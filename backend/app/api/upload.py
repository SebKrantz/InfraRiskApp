"""
File upload endpoints
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import asyncio
import os
import time
import zipfile
import tempfile
from pathlib import Path
from typing import Optional

from app.config import settings
from app.utils.geospatial import load_spatial_file, load_csv_points, validate_geometry_type, convert_polygons_to_centroids

router = APIRouter()

class _UploadStore(dict):
    """Uploaded datasets by file_id, in memory only. Reading an entry marks it as used,
    so expire_uploads() drops only the ones nobody has touched for UPLOAD_TTL_HOURS."""

    def __getitem__(self, file_id):
        info = super().__getitem__(file_id)
        if "last_used" in info:
            info["last_used"] = time.monotonic()
        return info

    def get(self, file_id, default=None):
        return self[file_id] if file_id in self else default


# Store uploaded file metadata and GeoDataFrame in memory (never on disk)
uploaded_files = _UploadStore()

# How often the expiry runs while the server is up, seconds.
_SWEEP_INTERVAL_S = 600
_sweeper: Optional[asyncio.Task] = None


def forget_upload(file_id: str) -> bool:
    """Delete an upload and every cached result computed from it."""
    from app.api.analyze import clear_raster_cache_for_file

    found = uploaded_files.pop(file_id, None) is not None
    clear_raster_cache_for_file(file_id)
    return found


def expire_uploads(now: Optional[float] = None) -> list:
    """Delete the uploads unused for longer than UPLOAD_TTL_HOURS. Only uploads made
    through this endpoint carry `last_used`; anything else is left alone."""
    now = time.monotonic() if now is None else now
    ttl = settings.UPLOAD_TTL_HOURS * 3600
    stale = [
        file_id for file_id, info in list(uploaded_files.items())
        if now - info.get("last_used", now) > ttl
    ]
    for file_id in stale:
        forget_upload(file_id)
    return stale


async def _sweep_forever() -> None:
    while True:
        await asyncio.sleep(_SWEEP_INTERVAL_S)
        expire_uploads()


def _start_sweeper() -> None:
    """Run the expiry on this event loop from the first upload on, so data expires even
    when no further request arrives."""
    global _sweeper
    loop = asyncio.get_running_loop()
    if _sweeper is None or _sweeper.done() or _sweeper.get_loop() is not loop:
        _sweeper = loop.create_task(_sweep_forever())


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    file_id: Optional[str] = None
):
    """
    Upload a spatial file (Shapefile, GeoPackage, GeoJSON, or CSV with coordinates)

    Accepts:
    - .shp (zipped shapefile)
    - .gpkg (GeoPackage)
    - .geojson (GeoJSON)
    - .csv (with lat/lon, lat/lng, latitude/longitude, or y/x columns, or a
            WKT geometry column such as geometry/geom/wkt holding POINT or
            LINESTRING geometries)

    Shapefile and GeoPackage may contain Polygon or MultiPolygon geometries
    (e.g. buildings); these are converted to centroid points for analysis.

    Returns file metadata including geometry type and feature count
    """
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type not supported. Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}"
        )
    
    # Check file size
    file_content = await file.read()
    if len(file_content) > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.MAX_UPLOAD_SIZE / (1024*1024)} MB"
        )
    
    expire_uploads()
    _start_sweeper()

    # Generate file ID if not provided
    if not file_id:
        import uuid
        file_id = str(uuid.uuid4())
    
    try:
        # Save uploaded file
        if file_ext == ".zip":
            # Handle zipped shapefile
            with tempfile.TemporaryDirectory() as temp_dir:
                zip_path = Path(temp_dir) / file.filename
                zip_path.write_bytes(file_content)
                
                # Extract zip file
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Find .shp file
                shp_files = list(Path(temp_dir).glob("*.shp"))
                if not shp_files:
                    raise HTTPException(status_code=400, detail="No .shp file found in zip archive")
                
                # Load and validate
                gdf = load_spatial_file(str(shp_files[0]))
                
        elif file_ext == ".csv":
            # Handle CSV file with point coordinates
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.csv') as tmp_file:
                tmp_file.write(file_content)
                tmp_path = tmp_file.name
            
            try:
                gdf = load_csv_points(tmp_path)
            finally:
                os.unlink(tmp_path)
                
        else:
            # Handle GPKG or other spatial files
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                tmp_file.write(file_content)
                tmp_path = tmp_file.name
            
            try:
                gdf = load_spatial_file(tmp_path)
            finally:
                os.unlink(tmp_path)

        # Convert Polygon/MultiPolygon to centroids (point dataset) before validation
        gdf = convert_polygons_to_centroids(gdf)

        # Validate geometry type (points or lines)
        geometry_type = validate_geometry_type(gdf)
        
        # Clean NaN values from GeoDataFrame before storing (clean once, not during analysis)
        import numpy as np
        import pandas as pd
        for col in gdf.columns:
            if col != 'geometry':
                if gdf[col].dtype in ['float64', 'float32']:
                    gdf[col] = gdf[col].replace([np.nan, np.inf, -np.inf], [None, None, None])
                elif pd.api.types.is_numeric_dtype(gdf[col]):
                    gdf[col] = gdf[col].replace([np.nan], [None])
        
        # Transform to WGS84 in place (all hazard rasters use EPSG:4326, so store in WGS84)
        if gdf.crs and gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        
        # Store metadata and GeoDataFrame
        uploaded_files[file_id] = {
            "filename": file.filename,
            "geometry_type": geometry_type,
            "feature_count": len(gdf),
            "crs": str(gdf.crs) if gdf.crs else None,
            "bounds": {
                "minx": float(gdf.bounds.minx.min()),
                "miny": float(gdf.bounds.miny.min()),
                "maxx": float(gdf.bounds.maxx.max()),
                "maxy": float(gdf.bounds.maxy.max())
            },
            "gdf": gdf,  # Store GeoDataFrame for analysis (already cleaned and in WGS84)
            "last_used": time.monotonic(),
        }
        
        # Convert to GeoJSON for display (gdf is already in WGS84 and cleaned)
        # Strip per-feature bbox to avoid MapLibre geojson-vt tiler issues
        try:
            geo_json = gdf.__geo_interface__
            if "bbox" in geo_json:
                del geo_json["bbox"]
            for feature in geo_json.get("features", []):
                feature.pop("bbox", None)
        except Exception as e:
            print(f"Warning: Could not convert to GeoJSON for upload response: {e}")
            geo_json = None
        
        return JSONResponse(content={
            "file_id": file_id,
            "filename": file.filename,
            "geometry_type": geometry_type,
            "feature_count": len(gdf),
            "crs": uploaded_files[file_id]["crs"],
            "bounds": uploaded_files[file_id]["bounds"],
            "geojson": geo_json,  # Include GeoJSON for initial display
            "message": "File uploaded successfully"
        })
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )


@router.get("/upload/{file_id}")
async def get_upload_info(file_id: str):
    """Get metadata for an uploaded file"""
    if file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    # The stored entry also holds the GeoDataFrame, which is not JSON-serialisable
    info = uploaded_files[file_id]
    return JSONResponse(content={
        "file_id": file_id,
        "filename": info["filename"],
        "geometry_type": info["geometry_type"],
        "feature_count": info["feature_count"],
        "crs": info["crs"],
        "bounds": info["bounds"]
    })


@router.delete("/upload/{file_id}")
async def delete_upload(file_id: str):
    """Delete an uploaded file"""
    if not forget_upload(file_id):
        raise HTTPException(status_code=404, detail="File not found")

    return JSONResponse(content={"message": "File deleted successfully"})

