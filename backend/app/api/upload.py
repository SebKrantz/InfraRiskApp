"""
File upload endpoints
"""

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
import os
import zipfile
import tempfile
from pathlib import Path

from app import datasets
from app.config import settings
from app.identity import current_user
from app.utils.geospatial import load_spatial_file, load_csv_points, validate_geometry_type, convert_polygons_to_centroids

router = APIRouter()

# Uploaded file metadata and GeoDataFrames, in memory, each with an owner. The
# store lives in app/datasets.py; read it through datasets.get / list_for.
uploaded_files = datasets.STORE


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    user: str = Depends(current_user),
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

    Returns file metadata including geometry type and feature count. The
    file_id is always issued here and the dataset belongs to the uploader.
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

    # Parsing, reprojecting and serialising a large dataset takes seconds:
    # off the event loop, which serves every other user meanwhile.
    return await run_in_threadpool(_store_upload, user, file.filename, file_ext, file_content)


def _store_upload(user: str, filename: str, file_ext: str, file_content: bytes) -> JSONResponse:
    """Parse an upload, store it for `user` and build the response (blocking)."""
    try:
        # Save uploaded file
        if file_ext == ".zip":
            # Handle zipped shapefile
            with tempfile.TemporaryDirectory() as temp_dir:
                zip_path = Path(temp_dir) / (Path(filename).name or "upload.zip")
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
        
        # Store metadata and GeoDataFrame, owned by the uploader
        info = {
            "filename": filename,
            "geometry_type": geometry_type,
            "feature_count": len(gdf),
            "crs": str(gdf.crs) if gdf.crs else None,
            "bounds": {
                "minx": float(gdf.bounds.minx.min()),
                "miny": float(gdf.bounds.miny.min()),
                "maxx": float(gdf.bounds.maxx.max()),
                "maxy": float(gdf.bounds.maxy.max())
            },
            "gdf": gdf  # Store GeoDataFrame for analysis (already cleaned and in WGS84)
        }
        try:
            file_id = datasets.add(user, info)
        except datasets.QuotaExceeded as e:
            raise HTTPException(status_code=413, detail=str(e))
        
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
            "filename": filename,
            "geometry_type": geometry_type,
            "feature_count": len(gdf),
            "crs": info["crs"],
            "bounds": info["bounds"],
            "geojson": geo_json,  # Include GeoJSON for initial display
            "message": "File uploaded successfully"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )


def owned_dataset(user: str, file_id: str) -> dict:
    """The user's dataset, or a 404 exactly like an unknown id."""
    try:
        return datasets.get(user, file_id)
    except datasets.DatasetNotFound:
        raise HTTPException(status_code=404, detail="File not found") from None


@router.get("/upload/{file_id}")
async def get_upload_info(file_id: str, user: str = Depends(current_user)):
    """Get metadata for an uploaded file"""
    # The stored entry also holds the GeoDataFrame, which is not JSON-serialisable
    info = owned_dataset(user, file_id)
    return JSONResponse(content={
        "file_id": file_id,
        "filename": info["filename"],
        "geometry_type": info["geometry_type"],
        "feature_count": info["feature_count"],
        "crs": info["crs"],
        "bounds": info["bounds"]
    })


@router.delete("/upload/{file_id}")
async def delete_upload(file_id: str, user: str = Depends(current_user)):
    """Delete an uploaded file (and its cached rasters and analyses)"""
    try:
        datasets.remove(user, file_id)
    except datasets.DatasetNotFound:
        raise HTTPException(status_code=404, detail="File not found") from None
    return JSONResponse(content={"message": "File deleted successfully"})

