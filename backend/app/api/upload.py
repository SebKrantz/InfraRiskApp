"""
File upload endpoints
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import zipfile
import tempfile
from pathlib import Path
from typing import Optional

from app.config import settings
from app.utils.geospatial import load_spatial_file, load_csv_points, validate_geometry_type

router = APIRouter()

# Store uploaded file metadata and GeoDataFrame in memory (in production, use a database or cache)
uploaded_files = {}


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    file_id: Optional[str] = None
):
    """
    Upload a spatial file (Shapefile, GeoPackage, or CSV with coordinates)
    
    Accepts:
    - .shp (zipped shapefile)
    - .gpkg (GeoPackage)
    - .csv (with lat/lon, lat/lng, latitude/longitude, or y/x columns)
    
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
        
        # Validate geometry type (points or lines)
        geometry_type = validate_geometry_type(gdf)
        
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
            "gdf": gdf  # Store GeoDataFrame for analysis
        }
        
        # Convert to GeoJSON for display (if requested, can be done here or in separate endpoint)
        # For now, we'll include it in the response
        try:
            # Transform to WGS84 for web display
            display_gdf = gdf.copy()
            if display_gdf.crs and display_gdf.crs != "EPSG:4326":
                display_gdf = display_gdf.to_crs("EPSG:4326")
            
            # Clean NaN values
            import numpy as np
            import pandas as pd
            for col in display_gdf.columns:
                if col != 'geometry':
                    if display_gdf[col].dtype in ['float64', 'float32']:
                        display_gdf[col] = display_gdf[col].replace([np.nan, np.inf, -np.inf], [None, None, None])
                    elif pd.api.types.is_numeric_dtype(display_gdf[col]):
                        display_gdf[col] = display_gdf[col].replace([np.nan], [None])
            
            geo_json = display_gdf.__geo_interface__
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
    
    return JSONResponse(content=uploaded_files[file_id])


@router.delete("/upload/{file_id}")
async def delete_upload(file_id: str):
    """Delete an uploaded file"""
    if file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    del uploaded_files[file_id]
    return JSONResponse(content={"message": "File deleted successfully"})

