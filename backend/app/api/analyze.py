"""
Analysis endpoints for computing intersections
"""

import math
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from app.config import settings
from app.utils.geospatial import analyze_intersection
from app.api.upload import uploaded_files

router = APIRouter()


class AnalyzeRequest(BaseModel):
    """Request model for analysis"""
    file_id: str
    hazard_id: str
    hazard_url: str
    intensity_threshold: Optional[float] = None


@router.post("/analyze")
async def analyze_intersections(request: AnalyzeRequest):
    """
    Analyze intersections between uploaded infrastructure and hazard raster
    
    Parameters:
    - file_id: ID of uploaded infrastructure file
    - hazard_id: ID of hazard layer
    - hazard_url: URL/path to hazard raster (COG)
    - intensity_threshold: Optional threshold for hazard intensity filtering
    
    Returns:
    - Summary statistics (counts/meters affected)
    - Optional GeoJSON for visualization
    """
    # Check if file exists
    if request.file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="Uploaded file not found")
    
    file_info = uploaded_files[request.file_id]
    geometry_type = file_info["geometry_type"]
    infrastructure_gdf = file_info["gdf"]
    
    try:
        # Perform spatial intersection analysis
        analysis_result = analyze_intersection(
            infrastructure_gdf=infrastructure_gdf,
            hazard_raster_path=request.hazard_url,
            geometry_type=geometry_type,
            intensity_threshold=request.intensity_threshold
        )
        
        # Build response - ensure no NaN values
        def safe_float(value):
            """Convert float to JSON-safe value, replacing NaN/Inf with None"""
            if value is None:
                return None
            if isinstance(value, (int, float)):
                if math.isnan(value) or math.isinf(value):
                    return None
                return float(value)
            return value
        
        result = {
            "file_id": request.file_id,
            "hazard_id": request.hazard_id,
            "geometry_type": geometry_type,
            "summary": {
                "total_features": file_info["feature_count"]
            }
        }
        
        if geometry_type == "Point":
            result["summary"]["affected_count"] = analysis_result["affected_count"]
            result["summary"]["unaffected_count"] = analysis_result["unaffected_count"]
            result["summary"]["affected_meters"] = 0.0
            result["summary"]["unaffected_meters"] = 0.0
        elif geometry_type == "LineString":
            result["summary"]["affected_count"] = 0
            result["summary"]["unaffected_count"] = 0
            result["summary"]["affected_meters"] = safe_float(analysis_result["affected_meters"])
            result["summary"]["unaffected_meters"] = safe_float(analysis_result["unaffected_meters"])
        
        # Include full infrastructure GeoJSON with affected status
        try:
            # Get the full GeoDataFrame with affected status from analysis result
            if "full_gdf" in analysis_result:
                display_gdf = analysis_result["full_gdf"]
            else:
                # Fallback: create from original with affected status set to False
                display_gdf = infrastructure_gdf.copy()
                display_gdf['affected'] = False
            
            # GeoDataFrame is already in WGS84 (EPSG:4326) and cleaned from upload
            # Convert to GeoJSON
            geo_interface = display_gdf.__geo_interface__
            result["infrastructure_features"] = geo_interface
        except Exception as e:
            # If GeoJSON conversion fails, skip it
            print(f"Warning: Could not convert to GeoJSON: {e}")
        
        # Use JSONResponse with a simple encoder for any edge cases
        def json_encoder(obj):
            """Custom JSON encoder that handles NaN and Inf (shouldn't be needed if DataFrame cleaning works)"""
            if isinstance(obj, float):
                if math.isnan(obj):
                    return None
                if math.isinf(obj):
                    return None
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        # Try to serialize to catch any remaining NaN (but this should be rare now)
        try:
            json_str = json.dumps(result, default=json_encoder, allow_nan=False)
            # Parse back to dict for JSONResponse
            result = json.loads(json_str)
        except (ValueError, TypeError) as e:
            # If serialization fails, remove features and retry
            print(f"Warning: JSON serialization issue, removing features: {e}")
            if "infrastructure_features" in result:
                del result["infrastructure_features"]
            result = json.loads(json.dumps(result, default=json_encoder, allow_nan=False))
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during analysis: {str(e)}"
        )


@router.get("/analyze/{file_id}/status")
async def get_analysis_status(file_id: str):
    """Get status of analysis for a file"""
    if file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    return JSONResponse(content={
        "file_id": file_id,
        "status": "ready",
        "geometry_type": uploaded_files[file_id]["geometry_type"],
        "feature_count": uploaded_files[file_id]["feature_count"]
    })

