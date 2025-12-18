"""
Analysis endpoints for computing intersections
"""

import math
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Tuple
import numpy as np

from app.utils.geospatial import analyze_intersection, parse_vulnerability_curve
from app.api.upload import uploaded_files

router = APIRouter()

# Thread pool for blocking analysis operations
_analysis_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="analysis_worker")

# Cache for sampled raster values: (file_id, hazard_id) -> cached data
# For Points: np.ndarray of raster values
# For LineStrings: dict with 'line_data' and 'raster_values'
# This allows threshold changes without re-sampling the raster
_raster_values_cache: Dict[Tuple[str, str], any] = {}

# Cache for full analysis results: (file_id, hazard_id, threshold) -> full analysis result
# This allows export to use pre-computed results without recalculation
_analysis_results_cache: Dict[Tuple[str, str, Optional[float]], dict] = {}


def get_cached_raster_values(file_id: str, hazard_id: str) -> Optional[any]:
    """Get cached raster values if available."""
    return _raster_values_cache.get((file_id, hazard_id))


def set_cached_raster_values(file_id: str, hazard_id: str, values: any):
    """Cache raster values for future threshold changes."""
    _raster_values_cache[(file_id, hazard_id)] = values


def clear_raster_cache_for_file(file_id: str):
    """Clear cached raster values for a file (called when file is deleted)."""
    keys_to_remove = [k for k in _raster_values_cache if k[0] == file_id]
    for key in keys_to_remove:
        del _raster_values_cache[key]
    
    # Also clear analysis results cache
    keys_to_remove = [k for k in _analysis_results_cache if k[0] == file_id]
    for key in keys_to_remove:
        del _analysis_results_cache[key]


def get_cached_analysis_result(file_id: str, hazard_id: str, threshold: Optional[float]) -> Optional[dict]:
    """Get cached analysis result if available."""
    return _analysis_results_cache.get((file_id, hazard_id, threshold))


def set_cached_analysis_result(file_id: str, hazard_id: str, threshold: Optional[float], result: dict):
    """Cache full analysis result."""
    _analysis_results_cache[(file_id, hazard_id, threshold)] = result


class AnalyzeRequest(BaseModel):
    """Request model for analysis (JSON body)"""
    file_id: str
    hazard_id: str
    hazard_url: str
    intensity_threshold: Optional[float] = None


@router.post("/analyze")
async def analyze_intersections(
    request: Optional[AnalyzeRequest] = None,
    # FormData parameters (optional, used when vulnerability curve is provided)
    file_id: Optional[str] = Form(None),
    hazard_id: Optional[str] = Form(None),
    hazard_url: Optional[str] = Form(None),
    intensity_threshold: Optional[float] = Form(None),
    vulnerability_curve_file: Optional[UploadFile] = File(None),
    replacement_value: Optional[float] = Form(None)
):
    """
    Analyze intersections between uploaded infrastructure and hazard raster
    
    Parameters:
    - file_id: ID of uploaded infrastructure file
    - hazard_id: ID of hazard layer
    - hazard_url: URL/path to hazard raster (COG)
    - intensity_threshold: Optional threshold for hazard intensity filtering
    - vulnerability_curve_file: Optional CSV file with vulnerability curve (2 columns: intensity, proportion_destroyed)
    - replacement_value: Optional monetary replacement value (for lines, cost per meter)
    
    Returns:
    - Summary statistics (counts/meters affected, or total damage cost if vulnerability analysis enabled)
    - Optional GeoJSON for visualization
    """
    # Handle both JSON and FormData requests
    if request is not None:
        # JSON request
        actual_file_id = request.file_id
        actual_hazard_id = request.hazard_id
        actual_hazard_url = request.hazard_url
        actual_threshold = request.intensity_threshold
        actual_vulnerability_file = None
        actual_replacement_value = None
    else:
        # FormData request
        if file_id is None or hazard_id is None or hazard_url is None:
            raise HTTPException(status_code=400, detail="Missing required parameters: file_id, hazard_id, hazard_url")
        actual_file_id = file_id
        actual_hazard_id = hazard_id
        actual_hazard_url = hazard_url
        actual_threshold = intensity_threshold
        actual_vulnerability_file = vulnerability_curve_file
        actual_replacement_value = replacement_value
    
    # Check if file exists
    if actual_file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="Uploaded file not found")
    
    file_info = uploaded_files[actual_file_id]
    geometry_type = file_info["geometry_type"]
    infrastructure_gdf = file_info["gdf"]
    
    # Parse vulnerability curve if provided
    vulnerability_curve_interp = None
    if actual_vulnerability_file is not None:
        try:
            # Read vulnerability curve file
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.csv') as tmp_file:
                content = await actual_vulnerability_file.read()
                tmp_file.write(content)
                tmp_path = tmp_file.name
            
            try:
                vulnerability_curve_interp = parse_vulnerability_curve(tmp_path)
            finally:
                os.unlink(tmp_path)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error parsing vulnerability curve: {str(e)}"
            )
    
    try:
        # Check cache for previously sampled raster values
        cached_values = get_cached_raster_values(actual_file_id, actual_hazard_id)
        
        # Perform spatial intersection analysis in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        analysis_result = await loop.run_in_executor(
            _analysis_executor,
            lambda: analyze_intersection(
                infrastructure_gdf=infrastructure_gdf,
                hazard_raster_path=actual_hazard_url,
                geometry_type=geometry_type,
                intensity_threshold=actual_threshold,
                cached_raster_values=cached_values,
                vulnerability_curve_interp=vulnerability_curve_interp,
                replacement_value=actual_replacement_value
            )
        )
        
        # Cache raster values for future threshold changes
        if "raster_values" in analysis_result:
            if geometry_type == "Point":
                set_cached_raster_values(actual_file_id, actual_hazard_id, analysis_result["raster_values"])
            elif geometry_type == "LineString" and "line_data" in analysis_result:
                # For LineString, cache both line_data and raster_values
                set_cached_raster_values(actual_file_id, actual_hazard_id, {
                    "line_data": analysis_result["line_data"],
                    "raster_values": analysis_result["raster_values"]
                })
        
        # Cache full analysis result for export endpoints
        set_cached_analysis_result(actual_file_id, actual_hazard_id, actual_threshold, analysis_result)
        
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
            "file_id": actual_file_id,
            "hazard_id": actual_hazard_id,
            "geometry_type": geometry_type,
            "summary": {
                "total_features": file_info["feature_count"]
            }
        }
        
        # Add vulnerability analysis results if available
        if "total_damage_cost" in analysis_result:
            result["summary"]["total_damage_cost"] = safe_float(analysis_result["total_damage_cost"])
        
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
            result["infrastructure_features"] = display_gdf.__geo_interface__
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

