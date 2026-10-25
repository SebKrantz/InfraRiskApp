"""
Analysis endpoints for computing intersections
"""

import math
import json
import asyncio
import hashlib
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from typing import Any, Callable, Optional, Dict, Tuple
import numpy as np

from app import config, datasets, identity
from app.identity import current_user
from app.utils.sizing import approx_bytes
from app.utils.geospatial import analyze_intersection, parse_vulnerability_curve_data
from app.api.hazards import load_hazards_dict

router = APIRouter()

# Thread pool for the analyze endpoint's blocking work. The analysis slots
# (app/concurrency.py) bound the raster sampling; these threads also serve quick
# re-thresholds and the response's GeoJSON, so they outnumber the slots.
_analysis_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="analysis_worker")

# Both caches hold whole analyses — for a national line network that is a
# segment GeoDataFrame plus every sample point along it, hundreds of MB per
# entry. Dragging the threshold slider produces one analysis-result entry per
# stop, and several users do so at once. So each cache is bounded in bytes
# (RASTER_CACHE_MB, ANALYSIS_CACHE_MB) and fair between owners: over budget, the
# owner holding the most bytes loses their least recently used entry first. A
# raster cache miss costs a re-sample, an analysis cache miss only arithmetic.


class ByteBudgetCache:
    """An LRU cache bounded in bytes, fair between owners.

    Each entry records its owner and approximate size. An entry larger than the
    whole budget is not kept. Thread-safe; values are stored as given.
    """

    def __init__(self, budget_mb: float) -> None:
        self.budget = int(budget_mb * 2**20)
        self._entries: "OrderedDict[Any, Tuple[str, int, Any]]" = OrderedDict()
        self._by_owner: Dict[str, int] = {}
        self._lock = threading.Lock()

    def get(self, key) -> Any:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            self._entries.move_to_end(key)
            return entry[2]

    def put(self, key, value, owner: str, nbytes: int) -> bool:
        """Store `value`; False when it is larger than the whole budget."""
        with self._lock:
            self._drop(key)
            if nbytes > self.budget:
                return False
            self._entries[key] = (owner, nbytes, value)
            self._by_owner[owner] = self._by_owner.get(owner, 0) + nbytes
            while sum(self._by_owner.values()) > self.budget:
                self._drop(self._victim(spare=key))
            return True

    def discard_where(self, predicate: Callable[[Any], bool]) -> None:
        with self._lock:
            for key in [k for k in self._entries if predicate(k)]:
                self._drop(key)

    def bytes_by_owner(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._by_owner)

    def __contains__(self, key) -> bool:
        with self._lock:
            return key in self._entries

    def _victim(self, spare) -> Any:
        """The least recently used entry of the heaviest owner, never `spare`.
        Something other than `spare` exists: `spare` alone fits the budget."""
        for owner in sorted(self._by_owner, key=self._by_owner.get, reverse=True):
            for key, (who, _, _) in self._entries.items():  # least recently used first
                if who == owner and key != spare:
                    return key
        raise AssertionError("over budget with nothing to evict")

    def _drop(self, key) -> None:
        entry = self._entries.pop(key, None)
        if entry is not None:
            owner, nbytes, _ = entry
            left = self._by_owner[owner] - nbytes
            if left > 0:
                self._by_owner[owner] = left
            else:
                del self._by_owner[owner]


# Sampled raster values: (file_id, hazard_id) -> for Points an array of values,
# for LineStrings {'line_data', 'raster_values'}. Threshold changes reuse them.
_raster_values_cache = ByteBudgetCache(config.RASTER_CACHE_MB)

# Full analysis results, keyed by everything that changes them: (file_id,
# hazard_id, threshold, curve digest, replacement value). The sidebar's export
# buttons ask for "the analysis of this dataset, layer and threshold" without a
# curve; _latest remembers which full key that was most recently.
_analysis_results_cache = ByteBudgetCache(config.ANALYSIS_CACHE_MB)
_latest: Dict[Tuple[str, str, Optional[float]], tuple] = {}
_latest_lock = threading.Lock()


def _owner(file_id: str) -> str:
    return datasets.owner_for(file_id) or identity.local_user()


def _copy(result: Optional[dict]) -> Optional[dict]:
    """A copy of a cached result that a caller may add keys to (its
    _assistant_meta too) without changing the cached one. The frames inside are
    shared: callers copy them before writing columns."""
    if result is None:
        return None
    out = dict(result)
    if isinstance(out.get("_assistant_meta"), dict):
        out["_assistant_meta"] = dict(out["_assistant_meta"])
    return out


def get_cached_raster_values(file_id: str, hazard_id: str) -> Optional[Any]:
    """Get cached raster values if available."""
    return _raster_values_cache.get((file_id, hazard_id))


def set_cached_raster_values(file_id: str, hazard_id: str, values: Any) -> None:
    """Cache raster values for future threshold changes."""
    _raster_values_cache.put((file_id, hazard_id), values, _owner(file_id), approx_bytes(values))


def cache_raster_values(file_id: str, hazard_id: str, geometry_type: str, result: dict) -> None:
    """Keep a run's sampled raster for later threshold changes — unless a raster
    window failed to read: its NaNs mean "unknown", so the next run samples again
    rather than serving them as "unaffected"."""
    if "raster_values" not in result or not result.get("raster_complete", True):
        return
    if geometry_type == "Point":
        set_cached_raster_values(file_id, hazard_id, result["raster_values"])
    elif geometry_type == "LineString" and "line_data" in result:
        set_cached_raster_values(file_id, hazard_id, {
            "line_data": result["line_data"],
            "raster_values": result["raster_values"],
        })


def clear_raster_cache_for_file(file_id: str) -> None:
    """Clear a file's cached raster values and analyses (it was deleted or evicted)."""
    _raster_values_cache.discard_where(lambda key: key[0] == file_id)
    _analysis_results_cache.discard_where(lambda key: key[0] == file_id)
    with _latest_lock:
        for key in [k for k in _latest if k[0] == file_id]:
            del _latest[key]


def analysis_key(
    file_id: str,
    hazard_id: str,
    threshold: Optional[float],
    curve_digest: Optional[str] = None,
    replacement_value: Optional[float] = None,
) -> tuple:
    return (file_id, hazard_id, threshold, curve_digest, replacement_value)


def get_cached_analysis_result(file_id: str, hazard_id: str, threshold: Optional[float]) -> Optional[dict]:
    """The most recent analysis of this dataset, layer and threshold, whatever its
    curve: what the export buttons render. A copy (see _copy)."""
    with _latest_lock:
        key = _latest.get((file_id, hazard_id, threshold))
    return _copy(_analysis_results_cache.get(key)) if key is not None else None


def get_exact_analysis_result(
    file_id: str,
    hazard_id: str,
    threshold: Optional[float],
    curve_digest: Optional[str] = None,
    replacement_value: Optional[float] = None,
) -> Optional[dict]:
    """The analysis with exactly these inputs (no curve: exposure only). A copy."""
    key = analysis_key(file_id, hazard_id, threshold, curve_digest, replacement_value)
    return _copy(_analysis_results_cache.get(key))


def set_cached_analysis_result(
    file_id: str,
    hazard_id: str,
    threshold: Optional[float],
    result: dict,
    curve_digest: Optional[str] = None,
    replacement_value: Optional[float] = None,
) -> None:
    """Cache a full analysis result (a copy, so the caller keeps its own dict).
    Its size leaves out the sampled raster, which the raster cache counts."""
    key = analysis_key(file_id, hazard_id, threshold, curve_digest, replacement_value)
    nbytes = approx_bytes({k: v for k, v in result.items() if k not in ("raster_values", "line_data")})
    _analysis_results_cache.put(key, _copy(result), _owner(file_id), nbytes)
    with _latest_lock:
        _latest[(file_id, hazard_id, threshold)] = key


def curve_digest(content: bytes) -> str:
    """The vulnerability curve's identity in the cache key: sha256 of its CSV."""
    return hashlib.sha256(content).hexdigest()


class AnalyzeRequest(BaseModel):
    """Request model for analysis (JSON body)"""
    file_id: str
    hazard_id: str
    hazard_url: str
    intensity_threshold: Optional[float] = None


@router.post("/analyze")
async def analyze_intersections(
    request_obj: Request,
    request: Optional[AnalyzeRequest] = None,
    # FormData parameters (optional, used when vulnerability curve is provided)
    file_id: Optional[str] = Form(default=None),
    hazard_id: Optional[str] = Form(default=None),
    hazard_url: Optional[str] = Form(default=None),
    intensity_threshold: Optional[float] = Form(default=None),
    vulnerability_curve_file: Optional[UploadFile] = File(default=None),
    replacement_value: Optional[float] = Form(default=None),
    user: str = Depends(current_user),
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
    # Check content type to determine request format
    # Note: FastAPI may not parse JSON body when Form params are declared, so we check content-type header
    content_type = request_obj.headers.get("content-type", "")
    
    # Handle both JSON and FormData requests
    if "multipart/form-data" in content_type:
        # FormData request (when vulnerability curve file is included)
        if file_id is None or hazard_id is None or hazard_url is None:
            raise HTTPException(status_code=400, detail="Missing required parameters: file_id, hazard_id, hazard_url")
        actual_file_id = file_id
        actual_hazard_id = hazard_id
        actual_hazard_url = hazard_url
        actual_threshold = intensity_threshold
        actual_vulnerability_file = vulnerability_curve_file
        actual_replacement_value = replacement_value
    else:
        # JSON request - FastAPI should have parsed it into 'request'
        if request is None:
            # If request is None, try to parse JSON body manually
            try:
                body = await request_obj.json()
                request = AnalyzeRequest(**body)
            except Exception:
                raise HTTPException(status_code=400, detail="Missing or invalid request body")
        actual_file_id = request.file_id
        actual_hazard_id = request.hazard_id
        actual_hazard_url = request.hazard_url
        actual_threshold = request.intensity_threshold
        actual_vulnerability_file = None
        actual_replacement_value = None
    
    # The user's own dataset, or 404 as for an unknown id
    try:
        file_info = datasets.get(user, actual_file_id)
    except datasets.DatasetNotFound:
        raise HTTPException(status_code=404, detail="Uploaded file not found") from None
    geometry_type = file_info["geometry_type"]
    infrastructure_gdf = file_info["gdf"]
    
    # Parse vulnerability curve if provided
    vulnerability_curve_interp = None
    vulnerability_curve_lower_interp = None
    vulnerability_curve_upper_interp = None
    vulnerability_curve_intensity: Optional[list] = None
    vulnerability_curve_proportion: Optional[list] = None
    vulnerability_curve_digest: Optional[str] = None
    if actual_vulnerability_file is not None:
        try:
            # Read vulnerability curve file
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.csv') as tmp_file:
                content = await actual_vulnerability_file.read()
                tmp_file.write(content)
                tmp_path = tmp_file.name
            vulnerability_curve_digest = curve_digest(content)

            try:
                (
                    vulnerability_curve_interp,
                    vuln_intensity,
                    vuln_prop,
                    vulnerability_curve_lower_interp,
                    vulnerability_curve_upper_interp,
                ) = parse_vulnerability_curve_data(tmp_path)
                vulnerability_curve_intensity = vuln_intensity.tolist()
                vulnerability_curve_proportion = vuln_prop.tolist()
            finally:
                os.unlink(tmp_path)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error parsing vulnerability curve: {str(e)}"
            )

    if actual_vulnerability_file is not None and actual_replacement_value is None:
        raise HTTPException(
            status_code=400,
            detail="replacement_value is required when vulnerability_curve_file is provided",
        )
    if actual_replacement_value is not None and actual_vulnerability_file is None:
        raise HTTPException(
            status_code=400,
            detail="vulnerability_curve_file is required when replacement_value is provided",
        )
    if actual_replacement_value is not None and actual_replacement_value <= 0:
        raise HTTPException(
            status_code=400,
            detail="replacement_value must be greater than zero",
        )

    def run() -> str:
        """The analysis, the caching and the response body, all on a worker
        thread: for a large dataset the GeoJSON and its serialisation are as
        slow as the analysis, and the event loop serves every other user."""
        # Check cache for previously sampled raster values
        cached_values = get_cached_raster_values(actual_file_id, actual_hazard_id)
        
        analysis_result = analyze_intersection(
            infrastructure_gdf=infrastructure_gdf,
            hazard_raster_path=actual_hazard_url,
            geometry_type=geometry_type,
            intensity_threshold=actual_threshold,
            cached_raster_values=cached_values,
            vulnerability_curve_interp=vulnerability_curve_interp,
            replacement_value=actual_replacement_value,
            vulnerability_curve_lower_interp=vulnerability_curve_lower_interp,
            vulnerability_curve_upper_interp=vulnerability_curve_upper_interp,
        )

        if vulnerability_curve_intensity is not None:
            analysis_result["vulnerability_curve_intensity"] = vulnerability_curve_intensity
            analysis_result["vulnerability_curve_proportion"] = vulnerability_curve_proportion
            analysis_result["vulnerability_replacement_value"] = actual_replacement_value
        
        # Cache raster values for future threshold changes
        cache_raster_values(actual_file_id, actual_hazard_id, geometry_type, analysis_result)

        # Cache full analysis result for export endpoints (and the assistant)
        set_cached_analysis_result(
            actual_file_id, actual_hazard_id, actual_threshold, analysis_result,
            curve_digest=vulnerability_curve_digest,
            replacement_value=actual_replacement_value,
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
        if "total_damage_cost_lower" in analysis_result:
            result["summary"]["total_damage_cost_lower"] = safe_float(analysis_result["total_damage_cost_lower"])
        if "total_damage_cost_upper" in analysis_result:
            result["summary"]["total_damage_cost_upper"] = safe_float(analysis_result["total_damage_cost_upper"])
        if "total_replacement_value" in analysis_result:
            result["summary"]["total_replacement_value"] = safe_float(analysis_result["total_replacement_value"])
        
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
                display_gdf = analysis_result["full_gdf"].copy()
            else:
                # Fallback: create from original with affected status set to False
                display_gdf = infrastructure_gdf.copy()
                display_gdf['affected'] = False

            if "line_id" in display_gdf.columns:
                display_gdf = display_gdf.drop(columns=["line_id"])
            
            # GeoDataFrame is already in WGS84 (EPSG:4326) and cleaned from upload
            # Convert to GeoJSON, stripping per-feature bbox to avoid
            # MapLibre geojson-vt tiler issues
            geo_dict = display_gdf.__geo_interface__
            if "bbox" in geo_dict:
                del geo_dict["bbox"]
            for feature in geo_dict.get("features", []):
                feature.pop("bbox", None)
            result["infrastructure_features"] = geo_dict
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
        
        # Serialize once and hand the bytes straight back. Encoding, decoding and
        # letting JSONResponse encode again walked the whole feature collection
        # three times, which for a large upload is the most expensive step in
        # the request.
        try:
            body = json.dumps(result, default=json_encoder, allow_nan=False)
        except (ValueError, TypeError) as e:
            # If serialization fails, remove features and retry
            print(f"Warning: JSON serialization issue, removing features: {e}")
            result.pop("infrastructure_features", None)
            body = json.dumps(result, default=json_encoder, allow_nan=False)

        return body

    try:
        loop = asyncio.get_running_loop()
        body = await loop.run_in_executor(_analysis_executor, run)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during analysis: {str(e)}"
        )
    return Response(content=body, media_type="application/json")


@router.get("/analyze/{file_id}/status")
async def get_analysis_status(file_id: str, user: str = Depends(current_user)):
    """Get status of analysis for a file"""
    try:
        info = datasets.get(user, file_id)
    except datasets.DatasetNotFound:
        raise HTTPException(status_code=404, detail="File not found") from None

    return JSONResponse(content={
        "file_id": file_id,
        "status": "ready",
        "geometry_type": info["geometry_type"],
        "feature_count": info["feature_count"]
    })

