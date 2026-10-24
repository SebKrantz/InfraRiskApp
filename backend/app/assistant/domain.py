"""The bridge between assistant tools and the app's own analysis layer.

Everything here calls the same functions the HTTP API calls, on the same
in-memory state, so an assistant answer and the number on screen cannot drift.

Two invariants worth stating:
  * `analyze_intersection` mutates the GeoDataFrame it is given (it writes
    `affected` / `exposure_level` / `damage_cost` columns), and the stored
    object IS `uploaded_files[file_id]["gdf"]` — so we always pass a copy.
  * A run populates the app's raster-value and analysis-result caches, which is
    what makes the sidebar's own export buttons work on an assistant-run
    analysis afterwards.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from difflib import get_close_matches
from typing import Any, Callable, Optional

from ..api.analyze import (
    _analysis_results_cache,
    get_cached_analysis_result,
    get_cached_raster_values,
    set_cached_analysis_result,
    set_cached_raster_values,
)
from ..api.hazards import load_hazards_dict
from ..api.upload import uploaded_files
from ..utils.geospatial import analyze_intersection

log = logging.getLogger("infrarisk.assistant")

# The app caps `_analysis_results_cache` as a backstop, but that cap is blind to
# who made an entry. The assistant can add a dozen in one `compare_hazards`
# call, and each one holds a full GeoDataFrame — so we keep a bound on OUR OWN
# entries. Keys the user created through the UI are never evicted by us; their
# sidebar export buttons must keep working.
MAX_ASSISTANT_CACHED = 48
_OURS: OrderedDict[tuple, None] = OrderedDict()
_OURS_LOCK = threading.Lock()


def _remember_cached(key: tuple) -> None:
    with _OURS_LOCK:
        _OURS[key] = None
        _OURS.move_to_end(key)
        while len(_OURS) > MAX_ASSISTANT_CACHED:
            old, _ = _OURS.popitem(last=False)
            _analysis_results_cache.pop(old, None)
            log.info("assistant analysis cache evicted: %s", old)


# ---- hazards --------------------------------------------------------------- #


def catalogue() -> dict[str, dict]:
    return load_hazards_dict()


def _clean(value: Any) -> Optional[str]:
    """Hazard-CSV fields are free text and a few rows are column-misaligned;
    keep them short and never let one break a tool result."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def hazard_brief(h: dict) -> dict[str, Any]:
    """The compact form used in listings."""
    unit = _clean(h.get("unit"))
    category = _clean(h.get("category"))
    # Rows 22-24 of hazard_layers.csv are misaligned: a citation lands in `unit`
    # and "(class 1-5)" in `category`. Anything long is not a unit.
    if unit and len(unit) > 24:
        unit = None
    if category and len(category) > 40:
        category = None
    return {
        "hazard_id": h["hazard_id"],
        "name": _clean(h.get("hazard")),
        "unit": unit,
        "category": category,
    }


def resolve_hazard(ref: str) -> dict:
    """Accept a hazard_id, an exact name, or a close-enough name."""
    haz = catalogue()
    if ref in haz:
        return haz[ref]
    lowered = ref.strip().lower()
    for h in haz.values():
        if str(h.get("hazard", "")).strip().lower() == lowered:
            return h
    # substring, then fuzzy
    subs = [h for h in haz.values() if lowered in str(h.get("hazard", "")).lower()]
    if len(subs) == 1:
        return subs[0]
    names = [str(h.get("hazard", "")) for h in haz.values()]
    close = get_close_matches(ref, names, n=3, cutoff=0.6)
    hint = ""
    if subs:
        hint = f" Did you mean one of: {', '.join(str(s.get('hazard')) for s in subs[:5])}?"
    elif close:
        hint = f" Did you mean: {', '.join(close)}?"
    raise ValueError(
        f"no hazard layer matching {ref!r}.{hint} Use list_hazards to see the catalogue."
    )


def hazard_url(h: dict) -> str:
    url = _clean(h.get("dataset_url"))
    if not url:
        raise ValueError(f"hazard {h.get('hazard_id')!r} has no dataset_url")
    return url


# ---- datasets -------------------------------------------------------------- #


def get_dataset(file_id: str) -> dict:
    info = uploaded_files.get(file_id)
    if info is None:
        known = list(uploaded_files)
        raise ValueError(
            f"no dataset {file_id!r}. Loaded datasets: {known or 'none'}. "
            "Use load_infrastructure to add one, or list_datasets to see them."
        )
    return info


def dataset_brief(file_id: str, info: dict) -> dict[str, Any]:
    gdf = info["gdf"]
    return {
        "file_id": file_id,
        "filename": info.get("filename"),
        "geometry_type": info.get("geometry_type"),
        "feature_count": info.get("feature_count"),
        "bounds": info.get("bounds"),
        "columns": [c for c in gdf.columns if c != "geometry"][:40],
    }


# ---- analysis -------------------------------------------------------------- #


def run_exposure(
    file_id: str,
    hazard: dict,
    threshold: Optional[float] = None,
    vulnerability_curve_interp: Optional[Callable[[float], float]] = None,
    replacement_value: Optional[float] = None,
    vulnerability_curve_lower_interp: Optional[Callable[[float], float]] = None,
    vulnerability_curve_upper_interp: Optional[Callable[[float], float]] = None,
) -> dict[str, Any]:
    """Run one analysis exactly as POST /api/analyze would, and cache it there.

    Returns the raw `analyze_intersection` dict (with `full_gdf`, `raster_values`
    and, for lines, `line_data`).
    """
    info = get_dataset(file_id)
    hazard_id = hazard["hazard_id"]
    geometry_type = info["geometry_type"]

    wants_vulnerability = (
        vulnerability_curve_interp is not None and replacement_value is not None
    )
    # A plain exposure re-run at the same threshold is already on file.
    if not wants_vulnerability:
        cached = get_cached_analysis_result(file_id, hazard_id, threshold)
        if cached is not None and cached.get("total_damage_cost") is None:
            cached.setdefault(
                "_assistant_meta",
                {"file_id": file_id, "hazard_id": hazard_id, "threshold": threshold},
            )
            return cached

    result = analyze_intersection(
        infrastructure_gdf=info["gdf"].copy(),  # never mutate the stored frame
        hazard_raster_path=hazard_url(hazard),
        geometry_type=geometry_type,
        intensity_threshold=threshold,
        cached_raster_values=get_cached_raster_values(file_id, hazard_id),
        vulnerability_curve_interp=vulnerability_curve_interp,
        replacement_value=replacement_value,
        vulnerability_curve_lower_interp=vulnerability_curve_lower_interp,
        vulnerability_curve_upper_interp=vulnerability_curve_upper_interp,
    )

    # Cache the sampled raster so later threshold changes skip the COG reads.
    if "raster_values" in result:
        if geometry_type == "Point":
            set_cached_raster_values(file_id, hazard_id, result["raster_values"])
        elif geometry_type == "LineString" and "line_data" in result:
            set_cached_raster_values(
                file_id,
                hazard_id,
                {"line_data": result["line_data"], "raster_values": result["raster_values"]},
            )
    # Enough provenance for the figure tools to re-derive titles, units and the
    # threshold without the model having to thread them through by hand.
    result["_assistant_meta"] = {
        "file_id": file_id,
        "hazard_id": hazard_id,
        "threshold": threshold,
    }
    # Make the sidebar's export buttons work on what the assistant just ran.
    set_cached_analysis_result(file_id, hazard_id, threshold, result)
    _remember_cached((file_id, hazard_id, threshold))
    return result


def summarise(
    result: dict[str, Any], info: dict, hazard: dict, threshold: Optional[float]
) -> dict[str, Any]:
    """The compact, model-facing view of one analysis."""
    geometry_type = info["geometry_type"]
    total = info.get("feature_count") or 0
    out: dict[str, Any] = {
        "hazard": _clean(hazard.get("hazard")),
        "hazard_id": hazard["hazard_id"],
        "unit": hazard_brief(hazard)["unit"],
        "threshold": threshold,
        "threshold_rule": (
            f"affected when intensity >= {threshold}"
            if threshold is not None
            else "affected when intensity > 0 (no threshold set)"
        ),
        "geometry_type": geometry_type,
        "total_features": total,
    }
    if geometry_type == "Point":
        affected = int(result.get("affected_count", 0))
        out["affected_count"] = affected
        out["unaffected_count"] = int(result.get("unaffected_count", 0))
        out["affected_share_pct"] = round(100.0 * affected / total, 2) if total else None
    else:
        aff = float(result.get("affected_meters", 0.0))
        un = float(result.get("unaffected_meters", 0.0))
        out["affected_meters"] = round(aff, 1)
        out["unaffected_meters"] = round(un, 1)
        out["affected_km"] = round(aff / 1000.0, 3)
        out["total_km"] = round((aff + un) / 1000.0, 3)
        out["affected_share_pct"] = round(100.0 * aff / (aff + un), 2) if (aff + un) else None
    for key in ("total_damage_cost", "total_damage_cost_lower", "total_damage_cost_upper"):
        if key in result and result[key] is not None:
            out[key] = float(result[key])
    return out
