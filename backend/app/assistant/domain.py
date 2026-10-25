"""The bridge between assistant tools and the app's own analysis layer.

Everything here calls the same functions the HTTP API calls, on the same
in-memory state, so an assistant answer and the number on screen cannot drift.
Datasets are looked up for the conversation's owner (app/datasets.py): a
browser chat sees its user's datasets, the MCP scope its own.

Two invariants worth stating:
  * `analyze_intersection` mutates the GeoDataFrame it is given (it writes
    `affected` / `exposure_level` / `damage_cost` columns), and the stored
    object IS `uploaded_files[file_id]["gdf"]` — so we always pass a copy.
  * A run populates the app's raster-value and analysis-result caches, which is
    what makes the sidebar's own export buttons work on an assistant-run
    analysis afterwards. The analysis cache keys a damage run on the curve's
    digest and the replacement value, so a run with another curve never comes
    back from it.
"""

from __future__ import annotations

import logging
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Callable, Optional

from ..api.analyze import (
    cache_raster_values,
    curve_digest as _digest_of,
    get_cached_raster_values,
    get_exact_analysis_result,
    set_cached_analysis_result,
)
from .. import datasets, identity
from ..api.hazards import load_hazards_dict
from ..utils.geospatial import analyze_intersection

log = logging.getLogger("infrarisk.assistant")


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


def get_dataset(file_id: str, owner: Optional[str | tuple[str, ...]] = None) -> dict:
    """The owner's dataset (AEI_LOCAL_USER's when no owner is named). `owner` may
    name several principals: a conversation's `readers`."""
    owner = owner or identity.local_user()
    try:
        return datasets.get(owner, file_id)
    except datasets.DatasetNotFound:
        known = [fid for fid, _ in datasets.list_for(owner)]
        raise ValueError(
            f"no dataset {file_id!r}. Loaded datasets: {known or 'none'}. "
            "Use load_infrastructure to add one, or list_datasets to see them."
        ) from None


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


def check_id_column(info: dict, column: str) -> None:
    """An id column must exist and name every feature exactly once."""
    gdf = info["gdf"]
    if column not in gdf.columns:
        raise ValueError(
            f"no column {column!r} for id_column; columns: "
            f"{[c for c in gdf.columns if c != 'geometry'][:40]}"
        )
    ids = gdf[column]
    if ids.isna().any():
        raise ValueError(f"id_column {column!r} is empty for {int(ids.isna().sum())} features")
    dup = ids[ids.duplicated()]
    if len(dup):
        raise ValueError(
            f"id_column {column!r} is not unique — repeated values: "
            f"{sorted({str(v) for v in dup})[:10]}"
        )


def _key(value: Any) -> str:
    """A map key for an attribute value: 3.0 and "3" both become "3"."""
    if value is None or (isinstance(value, float) and value != value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def replacement_values(
    info: dict,
    column: str,
    mapping: Optional[dict] = None,
    default: Optional[float] = None,
) -> tuple[Any, dict[str, Any]]:
    """One replacement value per feature: the numbers in `column`, or — with
    `mapping` — the value its keys give each feature's `column` entry (an asset
    type, say). `default` fills features left without one. Returns the array,
    aligned with the dataset's rows, and a summary for the tool result."""
    import numpy as np
    import pandas as pd

    gdf = info["gdf"]
    if column not in gdf.columns:
        raise ValueError(
            f"no column {column!r} for replacement values; columns: "
            f"{[c for c in gdf.columns if c != 'geometry'][:40]}"
        )
    raw = gdf[column]
    if mapping:
        try:
            lookup = {_key(k): float(v) for k, v in mapping.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError(f"replacement_value_map values must be numbers: {exc}") from exc
        values = raw.map(_key).map(lookup)
    else:
        values = pd.to_numeric(raw, errors="coerce")
    values = values.astype(float)
    missing = ~np.isfinite(values.to_numpy())
    if missing.any():
        if default is None:
            what = (
                f"unmapped values {sorted({_key(v) for v in raw[missing]})[:10]}"
                if mapping
                else "empty or non-numeric entries"
            )
            raise ValueError(
                f"{int(missing.sum())} of {len(raw)} features get no replacement value "
                f"from {column!r} ({what}); extend replacement_value_map or pass "
                "replacement_value as the default for them"
            )
        values[missing] = float(default)
    if (values < 0).any():
        raise ValueError(f"replacement values from {column!r} must not be negative")
    arr = values.to_numpy(dtype=float)
    return arr, {
        "column": column,
        "mapped": bool(mapping),
        "defaulted": int(missing.sum()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


# ---- analysis -------------------------------------------------------------- #


def run_exposure(
    file_id: str,
    hazard: dict,
    threshold: Optional[float] = None,
    vulnerability_curve_interp: Optional[Callable[[float], float]] = None,
    replacement_value: Any = None,
    vulnerability_curve_lower_interp: Optional[Callable[[float], float]] = None,
    vulnerability_curve_upper_interp: Optional[Callable[[float], float]] = None,
    owner: Optional[str | tuple[str, ...]] = None,
    curve_digest: Optional[str] = None,
) -> dict[str, Any]:
    """Run one analysis exactly as POST /api/analyze would, and cache it there.
    `replacement_value` is a scalar or one value per dataset row.

    Returns the raw `analyze_intersection` dict (with `full_gdf`, `raster_values`
    and, for lines, `line_data`). `owner` is the conversation's readers; `curve_digest`
    (see curve_digest()) identifies the curve in the cache key.
    """
    info = get_dataset(file_id, owner)
    hazard_id = hazard["hazard_id"]
    geometry_type = info["geometry_type"]

    wants_vulnerability = (
        vulnerability_curve_interp is not None and replacement_value is not None
    )
    digest = rv = None
    if wants_vulnerability:
        # without a digest the run is cached under a key nothing else can match
        digest = curve_digest or f"interp:{id(vulnerability_curve_interp)}"
        rv = _value_key(replacement_value)
    # The same inputs are already on file (from the sidebar or an earlier call),
    # unless a raster read failed then.
    cached = get_exact_analysis_result(file_id, hazard_id, threshold, digest, rv)
    if cached is not None and cached.get("raster_complete", True):
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
    cache_raster_values(file_id, hazard_id, geometry_type, result)
    # Enough provenance for the figure tools to re-derive titles, units and the
    # threshold without the model having to thread them through by hand.
    result["_assistant_meta"] = {
        "file_id": file_id,
        "hazard_id": hazard_id,
        "threshold": threshold,
    }
    # Make the sidebar's export buttons work on what the assistant just ran.
    set_cached_analysis_result(file_id, hazard_id, threshold, result, digest, rv)
    return result


def _value_key(value: Any) -> Any:
    """The replacement value as part of the cache key: a scalar as it is (so the
    sidebar's run of the same inputs matches), per-asset values by their digest."""
    import numpy as np

    if np.ndim(value) == 0:
        return value
    return "values:" + _digest_of(np.asarray(value, dtype=np.float64).tobytes())


def curve_digest(curve: dict[str, Any]) -> str:
    """sha256 of a loaded curve's CSV, remembered on the curve: the analysis
    cache keys damage runs on it, and the sidebar's upload of the same file
    gives the same digest."""
    digest = curve.get("digest")
    if digest is None:
        try:
            data = Path(curve["path"]).read_bytes()
        except (KeyError, OSError, TypeError):
            data = repr(curve.get("path")).encode()
        digest = curve["digest"] = _digest_of(data)
    return digest


# Below this exposed share (percent, as summarise reports it) a map of the asset
# is a large figure with no content: the hazard raster is invisible and the whole
# line is green. make_map declines; the figures guide says the same.
MAP_MIN_EXPOSED_PCT = 1.0


def affected_share_pct(result: dict[str, Any], info: dict) -> Optional[float]:
    """The exposed share of the asset in percent, to two decimals: features for
    points, length for lines."""
    if info["geometry_type"] == "Point":
        affected = float(result.get("affected_count", 0))
        total = float(info.get("feature_count") or 0)
    else:
        affected = float(result.get("affected_meters", 0.0))
        total = affected + float(result.get("unaffected_meters", 0.0))
    return round(100.0 * affected / total, 2) if total else None


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
    else:
        aff = float(result.get("affected_meters", 0.0))
        un = float(result.get("unaffected_meters", 0.0))
        out["affected_meters"] = round(aff, 1)
        out["unaffected_meters"] = round(un, 1)
        out["affected_km"] = round(aff / 1000.0, 3)
        out["total_km"] = round((aff + un) / 1000.0, 3)
    out["affected_share_pct"] = affected_share_pct(result, info)
    for key in ("total_damage_cost", "total_damage_cost_lower", "total_damage_cost_upper"):
        if key in result and result[key] is not None:
            out[key] = float(result[key])
    return out
