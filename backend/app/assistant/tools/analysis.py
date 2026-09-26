"""Exposure, vulnerability, comparison and open-ended Python analysis."""

from __future__ import annotations

import time
from typing import Any, Optional

from ... import config
from .. import domain, kernel
from ..conversations import Conversation
from . import tool
from .data import resolve_dataset


def _curve(conv: Conversation, name: str | None) -> dict[str, Any] | None:
    if not name:
        return None
    curve = conv.curves.get(name)
    if curve is None:
        # tolerate the model passing the filename
        for key, val in conv.curves.items():
            if key.startswith(name) or name.startswith(key):
                return val
        raise ValueError(
            f"no vulnerability curve {name!r}; loaded curves: "
            f"{sorted(conv.curves) or 'none'}. Use load_vulnerability_curve first."
        )
    return curve


_ANALYSIS_PROPS = {
    "file_id": {
        "type": "string",
        "description": "Dataset to analyse: a file_id (from load_infrastructure, "
        "list_datasets or POST /api/upload), or the name of a file uploaded to "
        "this conversation, which is loaded on first use.",
    },
    "hazard": {"type": "string", "description": "hazard_id or layer name."},
    "threshold": {
        "type": "number",
        "description": "Intensity at or above which an asset counts as affected, "
        "in the layer's own units (mm of inundation, km/h gust, cm/s² PGA, "
        "landslide class 1-5). Omit to count any positive intensity.",
    },
    "curve": {
        "type": "string",
        "description": "Name of a curve from load_vulnerability_curve. Supply "
        "with replacement_value to get damage costs as well as exposure.",
    },
    "replacement_value": {
        "type": "number",
        "description": "Asset value driving the damage cost: per FEATURE for "
        "point datasets, per METRE for line datasets. Required with `curve`.",
    },
}


@tool(
    "run_analysis",
    "Run the app's exposure analysis for one dataset against one hazard layer, "
    "and optionally the vulnerability (damage-cost) analysis on top. This is "
    "the same computation the app performs on screen, on the same data, so the "
    "numbers agree exactly. Points return affected/unaffected counts; lines are "
    "split at 100 m sampling into affected and unaffected segments and return "
    "metres. Supply `curve` + `replacement_value` for damage costs (with a "
    "lower/upper band when the curve carries one). The full result — including "
    "the per-feature/per-segment table — is kept in the Python namespace under "
    "the returned `stored_as` name; use python_exec to dig into it rather than "
    "re-running.",
    {
        "type": "object",
        "properties": _ANALYSIS_PROPS,
        "required": ["file_id", "hazard"],
    },
)
def run_analysis(
    conv: Conversation,
    file_id: str,
    hazard: str,
    threshold: Optional[float] = None,
    curve: Optional[str] = None,
    replacement_value: Optional[float] = None,
) -> dict[str, Any]:
    haz = domain.resolve_hazard(hazard)
    file_id = resolve_dataset(conv, file_id)
    info = domain.get_dataset(file_id)
    vc = _curve(conv, curve)
    if vc is not None and replacement_value is None:
        raise ValueError(
            "replacement_value is required with a vulnerability curve "
            "(per feature for points, per metre for lines)"
        )
    if replacement_value is not None and vc is None:
        raise ValueError("a `curve` is required whenever replacement_value is given")
    if replacement_value is not None and replacement_value <= 0:
        raise ValueError("replacement_value must be greater than zero")

    result = domain.run_exposure(
        file_id,
        haz,
        threshold=threshold,
        vulnerability_curve_interp=vc["interp"] if vc else None,
        replacement_value=replacement_value,
        vulnerability_curve_lower_interp=vc["lower"] if vc else None,
        vulnerability_curve_upper_interp=vc["upper"] if vc else None,
    )
    out = domain.summarise(result, info, haz, threshold)
    out["file_id"] = file_id
    if vc is not None:
        out["curve"] = curve
        out["replacement_value"] = replacement_value
        out["replacement_value_basis"] = (
            "per feature" if info["geometry_type"] == "Point" else "per metre"
        )
    out["stored_as"] = conv.store_result("analysis", result)
    return out


@tool(
    "compare_hazards",
    "Run the same dataset against several hazard layers in one call and return "
    "a tidy comparison table. This is the workhorse for return-period ladders "
    "(25/50/100 year) and climate-scenario comparisons (existing climate vs "
    "SSP1 lower bound vs SSP5 upper bound). Thresholds may be one value applied "
    "to every layer, or one per layer in the same order. Each layer's full "
    "result is cached, so a follow-up run_analysis on any of them is instant. "
    "Remote raster reads dominate the runtime — expect tens of seconds per "
    "layer on a large network. No new layer is started once the call's time "
    "budget is spent; those come back under `not_run` to be requested again.",
    {
        "type": "object",
        "properties": {
            "file_id": _ANALYSIS_PROPS["file_id"],
            "hazards": {
                "type": "array",
                "items": {"type": "string"},
                "description": "hazard_ids or layer names, max 12.",
            },
            "threshold": _ANALYSIS_PROPS["threshold"],
            "thresholds": {
                "type": "array",
                "items": {"type": "number"},
                "description": "One threshold per hazard, same order. Overrides "
                "`threshold` when given.",
            },
            "curve": _ANALYSIS_PROPS["curve"],
            "replacement_value": _ANALYSIS_PROPS["replacement_value"],
        },
        "required": ["file_id", "hazards"],
    },
)
def compare_hazards(
    conv: Conversation,
    file_id: str,
    hazards: list[str],
    threshold: Optional[float] = None,
    thresholds: Optional[list[float]] = None,
    curve: Optional[str] = None,
    replacement_value: Optional[float] = None,
    *,
    _budget_s: Optional[float] = None,
) -> dict[str, Any]:
    import pandas as pd

    # Stop starting layers well before the caller's own timeout, so a long
    # ladder returns what it finished instead of nothing.
    budget = _budget_s if _budget_s is not None else 0.75 * config.ASSISTANT_TOOL_TIMEOUT
    started = time.monotonic()
    if not hazards:
        raise ValueError("pass at least one hazard")
    if len(hazards) > 12:
        raise ValueError(f"too many layers ({len(hazards)}); analyse at most 12 at a time")
    if thresholds is not None and len(thresholds) != len(hazards):
        raise ValueError(
            f"thresholds has {len(thresholds)} entries but hazards has {len(hazards)}"
        )

    file_id = resolve_dataset(conv, file_id)
    info = domain.get_dataset(file_id)
    vc = _curve(conv, curve)
    if (vc is None) != (replacement_value is None):
        raise ValueError("`curve` and `replacement_value` must be given together")

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    not_run: list[str] = []
    for i, ref in enumerate(hazards):
        if time.monotonic() - started > budget:
            not_run = list(hazards[i:])
            break
        thr = thresholds[i] if thresholds is not None else threshold
        try:
            haz = domain.resolve_hazard(ref)
            result = domain.run_exposure(
                file_id,
                haz,
                threshold=thr,
                vulnerability_curve_interp=vc["interp"] if vc else None,
                replacement_value=replacement_value,
                vulnerability_curve_lower_interp=vc["lower"] if vc else None,
                vulnerability_curve_upper_interp=vc["upper"] if vc else None,
            )
            rows.append(domain.summarise(result, info, haz, thr))
        except Exception as exc:  # noqa: BLE001 — one bad layer must not sink the batch
            failures.append({"hazard": ref, "error": str(exc)})

    if not rows:
        raise ValueError(
            "every layer failed: "
            + "; ".join(f"{f['hazard']}: {f['error']}" for f in failures)
            + (f"; not run (time budget of {budget:.0f}s spent): {not_run}" if not_run else "")
        )
    df = pd.DataFrame(rows)
    out: dict[str, Any] = {
        "file_id": file_id,
        "geometry_type": info["geometry_type"],
        "compared": len(rows),
        "table": rows,
        "stored_as": conv.store_result("comparison", df),
    }
    if failures:
        out["failed"] = failures
    if not_run:
        out["not_run"] = not_run
        out["note"] = (
            f"time budget of {budget:.0f}s spent after {len(rows) + len(failures)} "
            "layers; call again with the layers under not_run"
        )
    return out


@tool(
    "sweep_thresholds",
    "Exposure of one dataset to one hazard across a range of thresholds — the "
    "sensitivity check that shows how much a result depends on where the line "
    "is drawn. The raster is sampled once and re-thresholded, so this is fast "
    "after the first run.",
    {
        "type": "object",
        "properties": {
            "file_id": _ANALYSIS_PROPS["file_id"],
            "hazard": _ANALYSIS_PROPS["hazard"],
            "thresholds": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Thresholds in the layer's units, max 20.",
            },
        },
        "required": ["file_id", "hazard", "thresholds"],
    },
)
def sweep_thresholds(
    conv: Conversation, file_id: str, hazard: str, thresholds: list[float]
) -> dict[str, Any]:
    import pandas as pd

    if not thresholds:
        raise ValueError("pass at least one threshold")
    if len(thresholds) > 20:
        raise ValueError(f"too many thresholds ({len(thresholds)}); use at most 20")

    haz = domain.resolve_hazard(hazard)
    file_id = resolve_dataset(conv, file_id)
    info = domain.get_dataset(file_id)
    rows = []
    for thr in sorted(thresholds):
        result = domain.run_exposure(file_id, haz, threshold=thr)
        rows.append(domain.summarise(result, info, haz, thr))
    df = pd.DataFrame(rows)
    return {
        "file_id": file_id,
        "hazard_id": haz["hazard_id"],
        "unit": domain.hazard_brief(haz)["unit"],
        "geometry_type": info["geometry_type"],
        "table": rows,
        "stored_as": conv.store_result("sweep", df),
    }


@tool(
    "python_exec",
    "Run Python in this conversation's persistent namespace (variables survive "
    "across calls). Preloaded: pandas as pd, numpy as np, geopandas as gpd, "
    "matplotlib.pyplot as plt, rasterio, `uploaded_files` (a read-only view of "
    "the app's dataset store: file_id -> {filename, geometry_type, "
    "feature_count, crs, bounds, gdf}, each gdf a copy), `HAZARDS` (the "
    "catalogue), the app's analysis functions (geospatial, hazards, analyze, "
    "export, export_data), `uploads` (name -> Path of files uploaded to this "
    "conversation) and save_artifact(obj, filename, title). Results of earlier "
    "tools are available under their `stored_as` names — an analysis result is "
    "a dict whose 'full_gdf' is the per-feature (points) or per-segment (lines) "
    "GeoDataFrame. Confined: the current directory is this conversation's "
    "workdir; files can be read and written only there (the app's data/ "
    "directory is readable), no subprocesses, no importing the server package. "
    "Every file the call produces — save_artifact(), figures you draw, and "
    "files you write into the workdir — is returned under `artifacts` with a "
    "download url. Print what you want to see; the trailing expression is "
    f"returned. Stopped after {config.ASSISTANT_EXEC_TIMEOUT:g} s; keep steps small.",
    {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python source to execute."},
        },
        "required": ["code"],
    },
)
def python_exec(conv: Conversation, code: str) -> dict[str, Any]:
    res = kernel.execute(conv, code)
    out: dict[str, Any] = {
        "ok": res["ok"],
        "stdout": res["stdout"],
        "result": res["result"],
        "figures": [a.public() for a in res.get("figures", [])],
        "artifacts": res.get("artifacts", []),
    }
    if not res["ok"]:
        out["error"] = res["error"]
    return out
