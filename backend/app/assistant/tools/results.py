"""Per-feature results out: the analysis table as data and as files.

An analysis result's `full_gdf` holds one row per point, or one row per
affected/unaffected run of a line, with every original attribute carried
along. These tools hand it over inline (paginated) or as a CSV / GeoPackage
artifact, with stable ids: `id` is the row's own identity and, for lines,
`line_id` ties a segment back to its input feature.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from .. import artifacts, domain
from ..conversations import Conversation
from . import tool
from .data import resolve_dataset

MAX_ROWS = 5000
_LINE_COLUMNS = [
    "id", "line_id", "length_m", "affected", "exposure_level_avg", "exposure_level_max",
    "vulnerability", "damage_cost", "damage_cost_lower", "damage_cost_upper",
]
_POINT_COLUMNS = [
    "id", "lon", "lat", "affected", "exposure_level",
    "vulnerability", "damage_cost", "damage_cost_lower", "damage_cost_upper",
]


def feature_table(result: dict[str, Any], id_column: Optional[str] = None):
    """The result's per-feature (points) or per-segment (lines) GeoDataFrame,
    computed columns first, original attributes after, geometry last."""
    import geopandas as gpd
    import numpy as np

    gdf = result.get("full_gdf")
    meta = result.get("_assistant_meta") or {}
    id_column = id_column or meta.get("id_column")
    if gdf is None or len(gdf) == 0:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    t = gdf.copy()
    lines = "length_m" in t.columns and "line_id" in t.columns
    # Attributes that would collide with a computed name (or GeoPackage's own
    # fid) are kept under an _input suffix.
    for col in list(t.columns):
        if col.lower() in ("id", "fid") or (not lines and col in ("lon", "lat")):
            t = t.rename(columns={col: f"{col}_input"})
    if lines:
        position = t["line_id"].astype(int).to_numpy() - 1  # 1-based input row
        if id_column:
            source = domain.get_dataset(meta["file_id"])["gdf"][id_column].to_numpy()
            t["line_id"] = source[position]
        t.insert(0, "id", np.arange(1, len(t) + 1))
        computed = _LINE_COLUMNS
    else:
        own = f"{id_column}_input" if id_column and id_column.lower() in ("id", "fid") else id_column
        t.insert(0, "id", t[own].to_numpy() if own else np.arange(1, len(t) + 1))
        t["lon"] = t.geometry.x
        t["lat"] = t.geometry.y
        computed = _POINT_COLUMNS
    # exposure-only runs carry empty damage columns
    for col in ("vulnerability", "damage_cost"):
        if col in t.columns and t[col].isna().all():
            t = t.drop(columns=col)
    first = [c for c in computed if c in t.columns]
    rest = [c for c in t.columns if c not in first and c != "geometry"]
    return t[first + rest + ["geometry"]]


def rows_of(table, limit: int, offset: int = 0, columns: Optional[list[str]] = None):
    """A page of the table as JSON-ready records (geometry as WKT if asked)."""
    import pandas as pd

    wanted = columns or [c for c in table.columns if c != "geometry"]
    unknown = [c for c in wanted if c not in table.columns]
    if unknown:
        raise ValueError(f"unknown columns {unknown}; available: {list(table.columns)}")
    limit = max(0, min(int(limit), MAX_ROWS))
    page = table.iloc[int(offset): int(offset) + limit]
    frame = pd.DataFrame(page[[c for c in wanted if c != "geometry"]])
    if "geometry" in wanted:
        frame["geometry"] = page.geometry.to_wkt()
    return json.loads(frame[wanted].to_json(orient="records", date_format="iso"))


def export_table(conv: Conversation, table, fmt: str, stem: str) -> dict[str, Any]:
    """The whole table as a CSV (no geometry; points keep lon/lat) or a
    GeoPackage (with geometry), registered as an artifact of this conversation."""
    if fmt not in ("csv", "gpkg"):
        raise ValueError(f"export must be 'csv' or 'gpkg', got {fmt!r}")
    conv._counter += 1
    filename = f"{stem}.{fmt}"
    dest = conv.workdir / f"{conv._counter}_{filename}"
    if fmt == "csv":
        table.drop(columns="geometry").to_csv(dest, index=False)
    else:
        try:
            table.to_file(dest, driver="GPKG", layer=stem[:60])
        except Exception as exc:  # noqa: BLE001 — e.g. nested property values
            raise ValueError(f"could not write the GeoPackage: {exc}") from exc
    art = artifacts.add(dest, filename, fmt, filename, conv.id)
    return {**art.public(), "bytes": dest.stat().st_size, "rows": int(len(table))}


def stem_for(result: dict[str, Any], kind: str) -> str:
    meta = result.get("_assistant_meta") or {}
    thr = meta.get("threshold")
    stem = f"{kind}_{meta.get('hazard_id', 'analysis')}"
    if thr is not None:
        stem += f"_t{thr:g}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)[:100]


def _resolve(
    conv: Conversation,
    stored_as: Optional[str],
    file_id: Optional[str],
    hazard: Optional[str],
    threshold: Optional[float],
    run_if_missing: bool = False,
) -> dict[str, Any]:
    """An analysis result by `stored_as`, or by (dataset, hazard, threshold)
    from the app's analysis cache."""
    from ...api.analyze import get_cached_analysis_result

    if stored_as:
        obj = conv.namespace.get(stored_as)
        if not isinstance(obj, dict) or "full_gdf" not in obj:
            raise ValueError(
                f"{stored_as!r} is not an analysis result in this conversation; pass "
                "the `stored_as` returned by run_analysis, or file_id + hazard + threshold"
            )
        return obj
    if not (file_id and hazard):
        raise ValueError("pass `stored_as`, or `file_id` and `hazard` (and the threshold used)")
    fid = resolve_dataset(conv, file_id)
    haz = domain.resolve_hazard(hazard)
    result = get_cached_analysis_result(fid, haz["hazard_id"], threshold)
    if result is None and run_if_missing:
        result = domain.run_exposure(fid, haz, threshold=threshold)
    if result is None:
        raise ValueError(
            f"no analysis on file for dataset {fid}, hazard {haz['hazard_id']!r} and "
            f"threshold {threshold!r} — run run_analysis first with the same threshold, "
            "or pass its stored_as"
        )
    result.setdefault(
        "_assistant_meta", {"file_id": fid, "hazard_id": haz["hazard_id"], "threshold": threshold}
    )
    return result


_EXPORT = {"type": "string", "enum": ["csv", "gpkg"]}


@tool(
    "get_analysis_table",
    "The per-feature table of an analysis that has been run: one row per point, "
    "or one row per affected/unaffected run of a line (`id` = segment, "
    "`line_id` = the input line it belongs to, `length_m`). Columns: affected, "
    "exposure level (points: exposure_level; lines: exposure_level_avg / _max), "
    "vulnerability (damage ratio) and damage_cost[_lower/_upper] when a curve "
    "was used, then every original attribute. Rows come back a page at a time "
    "(limit ≤ 5000, offset); export='csv' or 'gpkg' returns the WHOLE table as a "
    "downloadable artifact (GeoPackage keeps the geometry). Identify the "
    "analysis by `stored_as` from run_analysis, or by file_id + hazard + the "
    "threshold it was run with.",
    {
        "type": "object",
        "properties": {
            "stored_as": {"type": "string", "description": "`stored_as` from run_analysis."},
            "file_id": {"type": "string", "description": "Dataset (file_id or uploaded name)."},
            "hazard": {"type": "string", "description": "hazard_id or layer name."},
            "threshold": {"type": "number", "description": "Threshold the analysis used."},
            "limit": {"type": "integer", "description": "Rows per page, default 500, max 5000."},
            "offset": {"type": "integer", "description": "First row, default 0."},
            "columns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Only these columns ('geometry' gives WKT). Default: all "
                "but geometry.",
            },
            "export": {**_EXPORT, "description": "Also write the whole table as an artifact."},
        },
    },
)
def get_analysis_table(
    conv: Conversation,
    stored_as: Optional[str] = None,
    file_id: Optional[str] = None,
    hazard: Optional[str] = None,
    threshold: Optional[float] = None,
    limit: int = 500,
    offset: int = 0,
    columns: Optional[list[str]] = None,
    export: Optional[str] = None,
) -> dict[str, Any]:
    result = _resolve(conv, stored_as, file_id, hazard, threshold)
    table = feature_table(result)
    rows = rows_of(table, limit, offset, columns)
    out: dict[str, Any] = {
        "total": int(len(table)),
        "offset": int(offset),
        "rows": rows,
        "truncated": int(offset) + len(rows) < len(table),
        "columns": [c for c in table.columns if c != "geometry"],
    }
    if export:
        lines = "length_m" in table.columns
        out["artifact"] = export_table(
            conv, table, export, stem_for(result, "segments" if lines else "features")
        )
    return out


@tool(
    "get_affected_segments",
    "The affected parts of a LINE dataset for one hazard and threshold — the "
    "hand-off to a transport model (e.g. flooded links to disable). Each row is "
    "one continuous affected run of an input line: id, line_id (the input line), "
    "length_m, exposure_level_avg / _max, damage_cost when a curve was used, "
    "every original attribute, and the geometry. Runs the exposure analysis if "
    "it has not been run at this threshold. Returns counts, the affected "
    "line_ids and, by default, a GeoPackage artifact of the segments "
    "(export='csv' for a table, 'none' for rows inline).",
    {
        "type": "object",
        "properties": {
            "file_id": {"type": "string", "description": "Line dataset (file_id or uploaded name)."},
            "hazard": {"type": "string", "description": "hazard_id or layer name."},
            "threshold": {
                "type": "number",
                "description": "Affected when intensity >= threshold (layer units); "
                "omit for any positive intensity.",
            },
            "min_length_m": {
                "type": "number",
                "description": "Drop affected runs shorter than this, metres. Default 0.",
            },
            "export": {"type": "string", "enum": ["gpkg", "csv", "none"]},
        },
        "required": ["file_id", "hazard"],
    },
)
def get_affected_segments(
    conv: Conversation,
    file_id: str,
    hazard: str,
    threshold: Optional[float] = None,
    min_length_m: float = 0.0,
    export: str = "gpkg",
) -> dict[str, Any]:
    result = _resolve(conv, None, file_id, hazard, threshold, run_if_missing=True)
    meta = result["_assistant_meta"]
    if domain.get_dataset(meta["file_id"])["geometry_type"] != "LineString":
        raise ValueError(
            "get_affected_segments is for line datasets; for points use "
            "get_analysis_table and keep the rows with affected = true"
        )
    table = feature_table(result)
    if len(table):
        table = table[table["affected"].astype(bool) & (table["length_m"] >= float(min_length_m))]
    line_ids = list(dict.fromkeys(table["line_id"].tolist())) if len(table) else []
    out: dict[str, Any] = {
        "file_id": meta["file_id"],
        "hazard_id": meta["hazard_id"],
        "threshold": threshold,
        "n_segments": int(len(table)),
        "affected_km": round(float(table["length_m"].sum()) / 1000.0, 3) if len(table) else 0.0,
        "n_lines": len(line_ids),
        "line_ids": json.loads(json.dumps(line_ids[:2000], default=str)),
        "line_ids_truncated": len(line_ids) > 2000,
    }
    if export == "none":
        out["rows"] = rows_of(table, 500)
        out["truncated"] = len(table) > 500
    elif len(table):
        out["artifact"] = export_table(conv, table, export, stem_for(result, "affected_segments"))
    return out
