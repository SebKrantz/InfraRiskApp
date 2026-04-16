"""
Tabular / vector exports for analysis results (CSV, GPKG).
"""

from __future__ import annotations

import io
from typing import Any, Callable, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Geod
from shapely.geometry import LineString

# Skip duplicate coordinate columns when emitting geometry-derived lon/lat
_COORD_ATTR_SKIP = frozenset({"lon", "lat", "longitude", "latitude"})

_LINE_SEGMENT_INTERNAL = frozenset(
    {
        "affected",
        "length_m",
        "exposure_level_avg",
        "exposure_level_max",
        "vulnerability",
        "damage_cost",
        "line_id",
    }
)


def points_analysis_to_csv_bytes(gdf: gpd.GeoDataFrame) -> bytes:
    """CSV: lon, lat, original attributes, hazard_intensity; optional damage_ratio, damage_cost."""
    n = len(gdf)
    if n == 0:
        return b"lon,lat,hazard_intensity\n"

    has_vuln = "vulnerability" in gdf.columns and gdf["vulnerability"].notna().any()

    out: dict[str, Any] = {
        "lon": gdf.geometry.x.values,
        "lat": gdf.geometry.y.values,
    }

    for col in gdf.columns:
        if col == "geometry":
            continue
        if col in ("affected", "exposure_level", "vulnerability", "damage_cost"):
            continue
        low = col.lower()
        if low in _COORD_ATTR_SKIP:
            continue
        out[col] = gdf[col].values

    out["hazard_intensity"] = gdf["exposure_level"].values
    if has_vuln:
        out["damage_ratio"] = gdf["vulnerability"].values
        out["damage_cost"] = gdf["damage_cost"].values

    df = pd.DataFrame(out)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def lines_aggregate_to_csv_bytes(segment_gdf: gpd.GeoDataFrame) -> bytes:
    """One row per line_id: original attributes (first of group) + length-weighted metrics."""
    if segment_gdf.empty or "line_id" not in segment_gdf.columns:
        return b"line_id,hazard_intensity\n"

    has_vuln = "vulnerability" in segment_gdf.columns and segment_gdf["vulnerability"].notna().any()

    attr_cols = [
        c
        for c in segment_gdf.columns
        if c not in _LINE_SEGMENT_INTERNAL and c != "geometry"
    ]

    rows: list[dict[str, Any]] = []
    for line_id, grp in segment_gdf.groupby("line_id", sort=True):
        w = grp["length_m"].astype(float).to_numpy()
        avg_exp = grp["exposure_level_avg"]
        mask = avg_exp.notna()
        if mask.any():
            wh = w[mask.to_numpy()]
            vals = avg_exp[mask].astype(float).to_numpy()
            hazard_intensity = float(np.sum(wh * vals) / np.sum(wh))
        else:
            hazard_intensity = np.nan

        first = grp.iloc[0]
        row: dict[str, Any] = {c: first[c] for c in attr_cols}
        row["line_id"] = int(line_id)
        row["hazard_intensity"] = hazard_intensity

        if has_vuln:
            mask_v = grp["vulnerability"].notna()
            if mask_v.any():
                wv = w[mask_v.to_numpy()]
                vv = grp.loc[mask_v, "vulnerability"].astype(float).to_numpy()
                row["damage_ratio"] = float(np.sum(wv * vv) / np.sum(wv))
            else:
                row["damage_ratio"] = np.nan
            row["damage_cost"] = float(grp["damage_cost"].fillna(0).sum())

        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        tail = ["hazard_intensity"]
        if has_vuln:
            tail.extend(["damage_ratio", "damage_cost"])
        front = ["line_id"] if "line_id" in df.columns else []
        middle = [c for c in df.columns if c not in front + tail]
        df = df[front + middle + tail]
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def lines_split_to_gpkg_bytes(
    line_data: list[dict[str, Any]],
    vulnerability_interp: Optional[Callable[[float], float]],
    replacement_value: Optional[float],
) -> bytes:
    """
    GPKG: consecutive sample pairs per line part; columns id, hazard_intensity,
    and optionally damage_ratio, damage_cost (no original line attributes).
    """
    geod = Geod(ellps="WGS84")
    vuln_on = vulnerability_interp is not None and replacement_value is not None

    rows: list[dict[str, Any]] = []
    for ld in line_data:
        lid = int(ld["line_id"])
        pts = ld["sampled_points"]
        rv = np.asarray(ld["raster_values"], dtype=float)

        for i in range(len(pts) - 1):
            pt1, pt2 = pts[i], pts[i + 1]
            geom = LineString([pt1, pt2])
            seg_len = geod.line_length(*zip(*[pt1, pt2]))

            def _val(j: int) -> Optional[float]:
                if j >= len(rv) or np.isnan(rv[j]):
                    return None
                return float(rv[j])

            v1, v2 = _val(i), _val(i + 1)
            if v1 is not None and v2 is not None:
                hi = (v1 + v2) / 2.0
            elif v1 is not None:
                hi = v1
            elif v2 is not None:
                hi = v2
            else:
                hi = np.nan

            rec: dict[str, Any] = {
                "id": lid,
                "hazard_intensity": hi,
                "geometry": geom,
            }
            if vuln_on:
                if v1 is not None and v2 is not None:
                    dr = (vulnerability_interp(float(v1)) + vulnerability_interp(float(v2))) / 2.0
                elif v1 is not None:
                    dr = vulnerability_interp(float(v1))
                elif v2 is not None:
                    dr = vulnerability_interp(float(v2))
                else:
                    dr = 0.0
                rec["damage_ratio"] = dr
                rec["damage_cost"] = float(replacement_value) * dr * seg_len
            rows.append(rec)

    if not rows:
        empty_cols: dict[str, Any] = {
            "id": pd.Series(dtype="int64"),
            "hazard_intensity": pd.Series(dtype="float64"),
            "geometry": gpd.GeoSeries([], crs="EPSG:4326"),
        }
        if vuln_on:
            empty_cols["damage_ratio"] = pd.Series(dtype="float64")
            empty_cols["damage_cost"] = pd.Series(dtype="float64")
        gdf = gpd.GeoDataFrame(empty_cols, crs="EPSG:4326")
    else:
        gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")

    buf = io.BytesIO()
    gdf.to_file(buf, driver="GPKG", layer="segments")
    return buf.getvalue()
