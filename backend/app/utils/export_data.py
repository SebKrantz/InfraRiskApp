"""
Tabular / vector exports for analysis results (CSV, GPKG).
"""

from __future__ import annotations

import io
from typing import Any, Callable, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Geod

from app.utils.geospatial import pairwise_mean

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
        "damage_cost_lower",
        "damage_cost_upper",
        "line_id",
    }
)

# Per-feature analysis output, never passed through as an input attribute
_ANALYSIS_COLS = frozenset(
    {
        "affected",
        "exposure_level",
        "vulnerability",
        "damage_cost",
        "damage_cost_lower",
        "damage_cost_upper",
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
        if col in _ANALYSIS_COLS:
            continue
        low = col.lower()
        if low in _COORD_ATTR_SKIP:
            continue
        out[col] = gdf[col].values

    out["hazard_intensity"] = gdf["exposure_level"].values
    if has_vuln:
        out["damage_ratio"] = gdf["vulnerability"].values
        out["damage_cost"] = gdf["damage_cost"].values
        for bound in ("damage_cost_lower", "damage_cost_upper"):
            if bound in gdf.columns:
                out[bound] = gdf[bound].values

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

    src = pd.DataFrame(segment_gdf.drop(columns="geometry", errors="ignore"))
    by = src.groupby("line_id", sort=True)
    weight = src["length_m"].astype(float)

    def _weighted(col: str) -> pd.Series:
        """Length-weighted mean of col per line, over segments where it is set."""
        values = src[col].astype(float)
        defined = values.notna()
        w = weight.where(defined, 0.0)
        num = (w * values.fillna(0.0)).groupby(src["line_id"], sort=True).sum()
        den = w.groupby(src["line_id"], sort=True).sum()
        return num.div(den).where(den > 0)

    df = by[attr_cols].first() if attr_cols else pd.DataFrame(index=by.size().index)
    df["hazard_intensity"] = _weighted("exposure_level_avg")

    if has_vuln:
        df["damage_ratio"] = _weighted("vulnerability")
        df["damage_cost"] = by["damage_cost"].sum(min_count=0)
        # Costs add along a line; the bounds are totals too, not attributes
        for bound in ("damage_cost_lower", "damage_cost_upper"):
            if bound in src.columns:
                df[bound] = by[bound].sum(min_count=0)

    df = df.reset_index()
    df["line_id"] = df["line_id"].astype("int64")

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

    Where neither endpoint of a span was measured, hazard_intensity is NaN but
    damage_ratio is 0.0. The asymmetry is deliberate: an intensity is a
    measurement, and writing 0 mm for a cell the raster does not cover would be
    an invention, whereas a damage ratio is a contribution to a total and the
    analysis totals count an unmeasured span as contributing nothing.
    """
    geod = Geod(ellps="WGS84")
    vuln_on = vulnerability_interp is not None and replacement_value is not None

    ids: list = []
    coords: list = []
    hazard: list = []
    ratio: list = []
    cost: list = []

    for ld in line_data:
        pts = ld["sampled_points"]
        n = len(pts)
        if n < 2:
            continue
        rv = np.asarray(ld["raster_values"], dtype=float)
        lons = np.fromiter((p[0] for p in pts), dtype=np.float64, count=n)
        lats = np.fromiter((p[1] for p in pts), dtype=np.float64, count=n)

        valid = ~np.isnan(rv)
        hazard.append(pairwise_mean(rv, valid))
        ids.append(np.full(n - 1, int(ld["line_id"]), dtype=np.int64))

        # Each span is a two-point line: interleave its start and end coordinates
        pair = np.empty((2 * (n - 1), 2), dtype=np.float64)
        pair[0::2, 0] = lons[:-1]
        pair[0::2, 1] = lats[:-1]
        pair[1::2, 0] = lons[1:]
        pair[1::2, 1] = lats[1:]
        coords.append(pair)

        if vuln_on:
            vuln_pt = np.asarray(vulnerability_interp(rv), dtype=np.float64)
            dr = np.nan_to_num(
                pairwise_mean(np.where(valid, vuln_pt, 0.0), valid), nan=0.0
            )
            # One geodesic call per line rather than one per 100 m span
            _, _, seg_len = geod.inv(
                lons[:-1], lats[:-1], lons[1:], lats[1:], return_back_azimuth=True
            )
            ratio.append(dr)
            cost.append(float(replacement_value) * dr * seg_len)

    if ids:
        all_coords = np.concatenate(coords)
        geoms = shapely.linestrings(
            all_coords, indices=np.repeat(np.arange(len(all_coords) // 2), 2)
        )
        data: dict[str, Any] = {
            "id": np.concatenate(ids),
            "hazard_intensity": np.concatenate(hazard),
        }
        if vuln_on:
            data["damage_ratio"] = np.concatenate(ratio)
            data["damage_cost"] = np.concatenate(cost)
        gdf = gpd.GeoDataFrame(data, geometry=geoms, crs="EPSG:4326")
    else:
        empty_cols: dict[str, Any] = {
            "id": pd.Series(dtype="int64"),
            "hazard_intensity": pd.Series(dtype="float64"),
            "geometry": gpd.GeoSeries([], crs="EPSG:4326"),
        }
        if vuln_on:
            empty_cols["damage_ratio"] = pd.Series(dtype="float64")
            empty_cols["damage_cost"] = pd.Series(dtype="float64")
        gdf = gpd.GeoDataFrame(empty_cols, crs="EPSG:4326")

    buf = io.BytesIO()
    gdf.to_file(buf, driver="GPKG", layer="segments")
    return buf.getvalue()
