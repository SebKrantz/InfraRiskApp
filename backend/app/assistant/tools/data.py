"""Turning uploaded files into things the app can analyse.

`load_infrastructure` registers a dataset in the app's own `uploaded_files`
store, so the very same object is available to the analysis tools AND can be
displayed on the map with ui_show_dataset. `load_vulnerability_curve` parses a
curve once and keeps the interpolators on the conversation.
"""

from __future__ import annotations

import uuid
import zipfile
from pathlib import Path
from typing import Any

from .. import domain
from ..conversations import Conversation
from . import tool


def _resolve_upload(conv: Conversation, name: str) -> Path:
    path = conv.uploads.get(name)
    if path is None:
        matches = [p for n, p in conv.uploads.items() if n.startswith(name)]
        if len(matches) == 1:
            path = matches[0]
    if path is None or not path.exists():
        raise ValueError(
            f"no uploaded file {name!r}; this conversation has: "
            f"{sorted(conv.uploads) or 'no uploads yet'}"
        )
    return path


@tool(
    "load_infrastructure",
    "Load an uploaded infrastructure dataset into the app so it can be "
    "analysed and displayed. Accepts a zipped shapefile (.zip), GeoPackage "
    "(.gpkg), GeoJSON (.geojson) or CSV with coordinates or a WKT geometry "
    "column. Polygons are converted to centroids; the result is either a Point "
    "or a LineString dataset. Returns a file_id to pass to run_analysis, plus "
    "the attribute columns and a preview of the table. Call ui_show_dataset "
    "with the file_id afterwards so the user sees it on the map.",
    {
        "type": "object",
        "properties": {
            "file": {
                "type": "string",
                "description": "Name of a file uploaded to this conversation "
                "(see list_files).",
            },
        },
        "required": ["file"],
    },
)
def load_infrastructure(conv: Conversation, file: str) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    from ...api.upload import uploaded_files
    from ...utils.geospatial import (
        convert_polygons_to_centroids,
        load_csv_points,
        load_spatial_file,
        validate_geometry_type,
    )

    path = _resolve_upload(conv, file)
    suffix = path.suffix.lower()

    if suffix == ".zip":
        target = conv.workdir / f"unzipped_{path.stem}"
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path) as zf:
            zf.extractall(target)
        shapefiles = list(target.rglob("*.shp"))
        if not shapefiles:
            raise ValueError(f"{path.name} contains no .shp file")
        gdf = load_spatial_file(str(shapefiles[0]))
    elif suffix == ".csv":
        gdf = load_csv_points(str(path))
    elif suffix in (".gpkg", ".geojson", ".json", ".shp"):
        gdf = load_spatial_file(str(path))
    else:
        raise ValueError(
            f"{path.name}: expected .zip, .gpkg, .geojson, .shp or .csv, got {suffix!r}"
        )

    gdf = convert_polygons_to_centroids(gdf)
    geometry_type = validate_geometry_type(gdf)

    # Same cleaning the upload endpoint does: NaN out of the attribute columns,
    # then WGS84, because every hazard raster is EPSG:4326.
    for col in gdf.columns:
        if col == "geometry":
            continue
        if gdf[col].dtype in ("float64", "float32"):
            gdf[col] = gdf[col].replace([np.nan, np.inf, -np.inf], [None, None, None])
        elif pd.api.types.is_numeric_dtype(gdf[col]):
            gdf[col] = gdf[col].replace([np.nan], [None])
    if gdf.crs and gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    file_id = str(uuid.uuid4())
    uploaded_files[file_id] = {
        "filename": path.name,
        "geometry_type": geometry_type,
        "feature_count": len(gdf),
        "crs": str(gdf.crs) if gdf.crs else None,
        "bounds": {
            "minx": float(gdf.bounds.minx.min()),
            "miny": float(gdf.bounds.miny.min()),
            "maxx": float(gdf.bounds.maxx.max()),
            "maxy": float(gdf.bounds.maxy.max()),
        },
        "gdf": gdf,
    }
    conv.datasets.append(file_id)
    conv.namespace[f"gdf_{len(conv.datasets)}"] = gdf

    out = domain.dataset_brief(file_id, uploaded_files[file_id])
    out["stored_as"] = f"gdf_{len(conv.datasets)}"
    out["head"] = gdf.drop(columns="geometry").head(5).to_string(max_cols=12)
    out["note"] = (
        f"loaded {len(gdf)} {geometry_type} features as file_id {file_id}; "
        "call ui_show_dataset to put it on the user's map"
    )
    return out


@tool(
    "load_vulnerability_curve",
    "Parse an uploaded vulnerability-curve CSV so it can be used in a damage "
    "analysis. Two layouts are accepted: two columns (intensity, "
    "proportion_destroyed) or four or more (intensity, lower, central, upper) "
    "where column 3 is the central curve and 2/4 give an uncertainty band. "
    "Intensity is in the hazard layer's own units and proportions must lie in "
    "[0, 1]. Returns the curve points and whether it carries bounds.",
    {
        "type": "object",
        "properties": {
            "file": {
                "type": "string",
                "description": "Name of an uploaded .csv (see list_files).",
            },
            "name": {
                "type": "string",
                "description": "Short label to refer to this curve later. "
                "Defaults to the filename.",
            },
        },
        "required": ["file"],
    },
)
def load_vulnerability_curve(
    conv: Conversation, file: str, name: str | None = None
) -> dict[str, Any]:
    from ...utils.geospatial import parse_vulnerability_curve_data

    path = _resolve_upload(conv, file)
    central, intensity, proportion, lower, upper = parse_vulnerability_curve_data(str(path))
    label = (name or path.stem).strip()
    conv.curves[label] = {
        "interp": central,
        "lower": lower,
        "upper": upper,
        "intensity": intensity,
        "proportion": proportion,
        "has_bounds": lower is not None and upper is not None,
        "path": path,
    }
    points = [
        {"intensity": float(i), "proportion_destroyed": float(p)}
        for i, p in zip(intensity, proportion)
    ]
    return {
        "curve": label,
        "points": len(points),
        "has_uncertainty_bounds": conv.curves[label]["has_bounds"],
        "intensity_range": [float(intensity.min()), float(intensity.max())],
        "proportion_range": [float(proportion.min()), float(proportion.max())],
        "curve_points": points[:40],
        "note": (
            f"curve {label!r} ready — pass curve='{label}' to run_analysis "
            "together with a replacement_value"
        ),
    }
