"""Turning uploaded files into things the app can analyse.

`load_infrastructure` (an uploaded file), `load_features` (inline GeoJSON) and
`load_from_url` (a download) register a dataset in the app's own
`uploaded_files` store, so the very same object is available to the analysis
tools AND can be displayed on the map with ui_show_dataset. `load_vulnerability_curve` parses a
curve once and keeps the interpolators on the conversation.
"""

from __future__ import annotations

import re
import uuid
import zipfile
from pathlib import Path
from typing import Any

from ... import config
from .. import domain
from ..conversations import Conversation
from . import tool


def _upload_name(conv: Conversation, name: str) -> str | None:
    """The upload called `name`, or the only one it is a prefix of."""
    if name in conv.uploads:
        return name
    matches = [n for n in conv.uploads if n.startswith(name)]
    return matches[0] if len(matches) == 1 else None


def _resolve_upload(conv: Conversation, name: str) -> Path:
    found = _upload_name(conv, name)
    path = conv.uploads.get(found) if found else None
    if path is None or not path.exists():
        raise ValueError(
            f"no uploaded file {name!r}; this conversation has: "
            f"{sorted(conv.uploads) or 'no uploads yet'}. Upload it with "
            "upload_file (MCP) or the conversation files route first."
        )
    return path


def resolve_dataset(conv: Conversation, ref: str) -> str:
    """A loaded dataset's file_id, or the name of a file uploaded to this
    conversation — loaded on first use and reused while the file is unchanged."""
    from ...api.upload import uploaded_files

    if ref in uploaded_files:
        return ref
    name = _upload_name(conv, ref)
    if name is None:
        raise ValueError(
            f"no dataset or uploaded file {ref!r}. Loaded datasets: "
            f"{list(uploaded_files) or 'none'}; files uploaded to this conversation: "
            f"{sorted(conv.uploads) or 'none'}. Upload a file and pass its name, "
            "or use load_infrastructure / load_features / load_from_url / list_datasets."
        )
    st = conv.uploads[name].stat()
    known = conv.upload_datasets.get((name, st.st_mtime_ns, st.st_size))
    if known in uploaded_files:
        return known
    return load_infrastructure(conv, name)["file_id"]


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
    from ...utils.geospatial import load_csv_points, load_spatial_file

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

    out = _register(conv, gdf, path.name)
    st = path.stat()
    conv.upload_datasets[(path.name, st.st_mtime_ns, st.st_size)] = out["file_id"]
    return out


def _register(conv: Conversation, gdf, filename: str) -> dict[str, Any]:
    """The common tail of every loader: centroids for polygons, one geometry
    type, the upload endpoint's cleaning, WGS84, and registration in the app's
    dataset store under a fresh file_id."""
    import numpy as np
    import pandas as pd

    from ...api.upload import uploaded_files
    from ...utils.geospatial import convert_polygons_to_centroids, validate_geometry_type

    if len(gdf) == 0:
        raise ValueError(f"{filename}: no features")
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
        "filename": filename,
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
    "load_features",
    "Load infrastructure passed inline as a GeoJSON object — a "
    "FeatureCollection, one Feature or a bare geometry — without any file "
    "upload. Feature properties become attribute columns. Same result as "
    "load_infrastructure: a file_id for run_analysis. Meant for small inputs "
    "(an alignment, a few hundred assets); send large datasets as files.",
    {
        "type": "object",
        "properties": {
            "geojson": {
                "type": "object",
                "description": "GeoJSON FeatureCollection, Feature or geometry.",
            },
            "crs": {
                "type": "string",
                "description": "CRS of the coordinates. Default EPSG:4326 "
                "(lon/lat); projected input is reprojected.",
            },
            "name": {
                "type": "string",
                "description": "Dataset name shown in list_datasets. Default "
                "features.geojson.",
            },
        },
        "required": ["geojson"],
    },
)
def load_features(
    conv: Conversation, geojson: Any, crs: str = "EPSG:4326", name: str | None = None
) -> dict[str, Any]:
    import json

    import geopandas as gpd

    if isinstance(geojson, str):
        try:
            geojson = json.loads(geojson)
        except ValueError as exc:
            raise ValueError(f"geojson is not valid JSON: {exc}") from exc
    if isinstance(geojson, list):
        features = geojson
    elif isinstance(geojson, dict) and geojson.get("type") == "FeatureCollection":
        features = geojson.get("features") or []
    elif isinstance(geojson, dict) and geojson.get("type") == "Feature":
        features = [geojson]
    elif isinstance(geojson, dict) and "coordinates" in geojson:
        features = [{"type": "Feature", "geometry": geojson, "properties": {}}]
    else:
        raise ValueError(
            "geojson must be a FeatureCollection, a list of Features, a Feature or "
            f"a geometry; got {str(geojson)[:120]!r}"
        )
    if not features:
        raise ValueError("geojson has no features")
    missing = [i for i, f in enumerate(features) if not (f or {}).get("geometry")]
    if missing:
        raise ValueError(f"features without a geometry at positions {missing[:10]}")
    try:
        gdf = gpd.GeoDataFrame.from_features(features, crs=crs)
    except Exception as exc:  # noqa: BLE001 — bad coordinates, unknown CRS
        raise ValueError(f"could not read the features: {exc}") from exc
    return _register(conv, gdf, Path(name).name if name else "features.geojson")


# The first bytes of the formats load_infrastructure reads, for downloads
# whose name carries no extension.
_MAGIC = ((b"SQLite format 3\x00", ".gpkg"), (b"PK", ".zip"), (b"{", ".geojson"))
_LOADABLE = (".gpkg", ".zip", ".geojson", ".json", ".csv", ".shp")


@tool(
    "load_from_url",
    "Download a dataset from a URL and load it like load_infrastructure — for a "
    "GeoPackage, GeoJSON, zipped shapefile or CSV that another service exports "
    "(e.g. a network export from another platform). An artifact url of this "
    "server (/api/assistant/artifacts/<id>) is read directly. The file is also "
    "kept in this conversation's uploads under its name.",
    {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "http(s) URL, or an artifact url."},
            "name": {
                "type": "string",
                "description": "File name to keep it under, with extension. "
                "Default: from the response or the URL.",
            },
        },
        "required": ["url"],
    },
)
def load_from_url(conv: Conversation, url: str, name: str | None = None) -> dict[str, Any]:
    from urllib.parse import unquote, urlparse

    from .. import artifacts

    cap = int(config.ASSISTANT_UPLOAD_MAX_MB * 1024 * 1024)
    local = re.search(r"/assistant/artifacts/([0-9a-f]{32})\b", url)
    header_name = None
    if local:
        art = artifacts.get(local.group(1))
        if art is None or not art.path.is_file():
            raise ValueError(f"artifact {local.group(1)} not found (it may have been evicted)")
        blob, header_name = art.path.read_bytes(), art.filename
    else:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(f"expected an absolute http(s) URL, got {url!r}")
        blob, header_name = _download(url, parsed.hostname or "", cap)
    if not blob:
        raise ValueError(f"{url} returned an empty body")

    filename = Path(name or header_name or unquote(urlparse(url).path)).name or "download"
    if Path(filename).suffix.lower() not in _LOADABLE:
        sniffed = next((ext for magic, ext in _MAGIC if blob.lstrip()[:16].startswith(magic)), None)
        if sniffed is None:
            raise ValueError(
                f"cannot tell the format of {filename!r}; pass name= with an extension "
                "(.gpkg, .geojson, .zip, .csv)"
            )
        filename = f"{Path(filename).stem or 'download'}{sniffed}"
    dest = conv.workdir / filename
    dest.write_bytes(blob)
    conv.uploads[filename] = dest
    out = load_infrastructure(conv, filename)
    out["source_url"] = url
    out["upload_name"] = filename
    return out


def _download(url: str, host: str, cap: int) -> tuple[bytes, str | None]:
    import httpx

    # A proxy configured for the internet must not swallow a call to a sibling
    # service on this machine.
    local = host in ("localhost", "127.0.0.1", "::1") or host.endswith(".localhost")
    chunks, size = [], 0
    try:
        with httpx.Client(timeout=httpx.Timeout(120.0, connect=15.0),
                          follow_redirects=True, trust_env=not local) as client:
            with client.stream("GET", url) as resp:
                if resp.status_code >= 400:
                    raise ValueError(f"{url} answered HTTP {resp.status_code}")
                for chunk in resp.iter_bytes():
                    size += len(chunk)
                    if size > cap:
                        raise ValueError(
                            f"{url} is larger than {config.ASSISTANT_UPLOAD_MAX_MB:g} MB"
                        )
                    chunks.append(chunk)
                disposition = resp.headers.get("content-disposition", "")
    except httpx.HTTPError as exc:
        raise ValueError(f"could not download {url}: {exc}") from exc
    match = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', disposition)
    return b"".join(chunks), (match.group(1) if match else None)


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
