"""Artifact-producing tools: the app's own exhibits, custom figures, deliverables.

`make_barchart` and `make_map` call `app.api.export`'s renderers with the same
arguments the sidebar's export buttons use, so what the assistant produces is
byte-for-byte the app's own output. `make_chart` and `make_custom_map` are the
general-purpose complement for comparisons the app has no built-in view for.
"""

from __future__ import annotations

from typing import Any, Optional

from .. import artifacts, domain, figures, reports
from ..conversations import Conversation
from . import tool


def _analysis(conv: Conversation, ref: str) -> dict[str, Any]:
    """Resolve a `stored_as` name (or a file_id+hazard pair already cached)."""
    obj = conv.namespace.get(ref)
    if obj is None:
        raise ValueError(
            f"no variable {ref!r} in the namespace — pass the `stored_as` name "
            "returned by run_analysis"
        )
    if not isinstance(obj, dict) or "full_gdf" not in obj:
        raise ValueError(
            f"{ref!r} is not an analysis result (expected the dict run_analysis stored)"
        )
    return obj


def _table_from(
    conv: Conversation,
    source: str | None,
    columns: list[str] | None,
    rows: list[list[Any]] | None,
):
    """Either inline (columns+rows) or a DataFrame-ish variable from the namespace."""
    import geopandas as gpd
    import pandas as pd

    if source:
        obj = conv.namespace.get(source)
        if obj is None:
            raise ValueError(f"no variable {source!r} in the namespace")
        if isinstance(obj, gpd.GeoDataFrame):
            return pd.DataFrame(obj.drop(columns="geometry"))
        if isinstance(obj, pd.DataFrame):
            return obj
        try:
            return pd.DataFrame(obj)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"variable {source!r} is not table-like: {exc}") from exc
    if columns and rows is not None:
        return pd.DataFrame(rows, columns=columns)
    raise ValueError("pass either `source` (a namespace variable) or columns+rows")


# --------------------------------------------------------------------------- #
# The app's own two exhibits
# --------------------------------------------------------------------------- #


@tool(
    "make_barchart",
    "The app's own analysis barchart, rendered exactly as the sidebar's "
    "'Export Barchart' button produces it. In exposure mode it shows affected "
    "vs unaffected (counts for points, metres for lines) with the threshold in "
    "the category labels; in vulnerability mode it shows damage cost on the "
    "left axis with exposure and damage-ratio percentages on the right, and an "
    "error bar when the curve carried an uncertainty band. Use this as the "
    "standard exhibit for a single dataset-hazard pair; use make_chart for "
    "comparisons across layers.",
    {
        "type": "object",
        "properties": {
            "analysis": {
                "type": "string",
                "description": "`stored_as` name from run_analysis.",
            },
            "title": {"type": "string", "description": "Overrides the default title."},
            "filename": {"type": "string", "description": "e.g. flood100_exposure.png"},
        },
        "required": ["analysis"],
    },
)
def make_barchart(
    conv: Conversation,
    analysis: str,
    title: str | None = None,
    filename: str = "barchart.png",
) -> dict[str, Any]:
    from ...api.export import generate_barchart_png

    result = _analysis(conv, analysis)
    meta = result.get("_assistant_meta", {})
    haz = domain.resolve_hazard(meta.get("hazard_id", ""))
    info = domain.get_dataset(meta["file_id"])
    geometry_type = info["geometry_type"]

    summary = {
        "affected_count": result.get("affected_count", 0),
        "unaffected_count": result.get("unaffected_count", 0),
        "affected_meters": result.get("affected_meters", 0.0),
        "unaffected_meters": result.get("unaffected_meters", 0.0),
        "total_damage_cost": result.get("total_damage_cost"),
        "total_damage_cost_lower": result.get("total_damage_cost_lower"),
        "total_damage_cost_upper": result.get("total_damage_cost_upper"),
        "total_features": info["feature_count"],
    }
    hazard_name = domain._clean(haz.get("hazard")) or meta.get("hazard_id")
    default_title = (
        f"{hazard_name} - Vulnerability Analysis"
        if summary["total_damage_cost"] is not None
        else f"{hazard_name} - Infrastructure Exposure Analysis"
    )
    png = generate_barchart_png(
        summary,
        geometry_type,
        title or default_title,
        result.get("full_gdf"),
        meta.get("threshold"),
        domain.hazard_brief(haz)["unit"],
    )
    return _png_artifact(conv, png, filename, title or default_title, "chart")


@tool(
    "make_map",
    "The app's own hazard map, rendered exactly as the sidebar's 'Export Map' "
    "button produces it: the hazard raster as a blue intensity wash over a web "
    "basemap, with infrastructure coloured red/green by exposure — or along a "
    "green-amber-red damage-ratio ramp in vulnerability mode — plus colourbar, "
    "legend and title. Frames itself on the dataset's extent.",
    {
        "type": "object",
        "properties": {
            "analysis": {
                "type": "string",
                "description": "`stored_as` name from run_analysis.",
            },
            "basemap": {
                "type": "string",
                "description": "positron (default), dark-matter, osm, topo, "
                "esri-street, esri-topo, esri-imagery, google-maps, "
                "google-terrain, google-hybrid, google-satellite.",
            },
            "title": {"type": "string"},
            "filename": {"type": "string", "description": "e.g. flood100_map.png"},
        },
        "required": ["analysis"],
    },
)
def make_map(
    conv: Conversation,
    analysis: str,
    basemap: str = "positron",
    title: str | None = None,
    filename: str = "map.png",
) -> dict[str, Any]:
    from ...api.export import generate_map_png

    result = _analysis(conv, analysis)
    meta = result.get("_assistant_meta", {})
    haz = domain.resolve_hazard(meta.get("hazard_id", ""))
    info = domain.get_dataset(meta["file_id"])

    display_gdf = result["full_gdf"].copy()
    if "line_id" in display_gdf.columns:
        display_gdf = display_gdf.drop(columns=["line_id"])
    is_vulnerability = result.get("total_damage_cost") is not None
    hazard_name = domain._clean(haz.get("hazard")) or meta.get("hazard_id")
    default_title = (
        f"{hazard_name} - Vulnerability Analysis Map"
        if is_vulnerability
        else f"{hazard_name} - Infrastructure Exposure Map"
    )
    png = generate_map_png(
        display_gdf,
        domain.hazard_url(haz),
        info["geometry_type"],
        title or default_title,
        "turbo",
        1.0,  # exports always use full opacity
        is_vulnerability,
        meta.get("threshold"),
        basemap,
        domain.hazard_brief(haz)["unit"],
    )
    return _png_artifact(conv, png, filename, title or default_title, "map")


def _png_artifact(
    conv: Conversation, png: bytes, filename: str, title: str, kind: str
) -> dict[str, Any]:
    conv._counter += 1
    safe = filename if filename.lower().endswith(".png") else f"{filename}.png"
    dest = conv.workdir / f"{conv._counter}_{safe}"
    dest.write_bytes(png)
    art = artifacts.add(dest, safe, kind, title, conv.id)
    return {"artifact": art.public()}


# --------------------------------------------------------------------------- #
# General-purpose figures
# --------------------------------------------------------------------------- #


@tool(
    "make_chart",
    "A custom chart for anything the app's built-in barchart does not cover — "
    "exposure across return periods, climate scenarios, hazard types, datasets "
    "or thresholds. Series named 'Affected'/'Unaffected'/'Damage cost'/"
    "'Exposure' inherit the app's colours; everything else gets a "
    "colour-blind-safe palette. Axis ticks are humanised (12k, 3.4M). For "
    "anything more bespoke, draw it with matplotlib in python_exec.",
    {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": ["bar", "stacked_bar", "line", "pie"]},
            "title": {"type": "string"},
            "labels": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Category labels (x axis, or pie slices).",
            },
            "series": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "values": {"type": "array", "items": {"type": "number"}},
                    },
                    "required": ["name", "values"],
                },
            },
            "xlabel": {"type": "string"},
            "ylabel": {"type": "string", "description": "Always name the unit here."},
            "horizontal": {"type": "boolean", "description": "Horizontal bars."},
            "sort": {"type": "boolean", "description": "Order categories by magnitude."},
            "filename": {"type": "string", "description": "e.g. return_periods.png"},
        },
        "required": ["kind", "labels", "series"],
    },
)
def make_chart(
    conv: Conversation,
    kind: str,
    labels: list[str],
    series: list[dict[str, Any]],
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    horizontal: bool = False,
    sort: bool = False,
    filename: str = "chart.png",
) -> dict[str, Any]:
    conv._counter += 1
    dest = conv.workdir / f"{conv._counter}_{filename}"
    figures.chart(
        dest, kind, labels, series, title=title, xlabel=xlabel, ylabel=ylabel,
        horizontal=horizontal, sort=sort,
    )
    art = artifacts.add(dest, filename, "chart", title or filename, conv.id)
    return {"artifact": art.public()}


@tool(
    "make_custom_map",
    "A custom map from GeoDataFrames in the python namespace, over a web "
    "basemap. Use it for cartography the app's own map does not do: several "
    "datasets together, a subset (one country, the affected segments only), or "
    "a choropleth of an attribute. Build the GeoDataFrames with python_exec "
    "first — an analysis result's `full_gdf` is the per-feature/per-segment "
    "table and carries `affected`, `exposure_level*` and, in vulnerability "
    "mode, `damage_cost`.",
    {
        "type": "object",
        "properties": {
            "layers": {
                "type": "array",
                "description": "Drawn in order, first at the bottom.",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": "string",
                            "description": "Namespace variable holding a GeoDataFrame.",
                        },
                        "label": {"type": "string", "description": "Legend entry."},
                        "color": {"type": "string", "description": "Hex colour, e.g. #d62728."},
                        "column": {
                            "type": "string",
                            "description": "Colour by this column instead (choropleth).",
                        },
                        "cmap": {"type": "string", "description": "Colormap for `column`."},
                        "size": {"type": "number", "description": "Point marker size."},
                        "linewidth": {"type": "number", "description": "Line width."},
                    },
                    "required": ["source"],
                },
            },
            "title": {"type": "string"},
            "basemap": {
                "type": "string",
                "description": "Default 'osm'. The carto styles (positron, "
                "dark-matter) currently render with an API-key watermark.",
                "enum": [
                    "osm", "esri-street", "esri-topo", "esri-terrain",
                    "esri-imagery", "topo", "positron", "dark-matter",
                ],
            },
            "bbox": {
                "type": "array",
                "items": {"type": "number"},
                "description": "[west, south, east, north] in degrees, optional.",
            },
            "filename": {"type": "string"},
        },
        "required": ["layers"],
    },
)
def make_custom_map(
    conv: Conversation,
    layers: list[dict[str, Any]],
    title: str | None = None,
    basemap: str = "osm",
    bbox: list[float] | None = None,
    filename: str = "custom_map.png",
) -> dict[str, Any]:
    import geopandas as gpd

    resolved = []
    for layer in layers:
        name = layer.get("source")
        obj = conv.namespace.get(str(name))
        if obj is None:
            raise ValueError(f"no variable {name!r} in the namespace")
        if not isinstance(obj, gpd.GeoDataFrame):
            raise ValueError(f"variable {name!r} is not a GeoDataFrame")
        resolved.append({**layer, "gdf": obj, "label": layer.get("label") or name})

    conv._counter += 1
    dest = conv.workdir / f"{conv._counter}_{filename}"
    figures.custom_map(dest, resolved, title=title, basemap=basemap, bbox=bbox)
    art = artifacts.add(dest, filename, "map", title or filename, conv.id)
    return {"artifact": art.public()}


# --------------------------------------------------------------------------- #
# Deliverables
# --------------------------------------------------------------------------- #


@tool(
    "write_report_docx",
    "Write a Word report: title page, sections with markdown-lite paragraphs "
    "(**bold**, *italic*, '- ' bullets, '## ' subheadings), styled tables, and "
    "figures embedded by artifact id (from make_barchart / make_map / "
    "make_chart / make_custom_map / python_exec). Tables and figures are "
    "auto-numbered in document order (sections in order; within a section: "
    "tables, then figures). NEVER write 'Figure 3' by hand: give each exhibit "
    "a short `ref` label and cite it in your prose as [[that_ref]] — the "
    "builder substitutes the correct number, so cross-references cannot drift. "
    "Read read_guide('reports') before calling this.",
    {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "subtitle": {"type": "string"},
            "filename": {"type": "string", "description": "e.g. flood_exposure.docx"},
            "sections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "heading": {"type": "string"},
                        "paragraphs": {"type": "array", "items": {"type": "string"}},
                        "bullets": {"type": "array", "items": {"type": "string"}},
                        "tables": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {
                                        "type": "string",
                                        "description": "Caption after 'Table N.'",
                                    },
                                    "ref": {
                                        "type": "string",
                                        "description": "Short label, e.g. "
                                        "'flood_rp'. Cite it in prose as "
                                        "[[flood_rp]] to get 'Table N'.",
                                    },
                                    "columns": {"type": "array", "items": {"type": "string"}},
                                    "rows": {
                                        "type": "array",
                                        "items": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                    },
                                    "note": {
                                        "type": "string",
                                        "description": "Small-print source or "
                                        "definition note under the table.",
                                    },
                                },
                                "required": ["title", "columns", "rows"],
                            },
                        },
                        "figures": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "artifact_id": {"type": "string"},
                                    "caption": {"type": "string"},
                                    "ref": {
                                        "type": "string",
                                        "description": "Short label, e.g. "
                                        "'flood_map'. Cite it in prose as "
                                        "[[flood_map]] to get 'Figure N'.",
                                    },
                                    "note": {"type": "string"},
                                },
                                "required": ["artifact_id", "caption"],
                            },
                        },
                    },
                },
            },
        },
        "required": ["title", "sections"],
    },
)
def write_report_docx(
    conv: Conversation,
    title: str,
    sections: list[dict[str, Any]],
    subtitle: str | None = None,
    filename: str = "report.docx",
) -> dict[str, Any]:
    figure_paths = {a.id: a.path for a in artifacts.for_conversation(conv.id)}
    conv._counter += 1
    dest = conv.workdir / f"{conv._counter}_{filename}"
    _, labels = reports.build_docx(dest, title, subtitle, sections, figure_paths)
    art = artifacts.add(dest, filename, "docx", title, conv.id)
    return {"artifact": art.public(), "numbering": labels}


@tool(
    "export_excel",
    "Write an Excel workbook, one sheet per table. Tables come inline "
    "(columns+rows) or from namespace variables via `source` (a DataFrame, or "
    "a comparison/sweep result). Exports are for machines: snake_case column "
    "names, raw unformatted numbers, one row per record.",
    {
        "type": "object",
        "properties": {
            "filename": {"type": "string"},
            "sheets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "source": {
                            "type": "string",
                            "description": "Namespace variable (DataFrame or records).",
                        },
                        "columns": {"type": "array", "items": {"type": "string"}},
                        "rows": {
                            "type": "array",
                            "items": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                },
            },
        },
        "required": ["sheets"],
    },
)
def export_excel(
    conv: Conversation, sheets: list[dict[str, Any]], filename: str = "data.xlsx"
) -> dict[str, Any]:
    resolved = []
    for i, sheet in enumerate(sheets):
        df = _table_from(conv, sheet.get("source"), sheet.get("columns"), sheet.get("rows"))
        resolved.append(
            {
                "name": sheet.get("name") or sheet.get("source") or f"Sheet{i + 1}",
                "columns": list(df.columns),
                "rows": df.values.tolist(),
            }
        )
    conv._counter += 1
    dest = conv.workdir / f"{conv._counter}_{filename}"
    reports.build_xlsx(dest, resolved)
    art = artifacts.add(dest, filename, "xlsx", filename, conv.id)
    return {"artifact": art.public(), "sheets": [s["name"] for s in resolved]}


@tool(
    "export_csv",
    "Write one table as CSV, inline (columns+rows) or from a namespace "
    "variable via `source`.",
    {
        "type": "object",
        "properties": {
            "filename": {"type": "string"},
            "source": {"type": "string"},
            "columns": {"type": "array", "items": {"type": "string"}},
            "rows": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
)
def export_csv(
    conv: Conversation,
    filename: str = "data.csv",
    source: str | None = None,
    columns: list[str] | None = None,
    rows: list[list[Any]] | None = None,
) -> dict[str, Any]:
    df = _table_from(conv, source, columns, rows)
    conv._counter += 1
    dest = conv.workdir / f"{conv._counter}_{filename}"
    df.to_csv(dest, index=False)
    art = artifacts.add(dest, filename, "csv", filename, conv.id)
    return {"artifact": art.public(), "rows": int(len(df))}


@tool(
    "export_analysis_data",
    "The app's own per-feature data exports for an analysis that has been run. "
    "Modes: 'csv_points' — one row per point with its coordinates, original "
    "attributes, hazard intensity and (in vulnerability mode) damage ratio and "
    "cost; 'csv_lines_aggregate' — one row per input line with its "
    "length-weighted intensity and summed damage; 'gpkg_lines_split' — a "
    "GeoPackage of the affected/unaffected segments as actually split by the "
    "analysis. Use these when the user wants the underlying data rather than a "
    "summary table.",
    {
        "type": "object",
        "properties": {
            "file_id": {"type": "string"},
            "hazard": {"type": "string", "description": "hazard_id or layer name."},
            "mode": {
                "type": "string",
                "enum": ["csv_points", "csv_lines_aggregate", "gpkg_lines_split"],
            },
            "threshold": {
                "type": "number",
                "description": "The threshold the analysis was run with, if any.",
            },
        },
        "required": ["file_id", "hazard", "mode"],
    },
)
def export_analysis_data(
    conv: Conversation,
    file_id: str,
    hazard: str,
    mode: str,
    threshold: Optional[float] = None,
) -> dict[str, Any]:
    from ...api.export import _run_data_export

    haz = domain.resolve_hazard(hazard)
    try:
        body, filename, _mime = _run_data_export(file_id, haz["hazard_id"], threshold, mode)
    except ValueError as exc:
        code = str(exc)
        if code == "NO_ANALYSIS_CACHE":
            raise ValueError(
                "no analysis on file for that dataset, hazard and threshold — "
                "run_analysis first with the same threshold"
            ) from exc
        if code == "MODE_GEOMETRY_MISMATCH":
            raise ValueError(
                f"export mode {mode!r} does not match the dataset's geometry type"
            ) from exc
        if code == "UPLOAD_NOT_FOUND":
            raise ValueError(f"no dataset {file_id!r}") from exc
        raise
    conv._counter += 1
    dest = conv.workdir / f"{conv._counter}_{filename}"
    dest.write_bytes(body)
    kind = "gpkg" if filename.endswith(".gpkg") else "csv"
    art = artifacts.add(dest, filename, kind, filename, conv.id)
    return {"artifact": art.public(), "bytes": len(body)}


@tool(
    "list_files",
    "Files uploaded to this conversation and artifacts generated in it, plus "
    "the variables currently in the python namespace.",
    {"type": "object", "properties": {}},
)
def list_files(conv: Conversation) -> dict[str, Any]:
    return {
        "uploads": [
            {"name": n, "bytes": p.stat().st_size if p.exists() else 0}
            for n, p in conv.uploads.items()
        ],
        "curves": sorted(conv.curves),
        "artifacts": [a.public() for a in artifacts.for_conversation(conv.id)],
        "namespace_variables": sorted(
            k
            for k, v in conv.namespace.items()
            if not k.startswith("_") and not callable(v) and not hasattr(v, "__spec__")
        ),
    }
