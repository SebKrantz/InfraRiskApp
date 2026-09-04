"""Read-only tools: the hazard catalogue, raster statistics, loaded datasets."""

from __future__ import annotations

from typing import Any

from .. import domain
from ..conversations import Conversation
from . import tool


@tool(
    "list_hazards",
    "The hazard catalogue: every layer the app can analyse against, with its "
    "id, name, unit and category. Categories are Flood hazard, Tropical "
    "cyclone, Drought, Landslides, Earthquakes and Socioeconomic. Layers come "
    "in families — return periods (25/50/100/250/475/975 years) and climate "
    "scenarios (existing climate, SSP1 lower bound, SSP5 upper bound) — so "
    "filter rather than listing everything when you already know the family.",
    {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "description": "Exact category, e.g. 'Flood hazard'. Optional.",
            },
            "search": {
                "type": "string",
                "description": "Case-insensitive substring of the layer name, "
                "e.g. 'flood 100' or 'cyclone'. Optional.",
            },
        },
    },
)
def list_hazards(
    conv: Conversation, category: str | None = None, search: str | None = None
) -> dict[str, Any]:
    items = [domain.hazard_brief(h) for h in domain.catalogue().values()]
    if category:
        wanted = category.strip().lower()
        items = [i for i in items if (i["category"] or "").lower() == wanted]
    if search:
        needle = search.strip().lower()
        items = [i for i in items if needle in (i["name"] or "").lower()]
    categories = sorted({i["category"] for i in items if i["category"]})
    return {"count": len(items), "categories": categories, "hazards": items}


@tool(
    "get_hazard",
    "Full metadata for one hazard layer: the modelling description and the "
    "background paper. Read this before writing a method or caveats section "
    "about a layer — it is the authoritative source for how the layer was "
    "produced and what its units mean.",
    {
        "type": "object",
        "properties": {
            "hazard": {
                "type": "string",
                "description": "hazard_id or layer name (close matches are resolved).",
            },
        },
        "required": ["hazard"],
    },
)
def get_hazard(conv: Conversation, hazard: str) -> dict[str, Any]:
    h = domain.resolve_hazard(hazard)
    out = domain.hazard_brief(h)
    out["dataset_url"] = h.get("dataset_url")
    out["description"] = domain._clean(h.get("description"))
    out["background_paper"] = domain._clean(h.get("background_paper"))
    return out


@tool(
    "get_hazard_stats",
    "Minimum and maximum intensity of a hazard raster, in its own units. Use "
    "it to choose a defensible threshold and to state the range a layer spans. "
    "The first call reads the remote COG overview and can take a few seconds; "
    "it also warms the cache the map tiles need.",
    {
        "type": "object",
        "properties": {
            "hazard": {"type": "string", "description": "hazard_id or layer name."},
        },
        "required": ["hazard"],
    },
)
def get_hazard_stats(conv: Conversation, hazard: str) -> dict[str, Any]:
    import numpy as np
    import rasterio

    from ...api.hazards import _hazard_stats_cache, _raster_intensity_min_max

    h = domain.resolve_hazard(hazard)
    with rasterio.open(domain.hazard_url(h)) as src:
        bounds = _raster_intensity_min_max(src)
    if bounds is None:
        raise ValueError(
            f"could not read statistics for {h['hazard_id']!r} — the raster may "
            "be unreachable right now"
        )
    lo, hi = bounds
    # The tile renderer normalises on sqrt, and caches the transformed bounds.
    _hazard_stats_cache[h["hazard_id"]] = (float(np.sqrt(lo)), float(np.sqrt(hi)))
    brief = domain.hazard_brief(h)
    return {
        "hazard_id": h["hazard_id"],
        "name": brief["name"],
        "unit": brief["unit"],
        "min": float(lo),
        "max": float(hi),
    }


@tool(
    "list_datasets",
    "Infrastructure datasets currently loaded in the app — the ones the user "
    "uploaded through the sidebar and the ones you loaded with "
    "load_infrastructure. Gives each dataset's file_id, geometry type, feature "
    "count, bounds and attribute columns.",
    {"type": "object", "properties": {}},
)
def list_datasets(conv: Conversation) -> dict[str, Any]:
    from ...api.upload import uploaded_files

    return {
        "count": len(uploaded_files),
        "datasets": [
            domain.dataset_brief(fid, info) for fid, info in uploaded_files.items()
        ],
    }
