"""The shipped vulnerability-curve library: matching an asset to a curve.

`data/vulnerability_curves/` holds 218 curves from Nirandjan et al. (2024) plus
`curves_index.csv` describing each one and `asset_costs.csv` giving replacement
values. See that folder's README for provenance and the file format.

This module is the lookup layer. It knows two things the tools need:

  * **which library family a given app hazard layer can legally draw on** —
    only flood (mm), PGA (cm/s²) and cyclone wind (km/h) have curves in the
    layer's own units, and the landslide and drought layers have none at all;
  * **how to rank candidate curves against a free-text asset description**,
    with enough infrastructure vocabulary that "fibre backbone" reaches the
    telecommunication curves and "trunk road" reaches the motorway ones.

Nothing here interpolates or applies a curve — `parse_vulnerability_curve_data`
does that, on the very same CSV files, so a library curve and a user-uploaded
one travel identical code paths.
"""

from __future__ import annotations

import csv
import functools
import re
from pathlib import Path
from typing import Any

from ..config import settings

CURVES_DIR: Path = settings.DATA_DIR / "vulnerability_curves"
INDEX_CSV = CURVES_DIR / "curves_index.csv"
COSTS_CSV = CURVES_DIR / "asset_costs.csv"

# Library families whose curves are in the same unit as an app hazard layer.
APPLICABLE = ("flood", "earthquake_pga", "cyclone_wind")

# How an app hazard layer maps onto the library. Matched on the layer NAME,
# not the `category` column, because rows 22-24 of hazard_layers.csv are
# column-misaligned and their category is unreliable.
#
# family is None where the app has layers but the library has no curve in a
# compatible unit; `note` is what the assistant must tell the user.
_LAYER_RULES: list[tuple[re.Pattern, str | None, str]] = [
    (
        re.compile(r"flood", re.I),
        "flood",
        "Flood layers are inundation depth in mm; the library's flood curves are "
        "in mm and apply directly.",
    ),
    (
        re.compile(r"cyclone|wind", re.I),
        "cyclone_wind",
        "Cyclone layers are 10 m gust speed in km/h; the library's wind curves "
        "are in km/h and apply directly.",
    ),
    (
        re.compile(r"peak ground acceleration|\bpga\b", re.I),
        "earthquake_pga",
        "PGA layers are in cm/s²; the library's PGA curves are in cm/s² and "
        "apply directly.",
    ),
    (
        re.compile(r"landslide|susceptibility", re.I),
        None,
        "Landslide layers are an ordinal susceptibility CLASS 1-5, not a "
        "physical intensity. No published curve is keyed to that axis — the "
        "library's landslide curves use ground deformation, triggering "
        "precipitation or landslide area, and no published mapping connects a "
        "susceptibility class to any of them. A curve for this layer has to be "
        "constructed; read_guide('vulnerability_curves') carries the recipe and "
        "the evidence behind it.",
    ),
    (
        re.compile(r"drought|spi", re.I),
        None,
        "Drought layers are an index, a duration in days or an event count. "
        "Drought does not damage most infrastructure directly — its effect is "
        "loss of capacity and service, not destruction of assets — and no "
        "drought vulnerability curves exist in the literature the library is "
        "drawn from. Damage-cost analysis is the wrong tool here; "
        "read_guide('vulnerability_curves') explains what to report instead.",
    ),
    (
        re.compile(r"population|building exposure|ghsl|\bbem\b", re.I),
        None,
        "This is a socioeconomic layer, not a hazard. It has no intensity to "
        "feed a vulnerability curve.",
    ),
]

# Free-text vocabulary -> tokens that appear in the library's own asset names.
# The library speaks FEMA/JRC English ("distribution circuit", "lift station");
# users speak project English ("fibre backbone", "trunk road"). Without this
# bridge the search misses on wording alone.
SYNONYMS: dict[str, tuple[str, ...]] = {
    # telecommunication
    "fibre": ("communication", "telecommunication", "cable"),
    "fiber": ("communication", "telecommunication", "cable"),
    "optic": ("communication", "telecommunication", "cable"),
    "backbone": ("communication", "telecommunication"),
    "telecom": ("communication", "telecommunication"),
    "telecoms": ("communication", "telecommunication"),
    "broadband": ("communication", "telecommunication"),
    "internet": ("communication", "telecommunication"),
    "antenna": ("tower", "communication"),
    "mast": ("tower", "communication"),
    "bts": ("tower", "communication", "broadcasting"),
    "cell": ("tower", "communication"),
    "cellular": ("tower", "communication"),
    "mobile": ("tower", "communication"),
    "exchange": ("central", "offices", "communication"),
    # roads
    "road": ("roads",),
    "highway": ("motorways", "trunk", "roads"),
    "motorway": ("motorways", "trunk", "roads"),
    "freeway": ("motorways", "trunk", "roads"),
    "trunk": ("motorways", "trunk", "roads"),
    "street": ("roads",),
    "carriageway": ("roads",),
    "pavement": ("roads",),
    "track": ("roads", "railways"),
    # rail
    "rail": ("railways",),
    "railway": ("railways",),
    "railroad": ("railways",),
    "train": ("railways", "train", "stations"),
    "metro": ("railways", "light", "rail"),
    "tram": ("light", "rail"),
    # air and sea
    "airport": ("airports",),
    "airstrip": ("airports",),
    "runway": ("airports",),
    "aerodrome": ("airports",),
    # power
    "electricity": ("power", "distribution", "circuit"),
    "electric": ("power", "distribution", "circuit"),
    "grid": ("power", "distribution", "circuit", "transmission"),
    "transmission": ("power", "lines", "circuit"),
    "distribution": ("distribution", "circuit"),
    "pylon": ("power", "tower", "pole"),
    "pole": ("power", "pole"),
    "powerline": ("power", "lines", "distribution", "circuit"),
    "feeder": ("distribution", "circuit"),
    "substation": ("substation",),
    "transformer": ("substation",),
    "switchyard": ("substation",),
    "generation": ("power", "plants"),
    "generator": ("power", "plants"),
    "powerplant": ("power", "plants"),
    "thermal": ("thermal", "power", "plants"),
    "coal": ("thermal", "power", "plants"),
    "gas": ("thermal", "power", "plants"),
    "diesel": ("thermal", "power", "plants"),
    "hydro": ("hydropower", "plant"),
    "hydropower": ("hydropower", "plant"),
    "dam": ("hydropower", "plant"),
    "turbine": ("wind", "turbine"),
    "windfarm": ("wind", "turbine"),
    # water and wastewater
    "pipeline": ("pipelines", "buried"),
    "pipe": ("pipelines", "buried"),
    "main": ("pipelines", "transmission"),
    "sewer": ("sewers", "interceptors", "collector"),
    "sewerage": ("sewers", "interceptors"),
    "drainage": ("sewers", "collector"),
    "wastewater": ("wastewater", "treatment", "plants"),
    "sanitation": ("wastewater", "treatment", "plants"),
    "wwtp": ("wastewater", "treatment", "plants"),
    "wtp": ("water", "treatment", "plants"),
    "treatment": ("treatment", "plants"),
    "pump": ("pumping", "plants"),
    "pumping": ("pumping", "plants"),
    "booster": ("pumping", "plants"),
    "borehole": ("wells",),
    "well": ("wells",),
    "reservoir": ("water", "storage", "tanks"),
    "tank": ("water", "storage", "tanks"),
    "standpipe": ("water", "storage", "tanks"),
    # buildings
    "school": ("school", "educational", "education"),
    "university": ("educational", "education"),
    "college": ("educational", "education"),
    "clinic": ("health", "facilities"),
    "hospital": ("hospitals", "health", "facilities"),
    "health": ("health", "facilities"),
    "depot": ("fuel", "facility"),
    "fuel": ("fuel", "facility"),
    "refinery": ("fuel", "facility"),
    # qualifiers that matter for matching
    "buried": ("buried", "underground"),
    "underground": ("buried", "underground"),
    "overhead": ("elevated", "exposed"),
    "aerial": ("elevated", "exposed"),
    "anchored": ("anchored",),
    "unanchored": ("unanchored",),
}

# Sector hints: a query token that reliably implies one library sector.
SECTOR_HINTS: dict[str, str] = {
    "communication": "telecommunication",
    "telecommunication": "telecommunication",
    "roads": "transportation",
    "railways": "transportation",
    "airports": "transportation",
    "power": "energy",
    "substation": "energy",
    "hydropower": "energy",
    "turbine": "energy",
    "pipelines": "water_and_wastewater",
    "sewers": "water_and_wastewater",
    "wastewater": "water_and_wastewater",
    "pumping": "water_and_wastewater",
    "wells": "water_and_wastewater",
    "tanks": "water_and_wastewater",
    "school": "education_and_health",
    "educational": "education_and_health",
    "hospitals": "education_and_health",
    "health": "education_and_health",
}

_STOPWORDS = {
    "a", "an", "and", "the", "of", "for", "to", "in", "on", "with", "network",
    "networks", "infrastructure", "asset", "assets", "system", "systems", "line",
    "lines", "or", "my", "our", "this", "that", "data", "dataset",
}


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", (text or "").lower()) if t and t not in _STOPWORDS]


@functools.lru_cache(maxsize=1)
def index() -> list[dict[str, Any]]:
    """Every row of curves_index.csv, with a pre-tokenised haystack."""
    if not INDEX_CSV.exists():
        raise FileNotFoundError(
            f"the curve library index is missing at {INDEX_CSV}. Rebuild it with "
            "scripts/build_curve_library.py."
        )
    rows: list[dict[str, Any]] = []
    with INDEX_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["_asset_tokens"] = set(_tokens(row["asset"]))
            row["_detail_tokens"] = set(_tokens(row["characteristics"]))
            row["_file_tokens"] = set(_tokens(Path(row["file"]).stem))
            rows.append(row)
    return rows


@functools.lru_cache(maxsize=1)
def costs() -> list[dict[str, str]]:
    if not COSTS_CSV.exists():
        return []
    with COSTS_CSV.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def available() -> bool:
    return INDEX_CSV.exists()


def family_for_layer(hazard_name: str) -> tuple[str | None, str]:
    """(library family, explanation) for an app hazard layer name.

    A None family means no curve in the library can be applied to that layer —
    the explanation says why and what to do instead.
    """
    for pattern, family, note in _LAYER_RULES:
        if pattern.search(hazard_name or ""):
            return family, note
    return None, (
        f"No library family is mapped to {hazard_name!r}. Check the layer's unit "
        "with get_hazard before assuming any curve applies to it."
    )


def path_for(row: dict[str, Any]) -> Path:
    return CURVES_DIR / row["file"]


def get(curve_id: str) -> dict[str, Any]:
    """One curve by its published id, e.g. 'F7.1'. Case-insensitive."""
    wanted = curve_id.strip().upper()
    for row in index():
        if row["curve_id"].upper() == wanted:
            return row
    raise ValueError(
        f"no curve {curve_id!r} in the library. Use search() / "
        "search_curve_library to find one."
    )


def _expand(query: str) -> list[tuple[str, float]]:
    """Query tokens with weights; synonyms ride along at a lower weight."""
    out: dict[str, float] = {}
    for token in _tokens(query):
        out[token] = max(out.get(token, 0.0), 1.0)
        # Simple de-pluralisation so "towers" reaches "tower".
        if token.endswith("s") and len(token) > 3:
            out.setdefault(token[:-1], 0.8)
        for extra in SYNONYMS.get(token, ()) or SYNONYMS.get(token.rstrip("s"), ()):
            out[extra] = max(out.get(extra, 0.0), 0.6)
    return sorted(out.items(), key=lambda kv: -kv[1])


def search(
    query: str = "",
    family: str | None = None,
    sector: str | None = None,
    include_reference: bool = False,
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Rank library curves against a free-text asset description.

    `family` restricts to the curves applicable to one hazard (see
    `family_for_layer`). Reference curves — those on intensity metrics no app
    layer provides — are excluded unless asked for, or unless `family` names one.
    """
    rows = index()
    if family:
        rows = [r for r in rows if r["hazard"] == family]
    elif not include_reference:
        rows = [r for r in rows if r["hazard"] in APPLICABLE]
    if sector:
        wanted = sector.strip().lower()
        rows = [r for r in rows if r["sector"] == wanted]

    weighted = _expand(query)
    if not weighted:
        return [_brief(r, 0.0, []) for r in rows[:limit]]

    # A sector implied by the query is a strong signal, but only when it does
    # not empty the result set (asking for "pipeline" under the wind family
    # should still return something).
    implied = {SECTOR_HINTS[t] for t, _ in weighted if t in SECTOR_HINTS}
    phrase = " ".join(t for t, w in weighted if w == 1.0)

    scored: list[tuple[float, dict, list[str]]] = []
    for row in rows:
        score = 0.0
        matched: list[str] = []
        for token, weight in weighted:
            if token in row["_asset_tokens"]:
                score += 3.0 * weight
                matched.append(token)
            elif token in row["_detail_tokens"]:
                score += 1.5 * weight
                matched.append(token)
            elif token in row["_file_tokens"]:
                score += 1.0 * weight
                matched.append(token)
        if phrase and phrase in row["asset"].lower():
            score += 4.0
        if row["sector"] in implied:
            score += 2.0
        if score <= 0:
            continue
        # Tie-breakers, deliberately small: a curve carrying an uncertainty band
        # gives the user a range rather than a false point estimate, and a
        # globally-derived curve transfers better than a national one.
        if row["has_bounds"] == "yes":
            score += 0.3
        if row["geography"].strip().lower() in ("global", "europe", "asia"):
            score += 0.2
        scored.append((score, row, matched))

    scored.sort(key=lambda item: (-item[0], item[1]["curve_id"]))
    return [_brief(row, score, matched) for score, row, matched in scored[:limit]]


def _brief(row: dict[str, Any], score: float, matched: list[str]) -> dict[str, Any]:
    """The model-facing view of one library curve."""
    out = {
        "curve_id": row["curve_id"],
        "asset": row["asset"],
        "characteristics": row["characteristics"],
        "sector": row["sector"],
        "hazard": row["hazard"],
        "unit": row["unit"],
        "intensity_range": [float(row["intensity_min"]), float(row["intensity_max"])],
        "damage_at": row["damage_at"],
        "damage_max": float(row["damage_max"]),
        "has_bounds": row["has_bounds"] == "yes",
        "geography": row["geography"],
        "source": row["source"],
        "derivation": row["derivation"],
        "output": row["output"],
        "applies_to": row["app_layer_match"] or None,
        "match_score": round(score, 2),
        "matched_on": matched,
    }
    if row["central_basis"] != "published":
        out["central_basis"] = row["central_basis"]
    if row.get("bounds_note"):
        out["bounds_note"] = row["bounds_note"]
    if row["monotonic"] == "no":
        out["note"] = "not strictly monotonic in the published data"
    return out


def search_costs(query: str = "", basis: str | None = None, limit: int = 8) -> list[dict]:
    """Replacement-cost rows from Table D3, ranked the same way.

    `basis` filters on the unit: 'unit' for point assets (euro/unit), 'metre'
    for line assets (euro/m), 'area' for euro/m².
    """
    rows = costs()
    if basis:
        wanted = {"unit": "euro/unit", "metre": "euro/m", "meter": "euro/m", "area": "euro/m2"}
        target = wanted.get(basis.strip().lower())
        if target:
            rows = [r for r in rows if r["unit"].replace("²", "2").strip() == target]

    weighted = _expand(query)
    scored = []
    for row in rows:
        haystack = set(_tokens(row["asset"])) | set(_tokens(row["characteristics"]))
        score = sum(w * (2.0 if t in set(_tokens(row["asset"])) else 1.0) for t, w in weighted if t in haystack)
        if weighted and score <= 0:
            continue
        amount = row["amount"]
        try:
            amount = round(float(amount), 2)
            usable = 1.0
        except (TypeError, ValueError):
            # The source leaves blanks and prose ("refer to the supplementary
            # material") in this column; those rows are last, never dropped.
            usable = 0.0
        scored.append(
            (
                score + usable,
                {
                    "asset": row["asset"],
                    "characteristics": row["characteristics"],
                    "amount": amount,
                    "lower": row["lower"],
                    "upper": row["upper"],
                    "unit": row["unit"],
                    "geography": row["geography"],
                    "cost_feature": row["cost_feature"],
                    "source": row["source"],
                    "curve_ids": row["curve_ids"],
                },
            )
        )
    scored.sort(key=lambda item: (-item[0], item[1]["asset"]))
    return [rec for _, rec in scored[:limit]]


def stats() -> dict[str, Any]:
    """Coverage summary, for the guide and for diagnostics."""
    rows = index()
    per_family: dict[str, int] = {}
    for row in rows:
        per_family[row["hazard"]] = per_family.get(row["hazard"], 0) + 1
    return {
        "curves": len(rows),
        "applicable": sum(per_family.get(f, 0) for f in APPLICABLE),
        "per_family": per_family,
        "with_bounds": sum(1 for r in rows if r["has_bounds"] == "yes"),
        "cost_rows": len(costs()),
    }
