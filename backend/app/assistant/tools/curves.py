"""Vulnerability curves: find one in the library, or build one.

The point of these tools is that the user should never have to produce a curve
by hand. `search_curve_library` finds the published curve that fits their asset
and the selected hazard layer; `use_library_curve` loads it exactly as an
uploaded CSV would be loaded, so everything downstream — run_analysis, the app's
own vulnerability mode, the barchart's error bars — behaves identically.

When nothing in the library fits, `create_curve` builds one from explicit
points. It demands a written basis and refuses curves that are not physically
sensible, because an invented damage function that nobody can trace is worse
than no damage analysis at all.
"""

from __future__ import annotations

import csv
from typing import Any, Optional

import numpy as np

from .. import artifacts, curve_library, domain, figures
from ..conversations import Conversation
from . import tool


def _register(
    conv: Conversation,
    label: str,
    path,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Parse a curve CSV and put it on the conversation under `label`.

    Identical to what load_vulnerability_curve does for an upload — same parser,
    same conversation slot — so a library curve, a generated curve and a
    user-supplied file are interchangeable everywhere downstream.
    """
    from ...utils.geospatial import parse_vulnerability_curve_data

    central, intensity, proportion, lower, upper = parse_vulnerability_curve_data(str(path))
    conv.curves[label] = {
        "interp": central,
        "lower": lower,
        "upper": upper,
        "intensity": intensity,
        "proportion": proportion,
        "has_bounds": lower is not None and upper is not None,
        "path": path,
        "provenance": provenance,
    }
    return {
        "curve": label,
        "points": len(intensity),
        "has_uncertainty_bounds": conv.curves[label]["has_bounds"],
        "intensity_range": [float(intensity.min()), float(intensity.max())],
        "proportion_range": [float(proportion.min()), float(proportion.max())],
    }


@tool(
    "search_curve_library",
    "Find a published vulnerability curve for an asset type, from the 218 "
    "curves that ship with the app (Nirandjan et al. 2024). ALWAYS try this "
    "before asking the user for a curve — the point of the library is that they "
    "should not have to supply one. Pass the hazard LAYER you are analysing "
    "against and a plain description of the asset ('fibre optic backbone', "
    "'trunk road', 'high voltage substation', 'mobile tower'); the vocabulary "
    "is bridged to the library's own FEMA/JRC wording. Results carry the damage "
    "factor at standard anchor intensities, the study's geography and its "
    "source, so you can compare candidates without opening any file. Only three "
    "hazard families have curves in an app layer's units — flood (mm), PGA "
    "(cm/s²) and cyclone wind (km/h); for the landslide and drought layers this "
    "returns nothing and tells you what to do instead.",
    {
        "type": "object",
        "properties": {
            "hazard": {
                "type": "string",
                "description": "hazard_id or layer name you will analyse against. "
                "Restricts results to curves in that layer's unit. Omit only to "
                "browse the whole library.",
            },
            "asset": {
                "type": "string",
                "description": "Free-text description of the infrastructure, e.g. "
                "'buried fibre optic cable' or 'medium water treatment plant'.",
            },
            "sector": {
                "type": "string",
                "enum": [
                    "energy",
                    "transportation",
                    "water_and_wastewater",
                    "telecommunication",
                    "education_and_health",
                    "other",
                ],
            },
            "include_reference": {
                "type": "boolean",
                "description": "Also return curves on intensity metrics no app "
                "layer provides (landslide ground deformation, MMI, spectral "
                "acceleration). They CANNOT be applied to a layer — use them "
                "only as evidence when constructing a curve with create_curve.",
            },
            "limit": {"type": "integer", "description": "Default 8, max 25."},
        },
    },
)
def search_curve_library(
    conv: Conversation,
    hazard: Optional[str] = None,
    asset: str = "",
    sector: Optional[str] = None,
    include_reference: bool = False,
    limit: int = 8,
) -> dict[str, Any]:
    family: Optional[str] = None
    layer_note: Optional[str] = None
    layer_name: Optional[str] = None

    if hazard:
        # Accept either an app layer or a library family name.
        if hazard in curve_library.APPLICABLE or hazard in curve_library.stats()["per_family"]:
            family = hazard
        else:
            layer = domain.resolve_hazard(hazard)
            layer_name = domain.hazard_brief(layer)["name"]
            family, layer_note = curve_library.family_for_layer(layer_name or "")
            if family is None:
                return {
                    "hazard": layer_name,
                    "count": 0,
                    "curves": [],
                    "no_curve_available": layer_note,
                    "next_step": "read_guide('vulnerability_curves') — it gives the "
                    "construction recipe for this layer and the evidence behind it, "
                    "then use create_curve.",
                }

    found = curve_library.search(
        query=asset,
        family=family,
        sector=sector,
        include_reference=include_reference,
        limit=max(1, min(int(limit), 25)),
    )
    out: dict[str, Any] = {
        "hazard": layer_name or family,
        "query": asset,
        "count": len(found),
        "curves": found,
    }
    if layer_note:
        out["unit_note"] = layer_note
    if not found:
        out["note"] = (
            "nothing matched. Try a broader `asset` (the library's vocabulary is "
            "FEMA/JRC English), drop `sector`, or construct a curve with "
            "create_curve after reading read_guide('vulnerability_curves')."
        )
    else:
        out["next_step"] = (
            "pick one and call use_library_curve(curve_id) — check `geography` and "
            "`characteristics` first, and say in your answer which curve you chose "
            "and why"
        )
    return out


@tool(
    "use_library_curve",
    "Load a curve from the shipped library by its id (e.g. 'F7.1') so it can be "
    "used in an analysis. It is parsed exactly like an uploaded CSV, so pass "
    "the returned name to run_analysis as `curve`, and to ui_set_vulnerability "
    "so the app shows the user the same thing. Curves with published lower and "
    "upper bounds automatically produce a damage-cost range and error bars. "
    "Whenever you use one, state in your answer which curve it is, what asset "
    "it was derived for, where the study was done, and whether that is a good "
    "match for the user's network.",
    {
        "type": "object",
        "properties": {
            "curve_id": {
                "type": "string",
                "description": "From search_curve_library, e.g. 'F7.1'.",
            },
            "name": {
                "type": "string",
                "description": "Short label to refer to it later. Defaults to the "
                "curve id.",
            },
        },
        "required": ["curve_id"],
    },
)
def use_library_curve(
    conv: Conversation, curve_id: str, name: Optional[str] = None
) -> dict[str, Any]:
    row = curve_library.get(curve_id)
    if row["output"] != "damage factor 0-1":
        raise ValueError(
            f"curve {row['curve_id']} outputs a {row['output']}, so multiplying it "
            "by a replacement value is meaningless. Pick a curve whose `output` is "
            "'damage factor 0-1'."
        )
    label = (name or row["curve_id"]).strip()
    provenance = {
        "kind": "library",
        "curve_id": row["curve_id"],
        "asset": row["asset"],
        "characteristics": row["characteristics"],
        "geography": row["geography"],
        "source": row["source"],
        "derivation": row["derivation"],
        "citation": "Nirandjan et al. (2024), NHESS 24, 4341-4369",
    }
    out = _register(conv, label, curve_library.path_for(row), provenance)
    out.update(
        {
            "curve_id": row["curve_id"],
            "asset": row["asset"],
            "characteristics": row["characteristics"],
            "geography": row["geography"],
            "source": row["source"],
            "applies_to": row["app_layer_match"] or None,
            "unit": row["unit"],
            "damage_at": row["damage_at"],
            "cite": provenance["citation"],
            "note": (
                f"curve {label!r} ready — pass curve='{label}' to run_analysis with a "
                "replacement_value, and to ui_set_vulnerability so the app agrees"
            ),
        }
    )
    if row["central_basis"] != "published":
        out["caveat"] = (
            f"the central curve is the {row['central_basis']} — the source published "
            "only bounds"
        )
    if not row["app_layer_match"]:
        # Loading a reference curve is allowed — plot_curve needs it loaded to
        # draw it as evidence — but running an analysis with one silently
        # compares an app layer against a different physical quantity.
        out["warning"] = (
            f"{row['curve_id']} is a REFERENCE curve: its axis is "
            f"{row['intensity_metric']} in {row['unit']}, which no app hazard layer "
            "provides. Do not pass it to run_analysis — the intensities would not "
            "mean the same thing. Use it as evidence for create_curve, or to plot "
            "alongside a constructed curve."
        )
    if row["bounds_note"]:
        out["bounds_note"] = row["bounds_note"]
    return out


@tool(
    "create_curve",
    "Build a vulnerability curve from explicit points, for when the library has "
    "nothing suitable — an asset type it does not cover, or a hazard layer with "
    "no curves at all (landslide susceptibility, drought). Read "
    "read_guide('vulnerability_curves') FIRST: it carries the functional forms, "
    "published anchor values and the validation rules this tool enforces. "
    "Intensity must be in the hazard LAYER's units (mm, km/h, cm/s², class "
    "1-5). Supply `lower` and `upper` whenever you can — a constructed curve "
    "without an uncertainty band claims a precision it does not have. The curve "
    "is saved as a downloadable CSV the user can inspect, edit and re-use. "
    "`basis` is mandatory and goes into the record: name the source, the "
    "analogue curve or the reasoning behind every anchor, and say plainly which "
    "parts are judgement.",
    {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Short label, e.g. 'fibre_flood'."},
            "intensity": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Increasing intensities in the hazard layer's unit. "
                "Start at 0 and cover the layer's full range (get_hazard_stats).",
            },
            "damage": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Central proportion destroyed at each intensity, "
                "0-1, non-decreasing.",
            },
            "lower": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Lower-bound proportions, same length. Optional but "
                "strongly preferred.",
            },
            "upper": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Upper-bound proportions, same length.",
            },
            "asset": {
                "type": "string",
                "description": "What the curve is for, e.g. 'buried fibre optic cable'.",
            },
            "hazard": {
                "type": "string",
                "description": "The hazard layer it is meant for — recorded and "
                "checked against the intensity range you gave.",
            },
            "basis": {
                "type": "string",
                "description": "REQUIRED. Where each anchor comes from: a published "
                "source, a library curve used as an analogue, or your own "
                "engineering judgement — stated as such.",
            },
        },
        "required": ["name", "intensity", "damage", "basis"],
    },
)
def create_curve(
    conv: Conversation,
    name: str,
    intensity: list[float],
    damage: list[float],
    basis: str,
    lower: Optional[list[float]] = None,
    upper: Optional[list[float]] = None,
    asset: Optional[str] = None,
    hazard: Optional[str] = None,
) -> dict[str, Any]:
    label = name.strip()
    if not label:
        raise ValueError("name must not be empty")
    if len((basis or "").strip()) < 20:
        raise ValueError(
            "basis must actually say where the numbers come from — cite the source, "
            "the analogue curve, or the reasoning, and mark what is judgement. A "
            "constructed curve with no traceable basis is not usable in a report."
        )

    x = np.asarray(intensity, dtype=float)
    y = np.asarray(damage, dtype=float)
    if x.size < 2:
        raise ValueError("a curve needs at least 2 points")
    if x.size != y.size:
        raise ValueError(f"intensity has {x.size} points but damage has {y.size}")
    if not np.all(np.diff(x) > 0):
        raise ValueError("intensity must be strictly increasing")
    if np.any(y < 0) or np.any(y > 1):
        raise ValueError("damage must be a proportion in [0, 1], not a percentage")
    if np.any(np.diff(y) < -1e-9):
        bad = int(np.argmin(np.diff(y)))
        raise ValueError(
            f"damage must be non-decreasing, but it falls from {y[bad]:g} at "
            f"{x[bad]:g} to {y[bad + 1]:g} at {x[bad + 1]:g}. More hazard cannot "
            "mean less damage."
        )

    bounds: dict[str, np.ndarray] = {}
    for key, values in (("lower", lower), ("upper", upper)):
        if values is None:
            continue
        arr = np.asarray(values, dtype=float)
        if arr.size != x.size:
            raise ValueError(f"{key} has {arr.size} points but intensity has {x.size}")
        if np.any(arr < 0) or np.any(arr > 1):
            raise ValueError(f"{key} must lie in [0, 1]")
        bounds[key] = arr
    if len(bounds) == 1:
        raise ValueError("give both `lower` and `upper`, or neither")
    if bounds and (
        np.any(bounds["lower"] > y + 1e-9) or np.any(bounds["upper"] < y - 1e-9)
    ):
        raise ValueError("every point must satisfy lower <= damage <= upper")

    # Sanity checks that do not block, but must reach the user.
    warnings: list[str] = []
    if x[0] > 0:
        warnings.append(
            f"the curve starts at {x[0]:g}, not 0 — below that the app clamps to "
            f"{y[0]:g}, so it will report damage at zero intensity if {y[0]:g} > 0"
        )
    if y[0] > 0:
        warnings.append(
            f"damage is {y[0]:g} at the lowest tabulated intensity; a damage "
            "function normally starts at zero and has a non-zero onset threshold"
        )
    if not bounds:
        warnings.append(
            "no uncertainty band: the result will be a single damage-cost number "
            "with no range, which overstates how well this is known"
        )
    if y[-1] >= 0.999:
        warnings.append(
            "the curve reaches total loss — plausible for a point asset, but for a "
            "linear asset priced per metre the per-metre replacement value embeds "
            "earthworks and right-of-way that a hazard does not destroy (HAZUS caps "
            "roadway damage at 0.70)"
        )
    if hazard:
        try:
            layer = domain.resolve_hazard(hazard)
            brief = domain.hazard_brief(layer)
            family, note = curve_library.family_for_layer(brief["name"] or "")
            if family is not None:
                warnings.append(
                    f"{brief['name']} HAS library curves ({family}) — check "
                    "search_curve_library before relying on a constructed one"
                )
            hazard = brief["name"]
        except ValueError:
            pass  # a free-text hazard label is fine; it is only recorded

    conv._counter += 1
    dest = conv.workdir / f"{conv._counter}_{label.replace(' ', '_')}.csv"
    with dest.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        if bounds:
            writer.writerow(["intensity", "lower", "central", "upper"])
            rows = zip(x, bounds["lower"], y, bounds["upper"])
        else:
            writer.writerow(["intensity", "proportion_destroyed"])
            rows = zip(x, y)
        for row in rows:
            writer.writerow([f"{v:.6g}" for v in row])

    provenance = {
        "kind": "constructed",
        "asset": asset,
        "hazard": hazard,
        "basis": basis.strip(),
    }
    out = _register(conv, label, dest, provenance)
    art = artifacts.add(dest, f"{label}.csv", "csv", f"Vulnerability curve: {label}", conv.id)
    out.update(
        {
            "asset": asset,
            "hazard": hazard,
            "basis": basis.strip(),
            "artifact": art.public(),
            "note": (
                f"curve {label!r} ready — pass curve='{label}' to run_analysis. This "
                "curve was CONSTRUCTED, not published: say so wherever its numbers "
                "appear, and give the basis above."
            ),
        }
    )
    if warnings:
        out["warnings"] = warnings
    return out


@tool(
    "plot_curve",
    "Draw one or more vulnerability curves on an intensity axis, with the "
    "uncertainty band shaded where a curve has one. Use it to show the user "
    "which curve you picked, to compare candidates before choosing, and as a "
    "report exhibit whenever the curve is doing real work in the answer — "
    "especially for a constructed curve, where the reader has to be able to see "
    "what was assumed.",
    {
        "type": "object",
        "properties": {
            "curves": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Names of curves already loaded in this conversation "
                "(from use_library_curve, create_curve or load_vulnerability_curve).",
            },
            "title": {"type": "string"},
            "xlabel": {
                "type": "string",
                "description": "The intensity axis with its unit, e.g. "
                "'Inundation depth (mm)'.",
            },
            "filename": {"type": "string", "description": "e.g. curve.png"},
        },
        "required": ["curves"],
    },
)
def plot_curve(
    conv: Conversation,
    curves: list[str],
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    filename: str = "vulnerability_curve.png",
) -> dict[str, Any]:
    series = []
    for name in curves:
        curve = conv.curves.get(name)
        if curve is None:
            raise ValueError(
                f"no curve {name!r}; loaded curves: {sorted(conv.curves) or 'none'}"
            )
        entry = {
            "name": name,
            "intensity": curve["intensity"],
            "central": curve["proportion"],
        }
        if curve["has_bounds"]:
            entry["lower"] = curve["lower"](curve["intensity"])
            entry["upper"] = curve["upper"](curve["intensity"])
        series.append(entry)

    conv._counter += 1
    dest = conv.workdir / f"{conv._counter}_{filename}"
    figures.curve_plot(dest, series, title=title, xlabel=xlabel)
    art = artifacts.add(dest, filename, "chart", title or filename, conv.id)
    return {"artifact": art.public()}


@tool(
    "find_replacement_cost",
    "Look up a replacement value for an asset in the 179-row cost database that "
    "ships with the curve library (Table D3 of Nirandjan et al. 2024). "
    "replacement_value is the other input a damage analysis needs, and the user "
    "should not have to invent it either — but these are EUROS from specific "
    "countries at the source studies' price levels, so treat any figure as an "
    "assumption, state its geography and cost basis, and sanity-check it "
    "against the rest of its asset group before using it. Set `basis` to 'unit' "
    "for a point dataset and 'metre' for a line dataset, which is what the app's "
    "replacement_value means in each case.",
    {
        "type": "object",
        "properties": {
            "asset": {
                "type": "string",
                "description": "Free-text asset description, e.g. 'railway' or "
                "'communication tower'.",
            },
            "basis": {
                "type": "string",
                "enum": ["unit", "metre", "area"],
                "description": "'unit' = per feature (point datasets), 'metre' = "
                "per metre (line datasets), 'area' = per m².",
            },
            "limit": {"type": "integer", "description": "Default 8, max 25."},
        },
    },
)
def find_replacement_cost(
    conv: Conversation,
    asset: str = "",
    basis: Optional[str] = None,
    limit: int = 8,
) -> dict[str, Any]:
    rows = curve_library.search_costs(asset, basis=basis, limit=max(1, min(int(limit), 25)))
    return {
        "query": asset,
        "basis": basis,
        "count": len(rows),
        "costs": rows,
        "currency": "EUR at the source studies' price levels (the paper harmonises "
        "to 2010 euro); no inflation or exchange-rate adjustment is applied",
        "caution": (
            "figures vary by an order of magnitude with geography, and a few rows "
            "are mislabelled in the source (some road costs are per kilometre in a "
            "per-metre column). Compare candidates before picking one, and report "
            "the value, its geography and its cost basis as an assumption."
        ),
    }
