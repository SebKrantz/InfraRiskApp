"""Expected annual damage: a hazard family's return-period losses, integrated
over annual exceedance probability.

`ead()` is the arithmetic and matches AGUI's `aei.cba.ead`: the losses against
p = 1/T by the trapezoid rule, the largest-T loss held down to p = 0, and — the
default — no loss at events more frequent than the smallest return period, a
lower bound. The tool gathers one damage figure per return-period layer, by
running the analyses or by reusing run_analysis results, and refuses anything
that would make the points incomparable: one return period, or a threshold,
curve or replacement value that differs between them.
"""

from __future__ import annotations

import re
from typing import Any, Optional

import numpy as np

from .. import curve_library, domain
from ..conversations import Conversation
from . import tool
from .analysis import _ANALYSIS_PROPS, _curve

# Catalogue layers of each family, by hazard_id prefix. Each has one return
# period per layer, and the flood and cyclone families several climate variants.
FAMILIES = {
    "flood": "flood_hazard_",
    "cyclone": "tropical_cyclone_wind_",
    "pga": "peak_ground_acceleration_",
}

_RP = re.compile(r"(\d+)[\s-]*years?\b", re.IGNORECASE)

LANDSLIDE_NOTE = (
    "Landslide susceptibility has no return period, so it has no EAD. It can be "
    "annualised only by assuming an annual frequency for its scenario, which the "
    "caller must state: EAD = scenario loss x that frequency, labelled as an "
    "assumption."
)


def ead(
    return_periods: list[float],
    losses: list[float],
    protection_rp: Optional[float] = None,
    upper_bound: bool = False,
) -> float:
    """Expected annual damage from one loss per return period T (years).

    The losses are integrated over the annual exceedance probability p = 1/T
    by the trapezoid rule, and the largest-T loss is held down to p = 0. By
    default there is no loss at events more frequent than the smallest T, which
    itself does its full loss: a lower bound for an asset without protection.
    `protection_rp` sets a known protection standard instead — the losses at
    return periods up to AND including it are zero, and the curve rises linearly
    from a zero loss at it to the next return period's. One equal to the
    smallest T therefore comes out below the default, which keeps that loss.
    `upper_bound` joins the curve linearly from a zero loss at T = 1 instead.
    """
    t = np.asarray(return_periods, dtype=float)
    d = np.asarray(losses, dtype=float)
    if t.shape != d.shape:
        raise ValueError(f"{t.size} return periods but {d.size} losses")
    if np.unique(t).size < 2:
        raise ValueError(
            "expected annual damage needs at least two return periods; one event "
            "loss is not an annual figure"
        )
    if np.unique(t).size != t.size:
        raise ValueError(f"a return period appears twice: {sorted(t.tolist())}")
    if (t < 1).any():
        raise ValueError(f"return periods must be at least 1 year, not {t.tolist()}")
    if not np.isfinite(d).all() or (d < 0).any():
        raise ValueError(f"losses must be finite and non-negative, not {d.tolist()}")
    if protection_rp is not None and upper_bound:
        raise ValueError(
            "protection_rp and upper_bound are alternatives: a protection standard "
            "already fixes the losses below the smallest return period"
        )

    order = np.argsort(t)
    t, d = t[order], d[order]
    if protection_rp is not None:
        if not 1 <= protection_rp < t[-1]:
            raise ValueError(
                f"protection_rp must be at least 1 year and below the largest return "
                f"period ({t[-1]:g}); got {protection_rp:g}"
            )
        keep = t > protection_rp
        t = np.concatenate(([protection_rp], t[keep]))
        d = np.concatenate(([0.0], d[keep]))
    elif upper_bound:
        t = np.concatenate(([1.0], t))
        d = np.concatenate(([0.0], d))
    p = 1.0 / t
    area = ((p[:-1] - p[1:]) * (d[:-1] + d[1:])).sum() / 2
    return float(area + p[-1] * d[-1])


def return_period(name: str) -> tuple[Optional[float], str]:
    """(T, variant) from a layer name: 'Flood Hazard 25 Years - SSP1 Lower bound'
    gives (25, 'Flood Hazard - SSP1 Lower bound'). T is None without one."""
    match = _RP.search(name)
    if match is None:
        return None, name
    variant = _RP.sub("", name, count=1)
    variant = re.sub(r"\s+", " ", variant)
    variant = re.sub(r"(\s*-\s*){2,}", " - ", variant)
    return float(match.group(1)), variant.strip(" -")


def _point(haz: dict) -> dict[str, Any]:
    hid = haz["hazard_id"]
    name = domain._clean(haz.get("hazard")) or hid
    rp, variant = return_period(name)
    if rp is None:
        raise ValueError(
            f"{name!r} has no return period, so it cannot enter an expected annual "
            f"damage. {LANDSLIDE_NOTE if 'landslide' in name.lower() else ''}".strip()
        )
    family = next((f for f, prefix in FAMILIES.items() if hid.startswith(prefix)), None)
    return {"hazard_id": hid, "hazard": name, "return_period": rp, "variant": variant,
            "family": family}


def _group(points: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Points by climate variant, each sorted by return period and checked, all
    from one family — one curve cannot fit flood depth and wind speed."""
    outside = [pt["hazard"] for pt in points if pt["family"] is None]
    if outside:
        raise ValueError(
            f"{outside} belong to no return-period family; expected annual damage "
            f"takes the layers of one of {sorted(FAMILIES)}"
        )
    families = sorted({pt["family"] for pt in points})
    if len(families) > 1:
        raise ValueError(
            f"the layers span the {' and '.join(families)} families; expected annual "
            "damage integrates one family, under one curve — run one per family"
        )
    groups: dict[str, list[dict[str, Any]]] = {}
    for pt in points:
        groups.setdefault(pt["variant"], []).append(pt)
    for variant, pts in groups.items():
        pts.sort(key=lambda pt: pt["return_period"])
        rps = [pt["return_period"] for pt in pts]
        if len(set(rps)) < 2:
            raise ValueError(
                f"{variant!r} has a single return period ({rps[0]:g} years); expected "
                "annual damage needs at least two. One event loss is not an annual "
                "figure."
            )
        if len(set(rps)) != len(rps):
            raise ValueError(f"{variant!r} has the same return period twice: {rps}")
    return groups


def _ead_curve(conv: Conversation, name: str) -> tuple[dict[str, Any], str]:
    """A loaded curve by name, or a library curve by id, loaded on first use."""
    try:
        return _curve(conv, name), name
    except ValueError:
        try:
            curve_library.get(name)
        except ValueError:
            raise ValueError(
                f"no curve {name!r}: neither a loaded curve "
                f"({sorted(conv.curves) or 'none'}) nor a library id"
            ) from None
    from .curves import use_library_curve

    label = use_library_curve(conv, name)["curve"]
    return conv.curves[label], label


def _losses(result: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    out = {
        "damage_cost": float(result["total_damage_cost"]),
        "affected_share_pct": summary.get("affected_share_pct"),
    }
    if result.get("total_damage_cost_lower") is not None:
        out["damage_cost_lower"] = float(result["total_damage_cost_lower"])
        out["damage_cost_upper"] = float(result["total_damage_cost_upper"])
    return out


@tool(
    "expected_annual_damage",
    "Expected annual damage (EAD) of a dataset from one hazard family's "
    "return-period layers: one damage analysis per layer, integrated over the "
    "annual exceedance probability p = 1/T by the trapezoid rule. Give the "
    "layers as `family` (flood 25/50/100 yr x existing climate, SSP1, SSP5; "
    "cyclone 25/50/100 yr with and without climate change; pga 250/475/975 yr), "
    "as `hazards`, or as `analyses` — run_analysis results to reuse, which must "
    "share one threshold, curve and replacement value. Returns one EAD per "
    "climate variant, never averaged, with the per-return-period losses, and "
    "a lower/upper range when the curve has bounds. Default is a LOWER BOUND: "
    "no loss at events more frequent than the smallest return period; "
    "`protection_rp` sets a known protection standard, `upper_bound` joins the "
    "curve from zero loss at T = 1 instead. The largest return period's loss is "
    "held to p = 0. Read read_guide('multi_hazard') first. Runs every layer it "
    "has not sampled before: tens of seconds each.",
    {
        "type": "object",
        "properties": {
            "file_id": _ANALYSIS_PROPS["file_id"],
            "family": {"type": "string", "enum": sorted(FAMILIES)},
            "hazards": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Return-period layers (hazard_ids or names) of one "
                "family instead of all of it; grouped by climate variant, at least two "
                "return periods in each.",
            },
            "analyses": {
                "type": "array",
                "items": {"type": "string"},
                "description": "`stored_as` names of run_analysis results with a "
                "curve, instead of running the layers. Takes no threshold, curve or "
                "replacement_value — they come from the analyses.",
            },
            "threshold": _ANALYSIS_PROPS["threshold"],
            "curve": {
                "type": "string",
                "description": "A loaded curve's name (use_library_curve, "
                "create_curve, load_vulnerability_curve) or a library curve id, "
                "loaded on first use.",
            },
            "replacement_value": _ANALYSIS_PROPS["replacement_value"],
            "currency": {
                "type": "string",
                "description": "Currency of the replacement value, e.g. 'USD'.",
            },
            "price_basis": {
                "type": "string",
                "description": "Price basis of the replacement value, e.g. '2024 "
                "prices' or '2010 EUR (find_replacement_cost)'.",
            },
            "protection_rp": {
                "type": "number",
                "description": "A known protection or design standard in years: "
                "losses at return periods up to and including it are zero, so one "
                "equal to the smallest return period gives less than the default, "
                "which keeps that return period's loss.",
            },
            "upper_bound": {
                "type": "boolean",
                "description": "Join the curve linearly from zero loss at T = 1 "
                "(the yearly event) instead of the lower-bound default.",
            },
        },
    },
)
def expected_annual_damage(
    conv: Conversation,
    file_id: Optional[str] = None,
    family: Optional[str] = None,
    hazards: Optional[list[str]] = None,
    analyses: Optional[list[str]] = None,
    threshold: Optional[float] = None,
    curve: Optional[str] = None,
    replacement_value: Optional[float] = None,
    currency: Optional[str] = None,
    price_basis: Optional[str] = None,
    protection_rp: Optional[float] = None,
    upper_bound: bool = False,
) -> dict[str, Any]:
    import pandas as pd

    if sum(bool(x) for x in (family, hazards, analyses)) != 1:
        raise ValueError("give the layers as exactly one of family, hazards or analyses")
    if protection_rp is not None and upper_bound:
        raise ValueError(
            "protection_rp and upper_bound are alternatives: a protection standard "
            "already fixes the losses below the smallest return period"
        )

    if analyses:
        if any(x is not None for x in (threshold, curve, replacement_value)):
            raise ValueError(
                "with `analyses`, the threshold, curve and replacement value are the "
                "analyses' own — leave them out"
            )
        points, setting = _reused(conv, analyses, file_id)
        info = domain.get_dataset(setting["file_id"])
    else:
        if not file_id:
            raise ValueError("file_id is required with `family` or `hazards`")
        info = domain.get_dataset(file_id)
        if not curve:
            raise ValueError(
                "expected annual damage needs a vulnerability curve: a loaded curve's "
                "name or a library id"
            )
        if replacement_value is None or replacement_value <= 0:
            raise ValueError(
                "replacement_value is required and must be greater than zero (per "
                "feature for points, per metre for lines)"
            )
        if family:
            if family not in FAMILIES:
                raise ValueError(f"family must be one of {sorted(FAMILIES)}, not {family!r}")
            layers = [
                h for hid, h in domain.catalogue().items() if hid.startswith(FAMILIES[family])
            ]
        else:
            layers = [domain.resolve_hazard(ref) for ref in hazards]
        points = [_point(h) for h in layers]

    # Check the whole set before the first slow raster read.
    groups = _group(points)
    if protection_rp is not None:
        top = min(pts[-1]["return_period"] for pts in groups.values())
        if not 1 <= protection_rp < top:
            raise ValueError(
                f"protection_rp must be at least 1 year and below the largest return "
                f"period ({top:g}); got {protection_rp:g}"
            )

    if not analyses:
        vc, curve = _ead_curve(conv, curve)
        layer_of = {h["hazard_id"]: h for h in layers}
        for pt in points:
            haz = layer_of[pt["hazard_id"]]
            result = domain.run_exposure(
                file_id,
                haz,
                threshold=threshold,
                vulnerability_curve_interp=vc["interp"],
                replacement_value=replacement_value,
                vulnerability_curve_lower_interp=vc["lower"],
                vulnerability_curve_upper_interp=vc["upper"],
            )
            pt.update(_losses(result, domain.summarise(result, info, haz, threshold)))
        setting = {
            "file_id": file_id,
            "threshold": threshold,
            "curve": curve,
            "replacement_value": replacement_value,
            "provenance": vc.get("provenance"),
        }

    bounded = all("damage_cost_lower" in pt for pts in groups.values() for pt in pts)
    variants = []
    for variant, pts in groups.items():
        rps = [pt["return_period"] for pt in pts]
        entry: dict[str, Any] = {
            "variant": variant,
            "return_periods": rps,
            "losses": [
                {
                    "annual_exceedance_probability": 1.0 / pt["return_period"],
                    **{k: v for k, v in pt.items() if k not in ("variant", "family")},
                }
                for pt in pts
            ],
        }
        central = [pt["damage_cost"] for pt in pts]
        entry["ead_central"] = ead(rps, central, protection_rp, upper_bound)
        if bounded:
            for side in ("lower", "upper"):
                entry[f"ead_curve_{side}"] = ead(
                    rps, [pt[f"damage_cost_{side}"] for pt in pts], protection_rp, upper_bound
                )
        if any(b < a for a, b in zip(central, central[1:])):
            entry["warning"] = (
                "the loss falls between two return periods; the layers are separate "
                "model runs — check the per-layer analyses before reporting this EAD"
            )
        variants.append(entry)

    t_min = min(min(pt["return_period"] for pt in pts) for pts in groups.values())
    t_max = max(max(pt["return_period"] for pt in pts) for pts in groups.values())
    if upper_bound:
        estimate = "upper bound"
        below = (
            f"Upper bound: the loss rises linearly from zero at T = 1 year (the yearly "
            f"event) to the {t_min:g}-year loss."
        )
    elif protection_rp is not None:
        estimate = f"protected to {protection_rp:g} years"
        below = (
            f"Protection standard {protection_rp:g} years: no loss at return periods up "
            "to and including it; the loss rises linearly from zero there to the next "
            "layer's."
        )
    else:
        estimate = "lower bound"
        below = (
            f"Lower bound: no loss at events more frequent than the {t_min:g}-year, "
            "which itself does its full loss."
        )
    geometry_type = info["geometry_type"]
    table = pd.DataFrame(
        [{"variant": e["variant"], **loss} for e in variants for loss in e["losses"]]
    )
    out: dict[str, Any] = {
        "file_id": setting["file_id"],
        "family": points[0]["family"],
        "geometry_type": geometry_type,
        "threshold": setting["threshold"],
        "curve": setting["curve"],
        "curve_provenance": setting["provenance"],
        "curve_has_bounds": bounded,
        "replacement_value": {
            "value": setting["replacement_value"],
            "basis": "per feature" if geometry_type == "Point" else "per metre",
            "currency": currency,
            "price_basis": price_basis,
        },
        "estimate": estimate,
        "conventions": [
            "EAD = the damage integrated over the annual exceedance probability "
            "p = 1/T by the trapezoid rule; it is in the replacement value's currency "
            "and price basis.",
            below,
            f"Tail: the {t_max:g}-year loss is held for every rarer event, down to p = 0.",
            "One EAD per climate variant; never average them across SSPs.",
            "ead_curve_lower / ead_curve_upper integrate the curve's lower and upper "
            "damage bounds — the curve's uncertainty, not the frequency convention, and "
            "not the climate range the 'SSP1 Lower bound' / 'SSP5 Upper bound' layers "
            "span.",
            "A damage figure from one layer is an event loss, never an annual one.",
        ],
        "variants": variants,
        "landslide": LANDSLIDE_NOTE,
        "stored_as": conv.store_result("ead", table),
    }
    if currency is None or price_basis is None:
        out["state"] = (
            "currency or price basis not given — state both beside the EAD in any report"
        )
    return out


def _reused(
    conv: Conversation, names: list[str], file_id: Optional[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return-period points from stored run_analysis results, which must agree on
    the dataset, threshold, curve and replacement value."""
    from .output import _analysis

    points, settings = [], []
    for name in names:
        result = _analysis(conv, name)
        meta = result.get("_assistant_meta", {})
        if result.get("total_damage_cost") is None or "curve" not in meta:
            raise ValueError(
                f"{name!r} has no damage cost — run it with a curve and a "
                "replacement_value"
            )
        haz = domain.resolve_hazard(meta["hazard_id"])
        info = domain.get_dataset(meta["file_id"])
        pt = _point(haz)
        pt.update(_losses(result, domain.summarise(result, info, haz, meta["threshold"])))
        pt["analysis"] = name
        points.append(pt)
        settings.append(meta)

    for key, what in (
        ("file_id", "datasets"),
        ("threshold", "thresholds"),
        ("curve_path", "curves"),
        ("replacement_value", "replacement values"),
    ):
        # 100 and 100.0 are the same threshold
        seen = {
            float(v) if isinstance(v, (int, float)) else v
            for v in (m.get(key) for m in settings)
        }
        if len(seen) > 1:
            per = ", ".join(
                f"{n}: {m.get('curve' if key == 'curve_path' else key)}"
                for n, m in zip(names, settings)
            )
            raise ValueError(
                f"the analyses mix {what} ({per}); every return period needs the same "
                "dataset, threshold, curve and replacement value"
            )
    first = settings[0]
    if file_id and file_id != first["file_id"]:
        raise ValueError(f"the analyses are of {first['file_id']!r}, not {file_id!r}")
    curve = conv.curves.get(first["curve"], {})
    return points, {
        "file_id": first["file_id"],
        "threshold": first["threshold"],
        "curve": first["curve"],
        "replacement_value": first["replacement_value"],
        "provenance": curve.get("provenance"),
    }
