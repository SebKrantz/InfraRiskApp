"""Expected annual damage: a hazard family's return-period losses, integrated
over annual exceedance probability.

`ead()` is the arithmetic: the losses against p = 1/T by the trapezoid rule, the
largest-T loss held down to p = 0, and — the default — no loss at events more
frequent than the smallest return period, a lower bound that matches AGUI's
`aei.cba.ead`. The upper bound does NOT: it extends the loss line log-linearly in
T to where it reaches zero (see `_extend_frequent`), not linearly in p from zero
loss at T = 1, which gave 7-13 % of an asset a year. The tool gathers one damage
figure per return-period layer, by running the analyses or by reusing
run_analysis results, and refuses anything that would make the points
incomparable: one return period, or a threshold, curve or replacement value that
differs between them. Each EAD also comes with its share of the asset's
replacement value and a plausibility flag.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

import numpy as np

from .. import curve_library, domain
from ..conversations import Conversation
from . import tool
from .analysis import _ANALYSIS_PROPS, _curve, _damage_inputs
from .data import resolve_dataset

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
    "caller must state: EAD = scenario loss x that frequency (central 1/20, range "
    "1/50-1/10 unless the brief gives one), labelled as an assumption."
)

# The upper bound looks no further than this factor below the smallest return
# period, and never under one year.
EXTENSION_FACTOR = 10.0
# Above this share of the asset's replacement value a year, an EAD is not a figure
# to plan with: the asset would be lost to hazards in under a century.
PLAUSIBLE_SHARE = 0.01


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
    `upper_bound` instead carries the curve on to more frequent events: see
    `_extend_frequent`.
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
        t, d = _extend_frequent(t, d)
    p = 1.0 / t
    area = ((p[:-1] - p[1:]) * (d[:-1] + d[1:])).sum() / 2
    return float(area + p[-1] * d[-1])


def _extend_frequent(t: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The upper bound: (T, loss) with the loss line carried on past the smallest
    return period to more frequent events (t sorted, d the losses).

    The line is straight in ln T through the two smallest return periods — losses
    shrink by a fixed amount per halving of T, as they do between the layers — and
    runs down to where it reaches zero, but no further than EXTENSION_FACTOR below
    the smallest return period and never under 1 year. Below that there is no loss,
    as in the lower bound. It assumes the trend between the two most frequent
    layers continues. (Joining zero loss at T = 1 linearly in p instead credited
    frequent events with losses no layer shows: 7-13 % of a railway a year.) The
    span is cut into short steps so the trapezoid follows the curve, which is
    convex in p.
    """
    if d[0] <= 0:
        return t, d
    slope = max(0.0, (d[1] - d[0]) / np.log(t[1] / t[0]))  # loss per e-fold of T
    zero = t[0] * np.exp(-d[0] / slope) if slope > 0 else 0.0
    reach = max(zero, t[0] / EXTENSION_FACTOR, 1.0)
    if reach >= t[0]:
        return t, d
    grid = np.geomspace(reach, t[0], 41)[:-1]
    loss = np.maximum(0.0, d[0] - slope * np.log(t[0] / grid))
    return np.concatenate((grid, t)), np.concatenate((loss, d))


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


def _asset_value(result: dict[str, Any], info: dict, values: Any) -> Optional[float]:
    """Replacement value of the whole dataset, in the EAD's currency: what a year's
    EAD is a share of. `values` is the scalar or per-row array the analyses used;
    a line's is per metre, so its length comes from the analysis's segments."""
    v = np.asarray(values, dtype=float)
    if info["geometry_type"] == "Point":
        return float(v.sum() if v.ndim else v * info["feature_count"])
    if v.ndim == 0:
        return float(v * (result["affected_meters"] + result["unaffected_meters"]))
    segments = result["full_gdf"]
    rows = segments["line_id"].to_numpy(dtype=int) - 1
    return float((v[rows] * segments["length_m"].to_numpy(dtype=float)).sum())


def _plausibility(ead_value: float, asset_value: float) -> dict[str, Any]:
    """The EAD as a share of the asset's replacement value per year, and a flag
    when that is more than PLAUSIBLE_SHARE."""
    share = ead_value / asset_value
    out: dict[str, Any] = {
        "ead_pct_of_asset_per_year": float(f"{100 * share:.3g}"),
        "plausibility": "ok" if share <= PLAUSIBLE_SHARE else "high",
    }
    if share > PLAUSIBLE_SHARE:
        out["plausibility_note"] = (
            f"{100 * share:.1f} % of the replacement value every year is more than "
            f"{100 * PLAUSIBLE_SHARE:g} % a year: as if the whole asset were lost to "
            f"this hazard in {1 / share:.0f} years. It is not a planning bound and "
            "not a stress case. Check the curve, the threshold, the replacement value "
            "and the frequency convention, and report it as implausible rather than "
            "as an upper figure."
        )
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
    "a lower/upper range when the curve has bounds. Default (the central "
    "figure) is a LOWER BOUND: no loss at events more frequent than the smallest "
    "return period; `protection_rp` sets a known protection standard; "
    "`upper_bound` instead extends the loss line to more frequent events, "
    "straight in ln T through the two smallest return periods, down to where it "
    "reaches zero (at most 10 times below the smallest return period, never under "
    "1 year). The largest return period's loss is held to p = 0. Each result also "
    "gives the EAD as a share of the asset's replacement value per year "
    "(ead_pct_of_asset_per_year) and a `plausibility` flag: above about 1 % a year "
    "it is not a planning figure and not a bound. EADs of different hazards add "
    "(capped at the replacement value); read read_guide('multi_hazard') first. "
    "Runs every layer it has not sampled before: tens of seconds each.",
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
                "replacement values — they come from the analyses.",
            },
            "threshold": _ANALYSIS_PROPS["threshold"],
            "curve": {
                "type": "string",
                "description": "A loaded curve's name (use_library_curve, "
                "create_curve, load_vulnerability_curve) or a library curve id, "
                "loaded on first use.",
            },
            "replacement_value": _ANALYSIS_PROPS["replacement_value"],
            "replacement_value_column": _ANALYSIS_PROPS["replacement_value_column"],
            "replacement_value_map": _ANALYSIS_PROPS["replacement_value_map"],
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
                "description": "Upper bound instead of the lower-bound default: extend "
                "the loss line to events more frequent than the smallest return "
                "period, log-linearly in return period, to where it reaches zero "
                "(no further than 10 times below it, never under 1 year).",
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
    replacement_value_column: Optional[str] = None,
    replacement_value_map: Optional[dict] = None,
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
        given = (threshold, curve, replacement_value, replacement_value_column,
                 replacement_value_map)
        if any(x is not None for x in given):
            raise ValueError(
                "with `analyses`, the threshold, curve and replacement value are the "
                "analyses' own — leave them out"
            )
        # An uploaded file's name names the dataset its analyses were run on.
        points, setting = _reused(conv, analyses, file_id and resolve_dataset(conv, file_id))
        info = domain.get_dataset(setting["file_id"], conv.owner)
    else:
        if not file_id:
            raise ValueError("file_id is required with `family` or `hazards`")
        file_id = resolve_dataset(conv, file_id)
        info = domain.get_dataset(file_id, conv.owner)
        if not curve:
            raise ValueError(
                "expected annual damage needs a vulnerability curve: a loaded curve's "
                "name or a library id"
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
        curve = _ead_curve(conv, curve)[1]
        vc, values, per_feature = _damage_inputs(
            conv, info, curve, replacement_value, replacement_value_column,
            replacement_value_map,
        )
        layer_of = {h["hazard_id"]: h for h in layers}
        for pt in points:
            haz = layer_of[pt["hazard_id"]]
            result = domain.run_exposure(
                file_id,
                haz,
                threshold=threshold,
                vulnerability_curve_interp=vc["interp"],
                replacement_value=values,
                vulnerability_curve_lower_interp=vc["lower"],
                vulnerability_curve_upper_interp=vc["upper"],
                owner=conv.owner,
                curve_digest=domain.curve_digest(vc),
            )
            pt.update(_losses(result, domain.summarise(result, info, haz, threshold)))
        asset_value = _asset_value(result, info, values)
        setting = {
            "file_id": file_id,
            "threshold": threshold,
            "curve": curve,
            "replacement_value": replacement_value,
            "replacement_value_column": replacement_value_column,
            "replacement_value_map": replacement_value_map,
            "per_feature": per_feature,
            "provenance": vc.get("provenance"),
        }
    else:
        asset_value = setting["asset_value"]

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
        if upper_bound:
            reach = _extend_frequent(np.asarray(rps, dtype=float), np.asarray(central))[0][0]
            if reach < rps[0]:
                entry["upper_bound_extends_to_years"] = round(float(reach), 1)
        if asset_value:
            entry.update(_plausibility(entry["ead_central"], asset_value))
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
            f"Upper bound: the loss line through the two smallest return periods is "
            f"extended to more frequent events, straight in ln T, down to where it "
            f"reaches zero (per variant: upper_bound_extends_to_years) — at most "
            f"{EXTENSION_FACTOR:g} times below the {t_min:g}-year, never under 1 year; "
            "no loss beyond that. A bound on what the frequent events could add, "
            "not a forecast."
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
            "asset_total": asset_value,
            **{
                k: setting[k]
                for k in ("replacement_value_column", "replacement_value_map", "per_feature")
                if setting.get(k)
            },
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
            "ead_pct_of_asset_per_year is the central EAD over the whole dataset's "
            "replacement value (asset_total); above about "
            f"{100 * PLAUSIBLE_SHARE:g} % a year `plausibility` is 'high' and the figure "
            "is not a planning bound.",
            "EADs of different hazards add (rare, independent events; the sum is "
            "capped at the replacement value). Event losses at given return periods "
            "do not.",
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

    points, settings, results = [], [], []
    for name in names:
        result = _analysis(conv, name)
        meta = result.get("_assistant_meta", {})
        if result.get("total_damage_cost") is None or "curve" not in meta:
            raise ValueError(
                f"{name!r} has no damage cost — run it with a curve and a "
                "replacement_value"
            )
        haz = domain.resolve_hazard(meta["hazard_id"])
        info = domain.get_dataset(meta["file_id"], conv.owner)
        pt = _point(haz)
        pt.update(_losses(result, domain.summarise(result, info, haz, meta["threshold"])))
        pt["analysis"] = name
        points.append(pt)
        settings.append(meta)
        results.append(result)

    for key, what in (
        ("file_id", "datasets"),
        ("threshold", "thresholds"),
        ("curve_path", "curves"),
        ("replacement_value", "replacement values"),
        ("replacement_value_column", "replacement value columns"),
        ("replacement_value_map", "replacement value maps"),
    ):
        seen = {_same(m.get(key)) for m in settings}
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
    info = domain.get_dataset(first["file_id"], conv.owner)
    column = first.get("replacement_value_column")
    values = (
        domain.replacement_values(
            info, column, first.get("replacement_value_map"), first["replacement_value"]
        )[0]
        if column
        else first["replacement_value"]
    )
    return points, {
        "asset_value": _asset_value(results[0], info, values),
        "file_id": first["file_id"],
        "threshold": first["threshold"],
        "curve": first["curve"],
        "replacement_value": first["replacement_value"],
        "replacement_value_column": first.get("replacement_value_column"),
        "replacement_value_map": first.get("replacement_value_map"),
        "provenance": curve.get("provenance"),
    }


def _same(value: Any) -> Any:
    """A hashable form in which 100 and 100.0, or two equal maps, are equal."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, default=str)
    return value
