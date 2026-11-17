"""Declarative matplotlib figures for anything the app's own exports don't cover.

The app already renders two exhibits — the exposure/vulnerability barchart and
the hazard map — through `app.api.export.generate_barchart_png` /
`generate_map_png`, and the assistant calls those directly so its standard
figures are pixel-identical to the sidebar downloads. THIS module is the
general-purpose complement: comparisons across hazard layers, across datasets
or across thresholds, and bespoke cartography.

House style is the pretty_plot look used elsewhere in this project: a real
black panel frame, long outward ticks, thin light gridlines, ~7 formatted
breaks on the value axis and a plain left-aligned title.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

_LOCK = threading.Lock()  # pyplot state is process-global

# The app's own semantics, so a custom chart reads like the built-in one.
SEMANTIC_COLORS = {
    "affected": "#d62728",
    "exposed": "#d62728",
    "unaffected": "#2ca02c",
    "not exposed": "#2ca02c",
    "damage": "#f59e0b",
    "damage cost": "#f59e0b",
    "exposure": "#3b82f6",
    "damage ratio": "#10b981",
    "vulnerability": "#10b981",
}

# Okabe-Ito, colour-blind safe, for arbitrary categories.
_FALLBACK = [
    "#0072b2", "#d55e00", "#009e73", "#cc79a7",
    "#e69f00", "#56b4e9", "#f0e442", "#666666",
]

NEGLIGIBLE_SHARE = 0.005  # categories below this share of the peak are invisible
FLAT_SPREAD = 0.05  # bars within this share of the tallest say nothing a sentence does not


# Estimates of one quantity named for their bound: the higher the damage or cost, the
# warmer the colour (lower blue, central amber, upper vermillion).
_BOUNDS = (
    ("upper", re.compile(r"\bupper\b", re.I), "#d55e00"),
    ("lower", re.compile(r"\blower\b", re.I), "#0072b2"),
    ("central", re.compile(r"\bcentral\b", re.I), "#e69f00"),
)


def _color(label: str, i: int) -> str:
    return SEMANTIC_COLORS.get(str(label).strip().lower(), _FALLBACK[i % len(_FALLBACK)])


def _series_colors(names: list[str]) -> list[str]:
    """One colour per series; a chart that has a lower or an upper estimate colours the
    lower / central / upper ones by their rank."""
    bound = [next((c for k, p, c in _BOUNDS if p.search(n)), None) for n in names]
    if not any(c in ("#d55e00", "#0072b2") for c in bound):
        bound = [None] * len(names)
    return [c or _color(n, i) for i, (n, c) in enumerate(zip(names, bound))]


def capitalise(text: str | None) -> str | None:
    """A title or caption starts with a capital."""
    return text[:1].upper() + text[1:] if text else text


_SNAKE_CASE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+(?![A-Za-z0-9])")


def _id_patterns(curve_names: Iterable[str]) -> list[tuple[str, re.Pattern]]:
    """The curve ids to look for in figure text: the library's and the conversation's
    own, those that look like an id (a separator or a digit in them), in any case and
    with any of space, _, - or . between their parts."""
    from . import curve_library

    ids = {str(n) for n in curve_names}
    if curve_library.available():
        ids |= {row["curve_id"] for row in curve_library.index()}
    out = []
    for i in sorted(ids):
        parts = [p for p in re.split(r"[^A-Za-z0-9]+", i) if p]
        if parts and (len(parts) > 1 or any(c.isdigit() for c in i)):
            joined = r"[\s_.\-]+".join(map(re.escape, parts))
            out.append((i, re.compile(rf"(?<![A-Za-z0-9]){joined}(?![A-Za-z0-9])", re.I)))
    return out


def require_plain(texts: Iterable[str | None], curve_names: Iterable[str] = ()) -> None:
    """Raise when text that goes into a figure (title, axis label, legend entry, category)
    carries an internal identifier: a curve id ('aei_rail_pga', 'F7.1') or any snake_case
    name. A reader sees plain words; the id says nothing to them."""
    patterns = _id_patterns(curve_names)
    for text in texts:
        if not text:
            continue
        text = str(text)
        found = _SNAKE_CASE.findall(text) + [i for i, p in patterns if p.search(text)]
        if found:
            raise ValueError(
                f"{text!r} has an internal identifier ({found[0]!r}) in it. A figure uses plain "
                "words for what it shows (the asset, the hazard, the estimate), never a curve id, "
                "variable or file name: say what it is and draw it again."
            )


def flat_series(series: list[dict[str, Any]]) -> list[str]:
    """Names of the series whose values, across the categories, all lie within
    FLAT_SPREAD of the largest: equal bars, which a sentence or a table row says
    better than a chart."""
    names = []
    for i, s in enumerate(series):
        v = [float(x) for x in s["values"]]
        if len(v) >= 2 and max(v) - min(v) <= FLAT_SPREAD * max(abs(x) for x in v):
            names.append(str(s.get("name", f"Series {i + 1}")))
    return names


def _human_ticks(v: float, _pos: int = 0) -> str:
    a = abs(v)
    for div, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "k")):
        if a >= div:
            return f"{v / div:.10g}{suffix}"
    return f"{v:.10g}"


_SCALED_UNIT = re.compile(r"\b(thousand|million|billion|trillion)\b", re.I)
_US_MILLION = re.compile(r"US\$\s*million\b", re.I)


def _plain_ticks(v: float, _pos: int = 0) -> str:
    return f"{v:,.10g}"


def _legend(ax: Any, n_items: int) -> None:
    """A single frameless row above the panel for a few items, a right-hand
    column for many."""
    if n_items <= 4:
        ax.legend(
            fontsize=8.5, frameon=False, loc="lower left",
            bbox_to_anchor=(0.0, 1.0), ncols=n_items, borderaxespad=0.2,
        )
        title = ax.get_title(loc="left")
        if title:
            ax.set_title(title, fontsize=12, loc="left", pad=30)
    else:
        ax.legend(fontsize=8.5, frameon=False, loc="center left",
                  bbox_to_anchor=(1.02, 0.5), ncols=1)


def _style(
    ax: Any,
    title: str | None,
    xlabel: str | None,
    ylabel: str | None,
    value_axis: str = "y",
    plain_ticks: bool = False,
) -> None:
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(0.6)
    ax.grid(True, color="#ebebeb", linewidth=0.4)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=5, width=0.5, color="black", labelsize=8.5)
    value = ax.yaxis if value_axis == "y" else ax.xaxis
    value.set_major_locator(MaxNLocator(nbins=7))
    # an axis whose unit already says million or billion never gets a "k" or "M" on top of it
    value.set_major_formatter(FuncFormatter(_plain_ticks if plain_ticks else _human_ticks))
    if title:
        ax.set_title(title, fontsize=12, loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9.5)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9.5)


def chart(
    dest: Path,
    kind: str,
    labels: list[str],
    series: list[dict[str, Any]],
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    horizontal: bool = False,
    sort: bool = False,
    stacked: bool = False,
) -> Path:
    """Render one chart. `series` = [{name, values: [...]}]; for pie, the first
    series' values slice by `labels`."""
    import numpy as np
    import matplotlib.pyplot as plt

    if not series or not labels:
        raise ValueError("labels and series must be non-empty")
    for s in series:
        if len(s["values"]) != len(labels):
            raise ValueError(
                f"series {s.get('name')!r} has {len(s['values'])} values but "
                f"there are {len(labels)} labels"
            )

    from ..api.export import tex_safe  # two `$` in one label would switch on mathtext

    values = [[float(v) for v in s["values"]] for s in series]
    # Money in US$ million that reaches a thousand is drawn in US$ billion: a "1k" or
    # "2.5k" tick on a million axis reads as billions.
    value_on_x = horizontal and kind in ("bar", "stacked_bar")
    unit = xlabel if value_on_x else ylabel
    plain_ticks = bool(unit and kind != "pie" and _SCALED_UNIT.search(unit))
    if plain_ticks and _US_MILLION.search(unit):
        stack = stacked or kind == "stacked_bar"
        top = max(
            (sum(col[i] for col in values) if stack else max(abs(col[i]) for col in values))
            for i in range(len(labels))
        )
        if top >= 1000:
            values = [[v / 1000.0 for v in col] for col in values]
            billion = _US_MILLION.sub("US$ billion", unit)
            xlabel, ylabel = (billion, ylabel) if value_on_x else (xlabel, billion)

    labels = [tex_safe(x) for x in labels]
    title = capitalise(title)
    title, xlabel, ylabel = (tex_safe(t) if t else t for t in (title, xlabel, ylabel))

    # Drop categories that would render as invisible slivers.
    peaks = [max(abs(col[i]) for col in values) for i in range(len(labels))]
    top = max(peaks) if peaks else 0.0
    if top > 0:
        keep = [i for i, pk in enumerate(peaks) if pk >= NEGLIGIBLE_SHARE * top]
        if 0 < len(keep) < len(labels):
            labels = [labels[i] for i in keep]
            values = [[col[i] for i in keep] for col in values]

    if sort and kind != "line":
        order = sorted(range(len(labels)), key=lambda i: -max(abs(c[i]) for c in values))
        labels = [labels[i] for i in order]
        values = [[col[i] for i in order] for col in values]

    with _LOCK:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        x = np.arange(len(labels))
        names = [tex_safe(s.get("name", f"Series {i + 1}")) for i, s in enumerate(series)]
        colors = _series_colors(names)

        if kind == "pie":
            colors = [_color(l, i) for i, l in enumerate(labels)]
            ax.pie(
                [abs(v) for v in values[0]], labels=labels, colors=colors,
                autopct="%1.1f%%", startangle=90,
                textprops={"fontsize": 9},
            )
            ax.axis("equal")
            if title:
                ax.set_title(title, fontsize=12, loc="left")
        elif kind == "line":
            for i, (name, col) in enumerate(zip(names, values)):
                ax.plot(x, col, marker="o", markersize=4, linewidth=1.6,
                        color=colors[i], label=name)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=8.5)
            _style(ax, title, xlabel, ylabel, "y", plain_ticks)
            if len(names) > 1:
                _legend(ax, len(names))
        elif kind in ("bar", "stacked_bar"):
            stacked = stacked or kind == "stacked_bar"
            n = len(values)
            if horizontal:
                if stacked:
                    left = np.zeros(len(labels))
                    for i, (name, col) in enumerate(zip(names, values)):
                        ax.barh(x, col, left=left, color=colors[i], label=name)
                        left = left + np.array(col)
                else:
                    h = 0.8 / n
                    for i, (name, col) in enumerate(zip(names, values)):
                        ax.barh(x + (i - (n - 1) / 2) * h, col, height=h,
                                color=colors[i], label=name)
                ax.set_yticks(x)
                ax.set_yticklabels(labels, fontsize=8.5)
                ax.invert_yaxis()
                _style(ax, title, xlabel, ylabel, "x", plain_ticks)
            else:
                if stacked:
                    bottom = np.zeros(len(labels))
                    for i, (name, col) in enumerate(zip(names, values)):
                        ax.bar(x, col, bottom=bottom, color=colors[i], label=name)
                        bottom = bottom + np.array(col)
                else:
                    w = 0.8 / n
                    for i, (name, col) in enumerate(zip(names, values)):
                        ax.bar(x + (i - (n - 1) / 2) * w, col, width=w,
                               color=colors[i], label=name)
                ax.set_xticks(x)
                ax.set_xticklabels(
                    labels, fontsize=8.5,
                    rotation=30 if max(len(l) for l in labels) > 10 else 0,
                    ha="right" if max(len(l) for l in labels) > 10 else "center",
                )
                _style(ax, title, xlabel, ylabel, "y", plain_ticks)
            if len(names) > 1:
                _legend(ax, len(names))
        else:
            plt.close(fig)
            raise ValueError(
                f"unknown chart kind {kind!r}; use bar, stacked_bar, line or pie"
            )

        fig.savefig(dest, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return dest


# --------------------------------------------------------------------------- #
# Custom cartography
# --------------------------------------------------------------------------- #

def _basemap_source(name: str) -> str:
    """The app's own basemap table, so a custom map matches an exported one.

    Goes through _basemap_tile_url() so CARTO tiles carry CARTO_API_KEY, as the
    app's own exports do; unknown names still fall back to OSM.
    """
    from ..api.export import BASEMAP_TILE_URLS, _basemap_tile_url

    return _basemap_tile_url(name if name in BASEMAP_TILE_URLS else "osm")


def custom_map(
    dest: Path,
    layers: list[dict[str, Any]],
    title: str | None = None,
    basemap: str = "osm",
    legend: bool = True,
    bbox: list[float] | None = None,
) -> Path:
    """Draw arbitrary GeoDataFrames over a web basemap.

    layers = [{gdf, label, color, column?, cmap?, size?, linewidth?}]. All
    geometry is reprojected to Web Mercator so the tiles line up.
    """
    import contextily as cx
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if not layers:
        raise ValueError("pass at least one layer")

    with _LOCK:
        fig, ax = plt.subplots(figsize=(11, 9))
        handles: list[Any] = []
        for i, layer in enumerate(layers):
            gdf = layer["gdf"]
            if gdf is None or len(gdf) == 0:
                continue
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
            merc = gdf.to_crs("EPSG:3857")
            label = str(layer.get("label") or f"Layer {i + 1}")
            color = layer.get("color") or _color(label, i)
            column = layer.get("column")
            is_point = merc.geom_type.iloc[0] in ("Point", "MultiPoint")
            kwargs: dict[str, Any] = {"ax": ax, "zorder": 2 + i}
            if is_point:
                kwargs["markersize"] = layer.get("size", 18)
            else:
                kwargs["linewidth"] = layer.get("linewidth", 1.6)
            if column:
                merc.plot(column=column, cmap=layer.get("cmap", "viridis"),
                          legend=True, **kwargs)
            else:
                merc.plot(color=color, **kwargs)
                handles.append(
                    Line2D(
                        [], [], color=color, label=label,
                        marker="o" if is_point else None,
                        linestyle="none" if is_point else "-",
                        markersize=6, linewidth=2,
                    )
                )

        if bbox:
            from pyproj import Transformer

            tf = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
            x0, y0 = tf.transform(bbox[0], bbox[1])
            x1, y1 = tf.transform(bbox[2], bbox[3])
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)

        try:
            cx.add_basemap(
                ax, crs="EPSG:3857", source=_basemap_source(basemap),
                zoom="auto", zorder=0,
            )
        except Exception:  # noqa: BLE001 — a map without tiles still beats no map
            pass

        ax.set_axis_off()
        if title:
            ax.set_title(capitalise(title), fontsize=13, loc="left", pad=10)
        if legend and handles:
            ax.legend(handles=handles, loc="lower right", fontsize=9,
                      frameon=True, framealpha=0.85)
        fig.savefig(dest, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return dest


def curve_plot(
    dest: Path,
    curves: list[dict[str, Any]],
    title: str | None = None,
    xlabel: str | None = None,
) -> Path:
    """Vulnerability curves on a numeric intensity axis, bounds as a band.

    Each entry is {name, intensity, central, lower?, upper?}. This is the one
    exhibit `chart()` cannot draw: its x axis is categorical, and a damage curve
    needs a real intensity scale and a shaded uncertainty ribbon.
    """
    import matplotlib.pyplot as plt

    from ..api.export import tex_safe

    title, xlabel = (tex_safe(t) if t else t for t in (capitalise(title), xlabel))
    with _LOCK:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        try:
            for i, curve in enumerate(curves):
                color = _FALLBACK[i % len(_FALLBACK)]
                x = curve["intensity"]
                ax.plot(x, curve["central"], color=color, linewidth=1.8,
                        label=tex_safe(curve["name"]))
                if curve.get("lower") is not None and curve.get("upper") is not None:
                    ax.fill_between(
                        x, curve["lower"], curve["upper"], color=color, alpha=0.15, linewidth=0
                    )
            # Headroom above the tallest curve, but never past 1: a proportion
            # destroyed has no meaning above total loss. A shallow curve keeps a
            # tight axis so its shape stays readable.
            peak = max(max(c.get("upper", c["central"])) for c in curves)
            ax.set_ylim(0, min(1.0, max(peak * 1.15, 1e-3)))
            ax.margins(x=0)
            _style(ax, title, xlabel, "Proportion destroyed")
            if len(curves) > 1 or curves[0]["name"]:
                _legend(ax, len(curves))
            fig.tight_layout()
            fig.savefig(dest, dpi=200, bbox_inches="tight", facecolor="white")
        finally:
            plt.close(fig)
    return dest
