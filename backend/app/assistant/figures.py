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

import threading
from pathlib import Path
from typing import Any

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


def _color(label: str, i: int) -> str:
    return SEMANTIC_COLORS.get(str(label).strip().lower(), _FALLBACK[i % len(_FALLBACK)])


def _human_ticks(v: float, _pos: int = 0) -> str:
    a = abs(v)
    for div, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "k")):
        if a >= div:
            return f"{v / div:.10g}{suffix}"
    return f"{v:.10g}"


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
    value.set_major_formatter(FuncFormatter(_human_ticks))
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

    labels = [str(x) for x in labels]
    values = [[float(v) for v in s["values"]] for s in series]

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
        names = [str(s.get("name", f"Series {i + 1}")) for i, s in enumerate(series)]

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
                        color=_color(name, i), label=name)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=8.5)
            _style(ax, title, xlabel, ylabel, "y")
            if len(names) > 1:
                _legend(ax, len(names))
        elif kind in ("bar", "stacked_bar"):
            stacked = stacked or kind == "stacked_bar"
            n = len(values)
            if horizontal:
                if stacked:
                    left = np.zeros(len(labels))
                    for i, (name, col) in enumerate(zip(names, values)):
                        ax.barh(x, col, left=left, color=_color(name, i), label=name)
                        left = left + np.array(col)
                else:
                    h = 0.8 / n
                    for i, (name, col) in enumerate(zip(names, values)):
                        ax.barh(x + (i - (n - 1) / 2) * h, col, height=h,
                                color=_color(name, i), label=name)
                ax.set_yticks(x)
                ax.set_yticklabels(labels, fontsize=8.5)
                ax.invert_yaxis()
                _style(ax, title, xlabel, ylabel, "x")
            else:
                if stacked:
                    bottom = np.zeros(len(labels))
                    for i, (name, col) in enumerate(zip(names, values)):
                        ax.bar(x, col, bottom=bottom, color=_color(name, i), label=name)
                        bottom = bottom + np.array(col)
                else:
                    w = 0.8 / n
                    for i, (name, col) in enumerate(zip(names, values)):
                        ax.bar(x + (i - (n - 1) / 2) * w, col, width=w,
                               color=_color(name, i), label=name)
                ax.set_xticks(x)
                ax.set_xticklabels(
                    labels, fontsize=8.5,
                    rotation=30 if max(len(l) for l in labels) > 10 else 0,
                    ha="right" if max(len(l) for l in labels) > 10 else "center",
                )
                _style(ax, title, xlabel, ylabel, "y")
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
    """The app's own basemap table, so a custom map matches an exported one."""
    from ..api.export import BASEMAP_TILE_URLS

    return BASEMAP_TILE_URLS.get(name, BASEMAP_TILE_URLS["osm"])


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
            ax.set_title(title, fontsize=13, loc="left", pad=10)
        if legend and handles:
            ax.legend(handles=handles, loc="lower right", fontsize=9,
                      frameon=True, framealpha=0.85)
        fig.savefig(dest, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return dest
