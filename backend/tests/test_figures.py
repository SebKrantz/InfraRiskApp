"""The report figures: the map that declines when almost nothing is exposed, the
two-panel damage chart and its `$` handling, flat charts, PNG sizing and the
hazard colourbar. No network: rasters are written to a temp dir, basemap tiles
and the map renderer are replaced.

Run from backend/ with the repo's venv:  python -m unittest discover -s tests
"""

from __future__ import annotations

import io
import re
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from matplotlib import cbook  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from PIL import Image  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402
from shapely.geometry import LineString, Point  # noqa: E402

from app.api import export  # noqa: E402
from app.api.upload import uploaded_files  # noqa: E402
from app.assistant import conversations, domain, figures, tools  # noqa: E402

tools.load()

from app.assistant.tools import output  # noqa: E402

matplotlib.use("Agg")

FILE_ID = "test-figures-line"


def png(size=(100, 80)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, "white").save(buf, "PNG")
    return buf.getvalue()


@contextmanager
def captured_figure():
    """Stop every savefig, keeping the figure, all its text and the labels that
    were actually drawn (a tick outside its axis is not)."""
    seen: dict = {}
    drawn: list = []
    real_draw = Text.draw

    def record(self, renderer):
        real_draw(self, renderer)
        if self.get_visible() and self.get_text():
            drawn.append(self)

    def grab(fig, *args, **kwargs):
        drawn.clear()
        fig.canvas.draw()
        seen["fig"] = fig
        seen["texts"] = [t.get_text() for t in fig.findobj(Text) if t.get_text()]
        seen["drawn"] = [(t.get_text(), t.get_window_extent()) for t in drawn]

    with mock.patch.object(Figure, "savefig", autospec=True, side_effect=grab), \
            mock.patch.object(Text, "draw", record):
        yield seen


def line_dataset(file_id: str, km_affected: float, km_total: float):
    """A dataset and the analysis result it would give, `km_affected` of `km_total` exposed."""
    uploaded_files[file_id] = {
        "gdf": gpd.GeoDataFrame(
            {"name": ["a"]}, geometry=[LineString([(15.0, -4.0), (15.1, -4.0)])], crs="EPSG:4326"
        ),
        "filename": "line.gpkg",
        "geometry_type": "LineString",
        "feature_count": 1,
        "bounds": [15.0, -4.0, 15.1, -4.0],
    }
    return {
        "affected_meters": km_affected * 1000.0,
        "unaffected_meters": (km_total - km_affected) * 1000.0,
        "full_gdf": gpd.GeoDataFrame(
            {"affected": [True]}, geometry=[LineString([(15.0, -4.0), (15.1, -4.0)])], crs="EPSG:4326"
        ),
        "_assistant_meta": {
            "file_id": file_id,
            "hazard_id": next(iter(domain.catalogue())),
            "threshold": 100.0,
        },
    }


class MapDeclinesWhenAlmostNothingIsExposed(unittest.TestCase):
    def setUp(self):
        self.conv = conversations.Conversation(id="test-figures", created=0.0)
        patcher = mock.patch("app.api.export.generate_map_png", return_value=png())
        self.render = patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        uploaded_files.pop(FILE_ID, None)
        conversations._cleanup(self.conv)

    def make_map(self, result, **kw):
        self.conv.namespace["analysis_1"] = result
        return output.make_map(self.conv, "analysis_1", **kw)

    def test_the_railway_case_gets_a_note_and_no_figure(self):
        # 4.567 km of 454.768 km: 1.004 %, reported as 1.0 %
        out = self.make_map(line_dataset(FILE_ID, 4.567, 454.768))
        self.render.assert_not_called()
        self.assertNotIn("artifact", out)
        self.assertTrue(out["skipped"])
        self.assertIn("4.567 km of 454.768 km", out["note"])
        self.assertIn("one sentence", out["note"])

    def test_nothing_exposed_is_declined_too(self):
        out = self.make_map(line_dataset(FILE_ID, 0.0, 454.768))
        self.render.assert_not_called()
        self.assertTrue(out["skipped"])

    def test_a_visible_share_is_drawn(self):
        out = self.make_map(line_dataset(FILE_ID, 9.59, 454.768))  # 2.1 %
        self.render.assert_called_once()
        self.assertIn("artifact", out)
        self.assertNotIn("skipped", out)

    def test_force_draws_it_anyway(self):
        out = self.make_map(line_dataset(FILE_ID, 4.567, 454.768), force=True)
        self.render.assert_called_once()
        self.assertIn("artifact", out)

    def test_points_use_the_share_of_features(self):
        info = {"geometry_type": "Point", "feature_count": 400}
        self.assertEqual(domain.affected_share_pct({"affected_count": 3}, info), 0.75)
        self.assertEqual(domain.affected_share_pct({"affected_count": 40}, info), 10.0)
        uploaded_files[FILE_ID] = {**info, "gdf": gpd.GeoDataFrame(geometry=[]), "filename": "p"}
        result = {
            "affected_count": 3,
            "full_gdf": gpd.GeoDataFrame({"affected": [True]}, geometry=[Point(15, -4)], crs="EPSG:4326"),
            "_assistant_meta": {"file_id": FILE_ID, "hazard_id": next(iter(domain.catalogue()))},
        }
        out = self.make_map(result)
        self.render.assert_not_called()
        self.assertIn("3 of 400 features", out["note"])

    def test_summarise_reports_the_share_the_rule_uses(self):
        result = line_dataset(FILE_ID, 4.567, 454.768)
        info = uploaded_files[FILE_ID]
        out = domain.summarise(result, info, domain.resolve_hazard(result["_assistant_meta"]["hazard_id"]), 100.0)
        self.assertEqual(out["affected_share_pct"], domain.affected_share_pct(result, info))
        self.assertEqual(out["affected_share_pct"], 1.0)


class MapFitsTheReportBuilder(unittest.TestCase):
    def test_a_wide_map_is_scaled_down_and_stays_a_png(self):
        out = output._fit_png(png((3600, 3000)), output.REPORT_MAX_PX)
        with Image.open(io.BytesIO(out)) as im:
            self.assertEqual(im.format, "PNG")
            self.assertEqual(im.size, (2400, 2000))

    def test_a_map_within_the_limit_is_untouched(self):
        small = png((2400, 1800))
        self.assertIs(output._fit_png(small, output.REPORT_MAX_PX), small)


class DollarSigns(unittest.TestCase):
    def test_tex_safe_escapes_once(self):
        self.assertEqual(export.tex_safe("US$ 1 - US$ 2"), r"US\$ 1 - US\$ 2")
        self.assertEqual(export.tex_safe(r"US\$ 1"), r"US\$ 1")  # not twice
        self.assertFalse(cbook.is_math_text(export.tex_safe("$15,166,246 - $45,498,737")))
        self.assertTrue(cbook.is_math_text("$15,166,246 - $45,498,737"))

    def test_damage_label_in_millions_without_math_mode(self):
        label = export._damage_label(30_332_492, 15_166_246, 45_498_737)
        self.assertFalse(cbook.is_math_text(label))
        self.assertEqual(label.replace("\\", ""), "US$ 30.3 m\n[US$ 15.2 - 45.5 m]")
        self.assertEqual(export._damage_label(1_250_000_000).replace("\\", ""), "US$ 1,250 m")
        self.assertEqual(export._damage_label(171_882).replace("\\", ""), "US$ 0.17 m")

    def test_the_damage_chart_has_two_panels_and_plain_text(self):
        gdf = gpd.GeoDataFrame(
            {"affected": [True, False], "vulnerability": [0.2, 0.0], "length_m": [100_000.0, 354_000.0]},
            geometry=[LineString([(0, 0), (1, 0)]), LineString([(1, 0), (2, 0)])],
            crs="EPSG:4326",
        )
        summary = {
            "affected_meters": 100_000.0,
            "unaffected_meters": 354_000.0,
            "total_damage_cost": 30_332_492.0,
            "total_damage_cost_lower": 15_166_246.0,
            "total_damage_cost_upper": 45_498_737.0,
        }
        with captured_figure() as seen:
            export.generate_barchart_png(summary, "LineString", "Flood damage (US$)", gdf, 100.0, "mm")
        self.assertEqual(len(seen["fig"].axes), 2)  # no twin axis
        for text in seen["texts"]:
            self.assertFalse(cbook.is_math_text(text), text)  # no $ pair left to garble
            self.assertIsNone(re.search(r"\d{7,}|\d,\d{3},\d{3}", text), text)  # no 8-digit dollars
        self.assertIn(r"US\$ 30.3 m" + "\n" + r"[US\$ 15.2 - 45.5 m]", seen["texts"])

    def test_make_chart_labels_survive_two_dollar_signs(self):
        with tempfile.TemporaryDirectory() as tmp, captured_figure() as seen:
            figures.chart(
                Path(tmp) / "c.png", "bar", ["A ($)", "B ($)"],
                [{"name": "Damage ($) and EAD ($)", "values": [1.0, 2.0]}],
                title="Losses ($ m, $ m)", ylabel="US$ million ($)",
            )
        self.assertTrue(seen["texts"])
        for text in seen["texts"]:
            self.assertFalse(cbook.is_math_text(text), text)


class ExposureChartUnits(unittest.TestCase):
    def texts(self, summary, geometry_type):
        with captured_figure() as seen:
            export.generate_barchart_png(summary, geometry_type, "t", None, 1.0, "mm")
        return seen["texts"]

    def test_a_line_is_counted_in_km_like_the_on_screen_chart(self):
        texts = self.texts({"affected_meters": 9_590.0, "unaffected_meters": 445_178.0}, "LineString")
        self.assertIn("Length (km)", texts)
        self.assertIn("9.59", texts)
        self.assertIn("445.2", texts)

    def test_points_are_counted(self):
        texts = self.texts({"affected_count": 12, "unaffected_count": 1_200}, "Point")
        self.assertIn("Count", texts)
        self.assertIn("1,200", texts)


class FlatCharts(unittest.TestCase):
    def test_flat_series(self):
        self.assertEqual(figures.flat_series([{"name": "Exposed km", "values": [4.567, 4.567, 4.567]}]),
                         ["Exposed km"])
        self.assertEqual(len(figures.flat_series([{"name": "n", "values": [100, 103, 105]}])), 1)  # within 5 %
        self.assertEqual(figures.flat_series([{"name": "n", "values": [4.6, 9.2, 13.8]}]), [])
        self.assertEqual(figures.flat_series([{"name": "n", "values": [7.0]}]), [])  # one bar: nothing to compare

    def chart(self, kind, series, labels=("25 yr", "50 yr", "100 yr")):
        conv = conversations.Conversation(id="test-flat", created=0.0)
        self.addCleanup(conversations._cleanup, conv)
        return output.make_chart(conv, kind, list(labels), series, ylabel="km")

    def test_make_chart_says_so_when_every_series_is_flat(self):
        out = self.chart("bar", [{"name": "Exposed length", "values": [4.567, 4.567, 4.567]}])
        self.assertIn("artifact", out)  # still drawn: the agent decides
        self.assertIn("Flat chart: Exposed length", out["note"])
        self.assertIn("leave this figure out", out["note"])

    def test_make_chart_is_quiet_when_the_bars_differ(self):
        out = self.chart("bar", [{"name": "Exposed length", "values": [4.0, 6.0, 9.0]}])
        self.assertNotIn("note", out)

    def test_one_flat_series_among_moving_ones_is_not_flagged(self):
        out = self.chart("bar", [{"name": "A", "values": [5.0, 5.0, 5.0]},
                                 {"name": "B", "values": [2.0, 4.0, 8.0]}])
        self.assertNotIn("note", out)

    def test_stacked_bars_and_pies_are_not_judged(self):
        flat = [{"name": "Share", "values": [5.0, 5.0, 5.0]}]
        self.assertNotIn("note", self.chart("stacked_bar", flat))
        self.assertNotIn("note", self.chart("pie", flat))


class HazardColourbar(unittest.TestCase):
    """Its ticks stay inside the data range, so the end labels cannot overprint
    (the old hand-rolled norm drew ticks past both ends: '200' over '150')."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        # 196 to 502 across the window the map reads (the line's box plus 10 %)
        self.raster = str(Path(self.tmp.name) / "pga.tif")
        lon = 14.9 + 0.001 * np.arange(300)
        data = np.tile(np.clip(196.0 + (lon - 14.99) * 306.0 / 0.12, 150.0, 600.0), (400, 1))
        with rasterio.open(
            self.raster, "w", driver="GTiff", height=400, width=300, count=1, dtype="float32",
            crs="EPSG:4326", transform=from_origin(14.9, -3.8, 0.001, 0.001),
        ) as dst:
            dst.write(data.astype("float32"), 1)
        self.gdf = gpd.GeoDataFrame(
            {"affected": [True, False], "vulnerability": [0.2, 0.0], "length_m": [5000.0, 5000.0]},
            geometry=[LineString([(15.0, -4.0), (15.05, -4.0)]), LineString([(15.05, -4.0), (15.1, -4.0)])],
            crs="EPSG:4326",
        )

    def draw(self, vulnerability: bool):
        with captured_figure() as seen, mock.patch("contextily.add_basemap"):
            export.generate_map_png(
                self.gdf, self.raster, "LineString", "t", is_vulnerability_mode=vulnerability,
                hazard_unit="cm/s²",
            )
        fig = seen["fig"]
        bars = [a.get_window_extent() for a in fig.axes
                if a.get_ylabel().startswith(("Hazard Intensity", "Damage Ratio"))]
        right = min(b.x1 for b in bars)
        labels = [(t, box) for t, box in seen["drawn"]
                  if box.x0 >= right - 1 and re.fullmatch(r"[\d.,]+", t)]
        return bars, labels

    def test_ticks_stay_inside_the_bar_and_do_not_overlap(self):
        for vulnerability in (False, True):
            with self.subTest(vulnerability=vulnerability):
                bars, labels = self.draw(vulnerability)
                self.assertGreaterEqual(len(labels), 5)
                for text, box in labels:
                    centre = (box.y0 + box.y1) / 2
                    self.assertTrue(
                        any(b.y0 - box.height <= centre <= b.y1 + box.height for b in bars),
                        f"tick label {text} is drawn past the end of its colourbar",
                    )
                for i, (ta, a) in enumerate(labels):
                    for tb, b in labels[i + 1:]:
                        self.assertFalse(a.overlaps(b), f"colourbar labels {ta} and {tb} overprint")


if __name__ == "__main__":
    unittest.main()
