"""The Risk platform's report figures: plain text in titles and legends, money axes that
never carry a 'k' on a million, class layers that do not flood the map, small map files,
capitalised captions, the warm-is-higher colour rule.

No network: rasters are written to a temp dir, basemap tiles are replaced.

Run from backend/ with the repo's venv:  python -m unittest discover -s tests
"""

from __future__ import annotations

import io
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from PIL import Image  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402
from shapely.geometry import LineString  # noqa: E402

from app.api import export  # noqa: E402
from app.api.upload import uploaded_files  # noqa: E402
from app.assistant import artifacts, conversations, curve_library, figures, reports, tools  # noqa: E402
from test_figures import captured_figure, line_dataset, png  # noqa: E402

tools.load()

from app.assistant.tools import curves, output  # noqa: E402

FILE_ID = "test-report-graphics"


class PlainTextInFigures(unittest.TestCase):
    def test_curve_ids_are_refused_in_any_spelling(self):
        for text in (
            "Earthquake damage (XYZ-RAIL-PGA proxy)",
            "xyz_rail_pga",
            "Vulnerability: Xyz Rail Pga",
            f"Curve {curve_library.index()[0]['curve_id']}",
        ):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "internal identifier"):
                figures.require_plain([text], ["xyz_rail_pga"])

    def test_snake_case_names_are_refused(self):
        with self.assertRaises(ValueError):
            figures.require_plain(["Damage by affected_segments"])

    def test_plain_words_pass(self):
        figures.require_plain(
            [
                None,
                "Railway earthquake vulnerability (proxy curve)",
                "250-year (0.40%/yr)",
                "Peak ground acceleration (cm/s²)",
                "Railway",  # a curve a user simply named "railway" is not an id
            ],
            ["railway", "xyz_rail_pga"],
        )

    def test_make_chart_refuses_an_id_in_a_series_name(self):
        conv = conversations.Conversation(id="test-plain-chart", created=0.0)
        self.addCleanup(conversations._cleanup, conv)
        with self.assertRaisesRegex(ValueError, "internal identifier"):
            output.make_chart(conv, "bar", ["A", "B"],
                              [{"name": "my_curve_central", "values": [1.0, 2.0]}], ylabel="km")

    def test_a_custom_map_legend_is_not_a_variable_name(self):
        conv = conversations.Conversation(id="test-plain-map", created=0.0)
        self.addCleanup(conversations._cleanup, conv)
        conv.namespace["affected_segments"] = gpd.GeoDataFrame(
            geometry=[LineString([(0, 0), (1, 1)])], crs="EPSG:4326")
        with mock.patch.object(figures, "custom_map") as draw:
            output.make_custom_map(conv, [{"source": "affected_segments"}])
        self.assertEqual(draw.call_args.args[1][0]["label"], "Affected segments")


class CurveLegends(unittest.TestCase):
    def setUp(self):
        self.conv = conversations.Conversation(id="test-curve-legend", created=0.0)
        self.addCleanup(conversations._cleanup, self.conv)

    def legend(self, names, **kw):
        with mock.patch.object(figures, "curve_plot") as draw:
            curves.plot_curve(self.conv, names, **kw)
        return [c["name"] for c in draw.call_args.args[1]]

    def test_the_default_curve_is_named_by_its_asset_not_its_id(self):
        curves.use_library_curve(self.conv, "AEI-RAIL-PGA", name="aei_rail_pga")
        (label,) = self.legend(["aei_rail_pga"])
        self.assertEqual(label, "Railways - track and formation (default curve)")

    def test_a_published_curve_is_named_by_asset_and_place(self):
        curves.use_library_curve(self.conv, "F8.1")
        (label,) = self.legend(["F8.1"])
        self.assertEqual(label, "Railways, Austria")

    def test_a_constructed_curve_says_so_and_uses_its_asset(self):
        curves.create_curve(
            self.conv, "ls_curve", [0, 1, 2], [0, 0.1, 0.2],
            basis="Test curve, judgement throughout, for the legend only.", asset="railway track")
        (label,) = self.legend(["ls_curve"])
        self.assertEqual(label, "Railway track (constructed curve)")

    def test_curves_that_read_alike_are_told_apart(self):
        curves.use_library_curve(self.conv, "E1.1")
        curves.use_library_curve(self.conv, "E1.2")
        a, b = self.legend(["E1.1", "E1.2"])
        self.assertEqual((a, b), ("Small power plants, USA: Anchored/seismic components",
                                 "Small power plants, USA: Unanchored/standard components"))

    def test_a_title_with_a_curve_id_is_refused(self):
        curves.use_library_curve(self.conv, "AEI-RAIL-PGA", name="aei_rail_pga")
        with self.assertRaisesRegex(ValueError, "internal identifier"):
            curves.plot_curve(self.conv, ["aei_rail_pga"], title="Vulnerability: AEI-RAIL-PGA proxy")


class MoneyAxes(unittest.TestCase):
    def texts(self, kind="bar", values=(500.0, 1500.0, 3000.0), ylabel="Event damage cost (US$ million)",
              series=None, **kw):
        series = series or [{"name": "Cost", "values": list(values)}]
        with tempfile.TemporaryDirectory() as tmp, captured_figure() as seen:
            figures.chart(Path(tmp) / "c.png", kind, ["a", "b", "c"], series, ylabel=ylabel, **kw)
        return [t.replace("\\", "") for t in seen["texts"]]

    def test_a_million_axis_that_reaches_a_thousand_is_redrawn_in_billions(self):
        texts = self.texts()
        self.assertIn("Event damage cost (US$ billion)", texts)
        self.assertNotIn("Event damage cost (US$ million)", texts)
        self.assertFalse([t for t in texts if re.fullmatch(r"[\d.]+k", t)], texts)

    def test_the_bars_keep_their_values_in_the_new_unit(self):
        with tempfile.TemporaryDirectory() as tmp, captured_figure() as seen:
            figures.chart(Path(tmp) / "c.png", "bar", ["a", "b"],
                          [{"name": "Cost", "values": [500.0, 2500.0]}], ylabel="US$ million")
        heights = [p.get_height() for p in seen["fig"].axes[0].patches]
        self.assertEqual(heights, [0.5, 2.5])

    def test_stacked_totals_decide(self):
        series = [{"name": "A", "values": [600.0, 100.0, 100.0]}, {"name": "B", "values": [600.0, 100.0, 100.0]}]
        self.assertIn("US$ billion", " ".join(self.texts("stacked_bar", series=series, ylabel="US$ million")))

    def test_horizontal_bars_rescale_their_value_axis(self):
        texts = self.texts(horizontal=True, ylabel=None, xlabel="Damage (US$ million)")
        self.assertIn("Damage (US$ billion)", texts)

    def test_a_small_million_axis_is_left_alone_and_never_shows_k(self):
        texts = self.texts(values=(120.0, 480.0, 950.0))  # the top tick lands on 1,000
        self.assertIn("Event damage cost (US$ million)", texts)
        self.assertFalse([t for t in texts if re.fullmatch(r"[\d.]+k", t)], texts)

    def test_other_axes_keep_their_suffixes(self):
        texts = self.texts(values=(12_000.0, 30_000.0, 41_000.0), ylabel="Exposed people")
        self.assertTrue([t for t in texts if re.fullmatch(r"[\d.]+k", t)], texts)


class ColourByRank(unittest.TestCase):
    BLUE, AMBER, VERMILLION = "#0072b2", "#e69f00", "#d55e00"

    def test_lower_central_upper_run_cool_to_warm(self):
        self.assertEqual(
            figures._series_colors(["Lower curve bound", "Central curve", "Upper curve bound"]),
            [self.BLUE, self.AMBER, self.VERMILLION],
        )

    def test_two_estimates_keep_blue_and_vermillion(self):
        self.assertEqual(
            figures._series_colors(["Central / Lower-bound EAD", "Upper-bound EAD (freq. sensitivity)"]),
            [self.BLUE, self.VERMILLION],
        )

    def test_other_series_are_untouched(self):
        self.assertEqual(figures._series_colors(["Affected", "Unaffected"]), ["#d62728", "#2ca02c"])
        self.assertEqual(figures._series_colors(["Central Asia", "Other"]), ["#0072b2", "#d55e00"])


class CaptionsStartWithACapital(unittest.TestCase):
    def test_caption_and_title(self):
        self.assertEqual(reports._caption("precipitation-triggered landslide susceptibility"),
                         "Precipitation-triggered landslide susceptibility")
        self.assertEqual(reports._caption("Figure 3. flood depth"), "Flood depth")
        self.assertEqual(figures.capitalise("475-year shaking"), "475-year shaking")
        self.assertIsNone(figures.capitalise(None))


class ReportMaps(unittest.TestCase):
    def setUp(self):
        self.conv = conversations.Conversation(id="test-report-map", created=0.0)
        self.addCleanup(conversations._cleanup, self.conv)
        self.addCleanup(uploaded_files.pop, FILE_ID, None)

    def test_make_map_files_a_small_jpeg_and_serves_it_as_one(self):
        self.conv.namespace["a1"] = line_dataset(FILE_ID, 40.0, 100.0)
        with mock.patch("app.api.export.generate_map_png", return_value=png((3600, 3000))) as render:
            out = output.make_map(self.conv, "a1", filename="shaking_map.png")
        art = artifacts.get(out["artifact"]["id"])
        self.assertEqual(art.filename, "shaking_map.jpg")
        self.assertEqual(art.mime, "image/jpeg")
        self.assertEqual(art.kind, "map")
        with Image.open(art.path) as im:
            self.assertEqual((im.format, im.size), ("JPEG", (1600, 1333)))
        self.assertEqual(render.call_args.args[-1], output.CLASS_FOCUS_KM)

    def test_make_map_capitalises_and_checks_its_title(self):
        self.conv.namespace["a1"] = line_dataset(FILE_ID, 40.0, 100.0)
        with mock.patch("app.api.export.generate_map_png", return_value=png()) as render:
            out = output.make_map(self.conv, "a1", title="475-year shaking and damage")
            self.assertEqual(out["artifact"]["title"], "475-year shaking and damage")
            out = output.make_map(self.conv, "a1", title="shaking and damage")
            self.assertEqual(out["artifact"]["title"], "Shaking and damage")
            with self.assertRaisesRegex(ValueError, "internal identifier"):
                output.make_map(self.conv, "a1", title="Damage with the pga_curve")
        self.assertEqual(render.call_count, 2)

    def test_png_charts_keep_their_mime(self):
        art = artifacts.Artifact("i", Path("c.png"), "c.png", "chart", "t", "x", 0.0)
        self.assertEqual(art.mime, "image/png")


class ClassRasterMap(unittest.TestCase):
    """A 1-5 class layer: only the upper classes, one flat colour each, only near the asset."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        # 0.001 degree (about 110 m) cells over lon 14.9-15.4, lat -3.9 to -4.1; class 1 except
        # bands of rows: 5 far above the line, 3 just above it, 2 just below it, 4 far below it
        data = np.ones((200, 500), dtype="uint8")
        data[70:80], data[90:100], data[101:110], data[120:130] = 5, 3, 2, 4
        self.raster = str(Path(self.tmp.name) / "classes.tif")
        with rasterio.open(
            self.raster, "w", driver="GTiff", height=200, width=500, count=1, dtype="uint8",
            crs="EPSG:4326", transform=from_origin(14.9, -3.9, 0.001, 0.001),
        ) as dst:
            dst.write(data, 1)
        self.gdf = gpd.GeoDataFrame(
            {"affected": [True], "vulnerability": [0.02], "length_m": [33_000.0]},
            geometry=[LineString([(15.0, -4.0), (15.3, -4.0)])], crs="EPSG:4326",
        )

    def draw(self, **kw):
        with captured_figure() as seen, mock.patch("contextily.add_basemap"):
            export.generate_map_png(
                self.gdf, self.raster, "LineString", "t", is_vulnerability_mode=True,
                intensity_threshold=2.0, hazard_unit="(class 1-5)", **kw)
        return seen

    def wash(self, seen):
        (image,) = [im for ax in seen["fig"].axes for im in ax.images]
        rgba = np.asarray(image.get_array())
        return rgba[rgba[:, :, 3] > 0]

    def test_only_classes_from_the_middle_up_within_the_corridor_are_drawn(self):
        seen = self.draw(focus_km=2.0)
        drawn = self.wash(seen)
        self.assertGreater(len(drawn), 100)  # the class 3 band beside the line
        self.assertEqual(len({tuple(px[:3]) for px in drawn}), 1)  # not the far 4 and 5, not the 2
        bar = next(a for a in seen["fig"].axes if a.get_ylabel().startswith("Hazard Intensity"))
        self.assertEqual([t.get_text() for t in bar.get_yticklabels()], ["3", "4", "5"])
        self.assertTrue(any("within 2 km" in t for t in seen["texts"]))

    def test_a_wide_corridor_draws_the_far_classes_in_darker_colours(self):
        drawn = self.wash(self.draw(focus_km=10.0))
        colours = {tuple(px[:3]) for px in drawn}
        self.assertEqual(len(colours), 3)  # classes 3, 4, 5, one flat colour each
        self.assertLess(min(sum(map(int, c)) for c in colours), max(sum(map(int, c)) for c in colours))

    def test_a_threshold_above_the_middle_raises_the_first_class(self):
        with captured_figure() as seen, mock.patch("contextily.add_basemap"):
            export.generate_map_png(
                self.gdf, self.raster, "LineString", "t", is_vulnerability_mode=True,
                intensity_threshold=4.0, hazard_unit="(class 1-5)", focus_km=10.0)
        bar = next(a for a in seen["fig"].axes if a.get_ylabel().startswith("Hazard Intensity"))
        self.assertEqual([t.get_text() for t in bar.get_yticklabels()], ["4", "5"])

    def test_without_focus_the_layer_is_drawn_as_before(self):
        seen = self.draw()
        drawn = self.wash(seen)
        self.assertEqual(len({tuple(px[:3]) for px in drawn}), 1)  # one blue ...
        self.assertGreater(len({int(px[3]) for px in drawn}), 1)  # ... graded by alpha
        bar = next(a for a in seen["fig"].axes if a.get_ylabel().startswith("Hazard Intensity"))
        self.assertNotEqual([t.get_text() for t in bar.get_yticklabels()], ["3", "4", "5"])

    def test_exposure_mode_gets_the_same_wash_and_bar(self):
        with captured_figure() as seen, mock.patch("contextily.add_basemap"):
            export.generate_map_png(
                self.gdf, self.raster, "LineString", "t", intensity_threshold=2.0,
                hazard_unit="(class 1-5)", focus_km=2.0)
        self.assertEqual(len({tuple(px[:3]) for px in self.wash(seen)}), 1)
        bar = next(a for a in seen["fig"].axes if a.get_ylabel().startswith("Hazard Intensity"))
        self.assertEqual([t.get_text() for t in bar.get_yticklabels()], ["3", "4", "5"])

    def test_a_threshold_above_every_class_draws_no_wash_and_no_bar(self):
        with captured_figure() as seen, mock.patch("contextily.add_basemap"):
            export.generate_map_png(
                self.gdf, self.raster, "LineString", "t", is_vulnerability_mode=True,
                intensity_threshold=9.0, hazard_unit="(class 1-5)", focus_km=2.0)
        self.assertFalse(any(ax.get_ylabel().startswith("Hazard Intensity") for ax in seen["fig"].axes))
        self.assertTrue(any(ax.get_ylabel().startswith("Damage Ratio") for ax in seen["fig"].axes))

    def test_a_continuous_layer_ignores_focus(self):
        raster = str(Path(self.tmp.name) / "continuous.tif")
        data = np.tile(np.linspace(0.0, 500.0, 500, dtype="float32"), (200, 1))
        with rasterio.open(
            raster, "w", driver="GTiff", height=200, width=500, count=1, dtype="float32",
            crs="EPSG:4326", transform=from_origin(14.9, -3.9, 0.001, 0.001),
        ) as dst:
            dst.write(data, 1)
        with captured_figure() as seen, mock.patch("contextily.add_basemap"):
            export.generate_map_png(self.gdf, raster, "LineString", "t", focus_km=2.0, hazard_unit="cm/s²")
        (image,) = [im for ax in seen["fig"].axes for im in ax.images]
        alpha = np.asarray(image.get_array())[:, :, 3]
        self.assertGreater((alpha > 0).mean(), 0.9)  # the whole window, not a corridor
        self.assertFalse(any("within" in t for t in seen["texts"]))


if __name__ == "__main__":
    unittest.main()
