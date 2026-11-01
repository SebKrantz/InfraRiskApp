"""No data in an analysis: how much there was, and what it means for the layer.

No data counts as not exposed, so a layer that is empty where the hazard is low
(the PGA layers hold nothing below 49 cm/s²) gives "0 km exposed" that is a known
zero, while one that is empty because it was never modelled gives one that is
unknown. The result has to say which, and how much. No network: rasters are
written to a temp dir.

Run from backend/ with the repo's venv:  python -m unittest discover -s tests
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402
from shapely.geometry import LineString, Point  # noqa: E402

import main  # noqa: E402
from app.api import analyze, hazards, upload  # noqa: E402
from app.api.upload import uploaded_files  # noqa: E402
from app.assistant import conversations, domain, guides, mcp_server, tools  # noqa: E402
from app.utils.geospatial import analyze_intersection  # noqa: E402

tools.load()

from app.assistant.tools import query  # noqa: E402

# The PGA COGs' own declared nodata (float32 max / 2).
NODATA = 1.701410009187828e38

PGA = {
    "hazard_id": "pga_test",
    "hazard": "Peak Ground Acceleration PGA - 475 Years",
    "unit": "cm/s²",
    "nodata_below": "49.0",
}
UNDOCUMENTED = {"hazard_id": "wind_test", "hazard": "Wind", "unit": "km/h", "nodata_below": None}


def write_raster(path: Path) -> str:
    """lon 0-10, lat 0-10 at 0.1°: no data west of lon 4, 120 east of it."""
    data = np.full((100, 100), 120.0, dtype=np.float32)
    data[:, :40] = NODATA
    with rasterio.open(
        path, "w", driver="GTiff", height=100, width=100, count=1, dtype="float32",
        crs="EPSG:4326", transform=from_origin(0.0, 10.0, 0.1, 0.1), nodata=NODATA,
    ) as dst:
        dst.write(data, 1)
    return str(path)


class Counts(unittest.TestCase):
    """analyze_intersection counts what it could not read."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tif = write_raster(Path(cls._tmp.name) / "pga.tif")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_a_line_reports_its_no_data_samples_and_length(self):
        gdf = gpd.GeoDataFrame(geometry=[LineString([(1, 5), (9, 5)])], crs="EPSG:4326")
        r = analyze_intersection(gdf, self.tif, "LineString", intensity_threshold=98)
        self.assertGreater(r["samples"], 800)
        # lon 1-4 of the 1-9 span is empty: 3/8 of it
        self.assertAlmostEqual(r["nodata_samples"] / r["samples"], 3 / 8, delta=0.01)
        total = r["affected_meters"] + r["unaffected_meters"]
        self.assertAlmostEqual(r["nodata_meters"] / total, 3 / 8, delta=0.01)
        # all the unaffected length is no data here, none of the affected
        self.assertAlmostEqual(r["nodata_meters"], r["unaffected_meters"], delta=1.0)

    def test_points_count_what_is_empty_or_off_the_raster(self):
        gdf = gpd.GeoDataFrame(
            geometry=[Point(2, 5), Point(6, 5), Point(7, 5), Point(11, 5)], crs="EPSG:4326"
        )
        r = analyze_intersection(gdf, self.tif, "Point", intensity_threshold=98)
        self.assertEqual((r["affected_count"], r["samples"], r["nodata_samples"]), (2, 4, 2))

    def test_a_layer_with_no_gaps_reports_zero(self):
        gdf = gpd.GeoDataFrame(geometry=[LineString([(5, 5), (9, 5)])], crs="EPSG:4326")
        r = analyze_intersection(gdf, self.tif, "LineString", intensity_threshold=98)
        self.assertEqual((r["nodata_samples"], r["nodata_meters"]), (0, 0.0))


class Note(unittest.TestCase):
    """What no data means: only the catalogue can say, and only a threshold above
    the layer's minimum makes it a known 'not exposed'."""

    def test_documented_layer_above_its_minimum(self):
        note = hazards.nodata_note(PGA, 98)
        self.assertIn("no data = below 49 cm/s²", note)
        self.assertIn("treated as not exposed", note)

    def test_documented_layer_with_a_threshold_at_or_under_its_minimum(self):
        for threshold in (30, None):
            with self.subTest(threshold=threshold):
                note = hazards.nodata_note(PGA, threshold)
                self.assertIn("no data = below 49 cm/s²", note)
                self.assertIn("could be exposed", note)
                self.assertNotIn("treated as not exposed", note)
        self.assertIn("no threshold", hazards.nodata_note(PGA, None))
        # at the minimum itself nothing below it reaches the threshold
        self.assertIn("treated as not exposed", hazards.nodata_note(PGA, 49))

    def test_undocumented_layer_is_unknown_not_below_anything(self):
        for hazard in (UNDOCUMENTED, None, {"nodata_below": "nan"}, {"nodata_below": ""}):
            with self.subTest(hazard=hazard):
                note = hazards.nodata_note(hazard, 98)
                self.assertIn("no data — hazard unknown here", note)
                self.assertNotIn("below", note)

    def test_the_catalogue_documents_the_pga_layers_and_only_them(self):
        with mock.patch.object(hazards, "_hazards_cache", None):
            catalogue = hazards.load_hazards_dict()
        pga = {k: h for k, h in catalogue.items() if h["category"] == "Earthquakes"}
        self.assertEqual(len(pga), 3)
        self.assertTrue(all(hazards.nodata_below(h) == 49.0 for h in pga.values()))
        others = [h for k, h in catalogue.items() if k not in pga]
        self.assertTrue(others)
        self.assertTrue(all(hazards.nodata_below(h) is None for h in others))


class Summary(unittest.TestCase):
    def line_result(self, nodata_m=3000.0, nodata_samples=30, samples=80):
        return {
            "affected_meters": 5000.0, "unaffected_meters": 3000.0,
            "nodata_meters": nodata_m, "samples": samples, "nodata_samples": nodata_samples,
        }

    def test_a_line_gives_shares_km_and_the_note(self):
        out = analyze.nodata_summary(self.line_result(), "LineString", PGA, 98)
        self.assertEqual(out["samples"], 80)
        self.assertEqual(out["nodata_samples"], 30)
        self.assertEqual(out["nodata_share_pct"], 37.5)
        self.assertEqual(out["nodata_km"], 3.0)
        self.assertEqual(out["nodata_length_share_pct"], 37.5)
        self.assertIn("no data = below 49 cm/s²", out["nodata_note"])

    def test_without_any_no_data_there_is_no_note(self):
        out = analyze.nodata_summary(
            self.line_result(nodata_m=0.0, nodata_samples=0), "LineString", PGA, 98
        )
        self.assertEqual(out["nodata_share_pct"], 0.0)
        self.assertNotIn("nodata_note", out)

    def test_points_have_no_length(self):
        out = analyze.nodata_summary({"samples": 4, "nodata_samples": 1}, "Point", None, None)
        self.assertEqual(out["nodata_share_pct"], 25.0)
        self.assertNotIn("nodata_km", out)
        self.assertIn("hazard unknown here", out["nodata_note"])

    def test_a_result_without_counts_adds_nothing(self):
        self.assertEqual(analyze.nodata_summary({"affected_meters": 1.0}, "LineString", PGA, 98), {})

    def test_the_assistant_summary_carries_it(self):
        info = {"geometry_type": "LineString", "feature_count": 1}
        out = domain.summarise(self.line_result(), info, PGA, 98)
        self.assertEqual(out["nodata_share_pct"], 37.5)
        self.assertEqual(out["nodata_km"], 3.0)
        self.assertIn("treated as not exposed", out["nodata_note"])

    def test_get_hazard_says_what_no_data_means(self):
        conv = conversations.Conversation(id="test-nodata", created=0.0)
        pga = {**PGA, "hazard_id": "peak_ground_acceleration_pga_475_years"}
        wind = {**UNDOCUMENTED, "category": None}
        with mock.patch.object(domain, "resolve_hazard", side_effect=[pga, wind]):
            self.assertEqual(
                query.get_hazard(conv, "pga")["nodata_means"],
                "below 49 cm/s²: the layer stores no value under its minimum",
            )
            self.assertIn("not documented", query.get_hazard(conv, "wind")["nodata_means"])


class Endpoint(unittest.TestCase):
    """The sidebar's POST /api/analyze, end to end on a temp raster."""

    def test_the_summary_has_the_no_data_share_and_note(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            upload, "_start_sweeper", lambda: None
        ):
            tif = write_raster(Path(tmp) / "pga.tif")
            client = TestClient(main.app)
            r = client.post(
                "/api/upload", files={"file": ("pts.csv", b"lat,lon\n5,2\n5,6\n5,7\n5,8\n")}
            )
            self.assertEqual(r.status_code, 200, r.text)
            file_id = r.json()["file_id"]
            self.addCleanup(upload.forget_upload, file_id)
            with mock.patch.object(hazards, "_hazards_cache", {"pga_test": PGA}):
                r = client.post(
                    "/api/analyze",
                    json={
                        "file_id": file_id, "hazard_id": "pga_test", "hazard_url": tif,
                        "intensity_threshold": 98,
                    },
                )
            self.assertEqual(r.status_code, 200, r.text)
            summary = r.json()["summary"]
            self.assertEqual(summary["affected_count"], 3)
            self.assertEqual((summary["samples"], summary["nodata_samples"]), (4, 1))
            self.assertEqual(summary["nodata_share_pct"], 25.0)
            self.assertIn("no data = below 49 cm/s²", summary["nodata_note"])


class Documented(unittest.TestCase):
    """The agent is told, in the places it reads, to report the share and what it means."""

    def test_the_guides_the_tools_and_the_mcp_instructions_name_the_fields(self):
        for text in (
            guides.EXPOSURE_ANALYSIS,
            tools.REGISTRY["run_analysis"].description,
            tools.REGISTRY["compare_hazards"].description,
            mcp_server.INSTRUCTIONS,
        ):
            self.assertIn("nodata_share_pct", text)
        self.assertIn("49 cm/s²", guides.EXPOSURE_ANALYSIS)
        self.assertIn("nodata_note", tools.REGISTRY["run_analysis"].description)
        self.assertIn("nodata_means", tools.REGISTRY["get_hazard"].description)


if __name__ == "__main__":
    unittest.main()
