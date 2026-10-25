"""The analysis caches (app/api/analyze.py): a byte budget fair between owners,
keys that include the curve and the replacement value, copies handed out, and
no caching of raster samples with failed window reads.

Run from backend/ with the repo's venv:  python -m unittest discover -s tests
"""

from __future__ import annotations

import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402
from shapely.geometry import Point  # noqa: E402

from app import datasets  # noqa: E402
from app.api import analyze  # noqa: E402
from app.assistant import domain  # noqa: E402
from app.utils import geospatial  # noqa: E402

MB = 2**20


class ByteBudget(unittest.TestCase):
    def test_stays_within_budget_and_the_heaviest_owner_pays(self):
        cache = analyze.ByteBudgetCache(10)  # 10 MB
        for i in range(8):
            cache.put(("heavy", i), i, "heavy@x", 1 * MB)
        cache.put(("light", 0), "keep", "light@x", 1 * MB)
        for i in range(8, 20):  # the heavy user keeps going
            cache.put(("heavy", i), i, "heavy@x", 1 * MB)
        self.assertLessEqual(sum(cache.bytes_by_owner().values()), 10 * MB)
        self.assertEqual(cache.get(("light", 0)), "keep")
        self.assertIn(("heavy", 19), cache)
        self.assertNotIn(("heavy", 0), cache)  # least recently used went first

    def test_reading_an_entry_keeps_it(self):
        cache = analyze.ByteBudgetCache(3)
        for i in range(3):
            cache.put(i, i, "a@x", 1 * MB)
        cache.get(0)
        cache.put(3, 3, "a@x", 1 * MB)
        self.assertIn(0, cache)
        self.assertNotIn(1, cache)

    def test_a_new_entry_is_never_its_own_victim(self):
        cache = analyze.ByteBudgetCache(10)
        cache.put("small", 1, "a@x", 4 * MB)
        self.assertTrue(cache.put("big", 2, "b@x", 8 * MB))
        self.assertIn("big", cache)
        self.assertNotIn("small", cache)

    def test_an_entry_larger_than_the_budget_is_not_kept(self):
        cache = analyze.ByteBudgetCache(1)
        self.assertFalse(cache.put("x", 1, "a@x", 2 * MB))
        self.assertNotIn("x", cache)


def fake_analysis(**kwargs):
    """analyze_intersection's shape for a two-point dataset, the damage cost
    telling which curve and replacement value produced it."""
    curve = kwargs.get("vulnerability_curve_interp")
    rv = kwargs.get("replacement_value")
    out = {"affected_count": 1, "unaffected_count": 1, "full_gdf": None,
           "raster_values": np.array([1.0, 0.0]), "raster_complete": True}
    if curve is not None:
        out["total_damage_cost"] = curve(1.0) * rv
    return out


class Keys(unittest.TestCase):
    def setUp(self):
        gdf = gpd.GeoDataFrame(geometry=[Point(15, -4), Point(15.1, -4)], crs="EPSG:4326")
        self.fid = datasets.add("local", {"filename": "p", "geometry_type": "Point",
                                          "feature_count": 2, "gdf": gdf})
        self.addCleanup(datasets.remove, "local", self.fid)
        self.hazard = {"hazard_id": "flood_test", "dataset_url": "unused.tif"}
        patch = mock.patch.object(domain, "analyze_intersection", side_effect=fake_analysis)
        self.run = patch.start()
        self.addCleanup(patch.stop)

    def damage(self, curve, digest, rv=10.0, threshold=100.0):
        return domain.run_exposure(
            self.fid, self.hazard, threshold=threshold, vulnerability_curve_interp=curve,
            replacement_value=rv, curve_digest=digest,
        )

    def test_another_curve_or_value_is_another_analysis(self):
        half, full = (lambda x: 0.5), (lambda x: 1.0)
        self.assertEqual(self.damage(half, "d-half")["total_damage_cost"], 5.0)
        self.assertEqual(self.damage(full, "d-full")["total_damage_cost"], 10.0)
        self.assertEqual(self.damage(full, "d-full", rv=20.0)["total_damage_cost"], 20.0)
        self.assertEqual(self.run.call_count, 3)
        # the same inputs again come from the cache
        self.assertEqual(self.damage(half, "d-half")["total_damage_cost"], 5.0)
        self.assertEqual(self.run.call_count, 3)

    def test_exposure_and_damage_runs_never_stand_in_for_each_other(self):
        domain.run_exposure(self.fid, self.hazard, threshold=100.0)
        out = self.damage(lambda x: 0.5, "d-half")
        self.assertEqual(out["total_damage_cost"], 5.0)
        again = domain.run_exposure(self.fid, self.hazard, threshold=100.0)
        self.assertNotIn("total_damage_cost", again)
        self.assertEqual(self.run.call_count, 2)

    def test_the_export_buttons_get_the_latest_analysis_of_the_threshold(self):
        self.damage(lambda x: 0.5, "d-half")
        self.damage(lambda x: 1.0, "d-full")
        latest = analyze.get_cached_analysis_result(self.fid, "flood_test", 100.0)
        self.assertEqual(latest["total_damage_cost"], 10.0)

    def test_the_sidebar_and_the_assistant_digest_a_curve_alike(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "curve.csv"
            path.write_bytes(b"intensity,proportion\n0,0\n1,1\n")
            self.assertEqual(domain.curve_digest({"path": path}),
                             analyze.curve_digest(path.read_bytes()))

    def test_callers_get_copies(self):
        first = domain.run_exposure(self.fid, self.hazard, threshold=1.0)
        first["_assistant_meta"]["curve"] = "scribbled"
        first["extra"] = True
        again = domain.run_exposure(self.fid, self.hazard, threshold=1.0)
        self.assertNotIn("curve", again["_assistant_meta"])
        self.assertNotIn("extra", again)
        self.assertEqual(self.run.call_count, 1)


class FailedReads(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tif = str(Path(tmp.name) / "hazard.tif")
        with rasterio.open(
            self.tif, "w", driver="GTiff", height=10, width=10, count=1, dtype="float32",
            crs="EPSG:4326", transform=from_origin(14.5, -3.5, 0.1, 0.1),
        ) as dst:
            dst.write(np.full((1, 10, 10), 5.0, dtype="float32"))

    def flaky_open(self):
        """rasterio.open that works for the metadata read and fails for the windows."""
        real = rasterio.open
        calls = {"n": 0}

        def opener(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] > 1:
                raise rasterio.errors.RasterioIOError("simulated network failure")
            return real(*args, **kwargs)

        return mock.patch.object(geospatial.rasterio, "open", side_effect=opener)

    def test_the_failure_is_reported(self):
        failures: list = []
        with self.flaky_open():
            values = geospatial.sample_raster_at_points(
                self.tif, np.array([15.0]), np.array([-4.0]), failures=failures)
        self.assertTrue(np.isnan(values).all())
        self.assertEqual(len(failures), 1)

    def test_samples_with_a_failed_read_are_not_cached(self):
        gdf = gpd.GeoDataFrame(geometry=[Point(15.0, -4.0)], crs="EPSG:4326")
        fid = str(uuid.uuid4())
        with self.flaky_open():
            result = geospatial.analyze_intersection(gdf, self.tif, "Point")
        self.assertFalse(result["raster_complete"])
        analyze.cache_raster_values(fid, "h", "Point", result)
        self.assertIsNone(analyze.get_cached_raster_values(fid, "h"))

        good = geospatial.analyze_intersection(gdf, self.tif, "Point")
        self.assertTrue(good["raster_complete"])
        self.assertEqual(good["affected_count"], 1)
        analyze.cache_raster_values(fid, "h", "Point", good)
        self.assertIsNotNone(analyze.get_cached_raster_values(fid, "h"))
        analyze.clear_raster_cache_for_file(fid)


if __name__ == "__main__":
    unittest.main()
