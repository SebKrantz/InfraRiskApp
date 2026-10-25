"""Heavy work is bounded and kept off the event loop (app/concurrency.py,
geospatial.analyze_intersection, the upload and analyze endpoints): at most
ANALYSIS_SLOTS analyses sample rasters at once under a flood of jobs, UI calls
and assistant runs; a cached re-threshold never waits; sampling threads are
capped; a slow upload does not stall other requests.

Run from backend/ with the repo's venv:  python -m unittest discover -s tests
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import threading
import time
import unittest
from concurrent import futures
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd  # noqa: E402
import httpx  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402
from shapely.geometry import Point  # noqa: E402

from app import concurrency, config, datasets  # noqa: E402
from app.api import upload  # noqa: E402
from app.assistant import domain  # noqa: E402
from app.utils import geospatial  # noqa: E402

import main  # noqa: E402

POINTS = gpd.GeoDataFrame(geometry=[Point(15, -4), Point(15.1, -4)], crs="EPSG:4326")


class Gauge:
    """A stand-in for the analysis that records how many run at once."""

    def __init__(self, seconds: float = 0.05) -> None:
        self.seconds, self.now, self.peak, self.calls = seconds, 0, 0, 0
        self.lock = threading.Lock()

    def __call__(self, *args, **kwargs):
        with self.lock:
            self.now += 1
            self.calls += 1
            self.peak = max(self.peak, self.now)
        time.sleep(self.seconds)
        with self.lock:
            self.now -= 1
        return {"affected_count": 0, "unaffected_count": 2, "full_gdf": None,
                "raster_values": np.zeros(2), "raster_complete": True}


class AnalysisSlots(unittest.TestCase):
    def setUp(self):
        self.gauge = Gauge()
        for patch in (
            mock.patch.object(concurrency, "ANALYSIS_SLOTS", threading.BoundedSemaphore(2)),
            mock.patch.object(geospatial, "_analyze_intersection", self.gauge),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def test_a_flood_of_jobs_ui_calls_and_assistant_runs_shares_two_slots(self):
        fid = datasets.add("local", {"filename": "p", "geometry_type": "Point",
                                     "feature_count": 2, "gdf": POINTS})
        self.addCleanup(datasets.remove, "local", fid)
        hazards = [{"hazard_id": f"h{i}", "dataset_url": "unused.tif"} for i in range(6)]
        jobs = [concurrency.submit_job(geospatial.analyze_intersection, POINTS, "x.tif", "Point")
                for _ in range(12)]
        ui = [threading.Thread(target=geospatial.analyze_intersection,
                               args=(POINTS, "x.tif", "Point")) for _ in range(6)]
        assistant = [threading.Thread(target=domain.run_exposure, args=(fid, h))
                     for h in hazards]
        for t in ui + assistant:
            t.start()
        for t in ui + assistant:
            t.join(10)
        futures.wait(jobs, timeout=10)
        self.assertEqual(self.gauge.calls, 24)
        self.assertEqual(self.gauge.peak, 2)

    def test_a_cached_rethreshold_does_not_wait_for_a_slot(self):
        concurrency.ANALYSIS_SLOTS.acquire()
        concurrency.ANALYSIS_SLOTS.acquire()
        try:
            started = time.monotonic()
            geospatial.analyze_intersection(POINTS, "x.tif", "Point",
                                            cached_raster_values=np.zeros(2))
            self.assertLess(time.monotonic() - started, 0.5)
        finally:
            concurrency.ANALYSIS_SLOTS.release()
            concurrency.ANALYSIS_SLOTS.release()

    def test_jobs_run_on_a_bounded_pool(self):
        names = set()

        def note():
            names.add(threading.current_thread().name)
            time.sleep(0.02)

        futures.wait([concurrency.submit_job(note) for _ in range(30)], timeout=10)
        self.assertLessEqual(len(names), config.JOB_WORKERS)


class SamplingThreads(unittest.TestCase):
    def test_capped_per_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            tif = str(Path(tmp) / "h.tif")
            with rasterio.open(
                tif, "w", driver="GTiff", height=40, width=400, count=1, dtype="float32",
                crs="EPSG:4326", transform=from_origin(0.0, 2.0, 0.1, 0.1),
            ) as dst:
                dst.write(np.ones((1, 40, 400), dtype="float32"))
            xs = np.arange(0.5, 39.5, 1.0)  # 39 one-degree tiles
            seen = []
            real = futures.ThreadPoolExecutor

            def recording(*args, **kwargs):
                seen.append(kwargs.get("max_workers"))
                return real(*args, **kwargs)

            with mock.patch.object(futures, "ThreadPoolExecutor", side_effect=recording):
                values = geospatial.sample_raster_at_points(tif, xs, np.full(len(xs), 1.0))
        self.assertTrue((values == 1.0).all())
        self.assertEqual(seen, [config.SAMPLING_THREADS])
        self.assertLessEqual(config.SAMPLING_THREADS, 8)


class EventLoop(unittest.IsolatedAsyncioTestCase):
    async def test_a_slow_upload_does_not_stall_other_requests(self):
        real = upload.load_csv_points

        def slow_parse(path):
            time.sleep(1.0)  # a large file's parse and reprojection
            return real(path)

        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://risk") as client:
            with mock.patch.object(upload, "load_csv_points", side_effect=slow_parse):
                sent = asyncio.create_task(client.post(
                    "/api/upload",
                    files={"file": ("t.csv", b"name,lat,lon\na,-4.0,15.0\n", "text/csv")},
                ))
                await asyncio.sleep(0.2)  # the upload is being parsed now
                started = time.monotonic()
                probe = await client.get("/health")
                waited = time.monotonic() - started
                self.assertFalse(sent.done())
                response = await sent
        self.assertEqual(probe.status_code, 200)
        self.assertLess(waited, 0.4)
        self.assertEqual(response.status_code, 200, response.text)
        datasets.STORE.pop(response.json()["file_id"], None)


if __name__ == "__main__":
    unittest.main()
