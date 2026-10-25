"""Datasets have owners (app/datasets.py): user B sees nothing of user A's through
any route, tool or python_exec, and the store's lifecycle — TTL, per-owner count
and byte quotas, eviction with cache purge, an MCP scope's datasets dropped with
the scope.

Run from backend/ with the repo's venv:  python -m unittest discover -s tests
"""

from __future__ import annotations

import sys
import time
import unittest
import uuid
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from shapely.geometry import Point  # noqa: E402

from app import config, datasets  # noqa: E402
from app.api import analyze  # noqa: E402
from app.api.hazards import load_hazards_dict  # noqa: E402
from app.assistant import conversations, domain, mcp_server, tools  # noqa: E402

tools.load()

from app.assistant.tools import analysis, output, query  # noqa: E402

import main  # noqa: E402
from multiuser_helpers import multi_user, signed_in  # noqa: E402

A, B = "a@example.org", "b@example.org"
CSV = b"name,lat,lon\na,-4.0,15.0\nb,-4.1,15.1\n"


def points_info(n: int = 2) -> dict:
    gdf = gpd.GeoDataFrame(
        {"name": [f"p{i}" for i in range(n)]},
        geometry=[Point(15.0 + i / 10, -4.0) for i in range(n)],
        crs="EPSG:4326",
    )
    return {"filename": "p.csv", "geometry_type": "Point", "feature_count": n,
            "crs": "EPSG:4326", "bounds": {}, "gdf": gdf}


class Isolation(unittest.TestCase):
    """Everything A owns is invisible to B, and B's attempts change nothing."""

    def setUp(self):
        patch = multi_user()
        patch.start()
        self.addCleanup(patch.stop)
        self.client = TestClient(main.app)
        r = self.client.post(
            "/api/upload?file_id=chosen-by-client",
            files={"file": ("towers.csv", CSV, "text/csv")},
            headers=signed_in(A),
        )
        self.assertEqual(r.status_code, 200, r.text)
        self.fid = r.json()["file_id"]
        self.addCleanup(datasets.STORE.pop, self.fid, None)
        self.hazard = next(iter(load_hazards_dict()))
        self.conv_a = conversations.get_or_create(uuid.uuid4().hex, owner=A)
        self.conv_b = conversations.get_or_create(uuid.uuid4().hex, owner=B)
        for conv in (self.conv_a, self.conv_b):
            self.addCleanup(conversations.drop, conv.id)

    def test_the_server_issues_the_file_id(self):
        self.assertNotEqual(self.fid, "chosen-by-client")
        self.assertEqual(datasets.STORE[self.fid]["owner"], A)

    def test_http_routes(self):
        b = signed_in(B)
        self.assertEqual(self.client.get(f"/api/upload/{self.fid}", headers=b).status_code, 404)
        self.assertEqual(self.client.delete(f"/api/upload/{self.fid}", headers=b).status_code, 404)
        self.assertEqual(
            self.client.get(f"/api/analyze/{self.fid}/status", headers=b).status_code, 404)
        self.assertEqual(
            self.client.get(f"/api/assistant/datasets/{self.fid}", headers=b).status_code, 404)
        with mock.patch.object(analyze, "analyze_intersection") as run:
            r = self.client.post(
                "/api/analyze", headers=b,
                json={"file_id": self.fid, "hazard_id": self.hazard, "hazard_url": "x"},
            )
        self.assertEqual(r.status_code, 404)
        run.assert_not_called()
        body = {"file_id": self.fid, "hazard_id": self.hazard}
        for route, extra in (("barchart", {}), ("map", {}), ("data", {"mode": "csv_points"})):
            with self.subTest(route=route):
                r = self.client.post(f"/api/export/{route}", json={**body, **extra}, headers=b)
                self.assertEqual(r.status_code, 404)
        # nothing of A's changed, and A still has it all
        a = signed_in(A)
        self.assertEqual(self.client.get(f"/api/upload/{self.fid}", headers=a).status_code, 200)
        self.assertEqual(
            self.client.get(f"/api/assistant/datasets/{self.fid}", headers=a).status_code, 200)

    def test_assistant_tools(self):
        self.assertEqual(query.list_datasets(self.conv_b)["count"], 0)
        self.assertEqual(query.list_datasets(self.conv_a)["count"], 1)
        with self.assertRaisesRegex(ValueError, "no dataset"):
            analysis.run_analysis(self.conv_b, self.fid, self.hazard)
        with self.assertRaisesRegex(ValueError, "no dataset"):
            analysis.sweep_thresholds(self.conv_b, self.fid, self.hazard, [1.0])
        with self.assertRaisesRegex(ValueError, "no dataset"):
            output.export_analysis_data(self.conv_b, self.fid, self.hazard, "csv_points")
        with self.assertRaisesRegex(ValueError, "no dataset"):
            domain.get_dataset(self.fid, B)

    def test_python_exec_view(self):
        res = analysis.python_exec(self.conv_b, f"print(list(uploaded_files)); {self.fid!r} in uploaded_files")
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["stdout"].strip(), "[]")
        self.assertEqual(res["result"], "False")
        res = analysis.python_exec(self.conv_a, "list(uploaded_files)")
        self.assertEqual(res["result"], repr([self.fid]))

    def test_the_mcp_scope_is_its_own_owner(self):
        conv = conversations.get_or_create(mcp_server._MCP_CONVERSATION_ID, owner=mcp_server._owner())
        self.assertEqual(query.list_datasets(conv)["count"], 0)


class SingleUserMode(unittest.TestCase):
    def test_everyone_sees_everything_as_before(self):
        fid = datasets.add(config.AEI_LOCAL_USER, points_info())
        self.addCleanup(datasets.STORE.pop, fid, None)
        conv = conversations.get_or_create(uuid.uuid4().hex)
        self.addCleanup(conversations.drop, conv.id)
        self.assertIn(fid, [d["file_id"] for d in query.list_datasets(conv)["datasets"]])
        self.assertEqual(TestClient(main.app).get(f"/api/upload/{fid}").status_code, 200)


class Lifecycle(unittest.TestCase):
    def setUp(self):
        # A private store: the quotas and the TTL must not touch other tests' datasets
        for patch in (
            mock.patch.object(datasets, "STORE", {}),
            mock.patch.multiple(config, DATASET_MAX_PER_OWNER=2, DATASET_MAX_MB_PER_OWNER=1.0),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def test_count_quota_evicts_the_owners_least_recently_used(self):
        b1 = datasets.add(B, points_info())
        a1 = datasets.add(A, points_info())
        a2 = datasets.add(A, points_info())
        datasets.get(A, a1)  # a1 is now the more recently used
        a3 = datasets.add(A, points_info())
        self.assertEqual({fid for fid, _ in datasets.list_for(A)}, {a1, a3})
        self.assertNotIn(a2, datasets.STORE)
        self.assertIn(b1, datasets.STORE)  # another owner's are never touched

    def test_byte_quota(self):
        with mock.patch.object(datasets, "approx_bytes", return_value=400 * 2**10), \
                mock.patch.object(config, "DATASET_MAX_PER_OWNER", 10):
            first = datasets.add(A, points_info())
            datasets.add(A, points_info())
            datasets.add(A, points_info())  # 3 x 400 KB > 1 MB: the first goes
        self.assertNotIn(first, datasets.STORE)
        self.assertEqual(len(datasets.list_for(A)), 2)

    def test_one_dataset_over_the_quota_is_refused(self):
        with mock.patch.object(datasets, "approx_bytes", return_value=2 * 2**20):
            with self.assertRaises(datasets.QuotaExceeded):
                datasets.add(A, points_info())
        self.assertEqual(datasets.list_for(A), [])

    def test_unused_datasets_expire(self):
        old = datasets.add(A, points_info())
        datasets.STORE[old]["last_used"] = time.time() - config.DATASET_TTL_HOURS * 3600 - 60
        with mock.patch.object(datasets, "_last_sweep", 0.0):
            with self.assertRaises(datasets.DatasetNotFound):
                datasets.get(A, old)
        self.assertNotIn(old, datasets.STORE)

    def test_eviction_purges_the_caches(self):
        doomed = datasets.add(A, points_info())
        analyze.set_cached_raster_values(doomed, "h", np.zeros(2))
        analyze.set_cached_analysis_result(doomed, "h", None, {"affected_count": 0})
        datasets.add(A, points_info())
        datasets.add(A, points_info())  # count quota 2: `doomed` goes
        self.assertIsNone(analyze.get_cached_raster_values(doomed, "h"))
        self.assertIsNone(analyze.get_cached_analysis_result(doomed, "h", None))

    def test_delete_purges_the_caches(self):
        fid = datasets.add(A, points_info())
        analyze.set_cached_raster_values(fid, "h", np.zeros(2))
        datasets.remove(A, fid)
        self.assertIsNone(analyze.get_cached_raster_values(fid, "h"))

    def test_a_scope_takes_its_own_datasets_with_it(self):
        with multi_user():
            scope = conversations.get_or_create("mcp:agui-1", owner="mcp:agui-1")
            chat = conversations.get_or_create(uuid.uuid4().hex, owner=A)
            mcp_fid = datasets.add(scope.owner, points_info(), scope=scope.id)
            chat_fid = datasets.add(chat.owner, points_info(), scope=chat.id)
            conversations.drop(scope.id)
            conversations.drop(chat.id)
        self.assertNotIn(mcp_fid, datasets.STORE)
        # a person's dataset stays for their map; TTL and quota retire it
        self.assertIn(chat_fid, datasets.STORE)

    def test_an_upload_over_the_quota_is_413(self):
        with mock.patch.object(datasets, "approx_bytes", return_value=2 * 2**20):
            r = TestClient(main.app).post(
                "/api/upload", files={"file": ("towers.csv", CSV, "text/csv")})
        self.assertEqual(r.status_code, 413)


if __name__ == "__main__":
    unittest.main()
