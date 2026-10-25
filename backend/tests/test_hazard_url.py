"""POST /api/analyze reads the hazard raster from the catalogue by hazard_id and
never opens a URL or path the client sends as hazard_url.

Run from backend/ with the repo's venv:  python -m unittest discover -s tests
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import datasets  # noqa: E402
from app.api import analyze  # noqa: E402
from app.api.hazards import load_hazards_dict  # noqa: E402

import main  # noqa: E402

CSV = b"name,lat,lon\na,-4.0,15.0\nb,-4.1,15.1\n"
CURVE = b"intensity,proportion_destroyed\n0,0\n100,0.5\n1000,1\n"
FORGED = ("/etc/passwd", "http://169.254.169.254/latest/meta-data", "file:///app/backend/.env")


def fake_analysis(infrastructure_gdf, **kwargs):
    gdf = infrastructure_gdf.copy()
    gdf["affected"] = False
    return {"affected_count": 0, "unaffected_count": len(gdf), "full_gdf": gdf,
            "raster_values": np.zeros(len(gdf)), "raster_complete": True}


class HazardUrl(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)
        r = self.client.post("/api/upload", files={"file": ("t.csv", CSV, "text/csv")})
        self.fid = r.json()["file_id"]
        self.addCleanup(datasets.STORE.pop, self.fid, None)
        self.hazard_id, hazard = next(iter(load_hazards_dict().items()))
        self.catalogue_url = hazard["dataset_url"]
        patch = mock.patch.object(analyze, "analyze_intersection", side_effect=fake_analysis)
        self.run = patch.start()
        self.addCleanup(patch.stop)

    def opened(self) -> list[str]:
        return [c.kwargs["hazard_raster_path"] for c in self.run.call_args_list]

    def test_json_body_hazard_url_is_ignored(self):
        for forged in FORGED:
            with self.subTest(forged=forged):
                r = self.client.post("/api/analyze", json={
                    "file_id": self.fid, "hazard_id": self.hazard_id, "hazard_url": forged,
                    "intensity_threshold": 1.5,
                })
                self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(set(self.opened()), {self.catalogue_url})

    def test_form_hazard_url_is_ignored(self):
        r = self.client.post(
            "/api/analyze",
            data={"file_id": self.fid, "hazard_id": self.hazard_id, "hazard_url": FORGED[1],
                  "replacement_value": "1000"},
            files={"vulnerability_curve_file": ("c.csv", CURVE, "text/csv")},
        )
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(self.opened(), [self.catalogue_url])

    def test_no_hazard_url_needed(self):
        r = self.client.post("/api/analyze", json={"file_id": self.fid, "hazard_id": self.hazard_id})
        self.assertEqual(r.status_code, 200, r.text)

    def test_unknown_hazard_is_404_and_nothing_is_opened(self):
        r = self.client.post("/api/analyze", json={
            "file_id": self.fid, "hazard_id": "no-such-layer", "hazard_url": FORGED[0],
        })
        self.assertEqual(r.status_code, 404)
        self.run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
