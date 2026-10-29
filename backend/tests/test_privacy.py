"""What happens to an uploaded file: nothing left on disk, gone when the user deletes
it, and no other website may call the API. Expiry, quotas and cache purging belong to
app/datasets.py here and are tested in test_datasets.py.

Run: cd backend && python -m unittest discover -s tests
"""

from __future__ import annotations

import io
import sqlite3
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from shapely.geometry import Point  # noqa: E402

import main  # noqa: E402

GDF = gpd.GeoDataFrame({"name": ["a", "b"], "value": [1.0, 2.0]},
                       geometry=[Point(10, 10), Point(11, 11)], crs="EPSG:4326")


def _samples(folder: Path) -> dict[str, bytes]:
    """One file per accepted format, plus two broken ones."""
    GDF.to_file(folder / "plain.gpkg", driver="GPKG")
    GDF.to_file(folder / "wal.gpkg", driver="GPKG")
    con = sqlite3.connect(folder / "wal.gpkg")  # WAL mode makes SQLite write side files
    con.execute("PRAGMA journal_mode=WAL")
    con.close()
    (folder / "shp").mkdir()
    GDF.to_file(folder / "shp" / "pts.shp")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for f in (folder / "shp").iterdir():
            z.write(f, f.name)
    return {
        "plain.gpkg": (folder / "plain.gpkg").read_bytes(),
        "wal.gpkg": (folder / "wal.gpkg").read_bytes(),
        "pts.zip": buf.getvalue(),
        "pts.csv": b"lat,lon,name\n10,10,a\n11,11,b\n",
        "pts.geojson": GDF.to_json().encode(),
        "broken.gpkg": b"not a geopackage",
        "broken.zip": b"PK\x03\x04 truncated",
    }


class UploadTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)
        self.ids: list[str] = []

    def tearDown(self):
        for file_id in self.ids:
            self.client.delete(f"/api/upload/{file_id}")

    def _upload(self, name: str, data: bytes) -> str:
        r = self.client.post("/api/upload", files={"file": (name, data)})
        self.assertEqual(r.status_code, 200, r.text)
        self.ids.append(r.json()["file_id"])
        return self.ids[-1]

    def test_nothing_is_left_in_the_temp_folder(self):
        with tempfile.TemporaryDirectory() as samples, tempfile.TemporaryDirectory() as tmp:
            files = _samples(Path(samples))
            with mock.patch.object(tempfile, "tempdir", tmp):
                for name, data in files.items():
                    with self.subTest(file=name):
                        r = self.client.post("/api/upload", files={"file": (name, data)})
                        self.assertEqual(r.status_code, 500 if "broken" in name else 200)
                        if r.status_code == 200:
                            self.ids.append(r.json()["file_id"])
                        self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_delete_drops_the_data(self):
        file_id = self._upload("pts.csv", b"lat,lon\n1,1\n")
        self.assertEqual(self.client.delete(f"/api/upload/{file_id}").status_code, 200)
        self.assertEqual(self.client.get(f"/api/upload/{file_id}").status_code, 404)
        self.assertEqual(self.client.delete(f"/api/upload/{file_id}").status_code, 404)


class CorsTest(unittest.TestCase):
    def test_other_websites_get_no_cors_grant(self):
        client = TestClient(main.app)
        r = client.get("/api/hazards", headers={"Origin": "https://evil.example", "Cookie": "s=1"})
        self.assertNotIn("access-control-allow-origin", r.headers)
        r = client.options("/api/upload", headers={
            "Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
        self.assertNotIn("access-control-allow-origin", r.headers)


if __name__ == "__main__":
    unittest.main()
