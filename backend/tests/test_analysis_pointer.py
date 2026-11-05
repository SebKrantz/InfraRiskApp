"""A follow-up call that names an analysis by dataset + hazard + threshold must get
the one its own conversation ran, not whatever another chat (or another curve)
wrote to the app's shared analysis cache since. No rasters: the analysis is faked.

Run from backend/ with the repo's venv:  python -m unittest discover -s tests
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd  # noqa: E402
from shapely.geometry import Point  # noqa: E402

from app.api.analyze import clear_raster_cache_for_file  # noqa: E402
from app.api.upload import uploaded_files  # noqa: E402
from app.assistant import conversations, domain, tools  # noqa: E402

tools.load()

from app.assistant.tools import analysis, output, results  # noqa: E402

FILE_ID = "test-pointer-points"
THRESHOLD = 100.0
VALUE = 1000.0
RATIO = {"low": 0.1, "high": 0.5}  # the two curves' damage ratios


def fake_analyze(infrastructure_gdf, hazard_raster_path, geometry_type,
                 intensity_threshold=None, vulnerability_curve_interp=None,
                 replacement_value=None, **kw):
    """Two exposed points; with a curve, a damage cost of ratio x value each."""
    gdf = infrastructure_gdf.copy()
    gdf["affected"] = True
    out = {"affected_count": len(gdf), "unaffected_count": 0, "full_gdf": gdf}
    if vulnerability_curve_interp is not None:
        gdf["damage_cost"] = vulnerability_curve_interp(1.0) * replacement_value
        out["total_damage_cost"] = float(gdf["damage_cost"].sum())
    return out


class AnalysisPointer(unittest.TestCase):
    def setUp(self):
        uploaded_files[FILE_ID] = {
            "gdf": gpd.GeoDataFrame(
                {"name": ["a", "b"]}, geometry=[Point(15.0, -4.0), Point(15.1, -4.0)],
                crs="EPSG:4326",
            ),
            "filename": "points.gpkg",
            "geometry_type": "Point",
            "feature_count": 2,
            "bounds": [15.0, -4.0, 15.1, -4.0],
        }
        self.hazard = next(iter(domain.catalogue()))
        # Two chats of one person, both reading the same dataset.
        self.chats = [
            conversations.Conversation(id=f"test-pointer-{n}", created=0.0) for n in (1, 2)
        ]
        for chat in self.chats:
            for name, ratio in RATIO.items():
                chat.curves[name] = {
                    "interp": lambda x, r=ratio: r, "lower": lambda x, r=ratio: r,
                    "upper": lambda x, r=ratio: r, "has_bounds": False,
                    "path": Path(f"{name}.csv"), "provenance": {"kind": "test"},
                }
        patcher = mock.patch.object(domain, "analyze_intersection", side_effect=fake_analyze)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(clear_raster_cache_for_file, FILE_ID)
        self.addCleanup(uploaded_files.pop, FILE_ID, None)
        for chat in self.chats:
            self.addCleanup(conversations._cleanup, chat)

    def run_analysis(self, chat, curve=None):
        return analysis.run_analysis(
            chat, FILE_ID, self.hazard, threshold=THRESHOLD, curve=curve,
            replacement_value=VALUE if curve else None,
        )["stored_as"]

    def costs(self, chat, **ref):
        """The damage cost of each point in the table a follow-up call gets."""
        ref = ref or dict(file_id=FILE_ID, hazard=self.hazard, threshold=THRESHOLD)
        rows = results.get_analysis_table(chat, **ref)["rows"]
        return [row.get("damage_cost") for row in rows]

    def test_another_chats_curve_does_not_replace_mine(self):
        mine, theirs = self.chats
        self.run_analysis(mine, "low")
        self.run_analysis(theirs, "high")
        self.assertEqual(self.costs(mine), [RATIO["low"] * VALUE] * 2)
        self.assertEqual(self.costs(theirs), [RATIO["high"] * VALUE] * 2)

    def test_another_chats_exposure_run_does_not_strip_my_damage(self):
        mine, theirs = self.chats
        self.run_analysis(mine, "low")
        self.run_analysis(theirs)
        self.assertEqual(self.costs(mine), [RATIO["low"] * VALUE] * 2)
        self.assertEqual(self.costs(theirs), [None, None])

    def test_in_one_chat_a_call_by_key_is_its_latest_run_and_stored_as_pins_one(self):
        chat = self.chats[0]
        first = self.run_analysis(chat, "low")
        self.run_analysis(chat, "high")
        self.assertEqual(self.costs(chat), [RATIO["high"] * VALUE] * 2)
        self.assertEqual(self.costs(chat, stored_as=first), [RATIO["low"] * VALUE] * 2)

    def test_a_figure_follows_the_analysis_it_is_given(self):
        mine, theirs = self.chats
        first = self.run_analysis(mine, "low")
        self.run_analysis(theirs, "high")
        with mock.patch("app.api.export.generate_barchart_png", return_value=b"png") as draw:
            output.make_barchart(mine, first)
        self.assertEqual(draw.call_args.args[0]["total_damage_cost"], 2 * RATIO["low"] * VALUE)


if __name__ == "__main__":
    unittest.main()
