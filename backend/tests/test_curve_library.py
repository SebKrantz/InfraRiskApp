"""The house default railway PGA curve (data/vulnerability_curves/house/).

The shipped CSV must be exactly what the documented parameters give, so a change
to either is caught here; and the library tools must carry the curve like any
other, with its own provenance and limitations.

Run from backend/ with the repo's venv:  python -m unittest discover -s tests
"""

from __future__ import annotations

import sys
import unittest
from math import erf, log, sqrt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from app.assistant import conversations, curve_library, tools  # noqa: E402

tools.load()

from app.assistant.tools.curves import search_curve_library, use_library_curve  # noqa: E402
from app.assistant.tools.ead import _ead_curve  # noqa: E402

CURVE_ID = "AEI-RAIL-PGA"
PGA_475 = "peak_ground_acceleration_pga_475_years"

# The documented parameters (house README, the index row, read_guide).
G = 980.665  # cm/s2 per g
MEDIANS_G = [0.15, 0.30, 0.50, 1.00]  # slight, moderate, extensive, complete
BETA = 0.6
DAMAGE_RATIOS = [0.05, 0.20, 0.50, 0.70]
BAND = 1.5


def mean_damage_factor(pga, shift=1.0):
    """MDF = sum_i DR_i x [P(DS >= i) - P(DS >= i+1)], P a lognormal CDF."""
    if pga <= 0:
        return 0.0
    exceed = [0.5 * (1 + erf(log(pga / (m * G * shift)) / BETA / sqrt(2))) for m in MEDIANS_G]
    exceed.append(0.0)
    return sum(dr * (exceed[i] - exceed[i + 1]) for i, dr in enumerate(DAMAGE_RATIOS))


class RailwayPGACurve(unittest.TestCase):
    def setUp(self):
        self.conv = conversations.Conversation(id="test-curve-library", created=0.0)
        self.addCleanup(conversations._cleanup, self.conv)

    def test_the_file_is_what_the_documented_parameters_give(self):
        frame = np.genfromtxt(
            curve_library.path_for(curve_library.get(CURVE_ID)),
            delimiter=",", names=True,
        )
        self.assertEqual(len(frame), 61)
        self.assertEqual((frame["intensity"][0], frame["intensity"][-1]), (0.0, 1500.0))
        for x, lo, mid, hi in frame:
            self.assertAlmostEqual(lo, mean_damage_factor(x, BAND), places=5)
            self.assertAlmostEqual(mid, mean_damage_factor(x), places=5)
            self.assertAlmostEqual(hi, mean_damage_factor(x, 1 / BAND), places=5)

    def test_it_is_a_sane_track_curve(self):
        frame = np.genfromtxt(
            curve_library.path_for(curve_library.get(CURVE_ID)),
            delimiter=",", names=True,
        )
        self.assertEqual(frame["central"][0], 0.0)
        for column in ("lower", "central", "upper"):
            self.assertTrue((np.diff(frame[column]) >= 0).all(), column)
        self.assertTrue((frame["lower"] <= frame["central"]).all())
        self.assertTrue((frame["central"] <= frame["upper"]).all())
        # Capped at 0.70 whatever the shaking: a per-metre value is not all lost.
        self.assertLessEqual(frame["upper"].max(), 0.70)
        # The index row quotes the same numbers.
        row = curve_library.get(CURVE_ID)
        self.assertEqual(float(row["damage_max"]), round(frame["central"].max(), 4))
        for part in row["damage_at"].split("; "):
            x, y = part.split(":")
            self.assertAlmostEqual(
                float(y), float(np.interp(float(x), frame["intensity"], frame["central"])),
                places=4,
            )

    def test_the_library_finds_it_for_a_railway_under_pga(self):
        for asset in ("railway", "railway track", "railway track and formation"):
            found = search_curve_library(self.conv, hazard=PGA_475, asset=asset)["curves"]
            self.assertEqual(found[0]["curve_id"], CURVE_ID, asset)
        top = search_curve_library(self.conv, hazard=PGA_475, asset="railway")["curves"][0]
        self.assertIn("proxy", top["limitations"])
        self.assertIn("ground", top["limitations"])
        self.assertTrue(top["has_bounds"])

    def test_loading_it_keeps_its_own_provenance(self):
        out = use_library_curve(self.conv, CURVE_ID)
        self.assertTrue(out["has_uncertainty_bounds"])
        self.assertIn("proxy", out["limitations"])
        self.assertNotIn("caveat", out)  # not a "midpoint of published bounds" curve
        self.assertNotIn("Nirandjan", out["cite"])
        self.assertIn("AEI Labs", out["cite"])
        curve = self.conv.curves[CURVE_ID]
        self.assertEqual(curve["provenance"]["curve_id"], CURVE_ID)
        self.assertIn("limitations", curve["provenance"])
        # At 0.1 g (98 cm/s2), the default exposure threshold: about 2 %.
        self.assertAlmostEqual(float(curve["interp"](0.1 * G)), 0.0186, places=3)
        self.assertGreater(float(curve["upper"](490.0)), float(curve["interp"](490.0)))

    def test_a_published_curve_still_cites_the_paper(self):
        out = use_library_curve(self.conv, "E8.9")
        self.assertIn("Nirandjan", out["cite"])
        self.assertNotIn("limitations", out)

    def test_expected_annual_damage_loads_it_by_id(self):
        curve, label = _ead_curve(self.conv, CURVE_ID.lower())
        self.assertEqual(label, CURVE_ID)
        self.assertTrue(curve["has_bounds"])

    def test_the_house_index_adds_one_row(self):
        rows = curve_library.index()
        self.assertEqual(sum(bool(r["_house"]) for r in rows), 1)
        self.assertEqual(len(rows), 219)


class GuidesPointAtTheDefault(unittest.TestCase):
    """The text agents read must name the curve that exists, with the numbers the
    file holds, and carry the conventions the appraisals need."""

    def test_guides_and_mcp_instructions_name_the_curve(self):
        from app.assistant import guides, mcp_server

        curve_library.get(CURVE_ID)
        for topic in ("vulnerability_analysis", "vulnerability_curves", "multi_hazard"):
            self.assertIn(CURVE_ID, guides.TOPICS[topic], topic)
        self.assertIn(CURVE_ID, mcp_server.INSTRUCTIONS)
        self.assertIn("plausibility", mcp_server.INSTRUCTIONS)
        text = " ".join(guides.TOPICS["vulnerability_curves"].split())
        for part in ("0.15 / 0.30 / 0.50 / 1.00 g", "147 / 294 / 490 / 981", "beta 0.6",
                     "0.05 / 0.20 / 0.50 / 0.70", "ground deformation", "sensitivity"):
            self.assertIn(part, text)

    def test_guides_state_the_conventions(self):
        from app.assistant import guides

        multi = " ".join(guides.TOPICS["multi_hazard"].split())
        reports = " ".join(guides.TOPICS["reports"].split())
        for part in ("98 cm/s²", "existing-climate", "1/20", "1/50 to 1/10",
                     "Expected annual damages add across hazards",
                     "event losses at given return periods do not"):
            self.assertIn(part, multi)
        self.assertIn("total expected annual damage", reports)
        self.assertIn("split by hazard", reports)
        self.assertIn("proxy", guides.TOPICS["vulnerability_analysis"])
        for topic, text in guides.TOPICS.items():
            self.assertNotIn("joins the curve linearly", text, topic)


if __name__ == "__main__":
    unittest.main()
