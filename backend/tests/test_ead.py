"""expected_annual_damage: the trapezoid, its frequency conventions, the curve's
bounds and the refusals. No rasters: the analyses are replaced by fakes.

Run from backend/ with the repo's venv:  python -m unittest discover -s tests
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd  # noqa: E402
from shapely.geometry import LineString  # noqa: E402

from app.api.upload import uploaded_files  # noqa: E402
from app.assistant import conversations, domain, tools  # noqa: E402

tools.load()

from app.assistant.tools import analysis  # noqa: E402
from app.assistant.tools.ead import ead, expected_annual_damage  # noqa: E402

FILE_ID = "test-ead-line"
RPS = [25, 50, 100]


def fake_run_exposure(file_id, hazard, threshold=None, **kw):
    """What run_exposure returns for a line dataset with a bounded curve. The
    loss is T / 2.5 (10, 20, 40 at 25, 50, 100 years), its bounds half and double
    that, and — to tell the climate variants apart — SSP1 twice and SSP5 three
    times the existing climate's."""
    hid = hazard["hazard_id"]
    rp = int(re.search(r"_(\d+)_years", hid).group(1))
    scale = 2.0 if "ssp1" in hid else 3.0 if "ssp5" in hid else 1.0
    central = scale * rp / 2.5
    lower, upper = central / 2, central * 2
    return {
        "affected_meters": 100.0 * rp,
        "unaffected_meters": 10_000.0 - 100.0 * rp,
        "full_gdf": None,
        "total_damage_cost": central,
        "total_damage_cost_lower": lower,
        "total_damage_cost_upper": upper,
        "_assistant_meta": {"file_id": file_id, "hazard_id": hid, "threshold": threshold},
    }


def flat(x):
    return 0.5


class Trapezoid(unittest.TestCase):
    """10, 20 and 40 at 25, 50 and 100 years: p = 0.04, 0.02, 0.01."""

    def test_three_return_periods_by_hand(self):
        # (0.04 - 0.02) x (10 + 20) / 2 = 0.3
        # (0.02 - 0.01) x (20 + 40) / 2 = 0.3
        # tail: the 100-year loss held to p = 0: 0.01 x 40 = 0.4
        self.assertAlmostEqual(ead([25, 50, 100], [10, 20, 40]), 1.0)

    def test_order_does_not_matter(self):
        self.assertAlmostEqual(ead([100, 25, 50], [40, 10, 20]), 1.0)

    def test_lower_bound_is_the_default(self):
        # Nothing below the 25-year: a lone 25-year loss of 10 adds no area
        # between p = 1 and p = 0.04.
        self.assertAlmostEqual(ead([25, 50], [10, 10]), 0.04 * 10)

    def test_protection_below_the_smallest_return_period(self):
        # Zero loss at 10 years (p = 0.1): + (0.1 - 0.04) x (0 + 10) / 2 = 0.3
        self.assertAlmostEqual(ead([25, 50, 100], [10, 20, 40], protection_rp=10), 1.3)

    def test_protection_inside_the_ladder_zeroes_what_it_covers(self):
        # Protected to 50 years: zero at p = 0.02, then (0.02 - 0.01) x 40 / 2 = 0.2,
        # plus the 0.4 tail.
        self.assertAlmostEqual(ead([25, 50, 100], [10, 20, 40], protection_rp=50), 0.6)

    def test_protection_at_the_smallest_return_period_is_below_the_default(self):
        # Protected to 25 years: the 25-year loss is dropped, not kept —
        # (0.04 - 0.02) x (0 + 20) / 2 = 0.2, + 0.3 + 0.4 = 0.9 < 1.0.
        self.assertAlmostEqual(ead([25, 50, 100], [10, 20, 40], protection_rp=25), 0.9)
        # Just short of it the default comes back: the 25-year loss stays.
        self.assertAlmostEqual(ead([25, 50, 100], [10, 20, 40], protection_rp=24.999), 1.0,
                               places=3)

    def test_upper_bound_joins_from_zero_at_one_year(self):
        # + (1 - 0.04) x (0 + 10) / 2 = 4.8
        self.assertAlmostEqual(ead([25, 50, 100], [10, 20, 40], upper_bound=True), 5.8)

    def test_refusals(self):
        with self.assertRaisesRegex(ValueError, "at least two return periods"):
            ead([100], [40])
        with self.assertRaisesRegex(ValueError, "at least two return periods"):
            ead([100, 100], [40, 40])
        with self.assertRaisesRegex(ValueError, "alternatives"):
            ead([25, 50], [1, 2], protection_rp=10, upper_bound=True)
        with self.assertRaisesRegex(ValueError, "below the largest return period"):
            ead([25, 50], [1, 2], protection_rp=50)
        with self.assertRaisesRegex(ValueError, "at least 1 year"):
            ead([0.5, 50], [1, 2])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            ead([25, 50], [-1, 2])


class Tool(unittest.TestCase):
    def setUp(self):
        line = LineString([(15.0, -4.0), (15.1, -4.0)])
        uploaded_files[FILE_ID] = {
            "gdf": gpd.GeoDataFrame({"name": ["a"], "kind": ["rail"], "value": [7000.0]},
                                    geometry=[line], crs="EPSG:4326"),
            "filename": "line.gpkg",
            "geometry_type": "LineString",
            "feature_count": 1,
            "bounds": [15.0, -4.0, 15.1, -4.0],
        }
        self.conv = conversations.Conversation(id="test-ead", created=0.0)
        for name, path in (("rail", "rail.csv"), ("other", "other.csv")):
            self.conv.curves[name] = {
                "interp": flat, "lower": flat, "upper": flat, "has_bounds": True,
                "path": Path(path), "provenance": {"kind": "test"},
            }
        patcher = mock.patch.object(domain, "run_exposure", side_effect=fake_run_exposure)
        self.run_exposure = patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        uploaded_files.pop(FILE_ID, None)
        conversations._cleanup(self.conv)

    def ead(self, **kw):
        return expected_annual_damage(self.conv, **kw)

    def analyse(self, hazard, threshold=100.0, curve="rail", replacement_value=500.0):
        return analysis.run_analysis(
            self.conv, FILE_ID, hazard, threshold=threshold, curve=curve,
            replacement_value=replacement_value,
        )["stored_as"]

    def test_flood_family_one_result_per_climate_variant_with_bounds(self):
        out = self.ead(file_id=FILE_ID, family="flood", threshold=100, curve="rail",
                       replacement_value=500, currency="USD", price_basis="2024 prices")
        self.assertEqual(self.run_exposure.call_count, 9)
        by_variant = {v["variant"]: v for v in out["variants"]}
        self.assertEqual(sorted(by_variant), [
            "Flood Hazard - Existing climate",
            "Flood Hazard - SSP1 Lower bound",
            "Flood Hazard - SSP5 Upper bound",
        ])
        for variant, scale in (("Existing climate", 1), ("SSP1 Lower bound", 2),
                               ("SSP5 Upper bound", 3)):
            v = by_variant[f"Flood Hazard - {variant}"]
            self.assertEqual(v["return_periods"], [25, 50, 100])
            self.assertEqual([x["damage_cost"] for x in v["losses"]],
                             [scale * 10, scale * 20, scale * 40])
            # The trapezoid by hand gives 1.0 per unit of scale; the bounds are
            # half and double the central losses, so their EADs are too.
            self.assertAlmostEqual(v["ead_central"], scale * 1.0)
            self.assertAlmostEqual(v["ead_curve_lower"], scale * 0.5)
            self.assertAlmostEqual(v["ead_curve_upper"], scale * 2.0)
        self.assertEqual((out["estimate"], out["family"]), ("lower bound", "flood"))
        self.assertTrue(out["curve_has_bounds"])
        self.assertEqual(out["replacement_value"], {
            "value": 500, "basis": "per metre", "currency": "USD", "price_basis": "2024 prices",
        })
        self.assertNotIn("state", out)
        self.assertIn("no return period", out["landslide"])
        self.assertTrue(any("never average" in c for c in out["conventions"]))
        self.assertEqual(len(self.conv.namespace[out["stored_as"]]), 9)

    def test_conventions_through_the_tool(self):
        hazards = [f"flood_hazard_{t}_years_existing_climate" for t in RPS]
        common = dict(file_id=FILE_ID, hazards=hazards, threshold=100, curve="rail",
                      replacement_value=500)
        protected = self.ead(**common, protection_rp=10)["variants"][0]
        self.assertAlmostEqual(protected["ead_central"], 1.3)
        upper = self.ead(**common, upper_bound=True)
        self.assertEqual(upper["estimate"], "upper bound")
        self.assertAlmostEqual(upper["variants"][0]["ead_central"], 5.8)
        self.assertAlmostEqual(upper["variants"][0]["ead_curve_upper"], 11.6)
        self.assertIn("state", upper)  # no currency or price basis given

    def test_reused_analyses_carry_the_bounds(self):
        names = [self.analyse(f"flood_hazard_{t}_years_existing_climate") for t in RPS]
        calls = self.run_exposure.call_count
        out = self.ead(analyses=names)
        self.assertEqual(self.run_exposure.call_count, calls)  # reused, not re-run
        v = out["variants"][0]
        self.assertAlmostEqual(v["ead_central"], 1.0)
        self.assertAlmostEqual(v["ead_curve_lower"], 0.5)
        self.assertAlmostEqual(v["ead_curve_upper"], 2.0)
        self.assertEqual([x["analysis"] for x in v["losses"]], names)
        self.assertEqual((out["threshold"], out["curve"]), (100.0, "rail"))

    def test_refuses_a_single_return_period(self):
        with self.assertRaisesRegex(ValueError, "single return period"):
            self.ead(file_id=FILE_ID, hazards=["flood_hazard_100_years_existing_climate"],
                     curve="rail", replacement_value=500)
        self.run_exposure.assert_not_called()
        # Two variants at one return period each are still one per variant.
        with self.assertRaisesRegex(ValueError, "single return period"):
            self.ead(file_id=FILE_ID, curve="rail", replacement_value=500, hazards=[
                "flood_hazard_100_years_existing_climate",
                "flood_hazard_100_years_ssp5_upper_bound",
            ])
        with self.assertRaisesRegex(ValueError, "single return period"):
            self.ead(analyses=[self.analyse("flood_hazard_100_years_existing_climate")])

    def test_an_int_and_a_float_threshold_are_the_same(self):
        names = [self.analyse("flood_hazard_25_years_existing_climate", threshold=100),
                 self.analyse("flood_hazard_50_years_existing_climate", threshold=100.0)]
        # (0.04 - 0.02) x (10 + 20) / 2 + 0.02 x 20
        self.assertAlmostEqual(self.ead(analyses=names)["variants"][0]["ead_central"], 0.7)

    def test_refuses_layers_of_two_families(self):
        with self.assertRaisesRegex(ValueError, "span the cyclone and flood families"):
            self.ead(file_id=FILE_ID, curve="rail", replacement_value=500, hazards=[
                "flood_hazard_25_years_existing_climate",
                "flood_hazard_50_years_existing_climate",
                "tropical_cyclone_wind_25_years",
                "tropical_cyclone_wind_50_years",
            ])
        names = [self.analyse("flood_hazard_25_years_existing_climate"),
                 self.analyse("tropical_cyclone_wind_50_years")]
        with self.assertRaisesRegex(ValueError, "span the cyclone and flood families"):
            self.ead(analyses=names)
        with self.assertRaisesRegex(ValueError, "no return-period family"):
            self.ead(file_id=FILE_ID, curve="rail", replacement_value=500, hazards=[
                "drought_hazard_spi_6_5_year_return_period_existing_climate",
                "flood_hazard_25_years_existing_climate",
            ])
        calls = self.run_exposure.call_count
        self.assertEqual(calls, 2)  # the two analyses above; EAD itself read nothing

    def test_refuses_mixed_thresholds(self):
        names = [self.analyse("flood_hazard_25_years_existing_climate", threshold=100),
                 self.analyse("flood_hazard_50_years_existing_climate", threshold=500)]
        with self.assertRaisesRegex(ValueError, "mix thresholds"):
            self.ead(analyses=names)

    def test_refuses_mixed_curves(self):
        names = [self.analyse("flood_hazard_25_years_existing_climate", curve="rail"),
                 self.analyse("flood_hazard_50_years_existing_climate", curve="other")]
        with self.assertRaisesRegex(ValueError, "mix curves"):
            self.ead(analyses=names)

    def test_refuses_mixed_replacement_values(self):
        names = [self.analyse("flood_hazard_25_years_existing_climate", replacement_value=500),
                 self.analyse("flood_hazard_50_years_existing_climate", replacement_value=900)]
        with self.assertRaisesRegex(ValueError, "mix replacement values"):
            self.ead(analyses=names)

    def test_refuses_exposure_only_analyses_and_layers_without_a_return_period(self):
        self.conv.namespace["plain"] = {
            "full_gdf": None,
            "_assistant_meta": {"file_id": FILE_ID, "threshold": None,
                                "hazard_id": "flood_hazard_25_years_existing_climate"},
        }
        with self.assertRaisesRegex(ValueError, "no damage cost"):
            self.ead(analyses=["plain", "plain"])
        with self.assertRaisesRegex(ValueError, "no return period.*Landslide"):
            self.ead(file_id=FILE_ID, curve="rail", replacement_value=500, hazards=[
                "susceptibility_class_of_landslides_triggered_by_earthquakes",
                "flood_hazard_25_years_existing_climate",
            ])

    def test_per_asset_replacement_values(self):
        hazards = [f"flood_hazard_{t}_years_existing_climate" for t in RPS]
        out = self.ead(file_id=FILE_ID, hazards=hazards, curve="rail",
                       replacement_value_column="kind", replacement_value_map={"rail": 7000})
        values = self.run_exposure.call_args.kwargs["replacement_value"]
        self.assertEqual(list(values), [7000.0])
        self.assertEqual(out["replacement_value"]["replacement_value_column"], "kind")
        self.assertEqual(out["replacement_value"]["per_feature"]["min"], 7000.0)

    def test_refuses_mixed_replacement_value_specs(self):
        names = [
            self.analyse("flood_hazard_25_years_existing_climate", replacement_value=7000),
            analysis.run_analysis(
                self.conv, FILE_ID, "flood_hazard_50_years_existing_climate", threshold=100,
                curve="rail", replacement_value_column="value",
            )["stored_as"],
        ]
        with self.assertRaisesRegex(ValueError, "mix replacement value"):
            self.ead(analyses=names)

    def test_loads_a_library_curve_by_id(self):
        out = self.ead(file_id=FILE_ID, family="pga", curve="E8.9", replacement_value=500)
        self.assertIn("E8.9", self.conv.curves)
        self.assertEqual(out["curve_provenance"]["curve_id"], "E8.9")
        self.assertEqual([v["return_periods"] for v in out["variants"]], [[250, 475, 975]])


if __name__ == "__main__":
    unittest.main()
