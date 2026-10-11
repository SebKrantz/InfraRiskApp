"""expected_annual_damage: the trapezoid, its frequency conventions, the curve's
bounds and the refusals. No rasters: the analyses are replaced by fakes.

Run from backend/ with the repo's venv:  python -m unittest discover -s tests
"""

from __future__ import annotations

import math
import re
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd  # noqa: E402
import pandas as pd  # noqa: E402
from shapely.geometry import LineString  # noqa: E402

from app.api.upload import uploaded_files  # noqa: E402
from app.assistant import conversations, domain, tools  # noqa: E402

tools.load()

from app.assistant.tools import analysis  # noqa: E402
from app.assistant.tools.ead import _asset_value, ead, expected_annual_damage  # noqa: E402

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
        # The 10 km line's segments: what the asset's total replacement value needs.
        "full_gdf": pd.DataFrame({"line_id": [1, 1], "length_m": [100.0 * rp, 10_000.0 - 100.0 * rp]}),
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

    def test_upper_bound_extends_the_loss_line_log_linearly_to_zero(self):
        # 10 and 20 at 25 and 50 years: 10 / ln 2 per e-fold of T, zero at 12.5
        # years. The area under L(p) = a ln(p0 / p) from p = 1 / 25 to p0 = 1 / 12.5
        # is (a / T0) x [u - u ln u] between u = T0 / 25 and 1, added to the 1.0
        # of the lower bound.
        a, t0, u0 = 10 / math.log(2), 12.5, 0.5
        extra = a / t0 * ((1 - 0) - (u0 - u0 * math.log(u0)))
        self.assertAlmostEqual(ead([25, 50, 100], [10, 20, 40], upper_bound=True), 1.0 + extra,
                               places=3)

    def test_upper_bound_is_at_most_a_factor_of_ten_below_the_smallest_return_period(self):
        # Flat losses never reach zero: held at 10 down to 25 / 10 = 2.5 years,
        # (0.4 - 0.04) x 10 = 3.6 on top of the lower bound's 0.04 x 10.
        self.assertAlmostEqual(ead([25, 50, 100], [10, 10, 10], upper_bound=True), 4.0)

    def test_upper_bound_never_goes_under_one_year(self):
        # Flat 10 from T = 3: 3 / 10 < 1, so it stops at the yearly event.
        self.assertAlmostEqual(ead([3, 6], [10, 10], upper_bound=True), 10.0)

    def test_upper_bound_adds_nothing_when_the_smallest_return_period_has_no_loss(self):
        self.assertAlmostEqual(ead([25, 50, 100], [0, 20, 40], upper_bound=True),
                               ead([25, 50, 100], [0, 20, 40]))

    def test_upper_bound_stays_plausible_on_the_pga_example(self):
        # A railway worth 8 bn, PGA losses of 1.10, 1.57 and 2.14 bn at 250, 475
        # and 975 years. Joining zero loss at T = 1 linearly in p gave 554 m a year
        # (7 % of the asset); the extension reaches zero at 56 years.
        rps, losses = [250, 475, 975], [1.10e9, 1.57e9, 2.14e9]
        lower = ead(rps, losses)
        upper = ead(rps, losses, upper_bound=True)
        self.assertAlmostEqual(lower / 1e6, 6.727, places=2)
        self.assertAlmostEqual(upper / 1e6, 12.555, places=2)
        self.assertGreater(upper, lower)
        self.assertLess(upper, 0.01 * 8e9)

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
        for file_id in self.conv.datasets:
            uploaded_files.pop(file_id, None)
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
            "asset_total": 500 * 10_000.0,  # per metre x the 10 km line
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
        default = self.ead(**common)["variants"][0]
        upper = self.ead(**common, upper_bound=True)
        self.assertEqual(upper["estimate"], "upper bound")
        v = upper["variants"][0]
        # The extension is linear in the losses: the curve's bounds scale it.
        self.assertGreater(v["ead_central"], default["ead_central"])
        self.assertLess(v["ead_central"], 1.5)  # was 5.8 when it joined zero loss at T = 1
        self.assertAlmostEqual(v["ead_curve_upper"], 2 * v["ead_central"])
        self.assertAlmostEqual(v["ead_curve_lower"], v["ead_central"] / 2)
        self.assertEqual(v["upper_bound_extends_to_years"], 12.5)
        self.assertNotIn("upper_bound_extends_to_years", default)
        self.assertIn("state", upper)  # no currency or price basis given

    def test_share_of_the_asset_and_the_plausibility_flag(self):
        hazards = [f"flood_hazard_{t}_years_existing_climate" for t in RPS]
        # EAD 1.0 on a 10 km line at 500 a metre (5 million): nothing to flag.
        ok = self.ead(file_id=FILE_ID, hazards=hazards, threshold=100, curve="rail",
                      replacement_value=500)["variants"][0]
        self.assertAlmostEqual(ok["ead_pct_of_asset_per_year"], 100 * 1.0 / 5_000_000, places=6)
        self.assertEqual(ok["plausibility"], "ok")
        self.assertNotIn("plausibility_note", ok)
        # At 0.001 a metre the asset is worth 10: the same EAD is 10 % of it a year.
        high = self.ead(file_id=FILE_ID, hazards=hazards, threshold=100, curve="rail",
                        replacement_value=0.001)["variants"][0]
        self.assertAlmostEqual(high["ead_pct_of_asset_per_year"], 10.0)
        self.assertEqual(high["plausibility"], "high")
        self.assertIn("not a planning bound", high["plausibility_note"])
        # Exactly 1 % a year is the edge of "about 1 %": still fine.
        edge = self.ead(file_id=FILE_ID, hazards=hazards, threshold=100, curve="rail",
                        replacement_value=0.01)["variants"][0]
        self.assertEqual(edge["plausibility"], "ok")

    def test_the_flag_judges_the_central_convention_not_the_bound(self):
        hazards = [f"flood_hazard_{t}_years_existing_climate" for t in RPS]
        common = dict(file_id=FILE_ID, hazards=hazards, threshold=100, curve="rail",
                      replacement_value=0.0095)  # asset 95: EAD 1.0 is 1.05 %
        self.assertEqual(self.ead(**common)["variants"][0]["plausibility"], "high")
        upper = self.ead(**common, upper_bound=True)["variants"][0]
        self.assertEqual(upper["plausibility"], "high")
        self.assertGreater(upper["ead_pct_of_asset_per_year"], 1.05)

    def test_asset_total_of_points_and_of_per_row_values(self):
        point = {"geometry_type": "Point", "feature_count": 4}
        self.assertEqual(_asset_value({}, point, 250.0), 1000.0)
        self.assertEqual(_asset_value({}, point, [100.0, 200.0, 300.0, 400.0]), 1000.0)
        line = {"geometry_type": "LineString", "feature_count": 2}
        result = {"affected_meters": 600.0, "unaffected_meters": 400.0,
                  "full_gdf": pd.DataFrame({"line_id": [1, 2, 2], "length_m": [600.0, 300.0, 100.0]})}
        self.assertEqual(_asset_value(result, line, 2.0), 2000.0)
        self.assertEqual(_asset_value(result, line, [1.0, 10.0]), 600.0 + 4000.0)

    def test_reused_analyses_carry_the_bounds(self):
        names = [self.analyse(f"flood_hazard_{t}_years_existing_climate") for t in RPS]
        calls = self.run_exposure.call_count
        out = self.ead(analyses=names)
        self.assertEqual(self.run_exposure.call_count, calls)  # reused, not re-run
        v = out["variants"][0]
        self.assertEqual(out["replacement_value"]["asset_total"], 5_000_000.0)
        self.assertEqual(v["plausibility"], "ok")
        self.assertAlmostEqual(v["ead_central"], 1.0)
        self.assertAlmostEqual(v["ead_curve_lower"], 0.5)
        self.assertAlmostEqual(v["ead_curve_upper"], 2.0)
        self.assertEqual([x["analysis"] for x in v["losses"]], names)
        self.assertEqual((out["threshold"], out["curve"]), (100.0, "rail"))

    def test_reused_analyses_of_an_upload_named_by_file_name(self):
        path = self.conv.workdir / "line.gpkg"
        uploaded_files[FILE_ID]["gdf"].to_file(path)
        self.conv.uploads["line.gpkg"] = path
        names = [
            analysis.run_analysis(
                self.conv, "line.gpkg", f"flood_hazard_{t}_years_existing_climate",
                threshold=100, curve="rail", replacement_value=500,
            )["stored_as"]
            for t in RPS
        ]
        out = self.ead(analyses=names, file_id="line.gpkg")
        self.assertEqual(out["file_id"], self.conv.datasets[0])
        self.assertAlmostEqual(out["variants"][0]["ead_central"], 1.0)
        with self.assertRaisesRegex(ValueError, "the analyses are of"):
            self.ead(analyses=names, file_id=FILE_ID)

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
        self.assertEqual(out["replacement_value"]["asset_total"], 7000.0 * 10_000.0)

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
