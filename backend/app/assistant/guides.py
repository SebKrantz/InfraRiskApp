"""On-demand how-to guides for the assistant — the skill layer.

The system prompt stays lean; before the heavyweight tasks the model calls
read_guide(topic) and follows the recipe. Content is written against the real
behaviour of app.utils.geospatial.analyze_intersection and the report tools —
not against what the code is assumed to do.
"""

from __future__ import annotations

EXPOSURE_ANALYSIS = """\
# Exposure analysis, end to end

## Workflow
1. `load_infrastructure` on the uploaded file, then `ui_show_dataset` with the
   returned file_id so the user SEES the network you are talking about.
2. `list_hazards` / `get_hazard` to pick the layer, `get_hazard_stats` to learn
   the intensity range before choosing a threshold.
3. `run_analysis`. Then `ui_select_hazard` + `ui_set_threshold` so the map shows
   the same thing your numbers describe.
4. Figures: `make_barchart` and `make_map` (the app's own exhibits).
5. `write_report_docx` if a deliverable was asked for — read_guide('reports') first.

## What the model actually computes (be precise about this)
- The hazard raster is sampled AT the asset. With a threshold, an asset is
  affected when **intensity >= threshold**. With no threshold, when
  **intensity > 0**. There is no other rule.
- **No hazard type is treated inversely.** Peak ground acceleration, flood depth
  and wind speed all use the same `>=` comparison. Higher intensity = worse.
- No-data (NaN) always counts as UNAFFECTED. For points this means
  `unaffected_count` silently includes assets that fell outside the raster —
  say so when a dataset extends beyond a layer's coverage.
- **Points** give counts: `affected_count`, `unaffected_count`.
- **Lines** are densified every 100 m, each sample point classified, and runs of
  the same class merged into segments. The result is **metres**
  (`affected_meters` / `unaffected_meters`) and a `full_gdf` with ONE ROW PER
  SEGMENT, not per input line. A single road can contribute several segments.
- Polygon inputs were converted to centroids at load time; they are points now.
  Say so if it matters — a large building footprint is being judged by one pixel.

## Choosing a threshold
State the threshold, its unit, and why you chose it. Never present an exposure
number without the threshold that produced it.
- Flood layers are inundation depth in **mm**. Common engineering anchors:
  ~100 mm (nuisance/access disruption), ~500 mm (damage to buried and
  ground-level equipment), ~1000 mm+ (severe structural damage).
- Tropical cyclone layers are gust speed in **km/h**: ~120 km/h is roughly the
  threshold for damage to overhead lines and masts, ~180 km/h for severe damage.
- PGA layers are in **cm/s²**: ~50 is a light-damage threshold, ~200 moderate.
- Landslide layers are a susceptibility **class 1-5**, not a physical quantity;
  threshold at a class boundary (e.g. >= 4) and call it "high susceptibility".
- Drought layers are indices, event counts or days — read `get_hazard` for the
  definition before interpreting them.
- Socioeconomic layers (population, building value) are not hazards at all.
  Analysing "exposure" against them is meaningless; if asked, explain and offer
  a real hazard layer instead.
If the user gives no threshold, pick a defensible one, say in one line what you
picked and why, and run `sweep_thresholds` to show how much it matters.

## Quality bar
- Every exposure figure needs: the dataset, the hazard layer, the threshold with
  its unit, and the share as well as the absolute number.
- Lines: report kilometres in prose (the tools return metres) and always give
  the share of total network length.
- Never read a number off the UI — the numbers come from `run_analysis`.
- Sanity-check: if affected share is 0% or 100%, say so explicitly and check the
  threshold against `get_hazard_stats` before reporting it as a finding.
"""

VULNERABILITY_ANALYSIS = """\
# Vulnerability (damage-cost) analysis

Exposure says *how much* infrastructure sits in a hazard zone. Vulnerability
says *what fraction is destroyed* and *what that costs*. They are different
numbers and must never be conflated.

## Workflow
1. `load_vulnerability_curve` on the uploaded CSV. Check the reported
   `intensity_range` against `get_hazard_stats` for the layer — a curve defined
   over 0-5000 mm applied to a layer that maxes at 800 mm is only exercising its
   lower limb, and you should say so.
2. `run_analysis` with `curve` and `replacement_value`.
3. `make_barchart` renders the vulnerability view automatically (damage cost on
   the left axis, exposure % and damage-ratio % on the right, error bars when the
   curve has bounds). `make_map` colours assets along a green-amber-red damage
   ratio ramp.
4. Drive the UI too: `ui_set_vulnerability` with the curve and replacement value.

## The curve file
Two accepted layouts, header optional:
- 2 columns: `intensity, proportion_destroyed`
- 4+ columns: `intensity, lower, central, upper` — **column 3 is the central
  curve**, columns 2 and 4 give the uncertainty band that becomes
  `total_damage_cost_lower` / `_upper` and the error bars on the chart.
Intensity is in the HAZARD LAYER'S units; proportions must be in [0, 1].
Between points the curve is interpolated linearly and it is clamped flat outside
its range — an intensity above the last point does not extrapolate.

## replacement_value — get this right
- **Point datasets: value per FEATURE** (per tower, per station, per building).
- **Line datasets: value per METRE** (so a USD 40 000/km cable is 40).
Damage = replacement_value x damage_ratio, summed; for lines it is also
multiplied by segment length. State the basis and the currency every time.

## Two caveats you must carry into any write-up
1. **Damage is computed over every feature, not only the "affected" ones.** An
   asset below the exposure threshold but with a small positive intensity still
   contributes damage cost through the curve. So total damage cost and the
   affected count answer different questions, and damage cost does not go to
   zero just because the threshold is high.
2. The damage ratio reported on the chart is the mean over affected features
   (length-weighted for lines), while the cost is the sum over all of them.

## What to report
- Total damage cost with its currency and, when the curve has bounds, the
  lower-upper band — as a range, not a false-precision single number.
- The mean damage ratio, and the exposure share, clearly labelled as different.
- The replacement value used, its basis (per feature / per metre) and where it
  came from (the user, an uploaded document, or your own stated assumption).
- The curve's provenance. A vulnerability curve is an engineering judgement; if
  it came from the user, cite them; if you assumed one, say so loudly.
"""

MULTI_HAZARD = """\
# Comparing across layers: return periods and climate scenarios

The catalogue is organised in families, and almost every interesting question is
a comparison within one.

- **Flood hazard** — 25 / 50 / 100 year return periods, each in three climate
  variants: *Existing climate*, *SSP1 Lower bound*, *SSP5 Upper bound*.
- **Tropical cyclone** — 25 / 50 / 100 year, with and without climate change.
- **Earthquakes** — PGA at 250 / 475 / 975 year return periods (no climate
  variants; seismic hazard is not climate-driven).
- **Landslides** — susceptibility classes, precipitation-triggered (existing /
  lower / upper) and earthquake-triggered.
- **Drought** — SPI-6 based hazard, duration and event count.

## How to run one
`compare_hazards` with the family's layer list and ONE threshold, so the only
thing varying is the hazard. It caches every layer, so a follow-up
`run_analysis` or `make_map` on any of them is instant. Expect tens of seconds
per layer — tell the user before starting a long batch.

## Return-period table
One row per return period; columns: layer, exposure (count or km), share of
total, and — with a curve — damage cost. Return periods are **annual exceedance
probabilities**: a 100-year layer is the intensity with a 1% chance of being
exceeded in any year, NOT "the flood that happens every 100 years". Say it that
way. Exposure rises with return period; if it does not, investigate before
reporting.

## Climate-scenario table
One row per scenario at a fixed return period; report the delta against existing
climate in both absolute and percentage terms.
Discipline about what these are:
- *SSP1 Lower bound* and *SSP5 Upper bound* are the ends of a modelled range,
  not a best estimate and a worst case for a given year. They bracket
  uncertainty across scenarios and models.
- Present them as a **range**: "exposed length rises from X km today to between
  Y and Z km". Never average the two, never call the upper bound "the
  projection", and never attach a date the layer does not carry.
- The underlying hazard model, not this app, is the source of the scenario
  definitions — `get_hazard` gives the description and background paper, and
  those belong in the method section.

## Cross-family comparison
Comparing flood exposure with seismic exposure is comparing different units at
different return periods. Only ever compare the SHARE of the network exposed,
state both thresholds, and note explicitly that the return periods differ
(250-975 years for PGA vs 25-100 for flood). Do not add exposures across
hazards into a single "total" — assets exposed to two hazards would be
double-counted.
"""

REPORTS = """\
# Written deliverables (Word, Excel, CSV)

## Structure of a good report
1. Title + a one-line subtitle naming the network, the area and the hazard set.
2. **Executive summary**: the headline numbers in the first five lines, then two
   or three sentences of interpretation. A reader who stops here should have the
   answer. Every figure quoted here MUST reappear in the body — the summary
   summarises, it never introduces.
3. **Data and method**: what infrastructure was analysed (source, feature count,
   total length), which hazard layers (name, return period, climate scenario,
   units) with their modelling provenance from `get_hazard`, the threshold and
   why, and — for damage — the curve and replacement value. State that assets
   are sampled at their location, lines every 100 m, and polygons at centroids.
4. **Results**: tables and figures, each with one short interpreting paragraph.
   Say what the number MEANS, not that it "is shown in the table".
5. **Caveats and limitations**: never skip this. The standard set for this app:
   raster resolution vs asset size; centroid treatment of polygons; the
   threshold is a modelling choice, not a physical certainty; no-data areas
   count as unaffected; damage curves are engineering judgements; return periods
   are annual exceedance probabilities; nothing here models cascading failure,
   redundancy or repair time.
6. **Conclusion**: brief and decisive. What the analysis shows, which assets or
   corridors deserve attention first, and what would most change the answer.
   No new numbers the body has not shown.

## Tables and figures: numbering, captions, references
- `write_report_docx` numbers tables and figures automatically in document order
  (per section: tables first, then figures) and renders "Table N. <title>" above
  each table and "Figure N. <caption>" below each figure.
- **Never type a number yourself.** Give every exhibit a short `ref` label and
  cite it in prose as `[[label]]`; the builder substitutes the right number, so
  a reference can never drift out of sync when you add or reorder an exhibit.
      figure: {artifact_id: "...", ref: "flood_map", caption: "..."}
      prose:  "The worst-affected corridors run north of Brazzaville ([[flood_map]])."
  Hand-counted "Figure 3" WILL eventually be wrong; `[[flood_map]]` cannot be.
  The tool returns the resolved `numbering` map so you can see what each became.
- EVERY table and figure gets a descriptive title/caption (what, where, units,
  scenario) and MUST be cited at least once in the text. No orphan exhibits, no
  "the table below".
- Use the `note` field for small print: data source, threshold, definitions
  ("Note: exposure at >= 500 mm inundation depth; GIRI 100-year flood layer,
  existing climate.").

## Quality bar
- Every number comes from a tool result or from python_exec on a stored
  variable. Never re-type a figure from memory, never estimate.
- Units on everything: mm, km/h, cm/s², km, USD. Say which currency.
- Say the threshold next to any exposure number, every time.
- Tables: 4-7 columns, sorted by the column that tells the story. Numbers are
  formatted strings — thousands separators, sensible precision, large magnitudes
  compacted with the unit in the header ("Damage cost (USD m)" holding "12.4"),
  never a 13-digit raw number in a Word table.
- Word tables and charts are for HUMANS: descriptive headers with units
  ("Exposed length (km)", "Flood 100-yr, SSP5"). Excel/CSV exports are for
  MACHINES: snake_case columns, raw unformatted numbers, one row per record.
- Generate figures BEFORE `write_report_docx`, then embed them by artifact_id.
- Length matches the ask: "comprehensive" means 2-4 pages of substance, not
  padding. No filler sentences, no restating the method in the results.
- After delivering a report, offer `export_excel` with the tables behind it and
  `export_analysis_data` for the per-asset detail.
- If the user uploaded context documents (terms of reference, inception notes,
  project descriptions), `read_document` them FIRST and weave them in: use their
  terminology and place names, answer their stated questions, and cite their
  figures alongside the model's — clearly distinguishing the two.
"""

FIGURES = """\
# Choosing and building figures

## The app's own two exhibits — use these by default
- `make_barchart(analysis)` — affected vs unaffected for one dataset-hazard
  pair, or the damage-cost view with exposure and damage-ratio percentages when
  the analysis had a curve. Identical to the sidebar's export button.
- `make_map(analysis)` — the hazard raster as a blue intensity wash over a
  basemap with the infrastructure coloured by exposure (or damage ratio).
  Frames itself on the dataset. Pick `basemap='esri-imagery'` when the physical
  setting matters, `positron` (default) otherwise.
These two belong in essentially every report: one chart, one map, per hazard.

## The general tools — for anything comparative
- `make_chart` — exposure across return periods, climate scenarios, hazard
  types or thresholds. Series named 'Affected'/'Unaffected'/'Damage cost'/
  'Exposure' inherit the app's colours. Always put the unit in `ylabel`.
  * Return-period ladder: `kind='bar'`, labels = 25/50/100 yr.
  * Climate scenarios: `kind='bar'` grouped, one series per scenario.
  * Threshold sensitivity from `sweep_thresholds`: `kind='line'`, x = threshold.
  * Split of a total: `kind='stacked_bar'`; a two-slice pie is never worth it.
- `make_custom_map` — several datasets on one map, a country subset, the
  affected segments alone, or a choropleth of an attribute. Build the
  GeoDataFrames in `python_exec` first from an analysis result's `full_gdf`
  (columns `affected`, `exposure_level_avg`/`_max`, `length_m`, and in
  vulnerability mode `vulnerability` and `damage_cost`).
- `python_exec` with matplotlib for anything else. Figures you draw are captured
  automatically and appear in the chat.

## Rules
- One message per figure. If a chart needs a paragraph to explain what it shows,
  the chart is wrong.
- **Do not plot affected against unaffected when the affected share is small.**
  A 0.03% bar next to a 99.97% bar is an invisible sliver and tells the reader
  nothing. Below roughly 5%, chart the affected quantity ALONE across return
  periods, scenarios or thresholds, or plot the share on a percentage axis.
  Reserve the affected/unaffected split for the app's own `make_barchart`,
  where it is the established house exhibit, and say the share in the caption.
- Caption every figure with what, where, units and scenario.
- Do not draw a chart of two numbers you already stated in a sentence.
- Generate every figure BEFORE writing the report, then embed by artifact_id.
"""

TOPICS: dict[str, str] = {
    "exposure_analysis": EXPOSURE_ANALYSIS,
    "vulnerability_analysis": VULNERABILITY_ANALYSIS,
    "multi_hazard": MULTI_HAZARD,
    "reports": REPORTS,
    "figures": FIGURES,
}
