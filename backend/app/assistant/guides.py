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
1. **Get a curve. Do not ask the user for one.** The app ships 218 published
   curves and the tools to build one when none fits, so a request for
   vulnerability analysis is answerable straight away:
   - the user uploaded a CSV → `load_vulnerability_curve`;
   - otherwise → `search_curve_library(hazard=<layer>, asset=<description>)`
     then `use_library_curve(curve_id)`;
   - nothing fits, or the layer has no curves at all (landslide, drought) →
     `create_curve`.
   **`read_guide('vulnerability_curves')` carries all of this**: the unit of
   every layer, how to judge between candidates, the published anchors for
   building a curve, and what the landslide and drought layers need instead.
2. Check the curve's `intensity_range` against `get_hazard_stats` for the layer —
   a curve defined over 0-5000 mm applied to a layer that maxes at 800 mm is only
   exercising its lower limb, and you should say so.
3. Get a `replacement_value`. If the user gave one, use it; otherwise
   `find_replacement_cost(asset, basis='unit'|'metre')` and state the figure, its
   geography and its basis as an assumption.
4. `run_analysis` with `curve` and `replacement_value`.
5. `make_barchart` renders the vulnerability view automatically (damage cost on
   the left axis, exposure % and damage-ratio % on the right, error bars when the
   curve has bounds). `make_map` colours assets along a green-amber-red damage
   ratio ramp. `plot_curve` shows the curve itself — include it whenever you
   chose or built the curve rather than the user.
6. Drive the UI too: `ui_set_vulnerability` with the curve and replacement value.

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
  it came from the user, cite them; if you chose one from the library, name it by
  id, asset and source study and say how close the match is; if you constructed
  one, say so loudly and give the basis. Library curves also carry the compilation
  citation: Nirandjan et al. (2024), NHESS 24, 4341-4369.
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
  A ref expands to the WHOLE label ("Figure 2"), so write "[[a]] and [[b]] show"
  — never "Figures [[a]] and [[b]]", which reads as "Figures Figure 1 and
  Figure 2".
- EVERY table and figure gets a descriptive title/caption (what, where, units,
  scenario) and MUST be cited at least once in the text. No orphan exhibits, no
  "the table below".
- Use the `note` field for small print: data source, threshold, definitions
  ("Note: exposure at >= 500 mm inundation depth; GIRI 100-year flood layer,
  existing climate.").

## The house template
`write_report_docx` applies the house style on its own — Arial throughout, a
deep-navy and bright-blue palette, navy headings, a blue rule under the title,
navy table headers with banded rows, muted captions and notes. You do not
choose fonts, colours or sizes, and there is no way to pass them.

So do not try to reproduce styling in your content: no ASCII rules, no
"(bold heading)" annotations, no colour instructions, no cover-page mock-up, and
no organisation name, logo or letterhead of any kind — the documents are
deliberately unattributed, and whoever issues one adds their own marks. Give the
builder clean structure — headings, paragraphs, bullets, tables, figures — and
let the template do the rest.

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

VULNERABILITY_CURVES = """\
# Choosing or building a vulnerability curve

**The user should not have to produce a curve.** The app ships 218 published
curves, and where none fits you build one. Asking the user to supply a CSV is
the last resort, not the first move. But a curve is an engineering judgement
about how much of an asset a hazard destroys, so whichever route you take, the
provenance goes in the answer — never a damage number without the curve behind
it.

## Step 0 — what is the layer's axis?

A curve's intensity axis must be the SAME QUANTITY IN THE SAME UNIT as the
hazard raster. Get this wrong by a factor of 1000 and the damage cost is wrong
by orders of magnitude, silently.

| App layers | Axis | Unit | Global range of the raster | Curves? |
|---|---|---|---|---|
| Flood Hazard 25/50/100 yr x existing / SSP1 / SSP5 (9) | inundation depth | **mm** | 1 to ~15 200 (100 yr) | 92 |
| Peak Ground Acceleration 250/475/975 yr (3) | peak ground acceleration | **cm/s2** | 49 to ~873 (475 yr) | 63 |
| Tropical Cyclone Wind 25/50/100 yr, +/- climate change (6) | 10 m gust speed | **km/h** | 50 to ~277 (100 yr) | 23 |
| Landslide susceptibility, precipitation- and earthquake-triggered (4) | ordinal **class**, 0 and 1-5 | 0 to 5 | **0 — construct** |
| Drought SPI-6 hazard (3) | drought severity index, NOT raw SPI | ~3.3 to ~11.5 | **0 — see below** |
| Drought mean event duration / event count (2) | days / count | 0 to ~28 days; 0 to 16 events | **0 — see below** |
| Building Exposure Model, Population (2) | USD, people | not a hazard | n/a |

Those ranges are the whole global raster; your study area will span far less, so
always `get_hazard_stats` — and note that a flood layer reaching 15 m means a
curve tabulated to 6 m is clamped over much of its upper range.

Unit traps, every one of which has bitten someone:
- Flood is **millimetres**. The literature is almost all in metres: 1 m = 1000.
- PGA is **cm/s2**. The literature is almost all in g: 1 g = 980.665 cm/s2, so
  0.15 g = 147, 0.30 g = 294, 0.50 g = 490, 1.0 g = 981.
- Wind is **km/h**. The literature is in m/s or mph: 1 m/s = 3.6 km/h,
  1 mph = 1.609 km/h. 42 m/s = 151 km/h; 150 mph = 241 km/h.
- Landslide class is **ordinal, not physical**. Class 4 is not "twice class 2".
- The drought SPI-6 layer is **not raw SPI** (which runs about -3 to +3). It is a
  severity magnitude at a 5-year return period, spanning roughly 3.3 to 11.5
  globally. Read `get_hazard` for its definition before interpreting it.

Always call `get_hazard_stats` on the actual layer before finalising a curve:
the curve has to cover the range the raster actually contains, and the app
**clamps flat outside the tabulated range** rather than extrapolating.

## Step 1 — search the library

`search_curve_library(hazard=<the layer>, asset=<plain description>)`. It maps
project English onto the library's FEMA/JRC wording, so 'fibre backbone',
'trunk road' and 'mobile tower' all land. Pass the hazard layer, not a guess at
the family — it returns the right unit automatically and tells you when a layer
has no curves at all.

Then `use_library_curve(curve_id)` and the curve behaves exactly like an
uploaded CSV: `run_analysis(curve=...)`, `ui_set_vulnerability(curve=...)`, and
the app's own vulnerability mode all work on it unchanged.

## Step 2 — judge the candidates before you pick

The search returns several. They are not interchangeable, and the differences
are often larger than anything else in the analysis.

- **`characteristics` usually matters more than `asset`.** Anchored vs
  unanchored seismic components roughly doubles the damage at a given PGA.
  A motorway "with sophisticated accessories" and one without differ by an order
  of magnitude at 2 m of water. Size classes (small / medium / large plants)
  are real distinctions. Pick on the evidence you have about the user's network,
  and if you have none, say which you assumed.
- **`geography` is the transferability question.** 117 of the 218 curves are
  from US studies and only 9 claim global applicability; **none was derived in
  sub-Saharan Africa**. Using a Dutch flood curve for a Congolese network is
  defensible and routine, but it is an assumption and it belongs in the caveats.
- **`damage_at`** gives the damage factor at standard anchor intensities so you
  can compare candidates without opening a file. If two candidates differ by 3x
  at the intensities your network actually sees, that spread IS the uncertainty
  — run both and report a range rather than picking one silently.
- **Prefer `has_bounds: true`.** Those curves produce
  `total_damage_cost_lower`/`_upper` and error bars for free.
- **`output` must be `damage factor 0-1`.** Three reference curves emit a repair
  rate or a pipe-break count; `use_library_curve` refuses them, and rightly so —
  multiplying a replacement value by "repairs per km" is meaningless.
- **`derivation`**: E empirical, A analytical, H hybrid, O expert opinion.
  Empirical beats expert opinion when both fit.

Curves under `reference/` (`include_reference: true`) are keyed to metrics no
app layer provides. **They cannot be applied to anything.** Read them as
evidence when constructing a curve; never load them.

## Step 3 — constructing a curve

Build a curve when the library has no reasonable analogue, or for a layer with
no curves at all. Use `create_curve`: it enforces monotonicity, [0,1] range and
bound ordering, demands a written `basis`, and saves the curve as a CSV the user
can download and check. Supply `lower` and `upper` — a constructed curve without
a band claims precision it has not got.

### Shape

- **Flood: piecewise linear and concave.** Every major source does this — the
  JRC global curves are tabulated at 0.5 m steps and linearly interpolated, and
  Nirandjan et al. harmonised the whole database the same way. Do not fit a
  lognormal to flood depth. Most of the damage happens in the first metre.
- **Wind: convex, roughly cubic.** Emanuel (2011):
  `f = v^3 / (1 + v^3)` where `v = max(V - V_thresh, 0) / (V_half - V_thresh)`.
  Damage scales with pressure (V^2) at least; Nordhaus found the 9th power for
  US losses, so cubic is the conservative end. A LINEAR wind curve badly
  over-predicts at low speeds.
- **PGA: lognormal fragility per damage state, combined into a mean damage
  factor.** `P(DS>=i | im) = Phi(ln(im/theta_i)/beta_i)`, then
  `MDF(im) = sum_i DR_i * [P(DS>=i) - P(DS>=i+1)]`. Compute it in `python_exec`
  and pass the resulting points to `create_curve`.
- **Onset threshold.** A damage function starts at zero and stays there for a
  while. Flood ~100-300 mm for equipment-bearing assets; wind ~90-150 km/h;
  PGA ~0.10 g (98 cm/s2). A curve rising from the very first pixel is wrong.

### Anchors you may use

**Flood, generic infrastructure — JRC global depth-damage (Huizinga et al.
2017), converted to mm:** 0:0.00, 500:0.22, 1000:0.43, 1500:0.58, 2000:0.67,
3000:0.79, 4000:0.89, 5000:0.97, 6000:1.00. The Asia variant is slightly
steeper early (500:0.25, 1000:0.42, 2000:0.65).

**Flood, buried assets.** The library's own answer for buried cable and pipeline
crossings is **zero damage from submergence** (F5.1, F16.2 — FEMA). Buried
fibre and pipe are damaged at scoured crossings and by washouts, not by depth.
A flat-zero curve is often the honest one; if you build a small positive curve
to represent crossing failures, say that it is a proxy for scour, which the app
does not model.

**Wind.** Transmission tower collapse, empirical from Cyclone Fani 2019:
lognormal median **292.5 km/h**, beta 0.104 — a very steep, almost step-like
curve. Wood distribution poles: median failure **266 km/h** (class 1) to
**283 km/h** (class 3). Tree-fall driven line outages start around **151 km/h**.
Emanuel's aggregate form with `V_thresh` 90-150 km/h and `V_half` 260-290 km/h
for lattice towers and poles, 200-230 km/h for light masts and PV racking.
PV modules certified to ~225 km/h are a DESIGN RATING, not a damage function.

**PGA — HAZUS 5.1 fragility medians (g) for slight/moderate/extensive/complete,
with beta:**
- LV substation anchored 0.15/0.29/0.45/0.90, beta 0.70/0.55/0.45/0.45;
  unanchored 0.13/0.26/0.34/0.74.
- HV substation anchored 0.11/0.15/0.20/0.47; unanchored 0.09/0.13/0.17/0.38.
- Communication facility anchored 0.15/0.32/0.60/1.25, beta 0.75/0.60/0.62/0.65.
Damage ratios per state (best estimate): substations 0.05/0.11/0.55/1.00;
comms central office 0.09/0.35/0.73/1.00; water treatment 0.08/0.40/0.77/1.00;
storage tanks 0.20/0.40/0.80/1.00; fuel/tank farms 0.13/0.40/0.80/1.00;
**roadways 0.05/0.20/0.70 — never 1.00**; bridges 0.03/0.08/0.25/1.00.
Default beta 0.5-0.6 for a lifeline when you have to choose one.
**Buried pipelines are the exception: HAZUS has no fragility curve for them at
all**, only a repair rate driven by PGV, which this app does not have. Converting
PGA to PGV needs an empirical ratio; if you do it, put the assumption in the
text, do not bury it.

### Uncertainty

Default to **+/- 0.15 absolute, or +/- 35% relative, whichever is wider**,
clipped to [0,1] — wider near the onset threshold, narrower near saturation.
Where two library curves for the same asset disagree, the band must be at least
as wide as that disagreement. For scale: Koks et al. found the choice of
fragility curve explains ~60% of total loss variance, against ~20% for
reconstruction cost, and reported global expected annual damage as a 7-fold
range. A single number is never the honest answer.

### Before you accept your own curve

- f(0) = 0, non-decreasing, never above 1. `create_curve` enforces these.
- Non-zero onset threshold (above).
- **Point assets may reach 1.0. Linear assets priced per metre usually must
  not.** A per-metre replacement value embeds subgrade, earthworks and
  right-of-way that floodwater does not destroy; real damage is resurfacing and
  shoulder repair. **Cap road carriageway around 0.70**, as HAZUS does.
- Flood curves are concave, wind curves convex. A convex flood curve is suspect
  unless the asset sits on a plinth.
- Sanity: road and infrastructure damage is typically 4-18% of total direct
  flood damage in observed events. A curve pushing far outside that is wrong.
- Re-check every unit conversion. Unit error is the most common failure mode.

## The two layers with no curves

### Landslide susceptibility, class 1-5

There is **no published mapping from a susceptibility class to any physical
landslide intensity**, and you must say so rather than inventing one. The GIRI
model computes susceptibility as a weighted product of slope, lithology, land
cover and antecedent rainfall; it predicts WHERE a slide may start, not how big,
fast or deep it is. The library's landslide curves are keyed to ground
deformation, triggering precipitation and landslide area — physically
independent of the class. Do not map one onto the other.

What IS published is the GIRI model's own **probability that a significant
landslide impacts a 1 km stretch of road or railway**, by class and trigger.
For a mid-severity rainfall scenario (roughly the 20-200 year band) the row is
approximately: class 1 ~0%, class 2 2%, class 3 3%, class 4 5%, class 5 10%.
Note the shape: **class 1 is zero under every trigger**, and the rise is
multiplicative, not linear.

Build the curve as **P(impact | class) x V(damage | impacted)**, take V from the
fixed damage values in the landslide literature (roads 0.3-1.0 depending on slide
type; power lines 1.0 for debris flow), and give all five class points so
interpolation never spans a gap. An indicative set, to be labelled as such:

| Asset | V | c0 | c1 | c2 | c3 | c4 | c5 |
|---|---|---|---|---|---|---|---|
| Roads | 0.50 | 0 | 0 | 0.010 | 0.015 | 0.025 | 0.050 |
| Railways | 0.60 | 0 | 0 | 0.012 | 0.018 | 0.030 | 0.060 |
| Buried pipes / cables | 0.40 | 0 | 0 | 0.008 | 0.012 | 0.020 | 0.040 |
| Transmission towers | 0.90 | 0 | 0 | 0.005 | 0.008 | 0.013 | 0.025 |
| Buildings | 0.50 | 0 | 0 | 0.010 | 0.015 | 0.025 | 0.050 |

The raster carries a class **0** as well as 1-5, so tabulate all six points.

**Only the P row is published, and only for roads and railways.** Every V, and
the reduction applied to point assets, is your judgement. Never describe the
result as "following the GIRI paper" — the paper supplies one factor of a
two-factor product. Say which half is published and which half is yours, in
those terms.

Two more things must reach the user, and one setting must be applied:
- The damage figure is an expected fraction **conditional on a mid-severity
  rainfall scenario**, not an annual expected loss, and **confidence is low** —
  use that word. Lead with the exposure result, which is solid, and give the
  damage cost as indicative.
- The published probabilities are calibrated on a **1 km x 600 m cell taking the
  maximum-susceptibility pixel**, while this app samples ONE pixel at the asset.
  They are not the same quantity; mention it in the caveats.
- Set the exposure threshold to **class >= 2, arguably >= 3**. The default
  "affected if > 0" rule counts class-0 and class-1 pixels whose modelled
  probability is ~zero.

### Drought

**Do not build a drought damage curve.** The GIRI drought background paper — the
source of these very layers — states that energy, transport and communication
systems are rarely affected directly, and models the impact as a *production
loss*, correlating catchment area under SPI-6 < -1 against hydropower output.
Nirandjan et al. contains no drought curves at all. A damage factor is the wrong
formulation, and `replacement_value x damage_factor` computes a quantity that
does not exist.

Report instead: how much of the network sits in the drought-affected zone, the
mean event duration and event count, and the mechanisms that actually matter —
capacity loss (drought years cut global hydropower ~5% and thermal output ~4%),
navigability, and shrink-swell subsidence, which damages buried pipes and
foundations but depends on **clay soils the app does not map**. If the user
insists on a cost, give a capacity-loss fraction and label it explicitly as
production loss, not asset damage.

## Replacement value

`find_replacement_cost(asset, basis='unit'|'metre')` searches 179 published cost
figures. **`basis='unit'` for point datasets, `basis='metre'` for line datasets**
— that is exactly what the app's `replacement_value` means in each case.

These are euros from specific countries at the source studies' price levels.
Geography dominates: the companion JRC road figures run from EUR 4/m in Asia to
EUR 267/m in Africa. A few source rows are mislabelled by a factor of 1000, so
compare a candidate against the rest of its asset group before using it. Always
report the value, its geography, its cost basis (replacement vs construction vs
repair) and the currency, as an assumption. If the user gives you a value, use
theirs and say so.

## What must always reach the user

One short paragraph, up front, whenever you supply the curve yourself:
- **which** curve, by id and asset description, or that you constructed it;
- **where** it came from — the source study and its geography, or the basis;
- **how good** the match is, stated plainly when it is approximate;
- the **replacement value**, its basis and its provenance;
- and the **band** when there is one, as a range rather than a point estimate.

Then run the analysis. Do not stop to ask permission — the user asked for
vulnerability analysis, and a stated assumption they can correct is more useful
than a question that blocks the work. Offer `plot_curve` so they can see the
curve, and name the runner-up if a different reasonable choice would move the
answer materially.
"""


TOPICS: dict[str, str] = {
    "exposure_analysis": EXPOSURE_ANALYSIS,
    "vulnerability_analysis": VULNERABILITY_ANALYSIS,
    "vulnerability_curves": VULNERABILITY_CURVES,
    "multi_hazard": MULTI_HAZARD,
    "reports": REPORTS,
    "figures": FIGURES,
}
