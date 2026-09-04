# AI Assistant for the Infrastructure Risk Analyzer

## Context

The app (FastAPI backend on :8000, React/Vite/MapLibre frontend on :5173) has no AI capability.
Goal: an in-app assistant — FAB bottom-right of the map opening a chat panel on the RHS — that
(a) drives the app, (b) runs exposure **and** vulnerability analysis in the backend, (c) accepts
infrastructure files (shapefile zip / GeoPackage / GeoJSON / CSV), vulnerability-curve CSVs and
context documents (Word, Markdown, Excel, PDF), (d) draws charts and maps, and (e) writes
Word / Excel / CSV / GeoPackage deliverables — agentically, over many steps.

**Model to replicate:** `/Users/sebastiankrantz/Documents/web-projects/transport-intelligence`
(`backend/app/assistant/`, `frontend/src/components/assistant/`). Architecture, SSE protocol,
tool registry, kernel, guides ("skills"), report writer and panel are ported near-verbatim;
the domain layer, tool inventory and guides are rewritten for hazard/exposure work.

### Verified facts this design leans on

**Backend**
- `main.py` has **CORS only** — no GZipMiddleware, no static/SPA catch-all, no lifespan. New
  `/api/assistant/*` routes mount freely. (SSE would be safe either way; Starlette 0.49.3
  exempts `text/event-stream` from GZip by default.)
- All state is in-process and unsynchronised: `uploaded_files` (`api/upload.py:19`),
  `_hazards_cache` / `_hazard_stats_cache` (`api/hazards.py:20,23`), `_raster_values_cache` /
  `_analysis_results_cache` (`api/analyze.py:28,32`), `_tile_cache` (locked). Backend must run
  `--workers 1`. Assistant tools calling the model layer directly inherit this warm state.
- **`analyze_intersection` mutates the passed GeoDataFrame in place** for Points
  (`geospatial.py:620-672`) — that object *is* `uploaded_files[fid]["gdf"]`. Assistant tools
  MUST pass `.copy()`.
- Threshold semantics (`geospatial.py:613-616`, `:861-864`): threshold given → **`>=` threshold**;
  threshold `None` → **`> 0`**. NaN (no-data) is always unaffected.
- **There is NO earthquake/PGA inverse logic.** `CLAUDE.md:56` claims it; a repo-wide grep finds
  no hazard-id/name/category/unit branch anywhere in the analysis path — `geospatial.py` never
  even receives `hazard_id`. PGA uses the identical `>=` comparison as floods. The guides must
  describe the real behaviour; CLAUDE.md gets corrected.
- Vulnerability curve CSV: 2 columns (`intensity, proportion_destroyed`) or **≥4 columns**
  (`intensity, lower, central, upper` — **index 2 is the central curve**), header optional,
  proportions validated to `[0,1]`. `replacement_value` is **per asset** for points and
  **per metre** for lines. Damage is computed over **all** features, not only affected ones.
- Result keys — Points: `affected_count`, `unaffected_count`, `affected_meters`(0.0),
  `unaffected_meters`(0.0), `full_gdf`, `raster_values`. Lines: `affected_count`(0),
  `unaffected_count`(0), `affected_meters`, `unaffected_meters`, `full_gdf` (**one row per
  segment**), `line_data`, `raster_values`. Plus `total_damage_cost[_lower|_upper]`.
- Reusable renderers returning **PNG bytes**: `export.generate_barchart_png(...)` and
  `export.generate_map_png(...)` (matplotlib+seaborn+contextily, dpi=300). These ARE the app's
  own barchart/map exports — the assistant calls them directly, so its figures are identical
  to the ones the user downloads from the sidebar.
- Data exports: `export._run_data_export(file_id, hazard_id, threshold, mode) -> (bytes, filename, mime)`
  with modes `csv_points` / `csv_lines_aggregate` / `gpkg_lines_split`.
- `GET /api/upload/{file_id}` is **broken** (serialises the GeoDataFrame → 500). Never build on it.
- `hazard_url` must be resolved server-side from `load_hazards_dict()[id]["dataset_url"]`.
- `_hazard_stats_cache` stores **sqrt-transformed** bounds while `GET /api/hazards/{id}/stats`
  returns raw ones; tiles render correctly only after stats have been fetched once.
- `config.py` is a plain class, no dotenv. venv is Python 3.13.2 with geopandas/rasterio/
  matplotlib/seaborn/contextily/python-dotenv present. Added: `anthropic`, `google-genai`,
  `mcp`, `python-docx`, `openpyxl` (installed and import-verified).
- `data/hazard_layers.csv`: 29 layers, `;`-delimited, columns `hazard;dataset_url;description;
  background_paper;unit;category`. Categories: Flood hazard (9), Tropical cyclone (6),
  Drought (5), Landslides (4), Earthquakes (3), Socioeconomic (2). IDs are slugs of the name.
  Rows 22-24 have misaligned `unit`/`category` — tolerate garbage there.

**Frontend**
- 16 `useState` atoms in `App.tsx:9-24`, prop-drilled, no context/store. StrictMode is on.
- **Analysis auto-runs, debounced 300 ms** (`App.tsx:87-134`) on `[uploadedFile, selectedHazard,
  intensityThreshold, vulnerabilityAnalysisEnabled, vulnerabilityCurveFile, replacementValue]`.
  Setting several in one batch produces exactly one run.
- Three sequencing hazards: (1) `setSelectedHazard` **asynchronously overwrites**
  `intensityThreshold` with `stats.min` (`App.tsx:52`) — a threshold must be applied *after*
  stats settle; (2) vulnerability mode nulls the result until curve **and** `replacementValue > 0`
  are both set; (3) `setBasemap` fires two duplicate `setStyle` effects (`MapView.tsx:833`, `:1631`)
  — don't flip it rapidly.
- `intensityThreshold` is in **raw hazard units**; the slider transform is internal to Sidebar.
- **No keyboard handlers anywhere** in the frontend (`grep keydown|keyup|keypress` → 0 hits), so a
  chat textarea needs no `stopPropagation` workaround. Only global listener is a `mousedown`
  outside-click in Sidebar.
- z-index: `z-10` map controls/legends, `z-30` sidebar dropdown, `z-50` dialogs → **FAB z-20,
  panel z-30**, leaving the Disclaimer/info dialogs on top.
- Mount seam: as siblings after `<MapView …/>`, between `App.tsx:179` and `:180`, inside the root
  `flex h-screen w-screen` div (add `relative` to it for explicitness).
- `tsconfig`: strict + `noUnusedLocals`/`noUnusedParameters`; `build` = `tsc && vite build`, so
  type errors block the build. There is **no ESLint config** — `npm run build` is the only gate.
- No markdown renderer and no `ui/textarea` exist → both hand-rolled (as in the model app).
- Style: sidebar/panel = dark zone (`bg-gray-900/800/700`, `text-gray-300`, accent `blue-600`);
  map overlays = light. No semicolons, single quotes in feature components.
- The map camera and layer-visibility toggles live inside `MapView` and are unreachable. A
  minimal `onMapReady?: (map) => void` prop is added so the assistant can fit the camera; layer
  toggles stay out of scope.

## Architecture

```
Browser (React)                          FastAPI (single worker, :8000)
AssistantPanel ── POST /api/assistant/chat (SSE) ──► agent loop (assistant/loop.py)
  useAssistant hook   ◄─ text/tool_call/tool_result/     ├ providers: anthropic | gemini
  client-tool executor   artifact/await_client/done      ├ tool registry (server + client specs)
  (App.tsx bindings) ── POST /chat {tool_results} ──►    ├ python_exec kernel (per-conversation)
       │                                                 ├ artifact registry (LRU) ─ GET /artifacts/{id}
       └─ GET /api/assistant/artifacts/{id}              └ MCPServer grafted at /mcp (server tools)
External MCP clients (Claude Code/Desktop) ── Streamable HTTP /mcp ──┘
```

- **Agent loop runs server-side** inside the SSE generator (plain sync `def` +
  `StreamingResponse`), matching the codebase's threadpool style.
- **Server tools call the model layer directly** (`analyze_intersection`, `load_hazards_dict`,
  `generate_barchart_png`, …) — never over HTTP — so the assistant's numbers and figures are the
  app's own.
- **Client tools** (`ui_*`) end the SSE leg with `await_client`; the browser executes them against
  an App.tsx bindings ref and POSTs results back, opening a new leg ("stream-per-leg").
- **One registry** exports Anthropic tool defs, Gemini function declarations and MCP tools.

### New backend package

```
backend/app/assistant/
  __init__.py  schema.py  conversations.py  loop.py  system_prompt.py  guides.py
  kernel.py  artifacts.py  figures.py  reports.py  documents.py  mcp_server.py  mcp_state.py
  providers/{__init__,anthropic,gemini}.py
  tools/{__init__,query,data,analysis,output,ui,guide,documents}.py
backend/app/api/assistant_api.py
```
Modified: `backend/main.py`, `backend/app/config.py`, `backend/requirements.txt`, `.gitignore`,
`CLAUDE.md` (PGA correction). New: `backend/.env.example`.

## Providers & models

`schema.py` holds the canonical message format (`role` + parts: text / tool_call / tool_result /
file, plus provider-native `raw` on assistant turns for verbatim mid-turn replay).
`providers/*.stream_turn(model, system, messages, tools) -> Iterator[ProviderEvent]` with
`TextDelta | ThinkingDelta | ToolCall | Usage | TurnEnd`.

- **Anthropic** (`anthropic>=1`): `client.messages.stream(...)`, system as a block with
  `cache_control: ephemeral`, no `thinking` param, raw content blocks replayed verbatim when
  continuing a tool-use turn, parallel tool results in ONE user message, `is_error: true` on
  failures, images/PDFs as base64 `image`/`document` blocks.
- **Gemini** (`google-genai`): `generate_content_stream`, `system_instruction`, automatic function
  calling disabled, uuid call ids synthesised, results via `Part.from_function_response`.
- JSON schemas stay in the simple dialect both providers and MCP accept (object/string/number/
  integer/boolean/array/enum + required; no `oneOf`, `$ref`, `additionalProperties`).
- Config-driven model table in `config.py`, env-overridable; a provider is offered iff its key is set.

## SSE protocol

`POST /api/assistant/chat` → `text/event-stream`, headers `Cache-Control: no-cache`,
`X-Accel-Buffering: no`. Events: `start` · `text{delta}` · `thinking{delta}` ·
`tool_call{id,name,args,side}` · `tool_result{id,name,ok,summary}` · `artifact{…}` ·
`await_client{calls[]}` · `usage` · `error` · `done{reason}`. Comment heartbeat `: ping` every
15 s while a server tool grinds (raster sampling over remote COGs takes 5-60 s).

Loop: stream turn → run all server calls sequentially, emitting call/result events → if the turn
contains client calls, persist `pending_calls`, emit `await_client` + `done{awaiting_client}` and
end the leg → the browser executes and POSTs `tool_results`, which are validated against
`pending_calls`, merged with buffered server results into ONE tool-result message, opening a fresh
leg. Cap `ASSISTANT_MAX_ITERATIONS=30`. Tool errors become `{ok:false, error}` results, never
aborts. Abandoned `pending_calls` auto-resolve as failed on the next user message.

## Tool inventory

**Client (`tools/ui.py`, schema only; executor in `frontend/src/lib/assistantTools.ts`)**
`ui_read_app_state` · `ui_show_dataset(file_id)` · `ui_select_hazard(hazard)` ·
`ui_set_threshold(value)` · `ui_set_vulnerability(enabled, curve_file?, replacement_value?)` ·
`ui_set_display(palette?, opacity?, basemap?)` · `ui_set_sidebar(open)` · `ui_clear_data` ·
`ui_fit_map(bbox)`.
Sequencing is handled *in the executor*: `ui_select_hazard` awaits the stats fetch before
returning, and `ui_set_threshold` applies after it, so the App's async clobber cannot bite.
`ui_set_vulnerability` fetches the curve bytes from the conversation and builds a real `File`.

**Server — query (`tools/query.py`)**
`list_hazards(category?, search?)` · `get_hazard(hazard_id)` (description + background paper +
unit) · `get_hazard_stats(hazard_id)` (raw min/max, warms the tile cache) · `list_datasets()`.

**Server — data (`tools/data.py`)**
`load_infrastructure(file, file_id?)` — parses an uploaded `.zip/.gpkg/.geojson/.csv` through the
app's own loaders and **registers it in `uploaded_files`**, so the UI can display the very same
dataset; returns `file_id`, geometry type, feature count, bounds, attribute columns, head.
`load_vulnerability_curve(file)` — validates and stores the curve; reports whether it carries
uncertainty bounds.

**Server — analysis (`tools/analysis.py`)**
`run_analysis(file_id, hazard_id, threshold?, curve?, replacement_value?)` — the core; calls
`analyze_intersection` on a **copy**, writes the app's analysis cache so the sidebar's own export
buttons work afterwards, returns a compact summary + `stored_as` (full result incl. `full_gdf` in
the kernel namespace).
`compare_hazards(file_id, hazard_ids[], threshold?, ...)` — one dataset across many layers
(return periods, climate scenarios) → a tidy comparison DataFrame. The workhorse for reports.
`sweep_thresholds(file_id, hazard_id, thresholds[])` — sensitivity of exposure to the threshold.
`python_exec(code)` — persistent namespace.

**Server — output (`tools/output.py`)**
`make_barchart(analysis, …)` → the app's `generate_barchart_png` (identical to the sidebar export).
`make_map(analysis, palette?, basemap?, …)` → the app's `generate_map_png`.
`make_chart(kind, labels, series, …)` → general custom charts (bar/stacked/line/pie/scatter) in a
house style, for anything the two app renderers don't cover.
`make_custom_map(layers, …)` → general geopandas + contextily map for bespoke cartography.
`write_report_docx(title, sections)` · `export_excel(sheets)` · `export_csv(...)` ·
`export_analysis_data(file_id, hazard_id, mode, threshold?)` (the app's own CSV/GPKG writers) ·
`list_files()`.

**Server — guides & documents**
`read_guide(topic)` over `exposure_analysis` · `vulnerability_analysis` · `multi_hazard` ·
`reports` · `figures`. `read_document(file, sheet?)` for uploaded context documents.

**MCP exposure:** server-side tools only, plus `read_app_state` fed by the frontend's debounced
snapshot (`POST /api/assistant/app_state`).

## python_exec kernel

Per-conversation namespace, lazy. Preloaded: `pd, np, gpd, plt, rasterio`, the app modules
(`geospatial`, `hazards`, `analyze`, `export`, `export_data`), `uploaded_files`, `HAZARDS`
(the catalogue dict), `uploads` (name→Path) and `save_artifact(obj, filename, title=None)`.
AST-split exec + eval-of-trailing-expression, per-thread stdout capture, ~20 kB output clip,
matplotlib figures harvested to PNG artifacts after each call, `_FIG_LOCK` around exec-to-capture,
worker thread + `join(ASSISTANT_EXEC_TIMEOUT=180)` (raster work is slow), refusal to stack a second
exec on a namespace whose previous call is still running.

## Guides (the skill layer)

The system prompt stays lean; heavyweight tasks read a guide first. Content written against the
real payload shapes above.

1. **`exposure_analysis`** — workflow; `>=` vs `>0` semantics; NaN = unaffected; points give
   counts, lines give metres (segment-split at 100 m sampling); what a threshold means in each
   unit (mm of inundation, km/h gusts, cm/s² PGA, landslide class 1-5); the honest statement that
   no hazard type is treated inversely; choosing thresholds defensibly.
2. **`vulnerability_analysis`** — curve formats (2-col and 4-col with bounds); `replacement_value`
   is per asset (points) / per metre (lines); damage runs over all features, not only affected
   ones; how to report central + lower/upper; damage ratio vs exposure share — different things.
3. **`multi_hazard`** — the catalogue's structure (return period × climate scenario), how to build
   a return-period table and a climate-delta table, and how to talk about SSP1 lower / SSP5 upper
   bounds without over-claiming.
4. **`reports`** — structure (exec summary → context/method → results → caveats → conclusion),
   auto-numbered tables and figures that MUST be referenced in the prose, units on everything,
   Word tables for humans vs Excel/CSV for machines, the quality bar. Ported from the model app's
   excellent version and re-pointed at this domain.
5. **`figures`** — which renderer for which job: `make_barchart`/`make_map` for the app's standard
   exhibits, `make_chart`/`make_custom_map` for comparisons across hazards or countries,
   `python_exec` for anything else.

## Frontend

New under `frontend/src/`:
- `types/assistant.ts` — SSE events, transcript items, client-tool outcomes.
- `lib/assistantApi.ts` — POST-based SSE reader (fetch + ReadableStream + manual frame parsing),
  raw-body upload helper, app-state push.
- `lib/assistantTools.ts` — `AssistantBindings` interface, `snapshot()`, and a whitelist `switch`
  executor over `ui_*` names.
- `hooks/useAssistant.ts` — chat state + the leg loop, AbortController, debounced state push,
  StrictMode-safe.
- `components/assistant/` — `Assistant.tsx` (FAB ↔ panel), `AssistantFab.tsx`,
  `AssistantPanel.tsx`, `MessageList.tsx`, `ToolChip.tsx`, `ArtifactCard.tsx`, `Markdown.tsx`.

Modified: `App.tsx` (a `useMemo`'d bindings object + a `stateRef` updated each render, mounted
after `<MapView/>`), `MapView.tsx` (optional `onMapReady` prop), `types/index.ts` (`Meta`),
`services/api.ts` (`getMeta`).

## Phases (all on `feature/ai-assistant`)

0. **Plumbing** — requirements, `config.py` dotenv + assistant block, `/api/meta` availability,
   `.gitignore`, `.env.example`. *Verify:* imports clean; `/api/meta` reports availability.
1. **Registry + artifacts + conversations + query/data tools.** *Verify:* a venv script exercises
   `REGISTRY`, the exporters and a direct `list_hazards` / `load_infrastructure` call.
2. **Providers + loop + SSE chat endpoint.** *Verify:* `curl -N` a real question; no
   `Content-Encoding` on the stream; kill mid-stream → clean cancel.
3. **Kernel + analysis tools + uploads + documents.** *Verify:* real exposure run on the
   AfTerFibre fibre network; two-step `python_exec` proves persistence; timeout doesn't wedge.
4. **Figures + reports + output tools.** *Verify:* barchart/map PNGs match the sidebar exports;
   a .docx opens with figures and numbered tables; xlsx/csv round-trip.
5. **Frontend panel + client tools.** *Verify:* `npm run build` (tsc strict) clean; in-browser the
   assistant loads a dataset, selects a hazard, sets a threshold and the map/chart update.
6. **MCP** — `mcp_server.py` + graft into `main.py` with the session-manager lifespan.
   *Verify:* `claude mcp add --transport http …`; existing `/api` routes unaffected.
7. **Hardening + end-to-end test** — heartbeats, max-iteration message, no-key gating, eviction,
   README section, and a full scripted run: upload fibre network → exposure vs flood →
   vulnerability with the curve → multi-hazard comparison → Word report + Excel.

## Risks / gotchas

`analyze_intersection` mutates its input → always `.copy()` · unbounded analysis caches (assistant
runs add to them) → cap what the assistant writes · StrictMode double-mount → idempotent effects ·
the 300 ms auto-run means UI-driven changes recompute in the browser too (server tools remain the
source of truth for numbers) · `setSelectedHazard` clobbers the threshold asynchronously → the
executor awaits stats · `noUnusedLocals`/`noUnusedParameters` break the build · remote COG reads
are slow and occasionally fail → heartbeats and honest tool errors · unsandboxed `python_exec`
documented, localhost stance · API keys only in `backend/.env` (gitignored), booleans outward ·
the misaligned landslide CSV rows must not crash the catalogue tools.
