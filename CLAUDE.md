# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Infrastructure Risk Analyzer — a full-stack web app for analyzing infrastructure assets (points/lines) against geospatial hazard raster layers (floods, earthquakes, cyclones, landslides, drought). Users upload spatial datasets, select hazard layers, set intensity thresholds, and view exposure results on an interactive map with bar charts.

## Development Commands

### Backend (FastAPI/Python)

```bash
# Activate venv and run with auto-reload (from repo root)
source venv/bin/activate
cd backend && uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Or simply:
cd backend && python main.py
```

### Frontend (React/TypeScript/Vite)

```bash
cd frontend && npm run dev      # Dev server on port 5173 (proxies /api to :8000)
cd frontend && npm run build    # Production build (tsc && vite build)
cd frontend && npm run lint     # ESLint
```

### Running Both

Start backend first (port 8000), then frontend (port 5173). Vite proxies `/api/*` requests to the backend.

## Architecture

### Monorepo Layout

- `backend/` — FastAPI Python API server
- `frontend/` — React + TypeScript + Vite SPA
- `data/` — Hazard layer config (`hazard_layers.csv`, semicolon-delimited), local rasters (`rasters/`) and the shipped vulnerability-curve library (`vulnerability_curves/`)

### Backend (`backend/`)

- **Entry:** `main.py` → FastAPI app with CORS, mounts all API routers
- **Config:** `app/config.py` — paths (BASE_DIR, UPLOAD_DIR, DATA_DIR), env vars (HOST, PORT, DEBUG, MAX_UPLOAD_SIZE)
- **API routers** in `app/api/`:
  - `upload.py` — `POST /api/upload` file parsing (zip/gpkg/csv), transforms to WGS84, stores GeoDataFrame in-memory
  - `hazards.py` — `GET /api/hazards` reads `data/hazard_layers.csv`, raster stats via rasterio
  - `analyze.py` — `POST /api/analyze` spatial intersection (raster sampling at infrastructure locations), caches raster values per `(file_id, hazard_id)` to avoid re-sampling on threshold changes
  - `tiles.py` — `GET /api/tiles/{hazard_id}/{z}/{x}/{y}.png` custom COG tile renderer with matplotlib colormaps and TTL cache
  - `export.py` — `POST /api/export/barchart`, `POST /api/export/map` high-res PNG exports
  - `assistant_api.py` — `POST /api/assistant/chat` (SSE), uploads, artifacts, `GET /api/assistant/meta`
- **Geospatial utils:** `app/utils/geospatial.py` — load spatial formats, raster-infrastructure intersection, vulnerability curves, line length via Geod
- **AI assistant:** `app/assistant/` — see the dedicated section below

**Critical constraint:** All uploaded data and analysis results live in-memory (`uploaded_files` dict + caches). Backend must run with `--workers 1`. Multi-worker needs Redis or similar.

**Analysis logic:** An asset is affected when the sampled hazard intensity is **`>= intensity_threshold`**, or **`> 0`** when no threshold is passed (`geospatial.py:613-616` for points, `:861-864` for lines). No-data (NaN) always counts as unaffected. There is **no hazard-specific branch** anywhere in the analysis path — `analyze_intersection` never receives `hazard_id`, so PGA, flood depth and wind speed are all compared the same way. (An earlier version of this file claimed PGA used inverted logic; that was never true of the code.)

**Vulnerability logic:** Curves are 2-column (`intensity, proportion_destroyed`) or 4+-column (`intensity, lower, central, upper`, index 2 being central). `replacement_value` is **per feature** for point datasets and **per metre** for line datasets — one scalar, or (assistant/MCP) an array with one value per input row. Damage is computed **only over affected features** — those at or above `intensity_threshold`, or above 0 when none is set. A curve typically returns a positive damage ratio well below any sensible threshold, so summing it over every feature made the total damage cost independent of the threshold while the reported damage ratio (already averaged over affected features only) moved with it; raising the threshold now lowers the cost and raises the mean ratio, which is the coherent pair. The threshold slider therefore stays visible in vulnerability mode.

### Frontend (`frontend/src/`)

- **Entry:** `main.tsx` → `App.tsx` (all top-level state via useState/useEffect)
- **Components:**
  - `Sidebar.tsx` — main control panel: file upload, hazard selector, color palette, opacity/threshold sliders, analysis results, vulnerability toggle, export buttons
  - `MapView.tsx` — MapLibre GL + deck.gl rendering, basemap selector (11 options), hazard tile overlay, infrastructure GeoJSON layer
  - `BarChart.tsx` — Recharts visualization of affected/unaffected counts or meters
  - `ui/` — shadcn-style reusable components (button, input, select, slider, card, dialog)
- **API client:** `services/api.ts` — typed fetch wrappers for all backend endpoints
- **Types:** `types/index.ts` — `Hazard`, `UploadedFile`, `AnalysisResult`, `ColorPalette`, `Basemap`
- **Path alias:** `@/*` → `./src/*` (configured in tsconfig.json)

### Data Flow

1. **Upload:** User file → `POST /api/upload` → validate/parse/transform to WGS84 → store GeoDataFrame in memory → return GeoJSON + metadata
2. **Hazard tiles:** Frontend requests `/api/tiles/{id}/{z}/{x}/{y}.png?palette=turbo` → backend reads COG, applies colormap, returns cached PNG
3. **Analysis:** User sets hazard + threshold → `POST /api/analyze` → sample raster at infrastructure locations (cached) → classify affected/unaffected → return summary + GeoJSON
4. **Export:** `POST /api/export/*` → matplotlib renders high-res PNG → download

### Hazard Layer Configuration

`data/hazard_layers.csv` uses semicolon (`;`) delimiter. Required columns: `hazard`, `dataset_url`. Optional: `description`, `background_paper`, `unit`, `category`. URLs point to Cloud Optimized GeoTIFFs (remote or local in `data/rasters/`). A few landslide rows have misaligned `unit`/`category` fields — code reading them must tolerate junk.

## AI Assistant (`backend/app/assistant/`)

An agentic chat panel that drives the app, runs the analyses, and writes deliverables. Disabled and invisible unless an API key is set in `backend/.env` (see `.env.example`).

- **Registry** (`tools/__init__.py`): one `@tool` declaration feeds the Anthropic, Gemini and OpenAI adapters and the MCP server. `side="server"` tools run in-process; `side="client"` (`ui_*`) tools are schema-only and executed by the browser.
- **Models & execution** (`models.py`): the ONE model table — providers (Claude / Gemini / OpenAI), models, per-model reasoning-effort levels and defaults, Standard/Flex tiers — kept in step with AGUI's `backend/agui/config.py`. `GET /api/assistant/meta` serves it; `POST /api/assistant/chat` accepts `provider`, `model`, `effort`, `service_tier` and validates them with `models.choose` (an unoffered model runs at the provider's default with a logged warning, an invalid effort is dropped, flex on Claude runs standard; an ignored `*_MODEL` override is also logged once). The adapters send effort as `output_config.effort` (Claude), `reasoning.effort` (OpenAI Responses) or `thinking_level` (Gemini); flex (OpenAI, Gemini) backs off 2/4/8/16 s on a capacity refusal, then goes out at the standard tier; `loop._provider_with_pings` keeps the SSE stream warm (a `: ping` after 15 s of provider silence) while a flex request waits. The panel's popover is `ModelPicker.tsx`; the choice is persisted in localStorage (`infrarisk.assistant.choice`) and AGUI's `ai_provider` / `ai_model` / `ai_effort` / `ai_tier` URL parameters (`lib/assistantChoice.ts`) replace it on load. Tests: `cd backend && python -m unittest discover -s tests`, `cd frontend && node --test tests/assistantChoice.test.ts`.
- **Loop** (`loop.py`): runs inside the SSE generator. Server tools execute inline with 15 s heartbeat pings; a turn containing client tools ends the leg with `await_client`, and the browser posts results back to open the next leg ("stream-per-leg"). Transient provider errors (503/429/…) retry with backoff, but only before any token has been emitted.
- **Server tools call the model layer directly** (`analyze_intersection`, `load_hazards_dict`, `generate_barchart_png`, `generate_map_png`, `_run_data_export`) — never over HTTP — so the assistant's numbers and figures ARE the app's. `domain.run_exposure` also writes the app's analysis cache, so the sidebar's export buttons work on an assistant-run analysis.
- **Always pass `gdf.copy()`** into `analyze_intersection`: it writes columns into the frame it is given, and that object is `uploaded_files[file_id]["gdf"]`.
- **Guides** (`guides.py`) are the skill layer: `exposure_analysis`, `vulnerability_analysis`, `vulnerability_curves`, `multi_hazard`, `reports`, `figures`. The system prompt stays lean and makes `read_guide` mandatory before the heavy tasks. Edit these to change how the assistant works and writes — that is the highest-leverage file in the package.
- **Kernel** (`kernel.py` + `sandbox.py`): a persistent per-conversation `python_exec` namespace preloaded with facades of the app modules and a **read-only** view of `uploaded_files`. Confined (AEI MCP contract §8, P1): the code runs on a worker thread with the conversation's workdir as cwd; an audit hook on that thread (and threads it starts) limits files to the workdir, temp files and a read-only `data/`, refuses listing elsewhere, subprocesses, ctypes and importing the server package; the GDAL entry points (pyogrio, `rasterio.open`), whose C-level file access no audit event sees, carry the same path check; an AST check refuses `__globals__`/frames/`sys.modules`. At the deadline the worker is stopped with an async exception, so a runaway loop cannot wedge the scope. It is a fence, not a wall — the subprocess sandbox is the next step. Every file a call produces (`save_artifact`, figures, files written into the workdir) comes back under `artifacts`.
- **Frontend**: `components/assistant/`, `hooks/useAssistant.ts`, `lib/assistantApi.ts` (SSE over POST), `lib/assistantTools.ts` (the `ui_*` executor + app-state snapshot). `App.tsx` publishes an `AssistantBindings` ref each render.
- **The threshold race**: selecting a hazard asynchronously overwrites `intensityThreshold` with the layer minimum. `App.tsx` tracks `statsHazardId` alongside `hazardStats`, and `ui_select_hazard` waits on it so a following `ui_set_threshold` cannot be clobbered.
- **Curve library** (`curve_library.py` + `tools/curves.py`): the assistant supplies the vulnerability curve itself rather than asking for one. `data/vulnerability_curves/` ships 218 curves from Nirandjan et al. (2024) with a searchable index and a replacement-cost table; see its README. `search_curve_library` ranks candidates against a free-text asset description (a synonym map bridges project English to the library's FEMA/JRC wording), `use_library_curve` loads one through the same parser an upload uses, `create_curve` builds one with validation when nothing fits, and `find_replacement_cost` proposes a value. **Only flood (mm), PGA (cm/s²) and cyclone wind (km/h) have curves in an app layer's units** — the landslide layers are an ordinal class and drought has no curves at all, so both are construct-or-decline paths spelled out in `read_guide('vulnerability_curves')`. Rebuild the library with `scripts/build_curve_library.py` (it validates every emitted file through the app's own parser and preserves the hand-written README).
- **MCP** (`mcp_server.py`, `mcp_tools.py`, `jobs.py`): server-side tools are also served at `/mcp` (Streamable HTTP) for external clients, implementing the AEI Labs MCP integration contract v1 (v1.1; the text is in the server `instructions`). One async wrapper runs every call in the caller's scope — `_meta["aeilabs/scope"]` or `X-AEI-Scope` → conversation `mcp:<scope>`, else the legacy `mcp-shared-scope` — on a worker thread under `ASSISTANT_TOOL_TIMEOUT`, and re-raises every failure as a `ToolError` (the SDK redacts anything else to "Error executing tool X"). Scopes isolate uploads, curves, namespace, artifacts and jobs; `uploaded_files` and the analysis caches stay process-global (the caches are locked). MCP-only tools: `upload_file` (base64, `ASSISTANT_MCP_UPLOAD_MAX_MB`), `reset_scope`, `delete_dataset`, `start_run_analysis` / `start_compare_hazards` / `get_job` / `cancel_job`. Grafted via `app.router.routes.extend(...)`, not `mount`, so the bare `/mcp` path matches. Check it with AGUI's `backend/tests/live/contract_check.py http://127.0.0.1:8060/mcp`.
- **Data in / results out**: `load_features` (inline GeoJSON) and `load_from_url` join `load_infrastructure`; any analysis tool accepts an uploaded file's name as `file_id`. `get_analysis_table` / `get_affected_segments` (and `run_analysis(include_features, export)`) hand over the per-feature table inline or as CSV/GeoPackage artifacts. `replacement_value_column` / `replacement_value_map` give each asset its own value (`analyze_intersection` takes one value per input row); `id_column` carries the caller's ids through as `id` / `line_id`.

## Key Technical Details

- **Geospatial stack:** geopandas, rasterio, xarray/rioxarray, pyogrio for I/O
- **Tile caching:** TTL 5min, max 2000 tiles, thread-local rasterio connections
- **Infrastructure types:** Point and LineString geometries only. Lines use Geod for accurate length calculations (meters).
- **Vulnerability analysis:** Optional CSV upload (intensity, proportion_destroyed) + replacement value → damage cost calculation
- **Frontend styling:** TailwindCSS 3.4 + class-variance-authority + tailwind-merge
- **TypeScript:** Strict mode, ES2020 target
