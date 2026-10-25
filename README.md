# Infrastructure Risk Analyzer

A high-performance web application for analyzing infrastructure assets against hazard layers. Upload spatial datasets (Shapefile, GeoPackage, or CSV) representing infrastructure and visualize their exposure to various hazard risks with interactive maps and statistics.

## Features

- **Multi-format Support**: Upload infrastructure data as Shapefiles, GeoPackages, or CSV files with coordinates
- **Hazard Layer Analysis**: Analyze infrastructure against multiple hazard layers (floods, earthquakes, etc.)
- **Interactive Visualization**: 
  - Dynamic map with multiple basemap options
  - Color-coded hazard layers with adjustable opacity
  - Real-time affected/unaffected infrastructure visualization
- **Customizable Analysis**:
  - Adjustable hazard intensity thresholds
  - Multiple color palettes for hazard visualization
  - Layer visibility controls
- **Statistical Summary**: Bar charts showing affected vs. unaffected infrastructure counts or lengths
- **User-Friendly Interface**: Collapsible sidebar with info tooltips and responsive design
- **AI Assistant** (optional): an in-app chat panel that loads your data, runs the exposure and
  vulnerability analyses, drives the map, and writes Word / Excel / CSV / GeoPackage deliverables
  — see [AI Assistant](#ai-assistant) below

## Tech Stack

### Backend
- **FastAPI** - Modern Python web framework
- **Geospatial Libraries**: `geopandas`, `xarray`, `rasterio`, `pyogrio`, `pandas`
- **Cloud Optimized GeoTIFF (COG)** support for efficient raster processing

### Frontend
- **React** + **TypeScript** - Component-based UI
- **Vite** - Fast build tool and dev server
- **MapLibre GL** - High-performance map rendering
- **Recharts** - Data visualization
- **TailwindCSS** + **shadcn/ui** - Modern styling and UI components

## Quick Start

### Prerequisites

- **Backend**: Python 3.9 or higher
- **Frontend**: Node.js 18+ and npm

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create data directory and configure hazard layers:
```bash
mkdir -p data
```

Create `data/hazard_layers.csv` with semicolon-delimited format:
```csv
hazard;dataset_url;description;background_paper
Flood Hazard 10 Years;https://example.com/flood.tif;Description;Paper URL
```

5. Start the backend server:
```bash
python main.py
# Or: uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The application will be available at `http://localhost:5173`

## Usage

1. **Upload Infrastructure Data**: 
   - Click "Upload Infrastructure Data" and select a Shapefile (zipped), GeoPackage, or CSV file
   - CSV files should contain coordinate columns (`lat`/`lon`, `latitude`/`longitude`, or `y`/`x`)

2. **Select Hazard Layer**: 
   - Choose a hazard layer from the dropdown
   - Click the info icon (ℹ️) to view metadata and background papers

3. **Customize Visualization**:
   - Select a color palette for the hazard layer
   - Adjust opacity (0-100%)
   - Set hazard intensity threshold using the slider

4. **View Results**:
   - The map shows affected (red) and unaffected (green) infrastructure
   - The bar chart displays summary statistics
   - Toggle layer visibility using the layer control panel

## Project Structure

### Backend

```
backend/
├── main.py                # FastAPI application entry point
├── app/
│   ├── config.py          # Configuration & paths
│   ├── api/
│   │   ├── upload.py      # File upload endpoints
│   │   ├── hazards.py     # Hazard layer endpoints
│   │   ├── analyze.py     # Analysis endpoints
│   │   └── tiles.py       # Tile serving endpoints
│   └── utils/
│       └── geospatial.py  # Geospatial utilities
├── requirements.txt       # Dependencies
└── README.md              # Backend setup instructions
```

### Frontend

```
frontend/
├── index.html             # HTML entry point
├── package.json           # NPM dependencies & scripts
├── vite.config.ts         # Vite configuration
├── tsconfig.json          # TypeScript configuration
├── tailwind.config.js     # TailwindCSS configuration
├── postcss.config.js      # PostCSS configuration
├── README.md              # Frontend setup instructions
└── src/
    ├── main.tsx           # React entry point
    ├── App.tsx            # Main app component
    ├── index.css          # Global styles
    ├── vite-env.d.ts      # Vite type definitions
    ├── components/
    │   ├── ui/            # Reusable UI components
    │   │   ├── button.tsx
    │   │   ├── input.tsx
    │   │   ├── select.tsx
    │   │   ├── slider.tsx
    │   │   ├── card.tsx
    │   │   └── dialog.tsx
    │   ├── Sidebar.tsx    # Collapsible sidebar with controls
    │   ├── MapView.tsx    # Map component with basemap selector
    │   └── BarChart.tsx   # Chart component (Recharts)
    ├── types/
    │   └── index.ts       # TypeScript type definitions
    ├── services/
    │   └── api.ts         # API service functions
    └── lib/
        └── utils.ts       # Utility functions
```

## API Endpoints

### Upload
- `POST /api/upload` - Upload spatial file (Shapefile, GeoPackage, or CSV)

### Hazards
- `GET /api/hazards` - List available hazard layers
- `GET /api/hazards/{hazard_id}` - Get hazard layer details
- `GET /api/hazards/{hazard_id}/stats` - Get hazard statistics (min, max, mean)

### Analysis
- `POST /api/analyze` - Perform spatial intersection analysis

### Tiles
- `GET /api/tiles/{hazard_id}/{z}/{x}/{y}.png` - Get hazard layer tiles

For detailed API documentation, visit `http://localhost:8000/docs` when the backend is running.

## AI Assistant

An agentic assistant lives in a chat panel behind the ✨ button at the bottom-right of the map.
It can do the whole job end to end: take a shapefile / GeoPackage / GeoJSON / CSV plus a
vulnerability curve and any context documents, run the exposure **and** damage analyses, drive the
app so you see the result on screen, and hand back a Word report, an Excel workbook, CSVs or a
GeoPackage of the split segments.

Its numbers are the app's numbers — the tools call the same analysis functions the UI calls, on the
same in-memory data, and its standard chart and map are rendered by the same code as the sidebar's
export buttons.

### Setup

```bash
cd backend
cp .env.example .env      # then add ANTHROPIC_API_KEY, GEMINI_API_KEY and/or OPENAI_API_KEY
pip install -r requirements.txt
```

Without a key the assistant is simply absent: the button never renders and the app behaves exactly
as before. A provider appears in the model picker only when its key is set (Gemini: when its
backend is configured, see below).

**Assistant model defaults** (optional; the same four variables in every AEI Labs app):
`ASSISTANT_DEFAULT_PROVIDER` is the provider the assistant starts on (`anthropic`, `gemini` or
`openai`), used only when that provider has a key; unset, it is the first keyed of anthropic,
gemini, openai. `ANTHROPIC_MODEL`, `GEMINI_MODEL` and `OPENAI_MODEL` replace that provider's
default model, honoured only when the value is one of the provider's models in the assistant's
model table; any other value is ignored with a warning in the log.

### Gemini on Vertex AI

The Gemini provider runs on one of two backends, chosen by `GEMINI_BACKEND` in `backend/.env`:

- `vertex`: Vertex AI, which Google now calls the Gemini Enterprise Agent Platform. Usage is billed
  to the Google Cloud project `VERTEX_PROJECT` and draws on that project's credits.
  `VERTEX_LOCATION` defaults to `global`.
- `developer`: the Gemini Developer API (an AI Studio key, `GEMINI_API_KEY`). It is also what runs
  when `GEMINI_BACKEND` is unset, so a deployment keeps working until its `.env` switches.

Gemini is offered in the model picker when its backend has what it needs: `VERTEX_PROJECT` or
`VERTEX_API_KEY` on `vertex`, `GEMINI_API_KEY` on `developer`. `backend/.env.example` shows the
`vertex` setup. The code is `make_client` and `_flex_options` in
`backend/app/assistant/providers/gemini.py`, and `gemini_configured` in `backend/app/config.py`; it
needs `google-genai` 1.74 or later, the first release with `Client(enterprise=...)`.

**Authentication on Vertex**, one of two:

1. *Application Default Credentials.* On a dev machine, enable the Vertex AI API
   (`aiplatform.googleapis.com`) in the project, link the billing account that carries the credit,
   and log in with `gcloud auth application-default login` (then
   `gcloud auth application-default set-quota-project <project>`); the account needs the Vertex AI
   User role. On a server, create a service account with that role, keep its JSON key outside the
   repository (the Docker image copies `backend/`), and set `GOOGLE_APPLICATION_CREDENTIALS` to its
   path. The server
   removes secret-named variables from its environment at startup (`app/redact.py`), so
   `config.py` reads that path first and the adapter loads the key file for the client itself.
2. *`VERTEX_API_KEY`.* A Google Cloud API key: requests go to the global endpoint with no project in
   the path, and the key's own project is billed. `VERTEX_PROJECT` and `VERTEX_LOCATION` are then
   unused. Google documents this form for express mode, whose model list (checked 2026-10-13) has
   `gemini-3.1-pro-preview` but not `gemini-3.8-flash` or `gemini-3.5-flash-lite`; try each model
   before relying on a key.

**Flex.** On Vertex, flex is [Flex PayGo](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/flex-paygo),
asked for with the header `X-Vertex-AI-LLM-Shared-Request-Type: flex` instead of `service_tier`. It
runs only at the `global` location and only for the models in `VERTEX_FLEX_MODELS`, which holds all
three models of the table; any other model or location logs a warning and runs at the standard
tier. The 15-minute timeout and the back-off-then-standard fallback are unchanged, and the fallback
drops the header.

**Docker.** `docker-compose.yml` passes no assistant settings to the container today, so the
assistant is off there. To run Gemini on Vertex it needs `GEMINI_BACKEND`, `VERTEX_PROJECT` and
`VERTEX_LOCATION` under `environment:`, and either `VERTEX_API_KEY` or a service-account key file
mounted read-only, from outside the repository, with `GOOGLE_APPLICATION_CREDENTIALS` naming the
mounted path.

**Credentials and `python_exec`.** The kernel runs in the server process, confined by
`app/assistant/sandbox.py`: it cannot read gcloud's `application_default_credentials.json` or the
key file `GOOGLE_APPLICATION_CREDENTIALS` names, wherever that file is kept, and it finds neither
that variable nor `VERTEX_API_KEY` in its environment. As a second layer, `app/redact.py` masks
Google refresh tokens (`1//…`), access tokens (`ya29.…`), PEM private-key blocks and the values of
`refresh_token`, `client_secret` and `private_key` fields in anything a tool returns.

### What it can do

| | |
|---|---|
| **Data** | `load_infrastructure` (zip/gpkg/geojson/csv), `load_vulnerability_curve` (2-col or 4-col with uncertainty bands) |
| **Analysis** | `run_analysis` (exposure + optional damage cost), `compare_hazards` (return periods, climate scenarios), `sweep_thresholds` (sensitivity), `python_exec` (open-ended, persistent namespace) |
| **Catalogue** | `list_hazards`, `get_hazard` (method + background paper), `get_hazard_stats`, `list_datasets` |
| **Figures** | `make_barchart` and `make_map` (the app's own exhibits), `make_chart` and `make_custom_map` (comparisons and bespoke cartography) |
| **Deliverables** | `write_report_docx`, `export_excel`, `export_csv`, `export_analysis_data` (the app's per-asset CSV/GeoPackage writers) |
| **App control** | `ui_show_dataset`, `ui_select_hazard`, `ui_set_threshold`, `ui_set_vulnerability`, `ui_set_display`, `ui_fit_map`, `ui_set_sidebar`, `ui_clear_data`, `ui_read_app_state` |

Report quality is governed by `backend/app/assistant/guides.py` — five written guides the model must
read before analysing or writing. **Edit those to change how the assistant works**; they carry the
workflow, the exact model semantics and the standing caveats, and they are far more effective than
editing the system prompt.

### MCP endpoint

The same server-side tools are exposed over MCP at `/mcp` (Streamable HTTP), so external agents can
drive the analysis layer directly. This needs no API key — the client brings its own model:

```bash
claude mcp add --transport http infrarisk http://127.0.0.1:8000/mcp
```

Set `ASSISTANT_MCP_ENABLED=0` to turn it off.

> **Security:** `python_exec` is deliberately unsandboxed, and the MCP endpoint is unauthenticated.
> Bind the backend to localhost, or put it behind authentication before exposing the port.

## Configuration

### Backend Environment Variables

- `HOST`: Server host (default: `0.0.0.0`)
- `PORT`: Server port (default: `8000`)
- `DEBUG`: Debug mode (default: `False`)
- `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` / `OPENAI_API_KEY`: enable the AI assistant (read from `backend/.env`)
- `GEMINI_BACKEND` (`vertex` | `developer`, unset = `developer`), `VERTEX_PROJECT`, `VERTEX_LOCATION` (`global`), `VERTEX_API_KEY`, `GOOGLE_APPLICATION_CREDENTIALS`: where Gemini runs and how it authenticates, see "Gemini on Vertex AI"
- `ASSISTANT_MAX_ITERATIONS` (30), `ASSISTANT_EXEC_TIMEOUT` (180 s), `ASSISTANT_UPLOAD_MAX_MB` (100), `ASSISTANT_MCP_ENABLED` (1)

The frontend dev proxy targets `http://localhost:8000`; override with `BACKEND_URL` if that port is
taken (`BACKEND_URL=http://127.0.0.1:8123 npm run dev`).

### Data Format

**Hazard Layers CSV** (`data/hazard_layers.csv`):
- Delimiter: semicolon (`;`)
- Required columns: `hazard`, `dataset_url`
- Optional columns: `description`, `background_paper`

**Infrastructure Upload Formats**:
- Shapefile: Zipped with `.shp`, `.shx`, `.dbf` (and optionally `.prj`)
- GeoPackage: Single `.gpkg` file
- CSV: Must contain coordinate columns (see usage section)

## Development

### Backend Development

Run with auto-reload:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Development

The Vite dev server includes hot module replacement (HMR) for instant updates.

### Building for Production

**Frontend**:
```bash
cd frontend
npm run build
```

**Backend**:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

> **Note**: Use `--workers 1` as the application uses in-memory state for uploaded files and analysis caching. Multi-worker deployment would require Redis or similar for shared state.

## License

GPL-3
