# How hazard layers are loaded (and why it can fail on Posit)

## Load chain

1. **Config** (`app/config.py`)
   - `BASE_DIR` = directory containing `app/` (the backend root).
   - `DATA_DIR` = `os.getenv("DATA_DIR", BASE_DIR / "data")` → default: `backend/data`.
   - `HAZARD_LAYERS_CSV` = `DATA_DIR / "hazard_layers.csv"` → default: `backend/data/hazard_layers.csv`.

2. **CSV** (`backend/data/hazard_layers.csv`)
   - Semicolon-delimited; required columns: `hazard`, `dataset_url`.
   - `dataset_url` holds **absolute paths** to GeoTIFFs (e.g. `/Data/shinydata/infrarisk/PGA_2475y.tif`).

3. **Where the CSV is used**
   - `app/api/hazards.py`: `load_hazards_dict()` reads the CSV from `settings.HAZARD_LAYERS_CSV`. Used by:
     - `GET /api/hazards` (list layers)
     - `GET /api/hazards/{id}` (layer info)
     - `GET /api/hazards/{id}/stats` (raster min/max)
   - `app/api/tiles.py`: gets `dataset_url` from the same dict and passes it to tile generation.
   - `app/api/analyze.py` and `app/api/export.py`: get `dataset_url` from the same dict for analysis/export.

4. **If the CSV is missing**
   - `load_hazards_dict()` returns an **empty dict** (no exception). The API returns zero hazard layers, so the app “does not work” (no layers to select).

## Why the Quarto doc works on Posit but the app doesn’t

- **Quarto** (`reading-tiff-files.qmd`) uses the **same** TIFF path (`/Data/shinydata/infrarisk/PGA_2475y.tif`) and runs in an R process on the same server, so it can read the file.
- The **FastAPI app** does not fail on the TIFF path; it fails earlier if it **never finds the CSV**:
  - The app resolves the CSV path from `DATA_DIR` (and optional env), which is derived from the **Python app’s** install location (`__file__` → `backend/app/config.py` → `backend/data/hazard_layers.csv`).
  - On Posit, the deployed app may:
    - Run from a different layout so that `backend/data/` (and thus `hazard_layers.csv`) is not present in the deployed bundle, or
    - Have `DATA_DIR` set (e.g. to `/Data/shinydata/infrarisk`), so the app looks for the CSV at `/Data/shinydata/infrarisk/hazard_layers.csv`, which does not exist there.

So: TIFFs are fine on the server; the likely failure is that **the CSV file is not found** by the FastAPI app.

## Fix for Posit

1. **Ensure the CSV is deployed**  
   Deploy `backend/data/hazard_layers.csv` with the app so the default path `{backend}/data/hazard_layers.csv` exists where the app runs.

2. **Or set the CSV path explicitly**  
   Set the environment variable **`HAZARD_LAYERS_CSV`** to the **absolute path** of `hazard_layers.csv` on the server (e.g. `/path/to/deployed/backend/data/hazard_layers.csv`). The app will use this path instead of `DATA_DIR / "hazard_layers.csv"`.

3. **Do not set `DATA_DIR` to the TIFF directory**  
   If you set `DATA_DIR=/Data/shinydata/infrarisk`, the app will look for the CSV at `/Data/shinydata/infrarisk/hazard_layers.csv`. The CSV lives in the repo at `backend/data/`, not next to the TIFFs. Either leave `DATA_DIR` unset (so the CSV is expected under backend’s `data/`) or use `HAZARD_LAYERS_CSV` to point directly at the CSV file.

## Summary

| Component        | Role |
|-----------------|------|
| `app/config.py` | Defines `HAZARD_LAYERS_CSV` (default: `DATA_DIR / "hazard_layers.csv"`). Can be overridden with `HAZARD_LAYERS_CSV` env. |
| `app/api/hazards.py` | Loads CSV from `settings.HAZARD_LAYERS_CSV`; returns empty dict if file missing. |
| `hazard_layers.csv` | Must exist at that path; contains `dataset_url` = absolute TIFF paths (e.g. `/Data/shinydata/infrarisk/...`). |
| TIFF paths        | Used as-is; no prefix or `DATA_DIR` is applied. Same paths as in the working Quarto doc. |

## If the CSV loads but nothing happens when you select a hazard

The app can list hazard layers (CSV is fine) but fail to **read the TIFFs** at the paths in `dataset_url`. Typical cause on Posit: **the process running the FastAPI app does not have read permission** for those paths (e.g. `/Data/shinydata/infrarisk/`). The R/Quarto process may run as a different user that can read the files.

- **What we do now:** When opening a TIFF fails (stats or first tile), the backend returns a 500 with the real error message (e.g. `Permission denied`, `No such file`), and the frontend shows that message in the UI instead of failing silently.
- **What to do on the server:** Ensure the user/process that runs the FastAPI app has **read** access to every path listed in `dataset_url` (e.g. `chmod` or ACLs on `/Data/shinydata/infrarisk/` so the app user can read the `.tif` files).
