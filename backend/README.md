# Hazard-Infrastructure Analyzer - Backend

FastAPI backend for analyzing infrastructure assets against hazard layers.

## Setup

### Prerequisites

- Python 3.9 or higher
- pip or conda

### Installation

1. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Configuration

The application uses environment variables for configuration (with defaults):

- `HOST`: Server host (default: `0.0.0.0`)
- `PORT`: Server port (default: `8000`)
- `DEBUG`: Debug mode (default: `False`)

### Data Setup

1. Create a `data` directory in the project root:
```bash
mkdir -p data
```

2. Create `data/hazard_layers.csv` with the following structure (semicolon-delimited):

```csv
hazard;dataset_url;description;background_paper
Flood Hazard 10 Years - Existing climate;https://hazards-data.unepgrid.ch/global_pc_h10glob.tif;Description text;Background paper URL
```

Required columns:
- `hazard`: Display name (used to generate unique ID)
- `dataset_url`: URL or path to Cloud Optimized GeoTIFF (COG)

Optional columns:
- `description`: Description of the hazard
- `background_paper`: URL or path to background paper/metadata

**Note:** The CSV file uses semicolon (`;`) as delimiter, not comma.

## Running the Server

### Development Mode

From the `backend` directory:

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Production Mode

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

> **Note**: Use `--workers 1` as the application uses in-memory state for uploaded files and analysis caching. Multi-worker deployment would require Redis or similar for shared state.

## API Endpoints

### Health Check
- `GET /health` - Health check endpoint
- `GET /` - API information

### Upload
- `POST /api/upload` - Upload spatial file (Shapefile, GeoPackage, or CSV with coordinates)
  - Accepts: 
    - `.shp` (zipped shapefile)
    - `.gpkg` (GeoPackage)
    - `.csv` (with point coordinates - see below)
  - Returns: File metadata with geometry type and bounds
  - **CSV Format**: CSV files must contain coordinate columns. Supported column name variations (case-agnostic):
    - `lat`/`lon`, `lat`/`lng`, `latitude`/`longitude`
    - `y`/`x` (where y = latitude, x = longitude)
    - Missing coordinates are automatically dropped
    - Coordinates are validated (lat: -90 to 90, lon: -180 to 180)

- `GET /api/upload/{file_id}` - Get upload metadata
- `DELETE /api/upload/{file_id}` - Delete uploaded file

### Hazards
- `GET /api/hazards` - Get list of available hazard layers
- `GET /api/hazards/{hazard_id}` - Get specific hazard layer info

### Analysis
- `POST /api/analyze` - Analyze intersections between infrastructure and hazards
  - Body: `{file_id, hazard_id, hazard_url, intensity_threshold?}`
  - Returns: Summary statistics (counts/meters affected)

- `GET /api/analyze/{file_id}/status` - Get analysis status

## API Documentation

Once the server is running, interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Testing the API Endpoints

### Method 1: Using Swagger UI (Recommended)

The easiest way to test the API is using the built-in Swagger UI:

1. Start the server: `python main.py`
2. Open your browser and go to: `http://localhost:8000/docs`
3. You'll see an interactive interface where you can:
   - See all available endpoints
   - Test each endpoint directly in the browser
   - See request/response schemas
   - Try out file uploads

### Method 2: Using curl Commands

You can test endpoints from the command line using `curl`:

#### Health Check
```bash
curl http://localhost:8000/health
curl http://localhost:8000/
```

#### Get Hazard Layers
```bash
# Get all hazards
curl http://localhost:8000/api/hazards

# Get specific hazard (use the generated ID from the list above)
curl http://localhost:8000/api/hazards/flood_hazard_10_years_existing_climate
```

#### Upload File
```bash
# Upload a Shapefile (zipped)
curl -X POST "http://localhost:8000/api/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/shapefile.zip;type=application/zip"

# Upload a GeoPackage
curl -X POST "http://localhost:8000/api/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/data.gpkg"

# Upload a CSV file with coordinates
# CSV should have columns like: lat,lon or latitude,longitude or y,x
curl -X POST "http://localhost:8000/api/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/points.csv;type=text/csv"

# The response will include a file_id that you'll need for analysis
```

#### Get Upload Info
```bash
# Replace {file_id} with the ID from upload response
curl http://localhost:8000/api/upload/{file_id}
```

#### Analyze Intersections
```bash
# Replace {file_id} with your uploaded file ID
# Replace {hazard_id} with a hazard ID from /api/hazards
# Replace {hazard_url} with the URL from the hazard info
curl -X POST "http://localhost:8000/api/analyze" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "your-file-id",
    "hazard_id": "flood_hazard_10_years_existing_climate",
    "hazard_url": "https://hazards-data.unepgrid.ch/global_pc_h10glob.tif",
    "intensity_threshold": 0.5
  }'
```

#### Get Analysis Status
```bash
curl http://localhost:8000/api/analyze/{file_id}/status
```

### Method 3: Using HTTPie (if installed)

HTTPie provides a more readable syntax:

```bash
# Install: pip install httpie

# Get hazards
http GET http://localhost:8000/api/hazards

# Upload file
http -f POST http://localhost:8000/api/upload file@path/to/file.gpkg

# Analyze
http POST http://localhost:8000/api/analyze \
  file_id=your-file-id \
  hazard_id=flood_hazard_10_years_existing_climate \
  hazard_url=https://hazards-data.unepgrid.ch/global_pc_h10glob.tif \
  intensity_threshold:=0.5
```

### Method 4: Using Python requests

You can also test programmatically with Python:

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Get hazards
response = requests.get("http://localhost:8000/api/hazards")
hazards = response.json()
print(hazards)

# Upload file
with open("path/to/file.gpkg", "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:8000/api/upload", files=files)
    file_data = response.json()
    file_id = file_data["file_id"]
    print(f"Uploaded file ID: {file_id}")

# Analyze
response = requests.post("http://localhost:8000/api/analyze", json={
    "file_id": file_id,
    "hazard_id": "flood_hazard_10_years_existing_climate",
    "hazard_url": "https://hazards-data.unepgrid.ch/global_pc_h10glob.tif",
    "intensity_threshold": 0.5
})
print(response.json())
```

## Project Structure

```
backend/
├── main.py                 # FastAPI application entry point
├── app/
│   ├── __init__.py
│   ├── config.py          # Configuration settings
│   ├── api/
│   │   ├── __init__.py
│   │   ├── upload.py      # File upload endpoints
│   │   ├── hazards.py     # Hazard layer endpoints
│   │   └── analyze.py     # Analysis endpoints
│   └── utils/
│       ├── __init__.py
│       └── geospatial.py  # Geospatial utility functions
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Notes

- Uploaded files and analysis results are stored in memory with intelligent caching for fast threshold adjustments.
- The tile server uses thread-local rasterio connections and tile caching for optimal performance.
- CORS is configured to allow all origins for development. Update for production.

## Troubleshooting

**Import errors with geospatial libraries:**
- On macOS, you may need: `brew install gdal`
- On Linux: `sudo apt-get install gdal-bin libgdal-dev`
- On Windows: Use conda or pre-built wheels

**Port already in use:**
- Change the port in `app/config.py` or use `--port` flag with uvicorn

