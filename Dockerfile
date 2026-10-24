# syntax=docker/dockerfile:1
# Multi-stage build: React frontend → FastAPI backend serving static SPA

# ---------------------------------------------------------------------------
# Stage 1: build frontend
# ---------------------------------------------------------------------------
FROM node:20-bookworm-slim AS frontend-build

WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---------------------------------------------------------------------------
# Stage 2: Python runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    DEBUG=false \
    # Repo-root layout inside the image (matches backend/app/config.py BASE_DIR)
    DATA_DIR=/app/data \
    HAZARD_LAYERS_CSV=/app/data/hazard_layers.csv \
    UPLOAD_DIR=/app/uploads \
    GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR \
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=.tif,.tiff,.vrt \
    GDAL_HTTP_MERGE_CONSECUTIVE_RANGES=YES \
    GDAL_HTTP_MULTIPLEX=YES \
    GDAL_CACHEMAX=512

# System libs for rasterio / geopandas / matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
        gdal-bin \
        libgdal-dev \
        libgeos-dev \
        libproj-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better layer caching)
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Application code
COPY backend/ /app/backend/
# Hazard catalog CSV (rasters are mounted at runtime — not baked into the image)
COPY data/hazard_layers.csv /app/data/hazard_layers.csv

# Frontend build → served by FastAPI from backend/static
COPY --from=frontend-build /frontend/dist /app/backend/static

RUN mkdir -p /app/uploads /app/data/rasters \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app

USER appuser

WORKDIR /app/backend
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

# Single worker required: uploads + analysis caches live in process memory
CMD ["sh", "-c", "uvicorn main:app --host ${HOST} --port ${PORT} --workers 1"]
