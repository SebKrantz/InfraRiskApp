"""
FastAPI backend for Hazard-Infrastructure Analyzer
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from app.api import upload, hazards, analyze, tiles, export
from app.config import settings

app = FastAPI(
    title="Hazard-Infrastructure Analyzer API",
    description="API for analyzing infrastructure assets against hazard layers",
    version="1.0.0"
)

# Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(upload.router, prefix="/api", tags=["upload"])
app.include_router(hazards.router, prefix="/api", tags=["hazards"])
app.include_router(analyze.router, prefix="/api", tags=["analyze"])
app.include_router(tiles.router, prefix="/api", tags=["tiles"])
app.include_router(export.router, prefix="/api", tags=["export"])

# Production SPA: frontend build is copied to backend/static by the Dockerfile
STATIC_DIR = Path(__file__).resolve().parent / "static"
_SPA_INDEX = STATIC_DIR / "index.html"
_SERVE_SPA = _SPA_INDEX.is_file()

if _SERVE_SPA:
    assets_dir = STATIC_DIR / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/")
async def root():
    """Serve SPA index in production; JSON stub for API-only local runs."""
    if _SERVE_SPA:
        return FileResponse(_SPA_INDEX)
    return {"message": "Hazard-Infrastructure Analyzer API", "version": "1.0.0"}


if _SERVE_SPA:
    @app.get("/{full_path:path}")
    async def spa_fallback(full_path: str):
        """Serve static files or fall back to index.html for client-side routes."""
        # Never shadow API / health (registered above; this is a safety net)
        if full_path == "health" or full_path.startswith("api/"):
            return {"message": "Hazard-Infrastructure Analyzer API", "version": "1.0.0"}
        candidate = STATIC_DIR / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_SPA_INDEX)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )

