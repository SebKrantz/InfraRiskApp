"""
FastAPI backend for Hazard-Infrastructure Analyzer
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from typing import Optional
from pathlib import Path

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

STATIC_DIR = Path(__file__).resolve().parent / "static"
ASSETS_DIR = STATIC_DIR / "assets"

if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint"""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Hazard-Infrastructure Analyzer API", "version": "1.0.0"}


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/{full_path:path}", include_in_schema=False)
async def spa_fallback(full_path: str):
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="Not Found")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )

