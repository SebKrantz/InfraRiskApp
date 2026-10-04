"""
Application configuration
"""

import os
from pathlib import Path
from typing import Optional

# backend/.env (gitignored) holds the API keys: CARTO_API_KEY below and the
# assistant's keys further down. Load it before anything reads the environment
# — Settings' attributes are evaluated once, when the class is defined.
# Variables already set in the environment win over the file.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:  # python-dotenv missing; the environment alone applies
    pass

class Settings:
    """Application settings"""
    
    # Server settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    DATA_DIR: Path = BASE_DIR / "data"
    HAZARD_LAYERS_CSV: Path = DATA_DIR / "hazard_layers.csv"
    
    # Ensure directories exist
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # CARTO basemap API key — CARTO's raster basemaps (Positron / Dark Matter)
    # now require ?key=<...>. Used server-side for PNG export basemaps and, where
    # the backend serves the built SPA, substituted into index.html for the
    # interactive map. Empty = request unkeyed.
    CARTO_API_KEY: str = os.getenv("CARTO_API_KEY", "")

    # File upload settings
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100 MB
    ALLOWED_EXTENSIONS: set = {".shp", ".gpkg", ".zip", ".csv", ".geojson"}

settings = Settings()


# --------------------------------------------------------------------------- #
# AI assistant (app/assistant/)
#
# Keys live in backend/.env (loaded at the top of this file). Nothing here is
# ever serialised outward — /api/meta reports availability booleans only.
# --------------------------------------------------------------------------- #

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()

# The model table (providers, models, effort, tiers) is app/assistant/models.py. A provider
# is offered iff its key is set. These env overrides pick the default MODEL of a provider
# and count only when they name a model of its table.
ASSISTANT_MODEL_ENV = {
    "anthropic": os.environ.get("ANTHROPIC_MODEL", "").strip(),
    "gemini": os.environ.get("GEMINI_MODEL", "").strip(),
    "openai": os.environ.get("OPENAI_MODEL", "").strip(),
}
# Unset: the first keyed of anthropic, gemini, openai.
ASSISTANT_DEFAULT_PROVIDER = os.environ.get("ASSISTANT_DEFAULT_PROVIDER", "").strip()

# Tool round-trips allowed per user turn before the loop gives up.
ASSISTANT_MAX_ITERATIONS = int(os.environ.get("ASSISTANT_MAX_ITERATIONS", "30"))
# Wall-clock cap on one provider request, seconds. Without it a hung stream
# hangs the whole turn; with it the failure looks transient and is retried.
ASSISTANT_PROVIDER_TIMEOUT = float(os.environ.get("ASSISTANT_PROVIDER_TIMEOUT", "300"))
# Wall-clock cap on one server-side tool call, seconds. Remote COG reads are
# slow and occasionally never return.
ASSISTANT_TOOL_TIMEOUT = float(os.environ.get("ASSISTANT_TOOL_TIMEOUT", "600"))
# Wall-clock cap on one background job (start_run_analysis /
# start_compare_hazards over MCP), seconds.
ASSISTANT_JOB_TIMEOUT = float(os.environ.get("ASSISTANT_JOB_TIMEOUT", "3600"))
# Wall-clock cap on one python_exec call, seconds. Generous: sampling a remote
# COG over a large network is minutes of work, not seconds.
ASSISTANT_EXEC_TIMEOUT = float(os.environ.get("ASSISTANT_EXEC_TIMEOUT", "180"))
# Per-file cap on uploads to the assistant, megabytes.
ASSISTANT_UPLOAD_MAX_MB = float(os.environ.get("ASSISTANT_UPLOAD_MAX_MB", "100"))
# Per-file cap on MCP upload_file (base64 inside a JSON-RPC body), megabytes
# decoded. Larger files go through the raw-body REST route above.
ASSISTANT_MCP_UPLOAD_MAX_MB = float(os.environ.get("ASSISTANT_MCP_UPLOAD_MAX_MB", "25"))
# Mount the MCP server at /mcp (external clients bring their own model, so this
# is independent of the API keys above).
ASSISTANT_MCP_ENABLED = os.environ.get("ASSISTANT_MCP_ENABLED", "1") == "1"

