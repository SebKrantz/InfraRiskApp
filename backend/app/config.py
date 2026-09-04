"""
Application configuration
"""

import os
from pathlib import Path
from typing import Optional

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
    
    # File upload settings
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100 MB
    ALLOWED_EXTENSIONS: set = {".shp", ".gpkg", ".zip", ".csv", ".geojson"}

settings = Settings()


# --------------------------------------------------------------------------- #
# AI assistant (app/assistant/)
#
# Keys live in backend/.env, which is gitignored. Nothing here is ever
# serialised outward — /api/meta reports availability booleans only.
# --------------------------------------------------------------------------- #

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:  # assistant deps not installed; the app runs without it
    pass

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

# The single source of model names. A provider is offered iff its key is set;
# defaults are env-overridable so nothing here needs editing when models move on.
ASSISTANT_PROVIDERS = {
    "anthropic": {
        "label": "Claude",
        "models": ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"],
        "default": os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5"),
    },
    "gemini": {
        "label": "Gemini",
        "models": ["gemini-3.1-pro-preview", "gemini-3-flash-preview", "gemini-3.1-flash-lite"],
        "default": os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview"),
    },
}
ASSISTANT_DEFAULT_PROVIDER = os.environ.get("ASSISTANT_DEFAULT_PROVIDER", "anthropic")

# Tool round-trips allowed per user turn before the loop gives up.
ASSISTANT_MAX_ITERATIONS = int(os.environ.get("ASSISTANT_MAX_ITERATIONS", "30"))
# Wall-clock cap on one python_exec call, seconds. Generous: sampling a remote
# COG over a large network is minutes of work, not seconds.
ASSISTANT_EXEC_TIMEOUT = float(os.environ.get("ASSISTANT_EXEC_TIMEOUT", "180"))
# Per-file cap on uploads to the assistant, megabytes.
ASSISTANT_UPLOAD_MAX_MB = float(os.environ.get("ASSISTANT_UPLOAD_MAX_MB", "100"))
# Mount the MCP server at /mcp (external clients bring their own model, so this
# is independent of the API keys above).
ASSISTANT_MCP_ENABLED = os.environ.get("ASSISTANT_MCP_ENABLED", "1") == "1"

