"""
Application configuration
"""

import os
import re
from pathlib import Path


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

    # Paths — BASE_DIR is the monorepo root (parent of backend/)
    # In Docker the layout is /app/{backend,data,uploads} so BASE_DIR=/app
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    UPLOAD_DIR: Path = Path(os.getenv("UPLOAD_DIR", str(BASE_DIR / "uploads")))
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
    HAZARD_LAYERS_CSV: Path = Path(
        os.getenv("HAZARD_LAYERS_CSV", str(DATA_DIR / "hazard_layers.csv"))
    )

    # Ensure directories exist
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # CARTO basemap API key — CARTO's raster basemaps (Positron / Dark Matter)
    # now require ?key=<...>. Used server-side for PNG export basemaps and, where
    # the backend serves the built SPA, substituted into index.html for the
    # interactive map. Empty = request unkeyed.
    CARTO_API_KEY: str = os.getenv("CARTO_API_KEY", "")

    # File upload settings
    MAX_UPLOAD_SIZE: int = int(
        os.getenv("MAX_UPLOAD_SIZE", str(100 * 1024 * 1024))
    )  # 100 MB default
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

# Where Gemini runs (README, "Gemini on Vertex AI"). "vertex": Vertex AI, billed to
# VERTEX_PROJECT and authenticated by Application Default Credentials, or by VERTEX_API_KEY
# when that is set. "developer": the Gemini Developer API with GEMINI_API_KEY. Unset means
# developer, so a deployment whose .env predates this keeps working.
GEMINI_BACKEND = os.environ.get("GEMINI_BACKEND", "").strip().lower() or "developer"
VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT", "").strip()
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "").strip() or "global"
VERTEX_API_KEY = os.environ.get("VERTEX_API_KEY", "").strip()
# A service-account key file for Application Default Credentials (the server case; gcloud's
# own login on a dev machine needs no variable). Its name holds "CREDENTIAL", so the scrub at
# the end of this file takes it out of the environment before the SDK could read it: it is
# kept here and the Gemini adapter loads the file for the client itself. Made absolute now,
# because python_exec changes the working directory while it runs.
_CREDENTIALS_FILE = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
GOOGLE_APPLICATION_CREDENTIALS = (
    os.path.abspath(os.path.expanduser(_CREDENTIALS_FILE)) if _CREDENTIALS_FILE else ""
)


def gemini_configured() -> bool:
    """Whether GEMINI_BACKEND has what it needs. On Vertex that is a project (the auth is
    Application Default Credentials, which cannot be checked without a call) or
    VERTEX_API_KEY: configuration only."""
    if GEMINI_BACKEND == "vertex":
        return bool(VERTEX_PROJECT or VERTEX_API_KEY)
    return GEMINI_BACKEND == "developer" and bool(GEMINI_API_KEY)


# The model table (providers, models, effort, tiers) is app/assistant/models.py. A provider
# is offered iff its key is set (Gemini: iff gemini_configured()). These env overrides pick
# the default MODEL of a provider and count only when they name a model of its table.
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
# Conversations held in memory, browser chats and MCP scopes together: at most
# this many, each dropped after this many idle hours. One with a turn, a
# python_exec or a tool call in flight is never dropped.
ASSISTANT_MAX_CONVERSATIONS = int(os.environ.get("ASSISTANT_MAX_CONVERSATIONS", "128"))
ASSISTANT_CONVERSATION_IDLE_HOURS = float(
    os.environ.get("ASSISTANT_CONVERSATION_IDLE_HOURS", "24")
)


# --------------------------------------------------------------------------- #
# Several users (app/identity.py). The AEI_* names are shared by every AEI Labs
# platform.
# --------------------------------------------------------------------------- #

# The identity header Caddy sets after the Google sign-in (X-Auth-Request-Email).
# Empty = single-user mode: no identity is checked and AEI_LOCAL_USER owns every
# dataset and conversation, which is how the local fleet runs.
AEI_AUTH_HEADER = os.environ.get("AEI_AUTH_HEADER", "").strip()
# sha256 (hex) of the X-AEI-Edge secret Caddy adds to every request it forwards.
# Required with AEI_AUTH_HEADER: without it any container on the internal network
# could claim any identity. Only the digest is kept (python_exec runs in-process).
AEI_EDGE_SECRET_SHA256 = os.environ.get("AEI_EDGE_SECRET_SHA256", "").strip().lower()
AEI_ADMIN_EMAILS = frozenset(
    e.strip().lower() for e in os.environ.get("AEI_ADMIN_EMAILS", "").split(",") if e.strip()
)
AEI_LOCAL_USER = os.environ.get("AEI_LOCAL_USER", "").strip() or "local"
# Browser origins other than the app's own that may call the API (comma list).
# Empty: same origin only — the backend serves the SPA, and Vite proxies /api in dev.
AEI_CORS_ORIGINS = [
    o.strip() for o in os.environ.get("AEI_CORS_ORIGINS", "").split(",") if o.strip()
]

# Datasets (app/datasets.py), in both modes: dropped after this many hours unused;
# per owner at most this many and this many megabytes in memory, the least
# recently used going first.
DATASET_TTL_HOURS = float(os.environ.get("DATASET_TTL_HOURS", "24"))
DATASET_MAX_PER_OWNER = int(os.environ.get("DATASET_MAX_PER_OWNER", "10"))
DATASET_MAX_MB_PER_OWNER = float(os.environ.get("DATASET_MAX_MB_PER_OWNER", "2048"))
# Analysis caches (app/api/analyze.py), megabytes each, fair between owners.
RASTER_CACHE_MB = float(os.environ.get("RASTER_CACHE_MB", "1024"))
ANALYSIS_CACHE_MB = float(os.environ.get("ANALYSIS_CACHE_MB", "2048"))
# Heavy work (app/concurrency.py): analyses sampling rasters at once, process-wide;
# threads each of them reads raster windows with; threads for background jobs.
ANALYSIS_SLOTS = int(os.environ.get("ANALYSIS_SLOTS", "2"))
SAMPLING_THREADS = int(os.environ.get("SAMPLING_THREADS", "8"))
JOB_WORKERS = int(os.environ.get("JOB_WORKERS", "4"))


def check_identity_settings() -> None:
    """Multi-user mode needs the edge secret's digest; refuse to start without it."""
    if AEI_AUTH_HEADER and not re.fullmatch(r"[0-9a-f]{64}", AEI_EDGE_SECRET_SHA256):
        raise RuntimeError(
            "AEI_AUTH_HEADER is set but AEI_EDGE_SECRET_SHA256 is missing or is not a "
            "sha256 hex digest: refusing to start in multi-user mode without it"
        )


check_identity_settings()

# Extra Host (and matching Origin) values the MCP HTTP transport accepts, comma-separated,
# e.g. "eps-mcp:*,aei-eps-mcp:*". The MCP SDK turns DNS-rebinding protection ON whenever the
# app is built for a localhost bind, with allowed_hosts of 127.0.0.1/localhost/[::1] only, and
# then answers any other Host header with 421 Misdirected Request. That is correct for a
# desktop install, but it refuses a client that reaches this service by its container or
# service name, which is how it is addressed inside a Docker network.
#
# Empty (the default) leaves the SDK's behaviour exactly as it was.
ASSISTANT_MCP_ALLOWED_HOSTS = [
    h.strip() for h in os.environ.get("ASSISTANT_MCP_ALLOWED_HOSTS", "").split(",") if h.strip()
]

# The localhost patterns the SDK itself would use, which stay allowed either way.
_MCP_LOCAL_HOSTS = ["127.0.0.1:*", "localhost:*", "[::1]:*"]
_MCP_LOCAL_ORIGINS = ["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"]


def mcp_transport_security():
    """Transport security for the MCP HTTP app.

    Returns None when ASSISTANT_MCP_ALLOWED_HOSTS is unset, which keeps the SDK's own
    default. Otherwise keeps DNS-rebinding protection ON and widens the allow-list to the
    configured names as well as localhost.
    """
    if not ASSISTANT_MCP_ALLOWED_HOSTS:
        return None
    from mcp.server.transport_security import TransportSecuritySettings

    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=_MCP_LOCAL_HOSTS + ASSISTANT_MCP_ALLOWED_HOSTS,
        allowed_origins=_MCP_LOCAL_ORIGINS
        + [f"http://{h}" for h in ASSISTANT_MCP_ALLOWED_HOSTS],
    )


# Every key is read above. Take the secret-named variables out of the environment
# now, so python_exec (which runs in this process) and any child process find none;
# app/redact.py keeps their values to scrub them from anything a tool returns.
from .redact import scrub_environment  # noqa: E402

scrub_environment()
