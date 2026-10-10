"""
FastAPI backend for Hazard-Infrastructure Analyzer
"""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from app.api import upload, hazards, analyze, tiles, export
from app import config
from app.config import settings

log = logging.getLogger("infrarisk")

# --------------------------------------------------------------------------- #
# AI assistant (optional): if its dependencies or configuration are missing the
# router simply is not mounted and the app behaves exactly as it did before.
# --------------------------------------------------------------------------- #

try:
    from app.api import assistant_api

    _ASSISTANT_ERROR: str | None = None
except Exception as exc:  # noqa: BLE001 — the app must start without the assistant
    assistant_api = None  # type: ignore[assignment]
    _ASSISTANT_ERROR = str(exc)
    logging.getLogger("infrarisk").warning("AI assistant disabled: %s", exc)

_MCP = None
_MCP_APP = None
if assistant_api is not None and config.ASSISTANT_MCP_ENABLED:
    try:
        from app.assistant import mcp_server

        _MCP = mcp_server.build()
        _MCP_APP = _MCP.streamable_http_app(
            streamable_http_path="/mcp",
            stateless_http=True,
            json_response=True,
            # upload_file carries base64 (4/3 of the file) in the JSON body. Room
            # for more than the cap, so an oversized upload reaches the tool and
            # gets its "use the REST route" answer instead of a bare 413.
            max_request_body_size=int(
                (config.ASSISTANT_MCP_UPLOAD_MAX_MB * 2 + 16) * 1024 * 1024
            ),
        )
    except Exception:  # noqa: BLE001
        log.exception("MCP server failed to build; /mcp disabled")
        _MCP = _MCP_APP = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    if _MCP is not None:
        async with _MCP.session_manager.run():
            yield
    else:
        yield


app = FastAPI(
    title="Hazard-Infrastructure Analyzer API",
    description="API for analyzing infrastructure assets against hazard layers",
    version="1.0.0",
    lifespan=lifespan,
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
if assistant_api is not None:
    app.include_router(assistant_api.router, prefix="/api")

if _MCP_APP is not None:
    # Not a Mount: Starlette mounts only match "/mcp/..." with the trailing
    # slash, while MCP clients POST the bare "/mcp". The streamable app is a
    # single route; graft it in directly so the exact path matches.
    app.router.routes.extend(_MCP_APP.routes)

# Production SPA: frontend build is copied to backend/static by the Dockerfile
STATIC_DIR = Path(__file__).resolve().parent / "static"
_SPA_INDEX = STATIC_DIR / "index.html"
_SERVE_SPA = _SPA_INDEX.is_file()

# The built index.html carries a "__CARTO_API_KEY__" placeholder (see
# frontend/index.html). It is filled in when the page is served rather than at
# build time, so the key stays in the environment (.env) instead of being baked
# into the image, and rotating it needs a restart, not a rebuild.
_CARTO_KEY_PLACEHOLDER = '"__CARTO_API_KEY__"'
_SPA_INDEX_HTML = ""

if _SERVE_SPA:
    _SPA_INDEX_HTML = _SPA_INDEX.read_text(encoding="utf-8").replace(
        _CARTO_KEY_PLACEHOLDER, json.dumps(settings.CARTO_API_KEY)
    )

    assets_dir = STATIC_DIR / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")


def _spa_index_response() -> HTMLResponse:
    """Serve the SPA shell with the CARTO API key substituted in."""
    return HTMLResponse(_SPA_INDEX_HTML)


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "assistant": assistant_api is not None,
        "mcp": _MCP_APP is not None,
    }


@app.get("/")
async def root():
    """Serve SPA index in production; JSON stub for API-only local runs."""
    if _SERVE_SPA:
        return _spa_index_response()
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
        return _spa_index_response()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
