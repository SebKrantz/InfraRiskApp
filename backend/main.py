"""
FastAPI backend for Hazard-Infrastructure Analyzer
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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
            streamable_http_path="/mcp", stateless_http=True, json_response=True
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


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Hazard-Infrastructure Analyzer API", "version": "1.0.0"}


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "assistant": assistant_api is not None,
        "mcp": _MCP_APP is not None,
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
