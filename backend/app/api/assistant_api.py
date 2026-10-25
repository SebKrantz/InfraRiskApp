"""HTTP surface of the AI assistant: SSE chat, uploads, artifacts, meta.

The chat response is a plain-`def` StreamingResponse of Server-Sent Events.
Client-executed UI tools suspend a leg with `await_client`; the browser posts
the results back to the same endpoint to continue the turn.

Every conversation, upload, artifact and curve belongs to the signed-in user who
created it (app/identity.py); anyone else gets a 404, as for an unknown id.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

from .. import config
from ..assistant import artifacts, conversations, loop, models, schema, tools
from ..identity import current_user, reserved_conversation_id

log = logging.getLogger("infrarisk.assistant")

router = APIRouter(prefix="/assistant", tags=["assistant"])

tools.load()

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}

# Uploads the model can also look at directly (multimodal parts).
_VISION_MIMES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".pdf": "application/pdf",
}


def assistant_meta() -> dict[str, Any]:
    """The model table and which providers have a key (booleans only, never the keys)."""
    return models.meta()


@router.get("/meta")
def meta() -> dict[str, Any]:
    """What the frontend needs to decide whether to show the assistant at all, and to
    fill the model / effort / execution popover."""
    return assistant_meta()


def _pick_model(body: dict[str, Any]) -> models.Choice:
    try:
        return models.choose(
            body.get("provider"),
            body.get("model"),
            body.get("effort"),
            body.get("service_tier"),
        )
    except ValueError as exc:
        if not models.default_provider() and not body.get("provider"):
            raise HTTPException(
                503,
                "no assistant provider configured — set ANTHROPIC_API_KEY, "
                "GEMINI_API_KEY or OPENAI_API_KEY in backend/.env",
            ) from exc
        raise HTTPException(422, str(exc)) from exc


def _conversation(conversation_id: str | None, user: str) -> conversations.Conversation:
    """The user's conversation, created on first use; another user's (or, in
    multi-user mode, an MCP scope's) is a 404."""
    if conversation_id is not None and not isinstance(conversation_id, str):
        raise HTTPException(422, "conversation_id must be a string")
    if reserved_conversation_id(conversation_id):
        raise HTTPException(404, "conversation not found")
    try:
        return conversations.get_or_create(conversation_id, owner=user)
    except conversations.NotYours:
        raise HTTPException(404, "conversation not found") from None


@router.post("/chat")
async def chat(request: Request, user: str = Depends(current_user)) -> StreamingResponse:
    """Body: {conversation_id?, provider?, model?, effort?, service_tier?,
    message? | tool_results?, files?, app_state?}. Returns an SSE stream (see
    assistant/loop.py); effort and service_tier are checked against /meta's table."""
    try:
        body = await request.json()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(422, f"invalid JSON body: {exc}") from exc
    if not isinstance(body, dict):
        raise HTTPException(422, "body must be a JSON object")
    message = body.get("message")
    tool_results = body.get("tool_results")
    if (message is None) == (tool_results is None):
        raise HTTPException(422, "pass exactly one of `message` or `tool_results`")

    choice = _pick_model(body)
    conv = _conversation(body.get("conversation_id"), user)

    # Uploads attached to this message: images and PDFs become multimodal file
    # parts; everything else becomes a context block with an automatic preview,
    # so the model always knows what arrived and how to open it.
    file_parts: list[dict[str, Any]] = []
    for name in body.get("files") or []:
        path = conv.uploads.get(str(name))
        if path is None:
            continue
        mime = _VISION_MIMES.get(path.suffix.lower())
        if mime:
            file_parts.append(
                {"type": "file", "name": name, "path": str(path), "mime": mime}
            )
        else:
            from ..assistant import documents

            file_parts.append(
                {"type": "text", "text": documents.context_note(str(name), path)}
            )

    def stream() -> Iterator[str]:
        if not conv.turn_lock.acquire(timeout=2.0):
            yield schema.sse(
                "error", {"message": "another turn is still running in this conversation"}
            )
            yield schema.sse("done", {"reason": "error"})
            return
        try:
            yield from loop.run(
                conv,
                choice.provider,
                choice.model,
                user_message=message,
                file_parts=file_parts,
                tool_results=tool_results,
                app_state=body.get("app_state"),
                effort=choice.effort,
                service_tier=choice.service_tier,
            )
        finally:
            conv.turn_lock.release()

    return StreamingResponse(
        schema.iter_sse_safe(stream()),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@router.post("/conversations/{conversation_id}/files")
async def upload_file(
    conversation_id: str,
    request: Request,
    filename: str = Query(..., min_length=1, max_length=200),
    user: str = Depends(current_user),
) -> dict[str, Any]:
    """Raw request body — no multipart, so a big GeoPackage streams straight in."""
    conv = _conversation(conversation_id, user)
    blob = await request.body()
    limit = int(config.ASSISTANT_UPLOAD_MAX_MB * 1024 * 1024)
    if len(blob) > limit:
        raise HTTPException(413, f"file exceeds {config.ASSISTANT_UPLOAD_MAX_MB:g} MB")
    if not blob:
        raise HTTPException(422, "empty file")
    safe = Path(filename).name
    dest = conv.workdir / safe
    with conv.hold():
        dest.write_bytes(blob)
        conv.uploads[safe] = dest
    log.info(
        "assistant upload: %s (%d bytes) -> conversation %s", safe, len(blob), conv.id
    )
    return {
        "conversation_id": conv.id,
        "name": safe,
        "bytes": len(blob),
        "kind": dest.suffix.lstrip(".").lower() or "file",
    }


@router.get("/artifacts/{artifact_id}")
def download_artifact(artifact_id: str, user: str = Depends(current_user)):
    art = artifacts.get(artifact_id)
    mine = art is not None and conversations.get(art.conversation_id, owner=user) is not None
    if not mine or not art.path.is_file():
        raise HTTPException(404, "artifact not found (it may have been evicted)")
    return FileResponse(
        art.path,
        media_type=art.mime,
        filename=art.filename,
        content_disposition_type="inline" if art.inline else "attachment",
    )


@router.get("/datasets/{file_id}")
def dataset_geojson(file_id: str) -> JSONResponse:
    """Metadata + GeoJSON for a dataset the assistant loaded, so the browser can
    display it. (`GET /api/upload/{file_id}` cannot serve this — it tries to
    serialise the GeoDataFrame itself.)"""
    from ..api.upload import uploaded_files

    info = uploaded_files.get(file_id)
    if info is None:
        raise HTTPException(404, "dataset not found")
    gdf = info["gdf"]
    geo = gdf.__geo_interface__
    geo.pop("bbox", None)
    for feature in geo.get("features", []):
        feature.pop("bbox", None)
    return JSONResponse(
        content={
            "file_id": file_id,
            "filename": info["filename"],
            "geometry_type": info["geometry_type"],
            "feature_count": info["feature_count"],
            "crs": info["crs"],
            "bounds": info["bounds"],
            "geojson": geo,
        }
    )


@router.get("/conversations/{conversation_id}/curves/{name}")
def download_curve(
    conversation_id: str, name: str, user: str = Depends(current_user)
) -> Response:
    """The raw bytes of a parsed vulnerability curve, so the browser can hand
    the app the very same file the analysis used."""
    conv = conversations.get(conversation_id, owner=user)
    if conv is None:
        raise HTTPException(404, "conversation not found")
    curve = conv.curves.get(name)
    if curve is None:
        raise HTTPException(404, f"no curve {name!r}; loaded: {sorted(conv.curves)}")
    path: Path = curve["path"]
    if not path.is_file():
        raise HTTPException(404, "curve file is gone")
    return Response(
        content=path.read_bytes(),
        media_type="text/csv",
        headers={"Content-Disposition": f'inline; filename="{path.name}"'},
    )


@router.post("/app_state")
async def push_app_state(request: Request, user: str = Depends(current_user)) -> dict[str, Any]:
    """The frontend's debounced snapshot of what the user sees; read by the
    MCP-facing read_app_state tool, per user."""
    from ..assistant import mcp_state

    body = await request.json()
    mcp_state.update(user, body if isinstance(body, dict) else {})
    return {"ok": True}


@router.delete("/conversations/{conversation_id}")
def delete_conversation(
    conversation_id: str, user: str = Depends(current_user)
) -> dict[str, Any]:
    return {"deleted": conversations.drop(conversation_id, owner=user)}
