"""Tools only an external MCP caller needs: files in without a browser, and
forgetting a scope.

Kept out of the shared registry on purpose — the in-app assistant has the
browser for uploads and no business resetting its own conversation — but built
from the same ToolSpec so mcp_server.py wraps them exactly like the rest.
"""

from __future__ import annotations

import base64
import binascii
from pathlib import Path
from typing import Any

from .. import config
from . import conversations
from .conversations import Conversation, file_digest, file_kind
from .tools import ToolSpec

_SPECS: list[ToolSpec] = []


def mcp_tool(name: str, description: str, params: dict[str, Any]):
    def register(fn):
        _SPECS.append(ToolSpec(name, description, params, "server", fn))
        return fn

    return register


def specs() -> list[ToolSpec]:
    return list(_SPECS)


def _rest_route(conv: Conversation, name: str) -> str:
    return (
        f"POST /api/assistant/conversations/{conv.id}/files?filename={name} with the "
        f"file as the raw request body (up to {config.ASSISTANT_UPLOAD_MAX_MB:g} MB, "
        "same scope, same name afterwards), or multipart POST /api/upload "
        "(field 'file', up to 100 MB) for a file_id run_analysis accepts directly"
    )


@mcp_tool(
    "upload_file",
    "Put a file into this scope's uploads so the file-taking tools can use it by "
    "name: load_infrastructure(file=name), load_vulnerability_curve(file=name), "
    "read_document(file=name), or run_analysis / compare_hazards with "
    f"file_id=name. content_base64 is the whole file, base64-encoded, at most "
    f"{config.ASSISTANT_MCP_UPLOAD_MAX_MB:g} MB decoded; larger files go through "
    "the REST route the error message names. Re-uploading identical content is a "
    "no-op; different content under an existing name needs overwrite=true.",
    {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "File name, with extension."},
            "content_base64": {"type": "string", "description": "The file, base64."},
            "overwrite": {
                "type": "boolean",
                "description": "Replace an existing upload of the same name.",
            },
        },
        "required": ["name", "content_base64"],
    },
)
def upload_file(
    conv: Conversation, name: str, content_base64: str, overwrite: bool = False
) -> dict[str, Any]:
    safe = Path(name).name
    if not safe or safe in (".", ".."):
        raise ValueError(f"invalid file name {name!r}")
    cap = int(config.ASSISTANT_MCP_UPLOAD_MAX_MB * 1024 * 1024)
    too_big = (
        f"upload_file takes at most {config.ASSISTANT_MCP_UPLOAD_MAX_MB:g} MB; send "
        f"{safe!r} through the REST route instead: {_rest_route(conv, safe)}"
    )
    if len(content_base64) * 3 // 4 > cap + 3:  # cheap check before decoding
        raise ValueError(too_big)
    try:
        blob = base64.b64decode(content_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"content_base64 is not valid base64: {exc}") from exc
    if len(blob) > cap:
        raise ValueError(too_big)
    if not blob:
        raise ValueError("empty file")

    import hashlib

    digest = hashlib.sha256(blob).hexdigest()
    existing = conv.uploads.get(safe)
    if existing is not None and existing.exists():
        if file_digest(existing) == digest:
            return {"name": safe, "bytes": len(blob), "sha256": digest,
                    "kind": file_kind(safe), "unchanged": True}
        if not overwrite:
            raise ValueError(
                f"{safe!r} already exists in this scope with different content; "
                "pass overwrite=true to replace it, or upload under another name"
            )
    dest = conv.workdir / safe
    dest.write_bytes(blob)
    conv.uploads[safe] = dest
    return {"name": safe, "bytes": len(blob), "sha256": digest, "kind": file_kind(safe)}


@mcp_tool(
    "reset_scope",
    "Forget this scope: its uploads, curves, python_exec namespace and "
    "workdir, stored results and artifacts, and the datasets it loaded into the "
    "app. Call it when the chat that owns the scope is deleted. Other scopes and "
    "datasets loaded by anyone else are untouched.",
    {"type": "object", "properties": {}},
)
def reset_scope(conv: Conversation) -> dict[str, Any]:
    from ..api.analyze import clear_raster_cache_for_file
    from ..api.upload import uploaded_files

    dropped = [fid for fid in conv.datasets if uploaded_files.pop(fid, None) is not None]
    for fid in dropped:
        clear_raster_cache_for_file(fid)
    conversations.drop(conv.id)
    return {"scope": conv.id, "reset": True, "datasets_deleted": dropped}
