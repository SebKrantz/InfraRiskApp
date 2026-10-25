"""Tools only an external MCP caller needs: files in without a browser,
forgetting a scope, removing a dataset, and job handles for long analyses.

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
from . import conversations, jobs
from .conversations import Conversation, file_digest, file_kind
from .tools import REGISTRY, ToolSpec
from .tools import load as load_tools

load_tools()  # the start_* tools reuse the blocking tools' schemas

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

    jobs.cancel_all(conv.id)
    dropped = [fid for fid in conv.datasets if uploaded_files.pop(fid, None) is not None]
    for fid in dropped:
        clear_raster_cache_for_file(fid)
    conversations.drop(conv.id)
    return {"scope": conv.id, "reset": True, "datasets_deleted": dropped}


@mcp_tool(
    "delete_dataset",
    "Remove a loaded dataset from the app, with its cached analyses — the same "
    "as DELETE /api/upload/{file_id}. Datasets are shared by every scope and the "
    "browser UI, so delete only ones you loaded.",
    {
        "type": "object",
        "properties": {"file_id": {"type": "string", "description": "Dataset to remove."}},
        "required": ["file_id"],
    },
)
def delete_dataset(conv: Conversation, file_id: str) -> dict[str, Any]:
    from ..api.analyze import clear_raster_cache_for_file
    from ..api.upload import uploaded_files

    if uploaded_files.pop(file_id, None) is None:
        raise ValueError(f"no dataset {file_id!r}; loaded datasets: {list(uploaded_files) or 'none'}")
    clear_raster_cache_for_file(file_id)
    if file_id in conv.datasets:
        conv.datasets.remove(file_id)
    return {"deleted": file_id}


def _estimate(conv: Conversation, file_id: str, hazards: list[str]) -> float:
    """Rough seconds: a first read of a remote layer costs tens of seconds, a
    layer already sampled for this dataset about one."""
    from ..api.analyze import get_cached_raster_values
    from . import domain

    total = 0.0
    for ref in hazards:
        try:
            hid = domain.resolve_hazard(ref)["hazard_id"]
        except ValueError:
            continue
        total += 1.0 if get_cached_raster_values(file_id, hid) is not None else 20.0
    return total


@mcp_tool(
    "start_compare_hazards",
    "compare_hazards as a background job: same arguments, returns {job_id, "
    "estimate_s} at once. Poll get_job(job_id) for progress (one step per layer) "
    "and the result — the same result compare_hazards returns; cancel_job stops "
    f"it before its next layer. A job may run up to {config.ASSISTANT_JOB_TIMEOUT:g} s.",
    REGISTRY["compare_hazards"].params,
)
def start_compare_hazards(conv: Conversation, file_id: str, hazards: list[str], **args: Any) -> dict[str, Any]:
    from .tools.analysis import compare_hazards
    from .tools.data import resolve_dataset

    if not hazards or len(hazards) > 12:
        raise ValueError(f"pass 1 to 12 hazards, got {len(hazards or [])}")
    fid = resolve_dataset(conv, file_id)
    estimate = _estimate(conv, fid, hazards)
    job = jobs.start(
        conv,
        "compare_hazards",
        lambda job: compare_hazards(
            conv, fid, hazards, **args, _budget_s=0.9 * config.ASSISTANT_JOB_TIMEOUT, _job=job
        ),
        estimate,
    )
    return {"job_id": job.id, "estimate_s": estimate, "status": job.status}


@mcp_tool(
    "start_run_analysis",
    "run_analysis as a background job: same arguments, returns {job_id, "
    "estimate_s} at once; poll get_job(job_id) for the result — the same result "
    "run_analysis returns. For large datasets on a layer not yet sampled.",
    REGISTRY["run_analysis"].params,
)
def start_run_analysis(conv: Conversation, file_id: str, hazard: str, **args: Any) -> dict[str, Any]:
    from .tools.analysis import run_analysis
    from .tools.data import resolve_dataset

    fid = resolve_dataset(conv, file_id)
    estimate = _estimate(conv, fid, [hazard])
    job = jobs.start(
        conv, "run_analysis", lambda job: run_analysis(conv, fid, hazard, **args), estimate
    )
    return {"job_id": job.id, "estimate_s": estimate, "status": job.status}


@mcp_tool(
    "get_job",
    "Status of a background job started in this scope: status (queued, "
    "running, done, error, cancelled), progress 0-1, a message, elapsed and "
    "estimated seconds, and — once done — the tool's result.",
    {
        "type": "object",
        "properties": {"job_id": {"type": "string"}},
        "required": ["job_id"],
    },
)
def get_job(conv: Conversation, job_id: str) -> dict[str, Any]:
    return jobs.get(conv, job_id).public()


@mcp_tool(
    "cancel_job",
    "Cancel a background job of this scope. compare_hazards stops before its "
    "next layer and keeps the layers it finished; a run_analysis already "
    "sampling finishes in the background but its result is dropped.",
    {
        "type": "object",
        "properties": {"job_id": {"type": "string"}},
        "required": ["job_id"],
    },
)
def cancel_job(conv: Conversation, job_id: str) -> dict[str, Any]:
    job = jobs.get(conv, job_id)
    if job.status not in ("queued", "running"):
        return {"cancelled": False, "status": job.status}
    job.cancel.set()
    return {"cancelled": True, "status": job.status}
