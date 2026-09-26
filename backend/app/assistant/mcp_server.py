"""The MCP endpoint: the assistant's server-side tools for external clients.

Mounted at /mcp (Streamable HTTP) on the same process, so Claude Code, Claude
Desktop and other agents can drive the analysis layer with the exact tools the
in-app assistant uses. Client-side ui_* tools are NOT exposed — they are
meaningless without a browser; `read_app_state` returns the frontend's last
pushed snapshot instead. External clients bring their own model, so this works
without any API key.

Implements the AEI Labs MCP integration contract v1 (v1.1): every failure
reaches the caller as a ToolError with its own message, each caller gets its
own scope, every call is bounded by a server-side timeout, and the MCP-only
tools in mcp_tools.py (upload_file, reset_scope, jobs, delete_dataset) fill in
what a browserless caller needs.
"""

from __future__ import annotations

import inspect
import logging
import math
from pathlib import Path
from typing import Any

import anyio
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from .. import config
from . import mcp_state, mcp_tools, tools
from .conversations import MAX_CONVERSATIONS, get_or_create

log = logging.getLogger("infrarisk.assistant")

# Callers that send no scope share this one, as every caller did before v1.
_MCP_CONVERSATION_ID = "mcp-shared-scope"
SCOPE_META = "aeilabs/scope"
SCOPE_HEADER = "x-aei-scope"

# How often a blocking call tells the caller it is still alive.
_HEARTBEAT_S = 10.0

_PY_TYPES = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def conversation_id(ctx: Context | None) -> str:
    """`mcp:<scope>` from `_meta["aeilabs/scope"]` or the X-AEI-Scope header."""
    scope = None
    if ctx is not None:
        try:
            scope = (ctx.request_context.meta or {}).get(SCOPE_META)
        except ValueError:  # no request context (in-process call)
            scope = None
        if not scope:
            scope = (ctx.headers or {}).get(SCOPE_HEADER)
    scope = str(scope).strip()[:128] if scope else ""
    return f"mcp:{scope}" if scope else _MCP_CONVERSATION_ID


def jsonsafe(value: Any) -> Any:
    """numpy scalars/arrays, paths and non-finite floats into plain JSON."""
    if isinstance(value, dict):
        return {str(k): jsonsafe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonsafe(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist") and type(value).__module__.startswith("numpy"):
        return jsonsafe(value.tolist())
    if hasattr(value, "isoformat"):  # dates, pandas Timestamps
        return value.isoformat()
    return value


async def _run(spec: tools.ToolSpec, ctx: Context | None, args: dict[str, Any]) -> Any:
    """One tool call: in the caller's scope, on a worker thread, bounded by the
    tool timeout, with heartbeats, and every failure turned into a ToolError
    that carries its own message (the SDK redacts any other exception)."""
    conv = get_or_create(conversation_id(ctx))
    timeout = config.ASSISTANT_TOOL_TIMEOUT

    def work() -> Any:
        return jsonsafe(spec.fn(conv, **args))

    async def heartbeat() -> None:
        waited = 0.0
        while True:
            await anyio.sleep(_HEARTBEAT_S)
            waited += _HEARTBEAT_S
            try:
                await ctx.report_progress(waited, None, f"{spec.name} running ({waited:.0f}s)")
            except Exception:  # noqa: BLE001 — progress is best effort
                pass

    # Raised outside the task group, which would wrap it in an ExceptionGroup.
    outcome: dict[str, Any] = {}
    async with anyio.create_task_group() as tg:
        if ctx is not None:
            tg.start_soon(heartbeat)
        try:
            with anyio.fail_after(timeout):
                outcome["value"] = await anyio.to_thread.run_sync(work, abandon_on_cancel=True)
        except Exception as exc:  # noqa: BLE001 — mapped below; cancellation passes
            outcome["error"] = exc
        finally:
            tg.cancel_scope.cancel()
    if "value" in outcome:
        return outcome["value"]
    try:
        raise outcome["error"]
    except TimeoutError as exc:
        raise ToolError(
            f"{spec.name} timed out after {timeout:.0f}s (server-side limit); the work "
            "may still finish in the background. Use start_run_analysis / "
            "start_compare_hazards with get_job for long analyses, or split the call."
        ) from exc
    except ToolError:
        raise
    except TypeError as exc:
        raise ToolError(f"bad arguments for {spec.name}: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 — model-facing error, never a crash
        log.info("mcp tool %s failed: %s", spec.name, exc)
        raise ToolError(str(exc) or type(exc).__name__) from exc


def _wrapper(spec: tools.ToolSpec):
    """A typed coroutine for MCPServer's schema introspection, delegating to the
    ToolSpec's fn with the caller's conversation."""

    async def impl(ctx: Context, **kwargs: Any) -> Any:
        # omitted optionals arrive as None — let the fn's own defaults apply
        args = {k: v for k, v in kwargs.items() if v is not None}
        return await _run(spec, ctx, args)

    props = spec.params.get("properties", {})
    required = set(spec.params.get("required", []))
    params = [
        inspect.Parameter("ctx", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=Context)
    ]
    annotations: dict[str, Any] = {"ctx": Context}
    for name, prop in props.items():
        py = _PY_TYPES.get(prop.get("type", "string"), str)
        annotations[name] = py
        params.append(
            inspect.Parameter(
                name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=py,
                default=inspect.Parameter.empty if name in required else None,
            )
        )
    # Signature demands non-default parameters first; JSON schemas do not care.
    params.sort(key=lambda q: q.default is not inspect.Parameter.empty)
    impl.__signature__ = inspect.Signature(params)  # type: ignore[attr-defined]
    impl.__annotations__ = annotations
    impl.__name__ = spec.name
    impl.__doc__ = spec.description
    return impl


INSTRUCTIONS = f"""\
Infrastructure Risk Analyzer — exposure and damage analysis of infrastructure \
assets (points and lines) against global natural-hazard rasters (flood, tropical \
cyclone, earthquake, landslide, drought). Implements AEI MCP contract v1 (v1.1).

Workflow: list_hazards → get data in (below) → run_analysis / compare_hazards → \
get_analysis_table or get_affected_segments for per-feature results. read_guide \
explains the model's exact semantics; call it before vulnerability work.

Scope: send an opaque scope id as _meta["aeilabs/scope"] or the X-AEI-Scope header \
on every call; it maps to conversation id "mcp:<scope>" (no scope = the legacy \
shared scope "mcp-shared-scope"). A scope isolates uploads, vulnerability curves, \
the python_exec namespace and workdir, stored results, jobs and artifacts \
(list_files shows only the calling scope's). NOT isolated, shared by every scope \
and by the browser UI: the loaded datasets (file_ids, list_datasets) and the \
analysis/raster caches. reset_scope forgets a scope and deletes the datasets it \
loaded. Scopes are LRU-evicted beyond {MAX_CONVERSATIONS} live conversations.

Data in: (1) upload_file(name, content_base64) — up to \
{config.ASSISTANT_MCP_UPLOAD_MAX_MB:g} MB decoded — then load_infrastructure(file=name), \
or pass the uploaded name straight to run_analysis / compare_hazards as file_id; \
(2) larger files (≤ {config.ASSISTANT_UPLOAD_MAX_MB:g} MB) as the raw request body of \
POST /api/assistant/conversations/mcp:<scope>/files?filename=<name>, then as (1); \
(3) multipart POST /api/upload (field "file", ≤ 100 MB) returns a file_id \
run_analysis accepts directly; (4) load_features(geojson) for inline features; \
(5) load_from_url(url) for a file another service exports (≤ \
{config.ASSISTANT_UPLOAD_MAX_MB:g} MB).

Units and values: thresholds are in the layer's own units (flood mm, PGA cm/s², \
cyclone km/h, landslide class). replacement_value is per feature for points and \
per metre for lines — one scalar, or per feature via replacement_value_column \
(numbers) / replacement_value_map (keyed by an attribute such as asset type). \
id_column carries your own feature ids into the result tables. Hazard layers are remote COGs: expect 5-60 s per layer \
for the first read of a dataset, near-instant re-thresholding after that.

Limits: every call has a {config.ASSISTANT_TOOL_TIMEOUT:g} s server-side timeout; \
compare_hazards stops starting new layers when its time budget runs out and \
reports the rest as not run. Long work: start_run_analysis / \
start_compare_hazards return {{job_id, estimate_s}} at once; poll get_job(job_id) \
for progress and the result; cancel_job(job_id). There is no solve budget. python_exec runs confined to the \
scope's workdir (cwd) with the app's data/ directory read-only, no \
subprocesses, a read-only view of the dataset store, and a hard stop after \
{config.ASSISTANT_EXEC_TIMEOUT:g} s.

Artifacts (charts, maps, reports, CSV/GeoPackage exports) are served at the \
root-relative url each result returns, /api/assistant/artifacts/<id>, on this \
server's origin; tables take export="csv" | "gpkg" for the full data. The \
artifact store keeps the 64 most recent files.
"""


def build() -> MCPServer:
    tools.load()
    server = MCPServer("infrastructure-risk-analyzer", instructions=INSTRUCTIONS)
    specs = tools.specs(side="server") + mcp_tools.specs()
    for spec in specs:
        server.add_tool(_wrapper(spec), name=spec.name, description=spec.description)

    def read_app_state() -> dict[str, Any]:
        return mcp_state.read()

    server.add_tool(
        read_app_state,
        name="read_app_state",
        description=(
            "What the app's browser UI currently shows (dataset, hazard, "
            "threshold, result summary) — a snapshot the frontend pushes every "
            "few seconds; check age_seconds for staleness."
        ),
    )
    log.info("mcp: %d tools registered", len(specs) + 1)
    return server
