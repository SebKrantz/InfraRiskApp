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

Several users (app/identity.py): in multi-user mode /mcp takes AGUI's internal
calls only (the service token). Each scope is its own principal `mcp:<scope>`,
bound to the person AGUI names in X-AEI-User, whose own datasets it also reads.
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

from .. import config, identity
from ..redact import redact, redact_text
from . import artifacts, mcp_state, mcp_tools, tools
from .conversations import MAX_CONVERSATIONS, NotYours, get_or_create

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


def _headers(ctx: Context | None) -> Any:
    """The call's HTTP headers ({} for an in-process call)."""
    try:
        return (ctx.headers if ctx is not None else None) or {}
    except ValueError:  # no request context
        return {}


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
    cid = conversation_id(ctx)
    try:
        conv = get_or_create(
            cid, owner=identity.scope_principal(cid), user=identity.service_user(_headers(ctx))
        )
    except NotYours:
        raise ToolError(f"scope {cid!r} works for another user; use a scope of your own") from None
    timeout = config.ASSISTANT_TOOL_TIMEOUT

    def work() -> Any:
        with conv.hold():  # never evicted while the call runs
            return redact(jsonsafe(spec.fn(conv, **args)))

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
    except ToolError as exc:
        raise ToolError(redact_text(str(exc))) from exc
    except TypeError as exc:
        raise ToolError(redact_text(f"bad arguments for {spec.name}: {exc}")) from exc
    except Exception as exc:  # noqa: BLE001 — model-facing error, never a crash
        message = redact_text(str(exc)) or type(exc).__name__
        log.info("mcp tool %s failed: %s", spec.name, message)
        raise ToolError(message) from exc


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
get_analysis_table or get_affected_segments for per-feature results; \
expected_annual_damage integrates a family's return-period losses (flood, cyclone, \
pga) into an EAD per climate variant — a lower bound by default. A railway under \
PGA takes the default curve use_library_curve('AEI-RAIL-PGA') (a proxy), and an EAD \
above about 1 % of the replacement value a year is flagged `plausibility: high` — not \
a planning bound. read_guide explains the model's exact semantics; call it before \
vulnerability work. A report follows read_guide('reports') and read_guide('figures').

Scope: send an opaque scope id as _meta["aeilabs/scope"] or the X-AEI-Scope header \
on every call; it maps to conversation id "mcp:<scope>" (no scope = the legacy \
shared scope "mcp-shared-scope"). A scope isolates uploads, vulnerability curves, \
the python_exec namespace and workdir, stored results, jobs and artifacts \
(list_files shows only the calling scope's). Datasets: on a multi-user server \
each scope owns the datasets it loads, and a call naming its user (AGUI's \
X-AEI-User, with the service token) also sees and uses that user's own datasets \
from the web app — nobody else's; on a single-user server every scope and the \
browser UI share the loaded datasets (file_ids, list_datasets). reset_scope \
forgets a scope and deletes the datasets it loaded, never the user's own. A scope \
idle for {config.ASSISTANT_CONVERSATION_IDLE_HOURS:g} h is dropped, and the least \
recently used beyond {MAX_CONVERSATIONS} live conversations, never one with a \
call in flight.

Data in: (1) upload_file(name, content_base64) — up to \
{config.ASSISTANT_MCP_UPLOAD_MAX_MB:g} MB decoded — then load_infrastructure(file=name), \
or pass the uploaded name straight to run_analysis / compare_hazards as file_id; \
(2) larger files (≤ {config.ASSISTANT_UPLOAD_MAX_MB:g} MB) as the raw request body of \
POST /api/assistant/conversations/mcp:<scope>/files?filename=<name>, then as (1); \
(3) multipart POST /api/upload (field "file", ≤ 100 MB; send the X-AEI-Scope \
header so the scope owns it) returns a file_id run_analysis accepts directly; \
(4) load_features(geojson) for inline features; (5) load_from_url(url) for a file \
another service exports (≤ {config.ASSISTANT_UPLOAD_MAX_MB:g} MB; a multi-user \
server downloads only from the hosts it allows).

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
for progress and the result; cancel_job(job_id). expected_annual_damage reruns every \
layer of a family (9 for flood) but reuses the rasters already sampled, so on a large \
network warm them with start_compare_hazards first. There is no solve budget. python_exec runs confined to the \
scope's workdir (cwd) with the app's data/ directory read-only, no \
subprocesses, a read-only view of the datasets the scope sees, and a hard stop after \
{config.ASSISTANT_EXEC_TIMEOUT:g} s.

Artifacts (charts, maps, reports, CSV/GeoPackage exports) are served at the \
root-relative url each result returns, /api/assistant/artifacts/<id>, on this \
server's origin (on a multi-user server with the service token); tables take \
export="csv" | "gpkg" for the full data. The artifact store keeps the \
{artifacts.MAX_ARTIFACTS} most recent files.
"""


def read_app_state(ctx: Context) -> dict[str, Any]:
    """The screen of the person the call names (X-AEI-User), else the scope's own."""
    owner = identity.service_user(_headers(ctx)) or identity.scope_principal(conversation_id(ctx))
    return redact(jsonsafe(mcp_state.read(owner)))


def build() -> MCPServer:
    tools.load()
    server = MCPServer("infrastructure-risk-analyzer", instructions=INSTRUCTIONS)
    specs = tools.specs(side="server") + mcp_tools.specs()
    for spec in specs:
        server.add_tool(_wrapper(spec), name=spec.name, description=spec.description)

    server.add_tool(
        read_app_state,
        name="read_app_state",
        description=(
            "What the app's browser UI currently shows (dataset, hazard, "
            "threshold, result summary) — a snapshot the frontend pushes every "
            "few seconds; check age_seconds for staleness. On a multi-user server, "
            "the screen of the user the call names (X-AEI-User)."
        ),
    )
    log.info("mcp: %d tools registered", len(specs) + 1)
    return server
