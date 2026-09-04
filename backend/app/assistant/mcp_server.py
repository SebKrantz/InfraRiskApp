"""The MCP endpoint: the assistant's server-side tools for external clients.

Mounted at /mcp (Streamable HTTP) on the same process, so Claude Code, Claude
Desktop and other agents can drive the analysis layer with the exact tools the
in-app assistant uses. Client-side ui_* tools are NOT exposed — they are
meaningless without a browser; `read_app_state` returns the frontend's last
pushed snapshot instead. External clients bring their own model, so this works
without any API key.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

from mcp.server.mcpserver import MCPServer

from . import mcp_state, tools
from .conversations import get_or_create

log = logging.getLogger("infrarisk.assistant")

# One shared scope for all external MCP traffic: stored variables, uploads and
# artifacts behave exactly as they do for one chat conversation.
_MCP_CONVERSATION_ID = "mcp-shared-scope"

_PY_TYPES = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _wrapper(spec: tools.ToolSpec):
    """A typed callable for MCPServer's schema introspection, delegating to the
    ToolSpec's fn with the shared MCP conversation."""

    def impl(**kwargs: Any) -> Any:
        conv = get_or_create(_MCP_CONVERSATION_ID)
        # omitted optionals arrive as None — let the fn's own defaults apply
        args = {k: v for k, v in kwargs.items() if v is not None}
        return spec.fn(conv, **args)

    props = spec.params.get("properties", {})
    required = set(spec.params.get("required", []))
    params = []
    annotations: dict[str, Any] = {}
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


def build() -> MCPServer:
    tools.load()
    server = MCPServer(
        "infrastructure-risk-analyzer",
        instructions=(
            "Infrastructure Risk Analyzer: exposure and damage analysis of "
            "infrastructure assets against global natural-hazard rasters "
            "(flood, tropical cyclone, earthquake, landslide, drought). Call "
            "list_hazards for the catalogue and list_datasets for the loaded "
            "infrastructure, then run_analysis. read_guide explains the model's "
            "exact semantics. Artifacts (charts, maps, reports) are downloadable "
            "from the backend at the url each result returns."
        ),
    )
    for spec in tools.specs(side="server"):
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
    log.info("mcp: %d tools registered", len(tools.specs(side="server")) + 1)
    return server
