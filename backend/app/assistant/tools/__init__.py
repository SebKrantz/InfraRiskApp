"""The tool registry: one definition, three consumers.

Every tool is declared once with a name, a description, a JSON schema and a
side. Server tools carry a function `(conv, **args) -> dict`; client tools are
schema-only — the browser executes them against App.tsx setters. The registry
feeds the Anthropic adapter, the Gemini adapter and the MCP server.

Schemas stay in the simple dialect all three accept: object / string / number /
integer / boolean / array / enum + required. No oneOf, no additionalProperties,
no $ref.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    params: dict[str, Any]  # JSON schema for the arguments object
    side: Literal["server", "client"]
    fn: Callable[..., Any] | None  # None for client tools


# Insertion-ordered — a deterministic tool list keeps provider prompt caching
# effective across turns.
REGISTRY: dict[str, ToolSpec] = {}


def tool(
    name: str,
    description: str,
    params: dict[str, Any],
    side: Literal["server", "client"] = "server",
) -> Callable:
    """Register a tool. For client tools, decorate a placeholder returning None."""

    def register(fn: Callable | None) -> Callable | None:
        if name in REGISTRY:
            raise ValueError(f"duplicate tool {name!r}")
        REGISTRY[name] = ToolSpec(
            name=name,
            description=description,
            params=params or {"type": "object", "properties": {}},
            side=side,
            fn=fn if side == "server" else None,
        )
        return fn

    return register


def client_tool(name: str, description: str, params: dict[str, Any]) -> None:
    """Declare a browser-executed tool (schema only)."""
    tool(name, description, params, side="client")(None)


def specs(side: str | None = None) -> list[ToolSpec]:
    return [t for t in REGISTRY.values() if side is None or t.side == side]


def load() -> None:
    """Import every tool module so the registry is complete. Idempotent."""
    from . import (  # noqa: F401
        analysis,
        data,
        documents,
        guide,
        output,
        query,
        ui,
    )
