"""Canonical message and event shapes shared by the loop, providers and API.

Messages are plain dicts (this codebase keeps Pydantic to request bodies, and
the loop needs no validation beyond what the providers enforce):

    {"role": "user" | "assistant",
     "parts": [
        {"type": "text", "text": str},
        {"type": "tool_call", "id": str, "name": str, "args": dict},
        {"type": "tool_result", "id": str, "name": str, "ok": bool, "content": Any},
        {"type": "file", "name": str, "path": str, "mime": str},
     ],
     # assistant turns only: the provider-native content blocks, replayed
     # verbatim when continuing a tool-use turn on the same provider (required
     # for Anthropic thinking blocks; harmless for Gemini).
     "raw": Any, "raw_provider": str}

Tool results travel in a *user*-role message — both providers model it that way.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterator


def text_message(role: str, text: str) -> dict[str, Any]:
    return {"role": role, "parts": [{"type": "text", "text": text}]}


def message_text(msg: dict[str, Any]) -> str:
    """All text parts of one message, joined."""
    return "".join(p["text"] for p in msg["parts"] if p["type"] == "text")


# ---- provider stream events ------------------------------------------------ #


@dataclass
class TextDelta:
    text: str


@dataclass
class ThinkingDelta:
    text: str


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int


@dataclass
class TurnEnd:
    stop_reason: str
    raw: Any = None  # provider-native assistant content, for verbatim replay


ProviderEvent = TextDelta | ThinkingDelta | ToolCall | Usage | TurnEnd


# ---- SSE encoding ---------------------------------------------------------- #


def sse(event: str, data: dict[str, Any]) -> str:
    """One Server-Sent Event. `default=str` so numpy scalars never break a stream."""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


SSE_PING = ": ping\n\n"


def iter_sse_safe(gen: Iterator[str]) -> Iterator[str]:
    """Wrap a stream generator so an unexpected exception ends the stream with
    a well-formed error + done pair instead of a truncated response."""
    try:
        yield from gen
    except GeneratorExit:  # client disconnected — normal cancellation path
        raise
    except Exception as exc:  # noqa: BLE001 — last-resort stream guard
        yield sse("error", {"message": f"internal error: {exc}"})
        yield sse("done", {"reason": "error"})
