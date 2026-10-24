"""Claude adapter over the anthropic SDK.

Thinking is left at the model's adaptive default (no `thinking` param); the
assistant's raw content blocks are replayed verbatim when continuing a tool-use
turn, which is what keeps thinking blocks legal mid-turn.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Iterator

import anthropic

from ... import config
from .. import schema
from ..tools import ToolSpec

MAX_TOKENS = 16_000


def _tool_defs(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {"name": t.name, "description": t.description, "input_schema": t.params}
        for t in tools
    ]


def _result_content(p: dict[str, Any]) -> str:
    content = p["content"]
    return content if isinstance(content, str) else json.dumps(content, default=str)


def _file_block(p: dict[str, Any]) -> dict[str, Any] | None:
    mime = p.get("mime", "")
    kind = "image" if mime.startswith("image/") else (
        "document" if mime == "application/pdf" else None
    )
    if kind is None:
        return None
    with open(p["path"], "rb") as fh:
        data = base64.standard_b64encode(fh.read()).decode()
    return {"type": kind, "source": {"type": "base64", "media_type": mime, "data": data}}


def _messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in messages:
        if (
            msg["role"] == "assistant"
            and msg.get("raw_provider") == "anthropic"
            and msg.get("raw")
        ):
            out.append({"role": "assistant", "content": msg["raw"]})
            continue
        blocks: list[dict[str, Any]] = []
        for p in msg["parts"]:
            if p["type"] == "text":
                if p["text"]:
                    blocks.append({"type": "text", "text": p["text"]})
            elif p["type"] == "tool_call":
                blocks.append(
                    {"type": "tool_use", "id": p["id"], "name": p["name"], "input": p["args"]}
                )
            elif p["type"] == "tool_result":
                blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": p["id"],
                        "content": _result_content(p),
                        **({} if p["ok"] else {"is_error": True}),
                    }
                )
            elif p["type"] == "file":
                block = _file_block(p)
                if block is not None:
                    blocks.append(block)
        if blocks:
            # Anthropic requires alternating roles; merge consecutive same-role
            # messages (e.g. a tool-result message followed by a new user turn).
            if out and out[-1]["role"] == msg["role"] and isinstance(out[-1]["content"], list):
                out[-1]["content"].extend(blocks)
            else:
                out.append({"role": msg["role"], "content": blocks})
    return out


class AnthropicProvider:
    name = "anthropic"

    def __init__(self) -> None:
        self.client = anthropic.Anthropic(
            api_key=config.ANTHROPIC_API_KEY,
            timeout=config.ASSISTANT_PROVIDER_TIMEOUT,
        )

    def stream_turn(
        self,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec],
    ) -> Iterator[schema.ProviderEvent]:
        with self.client.messages.stream(
            model=model,
            max_tokens=MAX_TOKENS,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            tools=_tool_defs(tools),
            messages=_messages(messages),
        ) as stream:
            for event in stream:
                if event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        yield schema.TextDelta(delta.text)
                    elif delta.type == "thinking_delta":
                        yield schema.ThinkingDelta(delta.thinking)
            final = stream.get_final_message()
        for block in final.content:
            if block.type == "tool_use":
                yield schema.ToolCall(id=block.id, name=block.name, args=dict(block.input))
        yield schema.Usage(
            input_tokens=int(final.usage.input_tokens),
            output_tokens=int(final.usage.output_tokens),
        )
        yield schema.TurnEnd(stop_reason=final.stop_reason or "end_turn", raw=final.content)
