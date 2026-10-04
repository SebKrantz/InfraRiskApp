"""Gemini adapter over the google-genai SDK.

Contract (shared with the Anthropic adapter): `stream_turn(model, system,
messages, tools)` yields schema.TextDelta / ThinkingDelta / ToolCall / Usage
and finally TurnEnd. `TurnEnd.raw` carries the provider-native assistant
content and is replayed verbatim on the next turn — Gemini 3 requires the
thought signatures inside it for multi-turn function calling.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Iterator

from google import genai
from google.genai import errors, types

from ... import config
from .. import schema
from ..tools import ToolSpec

log = logging.getLogger("infrarisk.assistant")

_SCHEMA_KEYS = {"type", "description", "enum", "items", "properties", "required"}
# `effort` (models.EFFORT_LEVELS) as Gemini 3's thinking level; none leaves the model's
# default (high).
_THINKING = {
    "minimal": types.ThinkingLevel.MINIMAL,
    "low": types.ThinkingLevel.LOW,
    "medium": types.ThinkingLevel.MEDIUM,
    "high": types.ThinkingLevel.HIGH,
}
# Flex: half price, best-effort, 1-15 min latency, and no fallback of its own. A flex
# request refused for capacity (503, or 429) is backed off FLEX_RETRIES times (2, 4, 8,
# 16 s), then sent at the standard tier; the client waits up to FLEX_TIMEOUT_MS.
FLEX_RETRIES = 4
FLEX_TIMEOUT_MS = 900_000
_CAPACITY = (429, 503)
_sleep = time.sleep  # a seam for the tests


def _clean_schema(node: Any) -> Any:
    """Keep the simple JSON-schema dialect the API accepts. `properties` maps
    argument NAMES to schemas — the keyword whitelist applies around it, never
    to it."""
    if not isinstance(node, dict):
        return node
    out: dict[str, Any] = {}
    for k, v in node.items():
        if k == "properties" and isinstance(v, dict):
            out[k] = {name: _clean_schema(sub) for name, sub in v.items()}
        elif k == "items":
            out[k] = _clean_schema(v)
        elif k in _SCHEMA_KEYS:
            out[k] = v
    return out


def _declarations(tools: list[ToolSpec]) -> list[types.FunctionDeclaration]:
    return [
        types.FunctionDeclaration(
            name=t.name,
            description=t.description,
            parameters_json_schema=_clean_schema(t.params),
        )
        for t in tools
    ]


def _contents(messages: list[dict[str, Any]]) -> list[types.Content]:
    out: list[types.Content] = []
    for msg in messages:
        if (
            msg["role"] == "assistant"
            and msg.get("raw_provider") == "gemini"
            and msg.get("raw")
        ):
            out.append(msg["raw"])
            continue
        role = "model" if msg["role"] == "assistant" else "user"
        parts: list[types.Part] = []
        for p in msg["parts"]:
            if p["type"] == "text":
                if p["text"]:
                    parts.append(types.Part.from_text(text=p["text"]))
            elif p["type"] == "tool_call":
                parts.append(types.Part.from_function_call(name=p["name"], args=p["args"]))
            elif p["type"] == "tool_result":
                payload = {"result": p["content"]} if p["ok"] else {"error": p["content"]}
                parts.append(
                    types.Part.from_function_response(name=p["name"], response=payload)
                )
            elif p["type"] == "file":
                with open(p["path"], "rb") as fh:
                    parts.append(
                        types.Part.from_bytes(data=fh.read(), mime_type=p["mime"])
                    )
        if parts:
            out.append(types.Content(role=role, parts=parts))
    return out


class GeminiProvider:
    name = "gemini"

    def __init__(self) -> None:
        self.client = genai.Client(
            api_key=config.GEMINI_API_KEY,
            # Without a cap a stalled stream hangs the whole turn; a timeout
            # reads as transient and the loop retries it.
            http_options=types.HttpOptions(
                timeout=int(config.ASSISTANT_PROVIDER_TIMEOUT * 1000)
            ),
        )

    def _open(
        self, model: str, contents: list[types.Content], cfg: types.GenerateContentConfig
    ) -> Iterator[types.GenerateContentResponse]:
        """The response stream, its first chunk already read: the SDK sends the request on
        the first read, so a refusal surfaces here, before anything was shown. A flex
        request refused for capacity is backed off, then sent at the standard tier."""
        tries = 0
        while True:
            flex = cfg.service_tier == types.ServiceTier.FLEX
            stream = self.client.models.generate_content_stream(
                model=model, contents=contents, config=cfg
            )
            try:
                first = next(stream)
            except StopIteration:
                return iter(())
            except errors.APIError as exc:
                if not flex or exc.code not in _CAPACITY:
                    raise
                if tries < FLEX_RETRIES:
                    tries += 1
                    log.info(
                        "gemini flex: no capacity (%s); retrying in %.0f s (%d of %d)",
                        exc.code, 2.0**tries, tries, FLEX_RETRIES,
                    )
                    _sleep(2.0**tries)
                else:
                    log.warning(
                        "gemini flex: no capacity after %d retries; sending the request "
                        "at the standard tier", FLEX_RETRIES,
                    )
                    cfg = cfg.model_copy(update={"service_tier": None, "http_options": None})
                continue

            def chunks(first=first, stream=stream) -> Iterator[types.GenerateContentResponse]:
                yield first
                yield from stream

            return chunks()

    def stream_turn(
        self,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec],
        effort: str | None = None,
        service_tier: str | None = None,
    ) -> Iterator[schema.ProviderEvent]:
        """`effort` is the thinking level (_THINKING); `service_tier="flex"` asks for flex
        processing, with a long client timeout (FLEX_RETRIES)."""
        flex = service_tier == "flex"
        cfg = types.GenerateContentConfig(
            system_instruction=system,
            tools=[types.Tool(function_declarations=_declarations(tools))] if tools else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            thinking_config=(
                types.ThinkingConfig(thinking_level=_THINKING[effort])
                if effort in _THINKING
                else None
            ),
            service_tier=types.ServiceTier.FLEX if flex else None,
            http_options=types.HttpOptions(timeout=FLEX_TIMEOUT_MS) if flex else None,
        )
        stream = self._open(model, _contents(messages), cfg)
        collected: list[types.Part] = []
        calls: list[schema.ToolCall] = []
        usage = None
        finish = "end_turn"
        for chunk in stream:
            usage = chunk.usage_metadata or usage
            for cand in chunk.candidates or []:
                if cand.finish_reason:
                    finish = str(cand.finish_reason.name).lower()
                if not cand.content:
                    continue
                for part in cand.content.parts or []:
                    collected.append(part)
                    if part.text:
                        if part.thought:
                            yield schema.ThinkingDelta(part.text)
                        else:
                            yield schema.TextDelta(part.text)
                    if part.function_call is not None:
                        fc = part.function_call
                        calls.append(
                            schema.ToolCall(
                                id=fc.id or f"call_{uuid.uuid4().hex[:12]}",
                                name=fc.name or "",
                                args=dict(fc.args or {}),
                            )
                        )
        for call in calls:
            yield call
        if usage is not None:
            yield schema.Usage(
                input_tokens=int(usage.prompt_token_count or 0),
                output_tokens=int(
                    (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
                ),
            )
        stop = "tool_use" if calls else finish
        raw = types.Content(role="model", parts=collected) if collected else None
        yield schema.TurnEnd(stop_reason=stop, raw=raw)
