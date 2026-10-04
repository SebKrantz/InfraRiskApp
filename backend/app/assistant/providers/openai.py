"""GPT adapter over the openai SDK's Responses API, streamed.

Contract (shared with the Anthropic and Gemini adapters): `stream_turn(model, system,
messages, tools, effort, service_tier)` yields schema.TextDelta / ThinkingDelta /
ToolCall / Usage and finally TurnEnd. Mirrors AGUI backend/agui/providers/openai.py.

Requests are stateless (`store=False`). `TurnEnd.raw` is the response's output items
(reasoning with its encrypted content, message, function_call) serialised to JSON and
replayed verbatim, ids included, so reasoning carries across a tool-use loop. Reasoning
summaries feed the thinking panel. Not the SDK's `responses.stream()` helper: it has no
final response after `response.incomplete` (the max-tokens stop).
"""

from __future__ import annotations

import base64
import json
import logging
import time
from typing import Any, Iterator

from openai import OpenAI, RateLimitError

from ... import config
from .. import schema
from ..tools import ToolSpec

log = logging.getLogger("infrarisk.assistant")

MAX_TOKENS = 32_000
# Reasoning counts against max_output_tokens, so the deepest levels get more room.
MAX_TOKENS_FOR = {"xhigh": 64_000, "max": 64_000}
# Flex: half price, best-effort, slower. A flex request the API has no capacity for is
# refused with a 429 "Resource Unavailable" (not charged): backed off FLEX_RETRIES times
# (2, 4, 8, 16 s), then sent at the standard tier. A flex response can take minutes, so
# the client waits FLEX_TIMEOUT seconds (OpenAI's advice: 15 min).
FLEX_RETRIES = 4
FLEX_TIMEOUT = 900.0
_sleep = time.sleep  # a seam for the tests


def _tool_defs(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    # non-strict: the schemas (optional fields, defaults) are not strict-mode schemas, and
    # the Responses API normalises to strict unless told otherwise
    return [
        {
            "type": "function",
            "name": t.name,
            "description": t.description,
            "parameters": t.params,
            "strict": False,
        }
        for t in tools
    ]


def _result_output(p: dict[str, Any]) -> str:
    content = p["content"]
    text = content if isinstance(content, str) else json.dumps(content, default=str)
    text = text if text.strip() else "(the tool returned no content)"
    # function_call_output has no error flag, so a failure says so in the output itself
    return text if p["ok"] else f"[tool error] {text}"


def _file_block(p: dict[str, Any]) -> dict[str, Any] | None:
    mime = p.get("mime", "")
    if not (mime.startswith("image/") or mime == "application/pdf"):
        return None
    with open(p["path"], "rb") as fh:
        url = f"data:{mime};base64,{base64.b64encode(fh.read()).decode()}"
    if mime == "application/pdf":
        return {"type": "input_file", "filename": p.get("name") or "document.pdf", "file_data": url}
    return {"type": "input_image", "image_url": url, "detail": "auto"}


def _message(role: str, content: list[Any]) -> dict[str, Any]:
    if role == "assistant":  # assistant input is text only (output_text, not input_text)
        return {"role": "assistant", "content": "".join(content)}
    return {"role": "user", "content": content}


def _input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The history as Responses input items. A turn's parts keep their order: text and
    files gather into a message, each tool call / result is an item of its own."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        if msg["role"] == "assistant" and msg.get("raw_provider") == "openai" and msg.get("raw"):
            out.extend(msg["raw"])
            continue
        role = msg["role"]
        content: list[Any] = []
        for p in msg["parts"]:
            kind, item = p["type"], None
            if kind == "text":
                if p["text"].strip():
                    content.append(
                        p["text"] if role == "assistant" else {"type": "input_text", "text": p["text"]}
                    )
            elif kind == "file" and role == "user":
                block = _file_block(p)
                if block is not None:
                    content.append(block)
            elif kind == "tool_call":
                item = {
                    "type": "function_call",
                    "call_id": p["id"],
                    "name": p["name"],
                    "arguments": json.dumps(p.get("args") or {}),
                }
            elif kind == "tool_result":
                item = {
                    "type": "function_call_output",
                    "call_id": p["id"],
                    "output": _result_output(p),
                }
            if item is not None:
                if content:
                    out.append(_message(role, content))
                    content = []
                out.append(item)
        if content:
            out.append(_message(role, content))
    return out


def _args(arguments: str | None) -> dict[str, Any]:
    # a non-strict call can carry malformed JSON: the tool then reports bad arguments
    try:
        args = json.loads(arguments or "{}")
    except ValueError:
        return {}
    return args if isinstance(args, dict) else {}


class OpenAIProvider:
    name = "openai"

    def __init__(self) -> None:
        self.client = OpenAI(
            api_key=config.OPENAI_API_KEY, timeout=config.ASSISTANT_PROVIDER_TIMEOUT
        )

    def _create(self, kwargs: dict[str, Any]) -> Any:
        """The request. A flex request refused with a 429 that is neither an exhausted
        quota nor a rate limit with a wait ("try again in ...") is flex capacity: backed
        off, then sent again at the standard tier. Anything else is raised."""
        tries = 0
        while True:
            try:
                return self.client.responses.create(**kwargs)
            except RateLimitError as exc:
                if (
                    kwargs.get("service_tier") != "flex"
                    or "insufficient_quota" in (exc.type, exc.code)
                    or "try again in" in str(exc)
                ):
                    raise
                if tries < FLEX_RETRIES:
                    tries += 1
                    log.info(
                        "openai flex: no capacity; retrying in %.0f s (%d of %d)",
                        2.0**tries, tries, FLEX_RETRIES,
                    )
                    _sleep(2.0**tries)
                else:
                    log.warning(
                        "openai flex: no capacity after %d retries; sending the request "
                        "at the standard tier", FLEX_RETRIES,
                    )
                    kwargs = {k: v for k, v in kwargs.items() if k not in ("service_tier", "timeout")}

    def stream_turn(
        self,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec],
        effort: str | None = None,
        service_tier: str | None = None,
    ) -> Iterator[schema.ProviderEvent]:
        """`effort` is the reasoning level (models.EFFORT_LEVELS); none leaves the model's
        default. `service_tier="flex"` asks for flex processing (FLEX_RETRIES)."""
        kwargs: dict[str, Any] = dict(
            model=model,
            instructions=system,
            input=_input(messages),
            max_output_tokens=MAX_TOKENS_FOR.get(effort or "", MAX_TOKENS),
            reasoning={"summary": "auto", **({"effort": effort} if effort else {})},
            store=False,
            # stateless replay needs the encrypted reasoning
            include=["reasoning.encrypted_content"],
            stream=True,
        )
        if tools:
            kwargs["tools"] = _tool_defs(tools)
        if service_tier == "flex":  # `timeout`: this request's client timeout, not the body's
            kwargs.update(service_tier="flex", timeout=FLEX_TIMEOUT)

        final = None
        done: dict[int, Any] = {}
        thought = False
        with self._create(kwargs) as stream:
            for event in stream:
                kind = event.type
                if kind == "response.output_text.delta":
                    yield schema.TextDelta(event.delta)
                elif kind == "response.reasoning_summary_part.added" and thought:
                    yield schema.ThinkingDelta("\n\n")  # summary parts are paragraphs
                elif kind == "response.reasoning_summary_text.delta":
                    thought = True
                    yield schema.ThinkingDelta(event.delta)
                elif kind == "response.output_item.done":
                    done[event.output_index] = event.item  # final encrypted reasoning
                elif kind in ("response.completed", "response.incomplete"):
                    final = event.response
                elif kind == "response.failed":
                    err = event.response.error
                    raise RuntimeError(err.message if err else "the response failed")
                elif kind == "error":
                    raise RuntimeError(event.message)
        if final is None:
            raise RuntimeError("the stream ended without a final response")

        items = [done[i] for i in sorted(done)] or list(final.output or [])
        calls = [
            schema.ToolCall(id=it.call_id, name=it.name, args=_args(it.arguments))
            for it in items
            if it.type == "function_call"
        ]
        for call in calls:
            yield call
        usage = final.usage
        yield schema.Usage(
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        )

        refusal = "".join(
            c.refusal
            for it in items
            if it.type == "message"
            for c in it.content
            if c.type == "refusal"
        ).strip()
        reason = final.incomplete_details.reason if final.incomplete_details else None
        stop = "end_turn"
        if calls:
            stop = "tool_use"
        elif refusal or reason == "content_filter":
            stop = "refusal"
        elif reason == "max_output_tokens":
            stop = "max_tokens"
        raw = [it.model_dump(mode="json", by_alias=True, exclude_none=True) for it in items]
        while raw and raw[-1]["type"] == "reasoning":  # a cut-off turn: the API rejects
            raw.pop()  # reasoning without its next item
        yield schema.TurnEnd(stop_reason=stop, raw=raw)
