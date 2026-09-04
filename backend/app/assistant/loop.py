"""The agent loop: provider turns, tool dispatch, SSE events.

Runs synchronously inside the chat endpoint's StreamingResponse generator.
Server tools execute here; client tools suspend the leg with an `await_client`
event and resume when the browser posts its results.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Iterator

from .. import config
from . import artifacts, schema, system_prompt, tools
from .conversations import Conversation
from .providers import get as get_provider

log = logging.getLogger("infrarisk.assistant")

MAX_RESULT_CHARS = 60_000  # cap on one serialized tool result fed to the model

# Providers rate-limit and shed load; a multi-step analysis is long enough that
# one blip should not throw away the whole turn.
TRANSIENT_MARKERS = (
    "503", "429", "500", "502", "504",
    "unavailable", "overloaded", "rate limit", "resource_exhausted",
    "timeout", "temporarily",
)
MAX_PROVIDER_RETRIES = 3
RETRY_BACKOFF_S = (2.0, 6.0, 15.0)


def _is_transient(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in TRANSIENT_MARKERS)


def _summary(name: str, ok: bool, content: Any) -> str:
    """The one-liner shown on the tool chip in the transcript."""
    if not ok:
        return str(content)[:200]
    if isinstance(content, dict):
        if "note" in content:
            return str(content["note"])[:200]
        if "artifact" in content and isinstance(content["artifact"], dict):
            return str(content["artifact"].get("filename", "artifact"))[:200]
        keys = [k for k in content if k not in ("stored_as",)][:6]
        return ", ".join(keys)
    return str(content)[:200]


def _shrink(content: Any) -> Any:
    """Bound what one tool result feeds back into the model's context."""
    try:
        blob = json.dumps(content, default=str)
    except TypeError:
        return str(content)[:MAX_RESULT_CHARS]
    if len(blob) <= MAX_RESULT_CHARS:
        return content
    return {
        "truncated": True,
        "note": (
            f"result was {len(blob)} chars; truncated — use the stored_as "
            "variable via python_exec for the full data"
        ),
        "head": blob[: MAX_RESULT_CHARS // 2],
    }


def _run_server_tool(conv: Conversation, call: schema.ToolCall) -> dict[str, Any]:
    spec = tools.REGISTRY.get(call.name)
    if spec is None or spec.fn is None:
        return {
            "id": call.id,
            "name": call.name,
            "ok": False,
            "content": f"unknown tool {call.name!r}",
        }
    try:
        result = spec.fn(conv, **call.args)
        return {"id": call.id, "name": call.name, "ok": True, "content": _shrink(result)}
    except TypeError as exc:
        # bad/missing arguments — tell the model what the schema wanted
        return {
            "id": call.id,
            "name": call.name,
            "ok": False,
            "content": f"bad arguments: {exc}",
        }
    except Exception as exc:  # noqa: BLE001 — model-facing error, never a crash
        log.info("tool %s failed: %s", call.name, exc)
        return {"id": call.id, "name": call.name, "ok": False, "content": str(exc)}


def _tool_with_pings(
    conv: Conversation, call: schema.ToolCall, holder: dict[str, Any]
) -> Iterator[str]:
    """Run one server tool in a worker and keep the SSE stream warm with comment
    pings while it grinds — sampling a remote COG runs for tens of seconds.

    Capped: a remote raster read can hang indefinitely, and one wedged tool
    would otherwise wedge the turn. On timeout the model gets an error result
    and can carry on; the worker is left to run out in the background (it holds
    no lock a later call needs).
    """
    done = threading.Event()

    def work() -> None:
        try:
            holder["res"] = _run_server_tool(conv, call)
        finally:
            done.set()

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    deadline = time.time() + config.ASSISTANT_TOOL_TIMEOUT
    while not done.wait(15.0):
        if time.time() > deadline:
            holder["res"] = {
                "id": call.id,
                "name": call.name,
                "ok": False,
                "content": (
                    f"timed out after {config.ASSISTANT_TOOL_TIMEOUT:.0f}s — the "
                    "hazard raster is unreachable or unusually slow right now. "
                    "Try a different layer, or fewer layers at once."
                ),
            }
            log.warning("tool %s timed out", call.name)
            return
        yield schema.SSE_PING


def _tool_result_message(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "role": "user",
        "parts": [
            {
                "type": "tool_result",
                "id": r["id"],
                "name": r["name"],
                "ok": r["ok"],
                "content": r["content"],
            }
            for r in results
        ],
    }


def resolve_abandoned(conv: Conversation) -> None:
    """A new user message arrived while client calls were pending: close them
    out as failed so the provider history stays legal."""
    if not conv.pending_calls:
        return
    stale = [
        {
            "id": c["id"],
            "name": c["name"],
            "ok": False,
            "content": "the user did not complete this UI action",
        }
        for c in conv.pending_calls
    ]
    conv.messages.append(_tool_result_message(conv.buffered_results + stale))
    conv.pending_calls = []
    conv.buffered_results = []


def run(
    conv: Conversation,
    provider_name: str,
    model: str,
    user_message: str | None = None,
    file_parts: list[dict[str, Any]] | None = None,
    tool_results: list[dict[str, Any]] | None = None,
    app_state: dict[str, Any] | None = None,
) -> Iterator[str]:
    """One SSE leg. Exactly one of user_message / tool_results drives it."""
    yield schema.sse(
        "start",
        {"conversation_id": conv.id, "provider": provider_name, "model": model},
    )

    if user_message is not None:
        resolve_abandoned(conv)
        parts: list[dict[str, Any]] = list(file_parts or [])
        note = system_prompt.state_note(app_state)
        parts.append({"type": "text", "text": user_message + (note or "")})
        conv.messages.append({"role": "user", "parts": parts})
    elif tool_results is not None:
        pending = {c["id"]: c for c in conv.pending_calls}
        unknown = [r for r in tool_results if r.get("id") not in pending]
        if unknown or not pending:
            yield schema.sse("error", {"message": "stale or unknown tool results"})
            yield schema.sse("done", {"reason": "error"})
            return
        client_results = [
            {
                "id": r["id"],
                "name": pending[r["id"]]["name"],
                "ok": bool(r.get("ok")),
                "content": r.get("result") if r.get("ok") else str(r.get("error", "failed")),
            }
            for r in tool_results
        ]
        # anything the browser did not answer fails closed
        answered = {r["id"] for r in client_results}
        for c in conv.pending_calls:
            if c["id"] not in answered:
                client_results.append(
                    {
                        "id": c["id"],
                        "name": c["name"],
                        "ok": False,
                        "content": "no result from the app",
                    }
                )
        conv.messages.append(_tool_result_message(conv.buffered_results + client_results))
        conv.pending_calls = []
        conv.buffered_results = []
    else:
        yield schema.sse("error", {"message": "nothing to do"})
        yield schema.sse("done", {"reason": "error"})
        return

    provider = get_provider(provider_name)
    conv.provider, conv.model = provider_name, model
    system = system_prompt.build()
    toolset = tools.specs()

    for _ in range(config.ASSISTANT_MAX_ITERATIONS):
        calls: list[schema.ToolCall] = []
        text_parts: list[str] = []
        stop = "end_turn"
        raw = None
        # Retrying is only safe while nothing has been emitted; once tokens have
        # streamed to the browser a second attempt would duplicate them.
        for attempt in range(MAX_PROVIDER_RETRIES + 1):
            failure: Exception | None = None
            try:
                for event in provider.stream_turn(
                    model=model, system=system, messages=conv.messages, tools=toolset
                ):
                    if isinstance(event, schema.TextDelta):
                        text_parts.append(event.text)
                        yield schema.sse("text", {"delta": event.text})
                    elif isinstance(event, schema.ThinkingDelta):
                        yield schema.sse("thinking", {"delta": event.text})
                    elif isinstance(event, schema.ToolCall):
                        calls.append(event)
                    elif isinstance(event, schema.Usage):
                        yield schema.sse(
                            "usage",
                            {
                                "input_tokens": event.input_tokens,
                                "output_tokens": event.output_tokens,
                            },
                        )
                    elif isinstance(event, schema.TurnEnd):
                        stop, raw = event.stop_reason, event.raw
                break
            except Exception as exc:  # noqa: BLE001 — provider errors are data
                failure = exc
            emitted = bool(text_parts or calls)
            retriable = (
                _is_transient(failure)
                and not emitted
                and attempt < MAX_PROVIDER_RETRIES
            )
            if not retriable:
                log.warning("provider error: %s", failure)
                yield schema.sse("error", {"message": f"{provider_name}: {failure}"})
                yield schema.sse("done", {"reason": "error"})
                return
            wait = RETRY_BACKOFF_S[min(attempt, len(RETRY_BACKOFF_S) - 1)]
            log.info("provider %s transient error, retrying in %.0fs: %s",
                     provider_name, wait, failure)
            yield schema.sse(
                "notice",
                {"message": f"{provider_name} is busy — retrying in {wait:.0f}s"},
            )
            deadline = time.time() + wait
            while time.time() < deadline:
                time.sleep(min(5.0, max(0.0, deadline - time.time())))
                yield schema.SSE_PING

        # record the assistant turn canonically + raw for replay
        parts = []
        if text_parts:
            parts.append({"type": "text", "text": "".join(text_parts)})
        parts.extend(
            {"type": "tool_call", "id": c.id, "name": c.name, "args": c.args} for c in calls
        )
        conv.messages.append(
            {"role": "assistant", "parts": parts, "raw": raw, "raw_provider": provider_name}
        )

        if not calls:
            yield schema.sse("done", {"reason": "end_turn" if stop != "refusal" else stop})
            return

        by_side = {
            c.id: (
                "server"
                if (spec := tools.REGISTRY.get(c.name)) and spec.side == "server"
                else "client"
            )
            for c in calls
        }
        server_calls = [c for c in calls if by_side[c.id] == "server"]
        client_calls = [c for c in calls if by_side[c.id] == "client"]

        # Invariant from here until the results message lands: every call is in
        # pending_calls or buffered_results. A cancellation mid-way leaves the
        # unanswered ones pending, and resolve_abandoned() closes them out on
        # the next user message — history stays legal for the providers.
        conv.pending_calls = [{"id": c.id, "name": c.name, "args": c.args} for c in calls]
        conv.buffered_results = []

        for call in calls:
            yield schema.sse(
                "tool_call",
                {
                    "id": call.id,
                    "name": call.name,
                    "args": call.args,
                    "side": by_side[call.id],
                },
            )
        for call in server_calls:
            before = {a.id for a in artifacts.for_conversation(conv.id)}
            holder: dict[str, Any] = {}
            yield from _tool_with_pings(conv, call, holder)
            res = holder["res"]
            conv.buffered_results.append(res)
            conv.pending_calls = [c for c in conv.pending_calls if c["id"] != call.id]
            yield schema.sse(
                "tool_result",
                {
                    "id": res["id"],
                    "name": res["name"],
                    "ok": res["ok"],
                    "summary": _summary(res["name"], res["ok"], res["content"]),
                },
            )
            for art in artifacts.for_conversation(conv.id):
                if art.id not in before:
                    yield schema.sse("artifact", art.public())

        if client_calls:
            yield schema.sse(
                "await_client",
                {
                    "calls": [
                        {"id": c.id, "name": c.name, "args": c.args} for c in client_calls
                    ],
                },
            )
            yield schema.sse("done", {"reason": "awaiting_client"})
            return

        conv.messages.append(_tool_result_message(conv.buffered_results))
        conv.pending_calls = []
        conv.buffered_results = []

    closing = (
        "I hit the tool-call limit for one turn — tell me to continue if you "
        "want me to keep going."
    )
    conv.messages.append(schema.text_message("assistant", closing))
    yield schema.sse("text", {"delta": "\n\n" + closing})
    yield schema.sse("done", {"reason": "max_iterations"})
