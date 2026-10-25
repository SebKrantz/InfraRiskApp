"""Gemini adapter over the google-genai SDK.

Contract (shared with the Anthropic adapter): `stream_turn(model, system,
messages, tools)` yields schema.TextDelta / ThinkingDelta / ToolCall / Usage
and finally TurnEnd. `TurnEnd.raw` carries the provider-native assistant
content and is replayed verbatim on the next turn — Gemini 3 requires the
thought signatures inside it for multi-turn function calling.

The same code serves both backends (config.GEMINI_BACKEND, README "Gemini on Vertex AI"):
only the client (make_client) and the way flex is asked for (_flex_options) differ.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Iterator

import google.auth
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
# Vertex AI's flex is Flex PayGo, asked for with this header (its docs name no service_tier),
# at the global endpoint only, for the models below (docs.cloud.google.com/vertex-ai/
# generative-ai/docs/flex-paygo, read 2026-10-13). With no Provisioned Throughput, which the
# header would use first, the request runs at flex.
VERTEX_FLEX_HEADERS = {"X-Vertex-AI-LLM-Shared-Request-Type": "flex"}
VERTEX_FLEX_MODELS = frozenset({
    "gemini-3.8-flash-cyber", "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash",
    "gemini-3.5-flash-lite", "gemini-3.1-flash-lite-image", "gemini-3-pro-image",
    "gemini-3.1-flash-image", "gemini-3.5-flash", "gemini-3.1-flash-lite",
    "gemini-3.1-pro-preview", "gemini-3-flash-preview",
})
_CLOUD_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


def make_client() -> genai.Client:
    """The client for config.GEMINI_BACKEND. `enterprise` is always passed, so the SDK never
    picks the backend from GOOGLE_GENAI_USE_ENTERPRISE / _VERTEXAI in the environment.
    On both backends the client carries the provider timeout: without a cap a stalled stream
    hangs the whole turn, and a timeout reads as transient, so the loop retries it."""
    http_options = types.HttpOptions(timeout=int(config.ASSISTANT_PROVIDER_TIMEOUT * 1000))
    if config.GEMINI_BACKEND == "vertex":
        if config.VERTEX_API_KEY:
            # Google documents the key without project and location (global endpoint, the
            # key's project billed); SDK >= 2.15 would take all three, older ones refuse
            return genai.Client(enterprise=True, api_key=config.VERTEX_API_KEY,
                                http_options=http_options)
        if not config.VERTEX_PROJECT:   # else the SDK would look for one in ADC / the env
            raise ValueError("GEMINI_BACKEND=vertex needs VERTEX_PROJECT or VERTEX_API_KEY")
        # Application Default Credentials, loaded by the SDK on the first request. The
        # service-account file named by GOOGLE_APPLICATION_CREDENTIALS is loaded here instead:
        # the environment variable itself is scrubbed at startup (app/redact.py), so the SDK's
        # own lookup would no longer find it. gcloud's login needs neither.
        extra: dict[str, Any] = {}
        if config.GOOGLE_APPLICATION_CREDENTIALS:
            extra["credentials"], _ = google.auth.load_credentials_from_file(
                config.GOOGLE_APPLICATION_CREDENTIALS, scopes=_CLOUD_SCOPES)
        return genai.Client(enterprise=True, project=config.VERTEX_PROJECT,
                            location=config.VERTEX_LOCATION, http_options=http_options,
                            **extra)
    if config.GEMINI_BACKEND == "developer":
        return genai.Client(enterprise=False, api_key=config.GEMINI_API_KEY,
                            http_options=http_options)
    raise ValueError(f"GEMINI_BACKEND must be vertex or developer, not {config.GEMINI_BACKEND!r}")


def _flex_options(model: str) -> dict[str, Any] | None:
    """The GenerateContentConfig fields that ask for flex with the long client timeout, or
    None (with a warning) when this backend cannot serve `model` at flex: the standard tier."""
    if config.GEMINI_BACKEND != "vertex":
        return {"service_tier": types.ServiceTier.FLEX,
                "http_options": types.HttpOptions(timeout=FLEX_TIMEOUT_MS)}
    if model not in VERTEX_FLEX_MODELS:
        log.warning("gemini flex: %s has no Flex PayGo on Vertex AI; sending the request at "
                    "the standard tier", model)
        return None
    if config.VERTEX_LOCATION != "global" and not config.VERTEX_API_KEY:   # a key: global
        log.warning("gemini flex: Flex PayGo needs VERTEX_LOCATION=global, not %s; sending "
                    "the request at the standard tier", config.VERTEX_LOCATION)
        return None
    return {"http_options": types.HttpOptions(timeout=FLEX_TIMEOUT_MS,
                                              headers=dict(VERTEX_FLEX_HEADERS))}


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
        self.client = make_client()

    def _open(
        self,
        model: str,
        contents: list[types.Content],
        cfg: types.GenerateContentConfig,
        flex: bool,
    ) -> Iterator[types.GenerateContentResponse]:
        """The response stream, its first chunk already read: the SDK sends the request on
        the first read, so a refusal surfaces here, before anything was shown. `flex` says
        whether `cfg` asks for flex (the field for it differs by backend, _flex_options). A
        flex request refused for capacity is backed off, then sent at the standard tier: no
        service_tier, no Vertex header, no long timeout."""
        tries = 0
        while True:
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
                    flex = False
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
        processing, with a long client timeout (FLEX_RETRIES), where the backend has it
        (_flex_options)."""
        tier = _flex_options(model) if service_tier == "flex" else None
        cfg = types.GenerateContentConfig(
            system_instruction=system,
            tools=[types.Tool(function_declarations=_declarations(tools))] if tools else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            thinking_config=(
                types.ThinkingConfig(thinking_level=_THINKING[effort])
                if effort in _THINKING
                else None
            ),
            **(tier or {}),
        )
        stream = self._open(model, _contents(messages), cfg, tier is not None)
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
