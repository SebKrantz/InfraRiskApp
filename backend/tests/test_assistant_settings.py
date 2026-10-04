"""The assistant's model table, /api/assistant/meta, request validation and the adapters'
effort / flex / backoff behaviour. No network: the SDK clients are replaced by fakes.

Run from backend/ with the repo's venv:  python -m unittest discover -s tests
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx2  # noqa: E402  (the openai SDK's HTTP layer)
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from google.genai import errors as genai_errors  # noqa: E402
from google.genai import types  # noqa: E402
from openai import RateLimitError  # noqa: E402

from app import config  # noqa: E402
from app.api import assistant_api  # noqa: E402
from app.assistant import loop, models, schema  # noqa: E402
from app.assistant.providers import anthropic as anthropic_provider  # noqa: E402
from app.assistant.providers import gemini as gemini_provider  # noqa: E402
from app.assistant.providers import openai as openai_provider  # noqa: E402

ALL_KEYS = {"ANTHROPIC_API_KEY": "a", "GEMINI_API_KEY": "g", "OPENAI_API_KEY": "o"}


def with_keys(**keys: str):
    """Patch the three API keys (unnamed ones become empty)."""
    return mock.patch.multiple(
        config, **{k: keys.get(k, "") for k in ALL_KEYS}, ASSISTANT_DEFAULT_PROVIDER="",
        ASSISTANT_MODEL_ENV={"anthropic": "", "gemini": "", "openai": ""},
    )


class ModelTable(unittest.TestCase):
    def test_models_defaults_and_order(self):
        table = {p: (s["models"], s["default"]) for p, s in models.PROVIDERS.items()}
        self.assertEqual(list(table), ["anthropic", "gemini", "openai"])
        self.assertEqual(
            table["anthropic"],
            (["claude-opus-5-5", "claude-fable-5-1", "claude-sonnet-5-5", "claude-haiku-4-5"],
             "claude-sonnet-5-5"),
        )
        self.assertEqual(
            table["gemini"],
            (["gemini-3.8-flash", "gemini-3.1-pro-preview", "gemini-3.5-flash-lite"],
             "gemini-3.8-flash"),
        )
        self.assertEqual(
            table["openai"], (["gpt-6-luna", "gpt-6.1-sol", "gpt-6-astra"], "gpt-6-luna")
        )
        self.assertEqual([s["label"] for s in models.PROVIDERS.values()],
                         ["Claude", "Gemini", "OpenAI"])

    def test_effort_levels_and_defaults(self):
        five = ["low", "medium", "high", "xhigh", "max"]
        three = ["low", "medium", "high"]  # 3.8 Flash and 3.1 Pro answer `minimal` with a 400
        cases = {
            ("anthropic", "claude-opus-5-5"): (five, "medium"),
            ("anthropic", "claude-fable-5-1"): (five, "high"),
            ("anthropic", "claude-sonnet-5-5"): (five, "high"),
            ("anthropic", "claude-haiku-4-5"): ([], None),
            ("gemini", "gemini-3.8-flash"): (three, "high"),
            ("gemini", "gemini-3.5-flash-lite"): (["minimal", *three], "high"),
            ("gemini", "gemini-3.1-pro-preview"): (three, "high"),
            ("openai", "gpt-6-luna"): (five, "high"),
            ("openai", "gpt-6.1-sol"): (five, "medium"),
            ("openai", "gpt-6-astra"): (five, "medium"),
        }
        for (provider, model), (levels, default) in cases.items():
            with self.subTest(model=model):
                self.assertEqual(models.effort_levels(provider, model), levels)
                self.assertEqual(models.effort_default(provider, model), default)

    def test_tiers(self):
        self.assertEqual(models.service_tiers("anthropic"), ["standard"])
        self.assertEqual(models.service_tiers("gemini"), ["standard", "flex"])
        self.assertEqual(models.service_tiers("openai"), ["standard", "flex"])

    def test_default_provider(self):
        with with_keys(**ALL_KEYS):
            self.assertEqual(models.default_provider(), "anthropic")
        with with_keys(GEMINI_API_KEY="g", OPENAI_API_KEY="o"):
            self.assertEqual(models.default_provider(), "gemini")
        with with_keys(OPENAI_API_KEY="o"):
            self.assertEqual(models.default_provider(), "openai")
        with with_keys():
            self.assertIsNone(models.default_provider())
        with with_keys(**ALL_KEYS), mock.patch.object(config, "ASSISTANT_DEFAULT_PROVIDER", "openai"):
            self.assertEqual(models.default_provider(), "openai")
        # an env default naming an unkeyed provider falls back to the first keyed one
        with with_keys(GEMINI_API_KEY="g"), mock.patch.object(
            config, "ASSISTANT_DEFAULT_PROVIDER", "openai"
        ):
            self.assertEqual(models.default_provider(), "gemini")

    def test_model_env_override_counts_only_inside_the_table(self):
        with mock.patch.object(
            config, "ASSISTANT_MODEL_ENV",
            {"anthropic": "claude-sonnet-5", "gemini": "gemini-3.1-pro-preview", "openai": ""},
        ):
            self.assertEqual(models.default_model("anthropic"), "claude-sonnet-5-5")  # stale ID
            self.assertEqual(models.default_model("gemini"), "gemini-3.1-pro-preview")
            self.assertEqual(models.default_model("openai"), "gpt-6-luna")

    def test_ignored_model_override_warns_once_per_variable_and_value(self):
        env = {"anthropic": "", "gemini": "gemini-3-flash-preview", "openai": ""}
        models._warned.clear()
        with mock.patch.object(config, "ASSISTANT_MODEL_ENV", env):
            with self.assertLogs("infrarisk.assistant", level="WARNING") as cm:
                for _ in range(3):  # /meta asks on every request: still one line
                    self.assertEqual(models.default_model("gemini"), "gemini-3.8-flash")
            self.assertEqual(len(cm.records), 1)
            msg = cm.records[0].getMessage()
            self.assertIn("GEMINI_MODEL='gemini-3-flash-preview' is not one of Gemini's models", msg)
            self.assertIn("gemini-3.8-flash, gemini-3.1-pro-preview", msg)
            self.assertTrue(msg.endswith("using gemini-3.8-flash"))
            # another value is another line; a valid or empty override is silent
            env["gemini"] = "gemini-2.5-pro"
            with self.assertLogs("infrarisk.assistant", level="WARNING") as cm:
                models.default_model("gemini")
            self.assertEqual(len(cm.records), 1)
            env["gemini"] = "gemini-3.5-flash-lite"
            with self.assertNoLogs("infrarisk.assistant", level="WARNING"):
                self.assertEqual(models.default_model("gemini"), "gemini-3.5-flash-lite")
                self.assertEqual(models.default_model("openai"), "gpt-6-luna")

    def test_choose_validates(self):
        with with_keys(**ALL_KEYS):
            # nothing asked: default provider and model, nothing sent
            self.assertEqual(
                models.choose(None, None), ("anthropic", "claude-sonnet-5-5", None, "standard")
            )
            self.assertEqual(
                models.choose("anthropic", "claude-opus-5-5", "xhigh", "flex"),
                ("anthropic", "claude-opus-5-5", "xhigh", "standard"),  # no flex for Claude
            )
            self.assertEqual(
                models.choose("gemini", "gemini-3.1-pro-preview", "medium", "flex"),
                ("gemini", "gemini-3.1-pro-preview", "medium", "flex"),  # 3.1 Pro takes medium
            )
            self.assertEqual(  # `minimal` is a 400 on 3.8 Flash and 3.1 Pro: dropped
                models.choose("gemini", "gemini-3.8-flash", "minimal", "flex"),
                ("gemini", "gemini-3.8-flash", None, "flex"),
            )
            self.assertEqual(
                models.choose("gemini", "gemini-3.1-pro-preview", "minimal"),
                ("gemini", "gemini-3.1-pro-preview", None, "standard"),
            )
            self.assertEqual(  # only Flash-Lite takes it
                models.choose("gemini", "gemini-3.5-flash-lite", "minimal"),
                ("gemini", "gemini-3.5-flash-lite", "minimal", "standard"),
            )
            self.assertEqual(
                models.choose("anthropic", "claude-haiku-4-5", "high"),
                ("anthropic", "claude-haiku-4-5", None, "standard"),  # Haiku takes no effort
            )
            # a model the provider does not offer runs at the provider's default
            self.assertEqual(
                models.choose("gemini", "claude-opus-5-5"),
                ("gemini", "gemini-3.8-flash", None, "standard"),
            )
            self.assertEqual(
                models.choose("anthropic", "claude-sonnet-5"),
                ("anthropic", "claude-sonnet-5-5", None, "standard"),
            )
            # an unknown level is dropped, a Default for Luna is SENT as high
            self.assertEqual(
                models.choose("openai", "gpt-6-luna", "bogus"),
                ("openai", "gpt-6-luna", "high", "standard"),
            )
            self.assertEqual(
                models.choose("openai", "gpt-6.1-sol"), ("openai", "gpt-6.1-sol", None, "standard")
            )
            self.assertEqual(
                models.choose("openai", "gpt-6-luna", "max", "flex"),
                ("openai", "gpt-6-luna", "max", "flex"),
            )
        with with_keys(**ALL_KEYS):
            # C: a named model outside the table: the default, with ONE warning line ...
            with self.assertLogs("infrarisk.assistant", level="WARNING") as cm:
                models.choose("gemini", "gemini-3-flash-preview")
            self.assertEqual(
                [r.getMessage() for r in cm.records],
                ["chat: model 'gemini-3-flash-preview' is not offered for gemini; "
                 "using gemini-3.8-flash"],
            )
            # ... and none when no model was named, or a listed one
            with self.assertNoLogs("infrarisk.assistant", level="WARNING"):
                models.choose("gemini", None)
                models.choose("gemini", "")
                models.choose("gemini", "gemini-3.1-pro-preview")
        with with_keys(GEMINI_API_KEY="g"):
            with self.assertRaises(ValueError):
                models.choose("anthropic", None)
            with self.assertRaises(ValueError):
                models.choose("nope", None)
        with with_keys():
            with self.assertRaises(ValueError):
                models.choose(None, None)


class MetaEndpoint(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(assistant_api.router, prefix="/api")
        self.client = TestClient(app)

    def test_meta_shape(self):
        with with_keys(ANTHROPIC_API_KEY="a", OPENAI_API_KEY="o"):
            data = self.client.get("/api/assistant/meta").json()
        self.assertTrue(data["available"])
        self.assertEqual(data["default_provider"], "anthropic")
        self.assertEqual([p["id"] for p in data["providers"]], ["anthropic", "gemini", "openai"])
        self.assertEqual(
            [p["available"] for p in data["providers"]], [True, False, True]
        )
        by_id = {p["id"]: p for p in data["providers"]}
        for pid, p in by_id.items():
            self.assertEqual(
                set(p), {"id", "label", "available", "models", "default_model", "efforts",
                         "effort_default", "tiers"},
            )
            self.assertEqual(set(p["efforts"]), set(p["models"]))
            self.assertEqual(set(p["effort_default"]), set(p["models"]))
        self.assertEqual(by_id["anthropic"]["label"], "Claude")
        self.assertEqual(by_id["anthropic"]["efforts"]["claude-haiku-4-5"], [])
        self.assertIsNone(by_id["anthropic"]["effort_default"]["claude-haiku-4-5"])
        self.assertEqual(by_id["anthropic"]["effort_default"]["claude-opus-5-5"], "medium")
        self.assertEqual(by_id["gemini"]["efforts"]["gemini-3.1-pro-preview"], ["low", "medium", "high"])
        self.assertEqual(by_id["gemini"]["efforts"]["gemini-3.8-flash"], ["low", "medium", "high"])
        self.assertEqual(
            by_id["gemini"]["efforts"]["gemini-3.5-flash-lite"], ["minimal", "low", "medium", "high"]
        )
        self.assertEqual(by_id["openai"]["effort_default"]["gpt-6-luna"], "high")
        self.assertEqual(by_id["openai"]["tiers"], ["standard", "flex"])
        self.assertEqual(by_id["anthropic"]["tiers"], ["standard"])
        self.assertNotIn('"a"', str(data))  # availability booleans only, never a key

    def test_no_key_hides_the_assistant(self):
        with with_keys():
            data = self.client.get("/api/assistant/meta").json()
        self.assertFalse(data["available"])
        self.assertIsNone(data["default_provider"])

    def test_chat_passes_the_validated_choice_to_the_loop(self):
        seen = {}

        def fake_run(conv, provider, model, **kw):
            seen.update(provider=provider, model=model, **kw)
            yield schema.sse("done", {"reason": "end_turn"})

        with with_keys(**ALL_KEYS), mock.patch.object(loop, "run", fake_run):
            res = self.client.post(
                "/api/assistant/chat",
                json={"message": "hi", "provider": "openai", "model": "gpt-6.1-sol",
                      "effort": "xhigh", "service_tier": "flex"},
            )
            self.assertEqual(res.status_code, 200)
            self.assertEqual(
                (seen["provider"], seen["model"], seen["effort"], seen["service_tier"]),
                ("openai", "gpt-6.1-sol", "xhigh", "flex"),
            )
            self.client.post("/api/assistant/chat", json={"message": "hi"})
            self.assertEqual(
                (seen["provider"], seen["model"], seen["effort"], seen["service_tier"]),
                ("anthropic", "claude-sonnet-5-5", None, "standard"),
            )
            # an unkeyed provider is refused; so is a chat with no key at all
            gem = self.client.post(
                "/api/assistant/chat", json={"message": "hi", "provider": "unknown"}
            )
            self.assertEqual(gem.status_code, 422)
        with with_keys(ANTHROPIC_API_KEY="a"), mock.patch.object(loop, "run", fake_run):
            res = self.client.post("/api/assistant/chat", json={"message": "hi", "provider": "gemini"})
            self.assertEqual(res.status_code, 422)
        with with_keys():
            res = self.client.post("/api/assistant/chat", json={"message": "hi"})
            self.assertEqual(res.status_code, 503)


class AnthropicAdapter(unittest.TestCase):
    def _kwargs(self, **opts):
        captured = {}

        class Stream:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *a):
                return False

            def __iter__(self_inner):
                return iter(())

            def get_final_message(self_inner):
                return SimpleNamespace(
                    content=[], stop_reason="end_turn",
                    usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                )

        def stream(**kw):
            captured.update(kw)
            return Stream()

        with mock.patch.object(config, "ANTHROPIC_API_KEY", "a"):
            prov = anthropic_provider.AnthropicProvider()
        prov.client = SimpleNamespace(messages=SimpleNamespace(stream=stream))
        list(prov.stream_turn("claude-opus-5-5", "sys", [], [], **opts))
        return captured

    def test_output_cap_is_agui_s(self):
        self.assertEqual(anthropic_provider.MAX_TOKENS, 32_000)
        self.assertEqual(self._kwargs()["max_tokens"], 32_000)

    def test_effort_goes_to_output_config(self):
        self.assertEqual(self._kwargs(effort="xhigh")["output_config"], {"effort": "xhigh"})

    def test_default_sends_nothing_and_flex_is_ignored(self):
        kw = self._kwargs(service_tier="flex")
        self.assertNotIn("output_config", kw)
        self.assertNotIn("service_tier", kw)


def _openai_event(kind, **kw):
    return SimpleNamespace(type=kind, **kw)


def _openai_final():
    return SimpleNamespace(
        output=[], usage=SimpleNamespace(input_tokens=3, output_tokens=4), incomplete_details=None
    )


class _OpenAIStream:
    def __init__(self, events):
        self.events = events

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        return iter(self.events)


def _rate_limit(message="Resource Unavailable", body=None):
    request = httpx2.Request("POST", "https://api.openai.com/v1/responses")
    return RateLimitError(
        message, response=httpx2.Response(429, request=request), body=body or {}
    )


class OpenAIAdapter(unittest.TestCase):
    def _provider(self, create):
        with mock.patch.object(config, "OPENAI_API_KEY", "o"):
            prov = openai_provider.OpenAIProvider()
        prov.client = SimpleNamespace(responses=SimpleNamespace(create=create))
        return prov

    def _run(self, create, **opts):
        prov = self._provider(create)
        slept: list[float] = []
        with mock.patch.object(openai_provider, "_sleep", slept.append):
            events = list(prov.stream_turn("gpt-6-luna", "sys", [], [], **opts))
        return events, slept

    def _ok_stream(self):
        item = SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text")],
                               model_dump=lambda **kw: {"type": "message"})
        return _OpenAIStream([
            _openai_event("response.output_text.delta", delta="Hi"),
            _openai_event("response.output_item.done", output_index=0, item=item),
            _openai_event("response.completed", response=_openai_final()),
        ])

    def test_effort_and_standard_request(self):
        calls = []

        def create(**kw):
            calls.append(kw)
            return self._ok_stream()

        events, slept = self._run(create, effort="xhigh", service_tier="standard")
        kw = calls[0]
        self.assertEqual(kw["reasoning"], {"summary": "auto", "effort": "xhigh"})
        self.assertEqual(kw["max_output_tokens"], 64_000)
        self.assertNotIn("service_tier", kw)
        self.assertNotIn("timeout", kw)
        self.assertEqual(slept, [])
        self.assertIsInstance(events[0], schema.TextDelta)
        self.assertIsInstance(events[-1], schema.TurnEnd)

    def test_default_effort_sends_none(self):
        calls = []
        self._run(lambda **kw: calls.append(kw) or self._ok_stream())
        self.assertEqual(calls[0]["reasoning"], {"summary": "auto"})
        self.assertEqual(calls[0]["max_output_tokens"], 32_000)

    def test_flex_request(self):
        calls = []
        self._run(lambda **kw: calls.append(kw) or self._ok_stream(), service_tier="flex")
        self.assertEqual(calls[0]["service_tier"], "flex")
        self.assertEqual(calls[0]["timeout"], 900.0)

    def test_flex_capacity_backs_off_then_succeeds_at_flex(self):
        calls = []

        def create(**kw):
            calls.append(kw)
            if len(calls) <= 2:
                raise _rate_limit()
            return self._ok_stream()

        _, slept = self._run(create, service_tier="flex")
        self.assertEqual(slept, [2.0, 4.0])
        self.assertEqual(calls[-1]["service_tier"], "flex")

    def test_flex_capacity_exhausted_falls_back_to_standard(self):
        calls = []

        def create(**kw):
            calls.append(kw)
            if "service_tier" in kw:
                raise _rate_limit()
            return self._ok_stream()

        _, slept = self._run(create, service_tier="flex")
        self.assertEqual(slept, [2.0, 4.0, 8.0, 16.0])
        self.assertEqual(len(calls), 6)  # 5 flex attempts, one standard
        self.assertNotIn("service_tier", calls[-1])
        self.assertNotIn("timeout", calls[-1])

    def test_quota_and_rate_waits_are_not_flex_capacity(self):
        quota = _rate_limit("You exceeded your current quota",
                            {"type": "insufficient_quota", "code": "insufficient_quota"})
        wait = _rate_limit("Rate limit reached. Please try again in 20s.")
        for exc in (quota, wait):
            calls = []

            def create(exc=exc, calls=calls, **kw):
                calls.append(kw)
                raise exc

            with self.assertRaises(RateLimitError):
                self._run(create, service_tier="flex")
            self.assertEqual(len(calls), 1)

    def test_standard_request_never_backs_off(self):
        def create(**kw):
            raise _rate_limit()

        with self.assertRaises(RateLimitError):
            self._run(create)

    def test_input_items_and_tool_defs(self):
        history = [
            {"role": "user", "parts": [{"type": "text", "text": "q"}]},
            {"role": "assistant", "parts": [
                {"type": "text", "text": "a"},
                {"type": "tool_call", "id": "c1", "name": "t", "args": {"x": 1}}]},
            {"role": "user", "parts": [
                {"type": "tool_result", "id": "c1", "name": "t", "ok": False, "content": {"e": 1}}]},
        ]
        items = openai_provider._input(history)
        self.assertEqual(items[0], {"role": "user", "content": [{"type": "input_text", "text": "q"}]})
        self.assertEqual(items[1], {"role": "assistant", "content": "a"})
        self.assertEqual(items[2]["type"], "function_call")
        self.assertEqual(items[2]["arguments"], '{"x": 1}')
        self.assertEqual(items[3]["output"], '[tool error] {"e": 1}')


class GeminiAdapter(unittest.TestCase):
    def _provider(self, generate):
        with mock.patch.object(config, "GEMINI_API_KEY", "g"):
            prov = gemini_provider.GeminiProvider()
        prov.client = SimpleNamespace(models=SimpleNamespace(generate_content_stream=generate))
        return prov

    def _chunk(self):
        return SimpleNamespace(
            usage_metadata=None,
            candidates=[SimpleNamespace(
                finish_reason=None,
                content=SimpleNamespace(parts=[SimpleNamespace(
                    text="Hi", thought=False, function_call=None)]),
            )],
        )

    def _run(self, generate, **opts):
        prov = self._provider(generate)
        slept: list[float] = []
        with mock.patch.object(gemini_provider, "_sleep", slept.append):
            events = list(prov.stream_turn("gemini-3.8-flash", "sys", [], [], **opts))
        return events, slept

    def test_effort_is_a_thinking_level_and_default_sends_none(self):
        cfgs = []

        def generate(model, contents, config):
            cfgs.append(config)
            return iter([self._chunk()])

        self._run(generate, effort="minimal")
        self.assertEqual(cfgs[0].thinking_config.thinking_level, types.ThinkingLevel.MINIMAL)
        self.assertIsNone(cfgs[0].service_tier)
        self.assertIsNone(cfgs[0].http_options)
        self._run(generate)
        self.assertIsNone(cfgs[1].thinking_config)

    def test_flex_config(self):
        cfgs = []

        def generate(model, contents, config):
            cfgs.append(config)
            return iter([self._chunk()])

        events, slept = self._run(generate, service_tier="flex")
        self.assertEqual(cfgs[0].service_tier, types.ServiceTier.FLEX)
        self.assertEqual(cfgs[0].http_options.timeout, 900_000)
        self.assertEqual(slept, [])
        self.assertIsInstance(events[0], schema.TextDelta)

    def _refusing(self, code):
        def stream():
            raise genai_errors.APIError(code, {"error": {"message": "overloaded"}})
            yield  # pragma: no cover

        return stream()

    def test_flex_capacity_backs_off_then_falls_back_to_standard(self):
        for code in (503, 429):
            cfgs = []

            def generate(model, contents, config, code=code, cfgs=cfgs):
                cfgs.append(config)
                if config.service_tier == types.ServiceTier.FLEX:
                    return self._refusing(code)
                return iter([self._chunk()])

            events, slept = self._run(generate, service_tier="flex")
            self.assertEqual(slept, [2.0, 4.0, 8.0, 16.0])
            self.assertEqual(len(cfgs), 6)
            self.assertIsNone(cfgs[-1].service_tier)
            self.assertIsNone(cfgs[-1].http_options)
            self.assertIsInstance(events[0], schema.TextDelta)

    def test_standard_errors_and_other_codes_are_raised(self):
        def standard(model, contents, config):
            return self._refusing(503)

        with self.assertRaises(genai_errors.APIError):
            self._run(standard)

        def flex_bad_request(model, contents, config):
            return self._refusing(400)

        with self.assertRaises(genai_errors.APIError):
            self._run(flex_bad_request, service_tier="flex")


class LoopHandsOverTheChoice(unittest.TestCase):
    def test_stream_turn_receives_effort_and_tier(self):
        seen = {}

        class Fake:
            def stream_turn(self, model, system, messages, tools, effort=None, service_tier=None):
                seen.update(model=model, effort=effort, service_tier=service_tier)
                yield schema.TextDelta("ok")
                yield schema.TurnEnd(stop_reason="end_turn")

        conv = SimpleNamespace(
            id="c", messages=[], pending_calls=[], buffered_results=[], provider=None, model=None
        )
        with mock.patch.object(loop, "get_provider", lambda name: Fake()):
            out = "".join(loop.run(conv, "gemini", "gemini-3.8-flash", user_message="hi",
                                   effort="low", service_tier="flex"))
        self.assertEqual(seen, {"model": "gemini-3.8-flash", "effort": "low", "service_tier": "flex"})
        self.assertIn('"effort": "low"', out)
        self.assertIn('"service_tier": "flex"', out)

    def test_exhausted_quota_is_not_retried(self):
        self.assertFalse(loop._is_transient(RuntimeError("Error code: 429 - insufficient_quota")))
        self.assertTrue(loop._is_transient(RuntimeError("Error code: 429 - rate limit")))


class KeepAlive(unittest.TestCase):
    """F: the SSE stream is not silent while the provider is (flex backoff, long thinking)."""

    def _run(self, provider):
        conv = SimpleNamespace(
            id="c", messages=[], pending_calls=[], buffered_results=[], provider=None, model=None
        )
        with mock.patch.object(loop, "PING_EVERY", 0.05), mock.patch.object(
            loop, "get_provider", lambda name: provider
        ), mock.patch.object(loop, "RETRY_BACKOFF_S", (0.0,)):
            return list(loop.run(conv, "openai", "gpt-6-luna", user_message="hi"))

    def test_ping_comes_before_a_slow_first_event(self):
        class Slow:
            def stream_turn(self, **kw):
                time.sleep(0.4)  # a flex request still waiting for its first byte
                yield schema.TextDelta("ok")
                yield schema.TurnEnd(stop_reason="end_turn")

        out = self._run(Slow())
        first_text = next(i for i, c in enumerate(out) if c.startswith("event: text"))
        self.assertIn(schema.SSE_PING, out[:first_text])
        self.assertGreaterEqual(out[:first_text].count(schema.SSE_PING), 2)
        self.assertEqual(out[-1], schema.sse("done", {"reason": "end_turn"}))

    def test_a_quick_provider_adds_no_ping(self):
        class Quick:
            def stream_turn(self, **kw):
                yield schema.TextDelta("ok")
                yield schema.TurnEnd(stop_reason="end_turn")

        with mock.patch.object(loop, "PING_EVERY", 5.0):
            out = self._run(Quick())
        self.assertNotIn(schema.SSE_PING, out)

    def test_a_provider_error_still_reaches_the_loops_error_handling(self):
        class Boom:
            def stream_turn(self, **kw):
                time.sleep(0.15)
                raise RuntimeError("boom")
                yield  # pragma: no cover

        out = self._run(Boom())
        self.assertIn(schema.SSE_PING, out)
        self.assertEqual(out[-2:], [schema.sse("error", {"message": "openai: boom"}),
                                    schema.sse("done", {"reason": "error"})])

    def test_a_transient_error_is_still_retried(self):
        attempts = []

        class Flaky:
            def stream_turn(self, **kw):
                attempts.append(1)
                if len(attempts) == 1:
                    raise RuntimeError("503 unavailable")
                yield schema.TextDelta("ok")
                yield schema.TurnEnd(stop_reason="end_turn")

        out = self._run(Flaky())
        self.assertEqual(len(attempts), 2)
        self.assertTrue(any(c.startswith("event: notice") for c in out))
        self.assertEqual(out[-1], schema.sse("done", {"reason": "end_turn"}))

    def test_the_pump_stops_when_the_consumer_leaves(self):
        closed = []

        def source():
            try:
                for i in range(1000):
                    time.sleep(0.01)
                    yield i
            finally:
                closed.append(True)

        gen = loop._provider_with_pings(source())
        self.assertEqual(next(gen), 0)
        gen.close()  # the client left
        time.sleep(0.2)
        self.assertEqual(closed, [True])


if __name__ == "__main__":
    unittest.main()
