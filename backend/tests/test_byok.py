"""Bring your own key: a user's API key is used for the request that brings it and appears
nowhere afterwards — not in the stream, the conversation, its export, or a log line, even
when the provider's error quotes it.

Run: cd backend && python -m unittest discover -s tests
"""

from __future__ import annotations

import json
import logging
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402
from app import config  # noqa: E402
from app.assistant import conversations, loop, models, providers, schema  # noqa: E402

USER_KEY = "sk-ant-api03-USERKEY-0123456789abcdefghijklmnop"
# a key in no provider's known shape: only the request's own scrubbing can catch it
PLAIN_KEY = "plainkey-0123456789abcdefXYZ"
NO_SERVER_KEYS = {"ANTHROPIC_API_KEY": "", "GEMINI_API_KEY": "", "OPENAI_API_KEY": ""}


class FakeProvider:
    """Stands in for a provider adapter; records the key it was built with."""

    built_with: list[str | None] = []

    def __init__(self, api_key=None, fail: str | None = None):
        FakeProvider.built_with.append(api_key)
        self.fail = fail

    def stream_turn(self, model, system, messages, tools, effort=None, service_tier=None):
        if self.fail:
            raise RuntimeError(self.fail)
        yield schema.TextDelta("hello from the model")
        yield schema.Usage(input_tokens=10, output_tokens=3)
        yield schema.TurnEnd(stop_reason="end_turn")


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(self.format(record))


def _everywhere(conv: conversations.Conversation) -> str:
    """Everything the server keeps about a conversation, as text."""
    return json.dumps(
        {"messages": conv.messages, "events": conv.events, "vars": {k: repr(v) for k, v in vars(conv).items()}},
        default=repr,
    )


class ByokTest(unittest.TestCase):
    def setUp(self):
        for name, value in NO_SERVER_KEYS.items():
            patcher = mock.patch.object(config, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        FakeProvider.built_with = []
        self.client = TestClient(main.app)
        self.log = _Capture()
        logging.getLogger().addHandler(self.log)
        self.addCleanup(logging.getLogger().removeHandler, self.log)
        self.root_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.DEBUG)
        self.addCleanup(logging.getLogger().setLevel, self.root_level)

    def _chat(self, conv_id: str, **extra) -> str:
        body = {"conversation_id": conv_id, "provider": "anthropic", "message": "hi", **extra}
        r = self.client.post("/api/assistant/chat", json=body)
        self.assertEqual(r.status_code, 200, r.text)
        return r.text

    def test_meta_offers_byok_without_server_keys(self):
        meta = self.client.get("/api/assistant/meta").json()
        self.assertTrue(meta["byok"])
        self.assertFalse(meta["available"])
        self.assertNotIn(USER_KEY, json.dumps(meta))

    def test_choose_needs_a_server_key_or_the_users(self):
        with self.assertRaises(ValueError):
            models.choose("anthropic", None)
        self.assertEqual(models.choose("anthropic", None, user_key=True).provider, "anthropic")
        with mock.patch.object(config, "ASSISTANT_BYOK", False), self.assertRaises(ValueError):
            models.choose("anthropic", None, user_key=True)

    def test_adapters_take_the_users_key(self):
        with self.assertRaises(ValueError):
            providers.get("openai")
        self.assertEqual(providers.get("anthropic", USER_KEY).client.api_key, USER_KEY)
        self.assertEqual(providers.get("openai", USER_KEY).client.api_key, USER_KEY)
        self.assertEqual(providers.get("gemini", USER_KEY).client._api_client.api_key, USER_KEY)

    def test_a_turn_with_the_users_key_leaves_no_trace(self):
        with mock.patch.object(loop, "get_provider", lambda name, key=None: FakeProvider(key)):
            stream = self._chat("byok-ok", api_key=USER_KEY)
        self.assertEqual(FakeProvider.built_with, [USER_KEY])
        self.assertIn("hello from the model", stream)
        conv = conversations.get_or_create("byok-ok")
        export = self.client.get("/api/assistant/conversations/byok-ok/export?format=json").text
        for name, text in {"stream": stream, "conversation": _everywhere(conv),
                           "export": export, "logs": "\n".join(self.log.lines)}.items():
            with self.subTest(where=name):
                self.assertNotIn(USER_KEY, text)

    def test_a_provider_error_quoting_the_key_is_scrubbed(self):
        failing = lambda name, key=None: FakeProvider(key, fail=f"request rejected (sent with {key})")  # noqa: E731
        for key in (USER_KEY, PLAIN_KEY):
            with mock.patch.object(loop, "get_provider", failing):
                stream = self._chat(f"byok-error-{key[:5]}", api_key=key)
            self.assertIn("[redacted]", stream)
            conv = conversations.get_or_create(f"byok-error-{key[:5]}")
            for name, text in {"stream": stream, "conversation": _everywhere(conv),
                               "logs": "\n".join(self.log.lines)}.items():
                with self.subTest(key=key[:5], where=name):
                    self.assertNotIn(key, text)

    def test_an_internal_error_quoting_the_key_is_scrubbed(self):
        def explode(name, key=None):
            raise RuntimeError(f"could not build a client for {key}")

        for key in (USER_KEY, PLAIN_KEY):
            with mock.patch.object(loop, "get_provider", explode):
                stream = self._chat(f"byok-internal-{key[:5]}", api_key=key)
            self.assertIn("internal error", stream)
            conv = conversations.get_or_create(f"byok-internal-{key[:5]}")
            for name, text in {"stream": stream, "conversation": _everywhere(conv),
                               "logs": "\n".join(self.log.lines)}.items():
                with self.subTest(key=key[:5], where=name):
                    self.assertNotIn(key, text)

    def test_a_refused_key_says_so_plainly(self):
        class Refused(Exception):
            status_code = 401

        class Failing(FakeProvider):
            def stream_turn(self, *a, **k):
                raise Refused("Error code: 401 - invalid x-api-key")
                yield  # pragma: no cover

        with mock.patch.object(loop, "get_provider", lambda name, key=None: Failing(key)):
            stream = self._chat("byok-refused", api_key=USER_KEY)
        self.assertIn("Claude refused the API key you entered", stream)
        self.assertNotIn(USER_KEY, stream)

    def test_refusals_never_repeat_the_key(self):
        cases = [({"api_key": "short"}, 422), ({"api_key": "has a space in it"}, 422)]
        for extra, status in cases:
            r = self.client.post("/api/assistant/chat",
                                 json={"provider": "anthropic", "message": "hi", **extra})
            self.assertEqual(r.status_code, status)
            self.assertNotIn(extra["api_key"], r.text)
        with mock.patch.object(config, "ASSISTANT_BYOK", False):
            r = self.client.post("/api/assistant/chat",
                                 json={"provider": "anthropic", "message": "hi", "api_key": USER_KEY})
        self.assertEqual(r.status_code, 422)
        self.assertNotIn(USER_KEY, r.text)
        # no key at all, and none on the server: refused as before
        r = self.client.post("/api/assistant/chat", json={"provider": "anthropic", "message": "hi"})
        self.assertEqual(r.status_code, 422)


class KeyShapeTest(unittest.TestCase):
    def test_provider_key_shapes_are_masked(self):
        from app.redact import redact_text

        for key in (USER_KEY, "sk-proj-abcdefghijklmnopqrstuvwx", "AIzaSyA1234567890abcdefghijklmnopqrs"):
            with self.subTest(key=key[:8]):
                self.assertEqual(redact_text(f"bad key {key}."), "bad key [redacted].")
        self.assertEqual(redact_text("sk-learn and AIza are fine"), "sk-learn and AIza are fine")


if __name__ == "__main__":
    unittest.main()
