"""The Gemini provider on its two backends: the Developer API and Vertex AI (README, "Gemini on
Vertex AI"). No network and no credentials: the real SDK builds each request and only its
transport (`_api_client._request`) is replaced, so URLs, headers and bodies are the SDK's own.

Run from backend/ with the repo's venv:  python -m unittest tests.test_gemini_backends
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from google.oauth2 import credentials as oauth2  # noqa: E402
from google.genai import _api_client  # noqa: E402
from google.genai import errors as genai_errors  # noqa: E402

from app import config  # noqa: E402
from app.assistant import models, schema  # noqa: E402
from app.assistant.providers import gemini as adapter  # noqa: E402

FLEX_HEADER = "X-Vertex-AI-LLM-Shared-Request-Type"
MODEL = "gemini-3.8-flash"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/"
VERTEX_URL = "https://aiplatform.googleapis.com/v1beta1/"
STREAM = f"{MODEL}:streamGenerateContent?alt=sse"


def settings(backend: str, **given: str):
    """config as backend/.env would set it (unnamed settings empty), patched for one test."""
    values = {"GEMINI_API_KEY": "", "VERTEX_PROJECT": "", "VERTEX_API_KEY": "",
              "VERTEX_LOCATION": "global", "GOOGLE_APPLICATION_CREDENTIALS": ""}
    values.update(given)
    return mock.patch.multiple(config, GEMINI_BACKEND=backend, **values)


def sdk_environment(vertex: bool):
    """The SDK's own environment variables, all pulling the other way from the backend in use:
    the client must follow its explicit arguments, never these."""
    other = "false" if vertex else "true"
    return mock.patch.dict(os.environ, {
        "GOOGLE_GENAI_USE_ENTERPRISE": other, "GOOGLE_GENAI_USE_VERTEXAI": other,
        "GOOGLE_API_KEY": "env-key", "GOOGLE_CLOUD_PROJECT": "env-project",
        "GOOGLE_CLOUD_LOCATION": "us-central1",
    })


def no_adc():
    def refuse(**_kw):
        raise AssertionError("Application Default Credentials looked up")

    return mock.patch.object(_api_client, "load_auth", refuse)


def reply(text: str = "ok") -> str:
    return json.dumps({
        "candidates": [{"content": {"role": "model", "parts": [{"text": text}]},
                        "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 1,
                          "totalTokenCount": 6},
    })


class Wire:
    """A provider on the real SDK whose transport is replaced: every request the SDK builds
    is kept in `sent` (url, headers, body) and answered by `answer(request)`, which returns
    the streamed chunks or raises."""

    def __init__(self, test: unittest.TestCase, answer=None) -> None:
        self.sent: list = []
        self.answer = answer or (lambda request: [reply()])
        with no_adc():
            self.provider = adapter.GeminiProvider()
        patch = mock.patch.object(self.provider.client._api_client, "_request", self._request)
        patch.start()
        test.addCleanup(patch.stop)
        test.addCleanup(self.provider.client.close)

    def _request(self, http_request, http_options=None, stream=False):
        self.sent.append(http_request)
        return _api_client.HttpResponse({}, self.answer(http_request))

    def turn(self, model: str = MODEL, service_tier: str | None = None) -> list:
        return list(self.provider.stream_turn(
            model, "sys", [schema.text_message("user", "hi")], [], service_tier=service_tier))


def flex_requested(request) -> bool:
    """Whether the request asks for flex, by either backend's means."""
    return FLEX_HEADER in request.headers or "serviceTier" in request.data


class MakeClient(unittest.TestCase):
    CASES = [
        ("developer", {"GEMINI_API_KEY": "dev-key"}, {"enterprise": False, "api_key": "dev-key"}),
        ("vertex", {"VERTEX_PROJECT": "p"},
         {"enterprise": True, "project": "p", "location": "global"}),
        ("vertex", {"VERTEX_PROJECT": "p", "VERTEX_LOCATION": "europe-west4"},
         {"enterprise": True, "project": "p", "location": "europe-west4"}),
        # Google documents the key without project and location; a project set beside it is unused
        ("vertex", {"VERTEX_PROJECT": "p", "VERTEX_API_KEY": "vx-key"},
         {"enterprise": True, "api_key": "vx-key"}),
    ]

    def test_the_client_is_built_explicitly_for_each_backend(self):
        for backend, given, expected in self.CASES:
            with self.subTest(backend=backend, given=sorted(given)):
                built: list[dict] = []
                with settings(backend, **given), \
                        mock.patch.object(config, "ASSISTANT_PROVIDER_TIMEOUT", 120.0), \
                        mock.patch.object(adapter.genai, "Client",
                                          lambda **kw: built.append(kw)):
                    adapter.make_client()
                self.assertEqual(len(built), 1)
                kwargs = built[0]
                # the provider timeout this adapter always passed stays, on both backends
                self.assertEqual(kwargs.pop("http_options").timeout, 120_000)
                self.assertEqual(kwargs, expected)

    def test_the_client_ignores_the_sdks_environment_variables(self):
        for backend, given, key, place in [
            ("developer", {"GEMINI_API_KEY": "dev-key"}, "dev-key", None),
            ("vertex", {"VERTEX_PROJECT": "p"}, None, ("p", "global")),
            ("vertex", {"VERTEX_API_KEY": "vx-key"}, "vx-key", (None, None)),
        ]:
            with self.subTest(backend=backend, given=sorted(given)):
                with settings(backend, **given), no_adc(), \
                        sdk_environment(backend == "vertex"):
                    client = adapter.make_client()
                self.addCleanup(client.close)
                api = client._api_client
                self.assertEqual((bool(api.vertexai), api.api_key), (backend == "vertex", key))
                if place:
                    self.assertEqual((api.project, api.location), place)

    def test_an_incomplete_or_unknown_backend_is_refused_and_not_offered(self):
        for backend, given in [("vertex", {"GEMINI_API_KEY": "dev-key"}),   # no project, no key
                               ("vertexai", {"VERTEX_PROJECT": "p"})]:       # not a backend
            with self.subTest(backend=backend), settings(backend, **given):
                with self.assertRaises(ValueError):
                    adapter.make_client()
                self.assertFalse(config.gemini_configured())

    def test_a_service_account_file_is_loaded_for_the_client(self):
        """GOOGLE_APPLICATION_CREDENTIALS is scrubbed from the environment at startup
        (app/redact.py), so the SDK's own lookup would never find it: the adapter loads the file
        named by config and hands the credentials over."""
        creds = oauth2.Credentials(token="fake-access-token")   # unexpiring: no refresh, no network
        loaded: list = []

        def load(path, scopes):
            loaded.append((path, scopes))
            return creds, "key-project"

        with settings("vertex", VERTEX_PROJECT="p",
                      GOOGLE_APPLICATION_CREDENTIALS="/run/secrets/sa.json"), \
                mock.patch.object(adapter.google.auth, "load_credentials_from_file", load), \
                no_adc(), sdk_environment(True):
            client = adapter.make_client()
        self.addCleanup(client.close)
        self.assertEqual(loaded, [("/run/secrets/sa.json",
                                   ["https://www.googleapis.com/auth/cloud-platform"])])
        api = client._api_client
        self.assertEqual((api.project, api.location), ("p", "global"))   # the setting, not the key's
        self.assertEqual(api._access_token(), "fake-access-token")        # and no ADC lookup

    def test_no_credentials_file_is_loaded_when_none_applies(self):
        load = mock.Mock()
        for backend, given in [
            ("vertex", {"VERTEX_PROJECT": "p"}),                   # gcloud's ADC login
            ("vertex", {"VERTEX_API_KEY": "vx-key",                # a key wins over the file
                        "GOOGLE_APPLICATION_CREDENTIALS": "/run/secrets/sa.json"}),
            ("developer", {"GEMINI_API_KEY": "dev-key",
                           "GOOGLE_APPLICATION_CREDENTIALS": "/run/secrets/sa.json"}),
        ]:
            with self.subTest(backend=backend, given=sorted(given)), settings(backend, **given), \
                    mock.patch.object(adapter.google.auth, "load_credentials_from_file", load):
                adapter.make_client().close()
        load.assert_not_called()


class Availability(unittest.TestCase):
    CASES = [
        ("developer", {"GEMINI_API_KEY": "dev-key"}, True),
        ("developer", {"VERTEX_PROJECT": "p", "VERTEX_API_KEY": "vx-key"}, False),
        ("vertex", {"VERTEX_PROJECT": "p"}, True),   # ADC: not checkable without a call
        ("vertex", {"VERTEX_API_KEY": "vx-key"}, True),
        ("vertex", {"GEMINI_API_KEY": "dev-key"}, False),
        ("vertex", {}, False),
        ("vertexai", {"VERTEX_PROJECT": "p", "GEMINI_API_KEY": "dev-key"}, False),
    ]

    def test_gemini_is_offered_per_the_backend(self):
        for backend, given, available in self.CASES:
            with self.subTest(backend=backend, given=sorted(given)):
                with settings(backend, **given), mock.patch.multiple(
                        config, ANTHROPIC_API_KEY="", OPENAI_API_KEY="",
                        ASSISTANT_DEFAULT_PROVIDER=""):
                    self.assertIs(config.gemini_configured(), available)
                    self.assertIs(models.keys()["gemini"], available)
                    by_id = {p["id"]: p for p in models.meta()["providers"]}
                    self.assertIs(by_id["gemini"]["available"], available)
                    self.assertEqual(models.default_provider(), "gemini" if available else None)
                    if not available:
                        with self.assertRaises(ValueError):
                            models.choose("gemini", MODEL)

    def test_settings_as_the_environment_gives_them(self):
        """Read at import, in a fresh interpreter. Empty counts as unset (python-dotenv
        leaves a variable that is set, even to nothing, alone): the live fleet's .env predates
        GEMINI_BACKEND, and until it is set the Developer API runs. The key file's path is
        made absolute, because python_exec changes the working directory."""
        probe = ("import json; from app import config as c; print(json.dumps([c.GEMINI_BACKEND, "
                 "c.VERTEX_LOCATION, c.gemini_configured(), c.GOOGLE_APPLICATION_CREDENTIALS]))")
        empty = {"GEMINI_BACKEND": "", "VERTEX_PROJECT": "", "VERTEX_LOCATION": "",
                 "VERTEX_API_KEY": "", "GEMINI_API_KEY": "", "GOOGLE_APPLICATION_CREDENTIALS": ""}
        for given, expected in [
            ({"GEMINI_API_KEY": "fake-gemini-key-0123456789"},
             ["developer", "global", True, ""]),
            ({}, ["developer", "global", False, ""]),
            ({"VERTEX_PROJECT": "p"}, ["developer", "global", False, ""]),   # backend unset
            ({"GEMINI_BACKEND": " Vertex ", "VERTEX_PROJECT": "p", "VERTEX_LOCATION": "us-east5",
              "GOOGLE_APPLICATION_CREDENTIALS": "keys/sa.json"},
             ["vertex", "us-east5", True, str(BACKEND / "keys" / "sa.json")]),
        ]:
            with self.subTest(given=sorted(given)):
                proc = subprocess.run(
                    [sys.executable, "-c", probe], cwd=BACKEND, capture_output=True, text=True,
                    env={**os.environ, **empty, **given}, timeout=120)
                self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
                self.assertEqual(json.loads(proc.stdout.strip().splitlines()[-1]), expected)


class Endpoints(unittest.TestCase):
    CASES = [
        ("developer", {"GEMINI_API_KEY": "dev-key"}, GEMINI_URL + STREAM, "dev-key"),
        # ADC adds its bearer token when the request is sent, which the replaced transport skips
        ("vertex", {"VERTEX_PROJECT": "p", "GEMINI_API_KEY": "dev-key"},
         VERTEX_URL + f"projects/p/locations/global/publishers/google/models/{STREAM}", None),
        ("vertex", {"VERTEX_PROJECT": "p", "VERTEX_LOCATION": "europe-west4"},
         "https://europe-west4-aiplatform.googleapis.com/v1beta1/projects/p/locations/"
         f"europe-west4/publishers/google/models/{STREAM}", None),
        ("vertex", {"VERTEX_API_KEY": "vx-key"},
         VERTEX_URL + f"publishers/google/models/{STREAM}", "vx-key"),
    ]

    def test_each_backend_streams_from_its_endpoint_at_the_standard_tier(self):
        for backend, given, url, key in self.CASES:
            with self.subTest(backend=backend, given=sorted(given)):
                with settings(backend, **given), sdk_environment(backend == "vertex"):
                    wire = Wire(self)
                    events = wire.turn()
                request = wire.sent[0]
                self.assertEqual([type(e).__name__ for e in events],
                                 ["TextDelta", "Usage", "TurnEnd"])
                self.assertEqual((request.url, request.headers.get("x-goog-api-key")), (url, key))
                self.assertFalse(flex_requested(request))
                self.assertEqual(request.headers["X-Server-Timeout"], "300")   # the client's cap


class Flex(unittest.TestCase):
    def test_flex_is_asked_for_as_each_backend_documents(self):
        for backend, given, header, tier in [
            ("developer", {"GEMINI_API_KEY": "k"}, None, "flex"),       # service_tier
            ("vertex", {"VERTEX_PROJECT": "p"}, "flex", None),          # Flex PayGo header
            ("vertex", {"VERTEX_API_KEY": "k", "VERTEX_LOCATION": "europe-west4"}, "flex", None),
        ]:
            with self.subTest(backend=backend, given=sorted(given)):
                with settings(backend, **given):
                    wire = Wire(self)
                    wire.turn(service_tier="flex")
                request = wire.sent[0]
                self.assertEqual(request.headers.get(FLEX_HEADER), header)
                serviceTier = request.data.get("serviceTier")
                self.assertEqual(getattr(serviceTier, "value", serviceTier), tier)
                self.assertEqual(request.headers["X-Server-Timeout"], "900")   # FLEX_TIMEOUT_MS

    def test_vertex_flex_that_cannot_be_had_runs_at_standard(self):
        for model, location in [("gemini-2.5-flash", "global"),         # not on the Flex PayGo list
                                (MODEL, "europe-west4")]:               # Flex PayGo is global only
            with self.subTest(model=model, location=location):
                with settings("vertex", VERTEX_PROJECT="p", VERTEX_LOCATION=location):
                    wire = Wire(self)
                    with self.assertLogs("infrarisk.assistant", "WARNING") as logged:
                        wire.turn(model, service_tier="flex")
                request = wire.sent[0]
                self.assertFalse(flex_requested(request))
                self.assertEqual(request.headers["X-Server-Timeout"], "300")
                self.assertIn("standard tier", logged.output[0])

    def test_every_gemini_model_of_the_table_has_flex_paygo(self):
        self.assertLessEqual(set(models.PROVIDERS["gemini"]["models"]), adapter.VERTEX_FLEX_MODELS)

    def test_a_standard_request_after_a_flex_one_carries_nothing_of_it(self):
        for backend, given in [("developer", {"GEMINI_API_KEY": "k"}),
                               ("vertex", {"VERTEX_PROJECT": "p"})]:
            with self.subTest(backend=backend), settings(backend, **given):
                wire = Wire(self)
                wire.turn(service_tier="flex")
                wire.turn()
                flexed, standard = wire.sent
                self.assertTrue(flex_requested(flexed))
                self.assertFalse(flex_requested(standard))
                self.assertEqual(standard.headers["X-Server-Timeout"], "300")

    def test_flex_without_capacity_is_backed_off_then_sent_at_standard(self):
        """The fallback drops everything that asked for flex: the Vertex header too."""
        overloaded = {429: genai_errors.ClientError, 503: genai_errors.ServerError}
        for backend, given in [("developer", {"GEMINI_API_KEY": "k"}),
                               ("vertex", {"VERTEX_PROJECT": "p"}),
                               ("vertex", {"VERTEX_API_KEY": "k"})]:
            for code in (503, 429):
                with self.subTest(backend=backend, given=sorted(given), code=code):
                    def answer(request, code=code):
                        if flex_requested(request):
                            raise overloaded[code](code, {"error": {
                                "code": code, "message": "overloaded", "status": "UNAVAILABLE"}})
                        return [reply()]

                    slept: list[float] = []
                    with settings(backend, **given), \
                            mock.patch.object(adapter, "_sleep", slept.append), \
                            self.assertLogs("infrarisk.assistant", "WARNING"):
                        wire = Wire(self, answer)
                        events = wire.turn(service_tier="flex")
                    self.assertEqual([type(e).__name__ for e in events],
                                     ["TextDelta", "Usage", "TurnEnd"])
                    self.assertEqual(slept, [2.0, 4.0, 8.0, 16.0])
                    self.assertEqual([flex_requested(r) for r in wire.sent], [True] * 5 + [False])
                    self.assertNotIn(FLEX_HEADER, wire.sent[-1].headers)
                    self.assertEqual(wire.sent[-1].headers["X-Server-Timeout"], "300")

    def test_flex_served_on_a_retry_keeps_the_header(self):
        refused = [2]

        def answer(request):
            if refused[0]:
                refused[0] -= 1
                raise genai_errors.ServerError(503, {"error": {"code": 503, "message": "busy"}})
            return [reply()]

        slept: list[float] = []
        with settings("vertex", VERTEX_PROJECT="p"), \
                mock.patch.object(adapter, "_sleep", slept.append):
            wire = Wire(self, answer)
            wire.turn(service_tier="flex")
        self.assertEqual(slept, [2.0, 4.0])
        self.assertTrue(all(r.headers.get(FLEX_HEADER) == "flex" for r in wire.sent))
        self.assertEqual(len(wire.sent), 3)

    def test_other_errors_are_not_retried_on_vertex(self):
        def answer(request):
            raise genai_errors.ClientError(400, {"error": {"code": 400, "message": "bad"}})

        with settings("vertex", VERTEX_PROJECT="p"), \
                mock.patch.object(adapter, "_sleep", lambda s: self.fail("no wait expected")):
            wire = Wire(self, answer)
            with self.assertRaises(genai_errors.APIError):
                wire.turn(service_tier="flex")
        self.assertEqual(len(wire.sent), 1)


if __name__ == "__main__":
    unittest.main()
