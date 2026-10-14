"""No key leaves the server: not in a tool result, an error message, python_exec's
view of the environment, or a file the static route serves.

Run: cd backend && python -m unittest tests.test_secrets
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import anyio  # noqa: E402
from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402

from app import config, redact  # noqa: E402
from app.api import export  # noqa: E402
from app.assistant import conversations, loop, mcp_server, sandbox, schema, tools  # noqa: E402
from app.assistant.tools import analysis  # noqa: E402

FAKE_CARTO = "fake-carto-0123456789abcdef"
FAKE_GEMINI = "fake-gemini-fedcba9876543210"
FAKE_TOKEN = "fake-token-aaaabbbbccccdddd"
FAKE_PASSWORD = "fake-password-1111222233334444"
FAKE_VERTEX = "fake-vertex-0011223344556677"
# What Google's credential files hold, made up: the shapes redact.py masks, nothing real.
FAKE_REFRESH = "1//0fake-refresh-token-0123456789abcdef"
FAKE_ACCESS = "ya29.a0fake-access-token_0123456789abcdef"
FAKE_CLIENT_SECRET = "fake-client-secret-0123456789"
FAKE_PEM_BODY = "ZmFrZS1wcml2YXRlLWtleS1tYXRlcmlhbA=="
FAKE_PEM = f"-----BEGIN PRIVATE KEY-----\n{FAKE_PEM_BODY}\n-----END PRIVATE KEY-----\n"
FAKE_SA_KEY = json.dumps({"type": "service_account", "private_key_id": "fake-id",
                          "private_key": FAKE_PEM, "client_email": "sa@fake.iam.example"})

# Run in a fresh interpreter with the fake keys in its environment, as the server
# starts: import the config (which reads and then scrubs the keys), then let
# python_exec look for them every way it might. It prints only booleans, so a
# failure cannot echo a real key from backend/.env into the test output.
CHILD = r"""
import json, os, sys
sys.path.insert(0, os.getcwd())
from app import config, redact
from app.assistant import conversations
from app.assistant.tools import analysis

fakes = json.loads(sys.argv[1])
env_file = sys.argv[2]
extra = json.loads(sys.argv[3])  # a readable sibling of the key file; text the key file holds
conv = conversations.get_or_create("secrets-test")
probes = {
    "environ": "import os; print(dict(os.environ))",
    "getenv": "import os; print([os.getenv(n) for n in %r])" % (list(fakes),),
    "posix_environ": "import posix; print(posix.environ)",
    "proc_environ": "print(open('/proc/self/environ').read())",
    "env_file": "print(open(%r).read())" % (env_file,),
    "subprocess": "import subprocess; print(subprocess.run(['env'], capture_output=True).stdout)",
    "os_system": "import os; os.system('env')",
    "os_popen": "import os; print(os.popen('env').read())",
    "basemap_url": "print(export.basemap_tile_url('positron'))",
    "basemap_table": "print(export.BASEMAP_TILE_URLS)",
    # Google credentials: the service-account key GOOGLE_APPLICATION_CREDENTIALS names (it sits
    # in the temp directory, which python_exec may otherwise read), gcloud's ADC file, and the
    # library route to the key
    "sa_key_file": "print(open(%r).read())" % (fakes["GOOGLE_APPLICATION_CREDENTIALS"],),
    "sa_key_loader": "import google.auth; google.auth.load_credentials_from_file(%r)"
                     % (fakes["GOOGLE_APPLICATION_CREDENTIALS"],),
    "gcloud_adc": "import os; print(open(os.path.expanduser("
                  "'~/.config/gcloud/application_default_credentials.json')).read())",
    "key_file_sibling": "print(open(%r).read())" % (extra["sibling"],),
}
secrets = (list(fakes.values()) + extra["markers"]
           + sorted(redact._secrets))  # real .env keys too
out = {}
key_path = fakes["GOOGLE_APPLICATION_CREDENTIALS"]
for name, code in probes.items():
    res = analysis.python_exec(conv, code)
    blob = json.dumps(res, default=str)
    # a refusal repeats the path the probe itself named; its text is what must not come out
    own = secrets if not name.startswith("sa_key") else [s for s in secrets if s != key_path]
    out[name] = {
        "leak": any(s in blob for s in own),
        "ok": res["ok"],
        "refused": "refused" in blob or "does not allow" in blob,
        "missing": "has no attribute 'basemap_tile_url'" in blob,
    }
print(json.dumps({
    "config_holds_keys": (config.settings.CARTO_API_KEY == fakes["CARTO_API_KEY"]
                          and config.GEMINI_API_KEY == fakes["GEMINI_API_KEY"]),
    "config_holds_vertex_settings": (
        config.VERTEX_API_KEY == fakes["VERTEX_API_KEY"]
        and config.GOOGLE_APPLICATION_CREDENTIALS == fakes["GOOGLE_APPLICATION_CREDENTIALS"]),
    "env_clean": not any(n in os.environ for n in fakes),
    "probes": out,
}))
"""


class StartupScrubTest(unittest.TestCase):
    """The keys reach the server's settings and nothing else."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        key_file = Path(os.path.realpath(cls.tmp.name)) / "sa-key.json"
        key_file.write_text(FAKE_SA_KEY)
        sibling = key_file.with_name("notes.txt")
        sibling.write_text("readable")
        fakes = {
            "CARTO_API_KEY": FAKE_CARTO,
            "GEMINI_API_KEY": FAKE_GEMINI,
            "SOME_SERVICE_TOKEN": FAKE_TOKEN,
            "DB_PASSWORD": FAKE_PASSWORD,
            "VERTEX_API_KEY": FAKE_VERTEX,
            "GOOGLE_APPLICATION_CREDENTIALS": str(key_file),
        }
        env = {**os.environ, **fakes}
        extra = {"sibling": str(sibling), "markers": [FAKE_PEM_BODY, "fake-id"]}
        proc = subprocess.run(
            [sys.executable, "-c", CHILD, json.dumps(fakes), str(BACKEND / ".env"),
             json.dumps(extra)],
            cwd=BACKEND, env=env, capture_output=True, text=True, timeout=300,
        )
        if proc.returncode != 0:  # a traceback, not a key: the child prints no values
            raise AssertionError(f"child failed:\n{proc.stderr[-3000:]}")
        cls.report = json.loads(proc.stdout.strip().splitlines()[-1])

    def test_config_still_holds_the_keys(self):
        self.assertTrue(self.report["config_holds_keys"])

    def test_config_holds_the_vertex_settings_the_scrub_removed(self):
        """VERTEX_API_KEY and GOOGLE_APPLICATION_CREDENTIALS are secret-named: out of the
        environment (test_environment_is_scrubbed), but read by config first."""
        self.assertTrue(self.report["config_holds_vertex_settings"])

    def test_environment_is_scrubbed(self):
        self.assertTrue(self.report["env_clean"])

    def test_python_exec_finds_no_key(self):
        for name, probe in self.report["probes"].items():
            with self.subTest(probe=name):
                self.assertFalse(probe["leak"], f"{name} returned a key")

    def test_the_ways_out_are_refused(self):
        p = self.report["probes"]
        for name in ("proc_environ", "env_file", "subprocess", "os_system", "os_popen"):
            with self.subTest(probe=name):
                self.assertFalse(p[name]["ok"])
                self.assertTrue(p[name]["refused"], name)

    def test_google_credential_files_are_refused(self):
        p = self.report["probes"]
        for name in ("sa_key_file", "sa_key_loader", "gcloud_adc"):
            with self.subTest(probe=name):
                self.assertFalse(p[name]["ok"])
                self.assertTrue(p[name]["refused"], name)

    def test_the_refusal_is_the_key_file_not_its_directory(self):
        probe = self.report["probes"]["key_file_sibling"]
        self.assertTrue(probe["ok"])
        self.assertFalse(probe["refused"])

    def test_keyed_basemap_url_is_not_in_the_facade(self):
        self.assertFalse(self.report["probes"]["basemap_url"]["ok"])
        self.assertTrue(self.report["probes"]["basemap_url"]["missing"])
        self.assertTrue(self.report["probes"]["basemap_table"]["ok"])


class ScrubEnvironmentTest(unittest.TestCase):
    def test_removes_secret_names_and_remembers_values(self):
        import posix

        with mock.patch.dict(os.environ, {"X_FAKE_API_KEY": FAKE_CARTO, "X_FAKE_PLAIN": "kept"}), \
                mock.patch.dict(posix.environ, {b"X_FAKE_START_TOKEN": FAKE_TOKEN.encode()}):
            removed = redact.scrub_environment()
            self.assertIn("X_FAKE_API_KEY", removed)
            self.assertIsNone(os.getenv("X_FAKE_API_KEY"))
            self.assertEqual(os.getenv("X_FAKE_PLAIN"), "kept")
            self.assertNotIn(b"X_FAKE_START_TOKEN", posix.environ)
        self.assertEqual(redact.redact_text(f"a {FAKE_CARTO} b {FAKE_TOKEN}"),
                         "a [redacted] b [redacted]")

    def test_python_exec_after_the_scrub(self):
        with mock.patch.dict(os.environ, {"X_FAKE_SECRET": FAKE_PASSWORD}):
            redact.scrub_environment()
            conv = conversations.get_or_create("secrets-test-inproc")
            res = analysis.python_exec(conv, "import os; print(dict(os.environ))")
        self.assertTrue(res["ok"])
        self.assertNotIn(FAKE_PASSWORD, json.dumps(res, default=str))


class RedactTest(unittest.TestCase):
    def test_url_credentials_are_stripped(self):
        cases = {
            "https://t.example/1/2/3.png?key=abc": "https://t.example/1/2/3.png",
            "https://t.example/a?key=abc&x=1": "https://t.example/a?x=1",
            "https://t.example/a?x=1&api_key=abc": "https://t.example/a?x=1",
            "https://t.example/a?x=1&token=abc&y=2": "https://t.example/a?x=1&y=2",
            "https://t.example/a?access_token=abc#frag": "https://t.example/a#frag",
            "https://t.example/a?keyword=kept&monkey=kept": "https://t.example/a?keyword=kept&monkey=kept",
            "sort key=value stays": "sort key=value stays",
        }
        for raw, want in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(redact.redact_text(raw), want)

    def test_nested_values(self):
        redact.remember(FAKE_GEMINI)
        out = redact.redact({"a": [f"x{FAKE_GEMINI}y", (1, FAKE_GEMINI)], "n": 3})
        self.assertEqual(out, {"a": ["x[redacted]y", (1, "[redacted]")], "n": 3})


def _leaky(fn):
    return tools.ToolSpec(name="leaky_test_tool", description="", params={"type": "object"},
                          side="server", fn=fn)


class ToolResultTest(unittest.TestCase):
    """What used to carry the CARTO key out — a keyed tile URL in a result or in a
    tile-fetch error — comes back without it, over MCP and in the app's own loop."""

    def setUp(self):
        self.patch = mock.patch.object(export.settings, "CARTO_API_KEY", FAKE_CARTO)
        self.patch.start()
        redact.remember(FAKE_CARTO)
        self.url = export._basemap_tile_url("positron")
        self.assertIn(FAKE_CARTO, self.url)  # the private helper still keys tiles

    def tearDown(self):
        self.patch.stop()

    def test_mcp_result(self):
        spec = _leaky(lambda conv: {"tiles": self.url, "note": f"Bearer {FAKE_CARTO}"})
        out = anyio.run(mcp_server._run, spec, None, {})
        blob = json.dumps(out)
        self.assertNotIn(FAKE_CARTO, blob)
        self.assertNotIn("key=", blob)
        self.assertTrue(out["tiles"].startswith("https://basemaps.cartocdn.com/"))

    def test_mcp_errors(self):
        def fails(conv):
            raise RuntimeError(f"401 Client Error: Unauthorized for url: {self.url}")

        def tool_error(conv):
            raise ToolError(f"tile fetch failed: {self.url}")

        for fn in (fails, tool_error):
            with self.subTest(fn=fn.__name__):
                with self.assertRaises(ToolError) as caught:
                    anyio.run(mcp_server._run, _leaky(fn), None, {})
                self.assertNotIn(FAKE_CARTO, str(caught.exception))
                self.assertIn("basemaps.cartocdn.com", str(caught.exception))

    def test_mcp_python_exec_output(self):
        spec = tools.REGISTRY["python_exec"]
        out = anyio.run(mcp_server._run, spec, None, {"code": f"print({self.url!r})"})
        self.assertTrue(out["ok"])
        self.assertNotIn(FAKE_CARTO, json.dumps(out))

    def test_app_loop(self):
        def fails(conv):
            raise RuntimeError(f"401 Client Error: Unauthorized for url: {self.url}")

        ok = _leaky(lambda conv: {"tiles": self.url})
        bad = _leaky(fails)
        conv = conversations.get_or_create("secrets-test-loop")
        call = schema.ToolCall(id="t1", name="leaky_test_tool", args={})
        for spec in (ok, bad):
            with self.subTest(ok=spec is ok), mock.patch.dict(tools.REGISTRY, {spec.name: spec}):
                res = loop._run_server_tool(conv, call)
                self.assertEqual(res["ok"], spec is ok)
                self.assertNotIn(FAKE_CARTO, json.dumps(res["content"]))


class GoogleCredentialTest(unittest.TestCase):
    """What python_exec cannot print of Google's credential files: gcloud's ADC file and a
    service-account key. The file reads are refused (StartupScrubTest); this is the second
    layer, the masking of anything shaped like them in what a tool returns, and the sandbox's
    own rule that these files stay unreadable wherever they are kept."""

    ADC = {"account": "", "client_id": "764086051850-fake.apps.example",
           "client_secret": FAKE_CLIENT_SECRET, "refresh_token": FAKE_REFRESH,
           "type": "authorized_user", "universe_domain": "googleapis.com"}
    SECRETS = (FAKE_REFRESH, FAKE_ACCESS, FAKE_CLIENT_SECRET, FAKE_PEM_BODY)

    def test_the_shapes_are_masked_in_text(self):
        cases = {
            "refresh token": (f"token {FAKE_REFRESH} end", "token [redacted] end"),
            "access token": (f"Authorization: Bearer {FAKE_ACCESS}\n",
                             "Authorization: Bearer [redacted]\n"),
            "PEM block": (f"key:\n{FAKE_PEM}done", "key:\n[redacted]\ndone"),
            "PEM RSA block": ("-----BEGIN RSA PRIVATE KEY-----\nabc\n-----END RSA PRIVATE KEY-----",
                              "[redacted]"),
            "PEM cut off inside the key": (f"head -----BEGIN PRIVATE KEY-----\n{FAKE_PEM_BODY[:12]}",
                                           "head [redacted]"),
        }
        for name, (raw, want) in cases.items():
            with self.subTest(name):
                self.assertEqual(redact.redact_text(raw), want)

    def test_the_fields_are_masked_in_json_and_in_a_printed_dict(self):
        for name, text in {"adc json": json.dumps(self.ADC, indent=2),
                           "adc dict": str(self.ADC),
                           "sa key json": FAKE_SA_KEY,
                           "sa key dict": str(json.loads(FAKE_SA_KEY))}.items():
            with self.subTest(name):
                out = redact.redact_text(text)
                self.assertFalse(any(secret in out for secret in self.SECRETS), out)
                self.assertIn("[redacted]", out)
        # what is not a secret stays, so the output is still readable
        out = redact.redact_text(json.dumps(self.ADC))
        self.assertIn(self.ADC["client_id"], out)
        self.assertIn('"type": "authorized_user"', out)
        self.assertEqual(json.loads(out)["refresh_token"], "[redacted]")
        self.assertEqual(json.loads(redact.redact_text(FAKE_SA_KEY))["private_key"], "[redacted]")

    def test_ordinary_text_is_left_alone(self):
        text = ("1//short and ya29.short, the private_keys of the bridge, "
                "refresh_token is a word, 'client_secret' alone, a -----BEGIN CERTIFICATE----- line")
        self.assertEqual(redact.redact_text(text), text)

    def test_python_exec_output_is_masked_over_mcp_and_in_the_app(self):
        """Credentials printed from memory, as a file read would print them if it ever got
        through: nothing of them reaches the model or the MCP client."""
        code = (f"import json; print({FAKE_ACCESS!r}); print(json.dumps({self.ADC!r}));"
                f"print({FAKE_SA_KEY!r})")
        out = anyio.run(mcp_server._run, tools.REGISTRY["python_exec"], None, {"code": code})
        self.assertTrue(out["ok"])
        self.assertFalse(any(secret in json.dumps(out) for secret in self.SECRETS))
        conv = conversations.get_or_create("secrets-test-google")
        res = loop._run_server_tool(conv, schema.ToolCall(id="g1", name="python_exec",
                                                          args={"code": code}))
        self.assertTrue(res["ok"])
        self.assertFalse(any(secret in json.dumps(res["content"]) for secret in self.SECRETS))

    def test_the_sandbox_keeps_credential_files_unreadable_wherever_they_are(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = os.path.realpath(tmp)  # the temp directory itself is readable to python_exec
            policy = sandbox.Policy.for_workdir(Path(tmp) / "work")
            key, adc, other = (os.path.join(tmp, n) for n in
                               ("sa-key.json", "application_default_credentials.json", "notes.txt"))
            for path in (key, adc, other):
                Path(path).write_text("{}")
                self.assertTrue(policy.can_read(path), path)  # readable, until named as a credential
            with mock.patch.object(config, "GOOGLE_APPLICATION_CREDENTIALS", key):
                self.assertFalse(policy.can_read(key))
                self.assertTrue(policy.can_read(other))
            with mock.patch.dict(os.environ, {"CLOUDSDK_CONFIG": tmp}):  # gcloud's config dir
                self.assertFalse(policy.can_read(adc))
                self.assertTrue(policy.can_read(other))
        default_adc = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
        self.assertFalse(policy.can_read(os.path.realpath(default_adc)))


class StaticRouteTest(unittest.TestCase):
    """The production SPA route serves files under static/ and nothing else."""

    def test_stays_inside_static(self):
        import main

        with tempfile.TemporaryDirectory() as tmp:
            static = Path(tmp) / "static"
            (static / "assets").mkdir(parents=True)
            (static / "assets" / "app.js").write_text("ok")
            (Path(tmp) / "outside.txt").write_text("secret")
            with mock.patch.object(main, "STATIC_DIR", static):
                self.assertEqual(main._static_file("assets/app.js"),
                                 (static / "assets" / "app.js").resolve())
                for path in ("/etc/hosts", "../outside.txt", "assets/../../outside.txt",
                             str(Path(tmp) / "outside.txt"), "", "assets"):
                    with self.subTest(path=path):
                        self.assertIsNone(main._static_file(path))


if __name__ == "__main__":
    unittest.main()
