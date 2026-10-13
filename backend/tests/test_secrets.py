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

from app import redact  # noqa: E402
from app.api import export  # noqa: E402
from app.assistant import conversations, loop, schema, tools  # noqa: E402
from app.assistant.tools import analysis  # noqa: E402

FAKE_CARTO = "fake-carto-0123456789abcdef"
FAKE_GEMINI = "fake-gemini-fedcba9876543210"
FAKE_TOKEN = "fake-token-aaaabbbbccccdddd"
FAKE_PASSWORD = "fake-password-1111222233334444"

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
}
secrets = list(fakes.values()) + sorted(redact._secrets)  # real .env keys too
out = {}
for name, code in probes.items():
    res = analysis.python_exec(conv, code)
    blob = json.dumps(res, default=str)
    out[name] = {
        "leak": any(s in blob for s in secrets),
        "ok": res["ok"],
        "refused": "refused" in blob or "does not allow" in blob,
        "missing": "has no attribute 'basemap_tile_url'" in blob,
    }
print(json.dumps({
    "config_holds_keys": (config.settings.CARTO_API_KEY == fakes["CARTO_API_KEY"]
                          and config.GEMINI_API_KEY == fakes["GEMINI_API_KEY"]),
    "env_clean": not any(n in os.environ for n in fakes),
    "probes": out,
}))
"""


class StartupScrubTest(unittest.TestCase):
    """The keys reach the server's settings and nothing else."""

    @classmethod
    def setUpClass(cls):
        fakes = {
            "CARTO_API_KEY": FAKE_CARTO,
            "GEMINI_API_KEY": FAKE_GEMINI,
            "SOME_SERVICE_TOKEN": FAKE_TOKEN,
            "DB_PASSWORD": FAKE_PASSWORD,
        }
        env = {**os.environ, **fakes}
        proc = subprocess.run(
            [sys.executable, "-c", CHILD, json.dumps(fakes), str(BACKEND / ".env")],
            cwd=BACKEND, env=env, capture_output=True, text=True, timeout=300,
        )
        if proc.returncode != 0:  # a traceback, not a key: the child prints no values
            raise AssertionError(f"child failed:\n{proc.stderr[-3000:]}")
        cls.report = json.loads(proc.stdout.strip().splitlines()[-1])

    def test_config_still_holds_the_keys(self):
        self.assertTrue(self.report["config_holds_keys"])

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
    tile-fetch error — comes back without it in the app's own loop."""

    def setUp(self):
        self.patch = mock.patch.object(export.settings, "CARTO_API_KEY", FAKE_CARTO)
        self.patch.start()
        redact.remember(FAKE_CARTO)
        self.url = export._basemap_tile_url("positron")
        self.assertIn(FAKE_CARTO, self.url)  # the private helper still keys tiles

    def tearDown(self):
        self.patch.stop()

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
