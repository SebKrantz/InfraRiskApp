"""AGUI's internal calls (app/identity.py, main.py, api/assistant_api.py): the
service token whose sha256 is AEI_SERVICE_TOKEN_SHA256, the start-up refusal
without it, /mcp refusing calls without it in multi-user mode, X-AEI-User
ignored on any request that does not carry it, and the REST routes AGUI calls
(the file push to `mcp:<scope>`, artifact downloads, an internal upload).

Multi-user mode is switched on by patching app.config. Run from backend/ with
the repo's venv:  python -m unittest discover -s tests
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi.testclient import TestClient  # noqa: E402
from starlette.datastructures import Headers  # noqa: E402

from app import config, datasets, identity  # noqa: E402
from app.assistant import artifacts, conversations  # noqa: E402

import main  # noqa: E402
from multiuser_helpers import HEADER, SECRET, TOKEN, internal, multi_user, signed_in  # noqa: E402

A, B = "a@example.org", "b@example.org"
CSV = b"name,lat,lon\na,-4.0,15.0\nb,-4.1,15.1\n"
INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "0"},
    },
}
MCP_HEADERS = {"Accept": "application/json, text/event-stream"}


class Token(unittest.TestCase):
    def test_only_the_right_bearer_token_is_internal(self):
        with multi_user():
            self.assertFalse(identity.internal(Headers({})))
            self.assertFalse(identity.internal(Headers(internal(token="a-guess"))))
            self.assertFalse(identity.internal(Headers({"Authorization": f"Basic {TOKEN}"})))
            self.assertFalse(identity.internal(Headers({"Authorization": TOKEN})))
            self.assertFalse(identity.internal(Headers({"Authorization": "Bearer "})))
            self.assertTrue(identity.internal(Headers(internal())))
            self.assertTrue(identity.internal(Headers({"Authorization": f"bearer {TOKEN}"})))
            with mock.patch.object(config, "AEI_SERVICE_TOKEN_SHA256", ""):
                self.assertFalse(identity.internal(Headers(internal())))

    def test_multi_user_mode_refuses_to_start_without_the_token_digest(self):
        edge = hashlib.sha256(SECRET.encode()).hexdigest()
        for digest in ("", "abc", "z" * 64, edge + "0"):
            with self.subTest(digest=digest), mock.patch.multiple(
                config,
                AEI_AUTH_HEADER=HEADER,
                AEI_EDGE_SECRET_SHA256=edge,
                AEI_SERVICE_TOKEN_SHA256=digest,
            ):
                with self.assertRaisesRegex(RuntimeError, "AEI_SERVICE_TOKEN_SHA256"):
                    config.check_identity_settings()

    def test_single_user_mode_needs_no_token(self):
        with mock.patch.multiple(
            config, AEI_AUTH_HEADER="", AEI_EDGE_SECRET_SHA256="", AEI_SERVICE_TOKEN_SHA256=""
        ):
            config.check_identity_settings()

    def test_mcp_takes_internal_calls_only_in_multi_user_mode(self):
        # One lifespan for the whole test: the MCP session manager runs once per process.
        with TestClient(main.app, base_url="http://127.0.0.1:8000") as client:

            def post(headers: dict[str, str]) -> int:
                return client.post("/mcp", json=INITIALIZE, headers={**MCP_HEADERS, **headers}).status_code

            with multi_user():
                self.assertEqual(post({}), 401)
                self.assertEqual(post(internal(token="a-guess")), 401)
                self.assertEqual(post(signed_in(A)), 401)  # a browser identity is not enough
                self.assertEqual(post({"X-AEI-User": A}), 401)
                self.assertEqual(post(internal(user=A, scope="chat-1")), 200)

                # over the real transport: a scope working for A lists A's web dataset
                r = client.post("/api/upload", files={"file": ("a.csv", CSV, "text/csv")},
                                headers=signed_in(A))
                fid = r.json()["file_id"]
                self.addCleanup(datasets.STORE.pop, fid, None)
                body = {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                        "params": {"name": "list_datasets", "arguments": {}}}
                for user, scope, sees in ((A, "chat-1", True), (B, "chat-2", False)):
                    self.addCleanup(conversations.drop, f"mcp:{scope}")
                    r = client.post("/mcp", json=body,
                                    headers={**MCP_HEADERS, **internal(user=user, scope=scope)})
                    self.assertEqual(r.status_code, 200)
                    self.assertEqual(fid in r.text, sees, r.text[:300])
            self.assertEqual(post({}), 200)  # single-user mode: as before


class ServiceUser(unittest.TestCase):
    def test_x_aei_user_counts_only_with_the_token(self):
        with multi_user():
            self.assertIsNone(identity.service_user(Headers({"X-AEI-User": A})))
            self.assertIsNone(identity.service_user(Headers(internal(user=A, token="a-guess"))))
            self.assertIsNone(identity.service_user(Headers({**signed_in(B), "X-AEI-User": A})))
            self.assertIsNone(identity.service_user(Headers(internal(user="not-an-email"))))
            self.assertEqual(identity.service_user(Headers(internal(user=" A@Example.org"))), A)
        # single-user mode: nobody is anybody but the local user
        self.assertIsNone(identity.service_user(Headers(internal(user=A))))

    def test_whom_a_request_acts_for(self):
        with multi_user():
            self.assertEqual(identity.resolve(Headers({**signed_in(B), "X-AEI-User": A})), B)
            self.assertIsNone(identity.resolve(Headers({"X-AEI-User": A})))
            self.assertEqual(identity.resolve(Headers(internal(user=A))), A)
            self.assertEqual(identity.resolve(Headers(internal(user=A, scope="chat-1"))), "mcp:chat-1")
            self.assertEqual(identity.resolve(Headers(internal())), "mcp:mcp-shared-scope")
        self.assertEqual(identity.resolve(Headers(internal(user=A))), config.AEI_LOCAL_USER)

    def test_x_aei_user_on_a_browser_request_changes_nothing(self):
        with multi_user():
            client = TestClient(main.app)
            r = client.post("/api/upload", files={"file": ("a.csv", CSV, "text/csv")},
                            headers=signed_in(A))
            fid = r.json()["file_id"]
            self.addCleanup(datasets.STORE.pop, fid, None)
            forged = {**signed_in(B), "X-AEI-User": A}
            self.assertEqual(client.get(f"/api/upload/{fid}", headers=forged).status_code, 404)
            self.assertEqual(client.get(f"/api/upload/{fid}", headers={"X-AEI-User": A}).status_code, 401)
            self.assertEqual(client.get(f"/api/upload/{fid}", headers=internal(user=A)).status_code, 200)


class InternalRoutes(unittest.TestCase):
    """The REST routes AGUI calls, in multi-user mode."""

    def setUp(self):
        patch = multi_user()
        patch.start()
        self.addCleanup(patch.stop)
        self.client = TestClient(main.app)

    def _push(self, cid: str, headers: dict[str, str]):
        return self.client.post(
            f"/api/assistant/conversations/{cid}/files?filename=a.csv", content=CSV, headers=headers
        )

    def test_the_file_push_acts_for_the_scope(self):
        self.addCleanup(conversations.drop, "mcp:push-1")
        self.assertEqual(self._push("mcp:push-1", {}).status_code, 401)
        self.assertEqual(self._push("mcp:push-1", internal(user=A, token="a-guess")).status_code, 401)
        self.assertEqual(self._push("mcp:push-1", signed_in(A)).status_code, 404)
        self.assertIsNone(conversations.get("mcp:push-1"))
        r = self._push("mcp:push-1", internal(user=A))
        self.assertEqual(r.status_code, 200, r.text)
        conv = conversations.get("mcp:push-1")
        self.assertEqual((conv.owner, conv.user), ("mcp:push-1", A))
        self.assertIn("a.csv", conv.uploads)
        # the scope works for A now: B's chat cannot use it
        self.assertEqual(self._push("mcp:push-1", internal(user=B)).status_code, 404)
        self.assertEqual(self._push("mcp:push-1", internal()).status_code, 200)

    def _artifact(self, conv: conversations.Conversation) -> str:
        dest = conv.workdir / "chart.png"
        dest.write_bytes(b"\x89PNG fake")
        return artifacts.add(dest, "chart.png", "image", "chart", conv.id).id

    def test_artifact_downloads(self):
        scope = conversations.get_or_create("mcp:art-1", owner="mcp:art-1", user=A)
        chat = conversations.get_or_create("chat-of-a", owner=A)
        for cid in (scope.id, chat.id):
            self.addCleanup(conversations.drop, cid)
        url = "/api/assistant/artifacts/{}".format
        made_by_scope, made_in_chat = self._artifact(scope), self._artifact(chat)
        cases = (
            (made_by_scope, internal(user=A), 200),
            (made_by_scope, internal(), 200),
            (made_by_scope, internal(user=B), 404),  # the scope works for A
            (made_by_scope, signed_in(A), 404),  # a browser never reads a scope's
            (made_by_scope, {}, 401),
            (made_by_scope, internal(user=A, token="a-guess"), 401),
            (made_in_chat, internal(user=A), 200),  # A's own
            (made_in_chat, internal(user=B), 404),
            (made_in_chat, internal(), 404),
            (made_in_chat, signed_in(A), 200),
            (made_in_chat, signed_in(B), 404),
        )
        for art, headers, status in cases:
            with self.subTest(art=art, headers=sorted(headers)):
                self.assertEqual(self.client.get(url(art), headers=headers).status_code, status)

    def test_an_internal_upload_with_a_scope_is_the_scopes(self):
        r = self.client.post("/api/upload", files={"file": ("a.csv", CSV, "text/csv")},
                             headers=internal(user=A, scope="up-1"))
        self.assertEqual(r.status_code, 200, r.text)
        fid = r.json()["file_id"]
        self.addCleanup(datasets.STORE.pop, fid, None)
        self.assertEqual(datasets.STORE[fid]["owner"], "mcp:up-1")
        self.assertEqual(self.client.get(f"/api/upload/{fid}", headers=signed_in(A)).status_code, 404)
        # the scope going takes it along
        conversations.get_or_create("mcp:up-1", owner="mcp:up-1")
        conversations.drop("mcp:up-1")
        self.assertNotIn(fid, datasets.STORE)


class EnvironmentScrub(unittest.TestCase):
    """config.py reads the AEI_* settings, then scrub_environment() pops every
    secret-named variable. Multi-user mode must survive that: the two digests
    leave the environment (and python_exec, and child processes, with them), the
    plain settings stay, and identity works from what config.py kept."""

    def test_multi_user_mode_survives_the_scrub(self):
        backend = Path(__file__).resolve().parents[1]
        edge = hashlib.sha256(SECRET.encode()).hexdigest()
        token = hashlib.sha256(TOKEN.encode()).hexdigest()
        env = {
            **os.environ,
            "AEI_AUTH_HEADER": HEADER,
            "AEI_EDGE_SECRET_SHA256": edge,
            "AEI_SERVICE_TOKEN_SHA256": token,
            "AEI_ADMIN_EMAILS": "boss@example.org",
            "AEI_LOCAL_USER": "someone",
        }
        probe = textwrap.dedent(f"""
            import json, os, posix, subprocess, sys
            sys.path.insert(0, {str(backend)!r})
            from starlette.datastructures import Headers
            from app import config, identity
            from app.redact import redact_text
            names = ("AEI_AUTH_HEADER", "AEI_EDGE_SECRET_SHA256", "AEI_SERVICE_TOKEN_SHA256",
                     "AEI_ADMIN_EMAILS", "AEI_LOCAL_USER")
            child = subprocess.run(
                [sys.executable, "-c", "import json, os; print(json.dumps(sorted(os.environ)))"],
                capture_output=True, text=True, check=True,
            )
            print(json.dumps({{
                "environ": [n for n in names if n in os.environ],
                "posix": [n for n in names if n.encode() in posix.environ],
                "child": [n for n in names if n in json.loads(child.stdout)],
                "multi_user": identity.multi_user(),
                "internal": identity.internal(Headers({{"Authorization": "Bearer {TOKEN}"}})),
                "guess": identity.internal(Headers({{"Authorization": "Bearer a-guess"}})),
                "user": identity.service_user(Headers({{"Authorization": "Bearer {TOKEN}",
                                                       "X-AEI-User": "a@example.org"}})),
                "browser": identity.resolve(Headers({{"{HEADER}": "a@example.org",
                                                     "X-AEI-Edge": "{SECRET}"}})),
                "admin": identity.is_admin("boss@example.org"),
                "local": config.AEI_LOCAL_USER,
                "plain_kept": redact_text("{HEADER} boss@example.org someone"),
                "digests_masked": "{edge}" not in redact_text("{edge} {token}"),
            }}))
        """)
        out = subprocess.run(
            [sys.executable, "-c", probe], env=env, cwd=backend,
            capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        got = json.loads(out.stdout.strip().splitlines()[-1])
        plain = ["AEI_AUTH_HEADER", "AEI_ADMIN_EMAILS", "AEI_LOCAL_USER"]
        self.assertEqual(got["environ"], plain)  # the two digests are gone
        self.assertEqual(got["posix"], plain)
        self.assertEqual(got["child"], plain)
        self.assertTrue(got["multi_user"])
        self.assertTrue(got["internal"])
        self.assertFalse(got["guess"])
        self.assertEqual(got["user"], "a@example.org")
        self.assertEqual(got["browser"], "a@example.org")
        self.assertTrue(got["admin"])
        self.assertEqual(got["local"], "someone")
        self.assertEqual(got["plain_kept"], f"{HEADER} boss@example.org someone")
        self.assertTrue(got["digests_masked"])

    def test_nothing_reads_the_aei_settings_from_the_environment_later(self):
        app = Path(__file__).resolve().parents[1] / "app"
        late = [
            str(path.relative_to(app.parent))
            for path in app.rglob("*.py")
            if path.name != "config.py" and "AEI_" in path.read_text()
            and ("environ" in path.read_text() or "getenv" in path.read_text())
        ]
        self.assertEqual(late, [])


if __name__ == "__main__":
    unittest.main()
