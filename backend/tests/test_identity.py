"""Identity and the trust boundary (app/identity.py, main.py): 401 without a
signed-in user or with a forged one in multi-user mode, single-user mode as
before, the startup refusal, cross-site writes, admin-only routes and CORS.

Multi-user mode is switched on by patching app.config. Run from backend/ with
the repo's venv:  python -m unittest discover -s tests
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from starlette.datastructures import Headers  # noqa: E402

from app import config, identity  # noqa: E402

import main  # noqa: E402
from multiuser_helpers import HEADER, multi_user, signed_in  # noqa: E402


class SingleUserMode(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)

    def test_no_identity_needed_and_everything_belongs_to_the_local_user(self):
        self.assertFalse(identity.multi_user())
        self.assertEqual(identity.resolve({}), config.AEI_LOCAL_USER)
        self.assertEqual(self.client.get("/api/hazards").status_code, 200)

    def test_identity_headers_are_ignored(self):
        self.assertEqual(identity.resolve(signed_in("a@example.org")), config.AEI_LOCAL_USER)

    def test_the_local_user_is_an_admin(self):
        self.assertEqual(self.client.post("/api/tiles/clear-cache").status_code, 200)

    def test_mcp_scopes_belong_to_the_local_user_too(self):
        self.assertEqual(identity.scope_principal("mcp:abc"), config.AEI_LOCAL_USER)
        self.assertFalse(identity.reserved_conversation_id("mcp:abc"))


class MultiUserMode(unittest.TestCase):
    def setUp(self):
        patch = multi_user(admins=("boss@example.org",))
        patch.start()
        self.addCleanup(patch.stop)
        self.client = TestClient(main.app)

    def test_no_identity_is_401(self):
        self.assertEqual(self.client.get("/api/hazards").status_code, 401)

    def test_a_forged_email_without_the_edge_secret_is_401(self):
        r = self.client.get("/api/hazards", headers={HEADER: "boss@example.org"})
        self.assertEqual(r.status_code, 401)

    def test_a_wrong_edge_secret_is_401(self):
        r = self.client.get("/api/hazards", headers=signed_in("boss@example.org", "guess"))
        self.assertEqual(r.status_code, 401)

    def test_the_edge_secret_without_an_email_is_401(self):
        for email in ("", "not-an-email", "two@at@signs"):
            with self.subTest(email=email):
                r = self.client.get("/api/hazards", headers=signed_in(email))
                self.assertEqual(r.status_code, 401)

    def test_a_signed_in_user_gets_through(self):
        r = self.client.get("/api/hazards", headers=signed_in("a@example.org"))
        self.assertEqual(r.status_code, 200)

    def test_emails_are_lower_cased(self):
        self.assertEqual(identity.resolve(Headers(signed_in(" A@Example.ORG "))), "a@example.org")

    def test_health_and_non_api_paths_stay_open(self):
        self.assertEqual(self.client.get("/health").status_code, 200)

    def test_cross_site_writes_are_refused(self):
        headers = {**signed_in("a@example.org"), "Sec-Fetch-Site": "same-site"}
        self.assertEqual(self.client.post("/api/upload", headers=headers).status_code, 403)
        headers["Sec-Fetch-Site"] = "same-origin"
        self.assertNotEqual(self.client.post("/api/upload", headers=headers).status_code, 403)

    def test_clear_cache_is_admin_only(self):
        r = self.client.post("/api/tiles/clear-cache", headers=signed_in("a@example.org"))
        self.assertEqual(r.status_code, 403)
        r = self.client.post("/api/tiles/clear-cache", headers=signed_in("boss@example.org"))
        self.assertEqual(r.status_code, 200)

    def test_mcp_scopes_are_their_own_principals(self):
        self.assertEqual(identity.scope_principal("mcp:abc"), "mcp:abc")
        self.assertEqual(identity.scope_principal("mcp-shared-scope"), "mcp:mcp-shared-scope")
        self.assertTrue(identity.reserved_conversation_id("mcp:abc"))
        self.assertFalse(identity.reserved_conversation_id("0f3a9c"))


class StartupRefusal(unittest.TestCase):
    def test_multi_user_mode_needs_the_edge_digest(self):
        for digest in ("", "abc", "z" * 64):
            with self.subTest(digest=digest), mock.patch.multiple(
                config, AEI_AUTH_HEADER=HEADER, AEI_EDGE_SECRET_SHA256=digest
            ):
                with self.assertRaisesRegex(RuntimeError, "AEI_EDGE_SECRET_SHA256"):
                    config.check_identity_settings()

    def test_a_digest_or_single_user_mode_starts(self):
        with multi_user():
            config.check_identity_settings()
        with mock.patch.multiple(config, AEI_AUTH_HEADER="", AEI_EDGE_SECRET_SHA256=""):
            config.check_identity_settings()


class Cors(unittest.TestCase):
    def test_no_cross_origin_access_by_default(self):
        client = TestClient(main.app)
        r = client.get("/api/hazards", headers={"Origin": "https://evil.example"})
        self.assertNotIn("access-control-allow-origin", r.headers)
        r = client.options(
            "/api/hazards",
            headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"},
        )
        self.assertNotIn("access-control-allow-origin", r.headers)

    def test_only_the_configured_origins(self):
        app = FastAPI()
        app.get("/api/x")(lambda: {"ok": True})
        with mock.patch.object(config, "AEI_CORS_ORIGINS", ["https://risk.aeilabs.xyz"]):
            main.configure_cors(app)
        client = TestClient(app)
        ok = client.get("/api/x", headers={"Origin": "https://risk.aeilabs.xyz"})
        self.assertEqual(ok.headers.get("access-control-allow-origin"), "https://risk.aeilabs.xyz")
        bad = client.get("/api/x", headers={"Origin": "https://evil.example"})
        self.assertNotIn("access-control-allow-origin", bad.headers)


if __name__ == "__main__":
    unittest.main()
