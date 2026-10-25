"""Conversations: bound to the principal that created it, and the store's limits
(app/assistant/conversations.py, app/api/assistant_api.py, mcp_state.py).

User B gets a 404 on every route that takes user A's conversation, and nothing
of A's changes; eviction never drops a busy conversation; idle ones expire.

Run from backend/ with the repo's venv:  python -m unittest discover -s tests
"""

from __future__ import annotations

import sys
import time
import unittest
import uuid
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi.testclient import TestClient  # noqa: E402

from app import identity  # noqa: E402
from app.api import assistant_api  # noqa: E402
from app.assistant import artifacts, conversations, mcp_server, mcp_state  # noqa: E402

import main  # noqa: E402
from multiuser_helpers import multi_user, signed_in  # noqa: E402

A, B = "a@example.org", "b@example.org"


class Binding(unittest.TestCase):
    def setUp(self):
        self.cid = uuid.uuid4().hex
        self.addCleanup(conversations.drop, self.cid)

    def test_the_creator_owns_it(self):
        conv = conversations.get_or_create(self.cid, owner=A)
        self.assertEqual(conv.owner, A)
        self.assertIs(conversations.get_or_create(self.cid, owner=A), conv)
        with self.assertRaises(conversations.NotYours):
            conversations.get_or_create(self.cid, owner=B)
        self.assertIsNone(conversations.get(self.cid, owner=B))
        self.assertFalse(conversations.drop(self.cid, owner=B))
        self.assertIs(conversations.get(self.cid, owner=A), conv)

    def test_single_user_mode_owner_is_the_local_user(self):
        conv = conversations.get_or_create(self.cid)
        self.assertEqual(conv.owner, "local")
        self.assertEqual(conversations.Conversation(id="x", created=0.0).owner, "local")


class RoutesRefuseOtherUsers(unittest.TestCase):
    def setUp(self):
        patch = multi_user()
        patch.start()
        self.addCleanup(patch.stop)
        self.client = TestClient(main.app)
        self.cid = uuid.uuid4().hex
        self.addCleanup(conversations.drop, self.cid)
        r = self.client.post(
            f"/api/assistant/conversations/{self.cid}/files?filename=curve.csv",
            content=b"intensity,proportion\n0,0\n1,1\n",
            headers=signed_in(A),
        )
        self.assertEqual(r.status_code, 200)
        self.conv = conversations.get(self.cid, owner=A)
        self.conv.curves["c"] = {"path": self.conv.uploads["curve.csv"]}
        self.art = artifacts.add(self.conv.uploads["curve.csv"], "curve.csv", "csv", "c", self.cid)

    def test_upload_into_anothers_conversation(self):
        r = self.client.post(
            f"/api/assistant/conversations/{self.cid}/files?filename=x.csv",
            content=b"a", headers=signed_in(B),
        )
        self.assertEqual(r.status_code, 404)
        self.assertEqual(sorted(self.conv.uploads), ["curve.csv"])

    def test_chat_in_anothers_conversation(self):
        choice = SimpleNamespace(provider="anthropic", model="m", effort=None, service_tier=None)
        with mock.patch.object(assistant_api, "_pick_model", return_value=choice), \
                mock.patch.object(assistant_api.loop, "run") as run:
            r = self.client.post(
                "/api/assistant/chat",
                json={"conversation_id": self.cid, "message": "hi"},
                headers=signed_in(B),
            )
        self.assertEqual(r.status_code, 404)
        run.assert_not_called()
        self.assertEqual(self.conv.messages, [])

    def test_curve_artifact_and_delete(self):
        r = self.client.get(f"/api/assistant/conversations/{self.cid}/curves/c", headers=signed_in(B))
        self.assertEqual(r.status_code, 404)
        r = self.client.get(f"/api/assistant/artifacts/{self.art.id}", headers=signed_in(B))
        self.assertEqual(r.status_code, 404)
        r = self.client.delete(f"/api/assistant/conversations/{self.cid}", headers=signed_in(B))
        self.assertEqual(r.json(), {"deleted": False})
        self.assertIs(conversations.get(self.cid, owner=A), self.conv)
        # and the owner still has all of it
        r = self.client.get(f"/api/assistant/artifacts/{self.art.id}", headers=signed_in(A))
        self.assertEqual(r.status_code, 200)
        r = self.client.get(f"/api/assistant/conversations/{self.cid}/curves/c", headers=signed_in(A))
        self.assertEqual(r.status_code, 200)

    def test_browsers_cannot_claim_mcp_scopes(self):
        for cid in ("mcp:agui-chat-1", "mcp-shared-scope"):
            with self.subTest(cid=cid):
                r = self.client.post(
                    f"/api/assistant/conversations/{cid}/files?filename=x.csv",
                    content=b"a", headers=signed_in(B),
                )
                self.assertEqual(r.status_code, 404)
                self.assertIsNone(conversations.get(cid))

    def test_app_state_is_per_user(self):
        self.client.post("/api/assistant/app_state", json={"file_id": "f-a"}, headers=signed_in(A))
        self.client.post("/api/assistant/app_state", json={"file_id": "f-b"}, headers=signed_in(B))
        self.assertEqual(mcp_state.read(A)["state"], {"file_id": "f-a"})
        self.assertEqual(mcp_state.read(B)["state"], {"file_id": "f-b"})
        # an MCP scope is its own principal and sees neither
        for cid in (mcp_server._MCP_CONVERSATION_ID, "mcp:agui-chat-1"):
            self.assertFalse(mcp_state.read(identity.scope_principal(cid))["available"])


class Limits(unittest.TestCase):
    def setUp(self):
        # A private store, so other test modules' conversations are not evicted
        for patch in (
            mock.patch.object(conversations, "_STORE", OrderedDict()),
            mock.patch.object(conversations, "MAX_CONVERSATIONS", 4),
        ):
            patch.start()
            self.addCleanup(patch.stop)
        self.addCleanup(lambda: [conversations._cleanup(c) for c in conversations._STORE.values()])

    def test_eviction_never_drops_a_busy_conversation(self):
        turn = conversations.get_or_create("turn")
        turn.turn_lock.acquire()
        self.addCleanup(turn.turn_lock.release)
        exe = conversations.get_or_create("exec")
        exe.exec_lock.acquire()
        self.addCleanup(exe.exec_lock.release)
        tool = conversations.get_or_create("tool")
        with tool.hold():
            for i in range(40):
                conversations.get_or_create(f"filler-{i}")
            for cid in ("turn", "exec", "tool"):
                self.assertIsNotNone(conversations.get(cid), cid)
        self.assertLessEqual(len(conversations._STORE), 4)
        self.assertIn("filler-39", conversations._STORE)

    def test_all_busy_runs_over_the_cap_rather_than_evicting(self):
        held = []
        for i in range(6):
            conv = conversations.get_or_create(f"busy-{i}")
            conv.turn_lock.acquire()
            held.append(conv)
        try:
            self.assertEqual(len(conversations._STORE), 6)
        finally:
            for conv in held:
                conv.turn_lock.release()

    def test_idle_conversations_expire_unless_busy(self):
        old = conversations.get_or_create("old")
        old_busy = conversations.get_or_create("old-busy")
        long_ago = time.time() - conversations.IDLE_TTL_S - 60
        old.last_used = old_busy.last_used = long_ago
        with old_busy.hold():
            old_busy.last_used = long_ago
            conversations.get_or_create("new")
            self.assertIsNone(conversations.get("old"))
            self.assertIsNotNone(conversations.get("old-busy"))

    def test_default_cap(self):
        from app import config

        self.assertGreaterEqual(config.ASSISTANT_MAX_CONVERSATIONS, 128)


if __name__ == "__main__":
    unittest.main()
