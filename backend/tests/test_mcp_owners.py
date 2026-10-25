"""MCP scopes as owners (multi-user mode): what a scope loads is `mcp:<scope>`'s;
a scope AGUI runs for a person (X-AEI-User, with the service token) also reads
that person's own web datasets, nobody else's, and stays bound to them;
reset_scope and delete_dataset touch only the scope's own; jobs run on the
bounded pool and belong to their scope. Single-user mode as before.

Calls go through mcp_server._run with a stand-in for the MCP Context, so the
scope and headers arrive exactly as over HTTP. Run from backend/ with the
repo's venv:  python -m unittest discover -s tests
"""

from __future__ import annotations

import sys
import threading
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import anyio  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402
from starlette.datastructures import Headers  # noqa: E402

from app import config, datasets  # noqa: E402
from app.assistant import conversations, jobs, mcp_server, mcp_tools, tools  # noqa: E402
from app.assistant.tools import data  # noqa: E402

import main  # noqa: E402
from multiuser_helpers import internal, multi_user, signed_in  # noqa: E402

A, B = "a@example.org", "b@example.org"
CSV = b"name,lat,lon\na,-4.0,15.0\nb,-4.1,15.1\n"
FEATURES = {
    "type": "FeatureCollection",
    "features": [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [15.0, -4.0]},
         "properties": {"name": "tower"}},
    ],
}
SPECS = {s.name: s for s in tools.specs(side="server") + mcp_tools.specs()}


def ctx(scope: str | None, headers: dict[str, str] | None = None) -> SimpleNamespace:
    """What the MCP Context gives _run: `_meta` and the HTTP headers."""
    return SimpleNamespace(
        request_context=SimpleNamespace(meta={"aeilabs/scope": scope} if scope else {}),
        headers=Headers(headers or {}),
    )


def call(tool: str, scope: str | None, headers: dict[str, str] | None = None, **args):
    return anyio.run(mcp_server._run, SPECS[tool], ctx(scope, headers), args)


def listed(scope: str, headers: dict[str, str] | None = None) -> set[str]:
    return {d["file_id"] for d in call("list_datasets", scope, headers)["datasets"]}


class Scopes(unittest.TestCase):
    def setUp(self):
        patch = multi_user()
        patch.start()
        self.addCleanup(patch.stop)
        self.client = TestClient(main.app)
        self.web = {}
        for who in (A, B):
            r = self.client.post("/api/upload", files={"file": ("t.csv", CSV, "text/csv")},
                                 headers=signed_in(who))
            self.assertEqual(r.status_code, 200, r.text)
            self.web[who] = r.json()["file_id"]
            self.addCleanup(datasets.STORE.pop, self.web[who], None)
        self.s = uuid.uuid4().hex[:8]  # this test's scope names
        self.addCleanup(self._drop_scopes)

    def _drop_scopes(self):
        for cid in [c for c in list(conversations._STORE) if self.s in c]:
            conversations.drop(cid)

    def scope(self, n: int) -> str:
        return f"{self.s}-{n}"

    def load(self, scope: str, headers: dict[str, str]) -> str:
        fid = call("load_features", scope, headers, geojson=FEATURES)["file_id"]
        self.addCleanup(datasets.STORE.pop, fid, None)
        return fid

    def test_what_a_scope_loads_is_the_scopes(self):
        one, two = self.scope(1), self.scope(2)
        fid = self.load(one, internal(user=A))
        self.assertEqual(datasets.STORE[fid]["owner"], f"mcp:{one}")
        self.assertNotIn(fid, listed(two, internal(user=A)))  # another scope, same user
        self.assertEqual(self.client.get(f"/api/upload/{fid}", headers=signed_in(A)).status_code, 404)

    def test_a_scope_reads_its_users_own_datasets_and_nobody_elses(self):
        one, two, three = self.scope(1), self.scope(2), self.scope(3)
        fid = self.load(one, internal(user=A))
        self.assertEqual(listed(one, internal(user=A)), {self.web[A], fid})
        self.assertEqual(listed(two, internal(user=B)), {self.web[B]})
        self.assertEqual(listed(three, internal()), set())  # no user: its own only
        # and uses them: python_exec's view, and the dataset lookups the analyses make
        out = call("python_exec", one, internal(user=A), code="sorted(uploaded_files)")
        self.assertEqual(out["result"], repr(sorted([self.web[A], fid])))
        conv = conversations.get(f"mcp:{one}")
        self.assertEqual(conv.readers, (f"mcp:{one}", A))
        self.assertEqual(data.resolve_dataset(conv, self.web[A]), self.web[A])
        with self.assertRaisesRegex(ValueError, "no dataset"):
            data.resolve_dataset(conv, self.web[B])

    def test_x_aei_user_without_the_token_grants_nothing(self):
        four = self.scope(4)
        self.assertEqual(listed(four, {"X-AEI-User": A}), set())
        self.assertIsNone(conversations.get(f"mcp:{four}").user)
        self.assertEqual(listed(four, internal(user=A, token="a-guess")), set())

    def test_a_scope_stays_with_its_user(self):
        one = self.scope(1)
        listed(one, internal(user=A))
        with self.assertRaisesRegex(ToolError, "another user"):
            listed(one, internal(user=B))
        self.assertEqual(listed(one, internal()), {self.web[A]})  # a call naming no one

    def test_reset_scope_deletes_the_scopes_datasets_not_the_users(self):
        one = self.scope(1)
        fid = self.load(one, internal(user=A))
        out = call("reset_scope", one, internal(user=A))
        self.assertEqual(out["datasets_deleted"], [fid])
        self.assertNotIn(fid, datasets.STORE)
        self.assertIn(self.web[A], datasets.STORE)
        self.assertIsNone(conversations.get(f"mcp:{one}"))

    def test_delete_dataset_takes_only_the_scopes_own(self):
        one = self.scope(1)
        fid = self.load(one, internal(user=A))
        with self.assertRaisesRegex(ToolError, "may delete"):
            call("delete_dataset", one, internal(user=A), file_id=self.web[A])
        self.assertIn(self.web[A], datasets.STORE)
        self.assertEqual(call("delete_dataset", one, internal(user=A), file_id=fid), {"deleted": fid})
        self.assertNotIn(fid, datasets.STORE)

    def test_read_app_state_shows_the_named_users_screen(self):
        self.client.post("/api/assistant/app_state", json={"file_id": "on-a"}, headers=signed_in(A))
        state = mcp_server.read_app_state(ctx(self.scope(1), internal(user=A)))
        self.assertEqual(state["state"], {"file_id": "on-a"})
        for headers in (internal(), {"X-AEI-User": A}):  # no one named, or not trusted
            with self.subTest(headers=sorted(headers)):
                self.assertFalse(mcp_server.read_app_state(ctx(self.scope(2), headers))["available"])

    def test_load_from_url_stays_within_its_bounds(self):
        one, two = self.scope(1), self.scope(2)
        conv = conversations.get_or_create(f"mcp:{one}", owner=f"mcp:{one}", user=A)
        with self.assertRaisesRegex(ValueError, "ASSISTANT_URL_ALLOW"):
            data.load_from_url(conv, "http://169.254.169.254/latest/meta-data")
        # an artifact of another scope is not this one's to read
        other = conversations.get_or_create(f"mcp:{two}", owner=f"mcp:{two}", user=B)
        dest = other.workdir / "net.geojson"
        dest.write_bytes(b"{}")
        art = mcp_server.artifacts.add(dest, "net.geojson", "file", "net", other.id)
        with self.assertRaisesRegex(ValueError, "not found"):
            data.load_from_url(conv, f"/api/assistant/artifacts/{art.id}")


class Jobs(unittest.TestCase):
    def test_jobs_run_on_the_pool_hold_their_scope_and_belong_to_it(self):
        mine = conversations.get_or_create(f"mcp:{uuid.uuid4().hex}")
        other = conversations.get_or_create(f"mcp:{uuid.uuid4().hex}")
        for conv in (mine, other):
            self.addCleanup(conversations.drop, conv.id)
        go, seen = threading.Event(), {}

        def work(job):
            seen.update(thread=threading.current_thread().name, busy=mine.busy)
            go.wait(5)
            return {"ok": True}

        job = jobs.start(mine, "test", work, 0.0)
        go.set()
        for _ in range(100):
            if jobs.get(mine, job.id).status == "done":
                break
            threading.Event().wait(0.05)
        self.assertEqual(jobs.get(mine, job.id).public()["result"], {"ok": True})
        self.assertTrue(seen["thread"].startswith("job"), seen)  # concurrency.JOBS
        self.assertTrue(seen["busy"])
        with self.assertRaisesRegex(ValueError, "no job"):
            jobs.get(other, job.id)

    def test_a_job_cancelled_while_queued_never_runs(self):
        conv = conversations.get_or_create(f"mcp:{uuid.uuid4().hex}")
        self.addCleanup(conversations.drop, conv.id)
        block, ran = threading.Event(), []
        blockers = [jobs.start(conv, "block", lambda job: block.wait(5), 0.0)
                    for _ in range(config.JOB_WORKERS)]
        queued = jobs.start(conv, "queued", lambda job: ran.append(1), 0.0)
        jobs.cancel_all(conv.id)
        block.set()
        for _ in range(100):
            if queued.status == "cancelled" and all(b.finished for b in blockers):
                break
            threading.Event().wait(0.05)
        self.assertEqual(queued.status, "cancelled")
        self.assertEqual(ran, [])


class SingleUserMode(unittest.TestCase):
    def test_scopes_share_datasets_and_reset_scope_deletes_what_it_loaded(self):
        scope = uuid.uuid4().hex[:8]
        self.addCleanup(conversations.drop, f"mcp:{scope}")
        fid = call("load_features", scope, internal(user=A), geojson=FEATURES)["file_id"]
        self.addCleanup(datasets.STORE.pop, fid, None)
        conv = conversations.get(f"mcp:{scope}")
        self.assertIsNone(conv.user)  # X-AEI-User means nothing here
        self.assertEqual(conv.readers, (config.AEI_LOCAL_USER,))
        self.assertEqual(datasets.STORE[fid]["owner"], config.AEI_LOCAL_USER)
        other = uuid.uuid4().hex[:8]
        self.addCleanup(conversations.drop, f"mcp:{other}")
        self.assertIn(fid, listed(other))  # shared by every scope, as before
        self.assertEqual(call("reset_scope", scope)["datasets_deleted"], [fid])
        self.assertNotIn(fid, datasets.STORE)


if __name__ == "__main__":
    unittest.main()
