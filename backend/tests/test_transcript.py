"""The panel's Export menu: GET /api/assistant/conversations/{id}/export and
assistant/transcript.py. The real loop runs a scripted provider (no network), so what the
export shows is what the loop recorded.

Run from backend/ with the repo's venv:  python -m unittest discover -s tests
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import redact  # noqa: E402
from app.api import assistant_api  # noqa: E402
from app.api.upload import uploaded_files  # noqa: E402
from app.assistant import conversations, loop, schema, transcript  # noqa: E402
from app.assistant.providers import anthropic as anthropic_provider  # noqa: E402
from app.assistant.providers import gemini as gemini_provider  # noqa: E402
from app.assistant.providers import openai as openai_provider  # noqa: E402

CONV_ID = "export-test-conversation"
FILE_ID = "export-test-file-id"


class Scripted:
    """A provider that plays one scripted turn (a list of events, or an exception to raise)
    per request."""

    def __init__(self, *turns):
        self.turns = list(turns)

    def stream_turn(self, model, system, messages, tools, effort=None, service_tier=None):
        turn = self.turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        yield from turn


def run_leg(conv, provider, **kw):
    with mock.patch.object(loop, "get_provider", lambda name: provider):
        return "".join(loop.run(conv, "fake", "fake-1", effort="high", service_tier="flex", **kw))


def client() -> TestClient:
    app = FastAPI()
    app.include_router(assistant_api.router, prefix="/api")
    return TestClient(app)


class Export(unittest.TestCase):
    """A conversation of three legs: a question answered with two server calls (one fails)
    and one browser call; the browser's answer and the final text; then a provider error."""

    @classmethod
    def setUpClass(cls):
        cls.conv = conversations.get_or_create(CONV_ID)
        conv = cls.conv
        cls.upload_dir = tempfile.TemporaryDirectory(prefix="export-test-")
        cls.png = Path(cls.upload_dir.name, "view.png")
        cls.png.write_bytes(b"\x89PNG\r\n\x1a\n")
        conv.datasets.append(FILE_ID)
        uploaded_files[FILE_ID] = {
            "filename": "line.gpkg", "geometry_type": "LineString", "feature_count": 12}
        provider = Scripted(
            [
                schema.ThinkingDelta("Check the data first."),
                schema.TextDelta("Looking.\n"),
                schema.ToolCall("c1", "read_guide", {"topic": "figures"}),
                schema.ToolCall("c2", "read_guide", {"topic": "nope"}),
                schema.ToolCall("c3", "ui_read_app_state", {}),
                schema.Usage(1200, 34),
                schema.TurnEnd("tool_use"),
            ],
            [
                schema.TextDelta("## Summary\nNothing exposed.\n\n```md\n# kept in code\n```"),
                schema.Usage(2000, 80),
                schema.TurnEnd("end_turn"),
            ],
            RuntimeError("provider said 400: bad request"),
        )
        cls.out = [
            run_leg(
                conv, provider,
                user_message="# not a section\nAssess the railway.",
                file_parts=[
                    {"type": "file", "name": "view.png", "path": str(cls.png),
                     "mime": "image/png"},
                    {"type": "text", "text": "[Attached file: curve.csv (9 bytes). Load it.]"
                                             "\n--- preview ---\nintensity,damage\n1,0.1"},
                ],
                app_state={"hazard": {"hazard_id": "flood_rp100"}},
            ),
            run_leg(conv, provider, tool_results=[
                {"id": "c3", "ok": True, "result": {"dataset": {"file_id": FILE_ID}, "nan": float("nan")}}]),
            run_leg(conv, provider, user_message="and again"),
        ]
        cls.http = client()

    @classmethod
    def tearDownClass(cls):
        conversations.drop(CONV_ID)
        uploaded_files.pop(FILE_ID, None)
        cls.upload_dir.cleanup()

    def export(self, fmt="json"):
        r = self.http.get(f"/api/assistant/conversations/{CONV_ID}/export?format={fmt}")
        self.assertEqual(r.status_code, 200, r.text)
        return r

    def test_the_loop_stamps_what_the_history_never_held(self):
        user, first, results, second, again = self.conv.messages[:5]
        self.assertTrue(user["meta"]["created"] <= first["meta"]["created"] <= results["meta"]["created"])
        self.assertEqual(
            {k: first["meta"][k] for k in ("provider", "model", "effort", "service_tier",
                                           "stop_reason", "usage", "thinking")},
            {"provider": "fake", "model": "fake-1", "effort": "high", "service_tier": "flex",
             "stop_reason": "tool_use", "usage": {"input_tokens": 1200, "output_tokens": 34},
             "thinking": "Check the data first."},
        )
        self.assertEqual(second["meta"]["usage"], {"input_tokens": 2000, "output_tokens": 80})
        self.assertEqual(second["meta"]["thinking"], "")
        self.assertEqual(len(self.conv.messages), 5)  # the third leg left its question
        self.assertEqual([(e["type"], e["at"]) for e in self.conv.events], [("error", 5)])
        # the three adapters read the history as before: `meta` and `seconds` are ignored
        anthropic_provider._messages(self.conv.messages)
        gemini_provider._contents(self.conv.messages)
        openai_provider._input(self.conv.messages)

    def test_json_is_the_full_record(self):
        r = self.export("json")
        self.assertRegex(r.headers["content-disposition"],
                         r'^attachment; filename="infrarisk-assistant-\d{8}-\d{4}-export-t\.json"$')

        def strict(const):  # NaN / Infinity would make the file unreadable to JSON.parse
            raise AssertionError(f"non-JSON constant {const}")

        doc = json.loads(r.text, parse_constant=strict)
        self.assertEqual(doc["format"], transcript.FORMAT)
        self.assertEqual(set(doc["app"]), {"commit", "branch"})
        self.assertEqual(doc["conversation"]["id"], CONV_ID)
        msgs = doc["messages"]
        self.assertEqual([m["role"] for m in msgs],
                         ["user", "assistant", "user", "assistant", "user"])
        # no provider-native blocks, no server temp path
        self.assertTrue(all("raw" not in m and "raw_provider" not in m for m in msgs))
        self.assertNotIn(self.upload_dir.name, r.text)
        self.assertEqual(msgs[0]["parts"][0], {"type": "file", "name": "view.png", "mime": "image/png"})

        first = msgs[1]
        self.assertEqual((first["model"], first["effort"], first["service_tier"], first["usage"]),
                         ("fake-1", "high", "flex", {"input_tokens": 1200, "output_tokens": 34}))
        self.assertEqual(first["parts"][0], {"type": "thinking", "text": "Check the data first."})
        calls = {p["id"]: p for p in first["parts"] if p["type"] == "tool_call"}
        self.assertEqual({i: c["side"] for i, c in calls.items()},
                         {"c1": "server", "c2": "server", "c3": "client"})
        res = {p["id"]: p for m in msgs for p in m["parts"] if p["type"] == "tool_result"}
        self.assertTrue(res["c1"]["ok"])
        self.assertEqual(res["c1"]["content"]["topic"], "figures")  # the whole guide, not a stub
        self.assertGreater(len(res["c1"]["content"]["guide"]), transcript.LONG_RESULT_CHARS)
        self.assertIn("seconds", res["c1"])
        self.assertFalse(res["c2"]["ok"])
        self.assertIn("unknown topic 'nope'", res["c2"]["content"])
        # the browser's answer, in full; NaN as text
        self.assertEqual(res["c3"]["content"]["dataset"], {"file_id": FILE_ID})
        self.assertEqual(res["c3"]["content"]["nan"], "nan")

        [event] = doc["events"]
        self.assertEqual((event["type"], event["at"]), ("error", 5))
        self.assertIn("bad request", event["message"])
        self.assertEqual(doc["in_flight"], {"pending_calls": [], "buffered_results": []})
        ids = doc["identifiers"]
        self.assertEqual(ids["conversation_id"], CONV_ID)
        self.assertEqual(ids["datasets"], [{"file_id": FILE_ID, "filename": "line.gpkg",
                                            "geometry_type": "LineString", "feature_count": 12}])

    def test_markdown_reads_top_to_bottom(self):
        r = self.export("md")
        self.assertTrue(r.headers["content-type"].startswith("text/markdown"))
        md = r.text
        self.assertTrue(md.startswith("# Assistant conversation (Infrastructure Risk Analyzer)\n"))
        self.assertIn(f"- Conversation: `{CONV_ID}`", md)
        self.assertIn("- Model: fake / fake-1 · effort high · flex", md)
        self.assertIn("- 5 messages, 3 tool calls (1 failed), 3,200 tokens in / 114 out", md)
        self.assertIn(f"- Dataset `{FILE_ID}` (line.gpkg, LineString, 12)", md)
        # the user's words are quoted (so their heading is no section of the export); the app's
        # additions are set apart
        self.assertIn("> # not a section\n> Assess the railway.", md)
        self.assertIn("Attached: `view.png`", md)
        self.assertIn("<summary>[Attached file: curve.csv (9 bytes). Load it.]</summary>", md)
        self.assertIn("intensity,damage", md)
        self.assertIn("<summary>App state sent with this message</summary>", md)
        self.assertIn('"hazard_id": "flood_rp100"', md)
        self.assertNotIn("Current app state", md)
        # thinking, call, result: in order, the result under its call
        self.assertIn("<details><summary>Thinking</summary>\n\nCheck the data first.", md)
        call = md.index("**Tool call** `read_guide` (server)")
        self.assertLess(call, md.index("**Result** (error"))
        self.assertIn("unknown topic 'nope'", md)
        self.assertIn("**Tool call** `read_guide` (server)\n\n```json\n{\n  \"topic\": \"figures\"", md)
        self.assertIn("**Result** (ok, ", md)
        self.assertIn("<summary>Full result</summary>", md)
        self.assertIn("[shortened: first 4,000 of ", md)  # the guide; the JSON test has it whole
        self.assertIn("**Tool call** `ui_read_app_state` (browser)", md)
        self.assertIn("## Assistant · ", md)
        self.assertIn("· fake / fake-1 · effort high · flex", md)
        self.assertIn("_1,200 in / 34 out · stopped: tool_use_", md)
        # the assistant's own headings sit below the message heading; code is untouched
        self.assertIn("#### Summary\nNothing exposed.", md)
        self.assertIn("```md\n# kept in code\n```", md)
        # the provider error comes after the last question
        self.assertGreater(md.index("> **Error** · "), md.index("> and again"))
        self.assertIn("bad request", md)

    def test_endpoint_refusals(self):
        self.assertEqual(self.http.get("/api/assistant/conversations/nope/export").status_code, 404)
        self.assertEqual(
            self.http.get(f"/api/assistant/conversations/{CONV_ID}/export?format=pdf").status_code,
            422)


class Rendering(unittest.TestCase):
    """Edge cases on a hand-built history."""

    def setUp(self):
        self.conv = conversations.get_or_create("export-test-edge")
        self.addCleanup(conversations.drop, "export-test-edge")
        self.http = client()

    def get(self, fmt):
        return self.http.get(f"/api/assistant/conversations/export-test-edge/export?format={fmt}").text

    def test_a_long_result_is_shortened_in_markdown_but_not_in_json(self):
        big = "x" * 10_000 + "END"
        self.conv.messages += [
            {"role": "assistant", "parts": [
                {"type": "tool_call", "id": "c1", "name": "python_exec", "args": {"code": "print('```')"}}]},
            {"role": "user", "parts": [
                {"type": "tool_result", "id": "c1", "name": "python_exec", "ok": True, "content": big}]},
        ]
        md = self.get("md")
        self.assertIn("[shortened: first 4,000 of 10,003 characters; the JSON export has the whole result]", md)
        self.assertNotIn("END", md)
        # the fence is longer than the backticks in the arguments
        self.assertIn('````json\n{\n  "code": "print(\'```\')"\n}\n````', md)
        self.assertIn("END", json.loads(self.get("json"))["messages"][1]["parts"][0]["content"])

    def test_an_unfinished_turn_exports_what_it_has(self):
        """c1 is done but not yet merged into the history, c2 waits for the browser."""
        self.conv.messages += [
            {"role": "user", "parts": [{"type": "text", "text": "go"}]},
            {"role": "assistant", "parts": [
                {"type": "tool_call", "id": "c1", "name": "list_datasets", "args": {}},
                {"type": "tool_call", "id": "c2", "name": "ui_read_app_state", "args": {}}]},
        ]
        self.conv.pending_calls = [{"id": "c2", "name": "ui_read_app_state", "args": {}}]
        self.conv.buffered_results = [
            {"id": "c1", "name": "list_datasets", "ok": True, "content": {"count": 0}}]
        md = self.get("md")
        self.assertIn("**Result** (ok): count", md)
        self.assertIn("**Result:** none recorded", md)
        self.assertIn("waiting on: `ui_read_app_state`", md)
        doc = json.loads(self.get("json"))
        self.assertEqual(doc["in_flight"]["pending_calls"][0]["id"], "c2")
        self.assertEqual(doc["in_flight"]["buffered_results"][0]["id"], "c1")

    def test_no_secret_leaves_in_either_format(self):
        secret = "fake-export-key-0123456789"
        with mock.patch.object(redact, "_secrets", {secret}):
            self.conv.messages += [
                {"role": "assistant", "parts": [
                    {"type": "tool_call", "id": "c1", "name": "python_exec",
                     "args": {"code": f"key = '{secret}'"}}],
                 "meta": {"model": "m", "provider": "p", "thinking": f"it is {secret}"}},
                {"role": "user", "parts": [
                    {"type": "tool_result", "id": "c1", "name": "python_exec", "ok": False,
                     "content": f"401 for https://tiles.example/{{z}}.png?key=abc&x=1 with {secret}"}]},
            ]
            self.conv.events.append({"at": 2, "time": time.time(), "type": "error",
                                     "message": f"provider said {secret}"})
            for fmt in ("json", "md"):
                text = self.get(fmt)
                self.assertNotIn(secret, text)
                self.assertNotIn("key=abc", text)
                self.assertIn(redact.REDACTED, text)

    def test_a_hostile_conversation_id_cannot_break_the_filename_header(self):
        conv = conversations.get_or_create('a"b;c d')
        self.addCleanup(conversations.drop, 'a"b;c d')
        r = self.http.get('/api/assistant/conversations/a"b;c d/export')
        self.assertEqual(r.status_code, 200)
        self.assertRegex(r.headers["content-disposition"],
                         r'^attachment; filename="infrarisk-assistant-\d{8}-\d{4}-abcd\.md"$')
        self.assertEqual(conv.id, 'a"b;c d')


if __name__ == "__main__":
    unittest.main()
