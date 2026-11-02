"""A whole assistant conversation as one JSON document or one Markdown file (the panel's
Export menu), for debugging.

The server's history is the fullest record there is: every message with its tool calls and
their full results (the browser keeps only a one-line summary of each). The loop stamps what
the history never held (`meta` on each message: time, model, effort, tier, tokens, thinking,
stop reason; `Conversation.events`: the errors and notices the stream sent), and this module
reads it all back. The provider-native `raw` blocks and the temp-file paths of uploads stay
out. Everything goes through `redact()`, the same scrubber the tool results pass, so a key
that slipped into a result, an argument or an error is replaced here too.

A tool result is the one the model was given: the loop caps one at MAX_RESULT_CHARS, and a
longer one is the `truncated` stub, with the full data in the conversation's namespace.
"""

from __future__ import annotations

import functools
import json
import math
import re
import subprocess
import time
from datetime import datetime
from typing import Any

from .. import config
from ..redact import redact
from . import artifacts, loop, tools
from .conversations import Conversation
from .system_prompt import STATE_NOTE_START

FORMAT = "infrarisk-assistant-export/1"

# The Markdown shortens a longer tool result (the JSON never does).
LONG_RESULT_CHARS = 4000


def _plain(value: Any) -> Any:
    """`value` as strict JSON: numpy scalars unwrapped, non-finite floats and other objects
    as text (a NaN would make the file unreadable to JSON.parse)."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(v) for v in value]
    if hasattr(value, "item") and not hasattr(value, "__len__"):  # a numpy scalar
        return _plain(value.item())
    return str(value)


@functools.cache
def _app() -> dict[str, str | None]:
    """The git commit and branch the server runs from. None in the Docker image (.git is
    dockerignored) or when git is not there."""
    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", "-C", str(config.settings.BASE_DIR), *args],
                capture_output=True, text=True, timeout=3, check=True,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
        return out or None

    return {"commit": git("rev-parse", "--short", "HEAD"), "branch": git("branch", "--show-current")}


def _part(p: dict[str, Any]) -> dict[str, Any]:
    if p["type"] == "file":  # the upload's name and type; its path is a server temp file
        return {k: v for k, v in p.items() if k != "path"}
    if p["type"] == "tool_call":
        spec = tools.REGISTRY.get(p["name"])
        return {**p, "side": spec.side if spec else "unknown"}
    return dict(p)


def _message(m: dict[str, Any]) -> dict[str, Any]:
    meta = dict(m.get("meta") or {})
    parts = [_part(p) for p in m["parts"]]
    thinking = meta.pop("thinking", "")
    if thinking.strip():
        parts.insert(0, {"type": "thinking", "text": thinking})
    return {"role": m["role"], **meta, "parts": parts}


def _as_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [v for v in value if isinstance(v, str)] if isinstance(value, list) else []


def _identifiers(conv: Conversation, messages: list[dict[str, Any]]) -> dict[str, Any]:
    """What the chat produced or used, to find it in the app or the logs: the datasets
    (file_id), hazards named in calls, analyses kept in the namespace (`stored_as`), curves,
    uploads and generated files."""
    from ..api.upload import uploaded_files

    hazards: list[str] = []
    stored: list[str] = []
    for m in messages:
        for p in m["parts"]:
            if p["type"] == "tool_call":
                for key in ("hazard", "hazards", "family"):
                    hazards += _as_list((p.get("args") or {}).get(key))
            elif p["type"] == "tool_result" and isinstance(p.get("content"), dict):
                stored += _as_list(p["content"].get("stored_as"))
    datasets = []
    for file_id in conv.datasets:
        info = uploaded_files.get(file_id) or {}
        datasets.append({"file_id": file_id, **{
            k: info[k] for k in ("filename", "geometry_type", "feature_count") if k in info}})
    return {
        "conversation_id": conv.id,
        "datasets": datasets,
        "hazards_named": list(dict.fromkeys(hazards)),
        "analyses": list(dict.fromkeys(stored)),
        "curves": sorted(conv.curves),
        "uploads": sorted(conv.uploads),
        "artifacts": [
            {**a.public(), "created": a.created} for a in artifacts.for_conversation(conv.id)
        ],
    }


def document(conv: Conversation) -> dict[str, Any]:
    """The JSON export. A turn may still be running (the lists are copied, and each message
    is complete once it is appended); `in_flight` holds the calls it is waiting on and the
    results it has not yet merged into the history."""
    messages = [_message(m) for m in list(conv.messages)]
    doc = {
        "format": FORMAT,
        "exported": time.time(),
        "app": _app(),
        "conversation": {
            "id": conv.id,
            "created": conv.created,
            "provider": conv.provider,
            "model": conv.model,
        },
        "identifiers": _identifiers(conv, messages),
        "messages": messages,
        # an error or notice sits after the first `at` messages
        "events": list(conv.events),
        "in_flight": {
            "pending_calls": list(conv.pending_calls),
            "buffered_results": list(conv.buffered_results),
        },
    }
    return redact(_plain(doc))


# ---- Markdown ----------------------------------------------------------------------- #


def _when(ts: float | None) -> str:
    return datetime.fromtimestamp(ts).astimezone().strftime("%Y-%m-%d %H:%M:%S") if ts else ""


def _fence(body: str, lang: str = "") -> str:
    """A code fence longer than any backtick run inside `body`."""
    ticks = "```"
    while ticks in body:
        ticks += "`"
    return f"{ticks}{lang}\n{body}\n{ticks}"


def _json(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, indent=2, ensure_ascii=False)


def _details(summary: str, body: str) -> str:
    return f"<details><summary>{summary}</summary>\n\n{body}\n\n</details>"


def _quote(text: str) -> str:
    return "\n".join(f"> {line}" if line else ">" for line in text.split("\n"))


def _demote(md: str, by: int) -> str:
    """Push the assistant's own headings below the message heading they sit under, so its
    `## Summary` does not read as a section of the export. Code blocks are left alone."""
    out, fence = [], ""
    for line in md.split("\n"):
        mark = re.match(r"\s*(`{3,}|~{3,})", line)
        if mark and (not fence or mark.group(1).startswith(fence)):
            fence = "" if fence else mark.group(1)[:3]
        elif not fence and (h := re.match(r"(#{1,6})\s", line)):
            line = "#" * min(6, len(h.group(1)) + by) + line[len(h.group(1)):]
        out.append(line)
    return "\n".join(out)


def _shorten(text: str) -> str:
    if len(text) <= LONG_RESULT_CHARS:
        return text
    return (f"{text[:LONG_RESULT_CHARS]}\n… [shortened: first {LONG_RESULT_CHARS:,} of "
            f"{len(text):,} characters; the JSON export has the whole result]")


def _result_md(res: dict[str, Any] | None) -> list[str]:
    if res is None:
        return ["**Result:** none recorded (still running, waiting for the browser, or "
                "the turn was cancelled)"]
    took = f", {res['seconds']:g} s" if res.get("seconds") is not None else ""
    head = f"**Result** ({'ok' if res.get('ok') else 'error'}{took})"
    content = res.get("content")
    if not res.get("ok"):
        return [f"{head}\n\n{_fence(_shorten(_json(content)))}"]
    out = [f"{head}: {loop._summary(res['name'], True, content)}"]
    if content not in (None, "", [], {}):
        out.append(_details("Full result", _fence(_shorten(_json(content)))))
    return out


def _model_line(m: dict[str, Any]) -> str:
    bits = [f"{m['provider']} / {m['model']}"] if m.get("model") else []
    if m.get("effort"):
        bits.append(f"effort {m['effort']}")
    if m.get("service_tier") == "flex":
        bits.append("flex")
    return " · ".join(bits)


def _user_md(m: dict[str, Any]) -> list[str]:
    """The user's words, quoted; then what the app added for the model: the note of each
    attached file, and the app-state snapshot the loop appends to the text."""
    words: list[str] = []
    context: list[str] = []
    for p in m["parts"]:
        if p["type"] != "text":
            continue
        text, _, note = p["text"].partition(STATE_NOTE_START)
        if text.startswith("[Attached file: "):
            head, _, preview = text.partition("\n")
            context.append(_details(head, _fence(preview)) if preview else head)
        elif text.strip():
            words.append(_quote(text.strip()))
        if note:
            state = note.removesuffix("]")
            try:
                state = json.dumps(json.loads(state), indent=2, ensure_ascii=False)
            except ValueError:
                pass
            context.append(_details("App state sent with this message", _fence(state, "json")))
    files = [p["name"] for p in m["parts"] if p["type"] == "file"]
    return [f"## User · {_when(m.get('created'))}", *words,
            *(["Attached: " + ", ".join(f"`{n}`" for n in files)] if files else []), *context]


def _assistant_md(m: dict[str, Any], results: dict[str, dict[str, Any]]) -> list[str]:
    label = _model_line(m) if m.get("model") else "added by the app"
    out = [f"## Assistant · {_when(m.get('created'))}" + (f" · {label}" if label else "")]
    for p in m["parts"]:
        if p["type"] == "text" and p["text"].strip():
            out.append(_demote(p["text"].strip(), 2))
        elif p["type"] == "thinking":
            out.append(_details("Thinking", p["text"].strip()))
        elif p["type"] == "tool_call":
            where = {"server": "server", "client": "browser"}.get(p.get("side"), "?")
            args = _fence(_json(p.get("args") or {}), "json")
            out.append(f"**Tool call** `{p['name']}` ({where})\n\n{args}")
            out.extend(_result_md(results.get(p["id"])))
    foot = []
    if m.get("usage"):
        foot.append(f"{m['usage']['input_tokens']:,} in / {m['usage']['output_tokens']:,} out")
    if m.get("stop_reason") and m["stop_reason"] != "end_turn":
        foot.append(f"stopped: {m['stop_reason']}")
    if foot:
        out.append(f"_{' · '.join(foot)}_")
    return out


def _event_md(e: dict[str, Any]) -> str:
    return _quote(f"**{e['type'].capitalize()}** · {_when(e.get('time'))}: {e.get('message', '')}")


def _identifiers_md(ids: dict[str, Any]) -> str:
    lines = []
    for d in ids["datasets"]:
        what = ", ".join(
            str(d[k]) for k in ("filename", "geometry_type", "feature_count") if k in d)
        lines.append(f"- Dataset `{d['file_id']}`" + (f" ({what})" if what else ""))
    for label, key in (("Hazards named", "hazards_named"), ("Analyses", "analyses"),
                       ("Curves", "curves"), ("Uploads", "uploads")):
        if ids[key]:
            lines.append(f"- {label}: " + ", ".join(f"`{v}`" for v in ids[key]))
    for a in ids["artifacts"]:
        lines.append(f"- Generated file `{a['filename']}` ({a['kind']}, id `{a['id']}`)")
    return "\n".join(lines)


def markdown(doc: dict[str, Any]) -> str:
    """The Markdown export of a `document()`: the same content, readable top to bottom, each
    tool result under its call, long results shortened."""
    msgs = doc["messages"]
    results = {p["id"]: p for m in msgs for p in m["parts"] if p["type"] == "tool_result"}
    results.update({r["id"]: r for r in doc["in_flight"]["buffered_results"]})
    called = {p["id"] for m in msgs for p in m["parts"] if p["type"] == "tool_call"}
    events = sorted(doc["events"], key=lambda e: e["at"])

    n_calls = len(called)
    n_failed = sum(1 for i in called if i in results and not results[i].get("ok"))
    tokens = [m["usage"] for m in msgs if m.get("usage")]
    models = list(dict.fromkeys(_model_line(m) for m in msgs if m.get("model")))
    conv = doc["conversation"]
    facts = [
        f"- Conversation: `{conv['id']}`",
        f"- Started: {_when(conv['created'])} · exported {_when(doc['exported'])}",
        "- Model: " + ("; ".join(models) or f"{conv['provider'] or ''} / {conv['model'] or ''}"),
    ]
    if doc["app"]["commit"]:
        facts.append(f"- App: commit `{doc['app']['commit']}` (branch `{doc['app']['branch']}`)")
    facts.append(
        f"- {len(msgs)} messages, {n_calls} tool calls ({n_failed} failed), "
        f"{sum(t['input_tokens'] for t in tokens):,} tokens in / "
        f"{sum(t['output_tokens'] for t in tokens):,} out")

    body: list[str] = []
    for i, m in enumerate(msgs):
        while events and events[0]["at"] <= i:
            body.append(_event_md(events.pop(0)))
        if m["role"] == "assistant":
            body.extend(_assistant_md(m, results))
        elif not all(p["type"] == "tool_result" for p in m["parts"]):
            body.extend(_user_md(m))
        else:  # results show under their calls; one without a call would otherwise vanish
            for p in m["parts"]:
                if p["id"] not in called:
                    body.append(f"**Result of `{p['name']}`** (its call is not in the history)")
                    body.extend(_result_md(p))
    body.extend(_event_md(e) for e in events)
    waiting = [c["name"] for c in doc["in_flight"]["pending_calls"]]
    if waiting:
        body.append("_A turn was still running at export; waiting on: "
                    + ", ".join(f"`{n}`" for n in waiting) + "_")

    head = ["# Assistant conversation (Infrastructure Risk Analyzer)", "\n".join(facts)]
    ids = _identifiers_md(doc["identifiers"])
    if ids:
        head += ["## Identifiers", ids]
    return "\n\n".join(head + ["---"] + body) + "\n"
