"""Per-conversation state: history, pending client calls, namespace, uploads.

One process, one store shared by browser chats and the MCP scope. Each
conversation belongs to the principal that created it (app/identity.py) and is
invisible to everyone else. A conversation idle for longer than IDLE_TTL_S is
dropped, and beyond MAX_CONVERSATIONS the least recently used go first — but
never one that is busy (a turn, a python_exec or a tool call in flight).
Eviction deletes the temp directory, artifacts and namespace. The backend
already runs single-worker for exactly this reason.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .. import config, identity
from . import artifacts

log = logging.getLogger("infrarisk.assistant")

# Browser chats and MCP scopes share this store; the MCP contract asks for >= 32
# so an orchestrator's scopes outlive a session's browser traffic.
MAX_CONVERSATIONS = config.ASSISTANT_MAX_CONVERSATIONS
IDLE_TTL_S = config.ASSISTANT_CONVERSATION_IDLE_HOURS * 3600.0


class NotYours(LookupError):
    """The conversation id belongs to another principal; callers answer 404."""


@dataclass
class Conversation:
    id: str
    created: float
    # The principal that created it: an email, `mcp:<scope>`, or AEI_LOCAL_USER
    # in single-user mode. Fixed for the conversation's life.
    owner: str = field(default_factory=identity.local_user)
    provider: str | None = None
    model: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    # Client-side tool calls the browser still owes us results for, plus the
    # server-side results of the same turn, buffered until they can be merged
    # into one tool-result message.
    pending_calls: list[dict[str, Any]] = field(default_factory=list)
    buffered_results: list[dict[str, Any]] = field(default_factory=list)
    # The python_exec namespace. Created empty and cheap; kernel.ensure()
    # preloads the heavy handles on first use. Server tools may stash large
    # results here at any time (store_result below).
    namespace: dict[str, Any] = field(default_factory=dict)
    kernel_ready: bool = False
    uploads: dict[str, Path] = field(default_factory=dict)
    # Vulnerability curves parsed in this conversation:
    # name -> {interp, lower, upper, intensity, proportion, has_bounds, path}
    curves: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Datasets this conversation registered into the app's uploaded_files.
    datasets: list[str] = field(default_factory=list)
    # Uploads already loaded as datasets: (name, mtime_ns, size) -> file_id, so
    # passing an upload's name as a file_id loads it once, not on every call.
    upload_datasets: dict[tuple, str] = field(default_factory=dict)
    # Serialises turns: one running /chat leg per conversation.
    turn_lock: threading.Lock = field(default_factory=threading.Lock)
    # Serialises python_exec against itself.
    exec_lock: threading.Lock = field(default_factory=threading.Lock)
    last_used: float = field(default_factory=time.time)
    _holds: int = 0
    _workdir: Path | None = None
    _counter: int = 0

    @contextmanager
    def hold(self) -> Iterator["Conversation"]:
        """Mark the conversation in use (a tool call, an upload) so it is not evicted."""
        with _LOCK:
            self._holds += 1
        try:
            yield self
        finally:
            with _LOCK:
                self._holds -= 1
                self.last_used = time.time()

    @property
    def busy(self) -> bool:
        stuck = getattr(self, "_exec_thread", None)
        return (
            self._holds > 0
            or self.turn_lock.locked()
            or self.exec_lock.locked()
            or (stuck is not None and stuck.is_alive())
        )

    @property
    def workdir(self) -> Path:
        if self._workdir is None:
            self._workdir = Path(tempfile.mkdtemp(prefix="infrarisk-assistant-"))
        return self._workdir

    def store_result(self, prefix: str, value: Any) -> str:
        """Put a large tool result into the namespace under a fresh name."""
        self._counter += 1
        name = f"{prefix}_{self._counter}"
        self.namespace[name] = value
        return name


_STORE: OrderedDict[str, Conversation] = OrderedDict()
_LOCK = threading.Lock()

_KINDS = {
    ".gpkg": "gpkg",
    ".csv": "csv",
    ".geojson": "geojson",
    ".xlsx": "xlsx",
    ".xls": "xlsx",
    ".txt": "text",
    ".md": "text",
}
_DIGESTS: dict[tuple, str] = {}


def file_kind(name: str) -> str:
    """The MCP contract's coarse file kind, from the extension."""
    return _KINDS.get(Path(name).suffix.lower(), "other")


def file_digest(path: Path) -> str:
    """sha256 of a file, cached until it changes."""
    st = path.stat()
    key = (str(path), st.st_mtime_ns, st.st_size)
    digest = _DIGESTS.get(key)
    if digest is None:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        digest = _DIGESTS[key] = h.hexdigest()
    return digest


def get_or_create(conversation_id: str | None, owner: str | None = None) -> Conversation:
    """The owner's conversation `conversation_id`, created on first use.

    Raises NotYours when the id belongs to another principal. `owner` defaults to
    AEI_LOCAL_USER, which in multi-user mode owns nothing a person created.
    """
    owner = owner or identity.local_user()
    now = time.time()
    with _LOCK:
        conv = _STORE.get(conversation_id) if conversation_id else None
        if conv is not None:
            if conv.owner != owner:
                raise NotYours(conversation_id)
            _STORE.move_to_end(conv.id)
            conv.last_used = now
            return conv
        conv = Conversation(
            id=conversation_id or uuid.uuid4().hex, created=now, owner=owner, last_used=now
        )
        _STORE[conv.id] = conv
        doomed = _evict(now, keep=conv.id)
    for old in doomed:
        _cleanup(old)
        log.info("conversation evicted: %s", old.id)
    return conv


def _evict(now: float, keep: str) -> list[Conversation]:
    """Under _LOCK: take out the conversations idle beyond IDLE_TTL_S, then the
    least recently used beyond MAX_CONVERSATIONS. A busy one is never taken; if
    every candidate is busy, the store runs over its cap for a while."""
    doomed = [
        c for cid, c in _STORE.items()
        if cid != keep and not c.busy and now - c.last_used > IDLE_TTL_S
    ]
    for c in doomed:
        del _STORE[c.id]
    for cid, c in list(_STORE.items()):  # least recently used first
        if len(_STORE) <= MAX_CONVERSATIONS:
            break
        if cid != keep and not c.busy:
            del _STORE[cid]
            doomed.append(c)
    return doomed


def get(conversation_id: str, owner: str | None = None) -> Conversation | None:
    """The conversation, or None when it is gone or (with `owner`) not the owner's.
    `owner=None` skips the check, for the server's own bookkeeping."""
    with _LOCK:
        conv = _STORE.get(conversation_id)
        if conv is None or (owner is not None and conv.owner != owner):
            return None
        _STORE.move_to_end(conversation_id)
        conv.last_used = time.time()
    return conv


def drop(conversation_id: str, owner: str | None = None) -> bool:
    """Forget a conversation; with `owner`, only the owner's."""
    with _LOCK:
        conv = _STORE.get(conversation_id)
        if conv is None or (owner is not None and conv.owner != owner):
            return False
        del _STORE[conversation_id]
    _cleanup(conv)
    return True


def _cleanup(conv: Conversation) -> None:
    """Drop the conversation's artifacts, namespace and temp files, and the
    datasets an MCP scope loaded for itself.

    Datasets a browser chat loaded belong to its user and are left alone on
    purpose: the user may still be looking at one on the map. Their own TTL
    and quota retire them (app/datasets.py).
    """
    from .. import datasets

    datasets.purge_scope(conv.id)
    artifacts.drop_for(conv.id)
    conv.namespace.clear()
    conv.curves.clear()
    if conv._workdir is not None:
        shutil.rmtree(conv._workdir, ignore_errors=True)
