"""Per-conversation state: history, pending client calls, namespace, uploads.

One process, one store, LRU-evicted once MAX_CONVERSATIONS are live; eviction
deletes the temp directory, artifacts and namespace. The backend already runs
single-worker for exactly this reason.
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import artifacts

log = logging.getLogger("infrarisk.assistant")

# Browser chats and MCP scopes share this LRU; the MCP contract asks for >= 32
# so an orchestrator's scopes outlive a session's browser traffic.
MAX_CONVERSATIONS = 32


@dataclass
class Conversation:
    id: str
    created: float
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
    _workdir: Path | None = None
    _counter: int = 0

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


def get_or_create(conversation_id: str | None) -> Conversation:
    with _LOCK:
        if conversation_id and conversation_id in _STORE:
            _STORE.move_to_end(conversation_id)
            return _STORE[conversation_id]
        conv = Conversation(id=conversation_id or uuid.uuid4().hex, created=time.time())
        _STORE[conv.id] = conv
        while len(_STORE) > MAX_CONVERSATIONS:
            old_id, old = _STORE.popitem(last=False)
            _cleanup(old)
            log.info("conversation evicted: %s", old_id)
        return conv


def get(conversation_id: str) -> Conversation | None:
    with _LOCK:
        conv = _STORE.get(conversation_id)
        if conv is not None:
            _STORE.move_to_end(conversation_id)
    return conv


def drop(conversation_id: str) -> bool:
    with _LOCK:
        conv = _STORE.pop(conversation_id, None)
    if conv is None:
        return False
    _cleanup(conv)
    return True


def _cleanup(conv: Conversation) -> None:
    """Drop the conversation's artifacts, namespace and temp files.

    Datasets the conversation pushed into the app's `uploaded_files` are left
    alone on purpose: the user may still be looking at one on the map.
    """
    artifacts.drop_for(conv.id)
    conv.namespace.clear()
    conv.curves.clear()
    if conv._workdir is not None:
        shutil.rmtree(conv._workdir, ignore_errors=True)
