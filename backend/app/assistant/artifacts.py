"""Generated files (charts, maps, reports, exports), addressable by id.

A uuid-keyed OrderedDict with LRU eviction, sized for one worker — the pattern
the rest of the backend already uses for its caches. Files live in each
conversation's temp directory; eviction unlinks them. Served only by
GET /api/assistant/artifacts/{id}.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("infrarisk.assistant")

MAX_ARTIFACTS = 64

# kind -> (mime, served inline in the chat transcript?)
KINDS = {
    "png": ("image/png", True),
    "chart": ("image/png", True),
    "map": ("image/png", True),
    "docx": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        False,
    ),
    "xlsx": (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        False,
    ),
    "csv": ("text/csv", False),
    "gpkg": ("application/geopackage+sqlite3", False),
    "file": ("application/octet-stream", False),
}


@dataclass
class Artifact:
    id: str
    path: Path
    filename: str
    kind: str
    title: str
    conversation_id: str
    created: float

    @property
    def mime(self) -> str:
        return KINDS.get(self.kind, KINDS["file"])[0]

    @property
    def inline(self) -> bool:
        return KINDS.get(self.kind, KINDS["file"])[1]

    def public(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "filename": self.filename,
            "title": self.title,
            "url": f"/api/assistant/artifacts/{self.id}",
            "inline": self.inline,
        }


_STORE: OrderedDict[str, Artifact] = OrderedDict()
_LOCK = threading.Lock()


def add(path: Path, filename: str, kind: str, title: str, conversation_id: str) -> Artifact:
    art = Artifact(
        id=uuid.uuid4().hex,
        path=Path(path),
        filename=filename,
        kind=kind if kind in KINDS else "file",
        title=title or filename,
        conversation_id=conversation_id,
        created=time.time(),
    )
    with _LOCK:
        _STORE[art.id] = art
        while len(_STORE) > MAX_ARTIFACTS:
            _, old = _STORE.popitem(last=False)
            old.path.unlink(missing_ok=True)
            log.info("artifact evicted: %s (%s)", old.filename, old.id)
    return art


def get(artifact_id: str) -> Artifact | None:
    with _LOCK:
        art = _STORE.get(artifact_id)
        if art is not None:
            _STORE.move_to_end(artifact_id)
    return art


def for_conversation(conversation_id: str) -> list[Artifact]:
    with _LOCK:
        return [a for a in _STORE.values() if a.conversation_id == conversation_id]


def drop_for(conversation_id: str) -> int:
    """Remove a conversation's artifacts (files go with its temp dir)."""
    with _LOCK:
        doomed = [k for k, a in _STORE.items() if a.conversation_id == conversation_id]
        for k in doomed:
            del _STORE[k]
    return len(doomed)
