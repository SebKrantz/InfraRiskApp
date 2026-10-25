"""Each user's last app-state snapshot from the browser, for MCP clients.

The in-app assistant gets the state with every chat turn; an external MCP client
has no browser, so it reads this instead — the snapshot of its own principal
only (app/identity.py), never another user's screen.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any

MAX_STATES = 256  # principals remembered; the least recently updated go first

_LOCK = threading.Lock()
_STATES: OrderedDict[str, tuple[dict[str, Any], float]] = OrderedDict()


def update(owner: str, state: dict[str, Any]) -> None:
    with _LOCK:
        _STATES[owner] = (dict(state), time.time())
        _STATES.move_to_end(owner)
        while len(_STATES) > MAX_STATES:
            _STATES.popitem(last=False)


def read(owner: str) -> dict[str, Any]:
    with _LOCK:
        entry = _STATES.get(owner)
    if entry is None or not entry[0]:
        return {"available": False, "note": "no browser session has reported state"}
    state, at = entry
    return {
        "available": True,
        "age_seconds": round(time.time() - at, 1),
        "state": dict(state),
    }
