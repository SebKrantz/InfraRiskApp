"""The frontend's last app-state snapshot, for MCP clients.

The in-app assistant gets the state with every chat turn; an external MCP client
has no browser, so it reads this instead.
"""

from __future__ import annotations

import threading
import time
from typing import Any

_LOCK = threading.Lock()
_STATE: dict[str, Any] = {}
_AT: float = 0.0


def update(state: dict[str, Any]) -> None:
    global _AT
    with _LOCK:
        _STATE.clear()
        _STATE.update(state)
        _AT = time.time()


def read() -> dict[str, Any]:
    with _LOCK:
        if not _STATE:
            return {"available": False, "note": "no browser session has reported state"}
        return {
            "available": True,
            "age_seconds": round(time.time() - _AT, 1),
            "state": dict(_STATE),
        }
