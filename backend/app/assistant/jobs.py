"""Background jobs for long MCP calls (AEI MCP contract §7).

`start_<tool>` returns a job id at once; the tool runs on a daemon thread with
a progress callback and a cancel flag it checks between steps (layers, for
compare_hazards). Jobs belong to the scope that started them, live in memory
only, and the store keeps the most recent MAX_JOBS.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable

from .. import config
from .conversations import Conversation

MAX_JOBS = 64


@dataclass
class Job:
    id: str
    conversation_id: str
    tool: str
    estimate_s: float
    created: float = field(default_factory=time.time)
    status: str = "queued"  # queued | running | done | error | cancelled
    progress: float = 0.0
    message: str = ""
    result: Any = None
    error: str | None = None
    started: float | None = None
    finished: float | None = None
    cancel: threading.Event = field(default_factory=threading.Event)

    def report(self, progress: float, message: str) -> None:
        self.progress = max(0.0, min(1.0, progress))
        self.message = message

    def public(self) -> dict[str, Any]:
        now = self.finished or time.time()
        out: dict[str, Any] = {
            "job_id": self.id,
            "tool": self.tool,
            "status": self.status,
            "progress": round(self.progress, 3),
            "message": self.message,
            "elapsed_s": round(now - (self.started or self.created), 1),
            "estimate_s": self.estimate_s,
        }
        if self.status in ("done", "cancelled") and self.result is not None:
            out["result"] = self.result  # a cancelled ladder keeps its finished layers
        if self.error:
            out["error"] = self.error
        return out


_STORE: OrderedDict[str, Job] = OrderedDict()
_LOCK = threading.Lock()


def start(
    conv: Conversation,
    tool: str,
    work: Callable[[Job], Any],
    estimate_s: float,
) -> Job:
    """Run `work(job)` in the background; it reports through job.report and
    should stop early once job.cancel is set."""
    job = Job(id=uuid.uuid4().hex, conversation_id=conv.id, tool=tool, estimate_s=estimate_s)
    with _LOCK:
        _STORE[job.id] = job
        while len(_STORE) > MAX_JOBS:
            _STORE.popitem(last=False)

    def run() -> None:
        job.status, job.started = "running", time.time()
        try:
            result = work(job)
        except Exception as exc:  # noqa: BLE001 — reported through get_job
            job.status = "cancelled" if job.cancel.is_set() else "error"
            job.error = str(exc) or type(exc).__name__
        else:
            if job.cancel.is_set():
                job.status, job.result = "cancelled", result
            else:
                job.status, job.result, job.progress = "done", result, 1.0
        finally:
            job.finished = time.time()

    threading.Thread(target=run, name=f"job {tool} {job.id[:8]}", daemon=True).start()
    return job


def get(conv: Conversation, job_id: str) -> Job:
    with _LOCK:
        job = _STORE.get(job_id)
        mine = [j.id for j in _STORE.values() if j.conversation_id == conv.id]
    if job is None or job.conversation_id != conv.id:
        raise ValueError(f"no job {job_id!r} in this scope; its jobs: {mine or 'none'}")
    if job.status == "running" and time.time() - (job.started or job.created) > config.ASSISTANT_JOB_TIMEOUT:
        job.status = "error"
        job.error = (
            f"exceeded the {config.ASSISTANT_JOB_TIMEOUT:g}s job time limit; the work was "
            "abandoned (a remote raster read may have hung)"
        )
        job.finished = time.time()
    return job


def cancel_all(conversation_id: str) -> None:
    with _LOCK:
        for job in _STORE.values():
            if job.conversation_id == conversation_id:
                job.cancel.set()
