"""Bounds on the heavy work, shared by the web UI, the assistant and jobs.

Sampling a hazard raster under a large network is minutes of remote COG reads
on a pool of threads and hundreds of MB of memory. At most ANALYSIS_SLOTS of
those run at once in the process, whoever starts them; the rest wait their
turn (geospatial.analyze_intersection takes the slot). Background work goes to
one bounded executor of JOB_WORKERS threads rather than a thread per job.
"""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Callable, Iterator

from . import config

ANALYSIS_SLOTS = threading.BoundedSemaphore(max(1, config.ANALYSIS_SLOTS))
JOBS = ThreadPoolExecutor(max_workers=max(1, config.JOB_WORKERS), thread_name_prefix="job")

# Waiting for a slot polls, so python_exec's deadline (an asynchronous exception)
# can still stop a script that is queueing for one.
_POLL_S = 0.5


@contextmanager
def analysis_slot() -> Iterator[None]:
    slots = ANALYSIS_SLOTS
    while not slots.acquire(timeout=_POLL_S):
        pass
    try:
        yield
    finally:
        slots.release()


def submit_job(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
    """Run `fn` on the bounded job executor."""
    return JOBS.submit(fn, *args, **kwargs)
