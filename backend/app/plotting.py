"""Matplotlib with several users at once.

pyplot keeps one process-wide registry of figures. Only python_exec draws with
it (kernel.py), and it collects and closes just the figures its own call made.
Everything else — the export endpoints, the assistant's charts, maps and curve
plots — draws on standalone Figure objects that pyplot never sees, so nothing
one user is drawing can vanish in another's close or land among their figures.
LOCK serialises the app's rendering, python_exec's figure harvest included:
matplotlib's internals are not thread-safe.
"""

from __future__ import annotations

import functools
import threading
from typing import Any, Callable, TypeVar

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

LOCK = threading.RLock()

F = TypeVar("F", bound=Callable[..., Any])


def subplots(nrows: int = 1, ncols: int = 1, *, figsize=None, **kwargs: Any):
    """plt.subplots without pyplot: a standalone Figure on an Agg canvas (as
    pyplot would give it under the Agg backend) and its axes."""
    fig = Figure(figsize=figsize)
    FigureCanvasAgg(fig)
    return fig, fig.subplots(nrows, ncols, **kwargs)


def locked(fn: F) -> F:
    """Run a rendering function under LOCK."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with LOCK:
            return fn(*args, **kwargs)

    return wrapper  # type: ignore[return-value]
