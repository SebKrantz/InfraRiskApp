"""The python_exec kernel: one persistent namespace per conversation.

IPython-style semantics — all statements exec'd, a trailing expression eval'd
and returned — with stdout/stderr capture and matplotlib figure harvesting.
Unsandboxed by design: this is a local, single-user tool and the whole backend
is already trusted code. The namespace holds the live `uploaded_files` store,
so `python_exec` sees exactly the datasets the app is serving.
"""

from __future__ import annotations

import ast
import io
import sys
import threading
import traceback
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # before pyplot, and pyplot only inside the functions below

from .. import config
from . import artifacts
from .conversations import Conversation

# pyplot state is process-global; one figure harvest at a time.
_FIG_LOCK = threading.Lock()

MAX_OUTPUT_CHARS = 20_000


class _ThreadTee(io.TextIOBase):
    """A sys.stdout/stderr stand-in that redirects per-thread.

    `contextlib.redirect_stdout` swaps the stream process-wide, so a runaway
    worker would keep swallowing every other thread's output after a timeout.
    This proxy sends a thread's writes to its registered buffer and everyone
    else's to the original stream.
    """

    def __init__(self, default) -> None:
        self.default = default
        self._local = threading.local()

    def capture(self, buf: io.StringIO) -> None:
        self._local.target = buf

    def release(self) -> None:
        self._local.target = None

    @property
    def _target(self):
        return getattr(self._local, "target", None) or self.default

    def write(self, s: str) -> int:
        return self._target.write(s)

    def flush(self) -> None:
        self._target.flush()

    def isatty(self) -> bool:
        return False


_TEE_LOCK = threading.Lock()


def _tees() -> tuple[_ThreadTee, _ThreadTee]:
    """Install the proxies once, lazily, and return them."""
    with _TEE_LOCK:
        if not isinstance(sys.stdout, _ThreadTee):
            sys.stdout = _ThreadTee(sys.stdout)
        if not isinstance(sys.stderr, _ThreadTee):
            sys.stderr = _ThreadTee(sys.stderr)
        return sys.stdout, sys.stderr


def ensure(conv: Conversation) -> dict[str, Any]:
    """Preload the heavy handles on first use, without clobbering stored results."""
    ns = conv.namespace
    if conv.kernel_ready:
        ns["uploads"] = conv.uploads  # always current
        return ns
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import rasterio

    from ..api import analyze, export, hazards
    from ..api.upload import uploaded_files
    from ..utils import export_data, geospatial

    ns.setdefault("pd", pd)
    ns.setdefault("np", np)
    ns.setdefault("gpd", gpd)
    ns.setdefault("plt", plt)
    ns.setdefault("rasterio", rasterio)
    ns.setdefault("geospatial", geospatial)
    ns.setdefault("export_data", export_data)
    ns.setdefault("hazards", hazards)
    ns.setdefault("analyze", analyze)
    ns.setdefault("export", export)
    ns.setdefault("uploaded_files", uploaded_files)
    ns.setdefault("HAZARDS", hazards.load_hazards_dict())
    ns["uploads"] = conv.uploads

    def save_artifact(obj: Any, filename: str, title: str | None = None) -> str:
        """Save a DataFrame / GeoDataFrame / bytes / matplotlib Figure / path as
        a downloadable artifact; returns its id. The extension picks the format
        (.csv / .xlsx / .png / .gpkg)."""
        return _save_artifact(conv, obj, filename, title).id

    ns["save_artifact"] = save_artifact
    conv.kernel_ready = True
    return ns


def _save_artifact(
    conv: Conversation, obj: Any, filename: str, title: str | None
) -> artifacts.Artifact:
    import geopandas as gpd
    import matplotlib.pyplot as plt  # noqa: F401 — ensures Agg is live
    import pandas as pd

    safe = Path(filename).name or "artifact"
    dest = conv.workdir / safe
    suffix = dest.suffix.lower()
    if isinstance(obj, gpd.GeoDataFrame) and suffix == ".gpkg":
        obj.to_file(dest, driver="GPKG")
        kind = "gpkg"
    elif isinstance(obj, pd.DataFrame):
        if suffix == ".xlsx":
            obj.to_excel(dest, index=False)
            kind = "xlsx"
        else:
            if suffix != ".csv":
                dest = dest.with_suffix(".csv")
            frame = obj.drop(columns="geometry") if "geometry" in obj.columns else obj
            frame.to_csv(dest, index=False)
            kind = "csv"
    elif hasattr(obj, "savefig"):  # matplotlib Figure
        if suffix != ".png":
            dest = dest.with_suffix(".png")
        obj.savefig(dest, dpi=150, bbox_inches="tight")
        kind = "png"
    elif isinstance(obj, (bytes, bytearray)):
        dest.write_bytes(bytes(obj))
        kind = _kind_for(suffix)
    elif isinstance(obj, (str, Path)) and Path(obj).is_file():
        dest.write_bytes(Path(obj).read_bytes())
        kind = _kind_for(suffix)
    else:
        raise TypeError(f"cannot save a {type(obj).__name__} as an artifact")
    return artifacts.add(dest, dest.name, kind, title or dest.name, conv.id)


def _kind_for(suffix: str) -> str:
    return {
        ".png": "png",
        ".csv": "csv",
        ".xlsx": "xlsx",
        ".docx": "docx",
        ".gpkg": "gpkg",
    }.get(suffix, "file")


def execute(conv: Conversation, code: str, timeout: float | None = None) -> dict[str, Any]:
    """Run `code` in the conversation namespace. Returns
    {ok, stdout, result, figures: [Artifact], error?}."""
    timeout = timeout or config.ASSISTANT_EXEC_TIMEOUT
    out: dict[str, Any] = {}

    def run() -> None:
        ns = ensure(conv)
        buf = io.StringIO()
        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError:
            out.update(
                ok=False,
                stdout="",
                result=None,
                figures=[],
                error=traceback.format_exc(limit=0).strip(),
            )
            return
        trailing: ast.Expression | None = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            trailing = ast.Expression(tree.body.pop(-1).value)
        figs: list[artifacts.Artifact] = []
        out_tee, err_tee = _tees()
        with _FIG_LOCK:
            import matplotlib.pyplot as plt

            plt.close("all")
            out_tee.capture(buf)
            err_tee.capture(buf)
            try:
                exec(compile(tree, "<assistant>", "exec"), ns)  # noqa: S102
                value = (
                    eval(compile(trailing, "<assistant>", "eval"), ns)  # noqa: S307
                    if trailing is not None
                    else None
                )
            except Exception:  # noqa: BLE001 — reported to the model, never raised
                tb = traceback.format_exc(limit=6)
                out.update(
                    ok=False,
                    stdout=_clip(buf.getvalue()),
                    result=None,
                    figures=[],
                    error=_clip(tb, 4000),
                )
                plt.close("all")
                return
            finally:
                out_tee.release()
                err_tee.release()
            # harvest any figures the code created
            for num in plt.get_fignums():
                fig = plt.figure(num)
                if not fig.get_axes():
                    continue
                conv._counter += 1
                path = conv.workdir / f"figure_{conv._counter}.png"
                fig.savefig(path, dpi=150, bbox_inches="tight")
                figs.append(
                    artifacts.add(path, path.name, "png", f"Figure {conv._counter}", conv.id)
                )
            plt.close("all")
        out.update(
            ok=True,
            stdout=_clip(buf.getvalue()),
            result=_clip(repr(value), 4000) if value is not None else None,
            figures=figs,
        )

    with conv.exec_lock:
        # A previous call that timed out may still be running (unkillable);
        # refuse to stack a second execution on the same namespace.
        prev = getattr(conv, "_exec_thread", None)
        if prev is not None and prev.is_alive():
            return {
                "ok": False,
                "stdout": "",
                "result": None,
                "figures": [],
                "error": "a previous python_exec is still running — wait for it "
                "or start a new conversation",
            }
        worker = threading.Thread(target=run, daemon=True)
        conv._exec_thread = worker
        worker.start()
        worker.join(timeout)
        if worker.is_alive():
            return {
                "ok": False,
                "stdout": "",
                "result": None,
                "figures": [],
                "error": (
                    f"timed out after {timeout:.0f}s — the code is still running "
                    "in the background and cannot be interrupted; avoid unbounded "
                    "loops and prefer smaller steps"
                ),
            }
    return out


def _clip(s: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n... [truncated at {limit} chars]"
