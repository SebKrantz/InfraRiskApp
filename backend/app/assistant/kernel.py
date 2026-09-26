"""The python_exec kernel: one persistent namespace per conversation.

IPython-style semantics — all statements exec'd, a trailing expression eval'd
and returned — with stdout/stderr capture and matplotlib figure harvesting.

Confined (sandbox.py, AEI MCP contract §8 step P1): code runs on a worker
thread with the conversation's workdir as the current directory, may read and
write only there (plus temp files) and read the app's data/ directory, cannot
start processes or import the server package, and is stopped at its deadline.
The namespace holds a read-only view of the app's dataset store and facades
of the app modules, so a script can use the analysis layer but cannot edit
what the app is serving. Every file the call produces — save_artifact(),
figures, and files written into the workdir — comes back as an artifact.
"""

from __future__ import annotations

import ast
import copy
import io
import os
import sys
import threading
import traceback
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")  # before pyplot, and pyplot only inside the functions below

from .. import config
from . import artifacts, sandbox
from .conversations import Conversation

# One python_exec at a time in the process: the working directory and pyplot's
# figure state are both process-global.
_RUN_LOCK = threading.Lock()
# How long a stopped worker gets to unwind before it counts as stuck.
_STOP_GRACE_S = 5.0

MAX_OUTPUT_CHARS = 20_000

sandbox.install()


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


def _facade(module) -> SimpleNamespace:
    """A module's own public functions, classes and constants — without the
    state it imports (the dataset store, the caches)."""
    public: dict[str, Any] = {}
    for name, obj in vars(module).items():
        if name.startswith("_"):
            continue
        if callable(obj) and getattr(obj, "__module__", None) == module.__name__:
            public[name] = obj
        elif name.isupper() and isinstance(obj, (str, int, float, tuple, frozenset)):
            public[name] = obj
        elif name.isupper() and isinstance(obj, dict):
            public[name] = MappingProxyType(dict(obj))
    return SimpleNamespace(**public)


def _datasets_view() -> Mapping:
    """The app's dataset store as python_exec sees it: always current, read-only,
    and handing out copies of the GeoDataFrames."""
    from ..api.upload import uploaded_files as store

    refusal = (
        "uploaded_files is read-only in python_exec: datasets are registered by "
        "the loading tools (load_infrastructure and friends), not by assignment"
    )

    class UploadedFiles(Mapping):
        def __getitem__(self, key):
            info = store[key]
            return MappingProxyType({**info, "gdf": info["gdf"].copy()})

        def __iter__(self):
            return iter(list(store))

        def __len__(self):
            return len(store)

        def __setitem__(self, key, value):
            raise TypeError(refusal)

        def __delitem__(self, key):
            raise TypeError(refusal)

        def __repr__(self):
            return f"<uploaded_files (read-only): {list(store)}>"

    return UploadedFiles()


def ensure(conv: Conversation) -> dict[str, Any]:
    """Preload the heavy handles on first use, without clobbering stored results."""
    ns = conv.namespace
    ns["uploads"] = MappingProxyType(dict(conv.uploads))  # always current, read-only
    if conv.kernel_ready:
        return ns
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import rasterio

    from ..api import analyze, export, hazards
    from ..utils import export_data, geospatial

    ns["__builtins__"] = sandbox.safe_builtins()
    ns.setdefault("pd", pd)
    ns.setdefault("np", np)
    ns.setdefault("gpd", gpd)
    ns.setdefault("plt", plt)
    ns.setdefault("rasterio", rasterio)
    ns.setdefault("geospatial", _facade(geospatial))
    ns.setdefault("export_data", _facade(export_data))
    ns.setdefault("hazards", _facade(hazards))
    ns.setdefault("analyze", _facade(analyze))
    ns.setdefault("export", _facade(export))
    ns.setdefault("uploaded_files", _datasets_view())
    ns.setdefault("HAZARDS", copy.deepcopy(hazards.load_hazards_dict()))

    def save_artifact(obj: Any, filename: str, title: str | None = None) -> str:
        """Save a DataFrame / GeoDataFrame / bytes / matplotlib Figure / path as
        a downloadable artifact; returns its id. The extension picks the format
        (.csv / .xlsx / .png / .gpkg / .geojson); a GeoDataFrame without one of
        those becomes a GeoPackage."""
        art = _save_artifact(conv, obj, filename, title)
        conv._exec_saved.append(art)
        return art.id

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
    if isinstance(obj, gpd.GeoDataFrame) and suffix == ".geojson":
        obj.to_file(dest, driver="GeoJSON")
        kind = "file"
    elif isinstance(obj, gpd.GeoDataFrame) and suffix not in (".csv", ".xlsx"):
        if suffix != ".gpkg":
            dest = dest.with_suffix(".gpkg")
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


def _snapshot(workdir: Path) -> dict[str, tuple[int, int]]:
    out = {}
    for entry in os.scandir(workdir):
        if entry.is_file(follow_symlinks=False):
            st = entry.stat()
            out[entry.name] = (st.st_mtime_ns, st.st_size)
    return out


def _worker(ns: dict, tree: ast.Module, trailing: ast.Expression | None,
            out: dict, ready: threading.Event) -> None:
    """Runs on the confined thread: the user's code and nothing else."""
    ready.wait()
    buf = io.StringIO()
    out_tee, err_tee = _tees()
    out_tee.capture(buf)
    err_tee.capture(buf)
    try:
        exec(compile(tree, "<assistant>", "exec"), ns)  # noqa: S102
        value = (
            eval(compile(trailing, "<assistant>", "eval"), ns)  # noqa: S307
            if trailing is not None
            else None
        )
        # repr runs user code (a class's __repr__) — keep it confined too
        out.update(ok=True, result=_clip(repr(value), 4000) if value is not None else None)
    except sandbox.Stop:
        out.update(ok=False, stopped=True)
    except Exception as exc:  # noqa: BLE001 — reported to the model, never raised
        # from the user's own frame down, not this worker's
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__.tb_next, limit=6))
        out.update(ok=False, error=_clip(tb, 4000))
    finally:
        out_tee.release()
        err_tee.release()
        out["stdout"] = buf.getvalue()


def _failed(error: str, stdout: str = "") -> dict[str, Any]:
    return {"ok": False, "stdout": stdout, "result": None, "figures": [],
            "artifacts": [], "error": error}


def execute(conv: Conversation, code: str, timeout: float | None = None) -> dict[str, Any]:
    """Run `code` in the conversation namespace, confined. Returns
    {ok, stdout, result, figures: [Artifact], artifacts: [Artifact], error?}."""
    timeout = timeout or config.ASSISTANT_EXEC_TIMEOUT
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError:
        return _failed(traceback.format_exc(limit=0).strip())
    refused = sandbox.check_source(tree)
    if refused:
        return _failed(
            f"python_exec refused this code ({refused}): the kernel is confined to "
            "its scope and does not expose interpreter internals"
        )
    trailing: ast.Expression | None = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        trailing = ast.Expression(tree.body.pop(-1).value)

    with conv.exec_lock:
        prev = getattr(conv, "_exec_thread", None)
        if prev is not None and prev.is_alive():
            return _failed(
                "a previous python_exec is still stuck in native code and could not "
                "be stopped — wait for it or start a new conversation"
            )
        if not _RUN_LOCK.acquire(timeout=min(timeout, 60.0)):
            return _failed(
                "python_exec is busy in another conversation; try again in a moment"
            )
        stuck = False
        try:
            ns = ensure(conv)
            workdir = conv.workdir
            policy = sandbox.Policy.for_workdir(workdir)
            before = _snapshot(workdir)
            home = os.getcwd()
            conv._exec_saved = []
            out: dict[str, Any] = {}
            ready = threading.Event()
            worker = threading.Thread(
                target=_worker, args=(ns, tree, trailing, out, ready),
                name=f"python_exec {conv.id}", daemon=True,
            )
            import matplotlib.pyplot as plt

            plt.close("all")
            os.chdir(workdir)
            worker.start()
            sandbox.confine(worker.ident, policy)
            conv._exec_thread = worker
            ready.set()
            worker.join(timeout)
            timed_out = worker.is_alive()
            if timed_out:
                sandbox.stop(sandbox.threads_of(policy) or [worker], _STOP_GRACE_S)
                stuck = worker.is_alive()
            if stuck:
                # Native code that never returns to Python cannot be stopped;
                # whoever finishes it restores the process state.
                threading.Thread(
                    target=_reap, args=(worker, policy, home), daemon=True
                ).start()
                return _failed(
                    f"timed out after {timeout:.0f}s and is stuck in native code; "
                    "python_exec is unavailable until that call returns"
                )
            sandbox.release(policy)
            os.chdir(home)
            made = list(conv._exec_saved)
            made += _register_written(conv, workdir, before, made)
            figs = _harvest_figures(conv)
        finally:
            if not stuck:
                _RUN_LOCK.release()

    made += figs
    listed = [_listing(a) for a in made]
    if timed_out:
        res = _failed(
            f"stopped after {timeout:.0f}s (python_exec time limit); the namespace "
            "is intact — split the work into smaller steps",
            _clip(out.get("stdout", "")),
        )
        res.update(figures=figs, artifacts=listed)
        return res
    res = {
        "ok": bool(out.get("ok")),
        "stdout": _clip(out.get("stdout", "")),
        "result": out.get("result"),
        "figures": figs,
        "artifacts": listed,
    }
    if not res["ok"]:
        res["error"] = out.get("error") or "stopped"
    return res


def _reap(worker: threading.Thread, policy: sandbox.Policy, home: str) -> None:
    worker.join()
    sandbox.release(policy)
    os.chdir(home)
    _RUN_LOCK.release()


def _register_written(
    conv: Conversation, workdir: Path, before: dict, saved: list[artifacts.Artifact]
) -> list[artifacts.Artifact]:
    """Files the code wrote into the workdir with plain I/O (to_csv, to_file…)
    become artifacts too, so nothing a script produces stays invisible."""
    skip = {Path(p).name for p in conv.uploads.values()} | {a.path.name for a in saved}
    made = []
    for name, stamp in _snapshot(workdir).items():
        if name in skip or before.get(name) == stamp:
            continue
        path = workdir / name
        made.append(
            artifacts.add(path, name, _kind_for(path.suffix.lower()), name, conv.id)
        )
    return made


def _harvest_figures(conv: Conversation) -> list[artifacts.Artifact]:
    import matplotlib.pyplot as plt

    figs: list[artifacts.Artifact] = []
    for num in plt.get_fignums():
        fig = plt.figure(num)
        if not fig.get_axes():
            continue
        conv._counter += 1
        path = conv.workdir / f"figure_{conv._counter}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        figs.append(artifacts.add(path, path.name, "png", f"Figure {conv._counter}", conv.id))
    plt.close("all")
    return figs


def _listing(art: artifacts.Artifact) -> dict[str, Any]:
    return {
        "url": art.public()["url"],
        "filename": art.filename,
        "bytes": art.path.stat().st_size if art.path.exists() else 0,
        "kind": art.kind,
    }


def _clip(s: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n... [truncated at {limit} chars]"
