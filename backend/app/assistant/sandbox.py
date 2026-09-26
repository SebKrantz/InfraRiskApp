"""Confinement for python_exec (AEI MCP contract §8, step P1).

User code runs in the server process, so this is a fence, not a wall: it
stops the moves an agent actually makes — reading or listing the user's disk,
reading the server's own source, spawning processes, importing the server
package, reaching into interpreter internals — and it can stop a runaway
script. A subprocess sandbox is the step after this one.

Mechanism: a `sys.addaudithook` hook that acts only on threads registered in
`_POLICIES` (the python_exec worker and any thread it starts), a restricted
`__import__` in the namespace's builtins, and an AST check before execution.
Allowed: read/write in the scope's workdir and the system temp directory
(except other conversations' workdirs); read-only in the app's data/
directory and the Python installation (library code reads its own files).
"""

from __future__ import annotations

import ast
import builtins
import ctypes
import os
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from ..config import settings

WORKDIR_PREFIX = "infrarisk-assistant-"
_BACKEND = os.path.realpath(Path(__file__).resolve().parents[2])
_TMP = os.path.realpath(tempfile.gettempdir())
_DEVICES = ("/dev/null", "/dev/urandom", "/dev/random")
# Never written, and read only from the Python installation: a module planted
# in the workdir would be importable process-wide.
_CODE_SUFFIXES = (".py", ".pyc", ".pyo", ".pyd", ".so", ".dylib", ".pth")

# Modules user code may not import: the server itself, and the handles that
# lead straight back into it or out of the process.
BLOCKED_IMPORTS = frozenset(
    {"app", "main", "ctypes", "importlib", "builtins", "gc", "inspect",
     "subprocess", "multiprocessing"}
)
# Attribute names that reach interpreter internals (globals of a function,
# frames, closures) — refused whether written as `x.attr` or as a string.
_DENIED_ATTRS = frozenset(
    {"__globals__", "__builtins__", "__subclasses__", "__code__", "__closure__",
     "__loader__", "f_globals", "f_locals", "f_builtins", "f_back", "gi_frame",
     "cr_frame", "ag_frame", "tb_frame", "cell_contents"}
)
_DENIED_EVENTS = frozenset(
    {"subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn",
     "os.fork", "os.forkpty", "os.kill", "os.killpg", "pty.spawn", "os.startfile",
     "os.startfile/2", "os.symlink", "ctypes.dlopen", "ctypes.dlsym",
     "ctypes.dlsym/handle", "sys.addaudithook", "sys.remote_exec",
     "_thread.start_new_thread"}
)
# Events whose first argument is a path being changed.
_WRITE_EVENTS = frozenset(
    {"os.mkdir", "os.rmdir", "os.remove", "os.truncate", "os.chmod", "os.chown",
     "os.utime", "os.chflags", "os.lchflags", "os.lchmod", "os.setxattr",
     "os.removexattr", "shutil.rmtree", "shutil.chown", "shutil.make_archive"}
)
# (source, destination) events: source read, destination written.
_COPY_EVENTS = frozenset(
    {"shutil.copyfile", "shutil.copymode", "shutil.copystat", "shutil.copytree"}
)


class ScopeViolation(PermissionError):
    """Raised inside user code when it reaches outside its scope. An OSError,
    so library code that probes paths treats it as 'not there'."""


class Stop(BaseException):
    """Raised asynchronously inside a python_exec worker at its deadline."""


def _inside(path: str, root: str) -> bool:
    return path == root or path.startswith(root + os.sep)


def _library_roots() -> tuple[str, ...]:
    roots = set()
    for entry in sys.path:
        if not entry or entry == ".":
            continue
        real = os.path.realpath(entry)
        if _inside(real, _BACKEND) or not os.path.exists(real):
            continue
        roots.add(real)
    try:
        import matplotlib

        roots.add(os.path.realpath(matplotlib.get_configdir()))
        roots.add(os.path.realpath(matplotlib.get_cachedir()))
    except Exception:  # noqa: BLE001
        pass
    try:
        import zoneinfo

        roots.update(os.path.realpath(p) for p in zoneinfo.TZPATH)
    except Exception:  # noqa: BLE001
        pass
    return tuple(sorted(roots))


_READ_ONLY = (os.path.realpath(settings.DATA_DIR),)
_LIBRARIES: tuple[str, ...] = ()


@dataclass(frozen=True)
class Policy:
    workdir: str

    @classmethod
    def for_workdir(cls, workdir: Path) -> "Policy":
        return cls(os.path.realpath(workdir))

    def _temp(self, path: str) -> bool:
        """Temp files, but not another conversation's workdir. (The temp root
        itself is opened by tempfile.TemporaryFile; listing it stays refused.)"""
        if path == _TMP:
            return True
        if not _inside(path, _TMP):
            return False
        first = path[len(_TMP) + 1:].split(os.sep, 1)[0]
        return not first.startswith(WORKDIR_PREFIX)

    def can_read(self, path: str) -> bool:
        if any(_inside(path, r) for r in _LIBRARIES):
            return True
        if path.endswith(_CODE_SUFFIXES):
            return False
        return (
            _inside(path, self.workdir)
            or any(_inside(path, r) for r in _READ_ONLY)
            or self._temp(path)
            or path in _DEVICES
        )

    def can_write(self, path: str) -> bool:
        if path.endswith(_CODE_SUFFIXES):
            return False
        return _inside(path, self.workdir) or self._temp(path) or path == "/dev/null"

    def can_list(self, path: str) -> bool:
        return (
            _inside(path, self.workdir)
            or any(_inside(path, r) for r in _READ_ONLY + _LIBRARIES)
        )

    def refuse(self, what: str, path: str | None = None) -> ScopeViolation:
        target = f" {path!r}" if path else ""
        return ScopeViolation(
            f"python_exec refused to {what}{target}: code runs confined to this "
            f"scope's workdir ({self.workdir}, the current directory) with the app's "
            "data/ directory read-only, and without subprocesses. Use `uploads` "
            "for uploaded files and save_artifact() to hand files back."
        )


# worker thread ident -> its policy; the hook ignores every other thread
_POLICIES: dict[int, Policy] = {}


def _path(value) -> str | None:
    if value is None:
        value = "."
    if isinstance(value, int):
        return None  # an already-open descriptor
    try:
        return os.path.realpath(os.fsdecode(value))
    except (TypeError, ValueError):
        return None


def _writes(mode, flags) -> bool:
    if isinstance(mode, str):
        return any(c in mode for c in "wax+")
    wanted = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
    return bool((flags or 0) & wanted)


def _audit(event: str, args: tuple) -> None:
    pol = _POLICIES.get(threading.get_ident())
    if pol is None:
        return
    if event == "open":
        path = _path(args[0])
        if path is None:
            return
        if _writes(args[1], args[2] if len(args) > 2 else 0):
            if not pol.can_write(path):
                raise pol.refuse("write", path)
        elif not pol.can_read(path):
            raise pol.refuse("read", path)
    elif event in ("os.listdir", "os.scandir"):
        path = _path(args[0] if args else None)
        if path is not None and not pol.can_list(path):
            raise pol.refuse("list", path)
    elif event == "os.chdir":
        path = _path(args[0])
        if path is not None and not _inside(path, pol.workdir):
            raise pol.refuse("change directory to", path)
    elif event in _WRITE_EVENTS:
        path = _path(args[0])
        if path is not None and not pol.can_write(path):
            raise pol.refuse("modify", path)
    elif event in ("os.rename", "os.link", "shutil.move"):
        for value in args[:2]:
            path = _path(value)
            if path is not None and not pol.can_write(path):
                raise pol.refuse("move or link", path)
    elif event in _COPY_EVENTS:
        src, dst = _path(args[0]), _path(args[1])
        if src is not None and not pol.can_read(src):
            raise pol.refuse("read", src)
        if dst is not None and not pol.can_write(dst):
            raise pol.refuse("write", dst)
    elif event == "shutil.unpack_archive":
        src, dst = _path(args[0]), _path(args[1])
        if src is not None and not pol.can_read(src):
            raise pol.refuse("read", src)
        if dst is not None and not pol.can_write(dst):
            raise pol.refuse("write", dst)
    elif event == "import":
        name = str(args[0])
        if name.split(".")[0] in ("app", "main"):
            raise ImportError(f"python_exec cannot import the server package ({name})")
    elif event in _DENIED_EVENTS:
        raise pol.refuse(f"use {event}")


_real_start = threading.Thread.start


def _start(self: threading.Thread) -> None:
    """Threads started by confined code are confined too, for their lifetime."""
    pol = _POLICIES.get(threading.get_ident())
    if pol is not None:
        target_run = self.run

        def run() -> None:
            _POLICIES[threading.get_ident()] = pol
            try:
                target_run()
            finally:
                _POLICIES.pop(threading.get_ident(), None)

        self.run = run  # type: ignore[method-assign]
    _real_start(self)


_INSTALLED = False
_INSTALL_LOCK = threading.Lock()


def install() -> None:
    """Install the audit hook and the thread-start wrapper, once per process."""
    global _INSTALLED, _LIBRARIES
    with _INSTALL_LOCK:
        if _INSTALLED:
            return
        import mimetypes

        mimetypes.init()  # reads /etc files; do it now, not inside user code
        _LIBRARIES = _library_roots()
        threading.Thread.start = _start  # type: ignore[method-assign]
        sys.addaudithook(_audit)
        _INSTALLED = True


def confine(ident: int, policy: Policy) -> None:
    _POLICIES[ident] = policy


def release(policy: Policy) -> None:
    for ident, pol in list(_POLICIES.items()):
        if pol is policy:
            _POLICIES.pop(ident, None)


# ---- the namespace side ----------------------------------------------------- #


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".")[0] in BLOCKED_IMPORTS:
        raise ImportError(
            f"python_exec does not allow importing {name!r} (confined kernel); the "
            "app's analysis functions are preloaded as geospatial, export_data, "
            "hazards, analyze and export"
        )
    return builtins.__import__(name, globals, locals, fromlist, level)


def safe_builtins() -> dict:
    """The builtins user code sees: the real ones with a guarded __import__."""
    b = dict(vars(builtins))
    b["__import__"] = _guarded_import
    return b


def check_source(tree: ast.AST) -> str | None:
    """Why this code may not run, or None."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if node.attr in _DENIED_ATTRS:
                return f"access to {node.attr}"
            if node.attr == "modules" and isinstance(node.value, ast.Name) and node.value.id == "sys":
                return "access to sys.modules"
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in _DENIED_ATTRS:
                return f"access to {node.value}"
        elif isinstance(node, ast.Name) and node.id in ("__builtins__", "__loader__"):
            return f"access to {node.id}"
    return None


# ---- stopping a worker ------------------------------------------------------ #

_PYAPI = ctypes.PyDLL(None)  # private handle: user code cannot reach its symbols
_SET_ASYNC_EXC = _PYAPI.PyThreadState_SetAsyncExc
_SET_ASYNC_EXC.argtypes = (ctypes.c_ulong, ctypes.py_object)


def stop(threads: list[threading.Thread], grace: float) -> bool:
    """Raise Stop in each thread until all have ended or `grace` runs out.
    Pure-Python loops stop at once; a call stuck in native code stops when it
    returns to Python."""
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        alive = [t for t in threads if t.is_alive()]
        if not alive:
            return True
        for t in alive:
            if t.ident is not None:
                _SET_ASYNC_EXC(t.ident, ctypes.py_object(Stop))
        time.sleep(0.05)
    return not any(t.is_alive() for t in threads)


def threads_of(policy: Policy) -> list[threading.Thread]:
    by_ident = {t.ident: t for t in threading.enumerate()}
    return [by_ident[i] for i, p in list(_POLICIES.items()) if p is policy and i in by_ident]
