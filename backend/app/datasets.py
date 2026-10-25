"""The infrastructure datasets the app analyses, each with an owner.

A dataset belongs to the principal that loaded it (app/identity.py): the
signed-in user for a sidebar upload and for what their in-app assistant loads
(one workspace per person, so the assistant and the sidebar see the same
datasets), or the MCP scope for what an external client loads. Every read names
the principal and finds only its own; another owner's file_id answers exactly
like one that does not exist. In single-user mode everything belongs to
AEI_LOCAL_USER, so every caller sees every dataset, as before.

Lifecycle: a dataset unused for DATASET_TTL_HOURS is dropped; an owner keeps at
most DATASET_MAX_PER_OWNER datasets and DATASET_MAX_MB_PER_OWNER megabytes, the
least recently used going first when a new one arrives; an MCP scope's datasets
go with the scope. Dropping a dataset purges its cached rasters and analyses.

STORE is the dict `app.api.upload.uploaded_files` has always been (file_id ->
{filename, geometry_type, feature_count, crs, bounds, gdf}) with `owner`,
`scope`, `bytes` and `last_used` beside them. Go through the functions here.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any

from . import config, identity
from .utils.sizing import approx_bytes

STORE: dict[str, dict[str, Any]] = {}
_LOCK = threading.RLock()
_SWEEP_EVERY_S = 60.0
_last_sweep = 0.0


class DatasetNotFound(LookupError):
    """No such dataset for this principal (absent, expired or another owner's)."""


class QuotaExceeded(ValueError):
    """A single dataset larger than an owner may hold."""


def owner_of(info: dict[str, Any]) -> str:
    return info.get("owner") or identity.local_user()


def owner_for(file_id: str) -> str | None:
    with _LOCK:
        info = STORE.get(file_id)
        return owner_of(info) if info is not None else None


def add(owner: str, info: dict[str, Any], scope: str | None = None) -> str:
    """Register a dataset for `owner` and return its new, server-issued file_id.
    `scope` is the conversation that loaded it, if any."""
    nbytes = approx_bytes(info["gdf"])
    limit = config.DATASET_MAX_MB_PER_OWNER * 2**20
    if nbytes > limit:
        raise QuotaExceeded(
            f"the dataset needs about {nbytes / 2**20:,.0f} MB in memory; one user may "
            f"hold at most {config.DATASET_MAX_MB_PER_OWNER:,.0f} MB"
        )
    file_id = str(uuid.uuid4())
    now = time.time()
    with _LOCK:
        doomed = _sweep(now, force=True)
        mine = sorted(
            (fid for fid, i in STORE.items() if owner_of(i) == owner),
            key=lambda fid: STORE[fid]["last_used"],
        )
        used = sum(STORE[fid].get("bytes", 0) for fid in mine)
        while mine and (len(mine) >= config.DATASET_MAX_PER_OWNER or used + nbytes > limit):
            fid = mine.pop(0)
            used -= STORE[fid].get("bytes", 0)
            del STORE[fid]
            doomed.append(fid)
        STORE[file_id] = {
            **info,
            "owner": owner,
            "scope": scope,
            "bytes": nbytes,
            "created": now,
            "last_used": now,
        }
    _purge_caches(doomed)
    return file_id


def get(owner: str, file_id: str) -> dict[str, Any]:
    """The owner's dataset `file_id`; raises DatasetNotFound otherwise."""
    with _LOCK:
        doomed = _sweep(time.time())
        info = STORE.get(file_id)
        if info is not None and owner_of(info) == owner:
            info["last_used"] = time.time()
    _purge_caches(doomed)
    if info is None or owner_of(info) != owner:
        raise DatasetNotFound(file_id)
    return info


def list_for(owner: str) -> list[tuple[str, dict[str, Any]]]:
    with _LOCK:
        doomed = _sweep(time.time())
        mine = [(fid, info) for fid, info in STORE.items() if owner_of(info) == owner]
    _purge_caches(doomed)
    return mine


def remove(owner: str, file_id: str) -> None:
    with _LOCK:
        info = STORE.get(file_id)
        if info is None or owner_of(info) != owner:
            raise DatasetNotFound(file_id)
        del STORE[file_id]
    _purge_caches([file_id])


def purge_scope(scope: str) -> int:
    """Drop the datasets an MCP scope loaded for itself, when the scope goes.
    A person's datasets stay (their map may still show them) until TTL or quota."""
    with _LOCK:
        doomed = [
            fid for fid, info in STORE.items()
            if info.get("scope") == scope and identity.is_scope_principal(owner_of(info))
        ]
        for fid in doomed:
            del STORE[fid]
    _purge_caches(doomed)
    return len(doomed)


def _sweep(now: float, force: bool = False) -> list[str]:
    """Under _LOCK: take out datasets unused for DATASET_TTL_HOURS (at most once a
    minute unless forced). Entries written without a timestamp start the clock now."""
    global _last_sweep
    if not force and now - _last_sweep < _SWEEP_EVERY_S:
        return []
    _last_sweep = now
    ttl = config.DATASET_TTL_HOURS * 3600.0
    doomed = [fid for fid, info in STORE.items() if now - info.setdefault("last_used", now) > ttl]
    for fid in doomed:
        del STORE[fid]
    return doomed


def _purge_caches(file_ids: list[str]) -> None:
    if not file_ids:
        return
    from .api.analyze import clear_raster_cache_for_file

    for fid in file_ids:
        clear_raster_cache_for_file(fid)
