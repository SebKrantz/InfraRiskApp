"""Rough in-memory sizes, for the dataset quotas and the cache budgets.

Exactness does not matter here, scale does: a national line network and its
analysis run to hundreds of MB, a district's points to kilobytes. Long lists
(the sample points along every line) are sized from a sample of their items.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pandas as pd

_SAMPLE = 64


def approx_bytes(obj: Any) -> int:
    if obj is None:
        return 0
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    if isinstance(obj, pd.DataFrame):
        return _frame_bytes(obj)
    if isinstance(obj, dict):
        return 232 + sum(approx_bytes(v) + 50 for v in obj.values())
    if isinstance(obj, (list, tuple)):
        if not obj:
            return 56
        head = obj[:_SAMPLE]
        per_item = sum(approx_bytes(v) for v in head) / len(head)
        return int(56 + len(obj) * (8 + per_item))
    if hasattr(obj, "geom_type"):  # a shapely geometry
        import shapely

        return 100 + 16 * int(shapely.get_num_coordinates(obj))
    return sys.getsizeof(obj)


def _frame_bytes(frame: pd.DataFrame) -> int:
    import shapely

    total = int(frame.memory_usage(index=True, deep=False).sum())
    for name in frame.columns:
        col = frame[name]
        if col.dtype.name == "geometry":
            coords = shapely.get_num_coordinates(np.asarray(col.array, dtype=object))
            total += int(coords.sum()) * 16 + 100 * len(col)
        elif col.dtype == object:
            total += 50 * len(col)  # a short string or a None per cell
    return total
