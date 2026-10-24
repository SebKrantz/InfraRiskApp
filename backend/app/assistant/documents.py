"""Uploaded document handling: previews and full-content extraction.

A document attached to a chat message gets an automatic preview block in the
message itself (so the model always knows what arrived and roughly what is in
it); `read_document` serves the full content on demand, and tabular files land
in the python namespace as DataFrames.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

PREVIEW_CHARS = 2_500
FULL_CHARS = 60_000

TEXT_SUFFIXES = {".md", ".txt", ".markdown", ".rst", ".json", ".yml", ".yaml"}
TABULAR_SUFFIXES = {".csv", ".xlsx", ".xls"}
SPATIAL_SUFFIXES = {".gpkg", ".zip", ".geojson", ".shp"}


def kind_of(path: Path) -> str:
    s = path.suffix.lower()
    if s in SPATIAL_SUFFIXES:
        return "spatial"
    if s in TEXT_SUFFIXES:
        return "text"
    if s == ".docx":
        return "docx"
    if s in TABULAR_SUFFIXES:
        return "tabular"
    return "other"


def _clip(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n... [truncated at {limit} chars — read_document returns more]"


def _docx_text(path: Path, limit: int) -> str:
    import docx

    doc = docx.Document(str(path))
    parts: list[str] = []
    total = 0
    for p in doc.paragraphs:
        if p.text.strip():
            line = f"# {p.text}" if p.style.name.startswith("Heading") else p.text
            parts.append(line)
            total += len(line)
            if total > limit:
                break
    for t in doc.tables:
        rows = [" | ".join(c.text for c in r.cells) for r in t.rows[:20]]
        parts.append("[table]\n" + "\n".join(rows))
        total += sum(len(r) for r in rows)
        if total > limit:
            break
    return _clip("\n".join(parts), limit)


def _tabular_preview(path: Path) -> str:
    import pandas as pd

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, nrows=6, sep=None, engine="python")
        return (
            f"CSV, {len(df.columns)} columns: {', '.join(map(str, df.columns[:20]))}\n"
            f"head:\n{df.to_string(max_cols=10)}"
        )
    xl = pd.ExcelFile(path)
    lines = [f"Excel workbook, sheets: {', '.join(xl.sheet_names)}"]
    for name in xl.sheet_names[:3]:
        df = xl.parse(name, nrows=5)
        lines.append(f"-- {name} ({len(df.columns)} cols):\n{df.to_string(max_cols=10)}")
    return "\n".join(lines)


def preview(path: Path) -> str | None:
    """A short look inside, for the message context block. None = no preview."""
    kind = kind_of(path)
    try:
        if kind == "text":
            return _clip(path.read_text(errors="replace"), PREVIEW_CHARS)
        if kind == "docx":
            return _docx_text(path, PREVIEW_CHARS)
        if kind == "tabular":
            return _clip(_tabular_preview(path), PREVIEW_CHARS)
    except Exception as exc:  # noqa: BLE001 — a broken file is context too
        return f"[could not read: {exc}]"
    return None


def context_note(name: str, path: Path) -> str:
    """The block attached to the user message for a non-image upload."""
    kind = kind_of(path)
    size = path.stat().st_size
    access = {
        "spatial": "Infrastructure data: call load_infrastructure to bring it "
        "into the app (read_guide('exposure_analysis') first).",
        "tabular": "If this is a vulnerability curve (intensity vs proportion "
        "destroyed), call load_vulnerability_curve. Otherwise read_document "
        "loads it into the python namespace as a DataFrame.",
        "text": "Context document — full content via read_document.",
        "docx": "Context document — full content via read_document.",
        "other": "Available at uploads[...] in python_exec.",
    }[kind]
    head = f"[Attached file: {name} ({size:,} bytes). {access}]"
    body = preview(path)
    return head if body is None else f"{head}\n--- preview ---\n{body}"


def full_content(path: Path) -> dict[str, Any]:
    """The read_document payload for text-like files."""
    kind = kind_of(path)
    if kind == "text":
        return {"kind": kind, "content": _clip(path.read_text(errors="replace"), FULL_CHARS)}
    if kind == "docx":
        return {"kind": kind, "content": _docx_text(path, FULL_CHARS)}
    raise ValueError(
        f"read_document handles text, markdown, Word, CSV and Excel, not {path.suffix!r}. "
        "Spatial files go through load_infrastructure."
    )
