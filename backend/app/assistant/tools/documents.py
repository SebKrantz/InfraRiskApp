"""read_document: full content of an uploaded context document."""

from __future__ import annotations

from typing import Any

from .. import documents
from ..conversations import Conversation
from . import tool


@tool(
    "read_document",
    "Full content of an uploaded context document. Markdown, text and Word "
    "return their text; CSV and Excel are loaded into the python namespace as "
    "DataFrames (one per sheet) and summarised. A short preview always arrives "
    "with the upload — call this when you need the rest before analysing or "
    "writing. Terms of reference, inception notes and project documents should "
    "always be read before a report so your wording, place names and framing "
    "match the user's own.",
    {
        "type": "object",
        "properties": {
            "file": {"type": "string", "description": "Uploaded filename (see list_files)."},
            "sheet": {
                "type": "string",
                "description": "Excel sheet name; default = all sheets.",
            },
        },
        "required": ["file"],
    },
)
def read_document(conv: Conversation, file: str, sheet: str | None = None) -> dict[str, Any]:
    path = conv.uploads.get(file)
    if path is None:
        matches = [p for n, p in conv.uploads.items() if n.startswith(file)]
        if len(matches) == 1:
            path = matches[0]
    if path is None or not path.exists():
        raise ValueError(f"no uploaded file {file!r}; uploads: {sorted(conv.uploads)}")

    kind = documents.kind_of(path)
    if kind == "tabular":
        import pandas as pd

        out: dict[str, Any] = {"kind": "tabular", "tables": []}
        if path.suffix.lower() == ".csv":
            frames = {"data": pd.read_csv(path, sep=None, engine="python")}
        else:
            xl = pd.ExcelFile(path)
            names = [sheet] if sheet else xl.sheet_names
            frames = {n: xl.parse(n) for n in names}
        for name, df in frames.items():
            var = conv.store_result(f"doc_{path.stem[:20]}", df)
            out["tables"].append(
                {
                    "sheet": name,
                    "stored_as": var,
                    "shape": list(df.shape),
                    "columns": [str(c) for c in df.columns][:40],
                    "head": df.head(8).to_string(max_cols=12),
                }
            )
        return out
    return documents.full_content(path)
