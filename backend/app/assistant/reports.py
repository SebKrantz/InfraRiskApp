"""Report writers: Word notes and reports in the AEI Labs house style, Excel workbooks.

`build_docx` keeps the app's section schema (what `write_report_docx` takes: headings,
paragraphs, bullets, tables, figures by artifact id, `ref` labels cited as `[[ref]]`) and
renders it through the AEI Labs builder vendored in `report_theme/` — the renderer AGUI uses
for its reports — by writing the sections as that builder's Markdown dialect (a `Table:`
caption over a pipe table, a `Figure:` caption over an image, a `Source:` or `Note:` line
under either). A `note` (2-10 pages: title block, numbered sections) is the default;
`report` adds the cover, contents and chapter openers.
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Any

# A caption's own "Figure 3." / "Table 2:" prefix: the builder numbers them.
_NUMBERED = re.compile(r"^\s*(?:\*\*)?\s*(?:Figure|Fig\.|Map|Table)\s+[A-Z]?\d+(?:\.\d+)?\s*[.:\-—–]?\s*(?:\*\*)?\s*",
                       re.I)
# A manual chapter number ("Chapter 2 —", "Section 3:"): the builder numbers headings itself.
_MANUAL_NUM = re.compile(r"^(?:Chapter|Section|Part)\s+(?:\d+|[IVX]+)\s*[—–:.\-]?\s*", re.I)
# A heading that STARTS with a quantity ("10 km buffer"): the builder would strip the number
# as manual section numbering, so a zero-width space shields it.
_QTY_HEAD = re.compile(r"^\d+(?:[.,]\d+)?\s+[a-z%°]")


def _cell(v: Any) -> str:
    s = "" if v is None else str(v)
    return s.replace("\n", " ").replace("|", "\\|").strip()


def _caption(v: Any) -> str:
    return _NUMBERED.sub("", _cell(v), count=1)


def _heading(text: str) -> str:
    text = _MANUAL_NUM.sub("", text, count=1).strip() or text
    return "​" + text if _QTY_HEAD.match(text) else text


def _note_line(text: Any) -> str:
    """A table's or figure's note as the builder's `Source:` line when it names a source,
    else as a `Note:` (a definition or a caveat is not a source)."""
    t = str(text).strip()
    m = re.match(r"^(source|sources|note|notes)\s*:\s*", t, re.I)
    if m:
        return f"{'Source' if m.group(1).lower().startswith('source') else 'Note'}: {t[m.end():]}"
    return f"Note: {t}"


def _paragraph_md(text: str) -> list[str]:
    """A paragraph's lines; its '## ' subheadings become the builder's level-2 headings."""
    out = []
    for line in str(text).splitlines():
        s = line.rstrip()
        if s.lstrip().startswith("## "):
            out += ["", "### " + s.lstrip()[3:], ""]
        else:
            out.append(s)
    return out + [""]


def _table_md(t: dict[str, Any]) -> list[str]:
    cols = [str(c) for c in t.get("columns") or []]
    if not cols:
        return []
    out = [f"Table: {_caption(t.get('title') or '')}".rstrip(), "",
           "| " + " | ".join(_cell(c) for c in cols) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    for r in t.get("rows") or []:
        out.append("| " + " | ".join(_cell(r[j] if j < len(r) else "") for j in range(len(cols))) + " |")
    out.append("")
    if t.get("note"):
        out += [_note_line(t["note"]), ""]
    return out


_REF = re.compile(r"\[\[([A-Za-z0-9_.:-]+)\]\]")


def _tables_of(sec: dict[str, Any]) -> list[dict[str, Any]]:
    tables = list(sec.get("tables") or [])
    if sec.get("table"):
        tables.insert(0, sec["table"])
    return [t for t in tables if t.get("columns")]


def _number_exhibits(sections: list[dict[str, Any]], figure_paths: dict[str, Path]) -> dict[str, str]:
    """Walk the document in render order and resolve every `ref` label to its final
    "Table N" / "Figure N" string, so prose can cite by name instead of counting. Must mirror
    `sections_to_markdown`'s traversal exactly."""
    labels: dict[str, str] = {}
    n_table = n_figure = 0
    for sec in sections:
        for table in _tables_of(sec):
            n_table += 1
            if table.get("ref"):
                labels[str(table["ref"])] = f"Table {n_table}"
        for figref in sec.get("figures") or []:
            path = figure_paths.get(str(figref.get("artifact_id", "")))
            if path is None or not path.is_file():
                continue  # skipped in the render too
            n_figure += 1
            if figref.get("ref"):
                labels[str(figref["ref"])] = f"Figure {n_figure}"
    return labels


def sections_to_markdown(sections: list[dict[str, Any]], figure_paths: dict[str, Path]) -> str:
    """The sections as the AEI builder's Markdown (a figure whose artifact is gone is skipped)."""
    md: list[str] = []
    for sec in sections:
        head = str(sec.get("heading") or "").strip()
        if head:
            md += [f"# {_heading(head)}", ""]
        for para in sec.get("paragraphs") or []:
            md += _paragraph_md(para)
        for b in sec.get("bullets") or []:
            md.append(f"- {b}")
        if sec.get("bullets"):
            md.append("")
        for t in _tables_of(sec):
            md += _table_md(t)
        for f in sec.get("figures") or []:
            fid = str(f.get("artifact_id", ""))
            path = figure_paths.get(fid)
            if path is None or not Path(path).is_file():
                continue
            # an explicit `Figure:` caption right above the image: the builder would take a
            # paragraph that starts "Figures 1 and 2 show…" above an image as its caption
            md += ["", f"Figure: {_caption(f.get('caption'))}",
                   f"![]({Path(path).resolve()}){{width=6in}}", ""]
            if f.get("note"):
                md += [_note_line(f["note"]), ""]
    return "\n".join(md)


def build_docx(dest: Path, title: str, subtitle: str | None, sections: list[dict[str, Any]],
               figure_paths: dict[str, Path], kind: str = "note") -> tuple[Path, dict[str, str]]:
    """sections = [{heading, paragraphs: [str], bullets: [str],
    table | tables: {columns: [...], rows: [[...]], title?, note?, ref?},
    figures: [{artifact_id, caption?, note?, ref?}]}], written as an AEI Labs Word document at
    `dest`: kind 'note' (default) or 'report'. Paragraph text supports **bold** / *italic*,
    '- ' bullets and '## ' subheadings. Tables and figures are numbered in document order
    ("Table 1", "Figure 1", ...); `[[ref]]` in any text resolves to the number of the exhibit
    carrying that `ref` (an unknown ref keeps its brackets, visible rather than silently wrong).
    Returns the path and the resolved label map."""
    from .report_theme import aeilabs_docx

    labels = _number_exhibits(sections, figure_paths)
    text = _REF.sub(lambda m: labels.get(m.group(1), m.group(0)),
                    sections_to_markdown(sections, figure_paths))
    meta = {"type": kind, "title": title, "subtitle": subtitle,
            "doc_label": f"Infrastructure risk {kind}", "date": dt.date.today().isoformat()}
    aeilabs_docx.build_document(text, dest, meta=meta, base_dir=str(Path(dest).parent),
                                paginate=False)
    return dest, labels


def build_xlsx(dest: Path, sheets: list[dict[str, Any]]) -> Path:
    """sheets = [{name, columns: [...], rows: [[...]]}] -> one workbook."""
    import pandas as pd

    with pd.ExcelWriter(dest, engine="openpyxl") as writer:
        for i, sheet in enumerate(sheets):
            name = str(sheet.get("name") or f"Sheet{i + 1}")[:31]
            df = pd.DataFrame(sheet.get("rows") or [], columns=sheet.get("columns"))
            df.to_excel(writer, sheet_name=name, index=False, freeze_panes=(1, 0))
            ws = writer.sheets[name]
            for j, col in enumerate(df.columns, start=1):
                widths = [len(str(col))] + [len(str(v)) for v in df.iloc[:200, j - 1]]
                ws.column_dimensions[ws.cell(row=1, column=j).column_letter].width = min(
                    max(widths) + 2, 44
                )
    return dest
