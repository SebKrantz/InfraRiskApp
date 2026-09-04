"""Report writers: a clean generic Word template, Excel workbooks, CSV.

python-docx / openpyxl only; figures are embedded from PNG artifacts already in
the conversation's workdir. Deliberately unbranded — the output is meant to be
dropped into whatever template the user's institution requires.
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Any

_MD_BOLD = re.compile(r"\*\*(.+?)\*\*")
_MD_ITALIC = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")


def _add_runs(paragraph: Any, text: str) -> None:
    """Bold/italic markdown subset -> runs."""
    pos = 0
    for m in _MD_BOLD.finditer(text):
        _italic_runs(paragraph, text[pos:m.start()])
        paragraph.add_run(m.group(1)).bold = True
        pos = m.end()
    _italic_runs(paragraph, text[pos:])


def _italic_runs(paragraph: Any, text: str) -> None:
    pos = 0
    for m in _MD_ITALIC.finditer(text):
        if text[pos:m.start()]:
            paragraph.add_run(text[pos:m.start()])
        paragraph.add_run(m.group(1)).italic = True
        pos = m.end()
    if text[pos:]:
        paragraph.add_run(text[pos:])


# Table accents — a single blue family, close to Word's own table styles.
_HEADER_FILL = "1F4E79"
_BORDER_COLOR = "1F4E79"


def _shade(cell: Any, hex_fill: str) -> None:
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), hex_fill)
    cell._tc.get_or_add_tcPr().append(shd)


def _borders(table: Any) -> None:
    """Box + horizontal rules in the accent colour, no vertical lines."""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "bottom", "left", "right", "insideH"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")  # 0.5 pt
        el.set(qn("w:color"), _BORDER_COLOR)
        borders.append(el)
    none = OxmlElement("w:insideV")
    none.set(qn("w:val"), "none")
    borders.append(none)
    table._tbl.tblPr.append(borders)


def _caption(doc: Any, label: str, text: str | None, above: bool = True) -> None:
    from docx.shared import Pt

    p = doc.add_paragraph()
    run = p.add_run(f"{label}. " if text else label)
    run.bold = True
    if text:
        _add_runs(p, str(text))
    for r in p.runs:
        r.font.size = Pt(9.5)
    p.paragraph_format.space_after = Pt(3 if above else 10)
    p.paragraph_format.space_before = Pt(0 if above else 2)


def _note(doc: Any, text: str) -> None:
    from docx.shared import Pt

    # The label is ours; models routinely write their own "Note:" in as well.
    body = re.sub(r"^\s*note\s*:\s*", "", str(text), flags=re.IGNORECASE)
    p = doc.add_paragraph()
    lead = p.add_run("Note: ")
    lead.italic = True
    _add_runs(p, body)
    for r in p.runs:
        r.italic = True
        r.font.size = Pt(8.5)
    p.paragraph_format.space_before = Pt(2)


def _col_widths(cols: list[str], rows: list[list[Any]]) -> list[Any]:
    """Column widths proportional to content, 16.5 cm total.

    Equal splits strangle a long name column into one word per line (and
    autofit is ignored by some renderers), so measure the longest entry per
    column, damp the extremes, and hand text-heavy columns the width they need.
    Word wraps anything that still overflows.
    """
    from docx.shared import Cm

    longest = []
    for j, c in enumerate(cols):
        entries = [c] + [str(r[j]) for r in rows if j < len(r) and r[j] is not None]
        longest.append(max(len(e) for e in entries))
    weights = [max(w, 6) ** 0.7 for w in longest]
    total = sum(weights)
    return [Cm(max(1.7, 16.5 * w / total)) for w in weights]


def _add_table(
    doc: Any, table: dict[str, Any], number: int, labels: dict[str, str] | None = None
) -> None:
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.shared import Pt, RGBColor

    labels = labels or {}
    cols = [str(c) for c in table["columns"]]
    rows = table.get("rows") or []
    _caption(doc, f"Table {number}", _resolve(table.get("title") or "", labels), above=True)

    t = doc.add_table(rows=1 + len(rows), cols=len(cols))
    t.alignment = WD_TABLE_ALIGNMENT.LEFT
    widths = _col_widths(cols, rows)
    for j, col in enumerate(t.columns):
        col.width = widths[j]
        for cell in col.cells:
            cell.width = widths[j]
    _borders(t)
    for j, c in enumerate(cols):
        cell = t.cell(0, j)
        cell.text = c
        _shade(cell, _HEADER_FILL)
        for r in cell.paragraphs[0].runs:
            r.bold = True
            r.font.size = Pt(10)
            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    for i, row in enumerate(rows, start=1):
        for j in range(len(cols)):
            val = row[j] if j < len(row) else ""
            cell = t.cell(i, j)
            cell.text = "" if val is None else str(val)
            for r in cell.paragraphs[0].runs:
                r.font.size = Pt(10)
                if j == 0:
                    r.bold = True
    if table.get("note"):
        _note(doc, _resolve(table["note"], labels))
    else:
        doc.add_paragraph().paragraph_format.space_after = Pt(0)


_REF = re.compile(r"\[\[([A-Za-z0-9_.:-]+)\]\]")


def _tables_of(sec: dict[str, Any]) -> list[dict[str, Any]]:
    tables = list(sec.get("tables") or [])
    if sec.get("table"):
        tables.insert(0, sec["table"])
    return [t for t in tables if t.get("columns")]


def _number_exhibits(
    sections: list[dict[str, Any]], figure_paths: dict[str, Path]
) -> dict[str, str]:
    """Walk the document in render order and resolve every `ref` label to its
    final "Table N" / "Figure N" string, so prose can cite by name instead of
    counting. Must mirror build_docx's traversal exactly."""
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


def _resolve(text: str, labels: dict[str, str]) -> str:
    """Replace [[ref]] with its number. An unknown ref keeps its brackets so it
    is visible in the output rather than silently wrong."""
    return _REF.sub(lambda m: labels.get(m.group(1), m.group(0)), str(text))


def build_docx(
    dest: Path,
    title: str,
    subtitle: str | None,
    sections: list[dict[str, Any]],
    figure_paths: dict[str, Path],
) -> tuple[Path, dict[str, str]]:
    """sections = [{heading, paragraphs: [str], bullets: [str],
    table | tables: {columns: [...], rows: [[...]], title?, note?, ref?},
    figures: [{artifact_id, caption?, note?, ref?}]}].
    Paragraph text supports **bold** / *italic*; a line starting with '- '
    becomes a bullet and '## ' a subheading. Tables and figures are numbered in
    document order ("Table 1", "Figure 1", ...); `[[ref]]` in any text resolves
    to the number of the exhibit carrying that `ref`. Returns the path and the
    resolved label map."""
    import docx
    from docx.shared import Cm, Pt

    labels = _number_exhibits(sections, figure_paths)

    doc = docx.Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10.5)

    doc.add_heading(title, level=0)
    if subtitle:
        p = doc.add_paragraph(subtitle)
        p.runs[0].italic = True
    meta = doc.add_paragraph(
        f"Infrastructure Risk Analyzer · generated {dt.date.today().isoformat()}"
    )
    meta.runs[0].font.size = Pt(8.5)

    n_table = 0
    n_figure = 0
    for sec in sections:
        if sec.get("heading"):
            doc.add_heading(str(sec["heading"]), level=1)
        for para in sec.get("paragraphs") or []:
            for line in _resolve(para, labels).splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("- "):
                    p = doc.add_paragraph(style="List Bullet")
                    _add_runs(p, line[2:])
                elif line.startswith("## "):
                    doc.add_heading(line[3:], level=2)
                else:
                    p = doc.add_paragraph()
                    _add_runs(p, line)
        for item in sec.get("bullets") or []:
            p = doc.add_paragraph(style="List Bullet")
            _add_runs(p, _resolve(item, labels))
        for table in _tables_of(sec):
            n_table += 1
            _add_table(doc, table, n_table, labels)
        for figref in sec.get("figures") or []:
            path = figure_paths.get(str(figref.get("artifact_id", "")))
            if path is None or not path.is_file():
                continue
            n_figure += 1
            doc.add_picture(str(path), width=Cm(16))
            _caption(doc, f"Figure {n_figure}",
                     _resolve(figref.get("caption") or "", labels), above=False)
            if figref.get("note"):
                _note(doc, _resolve(figref["note"], labels))

    doc.save(dest)
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
