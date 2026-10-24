"""Report writers: a themed Word template, Excel workbooks, CSV.

python-docx / openpyxl only; figures are embedded from PNG artifacts already in
the conversation's workdir.

The Word output follows the house style set out in THEME below: a deep-navy and
bright-blue institutional palette, Arial throughout. It carries **no logo, no
organisation name and no other identifying mark** — the styling is the house
look, and attribution is left to whoever issues the document.
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


# --------------------------------------------------------------------------- #
# House style
#
# Two institutional blues carry the whole document: a deep navy for type and
# the rules that frame content, a bright blue for the accents that separate it.
# Arial is the body face because it is the system font every Word installation
# already has — a themed document that silently falls back to Calibri on the
# recipient's machine is not themed at all.
#
# Nothing here identifies an organisation. Keep it that way: the palette is the
# only branding, and no name, logo or mark belongs in generated output.
# --------------------------------------------------------------------------- #
THEME = {
    "font": "Arial",
    "navy": "002244",      # headings, table header fill, framing rules
    "bright": "009FDA",    # accent rules, subheadings
    "band": "EAF3F9",      # alternating table rows, a light tint of `bright`
    "hairline": "BFD9E8",  # rules between table rows
    "muted": "5A5A5A",     # captions, notes, metadata
    "body": "262626",      # body text — softer than pure black on paper
}

_HEADER_FILL = THEME["navy"]


def _rgb(hex_colour: str) -> Any:
    from docx.shared import RGBColor

    return RGBColor.from_string(hex_colour)


def _shade(cell: Any, hex_fill: str) -> None:
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), hex_fill)
    cell._tc.get_or_add_tcPr().append(shd)


def _rule(paragraph: Any, hex_colour: str, points: float = 1.5) -> None:
    """A coloured rule under a paragraph — the accent beneath the title."""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    borders = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), str(int(points * 8)))  # eighths of a point
    bottom.set(qn("w:space"), "4")
    bottom.set(qn("w:color"), hex_colour)
    borders.append(bottom)
    paragraph._p.get_or_add_pPr().append(borders)


def _borders(table: Any) -> None:
    """Navy rules above and below the table, hairlines between rows.

    No outer box and no vertical lines: the columns are held apart by spacing,
    which reads cleaner in print than a full grid.
    """
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    borders = OxmlElement("w:tblBorders")
    spec = {
        "top": ("single", 12, THEME["navy"]),       # 1.5 pt
        "bottom": ("single", 12, THEME["navy"]),
        "insideH": ("single", 4, THEME["hairline"]),  # 0.5 pt
        "left": ("none", 0, "auto"),
        "right": ("none", 0, "auto"),
        "insideV": ("none", 0, "auto"),
    }
    for edge, (val, size, colour) in spec.items():
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), val)
        if val != "none":
            el.set(qn("w:sz"), str(size))
            el.set(qn("w:color"), colour)
        borders.append(el)
    table._tbl.tblPr.append(borders)


def _set_font(style: Any, font_name: str) -> None:
    """Pin a style to a real font face.

    `style.font.name` alone is not enough: the stock python-docx template gives
    its heading styles `w:asciiTheme="majorHAnsi"`, and a theme reference beats
    an explicit face — so the headings would keep rendering in Calibri Light
    while the body turned Arial. Drop the theme attributes, then set all four
    explicit ones (eastAsia included, or Word substitutes on non-Latin runs).
    """
    from docx.oxml.ns import qn

    rfonts = style.element.get_or_add_rPr().get_or_add_rFonts()
    for attr in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"):
        rfonts.attrib.pop(qn(f"w:{attr}"), None)
    for attr in ("ascii", "hAnsi", "eastAsia", "cs"):
        rfonts.set(qn(f"w:{attr}"), font_name)


def _apply_theme(doc: Any) -> None:
    """Restyle the built-in styles so every paragraph picks the theme up."""
    from docx.oxml.ns import qn
    from docx.shared import Pt

    normal = doc.styles["Normal"]
    _set_font(normal, THEME["font"])
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = _rgb(THEME["body"])
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.15

    for name, size, colour in (
        ("Title", 24, THEME["navy"]),
        ("Heading 1", 15, THEME["navy"]),
        ("Heading 2", 12, THEME["navy"]),
        ("Heading 3", 11, THEME["bright"]),
    ):
        try:
            style = doc.styles[name]
        except KeyError:  # a trimmed-down template may not define them all
            continue
        _set_font(style, THEME["font"])
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = _rgb(colour)
        # The stock Title and Heading styles carry their own borders; drop them
        # so the only rule in the document is the one we draw deliberately.
        pbdr = style.element.find(qn("w:pPr") + "/" + qn("w:pBdr"))
        if pbdr is not None:
            pbdr.getparent().remove(pbdr)

    for name in ("List Bullet", "List Number", "List Paragraph", "Caption"):
        try:
            _set_font(doc.styles[name], THEME["font"])
        except KeyError:
            continue


def _caption(doc: Any, label: str, text: str | None, above: bool = True) -> None:
    from docx.shared import Pt

    p = doc.add_paragraph()
    run = p.add_run(f"{label}. " if text else label)
    run.bold = True
    if text:
        _add_runs(p, str(text))
    for r in p.runs:
        r.font.size = Pt(9.5)
        r.font.color.rgb = _rgb(THEME["navy"] if r.bold else THEME["muted"])
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
        r.font.color.rgb = _rgb(THEME["muted"])
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
            r.font.name = THEME["font"]
            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    for i, row in enumerate(rows, start=1):
        for j in range(len(cols)):
            val = row[j] if j < len(row) else ""
            cell = t.cell(i, j)
            cell.text = "" if val is None else str(val)
            # Band every other row: at 4+ columns the eye loses the line.
            if i % 2 == 0:
                _shade(cell, THEME["band"])
            for r in cell.paragraphs[0].runs:
                r.font.size = Pt(10)
                r.font.name = THEME["font"]
                if j == 0:
                    r.bold = True
                    r.font.color.rgb = _rgb(THEME["navy"])
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
    _apply_theme(doc)

    heading = doc.add_heading(title, level=0)
    heading.paragraph_format.space_after = Pt(4)
    _rule(heading, THEME["bright"], points=2.0)
    if subtitle:
        p = doc.add_paragraph(subtitle)
        p.runs[0].italic = True
        p.runs[0].font.size = Pt(11.5)
        p.runs[0].font.color.rgb = _rgb(THEME["navy"])
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(2)
    meta = doc.add_paragraph(
        f"Infrastructure Risk Analyzer · generated {dt.date.today().isoformat()}"
    )
    meta.runs[0].font.size = Pt(8.5)
    meta.runs[0].font.color.rgb = _rgb(THEME["muted"])

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
