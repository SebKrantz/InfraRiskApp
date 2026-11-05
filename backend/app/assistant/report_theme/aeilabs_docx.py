"""
aeilabs_docx.py — Markdown -> Word (.docx) in the AEI Labs house style: reports and notes.

Library entry point (AGUI vendors this file, md_parser.py, render_docx.py and assets/ as a
package, e.g. backend/agui/report_theme/):

    from aeilabs_docx import build_document
    info = build_document("report.md", "out/Report.docx")                 # front matter decides type
    info = build_document(markdown_text, "out/Note.docx", meta={"type": "note", "title": "..."})
    # -> {"warnings": [...], "figures": 4, "tables": 6, "pages": 17, "paginated": True, "seconds": 9.8}

Design
- Every brand value (palette, fonts, sizes, strings, logos, page geometry) lives in one Theme
  object (AEI_LABS, WORLD_BANK below). The renderer never hard-codes a brand value; a house style
  is a Theme instance in THEMES, picked with the front-matter key `style:` (aeilabs | worldbank).
  AEI Labs is the default; an unknown style falls back to it with a warning.
- Inter everywhere, Arial declared as its fallback in the font table (w:altName); theme-font
  attributes are removed from styles so the explicit font always wins.
- Word's built-in styles (Normal, Title, Heading 1-4, Caption, toc 1/2 ...) are restyled; the
  builder's own paragraph styles carry the theme's prefix (Theme.style_prefix: "AEI Kicker" /
  AEIKicker, "WB Kicker" / WBKicker), so a document names only its own house style.
- Report: cover (own section, no running header) -> contents (real Word TOC field, cached entries
  pre-filled) + unnumbered front matter on roman pages -> numbered chapters from page 1 -> annexes
  (A, B ...). Chapters open on a new page with a kicker line ("CHAPTER 2" or a custom kicker) above
  the Heading 1 title. Note: title block, numbered sections, no cover.
- Captions above figures and tables, "Source:"/"Note:" lines below; boxes, KPI rows and
  `::: pipeline` flowcharts are tables (they render the same in Word and LibreOffice and never
  split); real Word footnotes.
- Contents page numbers: two-pass (LibreOffice -> pdftotext -> cached TOC entries) when soffice
  is available; Word refreshes all fields on open either way (updateFields is set).

Provenance: the brand-agnostic OOXML helpers (_insert_ordered, _el, set_* ...), the Footnotes
part writer, the cross-reference resolver, the list Numbering class, the glyph-metric column-width
algorithm, the landscape-section logic and the page-number fill are adapted from
~/.claude/skills/cpcs-report/scripts/cpcs_docx.py and cpcs_report.py (versions of 2026-09-08).
They now take colours and fonts from the Theme instead of module constants.
"""
from __future__ import annotations

import copy
import datetime as _dt
import io
import math
import os
import re
import tempfile
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.opc.packuri import PackURI
from docx.opc.part import Part
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Emu, Inches, Pt, RGBColor
from docx.text.paragraph import Paragraph
from docx.text.run import Run
from lxml import etree

try:  # package layout (AGUI: agui/report_theme/)
    from . import md_parser
    from . import render_docx
except ImportError:  # skill layout: scripts/ on sys.path or run as a script
    import md_parser
    import render_docx

Inline = md_parser.Inline
inline_text = md_parser.inline_text
parse_inline = md_parser.parse_inline

_HERE = Path(__file__).resolve().parent
ASSETS = _HERE / "assets" if (_HERE / "assets").is_dir() else _HERE.parent / "assets"


# =========================================================================== theme
@dataclass
class Theme:
    """All brand values in one place. Colours are hex strings without '#'. Logo, mark, footer and
    prepared-by fields are optional: a theme without them gets a typographic cover and a text-only
    running header/footer."""
    key: str = "aeilabs"
    label: str = "AEI Labs"                        # name shown in style pickers
    description: str = "Inter, navy and blue, logo"   # one short line (AGUI's style picker)
    style_prefix: str = "AEI"                      # names the builder's own Word styles: "AEI Kicker", id AEIKicker
    org: str = "AEI Labs"                          # document author property
    org_long: str = "Advanced Economic Intelligence"
    site: Optional[str] = "aeilabs.xyz"
    # typography
    font: str = "Inter"
    font_alt: Optional[str] = "Arial"              # font-table fallback (w:altName); None for a system font
    font_panose: str = "02000503000000020004"      # Inter OS/2 PANOSE
    font_sig: tuple = ("E00002FF", "1200A1FF", "00000000", "00000000", "0000019F", "00000000")
    code_font: str = "Courier New"
    # palette ("Deck-Light" system of deck V4 and the one-pagers)
    navy: str = "0B1B3A"        # headings, titles, strong text
    slate: str = "3D4A63"       # body text
    blue: str = "3D78D6"        # brand blue (logo): rules, bars, stat numbers, bullets
    blue_dk: str = "2E62C2"     # blue for small text: kickers, figure labels, H3, links
    kpi_value: Optional[str] = None   # KPI card values on `tint` (None: `blue`); for a blue under 3:1 on tint
    muted: str = "66738C"       # captions, sources, header/footer, table header text
    card: str = "F4F6FA"        # card fill (boxes, notes, code)
    tint: str = "EAF1FB"        # highlighted fill (takeaways, KPI cards, total rows)
    hairline: str = "E4E9F1"    # table row rules, header rule
    rule: str = "B9C4D6"        # stronger rule under table headers / at table ends
    frame: str = "D5DCE8"       # hairline frame around maps / screenshots
    cover_mark: str = "EAF1FB"  # the large pale brand mark on covers without an image (= tint)
    sky: str = "8FB2EA"         # light blue accent (note boxes)
    pale: str = "C9DAF5"
    white: str = "FFFFFF"
    warn: str = "D4952B"        # caution bar (the amber accent of the AEI dark deck/website)
    warn_fill: str = "FCF5E8"
    warn_text: str = "8A5A12"
    # logos in assets/ (None: no logo on the cover / in the header; no watermark mark on the cover)
    logo: Optional[str] = "logo_lockup_dark.png"   # blue mark + navy wordmark, for white backgrounds
    logo_on_dark: Optional[str] = "logo_lockup_light.png"
    mark: Optional[str] = "logo_mark_blue.png"
    # cover without a logo: a full-bleed colour panel behind the title block (None: white cover)
    cover_band: Optional[str] = None
    cover_band_accent: Optional[str] = None        # thin strip along the panel's lower edge
    # table header row: fill + text colour (None: no fill, bold muted text)
    table_header_fill: Optional[str] = None
    table_header_text: Optional[str] = None
    # fixed strings (US English)
    footer_text: Optional[str] = "AEI Labs · aeilabs.xyz"
    prepared_by: Optional[str] = "Prepared by AEI Labs · aeilabs.xyz"
    prepared_for: str = "Prepared for"
    chapter_word: str = "Chapter"
    annex_word: str = "Annex"
    section_word: str = "Section"
    figure_word: str = "Figure"
    table_word: str = "Table"
    contents_title: str = "Contents"
    lof_title: str = "Figures"
    lot_title: str = "Tables"
    takeaway_title: str = "Key takeaways"
    note_title: str = "Note"
    warning_title: str = "Caution"
    acronyms_title: str = "Abbreviations"
    summary_title: str = "Executive summary"
    summary_names: tuple = ("executive summary", "summary", "key messages", "key findings", "in brief",
                            "at a glance", "overview", "foreword", "preface", "abbreviations", "acronyms")
    # numbering
    figure_numbering: str = "sequential"       # or "chapter" (Figure 2.1)
    label_sep: str = "."
    # page geometry (inches)
    margin_lr: float = 1.0
    margin_top: float = 1.0
    margin_bottom: float = 0.9
    header_dist: float = 0.45
    footer_dist: float = 0.45
    # type sizes (pt)
    body_pt: float = 10.0
    body_line: int = 290          # line spacing in 240ths (290 = 1.21)
    body_after: float = 7.0
    cover_kicker_pt: float = 10.0
    cover_title_pt: float = 30.0
    cover_sub_pt: float = 17.0
    cover_lead_pt: float = 11.5
    note_title_pt: float = 24.0
    note_sub_pt: float = 13.5
    lead_pt: float = 11.0
    kicker_pt: float = 8.5
    kicker_spacing: int = 36      # letter spacing in 1/20 pt
    h1_pt: float = 24.0           # report chapter title
    h1_note_pt: float = 14.5      # note section title
    h2_pt: float = 13.5
    h3_pt: float = 11.0
    h4_pt: float = 10.0
    caption_pt: float = 9.5
    source_pt: float = 8.0
    table_pt: float = 8.5
    table_header_pt: float = 7.5
    table_header_spacing: int = 4 # letter spacing of table header text, 1/20 pt
    table_min_pt: float = 7.0
    box_pt: float = 9.5
    box_title_pt: float = 10.5
    stat_pt: float = 22.0
    stat_label_pt: float = 8.5
    footnote_pt: float = 8.0
    header_pt: float = 8.0
    code_pt: float = 8.5
    toc1_pt: float = 10.5
    toc2_pt: float = 9.5


AEI_LABS = Theme()

# World Bank publication look (sources and choices: references/themes.md). Unbranded: no World Bank
# name, logo or letterhead anywhere (AEI Labs is not the World Bank), and no AEI Labs logo either;
# the only attribution is the text line "Prepared by Advanced Economic Intelligence Labs".
WORLD_BANK = Theme(
    key="worldbank",
    label="World Bank",
    description="Arial, WB blues, no logo",
    style_prefix="WB",
    org="Advanced Economic Intelligence Labs",
    org_long="Advanced Economic Intelligence Labs",
    site=None,
    font="Arial",                  # WBG secondary (Office) typeface; Andes is proprietary, Open Sans not installed
    font_alt=None,
    navy="002244",                 # WBG solid blue (PMS 289)
    slate="111111",                # WB data-viz guide: text
    blue="009FDA",                 # WBG bright blue (process cyan)
    blue_dk="0071BC",              # WB data-viz guide: selection1 (small blue text)
    kpi_value="0071BC",            # selection1 for KPI values: 4.6:1 on tint (bright blue: 2.7:1)
    muted="666666",                # WB data-viz guide: textSubtle
    card="EBEEF4",                 # grey100
    tint="E3F6FD",                 # seqB1 (lightest monochrome blue)
    hairline="CED4DE",             # grey200 (grid lines)
    rule="8A969F",                 # grey300 (reference / zero lines)
    frame="CED4DE",
    cover_mark="E3F6FD",
    sky="75CCEC",                  # seqB2
    pale="E3F6FD",
    warn="FF9800",                 # cat2
    warn_fill="FFF3E0",
    warn_text="A4560B",            # cat2Text B65F0C 10 % darker: 4.9:1 on warn_fill (cat2Text: 4.1:1)
    logo=None, logo_on_dark=None, mark=None,
    cover_band="002244", cover_band_accent="009FDA",
    table_header_fill="002244", table_header_text="FFFFFF",
    footer_text="Advanced Economic Intelligence Labs",
    prepared_by="Prepared by Advanced Economic Intelligence Labs",
    takeaway_title="Key messages",
    figure_numbering="chapter",
)

THEMES: dict[str, Theme] = {
    "aeilabs": AEI_LABS, "aei": AEI_LABS, "aei labs": AEI_LABS, "aei-labs": AEI_LABS,
    "worldbank": WORLD_BANK, "world bank": WORLD_BANK, "world-bank": WORLD_BANK, "wb": WORLD_BANK,
}
DEFAULT_STYLE = "aeilabs"


def get_theme(style) -> tuple[Theme, str | None]:
    """(Theme, warning) for a style name or alias; unknown or empty -> AEI Labs (warning if unknown)."""
    key = str(style or DEFAULT_STYLE).strip().lower().replace("_", " ")
    T = THEMES.get(key) or THEMES.get(key.replace(" ", ""))
    if T is None:
        return AEI_LABS, f"unknown style '{key}': using the AEI Labs house style"
    return T, None


def list_styles() -> list[dict]:
    """The distinct house styles (no aliases): id, label, description, swatches (hex with '#'), default."""
    out, seen = [], set()
    for T in THEMES.values():
        if T.key in seen:
            continue
        seen.add(T.key)
        out.append({"id": T.key, "label": T.label, "description": T.description,
                    "swatches": ["#" + c for c in (T.navy, T.blue, T.blue_dk if T.blue_dk != T.blue else T.sky, T.tint)],
                    "default": T.key == DEFAULT_STYLE})
    return out


def asset(T: Theme, name: str) -> str:
    return str(ASSETS / name)


def style_name(T: Theme, base: str) -> str:
    """The Word name of one of the builder's own paragraph styles in this theme: 'AEI Kicker',
    'WB Kicker' (Theme.style_prefix + base). Word's built-in styles keep their names."""
    return f"{T.style_prefix} {base}".strip()


def style_id(T: Theme, base: str) -> str:
    """The style id that goes with style_name(): 'AEIKicker', 'WBKicker' (letters and digits only)."""
    return re.sub(r"[^0-9A-Za-z]", "", T.style_prefix) + base.replace(" ", "")


# =========================================================================== XML helpers (from cpcs_docx)
_PPR_ORDER = [qn("w:" + t) for t in (
    "pStyle keepNext keepLines pageBreakBefore framePr widowControl numPr suppressLineNumbers pBdr shd tabs "
    "suppressAutoHyphens kinsoku wordWrap overflowPunct topLinePunct autoSpaceDE autoSpaceDN bidi adjustRightInd "
    "snapToGrid spacing ind contextualSpacing mirrorIndents suppressOverlap jc textDirection textAlignment "
    "textboxTightWrap outlineLvl divId cnfStyle rPr sectPr pPrChange").split()]
_RPR_ORDER = [qn("w:" + t) for t in (
    "rStyle rFonts b bCs i iCs caps smallCaps strike dstrike outline shadow emboss imprint noProof snapToGrid "
    "vanish webHidden color spacing w kern position sz szCs highlight u effect bdr shd fitText vertAlign rtl cs "
    "em lang eastAsianLayout specVanish oMath").split()]
_TCPR_ORDER = [qn("w:" + t) for t in (
    "cnfStyle tcW gridSpan hMerge vMerge tcBorders shd noWrap tcMar textDirection tcFitText vAlign hideMark").split()]
_TBLPR_ORDER = [qn("w:" + t) for t in (
    "tblStyle tblpPr tblOverlap bidiVisual tblStyleRowBandSize tblStyleColBandSize tblW jc tblCellSpacing tblInd "
    "tblBorders shd tblLayout tblCellMar tblLook").split()]
_TRPR_ORDER = [qn("w:" + t) for t in (
    "cnfStyle divId gridBefore gridAfter wBefore wAfter cantSplit trHeight tblHeader tblCellSpacing jc hidden").split()]
_SECT_ORDER = [qn("w:" + t) for t in (
    "headerReference footerReference footnotePr endnotePr type pgSz pgMar paperSrc pgBorders lnNumType pgNumType "
    "cols formProt vAlign noEndnote titlePg textDirection bidi rtlGutter docGrid printerSettings sectPrChange").split()]


def _insert_ordered(parent, child, order):
    """Insert child into parent respecting the schema order; replaces an existing same-tag child."""
    for ex in parent.findall(child.tag):
        parent.remove(ex)
    idx = order.index(child.tag) if child.tag in order else len(order)
    for ex in list(parent):
        if ex.tag in order and order.index(ex.tag) > idx:
            ex.addprevious(child)
            return child
    parent.append(child)
    return child


def _el(tag: str, **attrs):
    e = OxmlElement(tag)
    for k, v in attrs.items():
        if v is not None:
            e.set(qn("w:" + k), str(v))
    return e


def pPr(p) -> Any:
    return p._p.get_or_add_pPr() if isinstance(p, Paragraph) else p.get_or_add_pPr()


def set_shd(prop, fill: str, order=_PPR_ORDER):
    _insert_ordered(prop, _el("w:shd", val="clear", color="auto", fill=fill), order)


def _border(tag, spec):
    if spec is None:
        return _el(tag, val="nil")
    sz, color = spec[0], spec[1]
    space = spec[2] if len(spec) > 2 else 0
    return _el(tag, val="single", sz=sz, space=space, color=color)


def set_borders(prop, kind: str, order, **sides):
    """kind: 'pBdr' | 'tcBorders' | 'tblBorders'. sides: top=(sz_eighths_pt, color[, space]) or None (nil)."""
    bdr = _el("w:" + kind)
    seq = ("top", "left", "bottom", "right", "insideH", "insideV") if kind != "pBdr" else ("top", "left", "bottom", "right", "between")
    for s in seq:
        if s in sides:
            bdr.append(_border("w:" + s, sides[s]))
    _insert_ordered(prop, bdr, order)


def set_spacing(p, before: int | None = None, after: int | None = None, line: int | None = None,
                rule: str = "auto"):
    """Spacing in twips (1/20 pt); line 240 = single (auto rule) or an exact height in twips."""
    sp = pPr(p).find(qn("w:spacing"))
    if sp is None:
        sp = _insert_ordered(pPr(p), _el("w:spacing"), _PPR_ORDER)
    if before is not None:
        sp.set(qn("w:before"), str(before))
    if after is not None:
        sp.set(qn("w:after"), str(after))
    if line is not None:
        sp.set(qn("w:line"), str(line)); sp.set(qn("w:lineRule"), rule)


def set_indent(p, left: int | None = None, right: int | None = None, hanging: int | None = None,
               first: int | None = None):
    ind = pPr(p).find(qn("w:ind"))
    if ind is None:
        ind = _insert_ordered(pPr(p), _el("w:ind"), _PPR_ORDER)
    if left is not None: ind.set(qn("w:left"), str(left))
    if right is not None: ind.set(qn("w:right"), str(right))
    if hanging is not None: ind.set(qn("w:hanging"), str(hanging))
    if first is not None: ind.set(qn("w:firstLine"), str(first))


def set_keep_next(p, on=True):
    _insert_ordered(pPr(p), _el("w:keepNext", val=None if on else "0"), _PPR_ORDER)


def set_keep_lines(p):
    _insert_ordered(pPr(p), _el("w:keepLines"), _PPR_ORDER)


def set_page_break_before(p):
    _insert_ordered(pPr(p), _el("w:pageBreakBefore"), _PPR_ORDER)


def set_jc(p, val: str):
    _insert_ordered(pPr(p), _el("w:jc", val=val), _PPR_ORDER)


def set_style(p, style_id: str):
    _insert_ordered(pPr(p), _el("w:pStyle", val=style_id), _PPR_ORDER)


def set_num(p, num_id: int, ilvl: int = 0):
    numpr = _el("w:numPr")
    numpr.append(_el("w:ilvl", val=ilvl))
    numpr.append(_el("w:numId", val=num_id))
    _insert_ordered(pPr(p), numpr, _PPR_ORDER)


def add_tabs(p, stops: list[tuple[int, str, str | None]]):
    """stops: [(pos_twips, 'left'|'right'|'center', leader|None)]"""
    tabs = _el("w:tabs")
    for pos, align, leader in stops:
        tabs.append(_el("w:tab", val=align, pos=pos, leader=leader))
    _insert_ordered(pPr(p), tabs, _PPR_ORDER)


def set_run_font(run, name: str, size_pt: float | None = None, color: str | None = None,
                 bold: bool | None = None, italic: bool | None = None, caps: bool | None = None,
                 spacing: int | None = None, hidden: bool = False, position: int | None = None):
    rpr = run._r.get_or_add_rPr()
    _insert_ordered(rpr, _el("w:rFonts", ascii=name, hAnsi=name, cs=name, eastAsia=name), _RPR_ORDER)
    if size_pt is not None:
        _insert_ordered(rpr, _el("w:sz", val=int(round(size_pt * 2))), _RPR_ORDER)
        _insert_ordered(rpr, _el("w:szCs", val=int(round(size_pt * 2))), _RPR_ORDER)
    if color is not None:
        _insert_ordered(rpr, _el("w:color", val=color), _RPR_ORDER)
    if bold is not None:
        _insert_ordered(rpr, _el("w:b", val=None if bold else "0"), _RPR_ORDER)
        _insert_ordered(rpr, _el("w:bCs", val=None if bold else "0"), _RPR_ORDER)
    if italic is not None:
        _insert_ordered(rpr, _el("w:i", val=None if italic else "0"), _RPR_ORDER)
        _insert_ordered(rpr, _el("w:iCs", val=None if italic else "0"), _RPR_ORDER)
    if caps is not None:
        _insert_ordered(rpr, _el("w:caps", val=None if caps else "0"), _RPR_ORDER)
    if spacing is not None:
        _insert_ordered(rpr, _el("w:spacing", val=spacing), _RPR_ORDER)
    if hidden:
        _insert_ordered(rpr, _el("w:vanish"), _RPR_ORDER)
    if position is not None:
        _insert_ordered(rpr, _el("w:position", val=position), _RPR_ORDER)


def add_text_run(p, text: str, T: Theme, size: float | None = None, color: str | None = None,
                 bold: bool | None = None, italic: bool | None = None, **kw):
    r = p.add_run(text)
    set_run_font(r, kw.pop("font", T.font), size, color, bold, italic, **kw)
    return r


def _field_runs(instr: str) -> tuple[list, list]:
    """(begin + instr + separate runs, end run) of a complex field, as bare w:r elements."""
    r1 = OxmlElement("w:r"); r1.append(_el("w:fldChar", fldCharType="begin"))
    r2 = OxmlElement("w:r"); it = _el("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = f" {instr} "; r2.append(it)
    r3 = OxmlElement("w:r"); r3.append(_el("w:fldChar", fldCharType="separate"))
    r4 = OxmlElement("w:r"); r4.append(_el("w:fldChar", fldCharType="end"))
    return [r1, r2, r3], [r4]


def add_field(p, instr: str, cached: str, T: Theme, size: float | None = None, color: str | None = None,
              bold: bool | None = None):
    """Append a complex field (begin / instr / separate / cached result / end). Returns the result run."""
    head, tail = _field_runs(instr)
    for r in head:
        p._p.append(r)
    res = p.add_run(cached)
    set_run_font(res, T.font, size, color, bold)
    p._p.append(tail[0])
    return res


def add_hyperlink(p, url: str, text: str, T: Theme, size=None, bold=None, italic=None):
    rid = p.part.relate_to(url, RT.HYPERLINK, is_external=True)
    h = OxmlElement("w:hyperlink"); h.set(qn("r:id"), rid)
    r = OxmlElement("w:r"); rpr = OxmlElement("w:rPr"); r.append(rpr)
    t = OxmlElement("w:t"); t.set(qn("xml:space"), "preserve"); t.text = text; r.append(t)
    h.append(r); p._p.append(h)
    run = Run(r, p)
    set_run_font(run, T.font, size, T.blue_dk, bold, italic)
    return run


def absorb_trailing_landscape(closing) -> bool:
    """If the element right before `closing` (a paragraph carrying a sectPr, or a body-level sectPr) is
    the landscape section break emitted by a ::: landscape block, merge it into `closing` so the
    document does not end the block with an empty portrait page."""
    prev = closing.getprevious()
    if prev is None or prev.tag != qn("w:p"):
        return False
    ppr = prev.find(qn("w:pPr"))
    sp = ppr.find(qn("w:sectPr")) if ppr is not None else None
    if sp is None:
        return False
    sz = sp.find(qn("w:pgSz"))
    if sz is None or sz.get(qn("w:orient")) != "landscape":
        return False
    target = closing if closing.tag == qn("w:sectPr") else closing.find(qn("w:pPr")).find(qn("w:sectPr"))
    tsz = target.find(qn("w:pgSz"))
    if tsz is not None:
        target.replace(tsz, copy.deepcopy(sz))
    prev.getparent().remove(prev)
    return True


def page_break(container, T: Theme):
    p = container.add_paragraph()
    p.add_run().add_break(WD_BREAK.PAGE)
    set_spacing(p, 0, 0)
    return p


def cell_set(cell, fill: str | None = None, margins: tuple | None = None, borders: dict | None = None,
             width: int | None = None, valign: str | None = None):
    tcpr = cell._tc.get_or_add_tcPr()
    if width is not None:
        _insert_ordered(tcpr, _el("w:tcW", w=width, type="dxa"), _TCPR_ORDER)
    if borders is not None:
        set_borders(tcpr, "tcBorders", _TCPR_ORDER, **borders)
    if fill is not None:
        set_shd(tcpr, fill, _TCPR_ORDER)
    if margins is not None:
        m = _el("w:tcMar")
        for side, v in zip(("top", "left", "bottom", "right"), margins):
            m.append(_el("w:" + side, w=v, type="dxa"))
        _insert_ordered(tcpr, m, _TCPR_ORDER)
    if valign is not None:
        _insert_ordered(tcpr, _el("w:vAlign", val=valign), _TCPR_ORDER)


def table_set(table, width: int, widths: list[int], borders: dict | None = None, align: str = "left",
              margins: tuple = (40, 80, 40, 80), indent: int | None = None):
    tbl = table._tbl
    tblpr = tbl.tblPr
    for ex in tblpr.findall(qn("w:tblStyle")):
        tblpr.remove(ex)
    _insert_ordered(tblpr, _el("w:tblW", w=width, type="dxa"), _TBLPR_ORDER)
    _insert_ordered(tblpr, _el("w:jc", val=align), _TBLPR_ORDER)
    if indent is not None:
        _insert_ordered(tblpr, _el("w:tblInd", w=indent, type="dxa"), _TBLPR_ORDER)
    if borders is not None:
        set_borders(tblpr, "tblBorders", _TBLPR_ORDER, **borders)
    _insert_ordered(tblpr, _el("w:tblLayout", type="fixed"), _TBLPR_ORDER)
    m = _el("w:tblCellMar")
    for side, v in zip(("top", "left", "bottom", "right"), margins):
        m.append(_el("w:" + side, w=v, type="dxa"))
    _insert_ordered(tblpr, m, _TBLPR_ORDER)
    _insert_ordered(tblpr, _el("w:tblLook", val="04A0", firstRow="1", lastRow="0", firstColumn="0",
                               lastColumn="0", noHBand="1", noVBand="1"), _TBLPR_ORDER)
    grid = tbl.tblGrid
    for gc in list(grid):
        grid.remove(gc)
    for w in widths:
        grid.append(_el("w:gridCol", w=w))
    table.autofit = False
    for row in table.rows:
        for i, cell in enumerate(row.cells):
            if i < len(widths):
                cell_set(cell, width=widths[i])


def row_set(row, header: bool = False, cant_split: bool = True):
    trpr = row._tr.get_or_add_trPr()
    if cant_split:
        _insert_ordered(trpr, _el("w:cantSplit"), _TRPR_ORDER)
    if header:
        _insert_ordered(trpr, _el("w:tblHeader"), _TRPR_ORDER)


def clear_cell(cell):
    """Remove the empty paragraph a new cell starts with (once content has been added)."""
    ps = cell._tc.findall(qn("w:p"))
    if len(ps) > 1 and not "".join(t.text or "" for t in ps[0].iter(qn("w:t"))) and ps[0].find(qn("w:r")) is None:
        cell._tc.remove(ps[0])


def para_text(p_el) -> str:
    return "".join(t.text or "" for t in p_el.iter(qn("w:t")))


def _spacer(container, T: Theme, after_twips: int, size: float = 2):
    p = container.add_paragraph()
    set_spacing(p, 0, after_twips, line=240)
    set_run_font(p.add_run(""), T.font, size)
    return p


def fmt_date(value, month_year: bool = False) -> str:
    """ISO date or date -> 'September 8, 2026' (US) or 'September 2026'; other strings pass through."""
    if value is None or value == "":
        d = _dt.date.today()
    elif isinstance(value, (_dt.date, _dt.datetime)):
        d = value if isinstance(value, _dt.date) and not isinstance(value, _dt.datetime) else value.date()
    else:
        s = str(value).strip()
        try:
            d = _dt.date.fromisoformat(s[:10])
        except ValueError:
            return s
    return f"{d.strftime('%B')} {d.year}" if month_year else f"{d.strftime('%B')} {d.day}, {d.year}"


# =========================================================================== package clean-up
_FONT_XML = (
    '<w:font {ns} w:name="{name}"><w:altName w:val="{alt}"/><w:panose1 w:val="{panose}"/><w:charset w:val="00"/>'
    '<w:family w:val="swiss"/><w:pitch w:val="variable"/>'
    '<w:sig w:usb0="{s0}" w:usb1="{s1}" w:usb2="{s2}" w:usb3="{s3}" w:csb0="{c0}" w:csb1="{c1}"/></w:font>'
)
_ARIAL_XML = (
    '<w:font {ns} w:name="Arial"><w:panose1 w:val="020B0604020202020204"/><w:charset w:val="00"/>'
    '<w:family w:val="swiss"/><w:pitch w:val="variable"/>'
    '<w:sig w:usb0="E0002EFF" w:usb1="C000785B" w:usb2="00000009" w:usb3="00000000" w:csb0="000001FF" w:csb1="00000000"/></w:font>'
)


def _prepare_package(doc, T: Theme):
    """Turn python-docx's default template into a clean modern file: Word 2013+ layout (no
    Compatibility Mode), fields updated on open, no stale thumbnail or Word-2010 styles part,
    theme fonts pointing at the brand font and the brand font declared with its fallback."""
    part = doc.part
    # --- settings: compatibility mode 15, update fields on open
    settings = doc.settings.element
    for cs in settings.iter(qn("w:compatSetting")):
        if cs.get(qn("w:name")) == "compatibilityMode":
            cs.set(qn("w:val"), "15")
    upd = settings.find(qn("w:updateFields"))
    if upd is None:
        upd = OxmlElement("w:updateFields")
        anchor = None                         # schema order: ... savePreviewPicture ... updateFields, hdrShapeDefaults,
        for tag in ("w:hdrShapeDefaults", "w:footnotePr", "w:endnotePr", "w:compat", "w:docVars", "w:rsids"):
            anchor = settings.find(qn(tag))    # footnotePr, endnotePr, compat, docVars, rsids ...
            if anchor is not None:
                break
        if anchor is not None:
            anchor.addprevious(upd)
        else:
            settings.append(upd)
    upd.set(qn("w:val"), "true")
    # --- drop the template thumbnail and the Word 2010 stylesWithEffects part
    pkg = part.package
    for rid, rel in list(pkg.rels.items()):
        if rel.reltype.endswith("/thumbnail"):
            pkg.rels.pop(rid)
    for rid, rel in list(part.rels.items()):
        if rel.reltype.endswith("/stylesWithEffects"):
            part.rels.pop(rid)
    # --- the template's own list definitions (Symbol-font bullets) are never used: drop them
    num_el = part.numbering_part.element
    for e in list(num_el):
        if e.tag in (qn("w:abstractNum"), qn("w:num"), qn("w:numPicBullet")):
            num_el.remove(e)
    # --- theme part: major/minor latin font -> brand font
    for rel in part.rels.values():
        if rel.reltype == RT.THEME:
            tp = rel.target_part
            xml = tp.blob.decode("utf-8")
            xml = re.sub(r'(<a:(?:major|minor)Font>\s*<a:latin typeface=")[^"]*(")', rf"\g<1>{T.font}\g<2>", xml)
            tp._blob = xml.encode("utf-8")
    # --- font table: declare the brand font with its fallback
    for rel in part.rels.values():
        if rel.reltype == RT.FONT_TABLE:
            ft = rel.target_part
            root = etree.fromstring(ft.blob)
            ns = nsdecls("w")
            names = {f.get(qn("w:name")) for f in root.findall(qn("w:font"))}
            if T.font not in names and T.font != "Arial":
                s = T.font_sig
                xml = _FONT_XML.format(ns=ns, name=T.font, alt=T.font_alt or "", panose=T.font_panose,
                                       s0=s[0], s1=s[1], s2=s[2], s3=s[3], c0=s[4], c1=s[5])
                if not T.font_alt:
                    xml = xml.replace('<w:altName w:val=""/>', "")
                root.insert(0, etree.fromstring(xml))
            if "Arial" not in names:
                root.insert(1 if T.font != "Arial" else 0, etree.fromstring(_ARIAL_XML.format(ns=ns)))
            ft._blob = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def _strip_theme_fonts(root, font: str):
    """Remove theme-font attributes (w:asciiTheme ...) so the explicit font wins; also clear theme
    colours on w:color."""
    for rf in root.iter(qn("w:rFonts")):
        themed = False
        for a in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"):
            if rf.get(qn("w:" + a)) is not None:
                del rf.attrib[qn("w:" + a)]
                themed = True
        if themed or rf.get(qn("w:ascii")) is None:
            for a in ("ascii", "hAnsi", "cs", "eastAsia"):
                rf.set(qn("w:" + a), font)
    for c in root.iter(qn("w:color")):
        for a in ("themeColor", "themeShade", "themeTint"):
            if c.get(qn("w:" + a)) is not None:
                del c.attrib[qn("w:" + a)]


# =========================================================================== styles
def _set_style_font(st, T: Theme, size=None, color=None, bold=None, italic=None, caps=None, spacing=None, font=None):
    rpr = st.element.get_or_add_rPr()
    f = font or T.font
    _insert_ordered(rpr, _el("w:rFonts", ascii=f, hAnsi=f, cs=f, eastAsia=f), _RPR_ORDER)
    if size is not None:
        st.font.size = Pt(size)
        _insert_ordered(rpr, _el("w:szCs", val=int(round(size * 2))), _RPR_ORDER)
    if color is not None:
        _insert_ordered(rpr, _el("w:color", val=color), _RPR_ORDER)
    st.font.bold = bold
    st.font.italic = italic
    if caps is not None:
        st.font.all_caps = caps
    if spacing is not None:
        _insert_ordered(rpr, _el("w:spacing", val=spacing), _RPR_ORDER)


def _set_style_para(st, before=None, after=None, line=None, keep_next=None, align=None, ind=None):
    ppr = st.element.get_or_add_pPr()
    for tag in ("w:pBdr", "w:shd", "w:ind", "w:numPr", "w:tabs", "w:contextualSpacing"):
        for e in ppr.findall(qn(tag)):
            ppr.remove(e)
    sp = _el("w:spacing", before=int(before * 20) if before is not None else 0,
             after=int(after * 20) if after is not None else 0,
             line=line if line is not None else 240, lineRule="auto")
    _insert_ordered(ppr, sp, _PPR_ORDER)
    if keep_next is not None:
        if keep_next:
            _insert_ordered(ppr, _el("w:keepNext"), _PPR_ORDER)
        else:
            for e in ppr.findall(qn("w:keepNext")):
                ppr.remove(e)
    st.paragraph_format.alignment = align
    if ind:
        _insert_ordered(ppr, _el("w:ind", **{k: v for k, v in ind.items()}), _PPR_ORDER)


def _style(doc, name: str, style_id: str | None = None, base: str | None = "Normal",
           kind=WD_STYLE_TYPE.PARAGRAPH, builtin: bool = False):
    """Get a style by (internal) name, creating it when missing. Built-in names that python-docx does
    not know (toc 1, table of figures, footnote text ...) are created with Word's internal name so Word
    recognises them as its own (a regenerated TOC then uses our formatting)."""
    for s in doc.styles:
        if s.name.lower() == name.lower() and s.type == kind:
            return s
    st = doc.styles.add_style(name, kind, builtin=builtin)
    if builtin and st.element.get(qn("w:customStyle")) is not None:
        del st.element.attrib[qn("w:customStyle")]
    if style_id:
        st.element.set(qn("w:styleId"), style_id)
    if base and kind == WD_STYLE_TYPE.PARAGRAPH:
        st.base_style = doc.styles[base]
    st.quick_style = True
    return st


def _style_tabs(st, stops):
    ppr = st.element.get_or_add_pPr()
    tabs = _el("w:tabs")
    for pos, align, leader in stops:
        tabs.append(_el("w:tab", val=align, pos=pos, leader=leader))
    _insert_ordered(ppr, tabs, _PPR_ORDER)


def _style_border(st, **sides):
    set_borders(st.element.get_or_add_pPr(), "pBdr", _PPR_ORDER, **sides)


def define_styles(doc, T: Theme, mode: str, text_width: int):
    """Define every paragraph/character style the builder uses. Built-in styles (Normal, Title,
    Subtitle, Heading 1-4, Caption, toc 1/2, table of figures, footnote text, header, footer) are
    restyled so Word's navigation pane, TOC and cross-references keep working."""
    styles_root = doc.styles.element
    # document defaults: brand font, body colour, US English, no spacing
    rpr_default = styles_root.find(qn("w:docDefaults")).find(qn("w:rPrDefault")).find(qn("w:rPr"))
    _insert_ordered(rpr_default, _el("w:rFonts", ascii=T.font, hAnsi=T.font, cs=T.font, eastAsia=T.font), _RPR_ORDER)
    _insert_ordered(rpr_default, _el("w:lang", val="en-US", eastAsia="en-US", bidi="ar-SA"), _RPR_ORDER)
    _strip_theme_fonts(styles_root, T.font)
    for st_el in styles_root.findall(qn("w:style")):
        ppr = st_el.find(qn("w:pPr"))
        if ppr is not None:
            for e in ppr.findall(qn("w:numPr")):
                ppr.remove(e)          # the template's list styles pointed at the dropped definitions
    for rf in styles_root.iter(qn("w:rFonts")):
        if "courier" in (rf.get(qn("w:ascii")) or "").lower():
            for a in ("ascii", "hAnsi", "cs", "eastAsia"):
                rf.set(qn("w:" + a), T.code_font)

    normal = doc.styles["Normal"]
    _set_style_font(normal, T, size=T.body_pt, color=T.slate)
    _set_style_para(normal, before=0, after=T.body_after, line=T.body_line)
    normal.paragraph_format.widow_control = True

    report = mode == "report"
    # --- titles
    st = _style(doc, "Title")
    _set_style_font(st, T, size=T.cover_title_pt if report else T.note_title_pt, color=T.navy, bold=True)
    _set_style_para(st, before=0, after=8, line=250)
    st = _style(doc, "Subtitle")
    _set_style_font(st, T, size=T.cover_sub_pt if report else T.note_sub_pt, color=T.blue, bold=True)
    _set_style_para(st, before=0, after=6, line=260)
    # --- headings
    st = _style(doc, "Heading 1")
    if report:
        _set_style_font(st, T, size=T.h1_pt, color=T.navy, bold=True)
        _set_style_para(st, before=0, after=10, line=250, keep_next=True)
    else:
        _set_style_font(st, T, size=T.h1_note_pt, color=T.navy, bold=True)
        _set_style_para(st, before=20, after=6, line=250, keep_next=True)
        _style_tabs(st, [(540, "left", None)])
    st = _style(doc, "Heading 2")
    _set_style_font(st, T, size=T.h2_pt, color=T.navy, bold=True)
    _set_style_para(st, before=16, after=5, line=250, keep_next=True)
    _style_tabs(st, [(620, "left", None)])
    st = _style(doc, "Heading 3")
    _set_style_font(st, T, size=T.h3_pt, color=T.blue_dk, bold=True)
    _set_style_para(st, before=12, after=3, line=250, keep_next=True)
    st = _style(doc, "Heading 4")
    _set_style_font(st, T, size=T.h4_pt, color=T.navy, bold=True)
    _set_style_para(st, before=9, after=2, line=260, keep_next=True)
    # linked character styles follow their paragraph styles
    for name in ("Heading 1", "Heading 2", "Heading 3", "Heading 4", "Title", "Subtitle"):
        pst = doc.styles[name]
        link = pst.element.find(qn("w:link"))
        if link is None:
            continue
        for s in doc.styles:
            if s.element.get(qn("w:styleId")) == link.get(qn("w:val")):
                old = s.element.find(qn("w:rPr"))
                if old is not None:
                    s.element.remove(old)
                s.element.append(copy.deepcopy(pst.element.find(qn("w:rPr"))))
    # --- captions (figure / table), sources
    st = _style(doc, "Caption")
    _set_style_font(st, T, size=T.caption_pt, color=T.navy, bold=True, italic=False)
    _set_style_para(st, before=12, after=5, line=250, keep_next=True, align=WD_ALIGN_PARAGRAPH.LEFT)
    st = _style(doc, "Table Caption", "TableCaption", base="Caption")
    st = _style(doc, style_name(T, "Source"), style_id(T, "Source"))
    _set_style_font(st, T, size=T.source_pt, color=T.muted)
    _set_style_para(st, before=4, after=12, line=250)
    # --- kicker (letter-spaced small caps line above titles), lead, contents title
    st = _style(doc, style_name(T, "Kicker"), style_id(T, "Kicker"))
    _set_style_font(st, T, size=T.kicker_pt, color=T.blue_dk, bold=True, spacing=T.kicker_spacing)
    _set_style_para(st, before=0, after=5, line=240, keep_next=True)
    st = _style(doc, style_name(T, "Lead"), style_id(T, "Lead"))
    _set_style_font(st, T, size=T.cover_lead_pt if report else T.lead_pt, color=T.slate)
    _set_style_para(st, before=0, after=10, line=300)
    st = _style(doc, style_name(T, "Contents Title"), style_id(T, "Contents Title"))
    _set_style_font(st, T, size=T.h1_pt if report else T.h2_pt, color=T.navy, bold=True)
    _set_style_para(st, before=0, after=10 if report else 4, line=250, keep_next=True)
    st = _style(doc, style_name(T, "Contents Subtitle"), style_id(T, "Contents Subtitle"))
    _set_style_font(st, T, size=T.h2_pt, color=T.navy, bold=True)
    _set_style_para(st, before=20, after=6, line=250, keep_next=True)
    # --- lists
    st = _style(doc, "List Bullet")
    _set_style_font(st, T, size=T.body_pt, color=T.slate)
    _set_style_para(st, before=0, after=3, line=T.body_line)
    st = _style(doc, "List Number")
    _set_style_font(st, T, size=T.body_pt, color=T.slate)
    _set_style_para(st, before=0, after=3, line=T.body_line)
    # --- contents (Word's own style names so a regenerated TOC keeps this look)
    num_col = 400
    st = _style(doc, "toc 1", "TOC1", builtin=True)
    _set_style_font(st, T, size=T.toc1_pt, color=T.navy, bold=True)
    _set_style_para(st, before=9, after=2, line=250)
    _style_tabs(st, [(num_col, "left", None), (text_width, "right", None)])
    st = _style(doc, "toc 2", "TOC2", builtin=True)
    _set_style_font(st, T, size=T.toc2_pt, color=T.slate)
    _set_style_para(st, before=1, after=1, line=250, ind={"left": num_col})
    _style_tabs(st, [(num_col + 480, "left", None), (text_width, "right", "dot")])
    st = _style(doc, "table of figures", "TableofFigures", builtin=True)
    _set_style_font(st, T, size=T.toc2_pt, color=T.slate)
    _set_style_para(st, before=1, after=2, line=250, ind={"left": 0})
    _style_tabs(st, [(text_width, "right", "dot")])
    st = _style(doc, "TOC Heading")
    _set_style_font(st, T, size=T.h1_pt, color=T.navy, bold=True)
    _set_style_para(st, before=0, after=10, line=250)
    # --- footnotes
    st = _style(doc, "footnote text", "FootnoteText", builtin=True)
    _set_style_font(st, T, size=T.footnote_pt, color=T.slate)
    _set_style_para(st, before=0, after=2, line=240)
    st = _style(doc, "footnote reference", "FootnoteReference", base=None, kind=WD_STYLE_TYPE.CHARACTER, builtin=True)
    st.font.superscript = True
    _insert_ordered(st.element.get_or_add_rPr(), _el("w:color", val=T.blue_dk), _RPR_ORDER)
    # --- header / footer
    for name in ("Header", "Footer"):
        st = _style(doc, name)
        _set_style_font(st, T, size=T.header_pt, color=T.muted)
        _set_style_para(st, before=0, after=0, line=240)
    # --- quote
    st = _style(doc, "Quote")
    _set_style_font(st, T, size=T.body_pt + 2.5, color=T.navy, italic=True)
    _set_style_para(st, before=10, after=10, line=300)
    _strip_theme_fonts(styles_root, T.font)


# =========================================================================== footnotes (from cpcs_docx)
_FOOTNOTES_XML = (
    f'<w:footnotes {nsdecls("w", "r")}>'
    '<w:footnote w:type="separator" w:id="-1"><w:p><w:pPr><w:spacing w:after="0" w:line="240" w:lineRule="auto"/></w:pPr>'
    '<w:r><w:separator/></w:r></w:p></w:footnote>'
    '<w:footnote w:type="continuationSeparator" w:id="0"><w:p><w:pPr><w:spacing w:after="0" w:line="240" w:lineRule="auto"/></w:pPr>'
    '<w:r><w:continuationSeparator/></w:r></w:p></w:footnote>'
    '</w:footnotes>'
)


class Footnotes:
    """Collects real Word footnotes and writes the footnotes part on finalize()."""

    def __init__(self, doc, defs: dict[str, Inline], T: Theme):
        self.doc, self.defs, self.T = doc, defs, T
        self.items: list[tuple[int, Inline]] = []
        self.part = None
        for rel in doc.part.rels.values():
            if rel.reltype == RT.FOOTNOTES:
                self.part = rel.target_part
        if self.part is None:
            self.part = Part(PackURI("/word/footnotes.xml"),
                             "application/vnd.openxmlformats-officedocument.wordprocessingml.footnotes+xml",
                             _FOOTNOTES_XML.encode("utf-8"), doc.part.package)
            doc.part.relate_to(self.part, RT.FOOTNOTES)
        self.next_id = 1

    def add_reference(self, p, key: str):
        inline = self.defs.get(key)
        if inline is None:
            inline = parse_inline(key)  # allow [^literal note text]
        fid = self.next_id; self.next_id += 1
        r = p.add_run()
        rpr = r._r.get_or_add_rPr()
        _insert_ordered(rpr, _el("w:rStyle", val="FootnoteReference"), _RPR_ORDER)
        _insert_ordered(rpr, _el("w:vertAlign", val="superscript"), _RPR_ORDER)
        r._r.append(_el("w:footnoteReference", id=fid))
        self.items.append((fid, inline))

    def finalize(self):
        if not self.items:
            return
        root = parse_xml(self.part.blob)
        for fid, inline in self.items:
            fn = _el("w:footnote", id=fid)
            p = OxmlElement("w:p")
            ppr = OxmlElement("w:pPr"); ppr.append(_el("w:pStyle", val="FootnoteText")); p.append(ppr)
            r = OxmlElement("w:r"); rpr = OxmlElement("w:rPr"); rpr.append(_el("w:rStyle", val="FootnoteReference"))
            rpr.append(_el("w:vertAlign", val="superscript")); r.append(rpr); r.append(OxmlElement("w:footnoteRef")); p.append(r)
            para = Paragraph(p, None)
            add_text_run(para, " ", self.T, self.T.footnote_pt)
            add_inline_runs(para, inline, self.T, size=self.T.footnote_pt, color=self.T.slate, footnotes=None, links=False)
            fn.append(p); root.append(fn)
        self.part._blob = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


# =========================================================================== inline runs
def add_inline_runs(p, inline: Inline, T: Theme, size: float | None = None, color: str | None = None,
                    bold: bool | None = None, italic: bool | None = None, footnotes: Footnotes | None = None,
                    links: bool = True, code_size_delta: float = -0.5):
    """Render parsed inline runs into paragraph p (brand font on every run)."""
    for text, fmt in inline:
        if fmt.get("footnote"):
            if footnotes is not None:
                footnotes.add_reference(p, fmt["footnote"])
            continue
        if fmt.get("br"):
            p.add_run().add_break(WD_BREAK.LINE)
            continue
        if not text:
            continue
        b = True if fmt.get("bold") else bold
        i = True if fmt.get("italic") else italic
        if fmt.get("link") and links:
            url = fmt["link"]
            if url.startswith(("http://", "https://", "mailto:", "www.")):
                add_hyperlink(p, url if not url.startswith("www.") else "https://" + url, text, T, size, b, i)
                continue
        if fmt.get("code"):
            r = p.add_run(text)
            set_run_font(r, T.code_font, (size or T.body_pt) + code_size_delta, color or T.navy, b, i)
            continue
        r = p.add_run(text)
        set_run_font(r, T.font, size, color, b, i)
        if fmt.get("sup"):
            r.font.superscript = True
        if fmt.get("sub"):
            r.font.subscript = True


def _upper(text: str) -> str:
    return text.upper()


def _upper_inline(inline: Inline) -> Inline:
    """Kickers are set in real capitals (not the caps property, which some renderers turn into small
    caps when a run carries its own size). Table headers are NOT uppercased: units (t/yr, MW) matter."""
    return [(t.upper(), f) for t, f in inline]


def _inline_is_bold(inline: Inline) -> bool:
    runs = [(t, f) for t, f in inline if t.strip()]
    return bool(runs) and all(f.get("bold") for _, f in runs)


def _unbold(inline: Inline) -> Inline:
    return [(t, {k: v for k, v in f.items() if k != "bold"}) for t, f in inline]


# =========================================================================== list numbering (from cpcs_docx)
class Numbering:
    """Creates bullet / decimal list definitions on demand (colours from the theme)."""

    def __init__(self, doc, T: Theme):
        self.doc, self.T = doc, T
        self.el = doc.part.numbering_part.element
        self._bullet_num = None
        self._decimal_abs = None

    def _max(self, tag, attr):
        vals = [int(e.get(qn(attr))) for e in self.el.findall(qn(tag))]
        return max(vals) if vals else 0

    def _add_abstract(self, levels: list[dict]) -> int:
        aid = self._max("w:abstractNum", "w:abstractNumId") + 1
        an = _el("w:abstractNum", abstractNumId=aid)
        an.append(_el("w:multiLevelType", val="hybridMultilevel"))
        for i, lv in enumerate(levels):
            lvl = _el("w:lvl", ilvl=i)
            lvl.append(_el("w:start", val=1))
            lvl.append(_el("w:numFmt", val=lv["fmt"]))
            lvl.append(_el("w:lvlText", val=lv["text"]))
            lvl.append(_el("w:lvlJc", val="left"))
            ppr = OxmlElement("w:pPr"); ppr.append(_el("w:ind", left=lv["left"], hanging=lv["hanging"])); lvl.append(ppr)
            rpr = OxmlElement("w:rPr")
            f = lv.get("font", self.T.font)
            rpr.append(_el("w:rFonts", ascii=f, hAnsi=f, cs=f, hint="default"))
            if lv.get("bold"):
                rpr.append(_el("w:b"))
            if lv.get("color"):
                rpr.append(_el("w:color", val=lv["color"]))
            if lv.get("sz"):
                rpr.append(_el("w:sz", val=lv["sz"]))
            lvl.append(rpr)
            an.append(lvl)
        nums = self.el.findall(qn("w:num"))
        if nums:
            nums[0].addprevious(an)
        else:
            self.el.append(an)
        return aid

    def _add_num(self, aid: int, restart: bool = False) -> int:
        nid = self._max("w:num", "w:numId") + 1
        num = _el("w:num", numId=nid)
        num.append(_el("w:abstractNumId", val=aid))
        if restart:
            ov = _el("w:lvlOverride", ilvl=0); ov.append(_el("w:startOverride", val=1)); num.append(ov)
        self.el.append(num)
        return nid

    def bullets(self) -> int:
        T = self.T
        if self._bullet_num is None:
            aid = self._add_abstract([
                dict(fmt="bullet", text="•", left=300, hanging=220, color=T.blue, sz=22),
                dict(fmt="bullet", text="–", left=600, hanging=220, color=T.muted, sz=20),
                dict(fmt="bullet", text="•", left=900, hanging=220, color=T.muted, sz=16),
                dict(fmt="bullet", text="–", left=1200, hanging=220, color=T.muted, sz=16),
            ])
            self._bullet_num = self._add_num(aid)
        return self._bullet_num

    def attach_styles(self):
        """Give Word's 'List Bullet' / 'List Number' styles the AEI definitions, so lists typed in Word
        look like generated ones."""
        for name, nid in (("List Bullet", self.bullets()), ("List Number", self.decimal())):
            try:
                st = self.doc.styles[name]
            except KeyError:
                continue
            numpr = _el("w:numPr"); numpr.append(_el("w:numId", val=nid))
            _insert_ordered(st.element.get_or_add_pPr(), numpr, _PPR_ORDER)

    def decimal(self) -> int:
        """A fresh numbered list (restarts at 1)."""
        T = self.T
        if self._decimal_abs is None:
            self._decimal_abs = self._add_abstract([
                dict(fmt="decimal", text="%1.", left=340, hanging=340, color=T.blue_dk, bold=True),
                dict(fmt="lowerLetter", text="%2)", left=680, hanging=340, color=T.navy),
                dict(fmt="lowerRoman", text="%3.", left=1020, hanging=340, color=T.navy),
                dict(fmt="decimal", text="%4.", left=1360, hanging=340, color=T.navy),
            ])
        return self._add_num(self._decimal_abs, restart=True)


# =========================================================================== text metrics (from cpcs_docx)
_NUM_RE = re.compile(r"^[\s$€£+\-−–(]*(?:US\$|USD|EUR|US\\\$)?\s*[\d.,]+\s*(?:%|pp|bn|m|k|x|×|billion|million|Mt|t)?[\s)]*$|^[-–—]$|^n/?a$", re.I)


def _looks_numeric(s: str) -> bool:
    s = s.strip()
    return bool(s) and bool(_NUM_RE.match(s))


_FONT_FILES = {
    "Inter": [str(Path.home() / "Library/Fonts/Inter-VariableFont_opsz,wght.ttf"),
              "/Library/Fonts/Inter-VariableFont_opsz,wght.ttf",
              "/usr/share/fonts/truetype/inter/Inter-VariableFont_opsz,wght.ttf",
              "/usr/share/fonts/opentype/inter/Inter-VariableFont_opsz,wght.ttf",
              "/usr/local/share/fonts/Inter-VariableFont_opsz,wght.ttf",
              "C:/Windows/Fonts/Inter-VariableFont_opsz,wght.ttf",
              str(Path.home() / "AppData/Local/Microsoft/Windows/Fonts/Inter-VariableFont_opsz,wght.ttf")],
    "Arial": ["/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf", "C:/Windows/Fonts/arial.ttf",
              "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
              "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"],
    "Arial Bold": ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/Library/Fonts/Arial Bold.ttf",
                   "C:/Windows/Fonts/arialbd.ttf", "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf",
                   "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                   "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"],
}
_FONT_CACHE: dict = {}


def _metric_font(bold: bool, size10: int):
    """(Pillow font, width factor): Inter itself when installed (the variable font is set to the
    Regular/Bold instance), else Arial scaled to Inter's average width (+6.5 % / +4 %)."""
    key = (bold, size10)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    res = (None, 1.0)
    try:
        from PIL import ImageFont
        for f in _FONT_FILES["Inter"]:
            if os.path.exists(f):
                font = ImageFont.truetype(f, size10)
                try:
                    font.set_variation_by_name("Bold" if bold else "Regular")
                except Exception:  # noqa: BLE001 - static font or no variation support
                    pass
                res = (font, 1.0); break
        else:
            for f in _FONT_FILES["Arial Bold" if bold else "Arial"]:
                if os.path.exists(f):
                    res = (ImageFont.truetype(f, size10), 1.04 if bold else 1.065); break
    except Exception:  # noqa: BLE001
        res = (None, 1.0)
    _FONT_CACHE[key] = res
    return res


def text_width_twips(s: str, size_pt: float, bold: bool = False) -> int:
    """Rendered width of `s` in Inter at size_pt, in twips (real glyph metrics via Pillow; an
    average-character estimate when no font file is available)."""
    font, factor = _metric_font(bold, int(size_pt * 10))
    if font is not None:
        return int(font.getlength(s) * factor / 10 * 20)
    return int(len(s) * size_pt * (0.6 if bold else 0.56) * 20)


# =========================================================================== numbering pre-pass & refs
def assign_numbers(parts: list[dict], T: Theme, sequential: bool, numbered: bool, labels: dict) -> None:
    """Number figures, tables and headings once, before rendering, writing the result onto the blocks
    (`_label` on figures/tables, `_num` on headings) and into `labels` (id -> 'Figure 3', 'Section 2.1',
    'Chapter 2', 'Annex A') for @fig:/@tbl:/@sec: references. The renderer only reads these."""
    fig = tab = 0
    sep = T.label_sep
    for part in parts:
        kind = part["kind"]            # front | chapter | annex | note
        prefix = part.get("prefix", "")
        if not sequential and kind in ("chapter", "annex", "front"):
            fig = tab = 0
        label_prefix = "" if sequential or kind in ("front", "note") else prefix
        h = [0, 0, 0, 0]
        hb = part.get("heading")
        if hb is not None and hb.get("id"):
            labels[hb["id"]] = (f"{T.chapter_word} {prefix}" if kind == "chapter" else
                                f"{T.annex_word} {prefix}" if kind == "annex" else inline_text(hb["inline"]).strip())

        def walk(blocks):
            nonlocal fig, tab
            for b in blocks:
                t = b["type"]
                if t in ("figure", "table") and not (inline_text(b.get("caption") or []).strip() or b.get("id")):
                    b["_label"] = ""           # uncaptioned exhibits are not numbered
                elif t == "figure":
                    fig += 1
                    b["_label"] = f"{T.figure_word} {label_prefix + sep if label_prefix else ''}{fig}"
                    if b.get("id"):
                        labels[b["id"]] = b["_label"]
                elif t == "table":
                    tab += 1
                    b["_label"] = f"{T.table_word} {label_prefix + sep if label_prefix else ''}{tab}"
                    if b.get("id"):
                        labels[b["id"]] = b["_label"]
                elif t == "heading":
                    lvl = b["level"]
                    b["_num"] = ""
                    if kind == "note":
                        if numbered and b.get("numbered", True) and lvl <= 2:
                            h[lvl - 1] += 1
                            for k in range(lvl, 4):
                                h[k] = 0
                            b["_num"] = ".".join(str(x) for x in h[:lvl])
                        if b.get("id"):
                            if b["_num"]:
                                labels[b["id"]] = f"{T.section_word} {b['_num']}"
                            else:
                                labels[b["id"]] = f"{T.section_word} {'.'.join(str(x) for x in h[:2] if x)}".strip() \
                                    if h[0] else inline_text(b["inline"]).strip()
                    else:
                        if lvl == 2 and kind in ("chapter", "annex") and b.get("numbered", True):
                            h[1] += 1
                            b["_num"] = f"{prefix}.{h[1]}"
                        if b.get("id"):
                            if b["_num"]:
                                labels[b["id"]] = f"{T.section_word} {b['_num']}"
                            elif kind in ("chapter", "annex") and h[1]:
                                labels[b["id"]] = f"{T.section_word} {prefix}.{h[1]}"
                            else:
                                labels[b["id"]] = inline_text(b["inline"]).strip()
                elif t == "box":
                    walk(b["blocks"])
        walk(part["blocks"])


def resolve_refs(blocks: list[dict], labels: dict, warnings: list | None = None) -> None:
    """Replace ('', {'ref': id}) runs with the label text (in place, recursively)."""
    def fix(inline):
        out = []
        for text, fmt in inline:
            if "ref" in fmt:
                rid = fmt["ref"]
                lab = labels.get(rid)
                if lab is None and ":" in rid:
                    kind, _, key = rid.partition(":")
                    for alt in ("fig", "figure", "tbl", "tab", "table", "sec", "section"):
                        if f"{alt}:{key}" in labels:
                            lab = labels[f"{alt}:{key}"]; break
                if lab is None:
                    if warnings is not None:
                        warnings.append(f"unresolved reference @{rid}")
                    out.append((f"@{rid}", {k: v for k, v in fmt.items() if k != "ref"}))
                else:
                    out.append((lab, {k: v for k, v in fmt.items() if k != "ref"}))
            else:
                out.append((text, fmt))
        return out

    for b in blocks:
        for key in ("inline", "caption", "source", "title"):
            if b.get(key):
                b[key] = fix(b[key])
        if b["type"] == "table":
            b["header"] = [fix(c) for c in b["header"]]
            b["rows"] = [[fix(c) for c in r] for r in b["rows"]]
        if b["type"] == "list":
            for it in b["items"]:
                it["inline"] = fix(it["inline"])
        if b["type"] == "box":
            resolve_refs(b["blocks"], labels, warnings)


# =========================================================================== images
_MAX_PX = 2400


def _prepare_image(path: str, max_px: int = _MAX_PX):
    """Open an image; downscale when its long side exceeds max_px (big platform maps run to 8 MB)
    and re-encode formats Word cannot take. Returns (stream_or_path, (w, h), needs_frame)."""
    from PIL import Image
    with Image.open(path) as im:
        im.load()
        w, h = im.size
        fmt = (im.format or "").upper()
        frame = _edge_is_busy(im)
        if max(w, h) <= max_px and fmt in ("PNG", "JPEG", "GIF", "BMP"):
            return path, (w, h), frame
        scale = min(1.0, max_px / float(max(w, h)))
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        img = im
        if img.mode == "P":
            img = img.convert("RGBA")
        if img.mode in ("RGBA", "LA"):
            alpha = img.getchannel("A")
            if alpha.getextrema()[0] == 255:      # fully opaque: drop the alpha channel
                img = img.convert("RGB")
        elif img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img = img.resize((nw, nh), Image.LANCZOS) if scale < 1.0 else img
        buf = io.BytesIO()
        if fmt == "JPEG" and img.mode in ("RGB", "L"):
            img.save(buf, "JPEG", quality=90, optimize=True)
        else:   # a map or chart stays PNG: JPEG blocks a raster's flat colours (2026-10-11)
            img.save(buf, "PNG", optimize=True)
        buf.seek(0)
        return buf, (nw, nh), frame


def _edge_is_busy(im) -> bool:
    """True when the image does not sit on white (maps, screenshots): those get a hairline frame."""
    try:
        from PIL import Image
        small = im.convert("RGB").resize((120, max(1, round(120 * im.size[1] / im.size[0]))), Image.BILINEAR)
        W, H = small.size
        px = small.load()
        edge = [px[x, 0] for x in range(W)] + [px[x, H - 1] for x in range(W)] + \
               [px[0, y] for y in range(H)] + [px[W - 1, y] for y in range(H)]
        nonwhite = sum(1 for r, g, b in edge if min(r, g, b) < 250)
        return nonwhite / max(1, len(edge)) > 0.5
    except Exception:  # noqa: BLE001
        return False


def _picture_outline(inline_shape, color: str, width_emu: int = 6350):
    """Hairline outline around an inline picture (a:ln on pic:spPr)."""
    spPr = inline_shape._inline.graphic.graphicData.pic.spPr
    ln = parse_xml(f'<a:ln {nsdecls("a")} w="{width_emu}"><a:solidFill><a:srgbClr val="{color}"/></a:solidFill></a:ln>')
    spPr.append(ln)


def _to_anchor(inline_shape, x_emu: int, y_emu: int, wrap: str = "topAndBottom", behind: bool = False, z: int = 251658240):
    """Convert an inline picture into a floating one positioned relative to the page."""
    inline = inline_shape._inline
    ns = nsdecls("wp", "a", "pic", "r")
    extent = inline.find(qn("wp:extent"))
    docpr = inline.find(qn("wp:docPr"))
    graphic = inline.find(qn("a:graphic"))
    wrap_xml = {"topAndBottom": "<wp:wrapTopAndBottom/>", "none": "<wp:wrapNone/>"}[wrap]
    anchor = parse_xml(
        f'<wp:anchor {ns} distT="0" distB="0" distL="0" distR="0" simplePos="0" relativeHeight="{z}" '
        f'behindDoc="{1 if behind else 0}" locked="1" layoutInCell="1" allowOverlap="1">'
        f'<wp:simplePos x="0" y="0"/>'
        f'<wp:positionH relativeFrom="page"><wp:posOffset>{int(x_emu)}</wp:posOffset></wp:positionH>'
        f'<wp:positionV relativeFrom="page"><wp:posOffset>{int(y_emu)}</wp:posOffset></wp:positionV>'
        f'<wp:extent cx="{extent.get("cx")}" cy="{extent.get("cy")}"/>'
        f'<wp:effectExtent l="0" t="0" r="0" b="0"/>{wrap_xml}'
        f'<wp:docPr id="{docpr.get("id")}" name="{docpr.get("name")}"/>'
        f'<wp:cNvGraphicFramePr/></wp:anchor>')
    anchor.append(graphic)
    inline.getparent().replace(inline, anchor)
    return anchor


# =========================================================================== renderer
class Renderer:
    """Turns parsed blocks into OOXML. Reads brand values from the Theme only."""

    def __init__(self, doc, T: Theme, mode: str, meta: dict, footnotes: Footnotes, base_dir: str, text_width: int):
        self.doc, self.T, self.mode, self.meta, self.fn = doc, T, mode, meta, footnotes
        self.base_dir = str(base_dir)
        self.num = Numbering(doc, T)
        self.width_stack = [text_width]
        self.text_width = text_width
        self.page_height_in = 11.0
        self.figures = 0
        self.tables = 0
        self.captions: list[dict] = []     # {"kind", "text", "bookmark"}
        self.toc: list[dict] = []          # {"level", "num", "title", "bookmark", "search"}
        self.warnings: list[str] = []
        self.sect_template = None          # sectPr cloned for ::: landscape blocks
        self.sect_started = False
        self._bm = 0

    # ---- utilities
    @property
    def width(self) -> int:
        return self.width_stack[-1]

    def resolve(self, path: str) -> str | None:
        """Find an image: absolute, relative to base_dir, with leading folders stripped (pandoc
        --extract-media prefixes the output folder), or by basename under media/ and figures/."""
        if not path:
            return None
        path = path.strip().strip("<>")
        if path.startswith("file://"):
            path = path[7:]
        parts = path.replace("\\", "/").split("/")
        # relative paths resolve against base_dir first; the working directory is the last resort
        cands = [path] if os.path.isabs(path) else [os.path.join(self.base_dir, path)]
        for k in range(1, len(parts)):
            cands.append(os.path.join(self.base_dir, *parts[k:]))
        cands.append(os.path.join(self.base_dir, "media", parts[-1]))
        cands.append(os.path.join(self.base_dir, "figures", parts[-1]))
        if not os.path.isabs(path):
            cands.append(path)
        for c in cands:
            if os.path.isfile(c):
                return c
        return None

    def bookmark(self, p) -> str:
        """Wrap the paragraph content in a _Toc bookmark (TOC/LOF hyperlinks and PAGEREF fields)."""
        self._bm += 1
        name = f"_Toc{90000000 + self._bm}"
        start = _el("w:bookmarkStart", id=self._bm, name=name)
        end = _el("w:bookmarkEnd", id=self._bm)
        ppr = p._p.find(qn("w:pPr"))
        if ppr is not None:
            ppr.addnext(start)
        else:
            p._p.insert(0, start)
        p._p.append(end)
        return name

    def para(self, container, inline: Inline, style: str | None = None, size=None, color=None, bold=None,
             italic=None, align=None, before=None, after=None, keep_next=False, line=None):
        p = container.add_paragraph()
        if style:
            p.style = self.doc.styles[style]
        add_inline_runs(p, inline, self.T, size=size, color=color, bold=bold, italic=italic, footnotes=self.fn)
        if align:
            set_jc(p, align)
        if before is not None or after is not None or line is not None:
            set_spacing(p, before, after, line)
        if keep_next:
            set_keep_next(p)
        return p

    def body_para(self, container, inline: Inline):
        return self.para(container, inline)

    # ---- dispatch
    def render_blocks(self, blocks: list[dict], container, in_box: bool = False, box_color: str | None = None):
        for b in blocks:
            t = b["type"]
            if t == "heading":
                self.heading(container, b, in_box)
            elif t == "para":
                if not inline_text(b["inline"]).strip() and not any(f.get("footnote") or f.get("br") for _, f in b["inline"]):
                    continue
                if in_box:
                    self.para(container, b["inline"], size=self.T.box_pt, color=box_color or self.T.slate,
                              before=0, after=80, line=270)
                else:
                    self.body_para(container, b["inline"])
            elif t == "list":
                self.list_block(container, b, in_box, box_color)
            elif t == "table":
                self.table(container, b)
            elif t == "figure":
                self.figure(container, b)
            elif t == "code":
                self.code(container, b)
            elif t == "box":
                self.box(container, b)
            elif t == "pipeline":
                self.pipeline(container, b)
            elif t == "pagebreak":
                page_break(container, self.T)
            elif t == "hr":
                p = container.add_paragraph(); set_spacing(p, 60, 120, line=240)
                set_borders(pPr(p), "pBdr", _PPR_ORDER, bottom=(4, self.T.hairline, 1))

    # ---- openers (report H1s)
    def kicker(self, container, text: str, page_break: bool = False, after: int | None = None):
        p = container.add_paragraph(style=self.doc.styles[style_name(self.T, "Kicker")])
        add_text_run(p, _upper(text), self.T)
        if page_break:
            set_page_break_before(p)
        if after is not None:
            set_spacing(p, after=after)
        return p

    def short_rule(self, container, width_in: float = 1.6, before: int = 0, after: int = 240, sz: int = 12):
        """The brand's short blue rule (cover, chapter openers): an empty paragraph with a bottom border
        and a right indent that limits it to `width_in`."""
        p = container.add_paragraph()
        set_spacing(p, before, after, line=120, rule="exact")
        set_indent(p, right=max(0, self.width - int(width_in * 1440)))
        set_borders(pPr(p), "pBdr", _PPR_ORDER, bottom=(sz, self.T.blue, 0))
        set_keep_next(p)
        set_run_font(p.add_run(""), self.T.font, 2)
        return p

    def opener(self, container, title_inline: Inline, kicker: str | None, prefix: str, page_break: bool):
        """Chapter / annex / front-matter opener: kicker line + Heading 1 (+ short rule). The chapter
        label ('3', 'A') is written into the heading as hidden text so Word's regenerated TOC can list
        it without the page showing the number twice (the kicker already says CHAPTER 3)."""
        if kicker:
            self.kicker(container, kicker, page_break=page_break, after=110)
        h = container.add_paragraph(style=self.doc.styles["Heading 1"])
        if prefix:
            add_text_run(h, f"{prefix}\t", self.T, hidden=True)
        add_inline_runs(h, title_inline, self.T, footnotes=self.fn)
        if page_break and not kicker:
            set_page_break_before(h)
        name = self.bookmark(h)
        title = inline_text(title_inline).strip()
        self.toc.append({"level": 1, "num": prefix, "title": title, "bookmark": name, "search": title})
        self.short_rule(container, width_in=1.1, before=40, after=300, sz=18)
        return h

    # ---- headings
    def heading(self, container, b: dict, in_box: bool = False):
        T = self.T
        level = b["level"]
        if in_box:
            return self.para(container, b["inline"], size=T.box_pt + 0.5, color=T.navy, bold=True, before=60, after=40,
                             keep_next=True)
        num = b.get("_num", "")
        title = inline_text(b["inline"]).strip()
        if self.mode == "note" and level == 1 and b.get("kicker"):
            k = self.kicker(container, b["kicker"], after=60)
            set_spacing(k, before=400)
            set_borders(pPr(k), "pBdr", _PPR_ORDER, top=(4, T.hairline, 8))
            p = container.add_paragraph(style=self.doc.styles["Heading 1"])
            set_spacing(p, before=0)
        else:
            style = {1: "Heading 1", 2: "Heading 2", 3: "Heading 3"}.get(level, "Heading 4")
            p = container.add_paragraph(style=self.doc.styles[style])
            if self.mode == "note" and level == 1:
                # note sections: hairline above, blue number
                set_borders(pPr(p), "pBdr", _PPR_ORDER, top=(4, T.hairline, 8))
        if num:
            add_text_run(p, num, T, color=T.blue if level <= 2 else None)
            add_text_run(p, "\t", T)
        add_inline_runs(p, b["inline"], T, footnotes=self.fn)
        toc_level = 1 if (self.mode == "note" and level == 1) else (2 if level == 2 else None)
        if toc_level:
            name = self.bookmark(p)
            self.toc.append({"level": toc_level, "num": num, "title": title, "bookmark": name,
                             "search": f"{num} {title}".strip()})
        return p

    # ---- lists
    def list_block(self, container, b: dict, in_box: bool = False, box_color: str | None = None):
        T = self.T
        items = b["items"]
        ordered_num = None
        size = T.box_pt if in_box else T.body_pt
        color = box_color if in_box else T.slate
        last = None
        for it in items:
            lvl = min(it["level"], 3)
            if it["ordered"]:
                if ordered_num is None:
                    ordered_num = self.num.decimal()
                p = container.add_paragraph(style=self.doc.styles["List Number"])
                set_num(p, ordered_num, lvl)
                set_indent(p, left=340 + 340 * lvl, hanging=340)
            else:
                p = container.add_paragraph(style=self.doc.styles["List Bullet"])
                set_num(p, self.num.bullets(), lvl)
                set_indent(p, left=300 + 300 * lvl, hanging=220)
            set_spacing(p, 0, 50 if in_box else 70, line=270 if in_box else T.body_line)
            add_inline_runs(p, it["inline"], T, size=size, color=color, footnotes=self.fn)
            last = p
        if last is not None:
            set_spacing(last, after=60 if in_box else 150)

    # ---- captions & sources
    def caption(self, container, kind: str, label: str, inline: Inline | None):
        T = self.T
        style = "Caption" if kind == "figure" else "Table Caption"
        p = container.add_paragraph(style=self.doc.styles[style])
        text = inline_text(inline or []).strip()
        add_text_run(p, label, T, color=T.blue_dk, bold=True)
        if text:
            add_text_run(p, "\u2002\u2002", T)
            add_inline_runs(p, inline, T, color=T.navy, bold=True, footnotes=self.fn)
        set_keep_next(p); set_keep_lines(p)
        name = self.bookmark(p)
        self.captions.append({"kind": kind, "text": f"{label}\u2002\u2002{text}" if text else label, "bookmark": name,
                              "search": f"{label} {text}".strip()})
        return p

    def source(self, container, inline: Inline):
        p = container.add_paragraph(style=self.doc.styles[style_name(self.T, "Source")])
        add_inline_runs(p, inline, self.T, footnotes=self.fn)
        return p

    # ---- tables
    def table(self, container, b: dict):
        T = self.T
        header, rows = b["header"], b["rows"]
        ncol = len(header)
        if ncol == 0:
            return
        self.tables += 1
        label = b.get("_label") or ""
        if label:
            self.caption(container, "table", label, b.get("caption"))
        total = self.width
        if b.get("width"):
            total = min(total, int(b["width"] * 1440))
        size = T.table_pt
        if ncol >= 7:
            size = max(T.table_min_pt, size - 0.5)
        if ncol >= 9:
            size = max(T.table_min_pt, size - 0.5)
            if ncol > 9 and self.width <= self.text_width:
                self.warnings.append(f"{label or 'a table'} has {ncol} columns: consider ::: landscape, a transpose or a split")
        hsize = T.table_header_pt
        # bold first cells: a few = highlighted rows (totals, the scenario that matters); most = a row-header column
        bold_first = [bool(r) and _inline_is_bold(r[0]) for r in rows]
        n_bold = sum(bold_first)
        highlight = [False] * len(rows)
        rowhead = False
        if rows and n_bold:
            if n_bold <= max(1, len(rows) // 3) and n_bold < len(rows):
                highlight = bold_first
            else:
                rowhead = True
        rows = [[_unbold(c) for c in r] if (highlight[i] or rowhead) else r for i, r in enumerate(rows)]
        header = [_unbold(c) for c in header]
        widths = self._col_widths(header, rows, total, size, hsize, bold_cols={0} if rowhead else set(),
                                  bold_rows={i for i, h in enumerate(highlight) if h})
        numeric = []
        for c in range(ncol):
            vals = [inline_text(r[c]) for r in rows if c < len(r) and inline_text(r[c]).strip()]
            numeric.append(bool(vals) and sum(_looks_numeric(v) for v in vals) >= 0.7 * len(vals))
        align = [b["align"][c] if c < len(b["align"]) and b["align"][c] else ("right" if numeric[c] and c > 0 else "left")
                 for c in range(ncol)]
        align = ["both" if a == "justify" else a for a in align]

        tbl = container.add_table(rows=len(rows) + 1, cols=ncol)
        table_set(tbl, total, widths, borders=dict(top=(6, T.rule), left=None, bottom=(6, T.rule), right=None,
                                                   insideH=(4, T.hairline), insideV=None),
                  margins=(62, 90, 62, 90))
        keep_together = len(rows) <= 14
        # header row: small bold muted text, no fill (or the theme's header fill), stronger rule below
        row_set(tbl.rows[0], header=True)
        for c, cell in enumerate(tbl.rows[0].cells):
            hb = dict(bottom=(6, T.rule))
            if T.table_header_fill:            # side borders in the fill colour: no seams between cells
                hb.update(left=(4, T.table_header_fill), right=(4, T.table_header_fill))
            cell_set(cell, fill=T.table_header_fill, valign="bottom", borders=hb)
            p = cell.paragraphs[0]
            # header text keeps the Markdown's case (units: t/yr, MW, kt): small, bold, muted, lightly tracked
            add_inline_runs(p, header[c], T, size=hsize, color=T.table_header_text or T.muted, bold=True)
            for r in p.runs:
                set_run_font(r, T.font, spacing=T.table_header_spacing)
            set_spacing(p, 0, 0, line=240); set_jc(p, align[c] if align[c] != "both" else "left")
            set_keep_next(p)
        # body rows
        for r_i, row in enumerate(rows):
            tr = tbl.rows[r_i + 1]
            row_set(tr)
            hl = highlight[r_i]
            for c, cell in enumerate(tr.cells):
                cell_set(cell, fill=T.tint if hl else None, valign="top")
                p = cell.paragraphs[0]
                if hl:
                    color, bold = T.blue_dk, True
                elif c == 0:
                    color, bold = T.navy, (True if rowhead else None)
                else:
                    color, bold = T.slate, None
                add_inline_runs(p, row[c] if c < len(row) else [], T, size=size, color=color, bold=bold,
                                footnotes=self.fn)
                set_spacing(p, 0, 0, line=250); set_jc(p, align[c])
                if keep_together and r_i < len(rows) - 1:
                    set_keep_next(p)
        if b.get("source"):
            self.source(container, b["source"])
        else:
            _spacer(container, T, 200)

    def _col_widths(self, header, rows, total: int, size: float, header_size: float,
                    bold_cols: set | None = None, bold_rows: set | None = None) -> list[int]:
        """Column widths from real text metrics (cpcs_docx algorithm, extended). Numbers and words never
        break. Space is then handed out in priority order: short body cells (<= 24 characters) on one
        line, headers on at most two lines, long text cells, and what is left goes to the text columns."""
        bold_cols = bold_cols or set(); bold_rows = bold_rows or set()
        ncol = len(header)
        pad = 2 * 90 + 60                      # cell margins + safety
        tracking = self.T.table_header_spacing
        hw = lambda t: text_width_twips(t, header_size, True) + len(t) * tracking   # + letter spacing
        min_w = [0] * ncol; short = [0] * ncol; full = [0] * ncol; head2 = [0] * ncol
        numeric_col = [True] * ncol
        for c in range(ncol):
            htxt = inline_text(header[c]).strip()
            words = htxt.split() or [""]
            for w in words:
                min_w[c] = max(min_w[c], hw(w) + pad)
            # narrowest width that sets the header on two lines
            best = hw(htxt)
            for k in range(1, len(words)):
                best = min(best, max(hw(" ".join(words[:k])), hw(" ".join(words[k:]))))
            head2[c] = best + pad
            seen = False
            for r_i, r in enumerate(rows):
                if c >= len(r):
                    continue
                txt = inline_text(r[c]).strip()
                if not txt:
                    continue
                seen = True
                bold = c in bold_cols or r_i in bold_rows or any(f.get("bold") for _, f in r[c])
                min_w[c] = max(min_w[c], max(text_width_twips(w, size, bold) for w in txt.split()) + pad)
                fw = text_width_twips(txt, size, bold) + pad
                if _looks_numeric(txt):
                    fw += 60
                    min_w[c] = max(min_w[c], fw)
                else:
                    numeric_col[c] = False
                full[c] = max(full[c], fw)
                if len(txt) <= 24:
                    short[c] = max(short[c], fw)
            if not seen:
                numeric_col[c] = False
        cap = int(total * 0.6)
        targets = [
            [max(m, s) for m, s in zip(min_w, short)],
            [max(m, s, min(h, cap)) for m, s, h in zip(min_w, short, head2)],
            [max(m, s, min(h, cap), min(f, cap)) for m, s, h, f in zip(min_w, short, head2, full)],
        ]
        s_min = sum(min_w)
        if s_min >= total:
            widths = [int(total * m / s_min) for m in min_w]
        else:
            widths = list(min_w)
            for tgt in targets:
                need = [max(0, t - w) for t, w in zip(tgt, widths)]
                room = total - sum(widths)
                if sum(need) <= room:
                    widths = [w + n for w, n in zip(widths, need)]
                else:
                    widths = [w + int(room * n / sum(need)) for w, n in zip(widths, need)]
                    break
            left = total - sum(widths)
            if left > 0:
                weight = [w if not numeric_col[i] else 0 for i, w in enumerate(widths)]
                if not sum(weight):
                    weight = list(widths)
                ws = sum(weight) or 1
                widths = [w + int(left * wt / ws) for w, wt in zip(widths, weight)]
        widths[-1] += total - sum(widths)
        return widths

    # ---- figures
    def figure(self, container, b: dict):
        T = self.T
        self.figures += 1
        label = b.get("_label") or ""
        path = self.resolve(b.get("path", ""))
        if label:
            self.caption(container, "figure", label, b.get("caption"))
        if path is None:
            self.warnings.append(f"missing figure: {b.get('path')}")
            p = self.para(container, parse_inline(f"[Figure file not found: {b.get('path')}]"), size=T.source_pt + 0.5,
                          color=T.muted, italic=True, before=40, after=40)
            set_shd(pPr(p), T.card)
            if b.get("source"):
                self.source(container, b["source"])
            return
        avail_in = self.width / 1440.0
        width_in = min(b.get("width") or avail_in, avail_in)
        try:
            stream, (iw, ih), frame = _prepare_image(path)
        except Exception as e:  # noqa: BLE001 - unreadable image: warn, never raise
            self.warnings.append(f"unreadable figure {b.get('path')}: {e}")
            self.para(container, parse_inline(f"[Figure could not be read: {b.get('path')}]"), size=T.source_pt + 0.5,
                      color=T.muted, italic=True)
            return
        text_h = self.page_height_in - self.T.margin_top - self.T.margin_bottom
        max_h = 0.60 * text_h - 0.65    # caption + figure + source within ~60 % of the text height
        if width_in * ih / iw > max_h:
            width_in = max_h * iw / ih
        attrs_frame = str((b.get("attrs") or {}).get("frame", "")).lower()
        if attrs_frame in ("false", "no", "0"):
            frame = False
        elif attrs_frame in ("true", "yes", "1"):
            frame = True
        p = container.add_paragraph()
        set_jc(p, "left")
        set_spacing(p, 20, 40, line=240)
        set_keep_next(p)
        shape = p.add_run().add_picture(stream, width=Inches(width_in))
        if frame:
            _picture_outline(shape, T.frame)
        if b.get("source"):
            self.source(container, b["source"])
        else:
            _spacer(container, T, 160)

    # ---- code
    def code(self, container, b: dict):
        T = self.T
        lines = b["text"].split("\n") or [""]
        for i, line in enumerate(lines):
            p = container.add_paragraph()
            add_text_run(p, line.replace("\t", "    "), T, T.code_pt, T.navy, font=T.code_font)
            set_shd(pPr(p), T.card)
            set_borders(pPr(p), "pBdr", _PPR_ORDER, left=(12, T.sky, 6))
            set_indent(p, left=160, right=160)
            set_spacing(p, 80 if i == 0 else 0, 80 if i == len(lines) - 1 else 0, line=250)
            set_keep_lines(p)
            if i < len(lines) - 1:
                set_keep_next(p)
        _spacer(container, T, 120)

    # ---- boxes
    _BOX_KINDS = {
        # kind: (fill attr, bar attr, default title attr, title style, text colour attr)
        "takeaway": ("tint", "blue", "takeaway_title", "kicker", "navy"),
        "box": ("card", "blue", None, "title", "slate"),
        "note": ("card", "sky", "note_title", "title", "slate"),
        "info": ("card", "sky", "note_title", "title", "slate"),
        "warning": ("warn_fill", "warn", "warning_title", "warn", "slate"),
        "caution": ("warn_fill", "warn", "warning_title", "warn", "slate"),
    }

    def box(self, container, b: dict):
        T = self.T
        kind = (b.get("kind") or "box").lower()
        if kind in ("quote", "callout", "pull"):
            return self._quote(container, b)
        if kind == "landscape":
            return self._landscape(container, b)
        if kind in ("kpis", "kpi", "stats", "stat", "numbers"):
            return self.kpis(container, b)
        fill_a, bar_a, title_a, title_style, text_a = self._BOX_KINDS.get(kind, self._BOX_KINDS["box"])
        fill, bar, text_color = getattr(T, fill_a), getattr(T, bar_a), getattr(T, text_a)
        title = b.get("title") or (parse_inline(getattr(T, title_a)) if title_a else None)
        if kind not in self._BOX_KINDS and not b.get("title"):
            title = parse_inline(kind.replace("-", " ").capitalize())
        margins = (150, 230, 150, 230)
        tbl = container.add_table(rows=1, cols=1)
        table_set(tbl, self.width, [self.width], align="left",
                  borders=dict(top=None, left=(24, bar), bottom=None, right=None, insideH=None, insideV=None),
                  margins=margins)
        cell = tbl.cell(0, 0)
        cell_set(cell, fill=fill, width=self.width)
        row_set(tbl.rows[0], cant_split=len(b["blocks"]) <= 4)
        self.width_stack.append(self.width - margins[1] - margins[3] - 60)
        if title:
            if title_style == "kicker":
                p = self.para(cell, _upper_inline(title), size=T.kicker_pt - 0.5, color=T.blue_dk, bold=True, before=0,
                              after=70, keep_next=True, line=240)
                for r in p.runs:
                    set_run_font(r, T.font, spacing=T.kicker_spacing - 6)
            elif title_style == "warn":
                self.para(cell, title, size=T.box_title_pt, color=T.warn_text, bold=True, before=0, after=60,
                          keep_next=True, line=250)
            else:
                self.para(cell, title, size=T.box_title_pt, color=T.navy, bold=True, before=0, after=60,
                          keep_next=True, line=250)
        self.render_blocks(b["blocks"], cell, in_box=True, box_color=text_color)
        self.width_stack.pop()
        clear_cell(cell)
        if cell.paragraphs:
            set_spacing(cell.paragraphs[-1], after=0)
        _spacer(container, T, 200)

    # ---- KPI stat cards
    @staticmethod
    def _kpi_split(inline: Inline) -> tuple[Inline, Inline]:
        """'**17.1 Mt/yr** freight diverted' -> (value runs, label runs)."""
        runs = list(inline)
        value: Inline = []
        i = 0
        while i < len(runs) and (runs[i][1].get("bold") or not runs[i][0].strip()) and not runs[i][1].get("footnote"):
            if runs[i][0].strip() or value:
                value.append((runs[i][0], {k: v for k, v in runs[i][1].items() if k != "bold"}))
            i += 1
        if value and inline_text(value).strip():
            label = runs[i:]
        else:
            txt = inline_text(runs)
            m = re.match(r"^\s*(.+?)\s+(?:—|–|-|:)\s+(.+)$", txt)
            if m:
                return parse_inline(m.group(1)), parse_inline(m.group(2))
            return runs, []
        # trim leading separators / spaces from the label
        while label and not label[0][0].strip():
            label = label[1:]
        if label:
            t0 = re.sub(r"^\s*(?:[—–:,-]\s*)?", "", label[0][0])
            label = ([(t0, label[0][1])] if t0 else []) + label[1:]
        return value, label

    def kpis(self, container, b: dict):
        T = self.T
        items: list[tuple[Inline, Inline]] = []
        for blk in b["blocks"]:
            if blk["type"] == "list":
                items += [self._kpi_split(it["inline"]) for it in blk["items"]]
            elif blk["type"] == "para":
                items.append(self._kpi_split(blk["inline"]))
        if not items:
            return
        if b.get("title"):
            p = self.kicker(container, "", after=90)
            add_inline_runs(p, _upper_inline(b["title"]), T)
        n = len(items)
        n_rows = math.ceil(n / 4)
        per_row = math.ceil(n / n_rows)
        gap = 180
        for r0 in range(0, n, per_row):
            chunk = items[r0:r0 + per_row]
            k = len(chunk)
            card_w = int((self.width - gap * (k - 1)) / k)
            widths = []
            for j in range(k):
                widths.append(card_w)
                if j < k - 1:
                    widths.append(gap)
            widths[-1] += self.width - sum(widths)
            tbl = container.add_table(rows=1, cols=len(widths))
            table_set(tbl, self.width, widths, align="left",
                      borders=dict(top=None, left=None, bottom=None, right=None, insideH=None, insideV=None),
                      margins=(0, 0, 0, 0))
            row_set(tbl.rows[0], cant_split=True)
            cells = tbl.rows[0].cells
            inner = card_w - 190 - 190
            size = T.stat_pt            # one size per row: the largest that fits every value (>= 15 pt;
            fit = 0.94 * inner          # longer values wrap: keep KPI values short, context in the label)
            while size > 15 and any(text_width_twips(inline_text(v).strip(), size, True) > fit for v, _ in chunk):
                size -= 0.5
            for j, (value, label) in enumerate(chunk):
                cell = cells[2 * j]
                cell_set(cell, fill=T.tint, margins=(170, 190, 180, 190), valign="top")
                p = cell.paragraphs[0]
                add_inline_runs(p, value, T, size=size, color=T.kpi_value or T.blue, bold=True)
                set_spacing(p, 0, 50, line=240)
                set_keep_next(p)
                if label:
                    q = cell.add_paragraph()
                    add_inline_runs(q, label, T, size=T.stat_label_pt, color=T.slate, footnotes=self.fn)
                    set_spacing(q, 0, 0, line=260)
            for j in range(k - 1):
                gc = cells[2 * j + 1]
                set_spacing(gc.paragraphs[0], 0, 0, line=240)
            if r0 + per_row < n:
                _spacer(container, T, 0, size=4)
                for c in tbl.rows[0].cells:
                    for q in c.paragraphs:
                        set_keep_next(q)
        _spacer(container, T, 220)

    # ---- pipeline flowcharts
    def pipeline(self, container, b: dict):
        """A `::: pipeline` block as a flowchart built from one table, so Word edits it and
        LibreOffice and PDF draw it the same: a row of boxes per stage (tint fill, a frame in the
        brand blue, a bold title over its detail lines) and between stages a row of arrows, over
        each box of a stage of several (a fork), else under each box of the stage above (a
        merge), else one. A stage of one box spans the full width, so the arrows of a fork leave
        it and those of a merge reach it. The grid has as many units as the stages' box counts
        need (2 and 3 boxes: 6), with a gap column between units."""
        T = self.T
        stages = b["stages"]
        if not stages:
            return
        if b.get("title"):
            p = self.kicker(container, "", after=90)
            add_inline_runs(p, _upper_inline(b["title"]), T)
            set_keep_next(p)
        units = math.lcm(*{len(s) for s in stages})
        gap = 240 if units > 1 else 0
        unit_w = (self.width - gap * (units - 1)) // units
        widths = [w for _ in range(units) for w in (unit_w, gap)][:-1]
        widths[-1] += self.width - sum(widths)

        def spans(n: int) -> list[tuple[int, int]]:      # (first, last) grid column of each box
            k = units // n
            return [(2 * k * j, 2 * k * (j + 1) - 2) for j in range(n)]

        tbl = container.add_table(rows=2 * len(stages) - 1, cols=len(widths))
        table_set(tbl, self.width, widths, align="left",
                  borders=dict(top=None, left=None, bottom=None, right=None, insideH=None, insideV=None),
                  margins=(0, 0, 0, 0))
        for row in tbl.rows:
            row_set(row)
            for cell in row.cells:          # empty cells take no height
                set_spacing(cell.paragraphs[0], 0, 0, line=40, rule="exact")
        frame = (8, T.blue)
        for i, stage in enumerate(stages):
            row = 2 * i
            for (a, z), box in zip(spans(len(stage)), stage):
                cell = tbl.cell(row, a).merge(tbl.cell(row, z)) if z > a else tbl.cell(row, a)
                cell_set(cell, fill=T.tint, margins=(90, 120, 100, 120), valign="center",
                         borders=dict(top=frame, left=frame, bottom=frame, right=frame))
                p = cell.paragraphs[0]
                add_inline_runs(p, box["title"], T, size=T.box_pt + 0.5, color=T.navy, bold=True,
                                footnotes=self.fn)
                set_spacing(p, 0, 20 if box["details"] else 0, line=250)
                set_jc(p, "center")
                for d in box["details"]:
                    q = cell.add_paragraph()
                    add_inline_runs(q, d, T, size=T.table_pt, color=T.slate, footnotes=self.fn)
                    set_spacing(q, 0, 0, line=250)
                    set_jc(q, "center")
            if i + 1 == len(stages):
                break
            nxt = stages[i + 1]
            for a, z in spans(len(nxt) if len(nxt) > 1 else len(stage)):
                cell = tbl.cell(row + 1, a).merge(tbl.cell(row + 1, z)) if z > a else tbl.cell(row + 1, a)
                p = cell.paragraphs[0]
                add_text_run(p, "▼", T, size=T.box_pt, color=T.blue)
                set_spacing(p, 30, 30, line=220, rule="exact")
                set_jc(p, "center")
        for row in tbl.rows[:-1]:            # the chart stays on one page
            for cell in row.cells:
                for p in cell.paragraphs:
                    set_keep_next(p)
        _spacer(container, T, 200)

    def _landscape(self, container, b: dict):
        """Render the blocks on landscape pages: close the current (portrait) section, emit the
        content, then close a landscape section. Headers/footers are inherited. (cpcs_docx logic.)"""
        tmpl = self.sect_template
        if tmpl is None or container is not self.doc:
            return self.render_blocks(b["blocks"], container)

        def sect_copy(landscape: bool):
            sp = copy.deepcopy(tmpl)
            for e in sp.findall(qn("w:type")):
                sp.remove(e)
            pg = sp.find(qn("w:pgNumType"))
            if pg is not None and (landscape or self.sect_started) and pg.get(qn("w:start")) is not None:
                del pg.attrib[qn("w:start")]
            if landscape:
                sz = sp.find(qn("w:pgSz"))
                w, h = sz.get(qn("w:w")), sz.get(qn("w:h"))
                sz.set(qn("w:w"), h); sz.set(qn("w:h"), w); sz.set(qn("w:orient"), "landscape")
            return sp

        def break_para(sp):
            p = self.doc.add_paragraph()
            set_spacing(p, 0, 0, line=240)
            set_run_font(p.add_run(""), self.T.font, 2)
            _insert_ordered(pPr(p), sp, _PPR_ORDER)
            return p

        break_para(sect_copy(False))
        self.sect_started = True
        sz = tmpl.find(qn("w:pgSz")); mar = tmpl.find(qn("w:pgMar"))
        land_width = int(sz.get(qn("w:h"))) - int(mar.get(qn("w:left"))) - int(mar.get(qn("w:right")))
        self.width_stack.append(land_width)
        self.render_blocks(b["blocks"], container)
        self.width_stack.pop()
        break_para(sect_copy(True))

    def _quote(self, container, b: dict):
        T = self.T
        paras = [x for x in b["blocks"] if x["type"] == "para"]
        if not paras:
            return self.render_blocks(b["blocks"], container)
        for i, x in enumerate(paras):
            p = self.para(container, x["inline"], style="Quote")
            set_indent(p, left=300, right=600)
            set_spacing(p, 160 if i == 0 else 0, 200 if i == len(paras) - 1 else 80)
            set_borders(pPr(p), "pBdr", _PPR_ORDER, left=(18, T.blue, 14))
            set_keep_lines(p)


# =========================================================================== page furniture
def _page_setup(sec, T: Theme, meta: dict) -> tuple[int, float]:
    page = str(meta.get("page") or meta.get("paper") or "letter").lower()
    if page in ("a4", "a-4"):
        sec.page_width, sec.page_height = Inches(8.27), Inches(11.69)
    else:
        sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    sec.left_margin = sec.right_margin = Inches(T.margin_lr)
    sec.top_margin = Inches(T.margin_top); sec.bottom_margin = Inches(T.margin_bottom)
    sec.header_distance = Inches(T.header_dist); sec.footer_distance = Inches(T.footer_dist)
    width = int((sec.page_width - sec.left_margin - sec.right_margin) / 635)   # EMU -> twips
    return width, sec.page_height / 914400


def _set_pgnum(sectPr, fmt: str | None, start: int | None):
    pg = _el("w:pgNumType", fmt=fmt, start=start)
    _insert_ordered(sectPr, pg, _SECT_ORDER)


def _short_title(meta: dict) -> str:
    for k in ("short_title", "running_title", "project", "title"):
        if meta.get(k):
            return str(meta[k]).strip()
    return ""


def _hf_table(hf, T: Theme, bottom_rule: bool):
    """A one-row, two-cell, full-width (percentage) borderless table in a header/footer part: it
    follows each section's text width, so landscape pages keep the right-hand item at the margin."""
    tbl_xml = (
        f'<w:tbl {nsdecls("w")}><w:tblPr><w:tblW w:w="5000" w:type="pct"/><w:jc w:val="left"/>'
        '<w:tblBorders><w:top w:val="nil"/><w:left w:val="nil"/>'
        + (f'<w:bottom w:val="single" w:sz="4" w:space="0" w:color="{T.hairline}"/>' if bottom_rule else '<w:bottom w:val="nil"/>')
        + '<w:right w:val="nil"/><w:insideH w:val="nil"/><w:insideV w:val="nil"/></w:tblBorders>'
        '<w:tblLayout w:type="autofit"/><w:tblCellMar><w:top w:w="0" w:type="dxa"/><w:left w:w="0" w:type="dxa"/>'
        '<w:bottom w:w="70" w:type="dxa"/><w:right w:w="0" w:type="dxa"/></w:tblCellMar>'
        '<w:tblLook w:val="0000" w:firstRow="0" w:lastRow="0" w:firstColumn="0" w:lastColumn="0" w:noHBand="1" w:noVBand="1"/>'
        '</w:tblPr><w:tblGrid><w:gridCol w:w="4680"/><w:gridCol w:w="4680"/></w:tblGrid>'
        '<w:tr><w:tc><w:tcPr><w:tcW w:w="2500" w:type="pct"/><w:vAlign w:val="center"/></w:tcPr><w:p/></w:tc>'
        '<w:tc><w:tcPr><w:tcW w:w="2500" w:type="pct"/><w:vAlign w:val="center"/></w:tcPr><w:p><w:pPr><w:jc w:val="right"/></w:pPr></w:p></w:tc>'
        '</w:tr></w:tbl>')
    tbl = parse_xml(tbl_xml)
    first = hf._element.find(qn("w:p"))
    hf._element.insert(0, tbl)
    if first is not None:                  # the part must end with a paragraph: keep a 1 pt empty one
        set_spacing(Paragraph(first, hf), 0, 0, line=20, rule="exact")
    cells = tbl.findall(".//" + qn("w:tc"))
    left = Paragraph(cells[0].find(qn("w:p")), hf)
    right = Paragraph(cells[1].find(qn("w:p")), hf)
    for q in (left, right):
        set_style(q, "Header")
        set_spacing(q, 0, 0, line=240)
    return left, right


def _running_header(hdr, T: Theme, meta: dict, width: int):
    """Small logo lockup left, short title right, hairline below (full width on every page shape)."""
    left, right = _hf_table(hdr, T, bottom_rule=True)
    title = _short_title(meta)
    if T.logo:
        left.add_run().add_picture(asset(T, T.logo), height=Inches(0.27))
    if title:                              # no logo: the short title sits on the left
        add_text_run(right if T.logo else left, title, T, T.header_pt, T.muted)


def _running_footer(ftr, T: Theme, width: int):
    left, right = _hf_table(ftr, T, bottom_rule=False)
    if T.footer_text:
        add_text_run(left, T.footer_text, T, T.header_pt, T.muted)
    add_field(right, "PAGE", "1", T, T.header_pt + 0.5, T.navy, True)


# =========================================================================== contents (TOC / LOF / LOT)
_MARKER = "AEITOCENDMARKER"


def _toc_paragraphs(doc, T: Theme, rows: list[dict], instr: str, style_of) -> tuple[list, list]:
    """Build the cached entries of a TOC-type field. rows: {"level", "num", "title", "bookmark"}.
    Returns (paragraph elements, fill targets [{"run": w:r of the PAGEREF result, ...}])."""
    paras, targets = [], []
    head, tail = _field_runs(instr)
    for i, row in enumerate(rows):
        p = doc.add_paragraph(style=doc.styles[style_of(row)])
        if i == 0:
            for r in head:
                p._p.append(r)
        h = OxmlElement("w:hyperlink"); h.set(qn("w:anchor"), row["bookmark"]); h.set(qn("w:history"), "1")
        def run(text=None, tab=False):
            r = OxmlElement("w:r")
            if tab:
                r.append(OxmlElement("w:tab"))
            if text is not None:
                t = OxmlElement("w:t"); t.set(qn("xml:space"), "preserve"); t.text = text; r.append(t)
            h.append(r)
            return r
        if row.get("num"):
            run(row["num"]); run(tab=True)
        run(row["title"]); run(tab=True)
        ph, pt = _field_runs(f"PAGEREF {row['bookmark']} \\h")
        for r in ph:
            h.append(r)
        res = run("")
        h.append(pt[0])
        p._p.append(h)
        if i == len(rows) - 1:
            p._p.append(tail[0])
        paras.append(p._p)
        targets.append({"run": res, "search": row.get("search") or row["title"]})
    return paras, targets


def build_contents(doc, T: Theme, R: Renderer, anchor, lof_anchor=None, lot_anchor=None) -> list[dict]:
    """Insert TOC (levels 1-2) before `anchor`, and optional lists of figures/tables before their
    anchors. Returns page-number fill targets in document order."""
    entries: list[dict] = []
    if R.toc:
        paras, targets = _toc_paragraphs(doc, T, R.toc, 'TOC \\o "1-2" \\h \\z \\u',
                                         lambda row: "toc 1" if row["level"] == 1 else "toc 2")
        for el in paras:
            anchor.addprevious(el)
        for t in targets:
            t["kind"] = "toc"
        entries += targets
    for kind, anc, style_name in (("figure", lof_anchor, "Caption"), ("table", lot_anchor, "Table Caption")):
        if anc is None:
            continue
        caps = [c for c in R.captions if c["kind"] == kind]
        title_p = anc.getprevious()
        if not caps:
            if title_p is not None and para_text(title_p).strip() in (T.lof_title, T.lot_title):
                title_p.getparent().remove(title_p)
            anc.getparent().remove(anc)
            continue
        rows = [{"level": 1, "num": "", "title": c["text"], "bookmark": c["bookmark"], "search": c["search"]} for c in caps]
        paras, targets = _toc_paragraphs(doc, T, rows, f'TOC \\h \\z \\t "{style_name},1"', lambda row: "table of figures")
        for el in paras:
            anc.addprevious(el)
        for t in targets:
            t["kind"] = "lof" if kind == "figure" else "lot"
        entries += targets
    return entries


# =========================================================================== page-number fill (two-pass)
_PAGE_LABEL_RE = re.compile(r"^(?:[ivxlc]+|\d{1,4}|[A-Z][.-]\d{1,3})$")


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u2011", "-").replace("–", "-").replace("—", "-").replace("’", "'").replace("‘", "'")
    s = s.replace("“", '"').replace("”", '"').replace("\u2002", " ").replace("\u00a0", " ")
    # a title wrapped at its hyphen ("economy-\nwide") reads as one word again
    return re.sub(r"(?<=\w)- (?=\w)", "-", re.sub(r"\s+", " ", s).strip().lower())


def fill_page_numbers(entries: list[dict], pages_text: list[str]) -> int:
    """Write page labels (read from each page's footer) into the cached TOC/LOF/LOT entries.
    Entries are searched in document order, only after the invisible end-of-contents marker, so the
    contents pages themselves never match."""
    labels = []
    for txt in pages_text:
        lines = [l.strip() for l in txt.splitlines() if l.strip()]
        label = ""
        for l in reversed(lines[-3:]):
            tok = l.split()[-1] if l.split() else ""
            if _PAGE_LABEL_RE.match(tok):
                label = tok; break
        labels.append(label)
    norm_pages = [_norm(t) for t in pages_text]
    start = 0
    mk = _MARKER.lower()
    for i, t in enumerate(norm_pages):
        if mk in t:
            start = i
            norm_pages[i] = t.split(mk, 1)[1]
    for i in range(start):
        norm_pages[i] = ""
    filled = 0
    last = {"toc": start, "lof": start, "lot": start}
    for e in entries:
        key = _norm(e["search"])[:70]
        kind = e.get("kind", "toc")
        hit = None
        for k in (key, key[:36]):
            if not k:
                continue
            for i in range(last[kind], len(norm_pages)):
                if k in norm_pages[i]:
                    hit = i; break
            if hit is not None:
                break
        if hit is None:
            continue
        if k == key:           # a match on the shortened key may sit far on: it moves nothing
            last[kind] = hit
        if labels[hit]:
            r = e["run"]
            t = r.find(qn("w:t"))
            if t is None:
                t = OxmlElement("w:t"); r.append(t)
            t.text = labels[hit]
            filled += 1
    return filled


def _remove_marker(doc):
    for t in doc.element.body.iter(qn("w:t")):
        if t.text and _MARKER in t.text:
            r = t.getparent()
            r.getparent().remove(r)
            return


def _marker_para(doc, T: Theme, after: int = 0):
    p = doc.add_paragraph()
    set_spacing(p, 0, after, line=240)
    r = p.add_run(_MARKER)
    set_run_font(r, T.font, 1, T.white)
    return p


# =========================================================================== document structure
def _normalize_levels(blocks: list[dict], meta: dict) -> list[dict]:
    """Agent-written Markdown often uses '# Title' + '## Sections' or only '##': promote so that
    sections become chapters (report) / numbered sections (note)."""
    heads = [b for b in blocks if b["type"] == "heading"]
    h1 = [b for b in heads if b["level"] == 1]
    h2 = [b for b in heads if b["level"] == 2]
    if not h2:
        return blocks
    promote = False
    if not h1:
        promote = True
    elif len(h1) == 1 and blocks and blocks[0] is h1[0]:
        title = inline_text(h1[0]["inline"]).strip()
        if not meta.get("title"):
            meta["title"] = title
        blocks = blocks[1:]
        promote = True
    if promote:
        def up(bl):
            for b in bl:
                if b["type"] == "heading" and b["level"] > 1:
                    b["level"] -= 1
                elif b["type"] == "box":
                    up(b["blocks"])
        up(blocks)
    return blocks


def _split_report(blocks: list[dict], T: Theme) -> tuple[list, list, list]:
    """front matter [(heading, blocks)], chapters, annexes (cpcs_report rules)."""
    front: list = []; chapters: list = []; annexes: list = []
    cur = None; where = "pre"
    names = set(T.summary_names)
    for b in blocks:
        if b["type"] == "heading" and b["level"] == 1:
            title = inline_text(b["inline"]).strip().lower().rstrip(".:")
            if b.get("appendix"):
                where = "app"; cur = (b, []); annexes.append(cur)
            elif where == "app":
                cur = (b, []); annexes.append(cur)
            elif (not b.get("numbered", True) or title in names) and not chapters:
                where = "front"; cur = (b, []); front.append(cur)
            else:
                where = "ch"; cur = (b, []); chapters.append(cur)
            continue
        if cur is None:
            if b["type"] == "para" and not inline_text(b["inline"]).strip():
                continue
            if not front:
                front.append(({"type": "heading", "level": 1, "inline": parse_inline(T.summary_title),
                               "numbered": False, "appendix": False, "id": None, "kicker": None}, []))
            front[0][1].append(b)
        else:
            cur[1].append(b)
    return front, chapters, annexes


def _acronym_block(meta: dict, T: Theme) -> dict | None:
    acr = meta.get("acronyms") or meta.get("abbreviations")
    if not acr:
        return None
    items = list(acr.items()) if isinstance(acr, dict) else [tuple(str(x).split(":", 1)) if isinstance(x, str) else tuple(x) for x in acr]
    items = sorted(((str(a).strip(), str(b).strip()) for a, b in items if b is not None), key=lambda x: x[0].lower())
    return {"type": "table", "header": [parse_inline("Abbreviation"), parse_inline("Meaning")],
            "rows": [[parse_inline(a), parse_inline(b)] for a, b in items], "align": ["left", "left"],
            "caption": None, "source": None, "width": 5.2, "id": None}


def _cover(doc, R: Renderer, T: Theme, meta: dict, sec, width: int, page_w_in: float, page_h_in: float):
    """Report cover: logo, kicker, title, subtitle, short rule, lead; optional full-bleed image band;
    date and 'Prepared by' at the foot (in the cover section's own footer). A theme without a logo
    but with `cover_band` gets a typographic cover: the title block set in white on a full-bleed
    colour panel that runs from the top of the page to below the lead (or down to the image band)."""
    p = doc.paragraphs[0] if doc.paragraphs else doc.add_paragraph()
    set_spacing(p, 0, 0, line=240)
    if T.logo:
        p.add_run().add_picture(asset(T, T.logo), height=Inches(0.6))
    # optional image band, anchored above the footer strip, full page width; without an image a
    # large pale brand mark bleeds off the lower-right corner so the page stays balanced
    img = meta.get("cover_image")
    band_h = float(meta.get("cover_image_height") or 3.4)
    foot = 1.25                                   # white strip for the date / prepared-by footer
    text_h = _cover_text_height(T, meta, width)   # logo + kicker + title + subtitle + rule + lead (inches)
    space_before = 1.45
    placed = False
    panel_bottom = None
    path = R.resolve(str(img)) if img else None
    if img and path is None:
        R.warnings.append(f"missing cover image: {img}")
    if path:
        band_top = page_h_in - foot - band_h
        need = T.margin_top + text_h + 0.35      # text bottom + gap, with the default space above the kicker
        space_before = max(0.55, 1.15 - max(0.0, need + 1.15 - band_top))
        text_bottom = T.margin_top + text_h + space_before + 0.35
        if text_bottom > band_top:                # long title/lead: shorten the band instead
            band_top = text_bottom
            band_h = page_h_in - foot - band_top
        if band_h < 1.5:
            R.warnings.append("cover image dropped: the title block leaves no room for it (shorten the lead)")
        else:
            try:
                stream = _band_image(path, page_w_in / band_h, str(meta.get("cover_image_crop") or "center"))
                shape = p.add_run().add_picture(stream, width=Inches(page_w_in))
                y = (page_h_in - foot - band_h) * 914400
                _to_anchor(shape, 0, y, wrap="topAndBottom")
                placed = True
                panel_bottom = page_h_in - foot - band_h
            except Exception as e:  # noqa: BLE001
                R.warnings.append(f"cover image could not be used ({img}): {e}")
    if not placed and T.mark:
        try:
            mark_h = 6.2
            shape = p.add_run().add_picture(_mark_graphic(T, T.cover_mark), height=Inches(mark_h))
            mark_w = shape.width / 914400
            top = max(page_h_in - mark_h * 0.86, T.margin_top + text_h + space_before + 0.3)
            _to_anchor(shape, (page_w_in - mark_w * 0.74) * 914400, top * 914400, wrap="none", behind=True)
        except Exception as e:  # noqa: BLE001
            R.warnings.append(f"cover mark could not be drawn: {e}")
    on_band = None                                # text colours on the panel: kicker, title, subtitle, lead
    if T.cover_band:
        if panel_bottom is None:
            panel_bottom = max(T.margin_top + text_h + space_before + 0.6, 0.46 * page_h_in)
        try:
            shape = p.add_run().add_picture(_panel_graphic(T, page_w_in, panel_bottom), width=Inches(page_w_in))
            _to_anchor(shape, 0, 0, wrap="none", behind=True)
            on_band = (T.blue, T.white, T.sky, T.white)
        except Exception as e:  # noqa: BLE001
            R.warnings.append(f"cover panel could not be drawn: {e}")
    kicker = meta.get("doc_label") or meta.get("kicker") or meta.get("doc_type")
    k = doc.add_paragraph(style=doc.styles[style_name(T, "Kicker")])
    set_spacing(k, before=int(space_before * 1440), after=160)
    if kicker:
        add_text_run(k, _upper(str(kicker)), T, T.cover_kicker_pt, on_band[0] if on_band else None)
    t = doc.add_paragraph(style=doc.styles["Title"])
    set_keep_next(t)
    add_inline_runs(t, parse_inline(str(meta.get("title") or "Untitled report")), T,
                    color=on_band[1] if on_band else None)
    if meta.get("subtitle"):
        s = doc.add_paragraph(style=doc.styles["Subtitle"])
        add_inline_runs(s, parse_inline(str(meta["subtitle"])), T, color=on_band[2] if on_band else None)
        set_spacing(s, after=0)
    R.short_rule(doc, width_in=1.9, before=260, after=300, sz=16)
    lead = meta.get("lead") or meta.get("abstract")
    if lead:
        lp = doc.add_paragraph(style=doc.styles[style_name(T, "Lead")])
        set_indent(lp, right=int(0.9 * 1440))
        add_inline_runs(lp, parse_inline(str(lead).strip()), T, color=on_band[3] if on_band else None)
    # footer strip: date, prepared for / by
    f = sec.footer
    f.is_linked_to_previous = False
    fp = f.paragraphs[0]
    add_text_run(fp, fmt_date(meta.get("date"), month_year=True), T, 10.5, T.navy, True)
    set_spacing(fp, 0, 30, line=240)
    if meta.get("client"):
        q = f.add_paragraph()
        add_text_run(q, f"{T.prepared_for} {meta['client']}", T, 9, T.slate)
        set_spacing(q, 0, 10, line=240)
    if T.prepared_by:
        q = f.add_paragraph()
        add_text_run(q, T.prepared_by, T, 9, T.muted)
        set_spacing(q, 0, 0, line=240)
    # nothing in the cover header
    sec.header.is_linked_to_previous = False


def _wrap_lines(text: str, size_pt: float, bold: bool, width_twips: int) -> int:
    """Greedy line count of `text` set at size_pt in a column width_twips wide."""
    words = str(text).split()
    if not words:
        return 0
    space = text_width_twips(" ", size_pt, bold)
    lines, cur = 1, 0
    for w in words:
        ww = text_width_twips(w, size_pt, bold)
        if cur == 0:
            cur = ww
        elif cur + space + ww <= width_twips:
            cur += space + ww
        else:
            lines += 1; cur = ww
    return lines


def _cover_text_height(T: Theme, meta: dict, width: int) -> float:
    """Height (in) of the cover's logo, kicker, title, subtitle, rule and lead, without the space above
    the kicker. Inter's natural line height is ~1.21 em."""
    lh = 1.21 / 72.0
    title = inline_text(parse_inline(str(meta.get("title") or "Untitled report")))
    h = (0.62 if T.logo else 0.17) + (T.cover_kicker_pt * lh + 160 / 1440)
    h += _wrap_lines(title, T.cover_title_pt, True, width) * T.cover_title_pt * lh * 250 / 240 + 8 / 72
    if meta.get("subtitle"):
        sub = inline_text(parse_inline(str(meta["subtitle"])))
        h += _wrap_lines(sub, T.cover_sub_pt, True, width) * T.cover_sub_pt * lh * 260 / 240
    h += (260 + 120 + 300) / 1440
    lead = meta.get("lead") or meta.get("abstract")
    if lead:
        h += _wrap_lines(inline_text(parse_inline(str(lead))), T.cover_lead_pt, False, width - int(0.9 * 1440)) \
            * T.cover_lead_pt * lh * 300 / 240
    return h


def _mark_graphic(T: Theme, color: str, height_px: int = 1800):
    """The brand mark recoloured (e.g. a pale tint for the cover watermark), as a PNG stream."""
    from PIL import Image
    with Image.open(asset(T, T.mark)) as im:
        alpha = im.convert("RGBA").getchannel("A")
    w = int(alpha.size[0] * height_px / alpha.size[1])
    alpha = alpha.resize((w, height_px), Image.LANCZOS)
    rgb = tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))
    out = Image.new("RGBA", alpha.size, rgb + (0,))
    out.putalpha(alpha)
    buf = io.BytesIO(); out.save(buf, "PNG", optimize=True); buf.seek(0)
    return buf


def _panel_graphic(T: Theme, width_in: float, height_in: float, dpi: int = 60):
    """A flat cover panel in T.cover_band with a thin T.cover_band_accent strip along its lower edge."""
    from PIL import Image, ImageDraw
    w, h = max(1, int(width_in * dpi)), max(1, int(height_in * dpi))
    rgb = lambda c: tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))
    im = Image.new("RGB", (w, h), rgb(T.cover_band))
    if T.cover_band_accent:
        strip = max(2, int(0.07 * dpi))
        ImageDraw.Draw(im).rectangle([0, h - strip, w, h], fill=rgb(T.cover_band_accent))
    buf = io.BytesIO(); im.save(buf, "PNG", optimize=True); buf.seek(0)
    return buf


def _band_image(path: str, aspect: float, crop: str = "center"):
    """Crop an image to a wide band of the given aspect ratio and cap its size."""
    from PIL import Image
    with Image.open(path) as im:
        im.load()
        img = im.convert("RGB")
    w, h = img.size
    if w / h > aspect:                      # too wide: crop the sides
        nw = int(h * aspect); x0 = (w - nw) // 2
        box = (x0, 0, x0 + nw, h)
    else:                                   # too tall: crop top/bottom
        nh = int(w / aspect)
        y0 = {"top": 0, "bottom": h - nh}.get(crop, (h - nh) // 2)
        box = (0, y0, w, y0 + nh)
    img = img.crop(box)
    if img.size[0] > _MAX_PX:
        img = img.resize((_MAX_PX, int(_MAX_PX / aspect)), Image.LANCZOS)
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=88, optimize=True); buf.seek(0)
    return buf


def _build_report(doc, T: Theme, meta: dict, blocks: list[dict], fn_defs: dict, base_dir: str, paginate: bool):
    sec = doc.sections[0]
    width, page_h = _page_setup(sec, T, meta)
    page_w = sec.page_width / 914400
    define_styles(doc, T, "report", width)
    fn = Footnotes(doc, fn_defs, T)
    R = Renderer(doc, T, "report", meta, fn, base_dir, width)
    R.page_height_in = page_h
    R.num.attach_styles()
    sequential = str(meta.get("figure_numbering") or T.figure_numbering).lower().startswith("seq")

    front, chapters, annexes = _split_report(blocks, T)
    acr = _acronym_block(meta, T)
    if acr:
        front.insert(0, ({"type": "heading", "level": 1, "inline": parse_inline(T.acronyms_title), "numbered": False,
                          "appendix": False, "id": None, "kicker": None, "_acronyms": True}, [acr]))
    parts = [{"kind": "front", "prefix": "", "heading": hb, "blocks": bl} for hb, bl in front]
    parts += [{"kind": "chapter", "prefix": str(i), "heading": hb, "blocks": bl} for i, (hb, bl) in enumerate(chapters, 1)]
    parts += [{"kind": "annex", "prefix": chr(ord("A") + i), "heading": hb, "blocks": bl} for i, (hb, bl) in enumerate(annexes)]
    labels: dict = {}
    assign_numbers(parts, T, sequential, True, labels)
    for pt in parts:
        resolve_refs(pt["blocks"], labels, R.warnings)
        resolve_refs([pt["heading"]], labels, R.warnings)

    # ---- S1: cover
    _cover(doc, R, T, meta, sec, width, page_w, page_h)
    # ---- S2: contents + front matter (roman page numbers)
    doc.add_section(WD_SECTION.NEW_PAGE)
    s2 = doc.sections[-1]
    s2.header.is_linked_to_previous = False
    s2.footer.is_linked_to_previous = False
    _running_header(s2.header, T, meta, width)
    _running_footer(s2.footer, T, width)
    _set_pgnum(s2._sectPr, "lowerRoman", 1)
    ct = doc.add_paragraph(style=doc.styles[style_name(T, "Contents Title")])
    add_text_run(ct, T.contents_title, T)
    R.short_rule(doc, width_in=1.1, before=40, after=300, sz=18)
    toc_anchor = doc.add_paragraph(); set_spacing(toc_anchor, 0, 0, line=240)
    lof_anchor = lot_anchor = None
    if meta.get("lof") or meta.get("list_of_figures"):
        tp = doc.add_paragraph(style=doc.styles[style_name(T, "Contents Subtitle")]); add_text_run(tp, T.lof_title, T)
        lof_anchor = doc.add_paragraph()._p
    if meta.get("lot") or meta.get("list_of_tables"):
        tp = doc.add_paragraph(style=doc.styles[style_name(T, "Contents Subtitle")]); add_text_run(tp, T.lot_title, T)
        lot_anchor = doc.add_paragraph()._p
    marker = _marker_para(doc, T) if paginate else None
    default_front_kicker = meta.get("doc_label") or meta.get("kicker")
    R.sect_template = doc.sections[-1]._sectPr
    R.sect_started = False
    first_front = next((j for j, (hb, _) in enumerate(front) if not hb.get("_acronyms")), None)
    for i, (hb, bl) in enumerate(front):
        kick = hb.get("kicker") or (default_front_kicker if i == first_front else None) or None
        R.opener(doc, hb["inline"], str(kick) if kick else None, "", page_break=True)
        R.render_blocks(bl, doc)
    if R.sect_started:
        body_sect = doc.element.body.find(qn("w:sectPr"))
        pg = body_sect.find(qn("w:pgNumType"))
        if pg is not None and pg.get(qn("w:start")):
            del pg.attrib[qn("w:start")]
        absorb_trailing_landscape(body_sect)
    # ---- S3: chapters and annexes (arabic page numbers from 1)
    if chapters or annexes:
        doc.add_section(WD_SECTION.NEW_PAGE)
        _set_pgnum(doc.sections[-1]._sectPr, "decimal", 1)
        R.sect_template = doc.sections[-1]._sectPr
        R.sect_started = False
        for i, (hb, bl) in enumerate(chapters, 1):
            kick = hb.get("kicker") or f"{T.chapter_word} {i}"
            R.opener(doc, hb["inline"], str(kick), str(i), page_break=i > 1)
            R.render_blocks(bl, doc)
        for i, (hb, bl) in enumerate(annexes):
            letter = chr(ord("A") + i)
            raw = str(hb.get("raw") or inline_text(hb["inline"])).strip()
            clean = re.sub(r"^(?:Appendix|Annexe?)\b\s*(?:[A-Z]\b)?\s*[.:–—-]?\s*", "", raw, flags=re.I).strip()
            title_inline = parse_inline(clean) if clean and clean != raw else hb["inline"]
            kick = hb.get("kicker") or f"{T.annex_word} {letter}"
            R.opener(doc, title_inline, str(kick), letter, page_break=bool(chapters) or i > 0)
            R.render_blocks(bl, doc)
        if R.sect_started:
            body_sect = doc.element.body.find(qn("w:sectPr"))
            pg = body_sect.find(qn("w:pgNumType"))
            if pg is not None and pg.get(qn("w:start")):
                del pg.attrib[qn("w:start")]
            absorb_trailing_landscape(body_sect)

    entries = build_contents(doc, T, R, toc_anchor._p, lof_anchor, lot_anchor)
    toc_anchor._p.getparent().remove(toc_anchor._p)
    fn.finalize()
    return R, entries, marker


def _build_note(doc, T: Theme, meta: dict, blocks: list[dict], fn_defs: dict, base_dir: str, paginate: bool):
    sec = doc.sections[0]
    width, page_h = _page_setup(sec, T, meta)
    define_styles(doc, T, "note", width)
    fn = Footnotes(doc, fn_defs, T)
    R = Renderer(doc, T, "note", meta, fn, base_dir, width)
    R.page_height_in = page_h
    R.num.attach_styles()
    numbered = meta.get("numbered", True)
    numbered = not (str(numbered).lower() in ("false", "no", "0"))
    labels: dict = {}
    parts = [{"kind": "note", "prefix": "", "heading": None, "blocks": blocks}]
    assign_numbers(parts, T, True, numbered, labels)
    resolve_refs(blocks, labels, R.warnings)

    # header/footer: first page carries a larger logo, later pages the running header
    sec.different_first_page_header_footer = True
    fh = sec.first_page_header.paragraphs[0]
    if T.logo:
        fh.add_run().add_picture(asset(T, T.logo), height=Inches(0.46))
    add_tabs(fh, [(width, "right", None)])
    fh.add_run("\t")
    add_text_run(fh, fmt_date(meta.get("date")), T, T.header_pt + 0.5, T.muted, position=12)
    set_spacing(fh, 0, 0, line=240)
    _running_header(sec.header, T, meta, width)
    _running_footer(sec.footer, T, width)
    _running_footer(sec.first_page_footer, T, width)

    # title block
    kicker = meta.get("doc_label") or meta.get("kicker") or meta.get("doc_type")
    if kicker:
        k = doc.add_paragraph(style=doc.styles[style_name(T, "Kicker")])
        set_spacing(k, before=160, after=120)
        add_text_run(k, _upper(str(kicker)), T)
    t = doc.add_paragraph(style=doc.styles["Title"])
    add_inline_runs(t, parse_inline(str(meta.get("title") or "Untitled note")), T)
    if not kicker:
        set_spacing(t, before=160)
    if meta.get("subtitle"):
        s = doc.add_paragraph(style=doc.styles["Subtitle"])
        add_inline_runs(s, parse_inline(str(meta["subtitle"])), T)
    R.short_rule(doc, width_in=1.4, before=120, after=220, sz=14)
    lead = meta.get("lead") or meta.get("abstract")
    if lead:
        lp = doc.add_paragraph(style=doc.styles[style_name(T, "Lead")])
        add_inline_runs(lp, parse_inline(str(lead).strip()), T)
        set_spacing(lp, after=200)
    marker = None
    toc_anchor = None
    if meta.get("toc"):
        ct = doc.add_paragraph(style=doc.styles[style_name(T, "Contents Title")])
        add_text_run(ct, T.contents_title, T)
        set_spacing(ct, before=80)
        toc_anchor = doc.add_paragraph(); set_spacing(toc_anchor, 0, 0, line=240)
        marker = _marker_para(doc, T, after=120) if paginate else _spacer(doc, T, 120)
    R.sect_template = sec._sectPr
    R.render_blocks(blocks, doc)
    acr = _acronym_block(meta, T)
    if acr:
        R.heading(doc, {"type": "heading", "level": 1, "inline": parse_inline(T.acronyms_title), "numbered": False,
                        "_num": ""})
        R.table(doc, acr)
    absorb_trailing_landscape(doc.element.body.find(qn("w:sectPr")))
    entries = []
    if toc_anchor is not None:
        entries = build_contents(doc, T, R, toc_anchor._p)
        toc_anchor._p.getparent().remove(toc_anchor._p)
    fn.finalize()
    return R, entries, marker


# =========================================================================== public API
def _read_source(source) -> tuple[str, Path | None]:
    if isinstance(source, Path):
        return source.read_text(encoding="utf-8"), source.resolve().parent
    s = str(source)
    if "\n" not in s and len(s) < 4096:
        try:
            p = Path(s).expanduser()
            if p.is_file():
                return p.read_text(encoding="utf-8"), p.resolve().parent
        except OSError:
            pass
    return s, None


def build_document(source: str | Path, out: str | Path, *, meta: dict | None = None,
                   base_dir: str | Path | None = None, paginate: bool = True,
                   soffice_timeout: int = 180) -> dict:
    """Build an AEI Labs Word document from Markdown.

    source: Markdown text or a path to a .md file (front matter optional); meta overrides
    front-matter keys (title, subtitle, type, doc_label, lead, date, page, cover_image, ...);
    image paths may be absolute; base_dir resolves relative ones (default: the .md's folder,
    or the current directory for Markdown text). `type` is "report" or "note" (default "note").
    paginate: when the document has a contents list and LibreOffice is available, render it once
    and write the real page numbers into the cached contents entries (Word refreshes them on
    open either way). Never prints and never raises for content problems (missing figures,
    unresolved references ...): those come back as warnings.
    meta["style"] (or front-matter `style:`) picks the house style: "aeilabs" (default) or
    "worldbank"; an unknown style falls back to AEI Labs with a warning.
    Returns {"warnings": [...], "figures": int, "tables": int, "pages": int | None,
             "paginated": bool, "seconds": float, "style": theme key used}."""
    t0 = time.perf_counter()
    text, src_dir = _read_source(source)
    fm, blocks, footnotes = md_parser.parse_markdown(text)
    m = dict(fm)
    m.update({k: v for k, v in (meta or {}).items() if v is not None})
    base = str(Path(base_dir).expanduser().resolve()) if base_dir else str(src_dir or Path.cwd())
    T, style_warning = get_theme(m.get("style") or m.get("theme"))
    warnings: list[str] = [style_warning] if style_warning else []
    mode = str(m.get("type") or "note").strip().lower()
    if mode not in ("note", "report"):
        warnings.append(f"unknown type '{mode}': building a note")
        mode = "note"
    blocks = _normalize_levels(blocks, m)
    out = Path(out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    doc = Document()
    _prepare_package(doc, T)
    builder = _build_report if mode == "report" else _build_note
    want_pages = bool(paginate) and render_docx.have_soffice()
    R, entries, marker = builder(doc, T, m, blocks, footnotes, base, want_pages)
    _strip_theme_fonts(doc.element, T.font)
    doc.core_properties.title = str(m.get("title") or "")
    doc.core_properties.subject = str(m.get("subtitle") or "")
    doc.core_properties.author = T.org
    doc.core_properties.last_modified_by = T.org
    doc.core_properties.comments = ""
    warnings += R.warnings

    pages = None
    paginated = False
    if entries and want_pages:
        doc.save(str(out))
        try:
            with tempfile.TemporaryDirectory(prefix="aei-paginate-") as td:
                pdf = render_docx.docx_to_pdf(str(out), td, timeout=soffice_timeout)
                pages_text = render_docx.pdf_pages_text(pdf)
            n = fill_page_numbers(entries, pages_text)
            pages = len(pages_text)
            paginated = True
            if n < len(entries):
                warnings.append(f"contents: {len(entries) - n} of {len(entries)} page numbers not found "
                                f"(Word fills them when it updates fields)")
        except Exception as e:  # noqa: BLE001 - LibreOffice/poppler trouble must not fail the build
            warnings.append(f"contents page numbers not filled ({e}); Word fills them when it updates fields")
    elif entries and paginate:
        warnings.append("LibreOffice not found: contents page numbers are filled by Word when it updates fields")
    if marker is not None:
        _remove_marker(doc)
    doc.save(str(out))
    return {"warnings": warnings, "figures": R.figures, "tables": R.tables, "pages": pages,
            "paginated": paginated, "seconds": round(time.perf_counter() - t0, 2), "style": T.key}


if __name__ == "__main__":  # minimal CLI; see build_aeilabs_doc.py for previews and options
    import json
    import sys
    if len(sys.argv) < 3:
        sys.exit("usage: aeilabs_docx.py input.md output.docx")
    print(json.dumps(build_document(sys.argv[1], sys.argv[2]), indent=1))
