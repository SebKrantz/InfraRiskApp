"""
md_parser.py — parse the report Markdown dialect into a list of block dicts.

Copied from ~/.claude/skills/cpcs-report/scripts/md_parser.py (version of 2026-09-08) so that
documents move between the cpcs-report and aeilabs-report skills unchanged. The parser is
brand-agnostic. Changes in this copy (all backwards compatible):
  - heading attributes of the form key="value" (e.g. {kicker="Freight platform"}) are kept on
    the heading block as `attrs`, and `kicker` is exposed directly;
  - attribute values may also use single or curly quotes;
  - front matter parses without PyYAML (a small fallback for simple keys, lists and maps);
  - HTML comments may span several lines;
  - figure blocks keep all their {...} attributes (e.g. frame=false) as `attrs`;
  - a `::: pipeline` block is a flowchart: one line per stage, `||` between boxes side by
    side, `Title: detail` in a box (`;` starts another detail line).

The dialect is ordinary Markdown (headings, paragraphs, bullet/numbered lists, pipe tables,
images, fenced code, blockquotes, footnotes) plus a few conventions that map onto report
document elements — see references/markdown-syntax.md.  The parser is deliberately
line-based and forgiving: content converted from Word with pandoc (`-t gfm`) parses as is.

Block shapes produced (all dicts with a "type" key):
  heading   level:int, inline:list, numbered:bool, appendix:bool, id:str|None,
            kicker:str|None, attrs:dict
  para      inline:list
  list      items:[{level:int, ordered:bool, inline:list}]
  table     header:[inline], rows:[[inline]], align:[str], caption:inline|None,
            source:inline|None, widths:list|None, size:str|None
  figure    path:str, caption:inline|None, source:inline|None, width:float|None (inches), attrs:dict
  code      text:str, lang:str
  box       kind:str (takeaway|box|note|warning|kpis|landscape|quote|callout), title:inline|None,
            blocks:[...]
  pipeline  title:inline|None, stages:[[{title:inline, details:[inline]}]] (top to bottom; a stage
            of several boxes forks from the stage above and merges into the one below)
  pagebreak
  hr
Inline content is a list of runs: (text, {"bold","italic","code","sup","sub","link","footnote","br"}).
"""
from __future__ import annotations

import re
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

Inline = list[tuple[str, dict[str, Any]]]

# --------------------------------------------------------------------------- inline
_INLINE_RE = re.compile(
    r"(?P<esc>\\[\\`*_{}\[\]()#+\-.!~^$|<>])"
    r"|(?P<code>`+)(?P<codetext>.+?)(?P=code)"
    r"|(?P<fn>\[\^(?P<fnid>[^\]\s]+)\])"
    r"|(?P<link>\[(?P<ltext>[^\]]*)\]\((?P<lurl>[^)\s]+)(?:\s+\"[^\"]*\")?\))"
    r"|(?P<bold>\*\*(?P<btext>.+?)\*\*)"
    r"|(?P<bold2>__(?P<b2text>.+?)__)"
    r"|(?P<ital>(?<![\w*])\*(?!\s)(?P<itext>[^*]+?)(?<!\s)\*(?![\w*]))"
    r"|(?P<ital2>(?<![\w_])_(?!\s)(?P<i2text>[^_]+?)(?<!\s)_(?![\w_]))"
    r"|(?P<sup>\^(?P<stext>[^\s^]+?)\^)"
    r"|(?P<sub>~(?P<subtext>[^\s~]+?)~)"
    r"|(?P<br><br\s*/?>)"
    r"|(?P<ref>@(?P<refid>(?:fig|figure|tbl|tab|table|sec|section|box):[\w.:-]+))"
    r"|(?P<html><[^>]+>)"
    r"|(?P<ent>&(?:nbsp|amp|lt|gt|quot|#\d+);)",
    re.S,
)
_ENT = {"&nbsp;": " ", "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"'}


def parse_inline(text: str, base: dict | None = None) -> Inline:
    """Turn a Markdown string into a list of (text, fmt) runs."""
    base = dict(base or {})
    out: Inline = []
    pos = 0

    def emit(s: str, fmt: dict):
        if s:
            if out and out[-1][1] == fmt and "br" not in fmt:
                out[-1] = (out[-1][0] + s, fmt)
            else:
                out.append((s, fmt))

    for m in _INLINE_RE.finditer(text):
        emit(text[pos:m.start()], base)
        pos = m.end()
        k = m.lastgroup
        if m.group("esc"):
            emit(m.group("esc")[1], base)
        elif m.group("code"):
            emit(m.group("codetext"), {**base, "code": True})
        elif m.group("fn"):
            out.append(("", {**base, "footnote": m.group("fnid")}))
        elif m.group("link"):
            out.extend(parse_inline(m.group("ltext") or m.group("lurl"), {**base, "link": m.group("lurl")}))
        elif m.group("bold"):
            out.extend(parse_inline(m.group("btext"), {**base, "bold": True}))
        elif m.group("bold2"):
            out.extend(parse_inline(m.group("b2text"), {**base, "bold": True}))
        elif m.group("ital"):
            out.extend(parse_inline(m.group("itext"), {**base, "italic": True}))
        elif m.group("ital2"):
            out.extend(parse_inline(m.group("i2text"), {**base, "italic": True}))
        elif m.group("sup"):
            emit(m.group("stext"), {**base, "sup": True})
        elif m.group("sub"):
            emit(m.group("subtext"), {**base, "sub": True})
        elif m.group("br"):
            out.append(("", {**base, "br": True}))
        elif m.group("ref"):
            out.append(("", {**base, "ref": m.group("refid").rstrip(".,;:")}))
            trail = m.group("refid")[len(m.group("refid").rstrip(".,;:")):]
            if trail:
                emit(trail, base)
        elif m.group("html"):
            pass  # drop stray HTML tags
        elif m.group("ent"):
            e = m.group("ent")
            emit(_ENT.get(e, chr(int(e[2:-1])) if e.startswith("&#") else e), base)
    emit(text[pos:], base)
    return out


def inline_text(inline: Inline) -> str:
    return "".join(t for t, _ in inline)


# --------------------------------------------------------------------------- block regexes
_H_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*(?:\{([^}]*)\})?\s*#*\s*$")
_LIST_RE = re.compile(r"^(\s*)(?:([-*+•])|(\d+)[.)])\s+(.*)$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_IMG_MD_RE = re.compile(r"^\s*!\[(?P<alt>[^\]]*)\]\((?P<path>[^)\s]+)(?:\s+\"[^\"]*\")?\)\s*(?:\{(?P<attrs>[^}]*)\})?\s*$")
_IMG_HTML_RE = re.compile(r"^\s*<img\s[^>]*src=\"(?P<path>[^\"]+)\"[^>]*?(?:style=\"[^\"]*width:(?P<w>[\d.]+)(?P<unit>in|cm|px)[^\"]*\")?[^>]*/?>\s*$")
_FIG_CAP_RE = re.compile(r"^\s*(?:\*\*|\*|__|_)?\s*(?:Figure|Fig\.|Graphique|Carte|Map)\s*(?:[\dA-Z]+[-.\d]*)?\s*[.:]?\s*(?:\*\*|\*|__|_)?\s*[.:]?\s*(?P<cap>.*?)\s*$", re.I)
_TAB_CAP_RE = re.compile(r"^\s*(?:\*\*|\*|__|_)?\s*(?:Table|Tableau)\s*(?:[\dA-Z]+[-.\d]*)?\s*[.:]?\s*(?:\*\*|\*|__|_)?\s*[.:]?\s*(?P<cap>.*?)\s*$", re.I)
_SRC_RE = re.compile(r"^\s*(?:\*|_)?\s*(?P<kind>Sources?|Notes?|Note et source|Sources? et notes?)\s*:\s*(?P<txt>.*?)\s*(?:\*|_)?\s*$", re.I)
_FENCE_RE = re.compile(r"^(```+|~~~+)\s*(\w+)?\s*$")
_DIV_OPEN_RE = re.compile(r"^:::+\s*(?P<kind>[\w-]+)?\s*(?P<title>.*?)\s*$")
_DIV_CLOSE_RE = re.compile(r"^:::+\s*$")
_FN_DEF_RE = re.compile(r"^\[\^([^\]\s]+)\]:\s*(.*)$")
_PAGEBREAK_RE = re.compile(r"^\s*(\\newpage|\\pagebreak|<!--\s*pagebreak\s*-->|<div[^>]*page-break[^>]*>\s*</div>)\s*$", re.I)
_HR_RE = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")
_COMMENT_RE = re.compile(r"^\s*<!--.*?-->\s*$")
_CAPTION_LINE_RE = re.compile(r"^\s*(?:Table|Figure)\s*:\s*(?P<cap>.+?)\s*$", re.I)
_ARROW_LINE_RE = re.compile(r"^[\s|│↓⇓▼vV→⟶>=\-–—]*$")


def parse_pipeline(lines: list[str]) -> list[list[dict]]:
    """The stages of a `::: pipeline` block: one line per stage, boxes split at `||`, a box's
    title and detail at its first ': ', detail lines at ';'. List markers and lines of arrows
    alone (an agent drawing its own) are ignored."""
    stages = []
    for line in lines:
        s = re.sub(r"^(?:[-*+•]|\d+[.)])\s+", "", line.strip())
        if _ARROW_LINE_RE.match(s):
            continue
        boxes = []
        for part in s.split("||"):
            part = re.sub(r"(\*\*|__)([^*_]+?):\1", r"\1\2\1:", part.strip())   # **Title:** detail
            title, _, detail = part.partition(": ")
            title = title.strip().rstrip(":").strip()
            if not title:
                continue
            boxes.append({"title": parse_inline(title),
                          "details": [parse_inline(d.strip()) for d in detail.split(";") if d.strip()]})
        if boxes:
            stages.append(boxes)
    return stages


_ID_RE = re.compile(r"\s*\{#([\w:.-]+)\}\s*$")
_ATTR_RE = re.compile(r"\s*\{([^{}]*)\}\s*$")


def split_id(text: str) -> tuple[str, str | None]:
    """'Caption text {#fig:map}' -> ('Caption text', 'fig:map')."""
    m = _ID_RE.search(text or "")
    if m:
        return text[:m.start()].rstrip(), m.group(1)
    return text, None


def split_attrs(text: str) -> tuple[str, dict]:
    """'Caption {#tbl:x width=4in}' -> ('Caption', {'id': 'tbl:x', 'width': '4in'})."""
    m = _ATTR_RE.search(text or "")
    if m and ("#" in m.group(1) or "=" in m.group(1) or m.group(1).strip().startswith(".")):
        return text[:m.start()].rstrip(), _parse_attrs(m.group(1))
    return text, {}


def _strip_caption_prefix(line: str, kind: str) -> str | None:
    """Return caption text if `line` is a '**Table 1.** ...' / 'Figure 2: ...' style caption."""
    rx = _TAB_CAP_RE if kind == "table" else _FIG_CAP_RE
    m = rx.match(line)
    if not m:
        return None
    # guard: must actually start with the keyword
    if not re.match(r"^\s*(?:\*\*|\*|__|_)?\s*(?:%s)" % ("Table|Tableau" if kind == "table" else "Figure|Fig\\.|Graphique|Carte|Map"), line, re.I):
        return None
    cap = m.group("cap").strip().strip("*_").strip()
    return cap


def _split_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    cells, cur, i = [], "", 0
    while i < len(s):
        ch = s[i]
        if ch == "\\" and i + 1 < len(s) and s[i + 1] == "|":
            cur += "|"; i += 2; continue
        if ch == "|":
            cells.append(cur.strip()); cur = ""
        else:
            cur += ch
        i += 1
    cells.append(cur.strip())
    return cells


def _parse_attrs(s: str | None) -> dict:
    d: dict[str, Any] = {}
    if not s:
        return d
    for tok in re.findall(r"[\w.-]+=\"[^\"]*\"|[\w.-]+='[^']*'|[\w.-]+=“[^”]*”|[\w.-]+=\S+|\.[\w-]+|#[\w:.-]+|-", s):
        if tok == "-" or tok == ".unnumbered":
            d["unnumbered"] = True
        elif tok == ".appendix":
            d["appendix"] = True
        elif tok.startswith("."):
            d.setdefault("classes", []).append(tok[1:])
        elif tok.startswith("#"):
            d["id"] = tok[1:]
        elif "=" in tok:
            k, v = tok.split("=", 1)
            d[k] = v.strip("\"'“”")
    return d


def _length_in(v: str | None, unit: str | None = None) -> float | None:
    if v is None:
        return None
    m = re.match(r"^([\d.]+)\s*(in|cm|mm|px|%)?$", str(v).strip())
    if not m:
        return None
    n = float(m.group(1)); u = m.group(2) or unit or "in"
    return {"in": n, "cm": n / 2.54, "mm": n / 25.4, "px": n / 96.0, "%": n / 100.0 * 6.5}[u]


# --------------------------------------------------------------------------- document parser
def split_front_matter(text: str) -> tuple[dict, str]:
    """Return (meta, body). Front matter is a YAML block between --- lines at the top."""
    m = re.match(r"^﻿?---\s*\n(.*?)\n(?:---|\.\.\.)\s*\n?", text, re.S)
    if not m:
        return {}, text
    raw = m.group(1)
    if yaml is not None:
        meta = yaml.safe_load(raw) or {}
    else:
        meta = _mini_yaml(raw)
    if not isinstance(meta, dict):
        meta = {}
    return meta, text[m.end():]


def _mini_scalar(v: str):
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
        return v[1:-1]
    low = v.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("null", "~", ""):
        return None
    if v.startswith("[") and v.endswith("]"):
        return [_mini_scalar(x) for x in v[1:-1].split(",") if x.strip()]
    if v.startswith("{") and v.endswith("}"):
        out = {}
        for part in v[1:-1].split(","):
            if ":" in part:
                a, b = part.split(":", 1); out[a.strip()] = _mini_scalar(b)
        return out
    return v


def _mini_yaml(raw: str) -> dict:
    """Fallback front-matter reader used only when PyYAML is not installed: top-level
    `key: value` scalars, inline [lists] / {maps}, block lists (`- item`), one level of nested
    `key: value` maps and folded/literal strings (`>` / `|`)."""
    meta: dict = {}
    key = None; mode = None; buf: list[str] = []

    def close():
        nonlocal mode, buf
        if key is not None and mode == "text":
            meta[key] = " ".join(s.strip() for s in buf).strip()
        mode = None; buf = []

    for line in raw.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith((" ", "\t")) and ":" in line:
            close()
            k, v = line.split(":", 1); key = k.strip(); v = v.strip()
            if v in (">", ">-", "|", "|-", ">+", "|+"):
                mode = "text"; meta[key] = ""
            elif v == "":
                meta[key] = None; mode = "block"
            else:
                meta[key] = _mini_scalar(v)
            continue
        s = line.strip()
        if key is None:
            continue
        if mode == "text":
            buf.append(s)
        elif mode == "block" and s.startswith("- "):
            if not isinstance(meta[key], list):
                meta[key] = []
            meta[key].append(_mini_scalar(s[2:]))
        elif mode == "block" and ":" in s:
            if not isinstance(meta[key], dict):
                meta[key] = {}
            a, b = s.split(":", 1); meta[key][a.strip()] = _mini_scalar(b)
    close()
    return meta


_ML_COMMENT_RE = re.compile(r"<!--(?!\s*pagebreak\s*-->).*?-->", re.S | re.I)


def parse_markdown(text: str) -> tuple[dict, list[dict], dict[str, Inline]]:
    """Parse a full document. Returns (meta, blocks, footnotes)."""
    meta, body = split_front_matter(text)
    # editorial comments may span lines (<!-- pagebreak --> is kept: it is a page break)
    body = _ML_COMMENT_RE.sub(lambda m: "\n" * m.group(0).count("\n"), body)
    lines = body.replace("\r\n", "\n").split("\n")
    footnotes: dict[str, Inline] = {}

    # pull footnote definitions out first (they may appear anywhere)
    kept: list[str] = []
    i = 0
    while i < len(lines):
        m = _FN_DEF_RE.match(lines[i])
        if m:
            fid, txt = m.group(1), m.group(2)
            i += 1
            while i < len(lines) and (lines[i].startswith("    ") or lines[i].startswith("\t")):
                txt += " " + lines[i].strip(); i += 1
            footnotes[fid] = parse_inline(txt.strip())
            continue
        kept.append(lines[i]); i += 1
    blocks = _parse_blocks(kept)
    return meta, blocks, footnotes


def _parse_blocks(lines: list[str]) -> list[dict]:
    blocks: list[dict] = []
    stack: list[dict] = []  # open ::: containers
    para: list[str] = []
    n = len(lines)
    i = 0

    def target() -> list[dict]:
        return stack[-1]["blocks"] if stack else blocks

    def flush_para():
        nonlocal para
        if para:
            txt = " ".join(s.strip() for s in para).strip()
            if txt:
                target().append({"type": "para", "inline": parse_inline(txt)})
            para = []

    def prev_para_caption(kind: str) -> Inline | None:
        """If the most recent block is a caption-looking paragraph, pop and return it."""
        tgt = target()
        if tgt and tgt[-1]["type"] == "para":
            cap = _strip_caption_prefix(inline_text(tgt[-1]["inline"]), kind)
            if cap is not None:
                tgt.pop()
                return parse_inline(cap)
            m = _CAPTION_LINE_RE.match(inline_text(tgt[-1]["inline"]))
            if m and m.group(0).lower().startswith(kind):
                tgt.pop()
                return parse_inline(m.group("cap"))
        return None

    def next_caption(j: int, kind: str) -> tuple[Inline | None, int]:
        """Look ahead (skipping blank lines) for a caption paragraph; return (caption, new index)."""
        k = j
        while k < n and not lines[k].strip():
            k += 1
        if k < n:
            cap = _strip_caption_prefix(lines[k], kind)
            if cap is not None:
                # caption may wrap onto following non-blank lines
                k += 1
                while k < n and lines[k].strip() and not _SRC_RE.match(lines[k]) and not _H_RE.match(lines[k]) \
                        and not _IMG_MD_RE.match(lines[k]) and not _IMG_HTML_RE.match(lines[k]) and not lines[k].lstrip().startswith("|"):
                    cap += " " + lines[k].strip(); k += 1
                return parse_inline(cap), k
        return None, j

    def next_source(j: int) -> tuple[Inline | None, int]:
        """Consume following 'Source:'/'Note:' lines (possibly several)."""
        k = j; parts: list[str] = []
        while True:
            while k < n and not lines[k].strip():
                k += 1
            if k < n and _SRC_RE.match(lines[k]):
                m = _SRC_RE.match(lines[k])
                seg = f"{m.group('kind')}: {m.group('txt')}"
                k += 1
                while k < n and lines[k].strip() and not _SRC_RE.match(lines[k]) and not _H_RE.match(lines[k]) \
                        and not lines[k].lstrip().startswith(("|", "!", "<img", "#", "-", "*", ":::")):
                    seg += " " + lines[k].strip(); k += 1
                parts.append(seg.rstrip("*_ ").strip())
                continue
            break
        if parts:
            return parse_inline(" ".join(parts)), k
        return None, j

    while i < n:
        line = lines[i]
        s = line.strip()

        if _COMMENT_RE.match(line):
            i += 1; continue
        if _PAGEBREAK_RE.match(line):
            flush_para(); target().append({"type": "pagebreak"}); i += 1; continue

        # fenced div open/close
        if s.startswith(":::"):
            flush_para()
            if _DIV_CLOSE_RE.match(s):
                if stack:
                    box = stack.pop(); target().append(box)
                i += 1; continue
            m = _DIV_OPEN_RE.match(s)
            kind = (m.group("kind") or "box").lower()
            title = m.group("title").strip().strip('"')
            if kind == "pipeline":           # its lines are stages, not blocks
                j = i + 1
                while j < n and not _DIV_CLOSE_RE.match(lines[j].strip()):
                    j += 1
                target().append({"type": "pipeline", "title": parse_inline(title) if title else None,
                                 "stages": parse_pipeline(lines[i + 1:j])})
                i = j + 1; continue
            stack.append({"type": "box", "kind": kind, "title": parse_inline(title) if title else None, "blocks": []})
            i += 1; continue

        # fenced code
        m = _FENCE_RE.match(s)
        if m:
            flush_para()
            fence, lang = m.group(1), m.group(2) or ""
            j = i + 1; buf = []
            while j < n and not lines[j].strip().startswith(fence[0] * 3):
                buf.append(lines[j]); j += 1
            # markdown in a fence is read as markdown, as the chat renders it (frontend
            # lib/markdown.ts): a `markdown`/`md` fence, or an unlabelled one holding a table
            rows = [b for b in buf if b.strip()]
            if lang.lower() in ("markdown", "md") or (
                    not lang and len(rows) >= 2 and all(r.strip().startswith("|") for r in rows)
                    and _TABLE_SEP_RE.match(rows[1])):
                lines[i:j + 1] = buf
                n = len(lines)
                continue
            target().append({"type": "code", "text": "\n".join(buf), "lang": lang})
            i = j + 1; continue

        # heading
        m = _H_RE.match(line)
        if m and not s.startswith("#!"):
            flush_para()
            level = len(m.group(1)); attrs = _parse_attrs(m.group(3))
            txt = m.group(2).strip()
            # strip a leading manual number like "1." / "3.2 " / "Appendix A"
            appendix = bool(attrs.get("appendix")) or bool(re.match(r"^(Appendix|Annex|Annexe)\b", txt, re.I))
            txt_clean = re.sub(r"^(?:\d+(?:\.\d+)*\.?|[A-Z]\.|Appendix\s+[A-Z][.:]?|Annexe?\s+[A-Z][.:]?)\s+", "", txt)
            target().append({"type": "heading", "level": level, "inline": parse_inline(txt_clean),
                             "numbered": not attrs.get("unnumbered"), "appendix": appendix, "id": attrs.get("id"),
                             "raw": txt, "kicker": attrs.get("kicker"), "attrs": attrs})
            i += 1; continue

        # table
        if s.startswith("|") and i + 1 < n and _TABLE_SEP_RE.match(lines[i + 1]):
            flush_para()
            caption = prev_para_caption("table")
            header = _split_row(lines[i]); seps = _split_row(lines[i + 1])
            align = []
            for c in seps:
                c = c.strip()
                align.append("center" if c.startswith(":") and c.endswith(":") else "right" if c.endswith(":") else "left" if c.startswith(":") else None)
            rows = []
            j = i + 2
            while j < n and lines[j].strip().startswith("|"):
                rows.append(_split_row(lines[j])); j += 1
            ncol = len(header)
            rows = [r + [""] * (ncol - len(r)) if len(r) < ncol else r[:ncol] for r in rows]
            if caption is None:
                caption, j = next_caption(j, "table")
            source, j = next_source(j)
            cap_id = None; tbl_width = None
            if caption:
                cap_txt, attrs = split_attrs(inline_text(caption))
                if attrs:
                    caption = parse_inline(cap_txt); cap_id = attrs.get("id"); tbl_width = _length_in(attrs.get("width"))
            target().append({"type": "table", "header": [parse_inline(h) for h in header],
                             "rows": [[parse_inline(c) for c in r] for r in rows], "align": align,
                             "caption": caption, "source": source, "widths": None, "id": cap_id, "width": tbl_width})
            i = j; continue

        # images (markdown or pandoc html form)
        m = _IMG_MD_RE.match(line) or _IMG_HTML_RE.match(line)
        if m:
            flush_para()
            path = m.group("path")
            width = None; fig_id = None; fig_attrs: dict = {}
            if "attrs" in m.groupdict() and m.group("attrs"):
                attrs = _parse_attrs(m.group("attrs"))
                fig_attrs = attrs
                width = _length_in(attrs.get("width")); fig_id = attrs.get("id")
            elif "w" in m.groupdict() and m.group("w"):
                width = _length_in(m.group("w"), m.group("unit"))
            alt = m.group("alt").strip() if "alt" in m.groupdict() and m.group("alt") else ""
            caption = prev_para_caption("figure")
            j = i + 1
            if caption is None:
                caption, j = next_caption(j, "figure")
            if caption is None and alt:
                cap = _strip_caption_prefix(alt, "figure")
                caption = parse_inline(cap if cap is not None else alt)
            source, j = next_source(j)
            if caption:
                cap_txt, attrs = split_attrs(inline_text(caption))
                if attrs:
                    caption = parse_inline(cap_txt); fig_id = fig_id or attrs.get("id")
                    width = width or _length_in(attrs.get("width"))
                    fig_attrs = {**attrs, **fig_attrs}
            target().append({"type": "figure", "path": path, "caption": caption, "source": source, "width": width,
                             "id": fig_id, "attrs": fig_attrs})
            i = j; continue

        # list
        m = _LIST_RE.match(line)
        if m and s:
            flush_para()
            items = []
            j = i
            while j < n:
                lm = _LIST_RE.match(lines[j])
                if lm and lines[j].strip():
                    indent = len(lm.group(1).replace("\t", "    "))
                    level = min(indent // 2, 3) if indent < 8 else min(indent // 4 + 1, 3)
                    items.append({"level": level, "ordered": lm.group(3) is not None, "text": lm.group(4).strip()})
                    j += 1
                    # continuation lines (indented, non-list)
                    while j < n and lines[j].strip() and not _LIST_RE.match(lines[j]) and lines[j].startswith((" ", "\t")) \
                            and not lines[j].strip().startswith(("|", ":::", "#")):
                        items[-1]["text"] += " " + lines[j].strip(); j += 1
                elif not lines[j].strip():
                    # blank line: list continues only if next non-blank line is a list item
                    k = j
                    while k < n and not lines[k].strip():
                        k += 1
                    if k < n and _LIST_RE.match(lines[k]) and lines[k].strip():
                        j = k; continue
                    break
                else:
                    break
            # normalise levels so the smallest indent is level 0
            if items:
                base = min(it["level"] for it in items)
                for it in items:
                    it["level"] -= base
                    it["inline"] = parse_inline(it.pop("text"))
            target().append({"type": "list", "items": items})
            i = j; continue

        # blockquote -> callout
        if s.startswith(">"):
            flush_para()
            buf = []
            j = i
            while j < n and lines[j].strip().startswith(">"):
                buf.append(lines[j].strip()[1:].strip()); j += 1
            inner = _parse_blocks(buf)
            target().append({"type": "box", "kind": "quote", "title": None, "blocks": inner})
            i = j; continue

        if _HR_RE.match(line):
            flush_para(); target().append({"type": "hr"}); i += 1; continue

        if not s:
            flush_para(); i += 1; continue

        para.append(line)
        i += 1

    flush_para()
    while stack:
        box = stack.pop(); (stack[-1]["blocks"] if stack else blocks).append(box)
    return blocks


if __name__ == "__main__":
    import json, sys
    meta, blocks, fns = parse_markdown(open(sys.argv[1], encoding="utf-8").read())
    print(json.dumps(meta, indent=1, default=str))
    for b in blocks:
        t = b["type"]
        if t == "heading":
            print("#" * b["level"], inline_text(b["inline"]), "" if b["numbered"] else "{-}", "[appendix]" if b["appendix"] else "",
                  f"[kicker: {b['kicker']}]" if b.get("kicker") else "")
        elif t == "para":
            print("P:", inline_text(b["inline"])[:80])
        elif t == "table":
            print(f"TABLE {len(b['header'])}x{len(b['rows'])} cap={inline_text(b['caption'] or [])[:50]!r} src={inline_text(b['source'] or [])[:40]!r}")
        elif t == "figure":
            print(f"FIGURE {b['path']} cap={inline_text(b['caption'] or [])[:50]!r} src={inline_text(b['source'] or [])[:40]!r} w={b['width']}")
        elif t == "list":
            print("LIST", [(it["level"], it["ordered"], inline_text(it["inline"])[:30]) for it in b["items"]][:4])
        elif t == "box":
            print("BOX", b["kind"], inline_text(b["title"] or []), len(b["blocks"]))
        elif t == "pipeline":
            print("PIPELINE", [[inline_text(x["title"]) for x in st] for st in b["stages"]])
        else:
            print(t.upper())
    print("footnotes:", {k: inline_text(v)[:40] for k, v in fns.items()})
