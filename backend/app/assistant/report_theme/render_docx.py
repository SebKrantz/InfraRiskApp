#!/usr/bin/env python3
"""
render_docx.py — convert a .docx to PDF with LibreOffice and (optionally) render page
images plus a contact sheet so the layout can be checked visually.

Copied from ~/.claude/skills/cpcs-report/scripts/render_docx.py (version of 2026-09-08) and
made safe for library use (AGUI vendors it next to aeilabs_docx.py):
  - failures raise RenderError (a RuntimeError) instead of SystemExit; only main() exits;
  - every LibreOffice call gets its own throwaway -env:UserInstallation profile (proper file://
    URI), so concurrent builds never collide, and a caller-supplied timeout;
  - pdf_pages_text() reads the whole PDF with one pdftotext call (pages split on form feeds).

    python render_docx.py report.docx                 # -> report.pdf next to it
    python render_docx.py report.docx --pages         # also report_pages/page-01.png ...
    python render_docx.py report.docx --sheet         # also report_sheet.png (all pages, small)
    python render_docx.py report.docx --outdir /tmp/x

Requires `soffice` (LibreOffice) and, for --pages/--sheet, `pdftoppm` (poppler) and Pillow.
Note: LibreOffice renders the *cached* TOC entries; build_aeilabs_doc.py fills those in
(two-pass), and Word refreshes all fields on open (updateFields is set).
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


class RenderError(RuntimeError):
    """LibreOffice / poppler missing or the conversion failed."""


def find_soffice() -> str:
    for c in ("soffice", "libreoffice", "/Applications/LibreOffice.app/Contents/MacOS/soffice",
              "/usr/bin/soffice", "/usr/local/bin/soffice", "/opt/homebrew/bin/soffice"):
        p = shutil.which(c) or (c if os.path.exists(c) else None)
        if p:
            return p
    raise RenderError("LibreOffice (soffice) not found; install it or export the PDF from Word.")


def have_soffice() -> bool:
    try:
        find_soffice()
        return True
    except RenderError:
        return False


def docx_to_pdf(docx: str, outdir: str | None = None, timeout: int = 240) -> str:
    """Convert with LibreOffice headless. Returns the PDF path."""
    docx = os.path.abspath(docx)
    outdir = os.path.abspath(outdir or os.path.dirname(docx))
    os.makedirs(outdir, exist_ok=True)
    soffice = find_soffice()
    # a private, throwaway profile per call: no clash with a running LibreOffice GUI or with
    # other conversions running at the same time
    with tempfile.TemporaryDirectory(prefix="lo-profile-") as prof:
        cmd = [soffice, f"-env:UserInstallation={Path(prof).as_uri()}", "--headless", "--norestore",
               "--convert-to", "pdf", "--outdir", outdir, docx]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            raise RenderError(f"LibreOffice timed out after {timeout} s") from e
    pdf = os.path.join(outdir, os.path.splitext(os.path.basename(docx))[0] + ".pdf")
    if not os.path.exists(pdf):
        raise RenderError(f"PDF conversion failed:\n{r.stdout}\n{r.stderr}")
    return pdf


def pdf_pages_text(pdf: str) -> list[str]:
    """Text of each page (needs pdftotext)."""
    if not shutil.which("pdftotext"):
        raise RenderError("pdftotext (poppler) not found")
    r = subprocess.run(["pdftotext", "-layout", pdf, "-"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RenderError(f"pdftotext failed: {r.stderr}")
    pages = r.stdout.split("\f")
    if pages and not pages[-1].strip():
        pages = pages[:-1]
    return pages


def pdf_page_count(pdf: str) -> int:
    r = subprocess.run(["pdfinfo", pdf], capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split()[1])
    return 0


def render_pages(pdf: str, outdir: str, dpi: int = 60, first: int | None = None, last: int | None = None) -> list[str]:
    if not shutil.which("pdftoppm"):
        raise RenderError("pdftoppm (poppler) not found")
    os.makedirs(outdir, exist_ok=True)
    for f in glob.glob(os.path.join(outdir, "page-*.png")):
        os.remove(f)
    cmd = ["pdftoppm", "-png", "-r", str(dpi)]
    if first:
        cmd += ["-f", str(first)]
    if last:
        cmd += ["-l", str(last)]
    cmd += [pdf, os.path.join(outdir, "page")]
    subprocess.run(cmd, check=True)
    return sorted(glob.glob(os.path.join(outdir, "page-*.png")))


def contact_sheet(pages: list[str], out: str, cols: int = 4, max_pages: int = 24) -> str:
    from PIL import Image
    pages = pages[:max_pages]
    ims = [Image.open(p) for p in pages]
    w = max(i.width for i in ims); h = max(i.height for i in ims)
    rows = (len(ims) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w + (cols + 1) * 6, rows * h + (rows + 1) * 6), "#BBBBBB")
    for k, im in enumerate(ims):
        sheet.paste(im, (6 + (k % cols) * (w + 6), 6 + (k // cols) * (h + 6)))
    sheet.save(out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("docx")
    ap.add_argument("--outdir", help="directory for the PDF (default: next to the docx)")
    ap.add_argument("--pages", action="store_true", help="render every page to PNG (<name>_pages/)")
    ap.add_argument("--sheet", action="store_true", help="write a contact sheet PNG of all pages")
    ap.add_argument("--dpi", type=int, default=60, help="DPI for page PNGs (default 60; use 110+ to read text)")
    ap.add_argument("--cols", type=int, default=4)
    args = ap.parse_args()

    try:
        pdf = docx_to_pdf(args.docx, args.outdir)
        print("PDF:", pdf, f"({pdf_page_count(pdf)} pages)")
        if args.pages or args.sheet:
            base = os.path.splitext(pdf)[0]
            pages = render_pages(pdf, base + "_pages", dpi=args.dpi)
            if args.sheet:
                print("Sheet:", contact_sheet(pages, base + "_sheet.png", cols=args.cols))
            if args.pages:
                print("Pages:", base + "_pages/")
    except RenderError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
