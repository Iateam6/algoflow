import os
from PIL import Image, ImageDraw
from PyPDF2 import PdfMerger
from docx import Document
from copy import deepcopy
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt
from docx.enum.table import WD_ALIGN_VERTICAL, WD_ROW_HEIGHT_RULE
from docx.enum.text import WD_BREAK, WD_PARAGRAPH_ALIGNMENT
from docxcompose.composer import Composer
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfparser import PDFSyntaxError
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

import os
from pathlib import Path
from copy import deepcopy
from typing import List, Tuple, Union, Optional


async def convert_to_pdf(file_path):
    """Convert image to PDF."""
    try:
        pdf_path = os.path.splitext(file_path)[0] + ".pdf"
        with Image.open(file_path) as img:
            img = img.convert("RGB")
            img.save(pdf_path, "PDF", resolution=100.0)
        print(f"[OK] Converted to PDF: {pdf_path}")
        return pdf_path
    except Exception as e:
        print(f"[ERROR] Image conversion failed for {file_path}: {e}")
        return file_path


async def merge_pdfs(pdf_paths, output_path):
    """
    Merge PDFs safely across Windows & Linux.
    - Validates PDFs before merging
    - Automatically skips or replaces corrupted files
    """
    merger = PdfMerger()
    valid_pdfs = []

    for path in pdf_paths:
        if not path or not os.path.exists(path):
            print(f"[WARN] Skipping missing file: {path}")
            continue

        # Validate PDF with pdfminer.six
        try:
            with open(path, "rb") as f:
                parser = PDFParser(f)
                PDFDocument(parser)  # attempt to parse
            valid_pdfs.append(path)
            print(f"[OK] Valid PDF: {path}")

        except (PDFSyntaxError, Exception) as e:
            print(f"[WARN] Invalid PDF detected: {path} ({e})")
            placeholder = output_path + f"_{os.path.basename(path)}_error.pdf"
            c = canvas.Canvas(placeholder, pagesize=letter)
            c.setFont("Helvetica", 12)
            c.drawCentredString(300, 500, f"⚠️ Skipped file: {os.path.basename(path)}")
            c.drawCentredString(300, 470, f"Error: {str(e)[:100]}")
            c.save()
            valid_pdfs.append(placeholder)
            print(f"[INFO] Added placeholder for bad file: {placeholder}")

    if not valid_pdfs:
        raise Exception("No valid PDF files to merge.")

    try:
        for pdf in valid_pdfs:
            merger.append(pdf)
        merger.write(output_path)
        merger.close()
        print(f"[OK] Successfully merged {len(valid_pdfs)} PDFs → {output_path}")
    except Exception as e:
        print(f"[ERROR] PDF merge failed: {e}")
        raise


async def create_blank_page_pdf(output_path, text=""):
    """Create a blank separator page."""
    width, height = 595, 842
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    if text:
        draw.text((100, height // 2), text, fill="black")
    img.save(output_path, "PDF")
    print(f"[OK] Created separator PDF: {output_path}")


# DOCX utilities

def _is_sectPr(element) -> bool:
    return element.tag.endswith('}sectPr') or element.tag.endswith('sectPr')


def _normalize(doc_entries):
    normalized = []
    for e in doc_entries:
        if isinstance(e, (list, tuple)) and len(e) >= 2:
            normalized.append((str(e[0]), str(e[1])))
        else:
            p = str(e)
            normalized.append((os.path.basename(p), p))
    return normalized


def prepend_cover_and_merge(
   doc_entries: List[Union[str, Tuple[str, str]]],
    merged_output_path: str,
    name_font_size: int = 16,
) -> str:
    """
    Each file -> make a copy with a true full-page cover (its own filename centered both
    vertically & horizontally).  Those new files are saved in 'with_covers' subfolder.
    Then merge all new files (once each, in order) into merged_output_path.
    """
    normalized = _normalize(doc_entries)
    merged_output = Path(merged_output_path)
    output_dir = merged_output.parent / "with_covers"
    output_dir.mkdir(parents=True, exist_ok=True)

    new_files = []

    # Step 1: create per-file cover + content
    for display_name, path in normalized:
        p = Path(path)
        if not p.exists() or not p.suffix.lower().endswith(".docx"):
            print(f"[SKIP] {path}")
            continue

        try:
            original = Document(str(p))
            new_doc = Document()

            # Copy section layout
            try:
                src = original.sections[0]
                dst = new_doc.sections[0]
                dst.page_width, dst.page_height = src.page_width, src.page_height
                dst.top_margin, dst.bottom_margin = src.top_margin, src.bottom_margin
                dst.left_margin, dst.right_margin = src.left_margin, src.right_margin
            except Exception:
                pass

            # Remove default empty paragraph
            if new_doc.paragraphs:
                p0 = new_doc.paragraphs[0]
                p0._element.getparent().remove(p0._element)

            # Printable area
            sec = new_doc.sections[0]
            printable_w = sec.page_width - sec.left_margin - sec.right_margin
            printable_h = sec.page_height - sec.top_margin - sec.bottom_margin

            # 1x1 table for true centered text
            tbl = new_doc.add_table(rows=1, cols=1)
            tbl.autofit = False
            cell = tbl.cell(0, 0)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            para = cell.paragraphs[0]
            para.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
            run = para.add_run(display_name)
            run.bold = True
            run.font.size = Pt(name_font_size)

            # borderless
            tbl_pr = tbl._tblPr
            borders = OxmlElement("w:tblBorders")
            for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
                b = OxmlElement(f"w:{side}")
                b.set(qn("w:val"), "nil")
                borders.append(b)
            tbl_pr.append(borders)

            # size table
            try:
                tbl.columns[0].width = printable_w
                row = tbl.rows[0]
                row.height = printable_h
                row.height_rule = WD_ROW_HEIGHT_RULE.EXACTLY
            except Exception:
                pass

            para.add_run().add_break(WD_BREAK.PAGE)

            # append original body
            for el in original.element.body:
                new_doc.element.body.append(deepcopy(el))

            # save in new folder
            new_path = output_dir / p.name
            new_doc.save(str(new_path))
            new_files.append(new_path)
            print(f"[OK] Created cover for {p.name}")

        except Exception as e:
            print(f"[ERROR] {path}: {e}")

    # Step 2: merge only those new files (once each)
    merged = Document()
    if merged.paragraphs:
        p0 = merged.paragraphs[0]
        p0._element.getparent().remove(p0._element)

    for nf in new_files:
        try:
            doc = Document(str(nf))
            for el in doc.element.body:
                if not _is_sectPr(el):
                    merged.element.body.append(deepcopy(el))
            print(f"[MERGED] {nf.name}")
        except Exception as e:
            print(f"[ERROR MERGING] {nf}: {e}")

    merged.save(str(merged_output))
    print(f"[DONE] Merged DOCX saved: {merged_output}")
    return str(merged_output)
