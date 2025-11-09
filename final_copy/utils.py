import os
from PIL import Image, ImageDraw
from PyPDF2 import PdfMerger
from docx import Document
from copy import deepcopy
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfparser import PDFSyntaxError
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter


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


async def create_docx_with_separators(doc_entries, output_path):
    """
    Merge multiple DOCX files into one DOCX with a page break between each file.

    Parameters:
      doc_entries: list of (name, path) tuples
      output_path: str, path to save the merged .docx
    """

    merged = Document()

    for i, (_, path) in enumerate(doc_entries):
        try:
            ext = os.path.splitext(path)[1].lower()
            if ext != ".docx":
                print(f"[SKIP] Unsupported file type for {path}")
                continue

            sub_doc = Document(path)

            # Append the content of each sub-document
            for element in sub_doc.element.body:
                merged.element.body.append(deepcopy(element))

            # Add a page break after each document except the last one
            if i < len(doc_entries) - 1:
                merged.add_page_break()

            print(f"[OK] Merged: {path}")

        except Exception as e:
            print(f"[ERROR] Could not merge {path}: {e}")
            continue

    merged.save(output_path)
    print(f"[OK] Created merged DOCX with page breaks: {output_path}")
