import os
from PIL import Image, ImageDraw
from PyPDF2 import PdfMerger
from docx import Document
from docx.shared import Inches
from docx2txt import process as docx_extract_text
from pdfminer.high_level import extract_text as pdf_extract_text
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
    Create a DOCX with actual embedded content:
    - If DOCX → append content
    - If PDF → extract text and append
    - If Image → embed image
    - Otherwise → include filename reference
    """
    doc = Document()

    for i, (name, path) in enumerate(doc_entries, start=1):
        ext = os.path.splitext(path)[1].lower()
        doc.add_page_break()
        doc.add_heading(f"{i}. {name}", level=1)
        doc.add_paragraph(f"Source file: {os.path.basename(path)}")

        try:
            if ext in [".docx"]:
                # Extract text from DOCX and append
                text = docx_extract_text(path)
                if text.strip():
                    doc.add_paragraph(text)
                else:
                    doc.add_paragraph("[No extractable text found in DOCX]")
                print(f"[OK] Embedded DOCX content: {path}")

            elif ext in [".pdf"]:
                # Extract text from PDF
                text = pdf_extract_text(path)
                if text.strip():
                    doc.add_paragraph(text)
                else:
                    doc.add_paragraph("[No extractable text found in PDF]")
                print(f"[OK] Embedded PDF text: {path}")

            elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
                # Embed image directly
                try:
                    doc.add_picture(path, width=Inches(5.5))
                    print(f"[OK] Embedded image: {path}")
                except Exception as e:
                    print(f"[WARN] Could not insert image {path}: {e}")

            else:
                # Unknown format → just mention it
                doc.add_paragraph(f"[Unsupported file type: {ext}]")

        except Exception as e:
            print(f"[ERROR] Failed embedding {path}: {e}")
            doc.add_paragraph(f"[Error embedding {os.path.basename(path)}: {e}]")

    doc.save(output_path)
    print(f"[OK] Created DOCX with full content: {output_path}")
