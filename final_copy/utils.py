import os
from PIL import Image, ImageDraw, ImageFont
from PyPDF2 import PdfMerger
from docx import Document


# Convert images to PDF
async def convert_to_pdf(file_path):
    """Convert supported image formats to PDF using Pillow."""
    try:
        pdf_path = os.path.splitext(file_path)[0] + ".pdf"
        with Image.open(file_path) as img:
            img = img.convert("RGB")
            img.save(pdf_path, "PDF", resolution=100.0)
        print(f"[OK] Converted image to PDF: {pdf_path}")
        return pdf_path
    except Exception as e:
        print(f"[ERROR] Image conversion failed: {e}")
        return file_path


# Merge multiple PDFs
async def merge_pdfs(pdf_paths, output_path):
    """Merge multiple PDF files into a single file."""
    merger = PdfMerger()
    try:
        for pdf in pdf_paths:
            if os.path.exists(pdf):
                merger.append(pdf)
        merger.write(output_path)
        merger.close()
        print(f"[OK] Merged PDFs into {output_path}")
    except Exception as e:
        print(f"[ERROR] Failed to merge PDFs: {e}")
        raise


# Create a blank separator page as PDF
async def create_blank_page_pdf(output_path, text=""):
    """Create a simple PDF separator page with optional text."""
    try:
        width, height = 595, 842  # A4 size in points
        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)

        # Basic text placement
        if text:
            draw.text((100, height // 2), text, fill="black")

        img.save(output_path, "PDF")
        print(f"[OK] Created separator page: {output_path}")
    except Exception as e:
        print(f"[ERROR] Failed to create blank page: {e}")
        raise


# Create a DOCX file with separator pages for each AI-generated document
async def create_docx_with_separators(docs, output_path):
    """Generate DOCX file with separators for each AI document entry."""
    try:
        doc = Document()
        for i, doc_text in enumerate(docs, start=1):
            doc.add_page_break()
            doc.add_heading(f"Document {i}", level=1)
            doc.add_paragraph(doc_text)
        doc.save(output_path)
        print(f"[OK] DOCX file created: {output_path}")
    except Exception as e:
        print(f"[ERROR] Failed to create DOCX file: {e}")
        raise
