import os
from PIL import Image
from PyPDF2 import PdfMerger
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from docx import Document
from docx.shared import Pt
import asyncio


# Convert images to PDF
async def convert_to_pdf(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        pdf_path = os.path.splitext(image_path)[0] + ".pdf"
        img.save(pdf_path, "PDF", resolution=100.0)
        return pdf_path
    except Exception as e:
        print(f"[ERROR] Failed to convert image to PDF: {e}")
        raise


# Merge multiple PDFs
async def merge_pdfs(pdf_paths, output_path):
    try:
        merger = PdfMerger()
        for pdf in pdf_paths:
            if os.path.exists(pdf):
                merger.append(pdf)
        merger.write(output_path)
        merger.close()
    except Exception as e:
        print(f"[ERROR] Failed to merge PDFs: {e}")
        raise


# Create a blank page or one with text
async def create_blank_page_pdf(output_path, text=None):
    c = canvas.Canvas(output_path, pagesize=A4)
    if text:
        c.setFont("Helvetica-Bold", 18)
        width, height = A4
        c.drawCentredString(width / 2, height / 2, text)
    c.showPage()
    c.save()


# Create DOCX with separators for each AI-generated document
async def create_docx_with_separators(docs, output_path):
    document = Document()

    for i, doc in enumerate(docs, start=1):
        name = doc.get("name", f"Document {i}")
        url = doc.get("url", "")

        # Add separator title page
        title = document.add_paragraph()
        run = title.add_run(name)
        run.bold = True
        run.font.size = Pt(20)
        document.add_paragraph(f"URL: {url}")

        # Add page break after each doc except the last
        if i < len(docs):
            document.add_page_break()

    document.save(output_path)
