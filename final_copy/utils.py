import os
from docx2pdf import convert
from PyPDF2 import PdfMerger

def convert_doc_to_pdf(doc_path, output_dir):
    """
    Converts a .docx file to PDF using Microsoft Word via docx2pdf.
    """
    filename = os.path.splitext(os.path.basename(doc_path))[0]
    output_pdf_path = os.path.join(output_dir, f"{filename}.pdf")

    print(f"[INFO] Converting '{doc_path}' to PDF using docx2pdf...")

    try:
        convert(doc_path, output_pdf_path)
        print(f"[SUCCESS] Converted: {output_pdf_path}")
        return output_pdf_path
    except Exception as e:
        print(f"[ERROR] Conversion failed for '{doc_path}'. Reason: {e}")
        return None


def merge_files(file_paths, output_pdf="merged_output.pdf"):
    """
    Detects DOC/DOCX files, converts them to PDFs, and merges all PDFs into one.
    """
    temp_dir = "temp_converted_pdfs"
    os.makedirs(temp_dir, exist_ok=True)
    merger = PdfMerger()
    pdf_list = []

    for path in file_paths:
        ext = os.path.splitext(path)[1].lower()

        if ext in [".doc", ".docx"]:
            pdf_path = convert_doc_to_pdf(path, temp_dir)
            if pdf_path:
                pdf_list.append(pdf_path)
        elif ext == ".pdf":
            pdf_list.append(path)
        else:
            print(f"[WARNING] Unsupported file skipped: {path}")

    if not pdf_list:
        print("[ERROR] No valid PDF or DOC files found.")
        return

    print(f"[INFO] Merging {len(pdf_list)} files...")

    for pdf in pdf_list:
        merger.append(pdf)

    merger.write(output_pdf)
    merger.close()

    print(f"[✅ DONE] All files merged into: {output_pdf}")

    # Clean up temporary PDFs
    for f in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, f))
    os.rmdir(temp_dir)
