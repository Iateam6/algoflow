import os
import environ
import requests
from PyPDF2 import PdfMerger

env = environ.Env()
env.read_env()

API_KEY = env("API_KEY")

BASE_URL = "https://api.pdf.co/v1"

def upload_to_pdfco(file_path):
    """Uploads a local file to PDF.co and returns the uploaded file URL."""
    url = f"{BASE_URL}/file/upload"
    headers = {"x-api-key": API_KEY}
    with open(file_path, "rb") as f:
        files = {"file": f}
        resp = requests.post(url, headers=headers, files=files)
    if resp.status_code == 200:
        data = resp.json()
        if not data.get("error"):
            return data["url"]
        raise Exception(f"Upload error: {data.get('message')}")
    raise Exception(f"Upload failed: {resp.status_code} {resp.reason}")


def convert_to_pdf(file_url, file_ext, output_name):
    """Converts DOC/DOCX/Image to PDF using PDF.co, returns PDF URL."""
    ext = file_ext.lower()
    if ext in [".doc", ".docx"]:
        endpoint = "/pdf/convert/from/doc"
    elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
        endpoint = "/pdf/convert/from/image"
    elif ext == ".pdf":
        return file_url  # already PDF
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    headers = {"x-api-key": API_KEY}
    payload = {"name": f"{output_name}.pdf", "url": file_url}

    resp = requests.post(f"{BASE_URL}{endpoint}", headers=headers, data=payload)
    if resp.status_code == 200:
        data = resp.json()
        if not data.get("error"):
            return data["url"]
        raise Exception(f"Conversion error: {data.get('message')}")
    raise Exception(f"Conversion failed: {resp.status_code} {resp.reason}")


def merge_pdfs(pdf_urls, output_path):
    """Merges multiple PDF URLs into one and saves locally."""
    headers = {"x-api-key": API_KEY}
    payload = {"name": os.path.basename(output_path), "url": ",".join(pdf_urls)}

    resp = requests.post(f"{BASE_URL}/pdf/merge", headers=headers, data=payload)
    if resp.status_code == 200:
        data = resp.json()
        if not data.get("error"):
            # Download merged file
            r = requests.get(data["url"], stream=True)
            if r.status_code == 200:
                with open(output_path, "wb") as f:
                    for chunk in r:
                        f.write(chunk)
                return True
            raise Exception(f"Download failed: {r.status_code} {r.reason}")
        raise Exception(f"Merge error: {data.get('message')}")
    raise Exception(f"Merge failed: {resp.status_code} {resp.reason}")
