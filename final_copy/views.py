import os
import json
import shutil
import tempfile
import urllib.parse
import cgi
import asyncio
import aiohttp
import aiofiles

from django.conf import settings
from django.http import JsonResponse, HttpResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
from asgiref.sync import async_to_sync

from .utils import (
    convert_to_pdf,
    merge_pdfs,
    create_blank_page_pdf,
    create_docx_with_separators,
)


def index(request):
    return HttpResponse("Final-copy API!")


@csrf_exempt
def final_copy(request):
    if request.method != "POST":
        return HttpResponseBadRequest("Invalid request method")

    try:
        payload = json.loads(request.body)
        docs = payload.get("docs", [])
        forms = payload.get("forms", [])
        files = payload.get("files", [])

        print("[DEBUG] Incoming request payload:", payload)

    except Exception as e:
        print("[ERROR] JSON Parse Failed:", e)
        return HttpResponseBadRequest(f"Invalid JSON: {e}")

    if not isinstance(files, list) or not isinstance(forms, list):
        print("[ERROR] Files or Forms is not a list")
        return HttpResponseBadRequest("Invalid input format.")

    return async_to_sync(process_final_copy)(request, docs, files, forms)


# ------------------- ASYNC HANDLER -------------------
async def process_final_copy(request, docs, files, forms):
    print("[DEBUG] Starting final_copy processing...")

    temp_dir = tempfile.mkdtemp()
    print("[DEBUG] Temporary directory created:", temp_dir)

    try:
        async with aiohttp.ClientSession() as session:

            async def download_file(url):
                print(f"[DEBUG] Downloading: {url}")
                async with session.get(url, timeout=60) as resp:
                    if resp.status != 200:
                        raise Exception(f"HTTP {resp.status}: {resp.reason}")

                    cd = resp.headers.get("content-disposition")
                    if cd:
                        _, params = cgi.parse_header(cd)
                        filename = params.get("filename") or os.path.basename(
                            urllib.parse.urlparse(url).path
                        )
                    else:
                        filename = os.path.basename(urllib.parse.urlparse(url).path)

                    if not os.path.splitext(filename)[1]:
                        filename += ".pdf"

                    local_path = os.path.join(temp_dir, filename)
                    print(f"[DEBUG] Saving downloaded file as: {local_path}")

                    async with aiofiles.open(local_path, "wb") as f:
                        async for chunk in resp.content.iter_chunked(8192):
                            await f.write(chunk)

                    print(f"[DEBUG] Download complete: {local_path}")
                    return local_path

            async def ensure_pdf(file_path):
                if not file_path.lower().endswith(".pdf"):
                    print(f"[DEBUG] Converting to PDF: {file_path}")
                    return await convert_to_pdf(file_path)
                print(f"[DEBUG] Already a PDF: {file_path}")
                return file_path

            print("[DEBUG] Downloading files...")
            file_paths = await asyncio.gather(*[download_file(url) for url in files])
            form_paths = await asyncio.gather(*[download_file(url) for url in forms])

            print("[DEBUG] Converting to PDF if needed...")
            file_pdfs = [await ensure_pdf(f) for f in file_paths]
            form_pdfs = [await ensure_pdf(f) for f in form_paths]

            print("[DEBUG] Creating separator page...")
            separator_pdf = os.path.join(temp_dir, "separator.pdf")
            await create_blank_page_pdf(separator_pdf, text="--- Supporting Documents ---")

            print("[DEBUG] Merging all PDFs...")
            merged_group_pdf = os.path.join(temp_dir, "final_copy.pdf")
            await merge_pdfs(file_pdfs + [separator_pdf] + form_pdfs, merged_group_pdf)

            print("[DEBUG] Generating DOCX with separators...")
            docx_output = os.path.join(temp_dir, "final_copy.docx")
            await create_docx_with_separators(docs, docx_output)

            print("[DEBUG] Moving output files to MEDIA_ROOT...")
            media_dir = os.path.join(settings.MEDIA_ROOT, "generated")
            os.makedirs(media_dir, exist_ok=True)

            final_pdf_path = os.path.join(media_dir, "final_copy.pdf")
            final_docx_path = os.path.join(media_dir, "final_copy.docx")

            shutil.move(merged_group_pdf, final_pdf_path)
            shutil.move(docx_output, final_docx_path)

            print("[DEBUG] Files saved:")
            print("       PDF :", final_pdf_path)
            print("       DOCX:", final_docx_path)

            pdf_rel_path = os.path.join(settings.MEDIA_URL.lstrip("/"), "generated", "final_copy.pdf")
            docx_rel_path = os.path.join(settings.MEDIA_URL.lstrip("/"), "generated", "final_copy.docx")

            pdf_url = request.build_absolute_uri(f"/{pdf_rel_path}")
            docx_url = request.build_absolute_uri(f"/{docx_rel_path}")

            print("[DEBUG] Returning response with URLs")
            return JsonResponse({
                "final_pdf_url": pdf_url,
                "final_docx_url": docx_url
            })

    finally:
        try:
            print("[DEBUG] Cleaning up temporary directory:", temp_dir)
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"[WARN] Cleanup failed: {e}")
