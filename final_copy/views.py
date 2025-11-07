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
from django.http import JsonResponse, HttpResponseBadRequest, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from asgiref.sync import async_to_sync
from .utils import convert_to_pdf, merge_pdfs, create_blank_page_pdf, create_docx_with_separators


def index(request):
    return HttpResponse("Final-copy API !")


@csrf_exempt
def final_copy(request):
    """Main entry for merging and generating final copy documents."""
    if request.method != "POST":
        return HttpResponseBadRequest("Invalid request method")

    try:
        payload = json.loads(request.body)
        docs = payload.get("docs", [])
        forms = payload.get("forms", [])
        files = payload.get("files", [])
        print("[DEBUG] Payload received:", payload)
    except Exception as e:
        return HttpResponseBadRequest(f"Invalid JSON payload: {e}")

    if not isinstance(docs, list) or not isinstance(forms, list) or not isinstance(files, list):
        return HttpResponseBadRequest("Invalid structure in JSON payload")

    return async_to_sync(process_final_copy)(request, docs, files, forms)


# ------------------- ASYNC HANDLER -------------------
async def process_final_copy(request, docs, files, forms):
    print("[DEBUG] Processing final copy from URLs...")
    temp_dir = tempfile.mkdtemp()
    print(f"[DEBUG] Temporary directory: {temp_dir}")

    try:
        async with aiohttp.ClientSession() as session:

            async def download_file(url, name_hint=None):
                """Download a file from a URL to the temporary directory."""
                try:
                    headers = {
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/120.0 Safari/537.36"
                        )
                    }
                    async with session.get(url, headers=headers, timeout=60) as resp:
                        if resp.status != 200:
                            print(f"[ERROR] Failed to download {url}: HTTP {resp.status}")
                            return None

                        cd = resp.headers.get("content-disposition")
                        if cd:
                            _, params = cgi.parse_header(cd)
                            filename = params.get("filename") or os.path.basename(urllib.parse.urlparse(url).path)
                        else:
                            filename = os.path.basename(urllib.parse.urlparse(url).path)

                        # If no filename, use the provided hint
                        if not filename:
                            filename = f"{name_hint or 'file'}.pdf"

                        local_path = os.path.join(temp_dir, filename)
                        async with aiofiles.open(local_path, "wb") as f:
                            async for chunk in resp.content.iter_chunked(8192):
                                await f.write(chunk)
                        print(f"[OK] Downloaded: {local_path}")
                        return local_path
                except Exception as e:
                    print(f"[ERROR] Exception downloading {url}: {e}")
                    return None

            async def ensure_pdf(file_path, is_image_only=False):
                """Convert images to PDF only if is_image_only=True."""
                if not file_path:
                    return None
                ext = os.path.splitext(file_path)[1].lower()
                valid_images = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"]
                if is_image_only and ext in valid_images:
                    return await convert_to_pdf(file_path)
                return file_path

            # Step 1: Download all files
            print("[STEP 1] Downloading files...")
            file_paths = await asyncio.gather(*[download_file(url) for url in files])
            form_paths = await asyncio.gather(*[download_file(url) for url in forms])

            # Step 2: Download docs (with names)
            print("[STEP 2] Downloading docs...")
            doc_paths = []
            for d in docs:
                name = d.get("name", "Document")
                url = d.get("url")
                if not url:
                    continue
                path = await download_file(url, name_hint=name)
                if path:
                    doc_paths.append((name, path))

            # Step 3: Convert images (only in files)
            print("[STEP 3] Converting images in 'files' only...")
            file_pdfs = [await ensure_pdf(f, is_image_only=True) for f in file_paths if f]
            form_pdfs = [await ensure_pdf(f) for f in form_paths if f]

            # Step 4: Create separator
            separator_pdf = os.path.join(temp_dir, "separator.pdf")
            await create_blank_page_pdf(separator_pdf, text="--- Supporting Documents ---")

            # Step 5: Merge PDFs (files + separator + forms)
            print("[STEP 5] Merging all PDFs...")
            merged_pdf = os.path.join(temp_dir, "final_copy.pdf")
            all_pdfs = [*file_pdfs, separator_pdf, *form_pdfs]
            all_pdfs = [p for p in all_pdfs if os.path.exists(p)]
            await merge_pdfs(all_pdfs, merged_pdf)

            # Step 6: Create DOCX (one section per doc entry)
            print("[STEP 6] Creating DOCX...")
            docx_path = os.path.join(temp_dir, "final_copy.docx")
            await create_docx_with_separators(doc_paths, docx_path)

            # Step 7: Move results to MEDIA_ROOT
            media_dir = os.path.join(settings.MEDIA_ROOT, "generated")
            os.makedirs(media_dir, exist_ok=True)
            final_pdf_path = os.path.join(media_dir, "final_copy.pdf")
            final_docx_path = os.path.join(media_dir, "final_copy.docx")
            shutil.move(merged_pdf, final_pdf_path)
            shutil.move(docx_path, final_docx_path)

            # Step 8: Return URLs
            pdf_url = request.build_absolute_uri(f"{settings.MEDIA_URL}generated/final_copy.pdf")
            docx_url = request.build_absolute_uri(f"{settings.MEDIA_URL}generated/final_copy.docx")

            print("[SUCCESS] Final outputs generated successfully")
            return JsonResponse({
                "final_pdf_url": pdf_url,
                "final_docx_url": docx_url
            })

    finally:
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"[WARN] Cleanup failed: {e}")
