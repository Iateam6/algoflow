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

from .utils import upload_to_pdfco, convert_to_pdf, merge_pdfs


def index(request):
    return HttpResponse('Final-copy API!')


@csrf_exempt
def final_copy(request):
    if request.method != 'POST':
        return HttpResponseBadRequest('Invalid request method')

    try:
        payload = json.loads(request.body)
        files = payload.get("files", [])
        if not files or not isinstance(files, list):
            return HttpResponseBadRequest('Invalid or missing "files" array')
    except Exception as e:
        print(f"[ERROR] Invalid JSON data: {e}")
        return HttpResponseBadRequest(f'Invalid JSON data: {e}')

    print(f"[INFO] Received {len(files)} file(s) to process.")

    # Run the async logic inside a helper
    return async_to_sync(process_final_copy)(request, files)


# 🔽 define async helper (can use await inside)
async def process_final_copy(request, files):
    temp_dir = tempfile.mkdtemp()
    temp_paths = []

    try:
        # Download all input files
        async with aiohttp.ClientSession() as session:
            for i, url in enumerate(files, start=1):
                print(f"[STEP 1] Downloading file {i}: {url}")
                try:
                    async with session.get(url, timeout=60) as resp:
                        if resp.status != 200:
                            raise Exception(f"HTTP {resp.status}: {resp.reason}")

                        cd = resp.headers.get('content-disposition')
                        if cd:
                            _, params = cgi.parse_header(cd)
                            filename = params.get('filename') or os.path.basename(urllib.parse.urlparse(url).path)
                        else:
                            filename = os.path.basename(urllib.parse.urlparse(url).path)

                        path = urllib.parse.urlparse(url).path
                        suffix = os.path.splitext(path)[1]
                        if not filename.endswith(suffix):
                            filename = f"{filename}{suffix}"

                        local_path = os.path.join(temp_dir, filename)
                        async with aiofiles.open(local_path, 'wb') as tmp_file:
                            async for chunk in resp.content.iter_chunked(8192):
                                await tmp_file.write(chunk)

                        temp_paths.append(local_path)
                        print(f"[INFO] Saved to: {local_path}")

                except Exception as e:
                    print(f"[ERROR] Failed to download {url}: {e}")
                    return HttpResponseBadRequest(f'Error downloading {url}: {e}')

        # Upload and Convert docx,doc and image file
        pdf_urls = []
        for i, file_path in enumerate(temp_paths, start=1):
            print(f"[STEP 2] Uploading & converting file {i}/{len(temp_paths)}: {file_path}")
            try:
                file_name, ext = os.path.splitext(os.path.basename(file_path))
                uploaded_url = await upload_to_pdfco(file_path)
                print(f"[INFO] Uploaded to: {uploaded_url}")

                pdf_url = await convert_to_pdf(uploaded_url, ext, file_name)
                print(f"[INFO] Converted to PDF: {pdf_url}")

                pdf_urls.append(pdf_url)
                await asyncio.sleep(1)
            except Exception as e:
                print(f"[ERROR] Conversion failed for {file_path}: {e}")

        # Merge
        if not pdf_urls:
            return HttpResponseBadRequest("No valid PDFs found to merge.")

        output_pdf = os.path.join(temp_dir, "final_copy.pdf")
        await merge_pdfs(pdf_urls, output_pdf)

        # Save to MEDIA_ROOT
        media_dir = os.path.join(settings.MEDIA_ROOT, 'generated')
        os.makedirs(media_dir, exist_ok=True)
        final_path = os.path.join(media_dir, "final_copy.pdf")
        shutil.move(output_pdf, final_path)

        # Return download URL
        rel_path = os.path.join(settings.MEDIA_URL.lstrip('/'), 'generated', 'final_copy.pdf')
        download_url = request.build_absolute_uri(f"/{rel_path}")

        print(f"[SUCCESS] Final file available at: {download_url}")
        return JsonResponse({'download_url': download_url})

    finally:
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"[WARN] Cleanup failed: {e}")
