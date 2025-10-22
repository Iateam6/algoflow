import os
import json
import shutil
import tempfile
import requests
import urllib.parse
import cgi
import time

from django.conf import settings
from django.http import HttpResponseBadRequest, JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt

from .utils import merge_files, convert_doc_to_pdf


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
        return HttpResponseBadRequest(f'Invalid JSON data: {e}')

    # Create temporary directory for downloads
    temp_dir = tempfile.mkdtemp()
    temp_paths = []

    try:
        # Download each file
        for url in files:
            try:
                resp = requests.get(url, stream=True, timeout=60)
                resp.raise_for_status()
            except Exception as e:
                return HttpResponseBadRequest(f'Error downloading {url}: {e}')

            # Extract filename
            cd = resp.headers.get('content-disposition')
            if cd:
                _, params = cgi.parse_header(cd)
                filename = params.get('filename') or os.path.basename(urllib.parse.urlparse(url).path)
            else:
                filename = os.path.basename(urllib.parse.urlparse(url).path)

            # Ensure proper extension
            path = urllib.parse.urlparse(url).path
            suffix = os.path.splitext(path)[1]
            if not filename.endswith(suffix):
                filename = f"{filename}{suffix}"

            # Save to temp directory
            local_path = os.path.join(temp_dir, filename)
            with open(local_path, 'wb') as tmp_file:
                for chunk in resp.iter_content(chunk_size=8192):
                    tmp_file.write(chunk)

            temp_paths.append(local_path)

        # Convert DOC/DOCX files to PDF
        converted_paths = []
        for file_path in temp_paths:
            ext = os.path.splitext(file_path)[1].lower()
            if ext in ['.doc', '.docx']:
                try:
                    if not os.path.exists(file_path):
                        print(f"[WARN] File missing before conversion: {file_path}")
                        continue

                    pdf_path = convert_doc_to_pdf(file_path, temp_dir)
                    time.sleep(1)  # allow Word COM to close properly

                    if pdf_path and os.path.exists(pdf_path):
                        print(f"[SUCCESS] Converted: {pdf_path}")
                        converted_paths.append(pdf_path)
                        os.remove(file_path)  # remove original DOC/DOCX
                    else:
                        print(f"[ERROR] Conversion failed for {file_path}, skipping.")
                except Exception as e:
                    print(f"[ERROR] Error converting {file_path}: {e}")
            elif ext == '.pdf':
                converted_paths.append(file_path)

        # Merge all PDFs into one big file
        if not converted_paths:
            return HttpResponseBadRequest("No valid PDF files found to merge.")

        output_pdf = os.path.join(temp_dir, "final_copy.pdf")
        merge_files(converted_paths, output_pdf)

        # Move merged PDF to MEDIA_ROOT/generated/
        media_dir = os.path.join(settings.MEDIA_ROOT, 'generated')
        os.makedirs(media_dir, exist_ok=True)
        final_path = os.path.join(media_dir, "final_copy.pdf")
        shutil.move(output_pdf, final_path)

        # Build absolute download URL
        rel_path = os.path.join(settings.MEDIA_URL.lstrip('/'), 'generated', 'final_copy.pdf')
        download_url = request.build_absolute_uri(f"/{rel_path}")

        return JsonResponse({'download_url': download_url})

    finally:
        # Cleanup temporary directory
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass
