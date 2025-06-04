import os
import json
import shutil
import tempfile
import requests
import urllib.parse
import cgi

from django.conf import settings
from django.http import HttpResponseBadRequest, JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt

from .utils import handle_doc_generation

def index(request):
    return HttpResponse('Welcome to DS-260 (Immigrant Visas) API!')

@csrf_exempt
def generate_doc(request):
    """
    Handle JSON uploads, download each allowed file URL to a temp directory,
    generate a single document based on one option, and return its download URL.
    """
    ALLOWED_NAMES = {
        "form-i-129","form-g-28","form-g-1145","form-i-907",
        "certified-lca","degree-evidence","employer-letter",
        "form-i-94","form-w2","form-i-797","form-w2-1099",
        "professional-certs"
    }

    if request.method != 'POST':
        return HttpResponseBadRequest('Invalid request method')

    try:
        payload = json.loads(request.body)
        options = payload.get("options", "")
        files = payload.get("files", [])
    except Exception as e:
        return HttpResponseBadRequest(f'Invalid JSON data: {e}')

    # Filter only allowed-named entries with a URL
    filtered = [
        f for f in files
        if isinstance(f, dict)
        and f.get("name") in ALLOWED_NAMES
        and f.get("url")
    ]
    if not filtered:
        return HttpResponseBadRequest('No allowed files provided')

    # Create a temporary directory to store downloads
    temp_dir = tempfile.mkdtemp()
    temp_paths = []
    try:
        for entry in filtered:
            url = entry["url"]
            resp = requests.get(url, stream=True, timeout=30)
            resp.raise_for_status()

            # Try to get filename from Content-Disposition header
            cd = resp.headers.get('content-disposition')
            if cd:
                _, params = cgi.parse_header(cd)
                filename = params.get('filename') or os.path.basename(urllib.parse.urlparse(url).path)
            else:
                filename = os.path.basename(urllib.parse.urlparse(url).path)

            # Determine suffix from path
            path = urllib.parse.urlparse(url).path
            suffix = os.path.splitext(path)[1]
            if not filename.endswith(suffix):
                filename = f"{filename}{suffix}"

            # Download into the temp directory
            local_path = os.path.join(temp_dir, filename)
            with open(local_path, 'wb') as tmp_file:
                for chunk in resp.iter_content(chunk_size=8192):
                    tmp_file.write(chunk)
            temp_paths.append(local_path)

        # Generate document(s) from the list of temp file paths
        generated_paths = handle_doc_generation(temp_paths, [options])
        if not generated_paths or len(generated_paths) != 1:
            return HttpResponseBadRequest('Document generation failed')

        # Move the generated file into MEDIA_ROOT/generated/
        gen_src = generated_paths[0]
        filename = os.path.basename(gen_src)
        media_dir = os.path.join(settings.MEDIA_ROOT, 'generated')
        os.makedirs(media_dir, exist_ok=True)
        final_path = os.path.join(media_dir, filename)
        os.replace(gen_src, final_path)

        # Build absolute download URL
        rel = os.path.join(settings.MEDIA_URL.lstrip('/'), 'generated', filename)
        download_url = request.build_absolute_uri(f"/{rel}")

        return JsonResponse({'download_url': download_url})

    finally:
        # Clean up the entire temp directory
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass



