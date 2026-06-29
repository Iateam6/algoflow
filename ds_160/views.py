import cgi
import hashlib
import json
import logging
import mimetypes
import os
import shutil
import tempfile
import urllib.parse

import requests
from django.conf import settings
from django.http import HttpResponse, HttpResponseBadRequest, JsonResponse
from django.views.decorators.csrf import csrf_exempt

from immigration_algoflow_APIs.file_inputs import (
    INVALID_FILE_URLS_ERROR,
    build_url_source_entries,
)

from .utils import handle_doc_generation


logger = logging.getLogger(__name__)

def index(request):
    return HttpResponse('Welcome to DS-260 (Immigrant Visas) API!')

def build_download_filename(response: requests.Response, url: str) -> str:
    content_disposition = response.headers.get("content-disposition")
    if content_disposition:
        _, params = cgi.parse_header(content_disposition)
        filename = params.get("filename") or os.path.basename(urllib.parse.urlparse(url).path)
    else:
        filename = os.path.basename(urllib.parse.urlparse(url).path)

    if not filename:
        filename = "uploaded_document"

    path = urllib.parse.urlparse(url).path
    suffix = os.path.splitext(path)[1]
    if not suffix:
        content_type = (response.headers.get("content-type") or "").split(";")[0].strip()
        guessed_extension = mimetypes.guess_extension(content_type or "")
        suffix = guessed_extension or ""

    if suffix and not filename.endswith(suffix):
        filename = f"{filename}{suffix}"

    return filename


def sha256_of_file(file_path: str) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_download_manifest(
    entry: dict,
    response: requests.Response,
    local_path: str,
    original_filename: str,
) -> dict:
    content_type = (response.headers.get("content-type") or "").split(";")[0].strip()
    extension = os.path.splitext(original_filename)[1].lower()
    if not extension:
        extension = mimetypes.guess_extension(content_type) or ""

    return {
        "name": entry["name"],
        "original_filename": original_filename,
        "url": entry["url"],
        "content_type": content_type,
        "extension": extension.lower(),
        "local_path": local_path,
        "file_hash": sha256_of_file(local_path),
    }


@csrf_exempt
def generate_doc(request):
    """
    Handle JSON URL uploads, download each file URL to a temp directory,
    generate a single document based on one option, and return its download URL.
    """
    if request.method != "POST":
        return HttpResponseBadRequest("Invalid request method")

    try:
        payload = json.loads(request.body)
        options = payload.get("options", "")
        files = payload.get("files")
    except Exception as exc:
        return HttpResponseBadRequest(f"Invalid JSON data: {exc}")

    filtered = build_url_source_entries(files)
    if not filtered:
        return JsonResponse(
            {
                "download_url": None,
                "error message": INVALID_FILE_URLS_ERROR,
            }
        )

    temp_dir = tempfile.mkdtemp()
    file_manifests: list[dict] = []

    try:
        for index, entry in enumerate(filtered, start=1):
            response = requests.get(entry["url"], stream=True, timeout=30)
            response.raise_for_status()

            original_filename = build_download_filename(response, entry["url"])
            local_filename = f"source_{index}{os.path.splitext(original_filename)[1]}"
            local_path = os.path.join(temp_dir, local_filename)

            with open(local_path, "wb") as temp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)

            file_manifest = build_download_manifest(
                entry=entry,
                response=response,
                local_path=local_path,
                original_filename=original_filename,
            )
            file_manifests.append(file_manifest)

        logger.info("Downloaded %s files for DS-160 generation", len(file_manifests))

        generated_paths = handle_doc_generation(file_manifests, [options])
        if not generated_paths or len(generated_paths) != 1:
            return JsonResponse(
                {
                    "download_url": None,
                    "error message": "Document generation failed",
                }
            )

        generated_source_path = generated_paths[0]
        filename = os.path.basename(generated_source_path)
        media_dir = os.path.join(settings.MEDIA_ROOT, "generated")
        os.makedirs(media_dir, exist_ok=True)
        final_path = os.path.join(media_dir, filename)

        if os.path.abspath(generated_source_path) != os.path.abspath(final_path):
            os.replace(generated_source_path, final_path)

        rel_path = os.path.join(settings.MEDIA_URL.lstrip("/"), "generated", filename)
        download_url = request.build_absolute_uri(f"/{rel_path}")

        return JsonResponse(
            {
                "download_url": download_url,
                "error message": None,
            }
        )
    except requests.RequestException:
        logger.exception("Failed to download one or more DS-160 source documents.")
        return JsonResponse(
            {
                "download_url": None,
                "error message": "Document generation failed",
            }
        )
    except Exception:
        logger.exception("Unexpected error while generating an DS-160 document.")
        return JsonResponse(
            {
                "download_url": None,
                "error message": "Document generation failed",
            }
        )
    finally:
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            logger.warning("Failed to clean temporary DS-160 directory %s", temp_dir)

