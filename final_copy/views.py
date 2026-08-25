import os
import json
import shutil
import tempfile
import urllib.parse
import cgi
import asyncio
import zipfile
import aiohttp
import aiofiles
from typing import Any, Dict, List
from django.conf import settings
from django.http import JsonResponse, HttpResponseBadRequest, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
from asgiref.sync import async_to_sync
from immigration_algoflow_APIs.media_cleanup import delete_file_if_exists
from .utils import convert_to_pdf, merge_pdfs, create_blank_page_pdf, convert_docx_to_pdf, stamp_pdf_with_firm_name


def _infer_extension_from_content_type(content_type: str) -> str | None:
    ct = (content_type or "").split(";")[0].strip().lower()
    if ct == "application/pdf":
        return ".pdf"
    if ct == "application/msword":
        return ".doc"
    if ct == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return ".docx"
    if ct == "image/jpeg":
        return ".jpg"
    if ct == "image/png":
        return ".png"
    if ct == "image/gif":
        return ".gif"
    if ct == "image/bmp":
        return ".bmp"
    if ct in ("image/tiff", "image/tif"):
        return ".tiff"
    return None


def _maybe_fix_extension_by_signature(file_path: str) -> str:
    """
    Best-effort extension fix when the download URL has no filename/extension.
    This helps Word conversion and PDF merge behave deterministically.
    """
    try:
        if not file_path or not os.path.exists(file_path):
            return file_path

        base, ext = os.path.splitext(file_path)
        if ext and ext.lower() not in (".bin", ".dat"):
            return file_path

        with open(file_path, "rb") as f:
            head = f.read(8)

        # PDF signature
        if head.startswith(b"%PDF-"):
            new_path = base + ".pdf"
            if new_path != file_path and not os.path.exists(new_path):
                os.replace(file_path, new_path)
                return new_path
            return file_path

        # DOCX is a zip; check for Word document markers
        if head.startswith(b"PK\x03\x04"):
            try:
                with zipfile.ZipFile(file_path) as zf:
                    names = set(zf.namelist())
                if "word/document.xml" in names:
                    new_path = base + ".docx"
                    if new_path != file_path and not os.path.exists(new_path):
                        os.replace(file_path, new_path)
                        return new_path
            except Exception:
                pass

        # Legacy DOC (OLE Compound File signature)
        if head.startswith(b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1"):
            new_path = base + ".doc"
            if new_path != file_path and not os.path.exists(new_path):
                os.replace(file_path, new_path)
                return new_path

        return file_path
    except Exception:
        return file_path


def parse_final_copy_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Final Copy API v2 payload:
      {
        "case_id": str,
        "document_type": str?,
        "document_slug": str?,
        "visa_type": str,
        "preparer": {"firm_name": str, ...},
        "petitioner": {"full_name": str, ...},
        "beneficiary": {"full_name": str, ...},
        "cover_letter": null | str | {"type"?: "document", "name"?: str, "url": str},
        "exhibits": [
          {
            "number": int,
            "title": str,
            "items": [
              {"type": "form"|"document", "name": str, "url": str}
              {"type": "file", "name": str, "files": [{"url": str}, ...]}
            ]
          }, ...
        ]
      }

    Normalized output:
      {
        "cover_lines": [str, ...],
        "front_matter": [{"url": str, "name_hint": str}],
        "exhibits": [{"number": int, "divider_lines": [str, ...], "items": [{"url": str, "name_hint": str}]}],
      }
    """
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a JSON object")

    # Reject previous payload variants explicitly.
    if "docs" in payload or "forms" in payload:
        raise ValueError("Invalid payload: legacy docs/forms/files payload is no longer supported.")
    if "files" in payload and "exhibits" not in payload:
        raise ValueError("Invalid payload: top-level files[] payload is no longer supported; use exhibits[].")

    case_id = payload.get("case_id")
    visa_type = payload.get("visa_type")
    preparer = payload.get("preparer") or {}
    petitioner = payload.get("petitioner") or {}
    beneficiary = payload.get("beneficiary") or {}
    exhibits = payload.get("exhibits")

    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("Invalid payload: case_id is required.")
    if not isinstance(visa_type, str) or not visa_type.strip():
        raise ValueError("Invalid payload: visa_type is required.")
    if not isinstance(preparer, dict) or not isinstance(preparer.get("firm_name"), str) or not preparer["firm_name"].strip():
        raise ValueError("Invalid payload: preparer.firm_name is required.")
    if not isinstance(petitioner, dict) or not isinstance(petitioner.get("full_name"), str) or not petitioner["full_name"].strip():
        raise ValueError("Invalid payload: petitioner.full_name is required.")
    if not isinstance(beneficiary, dict) or not isinstance(beneficiary.get("full_name"), str) or not beneficiary["full_name"].strip():
        raise ValueError("Invalid payload: beneficiary.full_name is required.")
    if not isinstance(exhibits, list):
        raise ValueError("Invalid payload: exhibits[] is required.")

    law_firm_name = preparer["firm_name"].strip()
    petitioner_name = petitioner["full_name"].strip()
    beneficiary_name = beneficiary["full_name"].strip()

    server_date = timezone.localdate().strftime("%B %d, %Y")
    cover_lines = [
        law_firm_name,
        f"I-129 ({visa_type.strip()}) APPLICATION",
        f"PETITIONER: {petitioner_name}",
        f"BENEFICIARY: {beneficiary_name}",
        f"CASE ID: {case_id.strip()}",
        f"DATE: {server_date}",
    ]

    front_matter: List[Dict[str, str]] = []

    cover_letter = payload.get("cover_letter")
    cover_letter_from_top_level = False
    if cover_letter is None:
        cover_letter_from_top_level = False
    elif isinstance(cover_letter, str):
        if not cover_letter.strip():
            raise ValueError("Invalid payload: cover_letter must be a non-empty URL string.")
        front_matter.append({"url": cover_letter.strip(), "name_hint": "Cover Letter"})
        cover_letter_from_top_level = True
    elif isinstance(cover_letter, dict):
        url = cover_letter.get("url")
        if not isinstance(url, str) or not url.strip():
            raise ValueError("Invalid payload: cover_letter.url is required.")
        name_hint = cover_letter.get("name")
        name_hint = str(name_hint).strip() if isinstance(name_hint, str) and name_hint.strip() else "Cover Letter"
        front_matter.append({"url": url.strip(), "name_hint": name_hint})
        cover_letter_from_top_level = True
    else:
        raise ValueError("Invalid payload: cover_letter must be null, a URL string, or an object with url.")

    # Backward-compat: if no top-level cover_letter is provided, extract the first
    # exhibit item of type=document with name='Cover Letter' into front matter.
    cover_letter_taken = False
    exhibits_copy: List[Dict[str, Any]] = []
    for ex in exhibits:
        if not isinstance(ex, dict):
            continue
        items = ex.get("items", [])
        if not isinstance(items, list):
            items = []
        new_items: List[Dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            if (
                not cover_letter_from_top_level
                and
                not cover_letter_taken
                and str(item.get("type") or "").lower() == "document"
                and str(item.get("name") or "").strip().lower() == "cover letter"
            ):
                url = item.get("url")
                if isinstance(url, str) and url.strip():
                    front_matter.append({"url": url.strip(), "name_hint": "Cover Letter"})
                    cover_letter_taken = True
                continue  # remove from exhibit
            new_items.append(item)

        ex_copy = dict(ex)
        ex_copy["items"] = new_items
        exhibits_copy.append(ex_copy)

    normalized_exhibits: List[Dict[str, Any]] = []
    for ex in exhibits_copy:
        number = ex.get("number")
        title = ex.get("title")
        items = ex.get("items", [])

        if not isinstance(number, int):
            raise ValueError("Invalid payload: each exhibit.number must be an integer.")
        if title is None:
            title = ""
        if not isinstance(title, str):
            raise ValueError("Invalid payload: each exhibit.title must be a string.")
        if not isinstance(items, list):
            raise ValueError("Invalid payload: each exhibit.items must be a list.")

        flattened_items: List[Dict[str, str]] = []
        first_item_name: str | None = None

        for item in items:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "").lower()
            item_name = str(item.get("name") or "").strip() or "File"
            if first_item_name is None and item_name:
                first_item_name = item_name

            if item_type in ("form", "document"):
                url = item.get("url")
                if not isinstance(url, str) or not url.strip():
                    raise ValueError(f"Invalid payload: {item_type} item '{item_name}' is missing url.")
                flattened_items.append({"url": url.strip(), "name_hint": item_name})
                continue

            if item_type == "file":
                nested = item.get("files")
                if not isinstance(nested, list) or not nested:
                    raise ValueError(f"Invalid payload: file item '{item_name}' must include files[].")
                for nested_item in nested:
                    if not isinstance(nested_item, dict):
                        continue
                    url = nested_item.get("url")
                    if not isinstance(url, str) or not url.strip():
                        raise ValueError(f"Invalid payload: file item '{item_name}' has an empty url.")
                    flattened_items.append({"url": url.strip(), "name_hint": item_name})
                continue

            raise ValueError(f"Invalid payload: unsupported item.type '{item.get('type')}'.")

        title_or_fallback = title.strip() or (first_item_name or "").strip() or "Supporting Documents"
        divider_lines = [
            law_firm_name,
            f"EXHIBIT {number}",
            title_or_fallback,
        ]

        normalized_exhibits.append(
            {
                "number": number,
                "divider_lines": divider_lines,
                "items": flattened_items,
            }
        )

    normalized_exhibits.sort(key=lambda e: int(e["number"]))

    return {
        "cover_lines": cover_lines,
        "front_matter": front_matter,
        "exhibits": normalized_exhibits,
    }


def index(request):
    return HttpResponse("Final-copy API !")


@csrf_exempt
def final_copy(request):
    """Main entry for merging and generating final copy documents."""
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=400)

    try:
        payload = json.loads(request.body)
        print("[DEBUG] Payload received:", payload)
    except Exception as e:
        return JsonResponse({"error": f"Invalid JSON payload: {e}"}, status=400)

    try:
        normalized = parse_final_copy_payload(payload)
    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=400)

    return async_to_sync(process_final_copy)(request, normalized)


# ------------------- ASYNC HANDLER -------------------
async def process_final_copy(request, normalized: Dict[str, Any]):
    print("[DEBUG] Processing final copy from URLs...")
    # Use MEDIA_ROOT for temp work to avoid restricted OS temp locations in some deployments/test runners.
    media_tmp_root = os.path.join(settings.MEDIA_ROOT, "tmp")
    os.makedirs(media_tmp_root, exist_ok=True)
    temp_dir = tempfile.mkdtemp(dir=media_tmp_root)
    print(f"[DEBUG] Temporary directory: {temp_dir}")

    try:
        async with aiohttp.ClientSession() as session:
            download_cache: Dict[str, Dict[str, str]] = {}

            async def download_file(url, name_hint=None):
                """Download a file from a URL to the temporary directory."""
                if not url:
                    print("[WARN] Empty URL encountered — skipping download.")
                    return None
                if url in download_cache:
                    return download_cache[url]
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
 
                        content_type = (resp.headers.get("content-type") or "").lower()
                        inferred_ext = _infer_extension_from_content_type(content_type)
                        cd = resp.headers.get("content-disposition")
                        if cd:
                            _, params = cgi.parse_header(cd)
                            filename = params.get("filename") or os.path.basename(urllib.parse.urlparse(url).path)
                        else:
                            filename = os.path.basename(urllib.parse.urlparse(url).path)

                        # Fallback filename if missing
                        if not filename:
                            filename = f"{name_hint or 'file'}{inferred_ext or '.pdf'}"

                        base, ext = os.path.splitext(filename)
                        if not ext:
                            ext = inferred_ext or ".pdf"
                            filename = f"{base}{ext}"
                        elif inferred_ext in (".doc", ".docx") and ext.lower() not in (".doc", ".docx"):
                            # Storage layers sometimes return a generic filename; prefer Word type when known.
                            filename = f"{base}{inferred_ext}"
                            ext = inferred_ext
                        candidate = os.path.join(temp_dir, filename)
                        counter = 2
                        while os.path.exists(candidate):
                            candidate = os.path.join(temp_dir, f"{base}_{counter}{ext}")
                            counter += 1
                        local_path = candidate
                        async with aiofiles.open(local_path, "wb") as f:
                            async for chunk in resp.content.iter_chunked(8192):
                                await f.write(chunk)
                        local_path = _maybe_fix_extension_by_signature(local_path)
                        print(f"[OK] Downloaded: {local_path}")
                        download_cache[url] = {"path": local_path, "filename": filename}
                        return download_cache[url]
                except Exception as e:
                    print(f"[ERROR] Exception downloading {url}: {e}")
                    return None

            async def ensure_pdf(file_path):
                """Convert images and DOCX to PDF; return PDF path or None."""
                if not file_path or not os.path.exists(file_path):
                    return None
                file_path = _maybe_fix_extension_by_signature(file_path)
                ext = os.path.splitext(file_path)[1].lower()
                valid_images = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"]
                if ext in valid_images:
                    return await convert_to_pdf(file_path)
                if ext in (".docx", ".doc"):
                    converted = await asyncio.to_thread(convert_docx_to_pdf, file_path)
                    if not converted:
                        raise RuntimeError(
                            f"Word-to-PDF conversion failed for {os.path.basename(file_path)}. "
                            "Ensure LibreOffice (`soffice`) is installed and available on PATH."
                        )
                    return converted
                return file_path

            cover_lines = normalized.get("cover_lines", [])
            front_matter = normalized.get("front_matter", [])
            exhibits = normalized.get("exhibits", [])

            if not isinstance(cover_lines, list) or not all(isinstance(x, str) for x in cover_lines):
                raise ValueError("Invalid normalized payload: cover_lines")
            if not isinstance(front_matter, list):
                raise ValueError("Invalid normalized payload: front_matter")
            if not isinstance(exhibits, list):
                raise ValueError("Invalid normalized payload: exhibits")

            firm_name = (cover_lines[0] if cover_lines else "") or ""
            firm_name = str(firm_name).strip()

            # Build ordered render steps: generated pages + URLs.
            cover_pdf = os.path.join(temp_dir, "cover.pdf")
            await create_blank_page_pdf(cover_pdf, text="\n".join([l for l in cover_lines if str(l).strip()]))

            render_steps: List[Dict[str, Any]] = [{"kind": "pdf_path", "path": cover_pdf}]

            for fm in front_matter:
                if not isinstance(fm, dict):
                    continue
                url = fm.get("url")
                if isinstance(url, str) and url.strip():
                    render_steps.append({"kind": "url", "url": url.strip(), "name_hint": fm.get("name_hint")})

            for ex in exhibits:
                if not isinstance(ex, dict):
                    continue
                ex_num = ex.get("number")
                divider_lines = ex.get("divider_lines", [])
                ex_items = ex.get("items", [])

                if not isinstance(ex_num, int):
                    raise ValueError("Invalid normalized payload: exhibit.number")
                if not isinstance(divider_lines, list) or not all(isinstance(x, str) for x in divider_lines):
                    raise ValueError("Invalid normalized payload: exhibit.divider_lines")
                if not isinstance(ex_items, list):
                    raise ValueError("Invalid normalized payload: exhibit.items")

                divider_pdf = os.path.join(temp_dir, f"exhibit_{ex_num:03d}_divider.pdf")
                await create_blank_page_pdf(divider_pdf, text="\n".join([l for l in divider_lines if str(l).strip()]))
                render_steps.append({"kind": "pdf_path", "path": divider_pdf})

                for it in ex_items:
                    if not isinstance(it, dict):
                        continue
                    url = it.get("url")
                    if isinstance(url, str) and url.strip():
                        render_steps.append({"kind": "url", "url": url.strip(), "name_hint": it.get("name_hint")})

            # Step 1: Download URL inputs (order preserved by render_steps)
            print("[STEP 1] Downloading URL inputs...")
            steps_with_urls = [s for s in render_steps if s.get("kind") == "url" and s.get("url")]
            download_results = await asyncio.gather(
                *[download_file(s.get("url"), name_hint=s.get("name_hint")) for s in steps_with_urls]
            )

            # Step 2: Convert inputs to PDFs and build merge list
            print("[STEP 2] Preparing PDFs for merge...")
            merge_inputs: List[str] = []
            url_idx = 0
            for step in render_steps:
                kind = step.get("kind")
                if kind == "pdf_path":
                    p = step.get("path")
                    if isinstance(p, str) and p and os.path.exists(p):
                        merge_inputs.append(p)
                    continue

                if kind == "url":
                    downloaded = download_results[url_idx] if url_idx < len(download_results) else None
                    url_idx += 1
                    if not downloaded:
                        continue
                    input_path = downloaded.get("path")
                    if not input_path:
                        continue
                    converted = await ensure_pdf(input_path)
                    if converted and os.path.exists(converted):
                        stamped = await asyncio.to_thread(stamp_pdf_with_firm_name, converted, firm_name)
                        merge_inputs.append(stamped if stamped and os.path.exists(stamped) else converted)
                    continue

            # Step 3: Merge all PDFs safely
            print("[STEP 3] Merging all PDFs...")
            merged_pdf = os.path.join(temp_dir, "final_copy.pdf")

            # Preserve input ordering; let `merge_pdfs` validate and placeholder bad/empty PDFs.
            all_pdfs = [p for p in merge_inputs if p and os.path.exists(p)]

            if not all_pdfs:
                print("[WARN] No valid PDFs found to merge - creating placeholder.")
                placeholder = os.path.join(temp_dir, "placeholder.pdf")
                await create_blank_page_pdf(placeholder, text="No valid PDF content available")
                all_pdfs = [placeholder]

            await merge_pdfs(all_pdfs, merged_pdf)

            # Step 4: Move results to MEDIA_ROOT
            print("[STEP 4] Moving output files to MEDIA_ROOT...")
            media_dir = os.path.join(settings.MEDIA_ROOT, "generated")
            os.makedirs(media_dir, exist_ok=True)
            final_pdf_path = os.path.join(media_dir, "final_copy.pdf")
            final_docx_path = os.path.join(media_dir, "final_copy.docx")
            delete_file_if_exists(final_pdf_path)
            delete_file_if_exists(final_docx_path)
            shutil.move(merged_pdf, final_pdf_path)

            # Step 5: Return URLs
            pdf_url = request.build_absolute_uri(f"{settings.MEDIA_URL}generated/final_copy.pdf")
            print("[SUCCESS] Final PDF URL:", pdf_url)

            print("[SUCCESS] Final outputs generated successfully")
            return JsonResponse({
                "download_url": pdf_url,
            })

    except Exception as e:
        print(f"[FATAL] Error during processing: {e}")
        return JsonResponse({"error": str(e)}, status=500)

    finally:
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"[WARN] Cleanup failed: {e}")
