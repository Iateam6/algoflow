import os
import json
import shutil
import tempfile
from django.conf import settings
from django.http import JsonResponse, HttpResponseBadRequest, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from asgiref.sync import async_to_sync
from .utils import (
    convert_to_pdf,
    merge_pdfs,
    create_blank_page_pdf,
    create_docx_with_separators,
)
from PIL import Image


def index(request):
    return HttpResponse("Final-copy API (file upload mode)!")


@csrf_exempt
def final_copy(request):
    if request.method != "POST":
        return HttpResponseBadRequest("Invalid request method")

    try:
        docs_json = request.POST.get("docs", "[]")
        docs = json.loads(docs_json) if docs_json else []
        uploaded_files = request.FILES.getlist("files")
        uploaded_forms = request.FILES.getlist("forms")

        if not uploaded_files and not uploaded_forms:
            return HttpResponseBadRequest("No files or forms provided")

        return async_to_sync(process_final_copy)(request, docs, uploaded_files, uploaded_forms)

    except Exception as e:
        print(f"[ERROR] Failed to process request: {e}")
        return HttpResponseBadRequest(f"Error: {e}")


async def process_final_copy(request, docs, uploaded_files, uploaded_forms):
    print("[DEBUG] Starting local file upload processing...")
    temp_dir = tempfile.mkdtemp()
    print("[DEBUG] Temporary directory created:", temp_dir)

    try:
        # Save uploaded files locally
        async def save_uploaded_file(django_file):
            local_path = os.path.join(temp_dir, django_file.name)
            with open(local_path, "wb+") as f:
                for chunk in django_file.chunks():
                    f.write(chunk)
            print(f"[OK] Saved: {local_path}")
            return local_path

        print("[DEBUG] Saving uploaded files...")
        file_paths = [await save_uploaded_file(f) for f in uploaded_files]
        form_paths = [await save_uploaded_file(f) for f in uploaded_forms]

        # Step 2: Convert images in 'files' to PDF
        print("[DEBUG] Converting only image files to PDF...")
        file_pdfs = []
        for path in file_paths:
            ext = os.path.splitext(path)[1].lower()
            if ext in [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"]:
                pdf_path = await convert_to_pdf(path)
                file_pdfs.append(pdf_path)
            else:
                file_pdfs.append(path)

        form_pdfs = form_paths  # No image conversion for forms

        # Step 3: Create a separator page
        separator_pdf = os.path.join(temp_dir, "separator.pdf")
        await create_blank_page_pdf(separator_pdf, text="--- Supporting Documents ---")

        # Step 4: Merge all PDFs
        print("[DEBUG] Merging all PDFs...")
        merged_pdf_path = os.path.join(temp_dir, "final_copy.pdf")
        all_pdfs = [*file_pdfs, separator_pdf, *form_pdfs]
        all_pdfs = [p for p in all_pdfs if os.path.exists(p)]
        await merge_pdfs(all_pdfs, merged_pdf_path)

        # Step 5: Generate DOCX
        print("[DEBUG] Creating DOCX with separators...")
        docx_path = os.path.join(temp_dir, "final_copy.docx")
        await create_docx_with_separators(docs, docx_path)

        # Step 6: Move results to MEDIA_ROOT
        media_dir = os.path.join(settings.MEDIA_ROOT, "generated")
        os.makedirs(media_dir, exist_ok=True)
        final_pdf_path = os.path.join(media_dir, "final_copy.pdf")
        final_docx_path = os.path.join(media_dir, "final_copy.docx")

        shutil.move(merged_pdf_path, final_pdf_path)
        shutil.move(docx_path, final_docx_path)

        # Step 7: Build response URLs
        pdf_url = request.build_absolute_uri(
            f"{settings.MEDIA_URL}generated/final_copy.pdf"
        )
        docx_url = request.build_absolute_uri(
            f"{settings.MEDIA_URL}generated/final_copy.docx"
        )

        print("[SUCCESS] Final PDF and DOCX created successfully")
        return JsonResponse({
            "final_pdf_url": pdf_url,
            "final_docx_url": docx_url
        })

    finally:
        try:
            shutil.rmtree(temp_dir)
            print("[CLEANUP] Temporary directory removed")
        except Exception as e:
            print(f"[WARN] Cleanup failed: {e}")
