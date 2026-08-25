from __future__ import annotations

import base64
import logging
import mimetypes
import os
import tempfile

import docx2txt
import pypdfium2 as pdfium
from django.conf import settings
from openai import OpenAI
from openai.types.responses import (
    EasyInputMessageParam,
    ResponseInputImageParam,
    ResponseInputParam,
    ResponseInputTextParam,
)
from pypdf import PdfReader

from case_jobs.exceptions import DownloadError


logger = logging.getLogger(__name__)


def _needs_ocr(text: str) -> bool:
    return len(text.strip()) < 40 or len(text.split()) < 8


def _ocr_image(image_path: str, source_name: str, page_number: int) -> str:
    mime_type = mimetypes.guess_type(image_path)[0] or "image/png"
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("ascii")
    input_payload: ResponseInputParam = [
        EasyInputMessageParam(
            role="user",
            content=[
                ResponseInputTextParam(
                    type="input_text",
                    text=(
                        "Transcribe this immigration evidence page exactly. Preserve "
                        "names, labels, addresses, headings, lists, and tables. Do not "
                        "summarize, correct, infer, or invent text. Return plain text only. "
                        f"Source: {source_name}; page: {page_number}."
                    ),
                ),
                ResponseInputImageParam(
                    type="input_image",
                    detail="auto",
                    image_url=f"data:{mime_type};base64,{encoded}",
                ),
            ],
        )
    ]
    response = OpenAI(api_key=settings.OPENAI_API_KEY).responses.create(
        model=settings.OCR_MODEL,
        input=input_payload,
    )
    return (response.output_text or "").strip()


def _render_pdf_page(pdf_path: str, page_index: int, output_path: str) -> None:
    document = pdfium.PdfDocument(pdf_path)
    try:
        bitmap = document[page_index].render(scale=2)
        bitmap.to_pil().save(output_path)
    finally:
        document.close()


def extract_source_text(
    manifest: dict,
    *,
    job_id: str | None = None,
    tenant_id: str | None = None,
) -> dict:
    path = manifest["local_path"]
    content_type = manifest["content_type"]
    try:
        if content_type == "application/pdf":
            pages = []
            for page_index, page in enumerate(PdfReader(path).pages):
                text = page.extract_text() or ""
                if _needs_ocr(text):
                    logger.info(
                        "ocr used job_id=%s tenant_id=%s file_hash=%s page_number=%d",
                        job_id,
                        tenant_id,
                        manifest.get("file_hash"),
                        page_index + 1,
                    )
                    image_path = os.path.join(
                        tempfile.gettempdir(),
                        f"ocr-{manifest['file_hash']}-{page_index + 1}.png",
                    )
                    try:
                        _render_pdf_page(path, page_index, image_path)
                        text = _ocr_image(
                            image_path,
                            manifest.get("original_filename", "source.pdf"),
                            page_index + 1,
                        )
                    finally:
                        try:
                            os.remove(image_path)
                        except FileNotFoundError:
                            pass
                pages.append(text)
        elif content_type.endswith("wordprocessingml.document"):
            pages = [docx2txt.process(path) or ""]
        elif content_type == "text/plain":
            with open(path, encoding="utf-8", errors="replace") as handle:
                pages = [handle.read()]
        elif content_type in {"image/jpeg", "image/png"}:
            pages = [
                _ocr_image(
                    path,
                    manifest.get("original_filename", "source image"),
                    1,
                )
            ]
        else:
            raise DownloadError("Unsupported source type for extraction")
    except Exception as exc:
        raise DownloadError("Could not extract text from source file") from exc

    logger.info(
        "source extraction completion job_id=%s tenant_id=%s file_hash=%s page_count=%d content_type=%s",
        job_id,
        tenant_id,
        manifest.get("file_hash"),
        len(pages),
        content_type,
    )
    return {**manifest, "pages": pages, "text": "\n\n".join(pages)}


def extract_sources(
    manifests: list[dict],
    *,
    job_id: str | None = None,
    tenant_id: str | None = None,
) -> list[dict]:
    return [
        extract_source_text(manifest, job_id=job_id, tenant_id=tenant_id)
        for manifest in manifests
    ]
