from __future__ import annotations

import logging
import os
import tempfile
from urllib.parse import urljoin, urlsplit

from django.conf import settings

from case_jobs.api.validators import validate_generation_request
from case_jobs.exceptions import CaseJobError
from case_jobs.jobs.progress import ProgressReporter
from case_jobs.pipeline.cleanup import cleanup_job_directory
from case_jobs.pipeline.extraction import extract_sources
from case_jobs.pipeline.file_downloader import download_sources
from case_jobs.pipeline.generation import generate_document
from case_jobs.pipeline.identity import build_identity_record
from case_jobs.pipeline.verification import VerificationAgent
from case_jobs.retrieval.corpus import build_case_chunks, build_corpus_hash


logger = logging.getLogger(__name__)

_VISA_URL_SEGMENT_OVERRIDES = {
    "o-1": "o1",
}


def _public_base_url() -> str:
    explicit = str(getattr(settings, "PUBLIC_BASE_URL", "") or "").strip()
    if explicit:
        return explicit
    webhook_root = str(getattr(settings, "WEBHOOK_ROOT_URL", "") or "").strip()
    if not webhook_root:
        return ""
    parsed = urlsplit(webhook_root)
    if not parsed.scheme or not parsed.netloc:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}"


def _visa_url_segment(visa_type: str) -> str:
    key = (visa_type or "").strip().lower()
    if not key:
        raise ValueError("visa_type is required to build a download URL")
    return _VISA_URL_SEGMENT_OVERRIDES.get(key, key)


def _relative_media_path(path: str) -> str:
    relative_path = os.path.relpath(path, str(settings.MEDIA_ROOT)).replace(os.sep, "/")
    if relative_path.startswith("../"):
        raise ValueError("Generated output is outside MEDIA_ROOT")
    return relative_path


def _job_download_url(*, visa_type: str, job_id: str) -> str:
    segment = _visa_url_segment(visa_type)
    relative = f"/api/{segment}/generate_doc/{job_id}/download/"
    base = _public_base_url()
    return urljoin(f"{base.rstrip('/')}/", relative.lstrip("/")) if base else relative
def _submitted_request_context(request) -> dict:
    request_dict = request.to_dict()
    context = {
        "beneficiary": request_dict.get("beneficiary"),
        "petitioner": request_dict.get("petitioner"),
        "preparer": request_dict.get("preparer"),
    }
    if request.exhibits:
        context["exhibits"] = request_dict.get("exhibits")
    return context


def _is_structured_exhibit_list(request) -> bool:
    return request.document_type == "Exhibit List" and bool(request.exhibits)


def _ingestion_sources(request) -> list[tuple[str, str | None]]:
    if not request.exhibits:
        return [(source.url, None) for source in request.files]

    sources: list[tuple[str, str | None]] = []
    for exhibit in request.exhibits:
        for item in exhibit.items:
            if item.type in {"form", "document"} and item.url:
                sources.append((item.url, item.name))
                continue
            file_count = len(item.files)
            for file_index, source in enumerate(item.files, start=1):
                preferred_name = (
                    item.name if file_count == 1 else f"{item.name} {file_index}"
                )
                sources.append((source.url, preferred_name))
    return sources


def execute_generation_job(job: dict) -> dict:
    tenant_id = job["tenant_id"]
    job_id = job["job_id"]
    reporter = ProgressReporter(tenant_id, job_id)
    work_dir = tempfile.mkdtemp(prefix=f"case-job-{job_id}-")
    current_stage = "queued"
    try:
        request = validate_generation_request(job["request"], job["visa_type"])
        submitted_parties = _submitted_request_context(request)
        logger.info(
            "job start job_id=%s tenant_id=%s visa_type=%s case_id=%s document_type=%s document_slug=%s",
            job_id,
            tenant_id,
            job.get("visa_type"),
            request.case_id,
            request.document_type,
            request.document_slug,
        )
        structured_exhibit_list = _is_structured_exhibit_list(request)
        if structured_exhibit_list:
            manifests = []
            identity = None
            evidence_text = ""
            logger.info(
                "structured exhibit list skips downloads and RAG job_id=%s tenant_id=%s",
                job_id,
                tenant_id,
            )
        else:
            ingestion_sources = _ingestion_sources(request)
            current_stage = "reading_uploaded_files"
            reporter.stage("reading_uploaded_files")
            manifests = download_sources(
                [source[0] for source in ingestion_sources],
                work_dir,
                job_id=job_id,
                tenant_id=tenant_id,
                preferred_names=[source[1] for source in ingestion_sources],
            )
            logger.info(
                "file download completion job_id=%s tenant_id=%s source_file_count=%d",
                job_id,
                tenant_id,
                len(manifests),
            )

            current_stage = "extracting_content"
            reporter.stage("extracting_content")
            extracted = extract_sources(manifests, job_id=job_id, tenant_id=tenant_id)
            logger.info(
                "content extraction completion job_id=%s tenant_id=%s extracted_source_count=%d",
                job_id,
                tenant_id,
                len(extracted),
            )

            current_stage = "verifying_identity"
            reporter.stage("verifying_identity")
            identity = build_identity_record(request, extracted)
            logger.info(
                "identity verification completion job_id=%s tenant_id=%s supporting_source_hits=%d",
                job_id,
                tenant_id,
                len(identity.supporting_sources),
            )

            current_stage = "building_search_indexes"
            reporter.stage("building_search_indexes")
            corpus_hash = build_corpus_hash(
                tenant_id=tenant_id,
                case_id=request.case_id,
                source_hashes=[item["file_hash"] for item in extracted],
            )
            chunks = build_case_chunks(extracted, tenant_id, request.case_id)

            current_stage = "retrieving_evidence"
            reporter.stage("retrieving_evidence", corpus_hash=corpus_hash)
            evidence_text = "\n\n".join(chunk.text for chunk in chunks)
            logger.info(
                "evidence chunk preparation completion job_id=%s tenant_id=%s evidence_chunk_count=%d",
                job_id,
                tenant_id,
                len(chunks),
            )

        current_stage = "generating_document"
        reporter.stage("generating_document")
        output_path = generate_document(
            job["visa_type"],
            manifests,
            request.document_type,
            tenant_id=tenant_id,
            case_id=request.case_id,
            job_id=job_id,
            submitted_parties=submitted_parties,
        )
        logger.info(
            "document generation completion job_id=%s tenant_id=%s output_name=%s",
            job_id,
            tenant_id,
            os.path.basename(output_path),
        )

        if getattr(settings, "ENABLE_DOCUMENT_VERIFICATION", False):
            current_stage = "verifying_document"
            reporter.stage("verifying_document")
            verifier = VerificationAgent()
            verification = verifier.verify(output_path, request, identity, evidence_text)
            if not verification.passed:
                logger.warning(
                    "verification failed first pass job_id=%s tenant_id=%s case_id=%s document_type=%s corrections=%s",
                    job_id,
                    tenant_id,
                    request.case_id,
                    request.document_type,
                    list(verification.corrections)[:10],
                )
                output_path = generate_document(
                    job["visa_type"],
                    manifests,
                    request.document_type,
                    tenant_id=tenant_id,
                    case_id=request.case_id,
                    job_id=job_id,
                    corrections=list(verification.corrections),
                    submitted_parties=submitted_parties,
                )
                logger.info(
                    "document generation completion job_id=%s tenant_id=%s output_name=%s pass=%s",
                    job_id,
                    tenant_id,
                    os.path.basename(output_path),
                    "corrections_applied",
                )
                verification = verifier.verify(output_path, request, identity, evidence_text)
                if not verification.passed:
                    logger.error(
                        "verification failed second pass job_id=%s tenant_id=%s case_id=%s document_type=%s corrections=%s",
                        job_id,
                        tenant_id,
                        request.case_id,
                        request.document_type,
                        list(verification.corrections)[:10],
                    )
                    raise CaseJobError("Generated document failed verification twice")
        else:
            logger.info(
                "document verification skipped job_id=%s tenant_id=%s case_id=%s document_type=%s",
                job_id,
                tenant_id,
                request.case_id,
                request.document_type,
            )

        # Current visa adapters still create DOCX files locally. S3 is not
        # imported here and the uploading_generated_file stage is intentionally
        # reserved for the later storage integration.
        current_stage = "creating_docx"
        reporter.stage("creating_docx")
        download_url = _job_download_url(
            visa_type=job.get("visa_type", ""),
            job_id=job_id,
        )
        output_relative_path = _relative_media_path(output_path)
        logger.info(
            "job completion job_id=%s tenant_id=%s download_url=%s",
            job_id,
            tenant_id,
            download_url,
        )
        print(f"download_url={download_url}", flush=True)
        return reporter.stage(
            "completed",
            download_url=download_url,
            output_relative_path=output_relative_path,
            beneficiary_name=request.beneficiary.full_name,
            petitioner_name=request.petitioner.full_name,
            attorney_name=request.preparer.full_name,
            error_code=None,
            error_message=None,
        )
    except CaseJobError as exc:
        logger.error(
            "job failed job_id=%s tenant_id=%s stage=%s error_code=%s error_message=%s",
            job_id,
            tenant_id,
            current_stage,
            exc.code,
            str(exc),
        )
        reporter.failed(exc.code, str(exc))
        raise
    except Exception:
        logger.exception(
            "job failed job_id=%s tenant_id=%s stage=%s error_code=%s",
            job_id,
            tenant_id,
            current_stage,
            "GENERATION_ERROR",
        )
        reporter.failed("GENERATION_ERROR", "Document generation failed")
        raise
    finally:
        cleanup_job_directory(work_dir)
