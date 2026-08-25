from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, cast
from urllib.parse import urlsplit

from django.conf import settings
from django.http import FileResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt

from case_jobs.api.authentication import resolve_generation_principal
from case_jobs.api.validators import validate_generation_request
from case_jobs.constants import DOCX_CONTENT_TYPE
from case_jobs.exceptions import (
    CapacityExceeded,
    CaseJobError,
    ServiceConfigurationError,
    ValidationError,
)
from case_jobs.integrations.webhook_client import resolve_webhook_url
from case_jobs.jobs.idempotency import (
    IdempotencyStore,
    request_fingerprint,
    resolve_idempotency_key,
)
from case_jobs.jobs.job_store import JobStore
from case_jobs.jobs.rate_limits import CapacityLimiter
from case_jobs.tasks.pipeline_tasks import run_generation_pipeline
from case_jobs.tasks.webhook_tasks import queue_job_webhook


logger = logging.getLogger(__name__)


def _error_response(exc: CaseJobError) -> JsonResponse:
    response = JsonResponse(
        {"error_code": exc.code, "error_message": str(exc)},
        status=exc.status_code,
    )
    if isinstance(exc, CapacityExceeded):
        response["Retry-After"] = "30"
    return response


@csrf_exempt
def create_generation(request, visa_type: str):
    if request.method != "POST":
        return JsonResponse(
            {"error_code": "METHOD_NOT_ALLOWED", "error_message": "POST is required"},
            status=405,
        )
    limiter = CapacityLimiter()
    acquired = False
    idempotency_store = IdempotencyStore()
    job_store = JobStore()
    reservation_created = False
    explicit_idempotency_key = request.headers.get("Idempotency-Key", "").strip()
    idempotency_key = ""
    job = None
    try:
        principal = resolve_generation_principal(request.headers.get("Authorization"))
        if not settings.AI_DOC_WEBHOOK_SECRET:
            raise ServiceConfigurationError(
                "AI_DOC_WEBHOOK_SECRET is not configured"
            )
        if len(explicit_idempotency_key) > 255:
            raise ValidationError("Idempotency-Key cannot exceed 255 characters")
        try:
            payload = json.loads(request.body)
        except (TypeError, ValueError) as exc:
            raise ValidationError("Request body must contain valid JSON") from exc
        generation_request = validate_generation_request(payload, visa_type)
        fingerprint = request_fingerprint(
            {"visa_type": visa_type, **generation_request.to_dict()}
        )
        idempotency_key = resolve_idempotency_key(
            explicit_idempotency_key,
            fingerprint,
        )
        proposed_job_id = str(uuid.uuid4())
        reservation = idempotency_store.reserve(
            principal.tenant_id,
            idempotency_key,
            fingerprint,
            proposed_job_id,
        )
        if not reservation.created:
            existing = job_store.get(principal.tenant_id, reservation.job_id) or {}
            if existing.get("status") == "failed":
                failed_job_id = reservation.job_id
                idempotency_store.delete_if_job_matches(
                    principal.tenant_id,
                    idempotency_key,
                    failed_job_id,
                )
                reservation = idempotency_store.reserve(
                    principal.tenant_id,
                    idempotency_key,
                    fingerprint,
                    proposed_job_id,
                )
                if reservation.created:
                    logger.info(
                        "generation retry reserved new job tenant_id=%s failed_job_id=%s new_job_id=%s",
                        principal.tenant_id,
                        failed_job_id,
                        reservation.job_id,
                    )
                else:
                    existing = (
                        job_store.get(principal.tenant_id, reservation.job_id) or {}
                    )
        if not reservation.created:
            return JsonResponse(
                {
                    "job_id": reservation.job_id,
                    "case_id": generation_request.case_id,
                    "status": existing.get("status", "accepted"),
                    "error_code": existing.get("error_code"),
                    "error_message": existing.get("error_message"),
                },
                status=202,
            )
        reservation_created = True

        limiter.acquire(principal.tenant_id)
        acquired = True
        webhook_url = resolve_webhook_url(settings.WEBHOOK_ROOT_URL)
        job = job_store.create(
            {
                "job_id": reservation.job_id,
                "tenant_id": principal.tenant_id,
                "case_id": generation_request.case_id,
                "visa_type": visa_type,
                "document_type": generation_request.document_type,
                "document_slug": generation_request.document_slug,
                "status": "accepted",
                "stage": "queued",
                "progress_percent": 0,
                "download_url": None,
                "error_code": None,
                "error_message": None,
                "webhook_url": webhook_url,
                "request": generation_request.to_dict(),
            }
        )
        logger.info(
            "generation accepted job_id=%s tenant_id=%s visa_type=%s case_id=%s document_type=%s document_slug=%s",
            job.get("job_id"),
            job.get("tenant_id"),
            job.get("visa_type"),
            job.get("case_id"),
            job.get("document_type"),
            job.get("document_slug"),
        )
        task = cast(Any, run_generation_pipeline)
        try:
            async_result = task.apply_async(kwargs={"job": job}, queue="file_reading")
            logger.info(
                "generation pipeline queued job_id=%s task_id=%s queue=%s",
                job.get("job_id"),
                getattr(async_result, "id", None),
                "file_reading",
            )
        except Exception:
            logger.warning(
                "Falling back to synchronous generation execution for job %s",
                job["job_id"],
            )
            task.apply(kwargs={"job": job})
            logger.info(
                "generation pipeline ran synchronously job_id=%s",
                job.get("job_id"),
            )
        try:
            queue_job_webhook(job)
            logger.info(
                "accepted webhook queued job_id=%s tenant_id=%s",
                job.get("job_id"),
                job.get("tenant_id"),
            )
        except Exception:
            logger.warning(
                "Webhook queueing failed for accepted job %s; continuing without blocking the request",
                job["job_id"],
                exc_info=True,
            )
        return JsonResponse(
            {
                "job_id": job["job_id"],
                "case_id": job["case_id"],
                "status": "accepted",
                "error_code": None,
                "error_message": None,
            },
            status=202,
        )
    except CaseJobError as exc:
        if acquired:
            limiter.release(principal.tenant_id)
        if reservation_created:
            idempotency_store.delete(principal.tenant_id, idempotency_key)
        return _error_response(exc)
    except Exception as exc:
        if acquired:
            limiter.release(principal.tenant_id)
        if job is not None:
            logger.exception(
                "Generation request was accepted but downstream processing failed for job %s",
                job.get("job_id"),
            )
            return JsonResponse(
                {
                    "job_id": job["job_id"],
                    "case_id": job["case_id"],
                    "status": "accepted",
                    "error_code": None,
                    "error_message": None,
                },
                status=202,
            )
        logger.exception(
            "Generation request could not be accepted for visa %s",
            visa_type,
            exc_info=exc,
        )
        if 'principal' in locals() and idempotency_key:
            try:
                idempotency_store.delete(principal.tenant_id, idempotency_key)
            except Exception:
                logger.debug(
                    "Skipping idempotency cleanup after acceptance failure for %s",
                    idempotency_key,
                    exc_info=True,
                )
        return JsonResponse(
            {
                "error_code": "SERVICE_UNAVAILABLE",
                "error_message": "Generation request could not be accepted",
            },
            status=503,
        )


def _relative_media_path_from_download_url(download_url: object) -> str | None:
    if not isinstance(download_url, str) or not download_url.strip():
        return None
    path = urlsplit(download_url).path or ""
    media_url = (settings.MEDIA_URL or "/media/").rstrip("/") + "/"
    if not path.startswith(media_url):
        return None
    candidate = path[len(media_url) :].lstrip("/")
    return candidate or None


@csrf_exempt
def download_generation(request, visa_type: str, job_id: str):
    if request.method != "GET":
        return JsonResponse(
            {"error_code": "METHOD_NOT_ALLOWED", "error_message": "GET is required"},
            status=405,
        )

    principal = resolve_generation_principal(request.headers.get("Authorization"))
    record = JobStore().get(principal.tenant_id, str(job_id))
    if not record or record.get("visa_type") != visa_type:
        return JsonResponse(
            {"error_code": "NOT_FOUND", "error_message": "Unknown generation job"},
            status=404,
        )

    relative_path = record.get("output_relative_path") or _relative_media_path_from_download_url(
        record.get("download_url")
    )
    if not isinstance(relative_path, str) or not relative_path.strip():
        return JsonResponse(
            {
                "error_code": "DOWNLOAD_UNAVAILABLE",
                "error_message": "Generated output is not available for download yet",
            },
            status=409,
        )

    media_root = os.path.abspath(str(settings.MEDIA_ROOT))
    absolute_path = os.path.abspath(os.path.join(media_root, relative_path))
    if os.path.commonpath([media_root, absolute_path]) != media_root:
        return JsonResponse(
            {"error_code": "INVALID_PATH", "error_message": "Invalid download path"},
            status=400,
        )
    if not os.path.exists(absolute_path):
        return JsonResponse(
            {"error_code": "NOT_FOUND", "error_message": "Generated file not found"},
            status=404,
        )

    handle = open(absolute_path, "rb")
    return FileResponse(
        handle,
        as_attachment=True,
        filename=os.path.basename(absolute_path),
        content_type=DOCX_CONTENT_TYPE,
    )
