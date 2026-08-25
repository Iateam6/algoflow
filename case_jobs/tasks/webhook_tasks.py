from __future__ import annotations

import logging
import random
import uuid
from datetime import datetime, timezone
from typing import Any, cast

from celery import shared_task
from django.conf import settings

from case_jobs.exceptions import WebhookDeliveryError
from case_jobs.integrations.webhook_client import WebhookClient


logger = logging.getLogger(__name__)


def _preview_text(value: object, limit: int = 500) -> str:
    text = "" if value is None else str(value)
    text = text.strip()
    if limit <= 0 or not text:
        return ""
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}…"


def _event_type(record: dict) -> str:
    status = record.get("status")
    if status == "accepted":
        return "generation.accepted"
    if status == "completed":
        return "generation.completed"
    if status == "failed":
        return "generation.failed"
    return "generation.processing"


def build_job_event(record: dict) -> dict:
    job_id = record.get("job_id")
    if not isinstance(job_id, str) or not job_id.strip():
        raise ValueError("Webhook event requires job.job_id")
    event_type = _event_type(record)
    if event_type == "generation.completed" and not record.get("download_url"):
        raise ValueError("generation.completed requires job.download_url")
    stage = record.get("stage", "queued")
    stable_name = f"{job_id}:{event_type}:{stage}"
    event_id = f"evt_{uuid.uuid5(uuid.NAMESPACE_URL, stable_name).hex}"
    allowed = {
        "job_id",
        "case_id",
        "document_type",
        "document_slug",
        "beneficiary_name",
        "petitioner_name",
        "attorney_name",
        "status",
        "stage",
        "progress_percent",
        "download_url",
        "error_code",
        "error_message",
    }
    job = {key: record.get(key) for key in allowed}
    return {
        "event_id": event_id,
        "event_type": event_type,
        "event_created_at": datetime.now(timezone.utc).isoformat(),
        "job": job,
    }


def _webhook_secret() -> str:
    secret = settings.AI_DOC_WEBHOOK_SECRET
    if not secret:
        raise ValueError("AI_DOC_WEBHOOK_SECRET is not configured")
    return secret


def queue_job_webhook(record: dict) -> None:
    task = cast(Any, deliver_webhook)
    event = build_job_event(record)
    if event.get("event_type") == "generation.completed":
        job = event.get("job") or {}
        print(
            f"webhook_url={record.get('webhook_url')} download_url={job.get('download_url')}",
            flush=True,
        )
    logger.info(
        "webhook queued job_id=%s event_id=%s event_type=%s stage=%s",
        record.get("job_id"),
        event.get("event_id"),
        event.get("event_type"),
        record.get("stage"),
    )
    task.apply_async(
        kwargs={
            "event": event,
            "webhook_url": record["webhook_url"],
        },
        queue="webhooks",
    )


@shared_task(bind=True, max_retries=None, name="case_jobs.deliver_webhook")
def deliver_webhook(self, event: dict, webhook_url: str):
    logger.info(
        "webhook delivery start event_id=%s event_type=%s",
        event.get("event_id"),
        event.get("event_type"),
    )
    if event.get("event_type") == "generation.failed":
        job = event.get("job") or {}
        logger.warning(
            "webhook generation.failed context event_id=%s job_id=%s stage=%s error_code=%s error_message=%r",
            event.get("event_id"),
            job.get("job_id"),
            job.get("stage"),
            job.get("error_code"),
            _preview_text(job.get("error_message")),
        )
    try:
        status_code = WebhookClient(timeout=settings.WEBHOOK_TIMEOUT_SECONDS).deliver(
            webhook_url,
            event,
            _webhook_secret(),
        ).status_code
        logger.info(
            "webhook delivered event_id=%s event_type=%s http_status=%s",
            event.get("event_id"),
            event.get("event_type"),
            status_code,
        )
        return status_code
    except WebhookDeliveryError as exc:
        if not exc.retryable:
            logger.error(
                "Webhook delivery permanently failed for event %s with HTTP %s",
                event.get("event_id", "unknown"),
                exc.response_status_code or "unavailable",
            )
            raise
        if self.request.retries >= settings.WEBHOOK_MAX_RETRIES:
            logger.error(
                "Webhook delivery retries exhausted for event %s",
                event.get("event_id", "unknown"),
            )
            raise
        countdown = exc.retry_after
        if countdown is None:
            countdown = min(3600, 2 ** self.request.retries) + random.randint(0, 5)
        raise self.retry(exc=exc, countdown=countdown)
