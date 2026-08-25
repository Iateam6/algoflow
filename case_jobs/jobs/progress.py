from __future__ import annotations

from case_jobs.constants import PROGRESS_STAGES
from case_jobs.jobs.job_store import JobStore


class ProgressReporter:
    def __init__(self, tenant_id: str, job_id: str, store: JobStore | None = None):
        self.tenant_id = tenant_id
        self.job_id = job_id
        self.store = store or JobStore()

    def stage(self, stage: str, **extra):
        if stage not in PROGRESS_STAGES:
            raise ValueError(f"Unknown progress stage: {stage}")
        current = self.store.get(self.tenant_id, self.job_id)
        if current and current.get("stage") == stage:
            return current
        status = "completed" if stage == "completed" else "processing"
        record = self.store.update(
            self.tenant_id,
            self.job_id,
            status=status,
            stage=stage,
            progress_percent=PROGRESS_STAGES[stage],
            **extra,
        )
        from case_jobs.tasks.webhook_tasks import queue_job_webhook

        queue_job_webhook(record)
        return record

    def failed(self, error_code: str, error_message: str, stage: str | None = None):
        current = self.store.get(self.tenant_id, self.job_id) or {}
        record = self.store.update(
            self.tenant_id,
            self.job_id,
            status="failed",
            stage=stage or current.get("stage", "queued"),
            error_code=error_code,
            error_message=error_message,
        )
        from case_jobs.tasks.webhook_tasks import queue_job_webhook

        queue_job_webhook(record)
        return record

