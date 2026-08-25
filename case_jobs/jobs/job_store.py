from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from django.conf import settings

from case_jobs.integrations.redis_client import get_redis_client


class JobStore:
    def __init__(self, client=None, ttl: int | None = None):
        self.client = client or get_redis_client()
        self.ttl = ttl or settings.JOB_STATE_TTL_SECONDS

    @staticmethod
    def key(tenant_id: str, job_id: str) -> str:
        return f"case_jobs:job:{tenant_id}:{job_id}"

    def create(self, job: dict[str, Any]) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        record = {**job, "created_at": now, "updated_at": now}
        self.client.setex(
            self.key(record["tenant_id"], record["job_id"]),
            self.ttl,
            json.dumps(record, separators=(",", ":"), sort_keys=True),
        )
        return record

    def get(self, tenant_id: str, job_id: str) -> dict[str, Any] | None:
        raw = self.client.get(self.key(tenant_id, job_id))
        return json.loads(raw) if raw else None

    def update(self, tenant_id: str, job_id: str, **changes) -> dict[str, Any]:
        key = self.key(tenant_id, job_id)
        lock = self.client.lock(f"{key}:lock", timeout=15, blocking_timeout=5)
        with lock:
            current = self.get(tenant_id, job_id)
            if current is None:
                raise KeyError(f"Unknown job {job_id}")
            current.update(changes)
            current["updated_at"] = datetime.now(timezone.utc).isoformat()
            self.client.setex(
                key,
                self.ttl,
                json.dumps(current, separators=(",", ":"), sort_keys=True),
            )
            return current

