from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from django.conf import settings

from case_jobs.exceptions import IdempotencyConflict
from case_jobs.integrations.redis_client import get_redis_client


_DELETE_IF_JOB_MATCHES_SCRIPT = """
local record_json = redis.call("GET", KEYS[1])
if not record_json then
    return 0
end
local record = cjson.decode(record_json)
if record["job_id"] ~= ARGV[1] then
    return 0
end
return redis.call("DEL", KEYS[1])
"""


@dataclass(frozen=True)
class IdempotencyReservation:
    job_id: str
    created: bool


def request_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def resolve_idempotency_key(explicit_key: str, fingerprint: str) -> str:
    return explicit_key or f"payload:{fingerprint}"


class IdempotencyStore:
    def __init__(self, client=None, ttl: int | None = None):
        self.client = client or get_redis_client()
        self.ttl = ttl or settings.IDEMPOTENCY_TTL_SECONDS

    @staticmethod
    def key(tenant_id: str, idempotency_key: str) -> str:
        digest = hashlib.sha256(idempotency_key.encode()).hexdigest()
        return f"case_jobs:idempotency:{tenant_id}:{digest}"

    def reserve(
        self,
        tenant_id: str,
        idempotency_key: str,
        fingerprint: str,
        proposed_job_id: str,
    ) -> IdempotencyReservation:
        key = self.key(tenant_id, idempotency_key)
        record = {"fingerprint": fingerprint, "job_id": proposed_job_id}
        created = self.client.set(
            key,
            json.dumps(record, separators=(",", ":"), sort_keys=True),
            nx=True,
            ex=self.ttl,
        )
        if created:
            return IdempotencyReservation(proposed_job_id, True)

        existing_raw = self.client.get(key)
        if not existing_raw:
            return self.reserve(
                tenant_id, idempotency_key, fingerprint, proposed_job_id
            )
        existing = json.loads(existing_raw)
        if existing["fingerprint"] != fingerprint:
            raise IdempotencyConflict(
                "Idempotency-Key was already used for a different request"
            )
        return IdempotencyReservation(existing["job_id"], False)

    def delete(self, tenant_id: str, idempotency_key: str) -> None:
        try:
            self.client.delete(self.key(tenant_id, idempotency_key))
        except Exception:
            return

    def delete_if_job_matches(
        self,
        tenant_id: str,
        idempotency_key: str,
        expected_job_id: str,
    ) -> bool:
        deleted = self.client.eval(
            _DELETE_IF_JOB_MATCHES_SCRIPT,
            1,
            self.key(tenant_id, idempotency_key),
            expected_job_id,
        )
        return bool(deleted)
