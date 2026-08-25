import json

from django.test import SimpleTestCase
from redis.connection import HIREDIS_AVAILABLE

from case_jobs.exceptions import IdempotencyConflict
from case_jobs.jobs.idempotency import (
    IdempotencyStore,
    request_fingerprint,
    resolve_idempotency_key,
)


class FakeRedis:
    def __init__(self):
        self.values = {}

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self.values:
            return None
        self.values[key] = value
        return True

    def get(self, key):
        return self.values.get(key)

    def delete(self, key):
        self.values.pop(key, None)

    def eval(self, script, numkeys, key, expected_job_id):
        raw = self.values.get(key)
        if not raw:
            return 0
        if json.loads(raw)["job_id"] != expected_job_id:
            return 0
        self.delete(key)
        return 1


class IdempotencyTests(SimpleTestCase):
    def test_hiredis_parser_is_available(self):
        self.assertTrue(HIREDIS_AVAILABLE)

    def test_missing_header_uses_deterministic_payload_key(self):
        fingerprint = request_fingerprint({"case_id": "case-1"})
        self.assertEqual(
            resolve_idempotency_key("", fingerprint),
            f"payload:{fingerprint}",
        )

    def test_explicit_header_remains_supported(self):
        self.assertEqual(resolve_idempotency_key("caller-key", "hash"), "caller-key")

    def test_duplicate_request_returns_original_job(self):
        store = IdempotencyStore(client=FakeRedis(), ttl=60)
        fingerprint = request_fingerprint({"case_id": "case-1"})
        first = store.reserve("tenant-1", "same-key", fingerprint, "job-1")
        second = store.reserve("tenant-1", "same-key", fingerprint, "job-2")
        self.assertTrue(first.created)
        self.assertFalse(second.created)
        self.assertEqual(second.job_id, "job-1")

    def test_reused_key_with_different_body_conflicts(self):
        store = IdempotencyStore(client=FakeRedis(), ttl=60)
        store.reserve("tenant-1", "same-key", "hash-a", "job-1")
        with self.assertRaises(IdempotencyConflict):
            store.reserve("tenant-1", "same-key", "hash-b", "job-2")

    def test_failed_job_reservation_can_be_deleted_conditionally(self):
        store = IdempotencyStore(client=FakeRedis(), ttl=60)
        store.reserve("tenant-1", "same-key", "hash-a", "job-1")

        deleted = store.delete_if_job_matches(
            "tenant-1", "same-key", "job-1"
        )

        self.assertTrue(deleted)
        replacement = store.reserve("tenant-1", "same-key", "hash-a", "job-2")
        self.assertTrue(replacement.created)
        self.assertEqual(replacement.job_id, "job-2")

    def test_conditional_delete_preserves_newer_reservation(self):
        store = IdempotencyStore(client=FakeRedis(), ttl=60)
        store.reserve("tenant-1", "same-key", "hash-a", "job-1")
        store.delete_if_job_matches("tenant-1", "same-key", "job-1")
        store.reserve("tenant-1", "same-key", "hash-a", "job-2")

        deleted = store.delete_if_job_matches(
            "tenant-1", "same-key", "job-1"
        )

        self.assertFalse(deleted)
        reservation = store.reserve("tenant-1", "same-key", "hash-a", "job-3")
        self.assertFalse(reservation.created)
        self.assertEqual(reservation.job_id, "job-2")
