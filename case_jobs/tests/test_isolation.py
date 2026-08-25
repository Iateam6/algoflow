from unittest.mock import patch

from django.test import SimpleTestCase, override_settings

from case_jobs.api.validators import validate_generation_request
from case_jobs.exceptions import BeneficiaryNotFound, DownloadError
from case_jobs.integrations.s3_client import build_generated_document_key
from case_jobs.pipeline.file_downloader import assert_safe_https_url
from case_jobs.pipeline.identity import build_identity_record
from case_jobs.retrieval.corpus import build_corpus_hash
from case_jobs.tests.test_validation import valid_payload


class IsolationTests(SimpleTestCase):
    def test_corpus_hash_changes_across_tenants_and_cases(self):
        first = build_corpus_hash(
            tenant_id="tenant-a", case_id="case-a", source_hashes=["same"]
        )
        other_tenant = build_corpus_hash(
            tenant_id="tenant-b", case_id="case-a", source_hashes=["same"]
        )
        other_case = build_corpus_hash(
            tenant_id="tenant-a", case_id="case-b", source_hashes=["same"]
        )
        self.assertEqual(len({first, other_tenant, other_case}), 3)

    def test_s3_key_is_tenant_case_and_job_scoped(self):
        self.assertEqual(
            build_generated_document_key("tenant-a", "case-a", "job-a"),
            "generated/tenant-a/case-a/job-a/Support_Letter.docx",
        )

    @patch(
        "case_jobs.pipeline.file_downloader.socket.getaddrinfo",
        return_value=[(None, None, None, None, ("127.0.0.1", 443))],
    )
    def test_downloader_rejects_private_and_loopback_destinations(self, _resolver):
        with self.assertRaises(DownloadError):
            assert_safe_https_url("https://evidence.example.com/file.pdf")

    @override_settings(
        WEBHOOK_ROOT_URL="https://api.visa26.com/webhooks/documents",
        IDENTITY_EVIDENCE_POLICY="warn",
    )
    def test_identity_evidence_is_case_scoped_and_required(self):
        request = validate_generation_request(valid_payload(), "eb-1a")
        source = {
            "file_hash": "hash-a",
            "pages": ["Beneficiary and petitioner: Amritpal Sandhu"],
        }
        identity = build_identity_record(request, [source])
        self.assertEqual(identity.case_id, request.case_id)
        self.assertEqual(identity.supporting_sources[0]["file_hash"], "hash-a")

    @override_settings(
        WEBHOOK_ROOT_URL="https://api.visa26.com/webhooks/documents",
        IDENTITY_EVIDENCE_POLICY="strict",
    )
    def test_identity_strict_raises_when_beneficiary_not_found(self):
        request = validate_generation_request(valid_payload(), "eb-1a")
        with self.assertRaises(BeneficiaryNotFound):
            build_identity_record(
                request,
                [{"file_hash": "hash-b", "pages": ["Other person"]}],
            )

    @override_settings(
        WEBHOOK_ROOT_URL="https://api.visa26.com/webhooks/documents",
        IDENTITY_EVIDENCE_POLICY="warn",
    )
    def test_identity_warn_does_not_raise_and_records_conflict(self):
        request = validate_generation_request(valid_payload(), "eb-1a")
        identity = build_identity_record(
            request,
            [{"file_hash": "hash-b", "pages": ["Other person"]}],
        )
        self.assertIn("beneficiary_not_found", identity.conflicting_identities)

    @override_settings(
        WEBHOOK_ROOT_URL="https://api.visa26.com/webhooks/documents",
        IDENTITY_EVIDENCE_POLICY="off",
    )
    def test_identity_off_skips_scanning(self):
        request = validate_generation_request(valid_payload(), "eb-1a")
        identity = build_identity_record(
            request,
            [{"file_hash": "hash-b", "pages": ["Other person"]}],
        )
        self.assertEqual(identity.supporting_sources, ())
        self.assertEqual(identity.conflicting_identities, ())

    @override_settings(
        WEBHOOK_ROOT_URL="https://api.visa26.com/webhooks/documents",
        IDENTITY_EVIDENCE_POLICY="warn",
    )
    def test_identity_tolerates_last_first_format(self):
        request = validate_generation_request(valid_payload(), "eb-1a")
        source = {
            "file_hash": "hash-a",
            "pages": ["Beneficiary: Sandhu, Amritpal"],
        }
        identity = build_identity_record(request, [source])
        self.assertEqual(len(identity.supporting_sources), 1)
