import json
from unittest.mock import Mock, patch

from django.test import RequestFactory, SimpleTestCase, override_settings
from kombu.exceptions import OperationalError
from django.urls import resolve

from case_jobs.api.views import create_generation
from case_jobs.api.validators import validate_generation_request
from case_jobs.jobs.idempotency import IdempotencyReservation
from case_jobs.pipeline.orchestrator import _ingestion_sources, _job_download_url
from case_jobs.pipeline.identity import IdentityRecord
from case_jobs.pipeline.orchestrator import execute_generation_job
from case_jobs.pipeline.verification import VerificationResult
from case_jobs.tests.test_validation import valid_exhibit_payload, valid_payload
from immigration_algoflow_APIs import settings as app_settings


class FakeReporter:
    stages = []

    def __init__(self, *args, **kwargs):
        self.stages = []

    def stage(self, stage, **extra):
        self.stages.append(stage)
        return {"stage": stage, **extra}

    def failed(self, *args, **kwargs):
        raise AssertionError("pipeline should not fail")


@override_settings(
    WEBHOOK_ROOT_URL="https://api.visa26.com/webhooks/documents",
    AI_DOC_WEBHOOK_SECRET="test-secret",
    GENERATION_AUTH_ENABLED=False,
    DEFAULT_TENANT_ID="public",
    ENABLE_MODEL_VERIFICATION=False,
)
class PipelineTests(SimpleTestCase):
    def test_exhibit_attachments_replace_files_and_keep_readable_names(self):
        payload = valid_payload()
        payload["exhibits"] = valid_exhibit_payload()["exhibits"]
        request = validate_generation_request(payload, "eb-1a")

        self.assertEqual(
            _ingestion_sources(request),
            [
                ("https://storage.example.com/g28.pdf", "Form G-28"),
                ("https://storage.example.com/company.pdf", "Company Records 1"),
                ("https://storage.example.com/license.pdf", "Company Records 2"),
                ("https://storage.example.com/support.pdf", "Support Letter"),
            ],
        )

    @patch("case_jobs.api.views.queue_job_webhook")
    @patch("case_jobs.api.views.run_generation_pipeline")
    @patch("case_jobs.api.views.CapacityLimiter")
    @patch("case_jobs.api.views.JobStore")
    @patch("case_jobs.api.views.IdempotencyStore")
    def test_tn_endpoint_accepts_special_exhibit_payload(
        self,
        idempotency_class,
        job_store_class,
        limiter_class,
        pipeline_task,
        queue_webhook,
    ):
        idempotency_class.return_value.reserve.return_value = IdempotencyReservation(
            "job-1", True
        )
        job_store_class.return_value.create.side_effect = lambda job: job
        request = RequestFactory().post(
            "/api/tn/generate_doc/",
            data=json.dumps(valid_exhibit_payload()),
            content_type="application/json",
        )

        response = create_generation(request, "tn")

        self.assertEqual(response.status_code, 202)
        job = pipeline_task.apply_async.call_args.kwargs["kwargs"]["job"]
        self.assertEqual(job["document_slug"], "exhibit-list")
        self.assertEqual(job["request"]["document_slug"], "exhibit-list")
        self.assertEqual(
            job["request"]["preparer"]["firm_name"],
            "Smith & Associates, LLC",
        )
        self.assertEqual([item["number"] for item in job["request"]["exhibits"]], [1, 3])
        queue_webhook.assert_called_once()

    def test_public_generation_route_is_registered(self):
        match = resolve("/api/eb-1aA/generate_doc/")
        self.assertEqual(match.url_name, "create_generation")

    @override_settings(
        PUBLIC_BASE_URL="",
        WEBHOOK_ROOT_URL="https://api.visa26.com/webhooks/documents",
    )
    def test_job_download_url_falls_back_to_webhook_host(self):
        url = _job_download_url(
            visa_type="tn",
            job_id="00000000-0000-0000-0000-000000000000",
        )
        self.assertTrue(
            url.startswith(
                "https://api.visa26.com/api/tn/generate_doc/00000000-0000-0000-0000-000000000000/download/"
            )
        )

    @override_settings(
        PUBLIC_BASE_URL="https://algoai.visa26.com",
        WEBHOOK_ROOT_URL="https://api.visa26.com/webhooks/documents",
    )
    def test_job_download_url_uses_public_base_url(self):
        url = _job_download_url(
            visa_type="tn",
            job_id="00000000-0000-0000-0000-000000000000",
        )
        self.assertTrue(
            url.startswith(
                "https://algoai.visa26.com/api/tn/generate_doc/00000000-0000-0000-0000-000000000000/download/"
            )
        )

    @override_settings(
        PUBLIC_BASE_URL="",
        WEBHOOK_ROOT_URL="",
    )
    def test_job_download_url_falls_back_to_relative_path(self):
        url = _job_download_url(
            visa_type="tn",
            job_id="00000000-0000-0000-0000-000000000000",
        )
        self.assertTrue(
            url.startswith(
                "/api/tn/generate_doc/00000000-0000-0000-0000-000000000000/download/"
            )
        )

    @override_settings(PUBLIC_BASE_URL="", WEBHOOK_ROOT_URL="")
    def test_job_download_urls_use_the_registered_public_visa_routes(self):
        expected_segments = {
            "aap": "aap",
            "aea": "aea",
            "ds-160": "ds-160",
            "ds-260": "ds-260",
            "eb-1aA": "eb-1aA",
            "eb-1aB": "eb-1aB",
            "n-400": "naturalization",
            "r-1": "reentry-permit",
        }

        for visa_type, segment in expected_segments.items():
            with self.subTest(visa_type=visa_type):
                self.assertEqual(
                    _job_download_url(visa_type=visa_type, job_id="job-id"),
                    f"/api/{segment}/generate_doc/job-id/download/",
                )

    def test_redis_urls_do_not_include_kombu_incompatible_protocol_query(self):
        self.assertEqual(
            app_settings._ensure_redis_resp2("redis://localhost:6379/0"),
            "redis://localhost:6379/0",
        )

    @override_settings(AI_DOC_WEBHOOK_SECRET="")
    def test_public_api_requires_default_tenant_webhook_secret(self):
        request = RequestFactory().post(
            "/api/eb-1a/generate_doc/",
            data=json.dumps(valid_payload()),
            content_type="application/json",
        )
        response = create_generation(request, "eb-1a")
        self.assertEqual(response.status_code, 503)

    @patch("case_jobs.pipeline.orchestrator.cleanup_job_directory")
    @patch("case_jobs.pipeline.orchestrator.VerificationAgent")
    @patch("case_jobs.pipeline.orchestrator.generate_document")
    @patch("case_jobs.pipeline.orchestrator.build_identity_record")
    @patch("case_jobs.pipeline.orchestrator.extract_sources")
    @patch("case_jobs.pipeline.orchestrator.download_sources")
    @patch("case_jobs.pipeline.orchestrator.ProgressReporter", FakeReporter)
    def test_pipeline_keeps_s3_disconnected(
        self,
        download_sources_mock,
        extract_sources_mock,
        identity_mock,
        generate_mock,
        verifier_class,
        cleanup_mock,
    ):
        payload = valid_payload()
        manifest = {
            "file_hash": "hash-1",
            "pages": ["Amritpal Sandhu evidence"],
            "local_path": "source.pdf",
            "content_type": "application/pdf",
        }
        download_sources_mock.return_value = [manifest]
        extract_sources_mock.return_value = [manifest]
        identity_mock.return_value = IdentityRecord(
            case_id="EB1A-2026-00124",
            attorney_name="Jane Smith",
            attorney_address={},
            beneficiary_name="Amritpal Sandhu",
            beneficiary_address={},
            petitioner_name="Amritpal Sandhu",
            petitioner_address={},
            service_center_name="CSC",
            service_center_address={},
            self_petition=True,
            supporting_sources=(),
        )
        generate_mock.return_value = "media/generated/Support_Letter.docx"
        verifier_class.return_value.verify.return_value = VerificationResult(True)
        result = execute_generation_job(
            {
                "tenant_id": "tenant-1",
                "job_id": "job-1",
                "visa_type": "eb-1a",
                "request": payload,
            }
        )
        self.assertEqual(result["stage"], "completed")
        self.assertNotIn("s3", execute_generation_job.__globals__)

    @patch("case_jobs.pipeline.orchestrator.cleanup_job_directory")
    @patch("case_jobs.pipeline.orchestrator.build_case_chunks")
    @patch("case_jobs.pipeline.orchestrator.build_corpus_hash")
    @patch("case_jobs.pipeline.orchestrator.build_identity_record")
    @patch("case_jobs.pipeline.orchestrator.extract_sources")
    @patch("case_jobs.pipeline.orchestrator.download_sources")
    @patch("case_jobs.pipeline.orchestrator.generate_document")
    @patch("case_jobs.pipeline.orchestrator.ProgressReporter", FakeReporter)
    def test_structured_exhibit_list_skips_downloads_and_rag(
        self,
        generate_mock,
        download_mock,
        extract_mock,
        identity_mock,
        corpus_hash_mock,
        chunks_mock,
        cleanup_mock,
    ):
        generate_mock.return_value = "media/generated/Exhibit_List.docx"

        result = execute_generation_job(
            {
                "tenant_id": "public",
                "job_id": "job-exhibits",
                "visa_type": "tn",
                "request": valid_exhibit_payload(),
            }
        )

        self.assertEqual(result["stage"], "completed")
        download_mock.assert_not_called()
        extract_mock.assert_not_called()
        identity_mock.assert_not_called()
        corpus_hash_mock.assert_not_called()
        chunks_mock.assert_not_called()
        self.assertEqual(generate_mock.call_args.args[1], [])

    @patch("case_jobs.api.views.queue_job_webhook")
    @patch("case_jobs.api.views.run_generation_pipeline")
    @patch("case_jobs.api.views.CapacityLimiter")
    @patch("case_jobs.api.views.JobStore")
    @patch("case_jobs.api.views.IdempotencyStore")
    def test_generation_endpoint_accepts_and_enqueues(
        self,
        idempotency_class,
        job_store_class,
        limiter_class,
        pipeline_task,
        queue_webhook,
    ):
        idempotency_class.return_value.reserve.return_value = IdempotencyReservation(
            "job-1", True
        )
        job_store_class.return_value.create.side_effect = lambda job: job
        request = RequestFactory().post(
            "/api/eb-1a/generate_doc/",
            data=json.dumps(valid_payload()),
            content_type="application/json",
        )
        response = create_generation(request, "eb-1a")
        self.assertEqual(response.status_code, 202)
        pipeline_task.apply_async.assert_called_once()
        queue_webhook.assert_called_once()
        job = pipeline_task.apply_async.call_args.kwargs["kwargs"]["job"]
        self.assertEqual(job["tenant_id"], "public")
        self.assertEqual(job["document_slug"], "support-letter")
        self.assertEqual(
            job["webhook_url"],
            "https://api.visa26.com/webhooks/documents",
        )
        reserved_key = idempotency_class.return_value.reserve.call_args.args[1]
        self.assertTrue(reserved_key.startswith("payload:"))

    @patch("case_jobs.api.views.queue_job_webhook")
    @patch("case_jobs.api.views.run_generation_pipeline")
    @patch("case_jobs.api.views.CapacityLimiter")
    @patch("case_jobs.api.views.JobStore")
    @patch("case_jobs.api.views.IdempotencyStore")
    def test_failed_e2_duplicate_reserves_and_enqueues_new_job(
        self,
        idempotency_class,
        job_store_class,
        limiter_class,
        pipeline_task,
        queue_webhook,
    ):
        idempotency_store = idempotency_class.return_value
        idempotency_store.reserve.side_effect = [
            IdempotencyReservation("failed-job", False),
            IdempotencyReservation("new-job", True),
        ]
        job_store = job_store_class.return_value
        job_store.get.return_value = {
            "status": "failed",
            "error_code": "VALIDATION_ERROR",
            "error_message": "Unsupported visa type: e-2",
        }
        job_store.create.side_effect = lambda job: job
        request = RequestFactory().post(
            "/api/e-2/generate_doc/",
            data=json.dumps(valid_payload()),
            content_type="application/json",
        )

        response = create_generation(request, "e-2")

        self.assertEqual(response.status_code, 202)
        response_body = json.loads(response.content)
        self.assertEqual(response_body["job_id"], "new-job")
        self.assertEqual(response_body["status"], "accepted")
        idempotency_store.delete_if_job_matches.assert_called_once()
        self.assertEqual(
            idempotency_store.delete_if_job_matches.call_args.args[2],
            "failed-job",
        )
        pipeline_task.apply_async.assert_called_once()
        queue_webhook.assert_called_once()

    @patch("case_jobs.api.views.queue_job_webhook")
    @patch("case_jobs.api.views.run_generation_pipeline")
    @patch("case_jobs.api.views.CapacityLimiter")
    @patch("case_jobs.api.views.JobStore")
    @patch("case_jobs.api.views.IdempotencyStore")
    def test_active_and_completed_duplicates_return_existing_job(
        self,
        idempotency_class,
        job_store_class,
        limiter_class,
        pipeline_task,
        queue_webhook,
    ):
        idempotency_store = idempotency_class.return_value
        job_store = job_store_class.return_value

        for status in ("accepted", "completed"):
            with self.subTest(status=status):
                existing_job_id = f"{status}-job"
                idempotency_store.reserve.return_value = IdempotencyReservation(
                    existing_job_id,
                    False,
                )
                job_store.get.return_value = {
                    "status": status,
                    "error_code": None,
                    "error_message": None,
                }
                request = RequestFactory().post(
                    "/api/eb-1a/generate_doc/",
                    data=json.dumps(valid_payload()),
                    content_type="application/json",
                )

                response = create_generation(request, "eb-1a")

                self.assertEqual(response.status_code, 202)
                response_body = json.loads(response.content)
                self.assertEqual(response_body["job_id"], existing_job_id)
                self.assertEqual(response_body["status"], status)
                idempotency_store.delete_if_job_matches.assert_not_called()
                job_store.create.assert_not_called()
                pipeline_task.apply_async.assert_not_called()
                queue_webhook.assert_not_called()

                idempotency_store.reset_mock()
                job_store.reset_mock()
                pipeline_task.reset_mock()
                queue_webhook.reset_mock()

    @patch("case_jobs.api.views.queue_job_webhook")
    @patch("case_jobs.api.views.run_generation_pipeline")
    @patch("case_jobs.api.views.CapacityLimiter")
    @patch("case_jobs.api.views.JobStore")
    @patch("case_jobs.api.views.IdempotencyStore")
    def test_standard_job_persists_files_and_exhibits(
        self,
        idempotency_class,
        job_store_class,
        limiter_class,
        pipeline_task,
        queue_webhook,
    ):
        idempotency_class.return_value.reserve.return_value = IdempotencyReservation(
            "job-1", True
        )
        job_store_class.return_value.create.side_effect = lambda job: job
        payload = valid_payload()
        payload["exhibits"] = valid_exhibit_payload()["exhibits"]
        request = RequestFactory().post(
            "/api/eb-1a/generate_doc/",
            data=json.dumps(payload),
            content_type="application/json",
        )

        response = create_generation(request, "eb-1a")

        self.assertEqual(response.status_code, 202)
        job = pipeline_task.apply_async.call_args.kwargs["kwargs"]["job"]
        self.assertEqual(job["request"]["files"], payload["files"])
        self.assertEqual(len(job["request"]["exhibits"]), 2)

    @patch("case_jobs.api.views.queue_job_webhook")
    @patch("case_jobs.api.views.run_generation_pipeline")
    @patch("case_jobs.api.views.CapacityLimiter")
    @patch("case_jobs.api.views.JobStore")
    @patch("case_jobs.api.views.IdempotencyStore")
    def test_generation_endpoint_falls_back_to_eager_execution_when_broker_is_unavailable(
        self,
        idempotency_class,
        job_store_class,
        limiter_class,
        pipeline_task,
        queue_webhook,
    ):
        idempotency_class.return_value.reserve.return_value = IdempotencyReservation(
            "job-1", True
        )
        job_store_class.return_value.create.side_effect = lambda job: job
        pipeline_task.apply_async.side_effect = OperationalError("broker unavailable")

        request = RequestFactory().post(
            "/api/eb-1a/generate_doc/",
            data=json.dumps(valid_payload()),
            content_type="application/json",
        )
        response = create_generation(request, "eb-1a")

        self.assertEqual(response.status_code, 202)
        pipeline_task.apply.assert_called_once()
        queue_webhook.assert_called_once()

    @patch("case_jobs.api.views.queue_job_webhook")
    @patch("case_jobs.api.views.run_generation_pipeline")
    @patch("case_jobs.api.views.CapacityLimiter")
    @patch("case_jobs.api.views.JobStore")
    @patch("case_jobs.api.views.IdempotencyStore")
    def test_generation_endpoint_accepts_request_when_webhook_queueing_fails(
        self,
        idempotency_class,
        job_store_class,
        limiter_class,
        pipeline_task,
        queue_webhook,
    ):
        idempotency_class.return_value.reserve.return_value = IdempotencyReservation(
            "job-1", True
        )
        job_store_class.return_value.create.side_effect = lambda job: job
        queue_webhook.side_effect = OperationalError("webhook broker unavailable")

        request = RequestFactory().post(
            "/api/eb-1a/generate_doc/",
            data=json.dumps(valid_payload()),
            content_type="application/json",
        )
        response = create_generation(request, "eb-1a")

        self.assertEqual(response.status_code, 202)
        pipeline_task.apply_async.assert_called_once()
        queue_webhook.assert_called_once()
