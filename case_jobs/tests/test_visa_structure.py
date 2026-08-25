import json
from importlib import import_module
from pathlib import Path
from unittest.mock import patch

from django.test import RequestFactory, SimpleTestCase, override_settings
from django.urls import resolve

from case_jobs.jobs.idempotency import IdempotencyReservation
from case_jobs.registry import get_adapter, registered_visa_types
from case_jobs.tests.test_validation import valid_payload


VISA_PACKAGES = {
    "h_1b": "h-1b",
    "eb_1a": "eb-1a",
    "eb_1c": "eb-1c",
    "eb_2niw": "eb-2niw",
    "eb_5": "eb-5",
    "e_2": "e-2",
    "l1": "l-1",
    "o1": "o-1",
    "tn": "tn",
}
EXPECTED_FILES = {
    "__init__.py",
    "agent.py",
    "config.py",
    "adapter.py",
    "views.py",
    "urls.py",
    "tests.py",
}


class VisaStructureTests(SimpleTestCase):
    def test_each_visa_has_only_standard_files(self):
        project_root = Path(__file__).resolve().parents[2]
        for package in VISA_PACKAGES:
            package_dir = project_root / package
            with self.subTest(package=package):
                self.assertEqual(
                    {path.name for path in package_dir.iterdir() if path.is_file()},
                    EXPECTED_FILES,
                )
                self.assertFalse((package_dir / "migrations").exists())

    def test_all_adapters_are_registered_and_isolated(self):
        self.assertEqual(set(registered_visa_types()), set(VISA_PACKAGES.values()))
        namespaces = {
            get_adapter(visa_type).cache_namespace
            for visa_type in VISA_PACKAGES.values()
        }
        self.assertEqual(len(namespaces), len(VISA_PACKAGES))

    def test_get_adapter_accepts_common_visa_type_aliases(self):
        # Some callers (especially job payloads) use underscores or compact spellings.
        aliases = {
            "h-1b": ["h_1b", "H_1B", "h1b"],
            "eb-1a": ["eb_1a", "EB_1A", "eb1a"],
            "eb-1c": ["eb_1c", "EB_1C", "eb1c"],
            "eb-2niw": ["eb_2niw", "EB_2NIW", "eb2niw"],
            "eb-5": ["eb_5", "EB_5", "eb5"],
            "e-2": ["e_2", "E_2", "e2"],
            "l-1": ["l1", "L1", "l_1"],
            "o-1": ["o1", "O1", "o_1"],
            "tn": ["TN", "tn ", " TN"],
        }
        for canonical, variants in aliases.items():
            adapter = get_adapter(canonical)
            for variant in variants:
                with self.subTest(canonical=canonical, variant=variant):
                    self.assertIs(get_adapter(variant), adapter)

    def test_agents_use_shared_openai_client(self):
        project_root = Path(__file__).resolve().parents[2]
        for package in VISA_PACKAGES:
            source = (project_root / package / "agent.py").read_text(encoding="utf-8")
            with self.subTest(package=package):
                self.assertIn(
                    "from case_jobs.integrations.openai_client import get_openai_client",
                    source,
                )
                self.assertNotIn("from .openai_client", source)

    @override_settings(
        WEBHOOK_ROOT_URL="https://api.visa26.com/webhooks/documents",
        AI_DOC_WEBHOOK_SECRET="test-secret",
        GENERATION_AUTH_ENABLED=False,
        DEFAULT_TENANT_ID="public",
    )
    @patch("case_jobs.api.views.queue_job_webhook")
    @patch("case_jobs.api.views.run_generation_pipeline")
    @patch("case_jobs.api.views.CapacityLimiter")
    @patch("case_jobs.api.views.JobStore")
    @patch("case_jobs.api.views.IdempotencyStore")
    def test_all_async_routes_enqueue_the_correct_visa(
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
        prefixes = {
            "h-1b": "/api/h-1b/",
            "eb-1a": "/api/eb-1a/",
            "eb-1c": "/api/eb-1c/",
            "eb-2niw": "/api/eb-2niw/",
            "eb-5": "/api/eb-5/",
            "e-2": "/api/e-2/",
            "l-1": "/api/l-1/",
            "o-1": "/api/o1/",
            "tn": "/api/tn/",
        }
        for visa_type, prefix in prefixes.items():
            payload = valid_payload()
            payload["document_type"] = sorted(
                get_adapter(visa_type).supported_document_types
            )[0]
            request = RequestFactory().post(
                f"{prefix}generate_doc/",
                data=json.dumps(payload),
                content_type="application/json",
            )
            with self.subTest(visa_type=visa_type):
                response = resolve(f"{prefix}generate_doc/").func(request)
                self.assertEqual(response.status_code, 202)
                job = pipeline_task.apply_async.call_args.kwargs["kwargs"]["job"]
                self.assertEqual(job["visa_type"], visa_type)
            pipeline_task.reset_mock()
            queue_webhook.reset_mock()

    def test_tn_download_route_is_registered(self):
        match = resolve(
            "/api/tn/generate_doc/00000000-0000-0000-0000-000000000000/download/"
        )
        self.assertEqual(match.url_name, "download_generation")
