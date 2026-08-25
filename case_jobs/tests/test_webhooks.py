from unittest.mock import Mock, patch

import requests
from django.test import SimpleTestCase, override_settings

from case_jobs.exceptions import ValidationError, WebhookDeliveryError
from case_jobs.integrations.webhook_client import WebhookClient, resolve_webhook_url
from case_jobs.integrations.webhook_signing import verify_webhook_signature
from case_jobs.tasks.webhook_tasks import build_job_event, deliver_webhook


STAGING_WEBHOOK_URL = "https://api.visa26.com/webhooks/documents"


class WebhookTests(SimpleTestCase):
    def test_webhook_url_is_fixed_and_extensionless(self):
        self.assertEqual(resolve_webhook_url(STAGING_WEBHOOK_URL), STAGING_WEBHOOK_URL)
        with self.assertRaises(ValidationError):
            resolve_webhook_url(f"{STAGING_WEBHOOK_URL}/")

    def test_delivery_signs_exact_body_and_disables_redirects(self):
        response = Mock(status_code=200, headers={})
        session = Mock()
        session.post.return_value = response
        event = {
            "event_id": "evt-1",
            "event_type": "generation.accepted",
            "job": {"job_id": "job-1", "case_id": "00124"},
        }

        delivery = WebhookClient(session=session).deliver(
            STAGING_WEBHOOK_URL, event, "secret"
        )

        self.assertEqual(delivery.status_code, 200)
        args, kwargs = session.post.call_args
        self.assertEqual(args[0], STAGING_WEBHOOK_URL)
        self.assertEqual(kwargs["headers"]["Content-Type"], "application/json")
        self.assertTrue(
            verify_webhook_signature(
                kwargs["data"],
                "secret",
                kwargs["headers"]["X-Signature"],
            )
        )
        self.assertNotIn("X-Webhook-Signature", kwargs["headers"])
        self.assertFalse(kwargs["allow_redirects"])

    def test_400_and_401_are_not_retryable(self):
        for status_code in (400, 401):
            with self.subTest(status_code=status_code):
                session = Mock()
                session.post.return_value = Mock(status_code=status_code, headers={})
                with self.assertRaises(WebhookDeliveryError) as caught:
                    WebhookClient(session=session).deliver(
                        STAGING_WEBHOOK_URL,
                        {"event_id": "evt-1"},
                        "secret",
                    )
                self.assertFalse(caught.exception.retryable)
                self.assertEqual(caught.exception.response_status_code, status_code)

    def test_retryable_responses_and_retry_after(self):
        for status_code in (408, 429, 500, 503):
            with self.subTest(status_code=status_code):
                session = Mock()
                headers = {"Retry-After": "17"} if status_code == 429 else {}
                session.post.return_value = Mock(
                    status_code=status_code,
                    headers=headers,
                )
                with self.assertRaises(WebhookDeliveryError) as caught:
                    WebhookClient(session=session).deliver(
                        STAGING_WEBHOOK_URL,
                        {"event_id": "evt-1"},
                        "secret",
                    )
                self.assertTrue(caught.exception.retryable)
                if status_code == 429:
                    self.assertEqual(caught.exception.retry_after, 17)

    def test_connection_timeout_is_retryable(self):
        session = Mock()
        session.post.side_effect = requests.Timeout("timed out")
        with self.assertRaises(WebhookDeliveryError) as caught:
            WebhookClient(session=session).deliver(
                STAGING_WEBHOOK_URL,
                {"event_id": "evt-1"},
                "secret",
            )
        self.assertTrue(caught.exception.retryable)

    def test_completed_event_requires_download_url(self):
        with self.assertRaisesMessage(
            ValueError,
            "generation.completed requires job.download_url",
        ):
            build_job_event(
                {
                    "job_id": "job-1",
                    "status": "completed",
                    "stage": "completed",
                }
            )

    def test_event_requires_job_id_and_includes_document_slug(self):
        with self.assertRaisesMessage(ValueError, "Webhook event requires job.job_id"):
            build_job_event({"status": "accepted"})

        event = build_job_event(
            {
                "job_id": "job-1",
                "case_id": "00124",
                "document_type": "Petition Cover Letter",
                "document_slug": "petition-cover",
                "status": "accepted",
                "stage": "queued",
            }
        )
        self.assertEqual(event["job"]["document_slug"], "petition-cover")

    @override_settings(
        AI_DOC_WEBHOOK_SECRET="test-secret",
        WEBHOOK_TIMEOUT_SECONDS=10,
        WEBHOOK_MAX_RETRIES=8,
    )
    @patch("case_jobs.tasks.webhook_tasks.WebhookClient.deliver")
    def test_permanent_failure_does_not_invoke_celery_retry(self, deliver_mock):
        deliver_mock.side_effect = WebhookDeliveryError(
            "bad signature",
            retryable=False,
            status_code=401,
        )
        with patch.object(deliver_webhook, "retry") as retry_mock:
            with self.assertRaises(WebhookDeliveryError):
                deliver_webhook.run({"event_id": "evt-1"}, STAGING_WEBHOOK_URL)
        retry_mock.assert_not_called()

    @override_settings(
        AI_DOC_WEBHOOK_SECRET="test-secret",
        WEBHOOK_TIMEOUT_SECONDS=10,
        WEBHOOK_MAX_RETRIES=8,
    )
    @patch("case_jobs.tasks.webhook_tasks.WebhookClient.deliver")
    def test_retryable_failure_invokes_celery_retry(self, deliver_mock):
        deliver_mock.side_effect = WebhookDeliveryError(
            "temporarily unavailable",
            retryable=True,
            status_code=503,
        )
        with patch.object(
            deliver_webhook,
            "retry",
            side_effect=RuntimeError("retry scheduled"),
        ) as retry_mock:
            with self.assertRaisesRegex(RuntimeError, "retry scheduled"):
                deliver_webhook.run({"event_id": "evt-1"}, STAGING_WEBHOOK_URL)
        retry_mock.assert_called_once()
