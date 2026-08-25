from django.test import SimpleTestCase, override_settings

from case_jobs.api.validators import validate_generation_request
from case_jobs.pipeline.verification import deterministic_verify
from case_jobs.tests.test_validation import valid_exhibit_payload, valid_payload


@override_settings(WEBHOOK_ROOT_URL="https://api.visa26.com/webhooks/documents")
class ExhibitVerificationTests(SimpleTestCase):
    def test_standard_document_treats_exhibits_as_context_only(self):
        payload = valid_payload()
        payload["exhibits"] = valid_exhibit_payload()["exhibits"]
        request = validate_generation_request(payload, "eb-1a")
        text = " ".join(
            (
                "Amritpal Sandhu Jane Smith",
                "This support letter is grounded in the submitted evidence. " * 30,
            )
        )

        result = deterministic_verify(text, request)

        self.assertTrue(result.passed, result.corrections)

    def test_accepts_ordered_exhibit_output_without_preparer(self):
        request = validate_generation_request(valid_exhibit_payload(), "tn")
        text = """
        List of Supporting Documents for the TN Visa Application of Amritpal Sandhu.
        Petitioner: Example Company. This index identifies the supporting evidence
        supplied with the application and follows the submitted organization.
        Exhibit 1 - Forms and Fees
        1.1 Form G-28
        1.2 Company Records
        Exhibit 3
        3.1 Support Letter
        The listed records support the requested TN classification and are organized
        for efficient review by the adjudicating officer. Each entry corresponds to
        evidence supplied with this application package.
        """

        result = deterministic_verify(text, request)

        self.assertTrue(result.passed, result.corrections)
        self.assertNotIn("Jane Smith", text)

    def test_rejects_missing_or_out_of_order_exhibit_items_and_urls(self):
        request = validate_generation_request(valid_exhibit_payload(), "tn")
        text = " ".join(
            (
                "Amritpal Sandhu Example Company Forms and Fees",
                "3.1 Support Letter",
                "1.1 Form G-28",
                "1.2 Company Records",
                "https://storage.example.com/g28.pdf",
                "supporting evidence " * 50,
            )
        )

        result = deterministic_verify(text, request)

        self.assertFalse(result.passed)
        self.assertTrue(
            any("submitted order" in correction for correction in result.corrections)
        )
        self.assertIn("Remove source URLs from the exhibit list", result.corrections)
