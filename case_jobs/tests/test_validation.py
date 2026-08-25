from datetime import datetime, timedelta, timezone

import jwt
from django.test import SimpleTestCase, override_settings

from case_jobs.api.authentication import (
    authenticate_bearer_header,
    resolve_generation_principal,
)
from case_jobs.api.validators import validate_generation_request
from case_jobs.exceptions import (
    AuthenticationError,
    ServiceConfigurationError,
    ValidationError,
)


def valid_payload():
    address = {
        "line_1": "451 Research Drive",
        "line_2": None,
        "city": "San Francisco",
        "state": "CA",
        "province": None,
        "postal_code": "94105",
        "country": "United States",
    }
    return {
        "case_id": "00124",
        "document_type": "Support Letter",
        "document_slug": "support-letter",
        "preparer": {
            "full_name": "Jane Smith",
            "firm_name": "Boston Legal",
            "address": address.copy(),
        },
        "beneficiary": {"full_name": "Amritpal Sandhu", "address": address.copy()},
        "petitioner": {"full_name": "Amritpal Sandhu", "address": address.copy()},
        "files": [{"url": "https://storage.example.com/passport.pdf"}],
    }


def valid_exhibit_payload():
    address = {
        "line_1": "451 Research Drive",
        "line_2": None,
        "city": "San Francisco",
        "state": "CA",
        "province": None,
        "postal_code": "94105",
        "country": "United States",
    }
    return {
        "case_id": "TN-2026-00124",
        "document_type": "Exhibit List",
        "document_slug": "exhibit-list",
        "preparer": {
            "full_name": "Jane Smith",
            "firm_name": "Smith & Associates, LLC",
            "address": address,
        },
        "beneficiary": {"full_name": "Amritpal Sandhu", "address": address},
        "petitioner": {"full_name": "Example Company", "address": address},
        "exhibits": [
            {
                "number": 1,
                "title": "Forms and Fees",
                "items": [
                    {
                        "type": "form",
                        "name": "Form G-28",
                        "url": "https://storage.example.com/g28.pdf",
                    },
                    {
                        "type": "file",
                        "name": "Company Records",
                        "files": [
                            {"url": "https://storage.example.com/company.pdf"},
                            {"url": "https://storage.example.com/license.pdf"},
                        ],
                    },
                ],
            },
            {
                "number": 3,
                "title": "",
                "items": [
                    {
                        "type": "document",
                        "name": "Support Letter",
                        "url": "https://storage.example.com/support.pdf",
                    }
                ],
            },
        ],
    }


@override_settings(
    WEBHOOK_ROOT_URL="https://api.visa26.com/webhooks/documents"
)
class GenerationValidationTests(SimpleTestCase):
    def test_validates_tn_exhibit_payload_and_preserves_source_order(self):
        request = validate_generation_request(valid_exhibit_payload(), "tn")

        self.assertEqual(request.document_slug, "exhibit-list")
        self.assertEqual(request.preparer.firm_name, "Smith & Associates, LLC")
        self.assertEqual([exhibit.number for exhibit in request.exhibits], [1, 3])
        self.assertEqual(request.exhibits[1].title, "")
        self.assertEqual(
            [source.url for source in request.files],
            [
                "https://storage.example.com/g28.pdf",
                "https://storage.example.com/company.pdf",
                "https://storage.example.com/license.pdf",
                "https://storage.example.com/support.pdf",
            ],
        )
        self.assertEqual(
            validate_generation_request(request.to_dict(), "tn"),
            request,
        )

    def test_rejects_tn_exhibit_payload_for_other_visas(self):
        with self.assertRaises(ValidationError):
            validate_generation_request(valid_exhibit_payload(), "eb-1a")

    def test_rejects_invalid_tn_exhibit_items(self):
        invalid_type = valid_exhibit_payload()
        invalid_type["exhibits"][0]["items"][0]["type"] = "unknown"
        with self.assertRaises(ValidationError):
            validate_generation_request(invalid_type, "tn")

        insecure_url = valid_exhibit_payload()
        insecure_url["exhibits"][0]["items"][1]["files"][0]["url"] = (
            "http://storage.example.com/company.pdf"
        )
        with self.assertRaises(ValidationError):
            validate_generation_request(insecure_url, "tn")

        empty_files = valid_exhibit_payload()
        empty_files["exhibits"][0]["items"][1]["files"] = []
        with self.assertRaises(ValidationError):
            validate_generation_request(empty_files, "tn")

    def test_rejects_duplicate_exhibit_numbers_and_extra_fields(self):
        duplicate = valid_exhibit_payload()
        duplicate["exhibits"][1]["number"] = 1
        with self.assertRaises(ValidationError):
            validate_generation_request(duplicate, "tn")

        extra_field = valid_exhibit_payload()
        extra_field["unsupported"] = "value"
        with self.assertRaises(ValidationError):
            validate_generation_request(extra_field, "tn")

        wrong_slug = valid_exhibit_payload()
        wrong_slug["document_slug"] = "support-letter"
        with self.assertRaises(ValidationError):
            validate_generation_request(wrong_slug, "tn")

    @override_settings(MAX_FILES_PER_JOB=3)
    def test_tn_exhibit_payload_ignores_standard_file_limit(self):
        request = validate_generation_request(valid_exhibit_payload(), "tn")
        self.assertEqual(len(request.files), 4)

    def test_validates_self_petition_payload(self):
        request = validate_generation_request(valid_payload(), "eb-1a")
        self.assertEqual(request.case_id, "00124")
        self.assertEqual(request.document_slug, "support-letter")
        self.assertEqual(request.preparer.full_name, "Jane Smith")
        self.assertEqual(request.preparer.firm_name, "Boston Legal")
        self.assertEqual(request.beneficiary, request.petitioner)

    def test_accepts_province_field_in_addresses(self):
        payload = valid_payload()
        payload["preparer"]["address"]["state"] = None
        payload["preparer"]["address"]["province"] = "Ontario"
        payload["beneficiary"]["address"]["state"] = None
        payload["beneficiary"]["address"]["province"] = "Punjab"
        payload["petitioner"]["address"]["state"] = None
        payload["petitioner"]["address"]["province"] = "Telangana"

        request = validate_generation_request(payload, "eb-1a")
        self.assertIsNone(request.preparer.address.state)
        self.assertEqual(request.preparer.address.province, "Ontario")
        self.assertIsNone(request.beneficiary.address.state)
        self.assertEqual(request.beneficiary.address.province, "Punjab")
        self.assertIsNone(request.petitioner.address.state)
        self.assertEqual(request.petitioner.address.province, "Telangana")

    def test_accepts_missing_blank_and_null_address_fields(self):
        payload = valid_payload()
        payload["preparer"]["address"] = {}
        payload["beneficiary"]["address"] = {
            "line_1": "",
            "city": None,
            "postal_code": "  ",
        }
        payload["petitioner"]["address"] = {"country": "India"}

        request = validate_generation_request(payload, "eb-1a")

        self.assertIsNone(request.preparer.address.line_1)
        self.assertIsNone(request.preparer.address.country)
        self.assertIsNone(request.beneficiary.address.city)
        self.assertIsNone(request.beneficiary.address.postal_code)
        self.assertEqual(request.petitioner.address.country, "India")

    def test_requires_party_names_and_preparer_firm_name(self):
        for path in (
            ("preparer", "full_name"),
            ("preparer", "firm_name"),
            ("beneficiary", "full_name"),
            ("petitioner", "full_name"),
        ):
            payload = valid_payload()
            payload[path[0]][path[1]] = ""
            with self.assertRaises(ValidationError):
                validate_generation_request(payload, "eb-1a")

    def test_standard_payload_accepts_exhibits_without_merging_source_files(self):
        payload = valid_payload()
        payload["exhibits"] = valid_exhibit_payload()["exhibits"]

        request = validate_generation_request(payload, "eb-1a")

        self.assertEqual(len(request.exhibits), 2)
        self.assertEqual(
            [source.url for source in request.files],
            ["https://storage.example.com/passport.pdf"],
        )
        self.assertEqual(
            validate_generation_request(request.to_dict(), "eb-1a"),
            request,
        )

    def test_standard_payload_best_effort_parses_exhibits(self):
        payload = valid_payload()
        exhibits = valid_exhibit_payload()["exhibits"]
        exhibits[0]["items"].extend(
            [
                {"type": "unknown", "name": "Ignored"},
                {
                    "type": "document",
                    "name": "Invalid URL",
                    "url": "http://storage.example.com/invalid.pdf",
                },
            ]
        )
        exhibits.append({"number": "bad", "title": None, "items": []})
        payload["exhibits"] = exhibits

        request = validate_generation_request(payload, "eb-1a")

        self.assertEqual([exhibit.number for exhibit in request.exhibits], [1, 3])
        self.assertEqual(len(request.exhibits[0].items), 2)

    def test_standard_payload_accepts_exhibits_without_top_level_files(self):
        payload = valid_payload()
        payload.pop("files")
        payload["exhibits"] = valid_exhibit_payload()["exhibits"]

        request = validate_generation_request(payload, "eb-1a")

        self.assertFalse(request.files)
        self.assertEqual(len(request.exhibits), 2)

    def test_files_based_exhibit_list_for_other_visas_still_round_trips(self):
        payload = valid_payload()
        payload["document_type"] = "Exhibit List"
        payload["document_slug"] = "exhibit-list"

        request = validate_generation_request(payload, "h-1b")

        self.assertFalse(request.exhibits)
        self.assertEqual(request.to_dict()["files"], payload["files"])
        self.assertEqual(validate_generation_request(request.to_dict(), "h-1b"), request)

    def test_rejects_legacy_request_shape(self):
        payload = valid_payload()
        payload["attorney"] = payload.pop("preparer")
        with self.assertRaises(ValidationError):
            validate_generation_request(payload, "eb-1a")

    def test_rejects_non_https_source(self):
        payload = valid_payload()
        payload["files"] = [{"url": "http://storage.example.com/passport.pdf"}]
        with self.assertRaises(ValidationError):
            validate_generation_request(payload, "eb-1a")

    def test_rejects_unknown_document_type(self):
        payload = valid_payload()
        payload["document_type"] = "Unknown"
        with self.assertRaises(ValidationError):
            validate_generation_request(payload, "eb-1a")

    def test_rejects_missing_preparer_firm_name(self):
        payload = valid_payload()
        payload["preparer"].pop("firm_name")
        with self.assertRaises(ValidationError):
            validate_generation_request(payload, "eb-1a")

    def test_rejects_firm_name_on_non_preparer_party(self):
        payload = valid_payload()
        payload["petitioner"]["firm_name"] = "Example Firm"
        with self.assertRaises(ValidationError):
            validate_generation_request(payload, "eb-1a")

        payload = valid_payload()
        payload["beneficiary"]["firm_name"] = "Example Firm"
        with self.assertRaises(ValidationError):
            validate_generation_request(payload, "eb-1a")


@override_settings(
    JWT_SECRET="test-secret-that-is-at-least-32-bytes-long",
    JWT_ALGORITHM="HS256",
    JWT_AUDIENCE="",
    JWT_ISSUER="",
)
class AuthenticationTests(SimpleTestCase):
    def test_derives_tenant_from_verified_claim(self):
        token = jwt.encode(
            {
                "sub": "user-1",
                "tenant_id": "tenant-1",
                "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
            },
            "test-secret-that-is-at-least-32-bytes-long",
            algorithm="HS256",
        )
        principal = authenticate_bearer_header(f"Bearer {token}")
        self.assertEqual(principal.tenant_id, "tenant-1")

    def test_rejects_unsafe_tenant_identifier(self):
        token = jwt.encode(
            {
                "sub": "user-1",
                "tenant_id": "../tenant",
                "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
            },
            "test-secret-that-is-at-least-32-bytes-long",
            algorithm="HS256",
        )
        with self.assertRaises(AuthenticationError):
            authenticate_bearer_header(f"Bearer {token}")

    @override_settings(
        GENERATION_AUTH_ENABLED=False,
        DEFAULT_TENANT_ID="public",
    )
    def test_public_mode_does_not_require_bearer_token(self):
        principal = resolve_generation_principal(None)
        self.assertEqual(principal.tenant_id, "public")
        self.assertEqual(principal.subject, "public-api")

    @override_settings(GENERATION_AUTH_ENABLED=True)
    def test_enabled_authentication_still_requires_bearer_token(self):
        with self.assertRaises(AuthenticationError):
            resolve_generation_principal(None)

    @override_settings(
        GENERATION_AUTH_ENABLED=False,
        DEFAULT_TENANT_ID="",
    )
    def test_public_mode_requires_a_default_tenant(self):
        with self.assertRaises(ServiceConfigurationError):
            resolve_generation_principal(None)
