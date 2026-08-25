import os
import tempfile

from django.test import SimpleTestCase

from .views import parse_final_copy_payload
from .utils import stamp_pdf_with_firm_name
from pypdf import PdfReader
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter


class ParseFinalCopyPayloadTests(SimpleTestCase):
    def test_payload_supports_top_level_cover_letter_string_and_does_not_extract_from_exhibits(self):
        payload = {
            "case_id": "EB1A-2026-00124",
            "document_type": "Final Copy",
            "document_slug": "final-copy",
            "visa_type": "TN",
            "preparer": {"firm_name": "Lakshmi"},
            "petitioner": {"full_name": "Sam Taylor M"},
            "beneficiary": {"full_name": "Padma Kumari Rao"},
            "cover_letter": "https://s3.example/cover-letter-top-level",
            "exhibits": [
                {
                    "number": 2,
                    "title": "",
                    "items": [
                        {
                            "type": "document",
                            "name": "Cover Letter",
                            "url": "https://s3.example/cover-letter-exhibit",
                        },
                        {
                            "type": "file",
                            "name": "Evidence of Nationality",
                            "files": [{"url": "https://s3.example/passport"}],
                        },
                    ],
                },
                {
                    "number": 1,
                    "title": "Company Formation Documents",
                    "items": [
                        {"type": "form", "name": "Form G-28", "url": "https://s3.example/g28"},
                        {
                            "type": "file",
                            "name": "Company Registration and Business Licenses",
                            "files": [
                                {"url": "https://s3.example/coi"},
                                {"url": "https://s3.example/license"},
                            ],
                        },
                    ],
                },
            ],
        }

        normalized = parse_final_copy_payload(payload)

        self.assertIn("cover_lines", normalized)
        self.assertEqual(normalized["front_matter"][0]["url"], "https://s3.example/cover-letter-top-level")

        # Exhibits are sorted by number: 1 then 2
        self.assertEqual([e["number"] for e in normalized["exhibits"]], [1, 2])

        ex2 = normalized["exhibits"][1]
        self.assertEqual(ex2["divider_lines"][1], "EXHIBIT 2")
        # Title fallback uses first remaining item name since title is empty
        self.assertEqual(ex2["divider_lines"][2], "Cover Letter")

        ex2_urls = [it["url"] for it in ex2["items"]]
        self.assertEqual(ex2_urls, ["https://s3.example/cover-letter-exhibit", "https://s3.example/passport"])

    def test_cover_letter_null_falls_back_to_legacy_extraction_from_exhibits(self):
        payload = {
            "case_id": "EB1A-2026-00124",
            "document_type": "Final Copy",
            "document_slug": "final-copy",
            "visa_type": "TN",
            "preparer": {"firm_name": "Lakshmi"},
            "petitioner": {"full_name": "Sam Taylor M"},
            "beneficiary": {"full_name": "Padma Kumari Rao"},
            "cover_letter": None,
            "exhibits": [
                {
                    "number": 1,
                    "title": "Company Formation Documents",
                    "items": [
                        {
                            "type": "document",
                            "name": "Cover Letter",
                            "url": "https://s3.example/cover-letter-exhibit",
                        },
                        {"type": "form", "name": "Form G-28", "url": "https://s3.example/g28"},
                    ],
                }
            ],
        }

        normalized = parse_final_copy_payload(payload)

        self.assertEqual(normalized["front_matter"][0]["url"], "https://s3.example/cover-letter-exhibit")
        ex1_urls = [it["url"] for it in normalized["exhibits"][0]["items"]]
        self.assertEqual(ex1_urls, ["https://s3.example/g28"])

    def test_legacy_payload_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_final_copy_payload({"docs": [], "forms": [], "files": []})

    def test_top_level_files_payload_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_final_copy_payload({"files": []})


class FinalCopySeparatorPagesTests(SimpleTestCase):
    def test_final_copy_generates_cover_and_exhibit_dividers_only(self):
        urls = [
            "https://example.com/files/g28.pdf",
            "https://example.com/cover-letter",
            "https://example.com/passport.pdf",
        ]
        payload = {
            "case_id": "EB1A-2026-00124",
            "document_type": "Final Copy",
            "document_slug": "final-copy",
            "visa_type": "TN",
            "preparer": {"firm_name": "Lakshmi"},
            "petitioner": {"full_name": "Sam Taylor M"},
            "beneficiary": {"full_name": "Padma Kumari Rao"},
            "cover_letter": urls[1],
            "exhibits": [
                {
                    "number": 1,
                    "title": "Company Formation Documents",
                    "items": [
                        {"type": "form", "name": "Form G-28", "url": urls[0]},
                    ],
                },
                {
                    "number": 2,
                    "title": "",
                    "items": [
                        {"type": "file", "name": "Evidence of Nationality", "files": [{"url": urls[2]}]},
                    ],
                },
            ],
        }

        normalized = parse_final_copy_payload(payload)

        merge_order = ["cover.pdf"]
        for fm in normalized["front_matter"]:
            merge_order.append(fm["url"])
        for ex in normalized["exhibits"]:
            merge_order.append(f"divider:{ex['number']}")
            for it in ex["items"]:
                merge_order.append(it["url"])

        self.assertEqual(
            merge_order,
            [
                "cover.pdf",
                urls[1],
                "divider:1",
                urls[0],
                "divider:2",
                urls[2],
            ],
        )


class FirmNameStampingTests(SimpleTestCase):
    def test_stamp_pdf_with_firm_name_writes_visible_text(self):
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmp:
            src = os.path.join(tmp, "src.pdf")
            c = canvas.Canvas(src, pagesize=letter)
            c.drawString(72, 72, "Hello")
            c.showPage()
            c.save()

            out = stamp_pdf_with_firm_name(src, "My Firm Name")
            self.assertTrue(os.path.exists(out))

            reader = PdfReader(out)
            text = (reader.pages[0].extract_text() or "").replace("\n", " ")
            self.assertIn("My Firm Name", text)
