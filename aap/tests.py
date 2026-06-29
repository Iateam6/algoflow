import json
import os
import tempfile
from unittest.mock import patch

from django.test import Client, SimpleTestCase, override_settings

from immigration_algoflow_APIs.file_inputs import INVALID_FILE_URLS_ERROR
from immigration_algoflow_APIs.media_cleanup import clean_media_files_for_generation


API_ENDPOINTS = (
    ("aap", "/api/aap/generate_doc/"),
    ("aea", "/api/aea/generate_doc/"),
    ("ds_160", "/api/ds-160/generate_doc/"),
    ("ds_260", "/api/ds-260/generate_doc/"),
    ("eb_1aA", "/api/eb-1aA/generate_doc/"),
    ("eb_1aB", "/api/eb-1aB/generate_doc/"),
    ("naturalization", "/api/naturalization/generate_doc/"),
    ("reentry_permit", "/api/reentry-permit/generate_doc/"),
)


class FakeDownloadResponse:
    headers = {
        "content-disposition": 'attachment; filename="input.pdf"',
        "content-type": "application/pdf",
    }

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=8192):
        yield b"%PDF-1.4\n"


class UrlOnlyGenerateDocTests(SimpleTestCase):
    def setUp(self):
        self.client = Client()

    def post_json(self, path, payload):
        return self.client.post(
            path,
            data=json.dumps(payload),
            content_type="application/json",
        )

    def test_url_only_payload_succeeds_for_all_generate_apis(self):
        payload = {
            "options": "Petition Cover Letter",
            "files": [{"url": "https://example.com/input.pdf"}],
        }

        for app_label, path in API_ENDPOINTS:
            with self.subTest(path=path), tempfile.TemporaryDirectory() as media_root:
                generated_dir = os.path.join(media_root, "generated")
                os.makedirs(generated_dir, exist_ok=True)
                generated_path = os.path.join(generated_dir, "result.docx")
                with open(generated_path, "wb") as generated_file:
                    generated_file.write(b"generated")

                with (
                    override_settings(MEDIA_ROOT=media_root),
                    patch(
                        f"{app_label}.views.requests.get",
                        return_value=FakeDownloadResponse(),
                    ) as mock_get,
                    patch(
                        f"{app_label}.views.handle_doc_generation",
                        return_value=[generated_path],
                    ) as mock_generation,
                ):
                    response = self.post_json(path, payload)

                self.assertEqual(response.status_code, 200)
                body = response.json()
                self.assertIsNone(body["error message"])
                self.assertIn("/media/", body["download_url"])
                self.assertTrue(body["download_url"].endswith("result.docx"))

                mock_get.assert_called_once_with(
                    "https://example.com/input.pdf",
                    stream=True,
                    timeout=30,
                )
                mock_generation.assert_called_once()
                file_manifests = mock_generation.call_args.args[0]
                self.assertEqual(file_manifests[0]["name"], "source-1")
                self.assertEqual(file_manifests[0]["url"], "https://example.com/input.pdf")
                self.assertEqual(os.path.basename(file_manifests[0]["local_path"]), "source_1.pdf")

    def test_name_slug_and_slag_payloads_are_rejected_before_download(self):
        for app_label, path in API_ENDPOINTS:
            for forbidden_key in ("name", "slug", "slag"):
                payload = {
                    "options": "Petition Cover Letter",
                    "files": [
                        {
                            "url": "https://example.com/input.pdf",
                            forbidden_key: "old-category",
                        }
                    ],
                }
                with (
                    self.subTest(path=path, forbidden_key=forbidden_key),
                    patch(f"{app_label}.views.requests.get") as mock_get,
                    patch(f"{app_label}.views.handle_doc_generation") as mock_generation,
                ):
                    response = self.post_json(path, payload)

                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["download_url"], None)
                self.assertEqual(response.json()["error message"], INVALID_FILE_URLS_ERROR)
                mock_get.assert_not_called()
                mock_generation.assert_not_called()

    def test_invalid_file_lists_are_rejected_before_download(self):
        invalid_payloads = (
            {"options": "Petition Cover Letter"},
            {"options": "Petition Cover Letter", "files": []},
            {"options": "Petition Cover Letter", "files": "https://example.com/input.pdf"},
            {"options": "Petition Cover Letter", "files": ["https://example.com/input.pdf"]},
            {"options": "Petition Cover Letter", "files": [{"url": ""}]},
            {
                "options": "Petition Cover Letter",
                "files": [{"url": "https://example.com/input.pdf", "extra": "value"}],
            },
        )

        for app_label, path in API_ENDPOINTS:
            for payload in invalid_payloads:
                with (
                    self.subTest(path=path, payload=payload),
                    patch(f"{app_label}.views.requests.get") as mock_get,
                    patch(f"{app_label}.views.handle_doc_generation") as mock_generation,
                ):
                    response = self.post_json(path, payload)

                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["download_url"], None)
                self.assertEqual(response.json()["error message"], INVALID_FILE_URLS_ERROR)
                mock_get.assert_not_called()
                mock_generation.assert_not_called()


class MediaCleanupTests(SimpleTestCase):
    def test_cleanup_removes_media_and_generated_files_but_keeps_directories(self):
        with tempfile.TemporaryDirectory() as media_root, override_settings(MEDIA_ROOT=media_root):
            root_file_path = os.path.join(media_root, "old-upload.pdf")
            generated_dir = os.path.join(media_root, "generated")
            generated_file_path = os.path.join(generated_dir, "old-output.docx")
            cache_dir = os.path.join(media_root, "aap_rag")
            cache_file_path = os.path.join(cache_dir, "manifest.json")

            os.makedirs(generated_dir, exist_ok=True)
            os.makedirs(cache_dir, exist_ok=True)

            with open(root_file_path, "wb") as root_file:
                root_file.write(b"root")
            with open(generated_file_path, "wb") as generated_file:
                generated_file.write(b"generated")
            with open(cache_file_path, "wb") as cache_file:
                cache_file.write(b"cache")

            clean_media_files_for_generation()

            self.assertFalse(os.path.exists(root_file_path))
            self.assertFalse(os.path.exists(generated_file_path))
            self.assertTrue(os.path.isdir(cache_dir))
            self.assertTrue(os.path.exists(cache_file_path))


class FinalCopyCleanupTests(SimpleTestCase):
    def setUp(self):
        self.client = Client()

    def test_final_copy_removes_stale_outputs_before_move(self):
        async def create_blank_page_pdf(output_path, text=""):
            with open(output_path, "wb") as output_file:
                output_file.write(b"separator")

        async def merge_pdfs(pdf_paths, output_path):
            with open(output_path, "wb") as output_file:
                output_file.write(b"new pdf")

        def prepend_cover_and_merge(doc_paths, output_path):
            with open(output_path, "wb") as output_file:
                output_file.write(b"new docx")
            return output_path

        with tempfile.TemporaryDirectory() as media_root, override_settings(MEDIA_ROOT=media_root):
            generated_dir = os.path.join(media_root, "generated")
            os.makedirs(generated_dir, exist_ok=True)
            final_pdf_path = os.path.join(generated_dir, "final_copy.pdf")
            final_docx_path = os.path.join(generated_dir, "final_copy.docx")

            with open(final_pdf_path, "wb") as pdf_file:
                pdf_file.write(b"old pdf")
            with open(final_docx_path, "wb") as docx_file:
                docx_file.write(b"old docx")

            with (
                patch("final_copy.views.create_blank_page_pdf", side_effect=create_blank_page_pdf),
                patch("final_copy.views.merge_pdfs", side_effect=merge_pdfs),
                patch("final_copy.views.prepend_cover_and_merge", side_effect=prepend_cover_and_merge),
            ):
                response = self.client.post(
                    "/api/final-copy/generate-final-copy/",
                    data=json.dumps({"docs": [], "forms": [], "files": []}),
                    content_type="application/json",
                )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["final_pdf_url"], "http://testserver/media/generated/final_copy.pdf")
            self.assertEqual(response.json()["final_docx_url"], "http://testserver/media/generated/final_copy.docx")

            with open(final_pdf_path, "rb") as pdf_file:
                self.assertEqual(pdf_file.read(), b"new pdf")
            with open(final_docx_path, "rb") as docx_file:
                self.assertEqual(docx_file.read(), b"new docx")
