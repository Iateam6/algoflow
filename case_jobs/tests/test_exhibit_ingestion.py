import asyncio
from unittest.mock import patch

from django.test import SimpleTestCase

from case_jobs.pipeline.file_downloader import _preferred_original_name
from case_jobs.pipeline.legacy_generation import async_handle_doc_generation
from case_jobs.retrieval.visa_rag import build_langchain_documents


class _ExhibitAdapter:
    display_name = "TN"
    visa_type = "tn"

    def __init__(self):
        self.generation_call = None

    async def generate_document(self, **kwargs):
        self.generation_call = kwargs
        return "# Exhibit List\n\n" + "supporting document " * 60


class ExhibitIngestionTests(SimpleTestCase):
    def test_payload_name_uses_detected_extension(self):
        self.assertEqual(
            _preferred_original_name("Intent to Depart", ".docx"),
            "Intent to Depart.docx",
        )
        self.assertEqual(
            _preferred_original_name("Intent to Depart.pdf", ".docx"),
            "Intent to Depart.docx",
        )

    def test_readable_download_name_becomes_vector_source_name(self):
        documents = build_langchain_documents(
            [
                {
                    "name": "source-1",
                    "original_filename": "Intent to Depart.docx",
                    "file_hash": "hash-1",
                    "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    "latex_path": "intent.tex",
                    "page_entries": [
                        {
                            "page_number": 1,
                            "latex_text": "Intent evidence",
                            "extraction_mode": "docx_text",
                        }
                    ],
                }
            ]
        )

        self.assertEqual(documents[0].metadata["source_name"], "Intent to Depart.docx")
        self.assertEqual(documents[0].metadata["source_category"], "source-1")

    @patch("case_jobs.pipeline.legacy_generation.convert_markdown_to_docx")
    @patch("case_jobs.pipeline.legacy_generation.clean_media_files_for_generation")
    @patch("case_jobs.pipeline.legacy_generation.get_or_build_corpus")
    def test_structured_exhibit_generation_bypasses_corpus(
        self,
        corpus_mock,
        cleanup_mock,
        convert_mock,
    ):
        adapter = _ExhibitAdapter()
        convert_mock.return_value = "Exhibit_List.docx"
        submitted = {
            "beneficiary": {"full_name": "Jane Doe"},
            "petitioner": {"full_name": "Example Company"},
            "exhibits": [
                {
                    "number": 1,
                    "title": "Intent",
                    "items": [
                        {
                            "type": "document",
                            "name": "Intent to Depart",
                            "url": "https://storage.example.com/secret.docx",
                        }
                    ],
                }
            ],
        }

        result = asyncio.run(
            async_handle_doc_generation(
                [],
                ["Exhibit List"],
                adapter,
                submitted_parties=submitted,
            )
        )

        self.assertEqual(result, ["Exhibit_List.docx"])
        corpus_mock.assert_not_called()
        self.assertEqual(adapter.generation_call["source_manifest"], [])
        context = adapter.generation_call["retrieved_context"]
        self.assertEqual(len(context), 1)
        self.assertNotIn("https://", context[0].page_content)
