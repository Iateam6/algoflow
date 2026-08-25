import json
from unittest.mock import AsyncMock, patch

from django.test import SimpleTestCase, override_settings

from case_jobs.api.validators import validate_generation_request
from case_jobs.pipeline.legacy_generation import (
    build_exhibit_request_prompt,
    build_submitted_parties_prompt,
    handle_doc_generation,
)
from case_jobs.pipeline.orchestrator import _submitted_request_context
from case_jobs.tests.test_validation import valid_exhibit_payload


class _FakeRetrievedDocument:
    def __init__(self, *, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata


class _FakeChunk:
    def __init__(self, *, text: str, source_name: str):
        self.text = text
        self.metadata = {
            "source_name": source_name,
            "source_category": "must-not-be-sent",
            "file_hash": "must-not-be-sent",
            "page_number": 99,
            "chunk_index": 99,
            "extraction_mode": "must-not-be-sent",
        }


class _FakeCorpusBundle:
    corpus_hash = "corpus-hash"
    cache_hit = True
    chunk_count = 3

    def __init__(self):
        self.processed_sources = [
            {
                "name": "source-1",
                "original_filename": "evidence.pdf",
                "url": "https://example.com/evidence.pdf",
                "content_type": "application/pdf",
                "extension": ".pdf",
                "local_path": "source_1.pdf",
                "file_hash": "hash-1",
                "latex_path": "evidence.tex",
                "page_count": 1,
                "extraction_mode": "pdf",
                "page_entries": [],
            }
        ]
        self.chunks = [
            _FakeChunk(text="First corpus chunk.", source_name="first.pdf"),
            _FakeChunk(text="Second corpus chunk.", source_name="second.docx"),
            _FakeChunk(text="Third corpus chunk.", source_name="third.pdf"),
        ]

    def retrieve(self, _query: str):
        raise AssertionError("ranked retrieval must not run")


@override_settings(PRINT_GENERATION_MARKDOWN=False)
class PromptInjectionTests(SimpleTestCase):
    @override_settings(WEBHOOK_ROOT_URL="https://api.visa26.com/webhooks/documents")
    def test_tn_exhibit_structure_is_injected_in_authoritative_order(self):
        request = validate_generation_request(valid_exhibit_payload(), "tn")
        context = _submitted_request_context(request)
        prompt = build_exhibit_request_prompt(context)

        self.assertIn("Submitted TN Exhibit Request", prompt.page_content)
        self.assertIn("must not appear", prompt.page_content)
        self.assertLess(
            prompt.page_content.index("Form G-28"),
            prompt.page_content.index("Company Records"),
        )
        self.assertLess(
            prompt.page_content.index("Company Records"),
            prompt.page_content.index("Support Letter"),
        )

    def test_submitted_parties_are_prepended_to_retrieved_context(self):
        submitted_parties = {
            "beneficiary": {
                "full_name": "Amritpal Sandhu",
                "address": {
                    "line_1": "1 Main St",
                    "line_2": None,
                    "city": "Nairobi",
                    "state": None,
                    "province": "Nairobi County",
                    "postal_code": "00100",
                    "country": "Kenya",
                },
            },
            "petitioner": {
                "full_name": "Amritpal Sandhu",
                "address": {
                    "line_1": "1 Main St",
                    "line_2": None,
                    "city": "Nairobi",
                    "state": None,
                    "province": "Nairobi County",
                    "postal_code": "00100",
                    "country": "Kenya",
                },
            },
            "preparer": {
                "full_name": "Jane Smith",
                "firm_name": "Boston Legal",
                "address": {
                    "line_1": "2 Court Ave",
                    "line_2": None,
                    "city": "Boston",
                    "state": "MA",
                    "province": None,
                    "postal_code": "02101",
                    "country": "USA",
                },
            },
        }
        expected_json = json.dumps(submitted_parties, sort_keys=True, separators=(",", ":"))

        class _FakeAdapter:
            display_name = "Test Visa"
            visa_type = "tn"

            def rag_config(self, *, tenant_id="legacy", case_id="legacy"):
                return object()

            def build_retrieval_query(self, _document_type: str) -> str:
                return "query"

            def document_template(self, _document_type: str) -> str:
                return "A document template without placeholder fields."

            async def generate_document(self, *, file_type, retrieved_context, source_manifest):
                first = retrieved_context[0]
                assert first.metadata.get("source_name") == "request_payload"
                assert "Submitted Parties" in first.page_content
                assert expected_json in first.page_content
                assert "authoritative" in first.page_content.lower()
                assert [item.page_content for item in retrieved_context[1:]] == [
                    "First corpus chunk.",
                    "Second corpus chunk.",
                    "Third corpus chunk.",
                ]
                assert all(
                    set(item.metadata) == {"source_name"}
                    for item in retrieved_context[1:]
                )
                assert source_manifest == []
                return "```markdown\nHello\n```"

        adapter = _FakeAdapter()

        with patch(
            "case_jobs.pipeline.legacy_generation.clean_media_files_for_generation",
            autospec=True,
        ), patch(
            "case_jobs.pipeline.legacy_generation.get_or_build_corpus",
            new=AsyncMock(return_value=_FakeCorpusBundle()),
        ), patch(
            "case_jobs.pipeline.legacy_generation.convert_markdown_to_docx",
            new=AsyncMock(return_value="media/generated/out.docx"),
        ):
            paths = handle_doc_generation(
                file_manifests=[
                    {
                        "name": "source-1",
                        "original_filename": "evidence.pdf",
                        "url": "https://example.com/evidence.pdf",
                        "content_type": "application/pdf",
                        "extension": ".pdf",
                        "local_path": "source_1.pdf",
                        "file_hash": "hash-1",
                    }
                ],
                selected_options=["support-letter"],
                adapter=adapter,
                tenant_id="tenant-1",
                case_id="case-1",
                submitted_parties=submitted_parties,
            )
        self.assertEqual(paths, ["media/generated/out.docx"])
