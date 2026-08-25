import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from django.test import SimpleTestCase, override_settings

from case_jobs.pipeline import context_enrichment
from case_jobs.pipeline.context_enrichment import ExhibitMarkdownAgent
from case_jobs.pipeline.legacy_generation import build_exhibit_request_prompt
from case_jobs.registry import ConfiguredVisaAdapter


class _FakeResponses:
    def __init__(self, outputs=None, error=None):
        self.outputs = list(outputs or [])
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(output_text=self.outputs.pop(0))


class _FakeClient:
    def __init__(self, outputs=None, error=None):
        self.responses = _FakeResponses(outputs=outputs, error=error)


@override_settings(
    AUXILIARY_CONTEXT_MODEL="test-context-model",
)
class ContextEnrichmentTests(SimpleTestCase):
    def test_placeholder_resolution_layer_is_not_exposed(self):
        self.assertFalse(hasattr(context_enrichment, "TemplateFieldResolver"))
        self.assertFalse(hasattr(context_enrichment, "TemplateResolution"))

    def test_exhibit_markdown_falls_back_with_order_and_without_urls(self):
        exhibits = [
            {
                "number": 1,
                "title": "Forms",
                "items": [
                    {
                        "type": "document",
                        "name": "Intent to Depart",
                        "url": "https://storage.example.com/intent.pdf",
                    },
                    {
                        "type": "file",
                        "name": "Education Credentials",
                        "files": [
                            {"url": "https://storage.example.com/one.pdf"},
                            {"url": "https://storage.example.com/two.pdf"},
                        ],
                    },
                ],
            }
        ]

        markdown_text = asyncio.run(
            ExhibitMarkdownAgent(
                client=_FakeClient(error=RuntimeError("model unavailable"))
            ).generate(exhibits)
        )

        self.assertLess(markdown_text.index("1.1"), markdown_text.index("1.2"))
        self.assertIn("2 files", markdown_text)
        self.assertNotIn("https://", markdown_text)

    def test_exhibit_generation_prompt_is_compact_and_url_free(self):
        submitted_request = {
            "beneficiary": {"full_name": "Jane Doe", "address": {"city": "Toronto"}},
            "petitioner": {"full_name": "Example Company", "address": {"city": "Austin"}},
            "preparer": {"full_name": "Counsel Name"},
            "exhibits": [
                {
                    "number": 2,
                    "title": "Intent to Depart",
                    "items": [
                        {
                            "type": "document",
                            "name": "Intent to Depart",
                            "url": "https://storage.example.com/opaque.docx",
                        }
                    ],
                }
            ],
        }

        prompt = build_exhibit_request_prompt(submitted_request).page_content

        self.assertIn('"beneficiary_name":"Jane Doe"', prompt)
        self.assertIn('"petitioner_name":"Example Company"', prompt)
        self.assertIn('"file_count":1', prompt)
        self.assertNotIn("https://", prompt)
        self.assertNotIn("Toronto", prompt)
        self.assertNotIn("Counsel Name", prompt)

    @patch("case_jobs.pipeline.legacy_generation.handle_doc_generation")
    def test_adapter_forwards_verification_corrections(self, generation_mock):
        generation_mock.return_value = ["output.docx"]
        adapter = ConfiguredVisaAdapter(
            visa_type="test",
            display_name="Test",
            cache_namespace="test",
            pipeline_version="v1",
            supported_document_types=frozenset({"Support Letter"}),
            retrieval_query_builder=lambda value: value,
            document_generator=lambda **_kwargs: None,
            template_provider=lambda _value: "[Beneficiary Full Name]",
        )

        result = adapter.generate(
            [],
            "Support Letter",
            corrections=["Fill the closing sign-off"],
        )

        self.assertEqual(result, ["output.docx"])
        self.assertEqual(
            generation_mock.call_args.kwargs["corrections"],
            ["Fill the closing sign-off"],
        )
