from django.test import SimpleTestCase

from case_jobs.pipeline.document_processing import is_scanned_page, text_to_latex
from case_jobs.retrieval.visa_rag import RAGConfig, build_corpus_hash


class SharedDocumentProcessingTests(SimpleTestCase):
    def test_scanned_page_thresholds(self):
        self.assertTrue(is_scanned_page("short scan"))
        self.assertFalse(
            is_scanned_page(
                "This native page contains enough visible characters and words to "
                "be treated as searchable text instead of a scanned image requiring OCR."
            )
        )

    def test_text_to_latex_preserves_labels_and_escapes_values(self):
        latex = text_to_latex("Beneficiary: Jane Doe\nIncome: $100 & verified")
        self.assertIn("Beneficiary", latex)
        self.assertIn(r"\$100", latex)
        self.assertIn(r"\&", latex)


class SharedVisaRAGTests(SimpleTestCase):
    def test_corpus_hash_isolated_by_visa_tenant_and_case(self):
        source = [{"file_hash": "same-source"}]
        first = build_corpus_hash(
            source,
            RAGConfig("cache", visa_type="eb-1a", tenant_id="a", case_id="one"),
        )
        other_visa = build_corpus_hash(
            source,
            RAGConfig("cache", visa_type="o-1", tenant_id="a", case_id="one"),
        )
        other_tenant = build_corpus_hash(
            source,
            RAGConfig("cache", visa_type="eb-1a", tenant_id="b", case_id="one"),
        )
        other_case = build_corpus_hash(
            source,
            RAGConfig("cache", visa_type="eb-1a", tenant_id="a", case_id="two"),
        )
        self.assertEqual(len({first, other_visa, other_tenant, other_case}), 4)
