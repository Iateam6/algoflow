from django.test import SimpleTestCase

from case_jobs.retrieval.bm25 import BM25Index
from case_jobs.retrieval.chunking import EvidenceChunk, chunk_sources
from case_jobs.retrieval.fusion import reciprocal_rank_fusion


class RetrievalTests(SimpleTestCase):
    def test_structural_chunking_keeps_label_and_value_together(self):
        chunks = chunk_sources(
            [
                {
                    "file_hash": "hash-1",
                    "pages": ["Beneficiary: Dr. Amritpal Sandhu\nPetitioner: Dr. Amritpal Sandhu"],
                }
            ],
            tenant_id="tenant-a",
            case_id="case-a",
        )
        self.assertEqual(len(chunks), 1)
        self.assertIn("Beneficiary: Dr. Amritpal Sandhu", chunks[0].text)
        self.assertIn("Petitioner: Dr. Amritpal Sandhu", chunks[0].text)

    def test_bm25_finds_exact_identity(self):
        chunks = [
            EvidenceChunk("a", "Unrelated evidence", "t", "c", "h1", 1, 0),
            EvidenceChunk("b", "Beneficiary Amritpal Sandhu", "t", "c", "h2", 1, 0),
        ]
        self.assertEqual(BM25Index(chunks).search("Amritpal Sandhu")[0].chunk_id, "b")

    def test_reciprocal_rank_fusion_deduplicates_chunks(self):
        a = EvidenceChunk("a", "A", "t", "c", "h1", 1, 0)
        b = EvidenceChunk("b", "B", "t", "c", "h2", 1, 0)
        result = reciprocal_rank_fusion([[a, b], [b, a]])
        self.assertEqual({chunk.chunk_id for chunk in result}, {"a", "b"})

