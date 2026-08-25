import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from django.test import SimpleTestCase
from langchain_core.documents import Document

from case_jobs.retrieval.chunking import EvidenceChunk
from case_jobs.retrieval.hybrid import HybridRetriever
from case_jobs.retrieval.semantic import SemanticIndex
from case_jobs.retrieval.visa_rag import RAGConfig, get_or_build_corpus


def _chunk(index: int) -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=f"chunk-{index}",
        text=f"Evidence chunk {index}",
        tenant_id="tenant",
        case_id="case",
        source_hash="source",
        page_number=1,
        chunk_index=index,
        metadata={"chunk_id": f"chunk-{index}"},
    )


class _FakeIndex:
    def __init__(self, ranking=None, error=None):
        self.ranking = list(ranking or [])
        self.error = error
        self.calls = []

    def search(self, query, limit):
        self.calls.append((query, limit))
        if self.error:
            raise self.error
        return self.ranking[:limit]


class HybridRetrieverTests(SimpleTestCase):
    def test_three_rankings_are_fused_to_twenty_unique_chunks(self):
        chunks = [_chunk(index) for index in range(35)]
        bm25 = _FakeIndex(chunks)
        semantic = _FakeIndex(list(reversed(chunks)))
        mmr_calls = []

        def mmr_search(query, limit, fetch_k):
            mmr_calls.append((query, limit, fetch_k))
            return chunks[5 : 5 + limit]

        retriever = HybridRetriever(
            bm25_index=bm25,
            semantic_index=semantic,
            mmr_search=mmr_search,
        )
        result = retriever.retrieve("beneficiary achievements")

        self.assertEqual(bm25.calls, [("beneficiary achievements", 30)])
        self.assertEqual(semantic.calls, [("beneficiary achievements", 30)])
        self.assertEqual(mmr_calls, [("beneficiary achievements", 30, 60)])
        self.assertEqual(len(result), 20)
        self.assertEqual(len({chunk.chunk_id for chunk in result}), 20)

    def test_backend_failures_return_available_ranking(self):
        chunks = [_chunk(index) for index in range(5)]
        retriever = HybridRetriever(
            bm25_index=_FakeIndex(chunks),
            semantic_index=_FakeIndex(error=RuntimeError("semantic unavailable")),
            mmr_search=lambda *_args: (_ for _ in ()).throw(RuntimeError("mmr unavailable")),
        )
        self.assertEqual(retriever.retrieve("query"), chunks)

    def test_limits_are_validated(self):
        with self.assertRaises(ValueError):
            RAGConfig("cache", hybrid_candidate_k=10, hybrid_final_k=20)
        with self.assertRaises(ValueError):
            RAGConfig("cache", hybrid_candidate_k=30, hybrid_mmr_fetch_k=20)

    def test_semantic_index_can_be_reloaded(self):
        chunks = [_chunk(0), _chunk(1)]

        def embed(values):
            return [[1.0, 0.0] if "zero" in value else [0.0, 1.0] for value in values]

        index = SemanticIndex(
            chunks,
            embed,
            vectors=[[1.0, 0.0], [0.0, 1.0]],
        )
        with patch("case_jobs.retrieval.semantic.faiss.write_index") as writer:
            index.save("semantic.index")
        writer.assert_called_once_with(index.index, "semantic.index")
        with patch(
            "case_jobs.retrieval.semantic.faiss.read_index",
            return_value=index.index,
        ):
            loaded = SemanticIndex.load("semantic.index", chunks, embed)
        self.assertEqual(loaded.search("zero", 1)[0].chunk_id, "chunk-0")

    def test_cache_miss_returns_fully_initialized_corpus_bundle(self):
        document = Document(
            page_content="Beneficiary: Jane Doe",
            metadata={
                "file_hash": "hash-1",
                "page_number": 1,
                "source_name": "source.pdf",
                "source_category": "identity",
                "extraction_mode": "native",
            },
        )
        splitter = SimpleNamespace(split_documents=lambda _documents: [document])
        vector_store = SimpleNamespace(
            save_local=lambda _path: None,
            index=SimpleNamespace(reconstruct=lambda _index: [1.0, 0.0]),
        )
        semantic = SimpleNamespace(save=lambda _path: None)
        hybrid = object()

        with (
            patch("case_jobs.retrieval.visa_rag.os.makedirs"),
            patch("case_jobs.retrieval.visa_rag.os.path.isdir", return_value=False),
            patch("case_jobs.retrieval.visa_rag.os.path.exists", return_value=False),
            patch("case_jobs.retrieval.visa_rag.ensure_embedding_environment", return_value=object()),
            patch(
                "case_jobs.retrieval.visa_rag.process_files_to_latex",
                return_value=[{"name": "identity"}],
            ),
            patch("case_jobs.retrieval.visa_rag.build_langchain_documents", return_value=[document]),
            patch("case_jobs.retrieval.visa_rag.summarise_processed_sources", return_value=[]),
            patch("case_jobs.retrieval.visa_rag.build_text_splitter", return_value=splitter),
            patch("case_jobs.retrieval.visa_rag.FAISS.from_documents", return_value=vector_store),
            patch("case_jobs.retrieval.visa_rag.SemanticIndex", return_value=semantic),
            patch("case_jobs.retrieval.visa_rag.write_canonical_chunks"),
            patch("case_jobs.retrieval.visa_rag.write_manifest"),
            patch("case_jobs.retrieval.visa_rag.build_hybrid_retriever", return_value=hybrid),
        ):
            bundle = asyncio.run(
                get_or_build_corpus(
                    [{"file_hash": "hash-1"}],
                    RAGConfig("cache"),
                )
            )

        self.assertEqual(len(bundle.chunks), 1)
        self.assertIs(bundle.hybrid_retriever, hybrid)
