from __future__ import annotations

import logging
from collections.abc import Callable

from case_jobs.retrieval.bm25 import BM25Index
from case_jobs.retrieval.chunking import EvidenceChunk
from case_jobs.retrieval.fusion import reciprocal_rank_fusion
from case_jobs.retrieval.semantic import SemanticIndex


logger = logging.getLogger(__name__)


class HybridRetriever:
    """Fuse lexical, cosine-semantic, and MMR rankings for one case corpus."""

    def __init__(
        self,
        *,
        bm25_index: BM25Index,
        semantic_index: SemanticIndex,
        mmr_search: Callable[[str, int, int], list[EvidenceChunk]],
        candidate_k: int = 30,
        final_k: int = 20,
        mmr_fetch_k: int = 60,
    ):
        if candidate_k < final_k:
            raise ValueError("candidate_k cannot be lower than final_k")
        if mmr_fetch_k < candidate_k:
            raise ValueError("mmr_fetch_k cannot be lower than candidate_k")
        self.bm25_index = bm25_index
        self.semantic_index = semantic_index
        self.mmr_search = mmr_search
        self.candidate_k = candidate_k
        self.final_k = final_k
        self.mmr_fetch_k = mmr_fetch_k

    def retrieve(self, query: str) -> list[EvidenceChunk]:
        rankings: list[list[EvidenceChunk]] = []
        backends = (
            ("bm25", lambda: self.bm25_index.search(query, self.candidate_k)),
            ("semantic", lambda: self.semantic_index.search(query, self.candidate_k)),
            (
                "mmr",
                lambda: self.mmr_search(
                    query,
                    self.candidate_k,
                    self.mmr_fetch_k,
                ),
            ),
        )
        for name, search in backends:
            try:
                ranking = search()
            except Exception:
                logger.warning(
                    "Hybrid retrieval backend failed backend=%s query=%r",
                    name,
                    query[:120],
                    exc_info=True,
                )
                continue
            if ranking:
                rankings.append(ranking)
            logger.info(
                "Hybrid retrieval backend=%s candidates=%d query=%r",
                name,
                len(ranking),
                query[:120],
            )
        if not rankings:
            return []
        return reciprocal_rank_fusion(rankings, limit=self.final_k)
