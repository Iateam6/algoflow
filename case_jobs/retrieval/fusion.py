from __future__ import annotations

from case_jobs.retrieval.chunking import EvidenceChunk


def reciprocal_rank_fusion(
    rankings: list[list[EvidenceChunk]],
    *,
    limit: int = 20,
    rank_constant: int = 60,
) -> list[EvidenceChunk]:
    scores: dict[str, float] = {}
    chunks: dict[str, EvidenceChunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (
                rank_constant + rank
            )
            chunks[chunk.chunk_id] = chunk
    ordered = sorted(scores, key=lambda key: (-scores[key], key))
    return [chunks[key] for key in ordered[:limit]]

