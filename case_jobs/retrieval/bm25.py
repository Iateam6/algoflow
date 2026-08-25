from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from case_jobs.retrieval.chunking import EvidenceChunk


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._'-]*")


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


class BM25Index:
    def __init__(self, chunks: list[EvidenceChunk]):
        self.chunks = chunks
        self.index = BM25Okapi([tokenize(chunk.text) for chunk in chunks]) if chunks else None

    def search(self, query: str, limit: int = 10) -> list[EvidenceChunk]:
        if not self.index:
            return []
        query_tokens = tokenize(query)
        query_set = set(query_tokens)
        query_phrase = " ".join(query_tokens)
        bm25_scores = self.index.get_scores(query_tokens)
        scores = []
        for chunk, bm25_score in zip(self.chunks, bm25_scores):
            document_tokens = tokenize(chunk.text)
            overlap = len(query_set.intersection(document_tokens))
            phrase_bonus = 2 if query_phrase and query_phrase in " ".join(document_tokens) else 0
            scores.append(float(bm25_score) + overlap + phrase_bonus)
        ranked = sorted(
            enumerate(scores),
            key=lambda item: (-float(item[1]), item[0]),
        )
        return [self.chunks[index] for index, score in ranked[:limit] if score > 0]
