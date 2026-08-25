from __future__ import annotations

from collections.abc import Callable
import os

import faiss
import numpy as np

from case_jobs.retrieval.chunking import EvidenceChunk


class SemanticIndex:
    """FAISS cosine-similarity index with an injected embedding function."""

    def __init__(
        self,
        chunks: list[EvidenceChunk],
        embed_texts: Callable[[list[str]], list[list[float]]],
        *,
        vectors: list[list[float]] | np.ndarray | None = None,
    ):
        self.chunks = chunks
        self.embed_texts = embed_texts
        self.index = None
        if chunks:
            matrix = np.asarray(
                vectors
                if vectors is not None
                else embed_texts([chunk.text for chunk in chunks]),
                dtype="float32",
            )
            if matrix.ndim != 2 or matrix.shape[0] != len(chunks):
                raise ValueError("Embedding function returned an invalid matrix")
            faiss.normalize_L2(matrix)
            self.index = faiss.IndexFlatIP(matrix.shape[1])
            self.index.add(matrix)

    @classmethod
    def load(
        cls,
        path: str,
        chunks: list[EvidenceChunk],
        embed_texts: Callable[[list[str]], list[list[float]]],
    ) -> "SemanticIndex":
        instance = cls.__new__(cls)
        instance.chunks = chunks
        instance.embed_texts = embed_texts
        instance.index = faiss.read_index(path)
        if instance.index.ntotal != len(chunks):
            raise ValueError("Persisted semantic index does not match canonical chunks")
        return instance

    def save(self, path: str) -> None:
        if self.index is None:
            raise ValueError("Cannot persist an empty semantic index")
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, path)

    def search(self, query: str, limit: int = 10) -> list[EvidenceChunk]:
        if self.index is None:
            return []
        vector = np.asarray(self.embed_texts([query]), dtype="float32")
        if vector.ndim != 2 or vector.shape[0] != 1:
            raise ValueError("Embedding function returned an invalid query vector")
        faiss.normalize_L2(vector)
        _, indices = self.index.search(vector, min(limit, len(self.chunks)))
        return [self.chunks[index] for index in indices[0] if index >= 0]

