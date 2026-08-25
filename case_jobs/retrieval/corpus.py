from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from case_jobs.retrieval.chunking import EvidenceChunk, chunk_sources


@dataclass(frozen=True)
class CorpusVersions:
    extraction: str = "1"
    chunking: str = "1"
    embedding_model: str = "text-embedding-3-small"
    retrieval: str = "1"
    pipeline: str = "1"


def build_corpus_hash(
    *,
    tenant_id: str,
    case_id: str,
    source_hashes: list[str],
    versions: CorpusVersions | None = None,
) -> str:
    versions = versions or CorpusVersions()
    payload = {
        "tenant_id": tenant_id,
        "case_id": case_id,
        "source_hashes": sorted(source_hashes),
        "versions": versions.__dict__,
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_case_chunks(
    sources: list[dict], tenant_id: str, case_id: str
) -> list[EvidenceChunk]:
    return chunk_sources(sources, tenant_id=tenant_id, case_id=case_id)

