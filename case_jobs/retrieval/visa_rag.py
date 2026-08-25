import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from django.conf import settings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np

from case_jobs.integrations.openai_client import get_openai_api_key
from case_jobs.pipeline.document_processing import build_file_header, process_files_to_latex
from case_jobs.retrieval.bm25 import BM25Index
from case_jobs.retrieval.chunking import EvidenceChunk
from case_jobs.retrieval.hybrid import HybridRetriever
from case_jobs.retrieval.semantic import SemanticIndex


logger = logging.getLogger(__name__)

LATEX_SEPARATORS = [
    r"\section{",
    r"\section*{",
    r"\subsection{",
    r"\subsection*{",
    r"\begin{itemize}",
    r"\begin{enumerate}",
    r"\newpage",
    "\n\n",
    "\n",
    " ",
    "",
]


@dataclass(frozen=True)
class RAGConfig:
    cache_root: str
    visa_type: str = "test"
    tenant_id: str = "legacy"
    case_id: str = "legacy"
    ocr_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 1200
    chunk_overlap: int = 200
    hybrid_candidate_k: int = 30
    hybrid_final_k: int = 20
    hybrid_mmr_fetch_k: int = 60
    retrieval_version: str = "hybrid-v1"
    pipeline_version: str = "rag-v1"

    def __post_init__(self) -> None:
        if self.hybrid_candidate_k < self.hybrid_final_k:
            raise ValueError(
                "HYBRID_RETRIEVAL_CANDIDATE_K cannot be lower than "
                "HYBRID_RETRIEVAL_FINAL_K"
            )
        if self.hybrid_mmr_fetch_k < self.hybrid_candidate_k:
            raise ValueError(
                "HYBRID_RETRIEVAL_MMR_FETCH_K cannot be lower than "
                "HYBRID_RETRIEVAL_CANDIDATE_K"
            )


@dataclass
class CorpusBundle:
    corpus_hash: str
    cache_dir: str
    vector_store: FAISS
    processed_sources: list[dict[str, Any]]
    chunk_count: int
    cache_hit: bool
    config: RAGConfig
    chunks: list[EvidenceChunk]
    hybrid_retriever: HybridRetriever

    def retrieve(self, query: str) -> list[Document]:
        chunks = self.hybrid_retriever.retrieve(query)
        documents = [
            Document(page_content=chunk.text, metadata=dict(chunk.metadata))
            for chunk in chunks
        ]

        logger.info(
            "Retrieved %s fused chunks for corpus %s using query '%s'",
            len(documents),
            self.corpus_hash,
            query[:120],
        )
        return documents


def build_rag_config(
    *,
    visa_type: str,
    cache_namespace: str,
    pipeline_version: str,
    tenant_id: str = "legacy",
    case_id: str = "legacy",
) -> RAGConfig:
    cache_root = os.path.join(
        settings.MEDIA_ROOT,
        cache_namespace,
        tenant_id,
        case_id,
    )
    return RAGConfig(
        visa_type=visa_type,
        tenant_id=tenant_id,
        case_id=case_id,
        cache_root=cache_root,
        pipeline_version=pipeline_version,
        embedding_model=getattr(
            settings,
            "RAG_EMBEDDING_MODEL",
            "text-embedding-3-small",
        ),
        hybrid_candidate_k=settings.HYBRID_RETRIEVAL_CANDIDATE_K,
        hybrid_final_k=settings.HYBRID_RETRIEVAL_FINAL_K,
        hybrid_mmr_fetch_k=settings.HYBRID_RETRIEVAL_MMR_FETCH_K,
    )


def build_corpus_hash(file_manifests: list[dict[str, Any]], config: RAGConfig) -> str:
    payload = {
        "visa_type": config.visa_type,
        "tenant_id": config.tenant_id,
        "case_id": config.case_id,
        "pipeline_version": config.pipeline_version,
        "retrieval_version": config.retrieval_version,
        "embedding_model": config.embedding_model,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "file_hashes": sorted(file_manifest["file_hash"] for file_manifest in file_manifests),
    }
    raw_payload = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw_payload).hexdigest()


def get_cache_paths(corpus_hash: str, config: RAGConfig) -> dict[str, str]:
    cache_dir = os.path.join(config.cache_root, corpus_hash)
    return {
        "cache_dir": cache_dir,
        "manifest_path": os.path.join(cache_dir, "manifest.json"),
        "vectorstore_dir": os.path.join(cache_dir, "faiss_index"),
        "semantic_index_path": os.path.join(cache_dir, "semantic.index"),
        "chunks_path": os.path.join(cache_dir, "chunks.json"),
        "latex_dir": os.path.join(cache_dir, "latex"),
    }


def ensure_embedding_environment(config: RAGConfig) -> OpenAIEmbeddings:
    os.environ.setdefault("OPENAI_API_KEY", get_openai_api_key())
    return OpenAIEmbeddings(model=config.embedding_model)


def summarise_processed_sources(processed_sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for source in processed_sources:
        summaries.append(
            {
                "name": source["name"],
                "original_filename": source["original_filename"],
                "url": source["url"],
                "content_type": source["content_type"],
                "extension": source["extension"],
                "local_path": source["local_path"],
                "file_hash": source["file_hash"],
                "latex_path": source["latex_path"],
                "page_count": source["page_count"],
                "extraction_mode": source["extraction_mode"],
                "page_entries": [
                    {
                        "page_number": page_entry["page_number"],
                        "extraction_mode": page_entry["extraction_mode"],
                    }
                    for page_entry in source.get("page_entries", [])
                ],
            }
        )
    return summaries


def build_langchain_documents(processed_sources: list[dict[str, Any]]) -> list[Document]:
    documents: list[Document] = []
    for source in processed_sources:
        for page_entry in source.get("page_entries", []):
            page_number = page_entry["page_number"]
            page_content = "\n\n".join(
                [
                    build_file_header(
                        source["original_filename"],
                        source["name"],
                        page_number,
                    ),
                    page_entry["latex_text"],
                ]
            ).strip()
            documents.append(
                Document(
                    page_content=page_content,
                    metadata={
                        "source_name": source["original_filename"],
                        "source_category": source["name"],
                        "file_hash": source["file_hash"],
                        "page_number": page_number,
                        "mime_type": source["content_type"],
                        "extraction_mode": page_entry["extraction_mode"],
                        "latex_path": source["latex_path"],
                    },
                )
            )
    return documents


def build_text_splitter(config: RAGConfig) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        separators=LATEX_SEPARATORS,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )


def assign_chunk_metadata(chunk_documents: list[Document]) -> list[Document]:
    counters: dict[tuple[str, int], int] = {}
    for document in chunk_documents:
        file_hash = document.metadata["file_hash"]
        page_number = document.metadata["page_number"]
        counter_key = (file_hash, page_number)
        counters[counter_key] = counters.get(counter_key, 0) + 1
        document.metadata["chunk_index"] = counters[counter_key]
        document.metadata["chunk_id"] = ":".join(
            (
                str(document.metadata.get("tenant_id", "")),
                str(document.metadata.get("case_id", "")),
                str(file_hash),
                str(page_number),
                str(counters[counter_key]),
            )
        )
    return chunk_documents


def build_canonical_chunks(
    chunk_documents: list[Document],
    config: RAGConfig,
) -> list[EvidenceChunk]:
    chunks: list[EvidenceChunk] = []
    for document in chunk_documents:
        metadata = dict(document.metadata)
        metadata["tenant_id"] = config.tenant_id
        metadata["case_id"] = config.case_id
        chunk_id = ":".join(
            (
                config.tenant_id,
                config.case_id,
                str(metadata["file_hash"]),
                str(metadata["page_number"]),
                str(metadata["chunk_index"]),
            )
        )
        metadata["chunk_id"] = chunk_id
        document.metadata.update(metadata)
        chunks.append(
            EvidenceChunk(
                chunk_id=chunk_id,
                text=document.page_content,
                tenant_id=config.tenant_id,
                case_id=config.case_id,
                source_hash=str(metadata["file_hash"]),
                page_number=int(metadata["page_number"]),
                chunk_index=int(metadata["chunk_index"]),
                heading=metadata.get("heading"),
                metadata=metadata,
            )
        )
    return chunks


def write_canonical_chunks(path: str, chunks: list[EvidenceChunk]) -> None:
    payload = [
        {
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "tenant_id": chunk.tenant_id,
            "case_id": chunk.case_id,
            "source_hash": chunk.source_hash,
            "page_number": chunk.page_number,
            "chunk_index": chunk.chunk_index,
            "heading": chunk.heading,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    ]
    with open(path, "w", encoding="utf-8") as chunks_file:
        json.dump(payload, chunks_file, ensure_ascii=False)


def load_canonical_chunks(path: str) -> list[EvidenceChunk]:
    with open(path, "r", encoding="utf-8") as chunks_file:
        payload = json.load(chunks_file)
    return [EvidenceChunk(**item) for item in payload]


def _query_embeddings(embeddings: OpenAIEmbeddings):
    def embed_texts(texts: list[str]) -> list[list[float]]:
        return [embeddings.embed_query(text) for text in texts]

    return embed_texts


def _vector_store_vectors(vector_store: FAISS, count: int) -> np.ndarray:
    return np.asarray(
        [vector_store.index.reconstruct(index) for index in range(count)],
        dtype="float32",
    )


def build_hybrid_retriever(
    *,
    vector_store: FAISS,
    chunks: list[EvidenceChunk],
    semantic_index: SemanticIndex,
    config: RAGConfig,
) -> HybridRetriever:
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}

    def mmr_search(query: str, limit: int, fetch_k: int) -> list[EvidenceChunk]:
        documents = vector_store.max_marginal_relevance_search(
            query,
            k=limit,
            fetch_k=fetch_k,
        )
        ranking: list[EvidenceChunk] = []
        for document in documents:
            chunk = chunk_by_id.get(str(document.metadata.get("chunk_id", "")))
            if chunk is not None:
                ranking.append(chunk)
        return ranking

    return HybridRetriever(
        bm25_index=BM25Index(chunks),
        semantic_index=semantic_index,
        mmr_search=mmr_search,
        candidate_k=config.hybrid_candidate_k,
        final_k=config.hybrid_final_k,
        mmr_fetch_k=config.hybrid_mmr_fetch_k,
    )


def write_manifest(
    manifest_path: str,
    corpus_hash: str,
    file_manifests: list[dict[str, Any]],
    processed_sources: list[dict[str, Any]],
    chunk_count: int,
    cache_hit: bool,
    config: RAGConfig,
) -> None:
    manifest_data = {
        "corpus_hash": corpus_hash,
        "pipeline_version": config.pipeline_version,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "cache_hit": cache_hit,
        "chunk_count": chunk_count,
        "file_hashes": sorted(file_manifest["file_hash"] for file_manifest in file_manifests),
        "processed_sources": summarise_processed_sources(processed_sources),
    }
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump(manifest_data, manifest_file, indent=2)


def load_manifest(manifest_path: str) -> dict[str, Any]:
    with open(manifest_path, "r", encoding="utf-8") as manifest_file:
        return json.load(manifest_file)


async def get_or_build_corpus(
    file_manifests: list[dict[str, Any]],
    config: RAGConfig | None = None,
) -> CorpusBundle:
    if config is None:
        raise ValueError("A visa-scoped RAGConfig is required")
    os.makedirs(config.cache_root, exist_ok=True)

    corpus_hash = build_corpus_hash(file_manifests, config)
    cache_paths = get_cache_paths(corpus_hash, config)
    embeddings = ensure_embedding_environment(config)
    embed_queries = _query_embeddings(embeddings)

    logger.info("Preparing %s corpus %s", config.visa_type, corpus_hash)

    cache_hit = all(
        (
            os.path.isdir(cache_paths["vectorstore_dir"]),
            os.path.exists(cache_paths["manifest_path"]),
            os.path.exists(cache_paths["semantic_index_path"]),
            os.path.exists(cache_paths["chunks_path"]),
        )
    )
    if cache_hit:
        logger.info("Cache hit for corpus %s", corpus_hash)
        vector_store = FAISS.load_local(
            cache_paths["vectorstore_dir"],
            embeddings,
            allow_dangerous_deserialization=True,
        )
        manifest_data = load_manifest(cache_paths["manifest_path"])
        processed_sources = manifest_data.get("processed_sources", [])
        chunk_count = manifest_data.get("chunk_count", 0)
        chunks = load_canonical_chunks(cache_paths["chunks_path"])
        semantic_index = SemanticIndex.load(
            cache_paths["semantic_index_path"],
            chunks,
            embed_queries,
        )
        hybrid_retriever = build_hybrid_retriever(
            vector_store=vector_store,
            chunks=chunks,
            semantic_index=semantic_index,
            config=config,
        )
        return CorpusBundle(
            corpus_hash=corpus_hash,
            cache_dir=cache_paths["cache_dir"],
            vector_store=vector_store,
            processed_sources=processed_sources,
            chunk_count=chunk_count,
            cache_hit=True,
            config=config,
            chunks=chunks,
            hybrid_retriever=hybrid_retriever,
        )

    logger.info("Cache miss for corpus %s", corpus_hash)
    os.makedirs(cache_paths["cache_dir"], exist_ok=True)
    os.makedirs(cache_paths["latex_dir"], exist_ok=True)

    processed_sources = await process_files_to_latex(
        file_manifests=file_manifests,
        latex_output_dir=cache_paths["latex_dir"],
        ocr_model=config.ocr_model,
    )
    langchain_documents = build_langchain_documents(processed_sources)
    splitter = build_text_splitter(config)
    chunk_documents = assign_chunk_metadata(splitter.split_documents(langchain_documents))
    chunks = build_canonical_chunks(chunk_documents, config)
    vector_store = FAISS.from_documents(chunk_documents, embeddings)
    vector_store.save_local(cache_paths["vectorstore_dir"])
    semantic_index = SemanticIndex(
        chunks,
        embed_queries,
        vectors=_vector_store_vectors(vector_store, len(chunks)),
    )
    semantic_index.save(cache_paths["semantic_index_path"])
    write_canonical_chunks(cache_paths["chunks_path"], chunks)
    hybrid_retriever = build_hybrid_retriever(
        vector_store=vector_store,
        chunks=chunks,
        semantic_index=semantic_index,
        config=config,
    )

    write_manifest(
        manifest_path=cache_paths["manifest_path"],
        corpus_hash=corpus_hash,
        file_manifests=file_manifests,
        processed_sources=processed_sources,
        chunk_count=len(chunk_documents),
        cache_hit=False,
        config=config,
    )

    logger.info(
        "Built corpus %s with %s processed sources and %s chunks",
        corpus_hash,
        len(processed_sources),
        len(chunk_documents),
    )

    return CorpusBundle(
        corpus_hash=corpus_hash,
        cache_dir=cache_paths["cache_dir"],
        vector_store=vector_store,
        processed_sources=summarise_processed_sources(processed_sources),
        chunk_count=len(chunk_documents),
        cache_hit=False,
        config=config,
        chunks=chunks,
        hybrid_retriever=hybrid_retriever,
    )
