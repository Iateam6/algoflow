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

from .document_processing import build_file_header, process_files_to_latex
from .openai_client import get_openai_api_key


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
    ocr_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 1200
    chunk_overlap: int = 200
    retrieval_k: int = 10
    retrieval_fetch_k: int = 30
    search_type: str = "mmr"
    pipeline_version: str = "reentry-permit-rag-v1"


@dataclass
class CorpusBundle:
    corpus_hash: str
    cache_dir: str
    vector_store: FAISS
    processed_sources: list[dict[str, Any]]
    chunk_count: int
    cache_hit: bool
    config: RAGConfig

    def retrieve(self, query: str) -> list[Document]:
        retriever = self.vector_store.as_retriever(
            search_type=self.config.search_type,
            search_kwargs={
                "k": self.config.retrieval_k,
                "fetch_k": self.config.retrieval_fetch_k,
            },
        )
        try:
            documents = retriever.invoke(query)
        except AttributeError:
            documents = self.vector_store.max_marginal_relevance_search(
                query,
                k=self.config.retrieval_k,
                fetch_k=self.config.retrieval_fetch_k,
            )

        logger.info(
            "Retrieved %s chunks for corpus %s using query '%s'",
            len(documents),
            self.corpus_hash,
            query[:120],
        )
        return documents


def get_default_config() -> RAGConfig:
    cache_root = os.path.join(settings.MEDIA_ROOT, "reentry_permit_rag")
    return RAGConfig(cache_root=cache_root)


def build_corpus_hash(file_manifests: list[dict[str, Any]], config: RAGConfig) -> str:
    payload = {
        "pipeline_version": config.pipeline_version,
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
    return chunk_documents


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
    config = config or get_default_config()
    os.makedirs(config.cache_root, exist_ok=True)

    corpus_hash = build_corpus_hash(file_manifests, config)
    cache_paths = get_cache_paths(corpus_hash, config)
    embeddings = ensure_embedding_environment(config)

    logger.info("Preparing reentry permit corpus %s", corpus_hash)

    cache_hit = os.path.isdir(cache_paths["vectorstore_dir"]) and os.path.exists(cache_paths["manifest_path"])
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
        return CorpusBundle(
            corpus_hash=corpus_hash,
            cache_dir=cache_paths["cache_dir"],
            vector_store=vector_store,
            processed_sources=processed_sources,
            chunk_count=chunk_count,
            cache_hit=True,
            config=config,
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
    vector_store = FAISS.from_documents(chunk_documents, embeddings)
    vector_store.save_local(cache_paths["vectorstore_dir"])

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
    )
