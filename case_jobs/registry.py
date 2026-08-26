from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Protocol

from case_jobs.retrieval.visa_rag import RAGConfig, build_rag_config


class VisaAdapter(Protocol):
    @property
    def visa_type(self) -> str: ...

    @property
    def display_name(self) -> str: ...

    @property
    def cache_namespace(self) -> str: ...

    @property
    def pipeline_version(self) -> str: ...

    @property
    def supported_document_types(self) -> frozenset[str]: ...

    def build_retrieval_query(self, document_type: str) -> str: ...

    def document_template(self, document_type: str) -> str: ...

    async def generate_document(
        self,
        file_type: str,
        retrieved_context,
        source_manifest: list[dict],
    ) -> str: ...

    def rag_config(
        self,
        *,
        tenant_id: str = "legacy",
        case_id: str = "legacy",
    ) -> RAGConfig: ...

    def generate(
        self,
        file_manifests: list[dict],
        document_type: str,
        *,
        corrections: list[str] | None = None,
        tenant_id: str = "legacy",
        case_id: str = "legacy",
        submitted_parties: dict | None = None,
    ) -> list[str]: ...


@dataclass(frozen=True)
class ConfiguredVisaAdapter:
    visa_type: str
    display_name: str
    cache_namespace: str
    pipeline_version: str
    supported_document_types: frozenset[str]
    retrieval_query_builder: Callable[[str], str]
    document_generator: Callable[..., Awaitable[str]]
    template_provider: Callable[[str], str]

    def build_retrieval_query(self, document_type: str) -> str:
        return self.retrieval_query_builder(document_type)

    def document_template(self, document_type: str) -> str:
        return self.template_provider(document_type)

    async def generate_document(
        self,
        file_type: str,
        retrieved_context,
        source_manifest: list[dict],
    ) -> str:
        return await self.document_generator(
            file_type=file_type,
            retrieved_context=retrieved_context,
            source_manifest=source_manifest,
        )

    def rag_config(
        self,
        *,
        tenant_id: str = "legacy",
        case_id: str = "legacy",
    ) -> RAGConfig:
        return build_rag_config(
            visa_type=self.visa_type,
            cache_namespace=self.cache_namespace,
            pipeline_version=self.pipeline_version,
            tenant_id=tenant_id,
            case_id=case_id,
        )

    def generate(
        self,
        file_manifests: list[dict],
        document_type: str,
        *,
        corrections: list[str] | None = None,
        tenant_id: str = "legacy",
        case_id: str = "legacy",
        submitted_parties: dict | None = None,
    ) -> list[str]:
        if document_type not in self.supported_document_types:
            raise ValueError(
                f"Unsupported {self.display_name} document type: {document_type}"
            )
        from case_jobs.pipeline.legacy_generation import handle_doc_generation

        return handle_doc_generation(
            file_manifests,
            [document_type],
            self,
            tenant_id=tenant_id,
            case_id=case_id,
            submitted_parties=submitted_parties,
            corrections=corrections,
        )


_ADAPTERS: dict[str, VisaAdapter] = {}
_VISA_MODULES = (
    "aap",
    "aea",
    "ds_160",
    "ds_260",
    "eb_1aA",
    "eb_1aB",
    "naturalization",
    "reentry_permit",
)
_adapters_registered = False


def _ensure_adapters_registered() -> None:
    global _adapters_registered
    if _adapters_registered:
        return
    from importlib import import_module

    for module_name in _VISA_MODULES:
        import_module(f"{module_name}.adapter").register_adapter()
    _adapters_registered = True

def _normalize_visa_type(visa_type: str) -> str:
    # Accept both underscore and hyphen variants (e.g. "h_1b" and "h-1b"),
    # plus common compact spellings like "o1" / "l1" used in some clients.
    key = (visa_type or "").strip().lower().replace("_", "-")
    compact_aliases = {
        "aap": "aap",
        "aea": "aea",
        "ds-160": "ds-160",
        "ds-260": "ds-260",
        "eb-1a-a": "eb-1aa",
        "eb-1a-b": "eb-1ab",
        "naturalization": "n-400",
        "reentry-permit": "r-1",
    }
    return compact_aliases.get(key, key)


def register(adapter: VisaAdapter) -> None:
    key = _normalize_visa_type(adapter.visa_type)
    if not key:
        raise ValueError("visa_type cannot be empty")
    _ADAPTERS[key] = adapter


def get_adapter(visa_type: str) -> VisaAdapter:
    _ensure_adapters_registered()
    key = _normalize_visa_type(visa_type)
    try:
        return _ADAPTERS[key]
    except KeyError as exc:
        raise LookupError(f"Unsupported visa type: {visa_type}") from exc


def registered_visa_types() -> tuple[str, ...]:
    _ensure_adapters_registered()
    return tuple(sorted(_ADAPTERS))
