from case_jobs.registry import ConfiguredVisaAdapter, register

from .agent import build_retrieval_query, generate_document, get_document_template
from .config import CACHE_NAMESPACE, DISPLAY_NAME, PIPELINE_VERSION, SUPPORTED_DOCUMENT_TYPES, VISA_TYPE


ADAPTER = ConfiguredVisaAdapter(
    VISA_TYPE, DISPLAY_NAME, CACHE_NAMESPACE, PIPELINE_VERSION,
    SUPPORTED_DOCUMENT_TYPES, build_retrieval_query, generate_document,
    get_document_template,
)


def register_adapter() -> None:
    register(ADAPTER)
