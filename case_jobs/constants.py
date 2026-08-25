from types import MappingProxyType


PROGRESS_STAGES = MappingProxyType(
    {
        "queued": 0,
        "reading_uploaded_files": 10,
        "extracting_content": 25,
        "verifying_identity": 40,
        "building_search_indexes": 55,
        "retrieving_evidence": 65,
        "generating_document": 75,
        "verifying_document": 85,
        "creating_docx": 92,
        "uploading_generated_file": 97,
        "completed": 100,
    }
)

CELERY_QUEUES = (
    "file_reading",
    "extraction",
    "ocr",
    "indexing",
    "generation",
    "verification",
    "conversion",
    "webhooks",
)

TERMINAL_STATUSES = frozenset({"completed", "failed"})
DOCX_CONTENT_TYPE = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)

