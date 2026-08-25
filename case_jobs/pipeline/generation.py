import os
import re
import shutil

from django.conf import settings

from case_jobs.jobs.locks import distributed_lock
from case_jobs.registry import get_adapter


SAFE_PATH_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _safe_path_segment(value: str) -> str:
    value = str(value).strip()
    if not SAFE_PATH_SEGMENT.fullmatch(value):
        raise ValueError("Unsafe local output scope")
    return value


def generate_document(
    visa_type: str,
    file_manifests: list[dict],
    document_type: str,
    *,
    tenant_id: str,
    case_id: str,
    job_id: str,
    corrections: list[str] | None = None,
    submitted_parties: dict | None = None,
) -> str:
    adapter = get_adapter(visa_type)
    # The current converter uses a shared top-level filename and performs
    # legacy cleanup. Serialize that small compatibility boundary, then copy
    # the completed result into an isolated local directory before releasing.
    with distributed_lock("legacy-local-document-generation", timeout=3600):
        paths = adapter.generate(
            file_manifests,
            document_type,
            corrections=corrections,
            tenant_id=tenant_id,
            case_id=case_id,
            submitted_parties=submitted_parties,
        )
        if len(paths) != 1:
            raise RuntimeError("Document generator did not return exactly one output")
        destination_dir = os.path.join(
            str(settings.MEDIA_ROOT),
            "generated",
            _safe_path_segment(tenant_id),
            _safe_path_segment(case_id),
            _safe_path_segment(job_id),
        )
        os.makedirs(destination_dir, exist_ok=True)
        destination = os.path.join(destination_dir, os.path.basename(paths[0]))
        shutil.copy2(paths[0], destination)
        return destination
