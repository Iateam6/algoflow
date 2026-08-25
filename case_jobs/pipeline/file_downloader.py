from __future__ import annotations

import hashlib
import ipaddress
import mimetypes
import os
import re
import socket
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit
import logging

import requests
from django.conf import settings
from pypdf import PdfReader

from case_jobs.exceptions import DownloadError


logger = logging.getLogger(__name__)


SUPPORTED_TYPES = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "text/plain": ".txt",
}


def _is_public_address(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    return not any(
        (
            ip.is_private,
            ip.is_loopback,
            ip.is_link_local,
            ip.is_multicast,
            ip.is_reserved,
            ip.is_unspecified,
        )
    )


def assert_safe_https_url(url: str, allowed_domains: tuple[str, ...] = ()) -> None:
    parsed = urlsplit(url)
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise DownloadError("Source URL must use HTTPS")
    if parsed.username or parsed.password:
        raise DownloadError("Source URL cannot contain credentials")
    hostname = parsed.hostname.lower().rstrip(".")
    if allowed_domains and not any(
        hostname == allowed or hostname.endswith(f".{allowed}")
        for allowed in allowed_domains
    ):
        raise DownloadError("Source host is not in the configured allowlist")
    try:
        addresses = {
            str(result[4][0])
            for result in socket.getaddrinfo(hostname, parsed.port or 443)
        }
    except socket.gaierror as exc:
        raise DownloadError("Source host could not be resolved") from exc
    if not addresses or not all(_is_public_address(address) for address in addresses):
        raise DownloadError("Source URL resolves to a non-public address")


def _detected_content_type(path: str, header: str) -> str:
    with open(path, "rb") as handle:
        signature = handle.read(16)
    if signature.startswith(b"%PDF-"):
        return "application/pdf"
    if signature.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if signature.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if signature.startswith(b"PK") and zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            if "word/document.xml" in archive.namelist():
                return (
                    "application/vnd.openxmlformats-officedocument."
                    "wordprocessingml.document"
                )
    normalized = header.split(";", 1)[0].strip().lower()
    if normalized == "text/plain":
        return normalized
    try:
        import magic

        detected = magic.from_file(path, mime=True)
        if detected in SUPPORTED_TYPES:
            return detected
    except (ImportError, OSError):
        pass
    raise DownloadError("Downloaded file type is unsupported or does not match content")


def _page_count(path: str, content_type: str) -> int:
    if content_type == "application/pdf":
        try:
            return len(PdfReader(path).pages)
        except Exception as exc:
            raise DownloadError("Downloaded PDF is corrupted") from exc
    return 1


@dataclass
class DownloadBudget:
    total_bytes: int = 0
    total_pages: int = 0


def _safe_original_name(url: str, content_disposition: str | None) -> str:
    filename = ""
    if content_disposition:
        match = re.search(
            r"filename\s*=\s*(?:\"([^\"]+)\"|'([^']+)'|([^;]+))",
            content_disposition,
            re.IGNORECASE,
        )
        if match:
            filename = next(value for value in match.groups() if value).strip()
    if not filename:
        filename = os.path.basename(unquote(urlsplit(url).path))
    return os.path.basename(filename) or "uploaded_document"


def _preferred_original_name(preferred_name: str | None, extension: str) -> str | None:
    if not isinstance(preferred_name, str) or not preferred_name.strip():
        return None
    filename = os.path.basename(preferred_name.strip())
    stem, current_extension = os.path.splitext(filename)
    if current_extension:
        filename = f"{stem}{extension}"
    elif not current_extension:
        filename = f"{filename}{extension}"
    return filename or None


def download_source(
    url: str,
    destination_dir: str,
    source_index: int,
    budget: DownloadBudget,
    *,
    session=None,
    job_id: str | None = None,
    tenant_id: str | None = None,
    preferred_name: str | None = None,
) -> dict:
    session = session or requests.Session()
    current_url = url
    response = None
    for redirect_number in range(settings.FILE_DOWNLOAD_MAX_REDIRECTS + 1):
        assert_safe_https_url(current_url, settings.FILE_DOWNLOAD_ALLOWED_DOMAINS)
        try:
            response = session.get(
                current_url,
                stream=True,
                allow_redirects=False,
                timeout=(
                    settings.FILE_DOWNLOAD_CONNECT_TIMEOUT,
                    settings.FILE_DOWNLOAD_READ_TIMEOUT,
                ),
            )
        except requests.RequestException as exc:
            raise DownloadError("Could not download source file") from exc
        if response.status_code in {301, 302, 303, 307, 308}:
            location = response.headers.get("Location")
            response.close()
            if not location or redirect_number >= settings.FILE_DOWNLOAD_MAX_REDIRECTS:
                raise DownloadError("Source URL exceeded the redirect limit")
            current_url = urljoin(current_url, location)
            continue
        try:
            response.raise_for_status()
        except requests.RequestException as exc:
            response.close()
            raise DownloadError("Source host returned an unsuccessful response") from exc
        break
    if response is None:
        raise DownloadError("Could not download source file")

    declared_size = int(response.headers.get("Content-Length") or 0)
    if declared_size > settings.MAX_FILE_SIZE_BYTES:
        response.close()
        raise DownloadError("Source file exceeds the per-file size limit")

    temp_path = os.path.join(destination_dir, f"source_{source_index}.download")
    hasher = hashlib.sha256()
    file_size = 0
    try:
        with open(temp_path, "wb") as output:
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                file_size += len(chunk)
                if file_size > settings.MAX_FILE_SIZE_BYTES:
                    raise DownloadError("Source file exceeds the per-file size limit")
                if budget.total_bytes + file_size > settings.MAX_TOTAL_DOWNLOAD_BYTES:
                    raise DownloadError("Job exceeds the total download size limit")
                output.write(chunk)
                hasher.update(chunk)
    finally:
        response.close()
    if file_size == 0:
        raise DownloadError("Source file is empty")

    content_type = _detected_content_type(
        temp_path, response.headers.get("Content-Type", "")
    )
    extension = SUPPORTED_TYPES[content_type]
    final_path = os.path.join(destination_dir, f"source_{source_index}{extension}")
    os.replace(temp_path, final_path)
    pages = _page_count(final_path, content_type)
    if budget.total_pages + pages > settings.MAX_PAGES_PER_JOB:
        raise DownloadError("Job exceeds the total page limit")
    budget.total_bytes += file_size
    budget.total_pages += pages

    hostname = urlsplit(url).hostname or "unknown"
    logger.info(
        "source file download completion job_id=%s tenant_id=%s source_index=%d host=%s content_type=%s page_count=%s file_size=%s",
        job_id,
        tenant_id,
        source_index,
        hostname,
        content_type,
        pages,
        file_size,
    )

    original_filename = _preferred_original_name(
        preferred_name, extension
    ) or _safe_original_name(
        current_url, response.headers.get("Content-Disposition")
    )
    return {
        "name": f"source-{source_index}",
        "original_filename": original_filename,
        "url": url,
        "content_type": content_type,
        "extension": extension,
        "local_path": final_path,
        "file_hash": hasher.hexdigest(),
        "page_count": pages,
        "file_size": file_size,
    }


def download_sources(
    urls: list[str],
    destination_dir: str,
    *,
    job_id: str | None = None,
    tenant_id: str | None = None,
    preferred_names: list[str | None] | None = None,
) -> list[dict]:
    Path(destination_dir).mkdir(parents=True, exist_ok=True)
    budget = DownloadBudget()
    names = preferred_names if preferred_names is not None else [None] * len(urls)
    if len(names) != len(urls):
        raise ValueError("preferred_names must match urls")
    return [
        download_source(
            url,
            destination_dir,
            index,
            budget,
            job_id=job_id,
            tenant_id=tenant_id,
            preferred_name=names[index - 1],
        )
        for index, url in enumerate(urls, start=1)
    ]
