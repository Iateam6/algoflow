from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import BinaryIO

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from django.conf import settings

from case_jobs.constants import DOCX_CONTENT_TYPE
from case_jobs.exceptions import (
    ObjectAlreadyExists,
    StorageConfigurationError,
    StorageError,
    ValidationError,
)


SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,255}$")


@dataclass(frozen=True)
class S3StorageConfig:
    bucket_name: str
    region_name: str = "us-east-1"
    presigned_url_expiry_seconds: int = 900

    @classmethod
    def from_settings(cls) -> "S3StorageConfig":
        return cls(
            bucket_name=settings.AWS_S3_BUCKET_NAME,
            region_name=settings.AWS_S3_REGION_NAME,
            presigned_url_expiry_seconds=(
                settings.AWS_S3_PRESIGNED_URL_EXPIRY_SECONDS
            ),
        )

    def validate(self) -> None:
        if not self.bucket_name:
            raise StorageConfigurationError("AWS_S3_BUCKET_NAME is not configured")
        if not 1 <= self.presigned_url_expiry_seconds <= 604800:
            raise StorageConfigurationError(
                "S3 signed URL expiry must be between 1 and 604800 seconds"
            )


@dataclass(frozen=True)
class StoredDocument:
    bucket: str
    key: str
    etag: str
    size: int


def _safe_segment(value: str, field: str) -> str:
    normalized = str(value).strip()
    if not SAFE_SEGMENT.fullmatch(normalized):
        raise ValidationError(f"{field} contains unsupported characters")
    return normalized


def build_generated_document_key(
    tenant_id: str,
    case_id: str,
    job_id: str,
    filename: str = "Support_Letter.docx",
) -> str:
    safe_filename = _safe_segment(filename, "filename")
    if not safe_filename.lower().endswith(".docx"):
        raise ValidationError("generated output must use a .docx filename")
    return "/".join(
        (
            "generated",
            _safe_segment(tenant_id, "tenant_id"),
            _safe_segment(case_id, "case_id"),
            _safe_segment(job_id, "job_id"),
            safe_filename,
        )
    )


class S3StorageClient:
    def __init__(self, config: S3StorageConfig | None = None, client=None):
        self.config = config or S3StorageConfig.from_settings()
        self.config.validate()
        self.client = client or boto3.client(
            "s3", region_name=self.config.region_name
        )

    @staticmethod
    def _read_document(document: bytes | BinaryIO) -> bytes:
        if isinstance(document, bytes):
            content = document
        elif isinstance(document, io.BufferedIOBase) or hasattr(document, "read"):
            content = document.read()
        else:
            raise ValidationError("document must be bytes or a binary stream")
        if not isinstance(content, bytes) or not content:
            raise ValidationError("completed DOCX cannot be empty")
        if not content.startswith(b"PK"):
            raise ValidationError("generated output is not a valid DOCX container")
        return content

    def upload_generated_document(
        self,
        document: bytes | BinaryIO,
        *,
        tenant_id: str,
        case_id: str,
        job_id: str,
        filename: str = "Support_Letter.docx",
    ) -> StoredDocument:
        content = self._read_document(document)
        key = build_generated_document_key(
            tenant_id, case_id, job_id, filename
        )
        try:
            response = self.client.put_object(
                Bucket=self.config.bucket_name,
                Key=key,
                Body=content,
                ContentLength=len(content),
                ContentType=DOCX_CONTENT_TYPE,
                ServerSideEncryption="AES256",
                IfNoneMatch="*",
            )
            head = self.client.head_object(
                Bucket=self.config.bucket_name,
                Key=key,
            )
        except ClientError as exc:
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            code = exc.response.get("Error", {}).get("Code")
            if status == 412 or code in {"PreconditionFailed", "412"}:
                raise ObjectAlreadyExists(
                    "A generated document already exists for this job"
                ) from exc
            raise StorageError(f"S3 upload failed: {code or 'unknown error'}") from exc
        except BotoCoreError as exc:
            raise StorageError(f"S3 upload failed: {exc}") from exc

        if head.get("ContentLength") != len(content):
            raise StorageError("S3 upload verification returned an unexpected size")
        if head.get("ServerSideEncryption") != "AES256":
            raise StorageError("S3 upload verification did not confirm AES256 encryption")
        return StoredDocument(
            bucket=self.config.bucket_name,
            key=key,
            etag=str(response.get("ETag", head.get("ETag", ""))).strip('"'),
            size=len(content),
        )

    def create_download_url(self, key: str, *, expires_in: int | None = None) -> str:
        expiry = expires_in or self.config.presigned_url_expiry_seconds
        if not 1 <= expiry <= 604800:
            raise ValidationError("signed URL expiry must be between 1 and 604800")
        try:
            return self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.config.bucket_name, "Key": key},
                ExpiresIn=expiry,
            )
        except (BotoCoreError, ClientError) as exc:
            raise StorageError(f"Could not create S3 download URL: {exc}") from exc

