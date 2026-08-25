import io
import os
import uuid
from unittest import skipUnless
from unittest.mock import Mock

from botocore.exceptions import ClientError
from django.test import SimpleTestCase
from docx import Document

from case_jobs.constants import DOCX_CONTENT_TYPE
from case_jobs.exceptions import ObjectAlreadyExists, ValidationError
from case_jobs.integrations.s3_client import S3StorageClient, S3StorageConfig


def docx_bytes() -> bytes:
    output = io.BytesIO()
    document = Document()
    document.add_paragraph("Completed generated document")
    document.save(output)
    return output.getvalue()


class S3StorageTests(SimpleTestCase):
    def setUp(self):
        self.client = Mock()
        self.client.put_object.return_value = {"ETag": '"etag-1"'}
        self.content = docx_bytes()
        self.client.head_object.return_value = {
            "ContentLength": len(self.content),
            "ServerSideEncryption": "AES256",
            "ETag": '"etag-1"',
        }
        self.storage = S3StorageClient(
            S3StorageConfig("generated-test", "us-east-1", 900),
            client=self.client,
        )

    def test_upload_is_private_encrypted_and_non_overwriting(self):
        stored = self.storage.upload_generated_document(
            self.content,
            tenant_id="tenant-1",
            case_id="case-1",
            job_id="job-1",
        )
        kwargs = self.client.put_object.call_args.kwargs
        self.assertEqual(kwargs["ServerSideEncryption"], "AES256")
        self.assertEqual(kwargs["IfNoneMatch"], "*")
        self.assertEqual(kwargs["ContentType"], DOCX_CONTENT_TYPE)
        self.assertNotIn("ACL", kwargs)
        self.assertEqual(
            stored.key,
            "generated/tenant-1/case-1/job-1/Support_Letter.docx",
        )

    def test_existing_object_is_not_overwritten(self):
        self.client.put_object.side_effect = ClientError(
            {
                "Error": {"Code": "PreconditionFailed", "Message": "exists"},
                "ResponseMetadata": {"HTTPStatusCode": 412},
            },
            "PutObject",
        )
        with self.assertRaises(ObjectAlreadyExists):
            self.storage.upload_generated_document(
                self.content,
                tenant_id="tenant-1",
                case_id="case-1",
                job_id="job-1",
            )

    def test_rejects_path_injection_and_empty_documents(self):
        with self.assertRaises(ValidationError):
            self.storage.upload_generated_document(
                self.content,
                tenant_id="../tenant",
                case_id="case-1",
                job_id="job-1",
            )
        with self.assertRaises(ValidationError):
            self.storage.upload_generated_document(
                b"", tenant_id="tenant", case_id="case", job_id="job"
            )

    def test_presigned_url_uses_short_expiry(self):
        self.client.generate_presigned_url.return_value = "https://signed.example.com"
        self.storage.create_download_url("generated/t/c/j/file.docx")
        self.assertEqual(
            self.client.generate_presigned_url.call_args.kwargs["ExpiresIn"], 900
        )


@skipUnless(
    os.getenv("RUN_S3_LIVE_TESTS") == "1" and os.getenv("AWS_S3_TEST_BUCKET_NAME"),
    "live S3 test is opt-in",
)
class LiveS3StorageTests(SimpleTestCase):
    def test_upload_verify_sign_and_cleanup(self):
        bucket = os.environ["AWS_S3_TEST_BUCKET_NAME"]
        job_id = str(uuid.uuid4())
        storage = S3StorageClient(
            S3StorageConfig(bucket, os.getenv("AWS_S3_REGION_NAME", "us-east-1"), 60)
        )
        stored = storage.upload_generated_document(
            docx_bytes(),
            tenant_id="codex-test",
            case_id="storage-smoke",
            job_id=job_id,
        )
        try:
            self.assertTrue(storage.create_download_url(stored.key))
        finally:
            storage.client.delete_object(Bucket=bucket, Key=stored.key)

