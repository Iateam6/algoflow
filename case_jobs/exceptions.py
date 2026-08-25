class CaseJobError(Exception):
    """Base exception carrying a stable API-safe error code."""

    code = "GENERATION_ERROR"
    status_code = 400

    def __init__(self, message: str | None = None):
        super().__init__(message or self.code)


class AuthenticationError(CaseJobError):
    code = "AUTHENTICATION_FAILED"
    status_code = 401


class ValidationError(CaseJobError):
    code = "VALIDATION_ERROR"
    status_code = 400


class IdempotencyConflict(CaseJobError):
    code = "IDEMPOTENCY_KEY_REUSED"
    status_code = 409


class CapacityExceeded(CaseJobError):
    code = "CAPACITY_EXCEEDED"
    status_code = 429


class ServiceConfigurationError(CaseJobError):
    code = "SERVICE_NOT_CONFIGURED"
    status_code = 503


class DownloadError(CaseJobError):
    code = "FILE_DOWNLOAD_FAILED"


class IdentityError(CaseJobError):
    pass


class BeneficiaryMismatch(IdentityError):
    code = "BENEFICIARY_MISMATCH"


class BeneficiaryNotFound(IdentityError):
    code = "BENEFICIARY_NOT_FOUND"


class PetitionerMismatch(IdentityError):
    code = "PETITIONER_MISMATCH"


class IdentityAmbiguous(IdentityError):
    code = "IDENTITY_AMBIGUOUS"


class StorageError(CaseJobError):
    code = "STORAGE_ERROR"
    status_code = 500


class StorageConfigurationError(StorageError):
    code = "STORAGE_NOT_CONFIGURED"


class ObjectAlreadyExists(StorageError):
    code = "STORAGE_OBJECT_EXISTS"
    status_code = 409


class WebhookDeliveryError(CaseJobError):
    code = "WEBHOOK_DELIVERY_FAILED"

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        retry_after: int | None = None,
        status_code: int | None = None,
    ):
        super().__init__(message)
        self.retryable = retryable
        self.retry_after = retry_after
        self.response_status_code = status_code
