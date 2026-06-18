"""Domain-specific exception hierarchy.

All custom exceptions inherit from `LawLensError` so callers can catch the
whole family with one clause. Each carries an HTTP status hint for the API
layer to map onto responses.
"""

from __future__ import annotations

from http import HTTPStatus


class LawLensError(Exception):
    """Base class for all application errors."""

    status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR
    error_code: str = "law_lens_error"

    def __init__(self, message: str, *, details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


# --- Ingestion -----------------------------------------------------------


class IngestionError(LawLensError):
    status_code = HTTPStatus.UNPROCESSABLE_ENTITY
    error_code = "ingestion_error"


class PDFParseError(IngestionError):
    error_code = "pdf_parse_error"


class OCRFailureError(IngestionError):
    error_code = "ocr_failure"


class PayloadTooLargeError(IngestionError):
    status_code = HTTPStatus.REQUEST_ENTITY_TOO_LARGE
    error_code = "payload_too_large"


# --- LLM -----------------------------------------------------------------


class LLMError(LawLensError):
    status_code = HTTPStatus.BAD_GATEWAY
    error_code = "llm_error"


class LLMRateLimitError(LLMError):
    status_code = HTTPStatus.TOO_MANY_REQUESTS
    error_code = "llm_rate_limit"


class LLMOverloadedError(LLMError):
    """Anthropic is transiently unavailable — HTTP 5xx / 529 / network failure.

    Always retried via `with_llm_retry`; never fails a document permanently.
    """

    status_code = HTTPStatus.SERVICE_UNAVAILABLE
    error_code = "llm_overloaded"


class LLMSchemaValidationError(LLMError):
    status_code = HTTPStatus.UNPROCESSABLE_ENTITY
    error_code = "llm_schema_invalid"


# --- API -----------------------------------------------------------------


class UnauthorizedError(LawLensError):
    status_code = HTTPStatus.UNAUTHORIZED
    error_code = "unauthorized"


class NotFoundError(LawLensError):
    status_code = HTTPStatus.NOT_FOUND
    error_code = "not_found"


class ValidationError(LawLensError):
    status_code = HTTPStatus.UNPROCESSABLE_ENTITY
    error_code = "validation_error"
