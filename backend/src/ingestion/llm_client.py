"""Anthropic real-time extraction client.

Stripped to the single-document real-time path — the Batch API is reserved
for the full ingestion pipeline. The demo uses this for interactive PDF
uploads where round-trip latency matters.

Key guarantees (same as production):
* Forced tool call — the model MUST invoke ``extract_headnote`` exactly once.
* Pydantic V2 validation of the tool input before returning.
* RPM + TPM rate limiting via a dual token-bucket.
* Tenacity exponential backoff on transient Anthropic errors.
* Prompt-cache hint on the system prompt (cuts per-call cost by ~90 %).
"""

from __future__ import annotations

from typing import Any

from anthropic import (
    APIConnectionError,
    APIStatusError,
    AsyncAnthropic,
    RateLimitError,
)
from pydantic import ValidationError

from src.core.config import AnthropicSettings, get_settings
from src.core.exceptions import (
    LLMError,
    LLMOverloadedError,
    LLMRateLimitError,
    LLMSchemaValidationError,
)
from src.core.logging import get_logger
from src.core.rate_limiter import RateLimiter
from src.core.retry import with_llm_retry
from src.ingestion.prompts import build_user_prompt, system_param
from src.schemas.headnote import LLMExtraction

log = get_logger(__name__)

_TOOL_NAME = "extract_headnote"

_RETRYABLE_HTTP_STATUSES = frozenset({408, 409, 500, 502, 503, 504, 529})


def _build_tool_schema() -> dict[str, Any]:
    """Derive the Anthropic tool schema from the LLMExtraction Pydantic model."""
    schema = LLMExtraction.model_json_schema()
    return {
        "name": _TOOL_NAME,
        "description": (
            "Persist the extracted headnote in the canonical schema. "
            "Always invoke this tool exactly once per judgment."
        ),
        "input_schema": schema,
    }


def _status_error_body(exc: APIStatusError) -> str:
    body = getattr(exc, "response", None)
    try:
        return body.text if body is not None else str(exc)
    except Exception:  # noqa: BLE001
        return str(exc)


class LLMClient:
    """Async client for real-time headnote extraction via Anthropic Messages API."""

    def __init__(
        self,
        settings: AnthropicSettings | None = None,
        client: AsyncAnthropic | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self._settings = settings or get_settings().anthropic
        self._client = client or AsyncAnthropic(
            api_key=self._settings.api_key.get_secret_value(),
            max_retries=0,  # tenacity handles retries
            timeout=self._settings.timeout_seconds,
        )
        self._tool_schema = _build_tool_schema()
        self._limiter = rate_limiter or RateLimiter(
            rpm=self._settings.rpm,
            tpm=self._settings.tpm,
            burst_factor=self._settings.burst_factor,
        )

    @property
    def rate_limiter(self) -> RateLimiter:
        return self._limiter

    @with_llm_retry()
    async def extract_one(
        self,
        *,
        custom_id: str,
        filename: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> LLMExtraction:
        """Extract a structured headnote from a single judgment text.

        ``custom_id`` is the document MD5 — used only for log correlation,
        never sent to Anthropic (no PII concern).
        """
        estimated = self._settings.estimated_tokens_per_call
        await self._limiter.acquire(estimated_tokens=estimated)

        try:
            response = await self._client.messages.create(
                model=self._settings.model,
                max_tokens=self._settings.max_tokens,
                system=system_param(),
                tools=[self._tool_schema],
                tool_choice={"type": "tool", "name": _TOOL_NAME},
                messages=[
                    {
                        "role": "user",
                        "content": build_user_prompt(filename, text, metadata=metadata),
                    },
                ],
            )
        except RateLimitError as exc:
            log.warning("llm.rate_limited", custom_id=custom_id, cause=str(exc))
            raise LLMRateLimitError(str(exc)) from exc
        except APIConnectionError as exc:
            log.warning("llm.connection_error", custom_id=custom_id, cause=str(exc))
            raise LLMOverloadedError(
                f"Could not reach Anthropic: {exc}",
                details={"custom_id": custom_id, "cause": str(exc)},
            ) from exc
        except APIStatusError as exc:
            body_text = _status_error_body(exc)
            if exc.status_code in _RETRYABLE_HTTP_STATUSES:
                log.warning(
                    "llm.api_overloaded",
                    status=exc.status_code,
                    custom_id=custom_id,
                    model=self._settings.model,
                )
                raise LLMOverloadedError(
                    f"Anthropic transiently unavailable ({exc.status_code}).",
                    details={"status": exc.status_code, "custom_id": custom_id},
                ) from exc
            log.error(
                "llm.api_status_error",
                status=exc.status_code,
                custom_id=custom_id,
                model=self._settings.model,
                body=body_text,
            )
            raise LLMError(
                f"Anthropic API error ({exc.status_code}): {body_text[:500]}",
                details={"status": exc.status_code, "body": body_text},
            ) from exc

        # Reconcile token spend with the rate limiter.
        usage = getattr(response, "usage", None)
        if usage is not None:
            cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
            cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
            actual = usage.input_tokens + usage.output_tokens + cache_read + cache_write
            log.info(
                "llm.extract_usage",
                custom_id=custom_id,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read_tokens=cache_read,
                cache_write_tokens=cache_write,
                cache_hit=cache_read > 0,
            )
        else:
            actual = estimated
        self._limiter.observe_tokens(actual, estimated_tokens=estimated)

        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and block.name == _TOOL_NAME:
                return self._validate_tool_input(custom_id, block.input)

        raise LLMSchemaValidationError(
            f"Anthropic response did not include the {_TOOL_NAME!r} tool call.",
            details={"custom_id": custom_id},
        )

    @staticmethod
    def _validate_tool_input(custom_id: str, tool_input: object) -> LLMExtraction:
        try:
            return LLMExtraction.model_validate(tool_input)
        except ValidationError as exc:
            raise LLMSchemaValidationError(
                "LLM produced a payload that did not match LLMExtraction.",
                details={"custom_id": custom_id, "errors": exc.errors()},
            ) from exc

    async def aclose(self) -> None:
        await self._client.close()
