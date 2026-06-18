"""Tenacity-backed retry helpers for Anthropic API calls.

Transient errors (rate limits, 5xx, network timeouts) get exponential
backoff; schema / auth errors propagate immediately.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

import httpx
from anthropic import APIConnectionError as AnthropicConnectionError
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from src.core.exceptions import LLMOverloadedError, LLMRateLimitError
from src.core.logging import get_logger

log = get_logger(__name__)

T = TypeVar("T")


def _log_before_sleep(retry_state: RetryCallState) -> None:
    """structlog-friendly before_sleep callback — avoids the str-level bug
    in tenacity's built-in ``before_sleep_log``."""
    outcome = retry_state.outcome
    if outcome is None or not outcome.failed:
        return
    exc = outcome.exception()
    fn = retry_state.fn
    fn_name = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", "?")
    next_action = retry_state.next_action
    sleep_for = round(next_action.sleep, 3) if next_action is not None else None
    log.warning(
        "retry.scheduled",
        function=fn_name,
        attempt=retry_state.attempt_number,
        exc_type=type(exc).__name__ if exc is not None else None,
        exc_message=str(exc) if exc is not None else None,
        sleep_seconds=sleep_for,
    )


_LLM_TRANSIENT = (
    LLMRateLimitError,
    LLMOverloadedError,
    AnthropicConnectionError,
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
)


def with_llm_retry(
    max_attempts: int = 5,
    initial_wait: float = 1.0,
    max_wait: float = 30.0,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Retry decorator for Anthropic API calls with random exponential backoff."""
    return retry(
        retry=retry_if_exception_type(_LLM_TRANSIENT),
        stop=stop_after_attempt(max_attempts),
        wait=wait_random_exponential(multiplier=initial_wait, max=max_wait),
        before_sleep=_log_before_sleep,
        reraise=True,
    )


async def retry_async(
    func: Callable[..., Any],
    *args: Any,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    max_attempts: int = 3,
    initial_wait: float = 0.5,
    max_wait: float = 10.0,
    **kwargs: Any,
) -> Any:
    """Inline async retry — use when a decorator isn't ergonomic."""
    try:
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(exceptions),
            stop=stop_after_attempt(max_attempts),
            wait=wait_random_exponential(multiplier=initial_wait, max=max_wait),
            before_sleep=_log_before_sleep,
            reraise=True,
        ):
            with attempt:
                return await func(*args, **kwargs)
    except RetryError as exc:
        raise exc.last_attempt.exception()  # type: ignore[misc]
    return None
