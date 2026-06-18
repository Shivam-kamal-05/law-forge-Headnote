"""Async token-bucket rate limiter for Anthropic API calls.

Enforces two independent ceilings:
* RPM — requests per minute.
* TPM — tokens per minute.

Both are rolling leaky-bucket implementations with continuous refill so no
background sweeper is needed.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass

from src.core.logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class _Bucket:
    capacity: float
    tokens: float
    refill_per_sec: float
    last_refill: float

    def _refill(self, now: float) -> None:
        elapsed = now - self.last_refill
        if elapsed > 0:
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_sec)
            self.last_refill = now

    def try_take(self, amount: float, now: float) -> tuple[bool, float]:
        """Return (success, seconds_until_available)."""
        self._refill(now)
        if amount > self.capacity:
            log.warning(
                "rate_limit.request_exceeds_bucket",
                amount=amount,
                capacity=self.capacity,
            )
            self.tokens = 0
            return True, 0.0
        if self.tokens >= amount:
            self.tokens -= amount
            return True, 0.0
        deficit = amount - self.tokens
        return False, deficit / self.refill_per_sec


class RateLimiter:
    """Combines RPM + TPM token buckets behind a single ``acquire()`` call."""

    def __init__(
        self,
        *,
        rpm: int | None,
        tpm: int | None = None,
        burst_factor: float = 1.0,
    ) -> None:
        now = time.monotonic()
        self._rpm_bucket: _Bucket | None = None
        self._tpm_bucket: _Bucket | None = None
        if rpm is not None and rpm > 0:
            cap = float(rpm) * burst_factor
            self._rpm_bucket = _Bucket(
                capacity=cap, tokens=cap, refill_per_sec=rpm / 60.0, last_refill=now
            )
        if tpm is not None and tpm > 0:
            cap = float(tpm) * burst_factor
            self._tpm_bucket = _Bucket(
                capacity=cap, tokens=cap, refill_per_sec=tpm / 60.0, last_refill=now
            )
        self._lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        return self._rpm_bucket is not None or self._tpm_bucket is not None

    async def acquire(self, *, estimated_tokens: int = 0) -> None:
        """Block until both buckets allow the call."""
        if not self.enabled:
            return

        while True:
            async with self._lock:
                now = time.monotonic()
                wait_secs = 0.0
                wait_rpm = 0.0

                if self._rpm_bucket is not None:
                    ok_rpm, wait_rpm = self._rpm_bucket.try_take(1, now)
                    if not ok_rpm:
                        wait_secs = max(wait_secs, wait_rpm)

                if self._tpm_bucket is not None and estimated_tokens > 0:
                    ok_tpm, wait_tpm = self._tpm_bucket.try_take(estimated_tokens, now)
                    if not ok_tpm:
                        wait_secs = max(wait_secs, wait_tpm)
                        if self._rpm_bucket is not None and wait_rpm == 0.0:
                            self._rpm_bucket.tokens = min(
                                self._rpm_bucket.capacity, self._rpm_bucket.tokens + 1
                            )

                if wait_secs == 0.0:
                    return

            await asyncio.sleep(min(wait_secs, 30.0) + 0.01)

    @asynccontextmanager
    async def reserve(self, *, estimated_tokens: int = 0):
        await self.acquire(estimated_tokens=estimated_tokens)
        try:
            yield self
        finally:
            pass

    def observe_tokens(self, actual_tokens: int, *, estimated_tokens: int = 0) -> None:
        """Reconcile actual token spend against an earlier estimate."""
        if self._tpm_bucket is None or actual_tokens == estimated_tokens:
            return
        delta = actual_tokens - estimated_tokens
        b = self._tpm_bucket
        b.tokens = max(0.0, min(b.capacity, b.tokens - delta))

    def snapshot(self) -> dict[str, float | None]:
        return {
            "rpm_remaining": self._rpm_bucket.tokens if self._rpm_bucket else None,
            "rpm_capacity": self._rpm_bucket.capacity if self._rpm_bucket else None,
            "tpm_remaining": self._tpm_bucket.tokens if self._tpm_bucket else None,
            "tpm_capacity": self._tpm_bucket.capacity if self._tpm_bucket else None,
        }
