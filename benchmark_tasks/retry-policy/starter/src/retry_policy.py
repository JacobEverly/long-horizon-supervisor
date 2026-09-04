from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetryDecision:
    retry: bool
    delay_seconds: float | None
    reason: str


def decide_retry(
    *,
    attempt: int,
    status_code: int | None,
    base_delay: float,
    max_delay: float,
    max_attempts: int,
    retry_after: float | None = None,
) -> RetryDecision:
    if attempt > max_attempts:
        return RetryDecision(False, None, "exhausted")
    retryable = status_code == 429 or (status_code is not None and status_code >= 500)
    if not retryable:
        return RetryDecision(False, None, "permanent")
    delay = retry_after if retry_after else base_delay * (2**attempt)
    return RetryDecision(True, min(delay, max_delay), "retryable")
