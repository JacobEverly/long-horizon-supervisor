from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class Record:
    fingerprint: str
    state: Literal["pending", "completed"]
    response: Any
    expires_at: float


@dataclass(frozen=True)
class BeginResult:
    status: Literal["started", "in_progress", "replay", "conflict"]
    response: Any = None


class IdempotencyStore:
    def __init__(self, ttl_seconds: float) -> None:
        self.ttl_seconds = ttl_seconds
        self._records: dict[str, Record] = {}

    def begin(self, key: str, fingerprint: str, now: float) -> BeginResult:
        record = self._records.get(key)
        if record is None:
            self._records[key] = Record(
                fingerprint, "pending", None, now + self.ttl_seconds
            )
            return BeginResult("started")
        if record.fingerprint != fingerprint:
            return BeginResult("conflict")
        if record.response:
            return BeginResult("replay", record.response)
        return BeginResult("in_progress")

    def complete(self, key: str, fingerprint: str, response: Any, now: float) -> None:
        record = self._records[key]
        record.state = "completed"
        record.response = response
