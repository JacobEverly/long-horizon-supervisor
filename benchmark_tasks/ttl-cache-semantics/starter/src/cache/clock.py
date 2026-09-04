from __future__ import annotations

from typing import Protocol


class Clock(Protocol):
    def now(self) -> float: ...


class ManualClock:
    def __init__(self, initial: float = 0.0) -> None:
        self.current = initial

    def now(self) -> float:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current += seconds
