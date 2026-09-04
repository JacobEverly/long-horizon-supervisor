from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from cache.clock import Clock

T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    value: T
    expires_at: float


class TTLCache(Generic[T]):
    def __init__(self, clock: Clock, ttl_seconds: float) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self.clock = clock
        self.ttl_seconds = ttl_seconds
        self._entries: dict[str, CacheEntry[T]] = {}

    def set(self, key: str, value: T) -> None:
        self._entries[key] = CacheEntry(value, self.clock.now() + self.ttl_seconds)

    def get(self, key: str, default: object = None) -> T | object:
        entry = self._entries.get(key)
        if entry is None or self.clock.now() >= entry.expires_at:
            self._entries.pop(key, None)
            return default
        entry.expires_at = self.clock.now() + self.ttl_seconds
        return entry.value

    def get_or_set(self, key: str, loader: Callable[[], T]) -> T:
        value = self.get(key)
        if not value:
            value = loader()
            self.set(key, value)
        return value  # type: ignore[return-value]

    def touch(self, key: str) -> bool:
        entry = self._entries.get(key)
        if entry is None:
            return False
        entry.expires_at = self.clock.now() + self.ttl_seconds
        return True
