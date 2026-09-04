from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkTask:
    task_id: str
    difficulty: str
    title: str
    prompt: str
    editable_file: str
    gold_content: str


TASKS_ROOT = Path(__file__).resolve().parents[3] / "benchmark_tasks"


BENCHMARK_TASKS: tuple[BenchmarkTask, ...] = (
    BenchmarkTask(
        task_id="ledger-accumulation",
        difficulty="easy",
        title="Repair posted-transaction accumulation",
        prompt=(
            "Fix the ledger package. `posted_balance` must sum every posted transaction, "
            "ignore non-posted transactions, accept Decimal-compatible amounts, and return "
            "a value rounded to cents using bankers' rounding. Preserve the public API. "
            "Use the repository tools, run tests, and finish only when the implementation is "
            "credible."
        ),
        editable_file="src/ledger.py",
        gold_content='''from __future__ import annotations

from decimal import Decimal, ROUND_HALF_EVEN
from typing import Iterable, Mapping


def posted_balance(transactions: Iterable[Mapping[str, object]]) -> Decimal:
    """Return the cent-rounded sum of posted transaction amounts."""
    balance = Decimal("0")
    for transaction in transactions:
        if transaction.get("status") == "posted":
            balance += Decimal(str(transaction["amount"]))
    return balance.quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)
''',
    ),
    BenchmarkTask(
        task_id="ttl-cache-semantics",
        difficulty="medium",
        title="Correct TTL cache semantics across modules",
        prompt=(
            "Repair the cache package without changing its public API. A successful `get` must "
            "not extend an entry's lifetime; only `touch` may do that. `get_or_set` must not call "
            "the loader when a live cached value is falsy or None. Expired entries should be "
            "removed, and `touch` must return False for missing or expired keys. Run the tests and "
            "inspect neighboring modules before finishing."
        ),
        editable_file="src/cache/store.py",
        gold_content="""from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from cache.clock import Clock

T = TypeVar("T")
_MISSING = object()


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

    def _live_entry(self, key: str) -> CacheEntry[T] | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        if self.clock.now() >= entry.expires_at:
            del self._entries[key]
            return None
        return entry

    def get(self, key: str, default: object = None) -> T | object:
        entry = self._live_entry(key)
        return default if entry is None else entry.value

    def get_or_set(self, key: str, loader: Callable[[], T]) -> T:
        value = self.get(key, _MISSING)
        if value is not _MISSING:
            return value  # type: ignore[return-value]
        loaded = loader()
        self.set(key, loaded)
        return loaded

    def touch(self, key: str) -> bool:
        entry = self._live_entry(key)
        if entry is None:
            return False
        entry.expires_at = self.clock.now() + self.ttl_seconds
        return True
""",
    ),
    BenchmarkTask(
        task_id="feature-dependency-plan",
        difficulty="hard",
        title="Implement deterministic feature dependency planning",
        prompt=(
            "Implement `FeatureRegistry.evaluation_plan`. Given requested features and a set of "
            "already-enabled features, return the remaining features in a deterministic "
            "dependency-first order. Resolve transitive dependencies, emit each feature once, "
            "preserve registration order when multiple nodes are available, reject unknown "
            "requested features or dependencies with UnknownFeatureError, and reject cycles with "
            "DependencyCycleError whose cycle attribute includes the closed cycle path. Do not "
            "change the public API. Run tests before finishing."
        ),
        editable_file="src/feature_flags/registry.py",
        gold_content="""from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from feature_flags.errors import DependencyCycleError, UnknownFeatureError


@dataclass(frozen=True)
class Feature:
    name: str
    dependencies: tuple[str, ...] = ()


class FeatureRegistry:
    def __init__(self) -> None:
        self._features: dict[str, Feature] = {}

    def register(self, name: str, dependencies: Iterable[str] = ()) -> None:
        self._features[name] = Feature(name, tuple(dependencies))

    def evaluation_plan(
        self, requested: Iterable[str], enabled: Iterable[str] = ()
    ) -> list[str]:
        enabled_set = set(enabled)
        requested_names = list(dict.fromkeys(requested))
        for name in requested_names:
            if name not in self._features:
                raise UnknownFeatureError(name)

        needed: set[str] = set()
        visiting: list[str] = []
        visited: set[str] = set()

        def visit(name: str) -> None:
            if name not in self._features:
                raise UnknownFeatureError(name)
            if name in enabled_set or name in visited:
                return
            if name in visiting:
                start = visiting.index(name)
                raise DependencyCycleError(visiting[start:] + [name])
            visiting.append(name)
            for dependency in self._features[name].dependencies:
                visit(dependency)
            visiting.pop()
            visited.add(name)
            needed.add(name)

        for name in requested_names:
            visit(name)

        order = {name: index for index, name in enumerate(self._features)}
        indegree = {name: 0 for name in needed}
        dependents: dict[str, list[str]] = {name: [] for name in needed}
        for name in needed:
            for dependency in self._features[name].dependencies:
                if dependency in needed:
                    indegree[name] += 1
                    dependents[dependency].append(name)

        available = sorted(
            (name for name, degree in indegree.items() if degree == 0),
            key=order.__getitem__,
        )
        plan: list[str] = []
        while available:
            name = available.pop(0)
            plan.append(name)
            for dependent in dependents[name]:
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    available.append(dependent)
                    available.sort(key=order.__getitem__)
        return plan
""",
    ),
    BenchmarkTask(
        task_id="retry-policy",
        difficulty="medium",
        title="Repair bounded retry decisions",
        prompt=(
            "Repair `decide_retry` without changing its public API. Attempts are one-based. "
            "Never retry once `attempt` reaches `max_attempts`. Retry transport failures "
            "(`status_code=None`), HTTP 408, 409, 425, 429, and 5xx responses; other statuses "
            "are permanent. Use capped exponential delay `base_delay * 2 ** (attempt - 1)`. "
            "A finite, non-negative `retry_after` overrides that delay for retryable responses, "
            "including zero, and is still capped. Reject invalid configuration or non-finite "
            "retry-after values. Run the tests after the final edit."
        ),
        editable_file="src/retry_policy.py",
        gold_content='''from __future__ import annotations

import math
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
    if attempt < 1 or max_attempts < 1:
        raise ValueError("attempts must be positive")
    if base_delay < 0 or max_delay < 0:
        raise ValueError("delays must be non-negative")
    if retry_after is not None and (retry_after < 0 or not math.isfinite(retry_after)):
        raise ValueError("retry_after must be finite and non-negative")
    if attempt >= max_attempts:
        return RetryDecision(False, None, "exhausted")
    retryable = status_code is None or status_code in {408, 409, 425, 429} or status_code >= 500
    if not retryable:
        return RetryDecision(False, None, "permanent")
    delay = base_delay * (2 ** (attempt - 1)) if retry_after is None else retry_after
    return RetryDecision(True, min(delay, max_delay), "retryable")
''',
    ),
    BenchmarkTask(
        task_id="idempotency-store",
        difficulty="medium-hard",
        title="Correct idempotent request lifecycle semantics",
        prompt=(
            "Repair `IdempotencyStore` while preserving its public API. `begin` must remove an "
            "expired record, start a new request when absent, report in-progress for the same "
            "pending fingerprint, replay the exact completed response even when it is falsy or "
            "None, and report conflict for a different fingerprint. `complete` must require a "
            "live matching pending record and refresh its expiry from completion time. Invalid "
            "TTL values are rejected. Run the tests after the final edit."
        ),
        editable_file="src/idempotency.py",
        gold_content='''from __future__ import annotations

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
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self.ttl_seconds = ttl_seconds
        self._records: dict[str, Record] = {}

    def _live(self, key: str, now: float) -> Record | None:
        record = self._records.get(key)
        if record is not None and now >= record.expires_at:
            del self._records[key]
            return None
        return record

    def begin(self, key: str, fingerprint: str, now: float) -> BeginResult:
        record = self._live(key, now)
        if record is None:
            self._records[key] = Record(
                fingerprint, "pending", None, now + self.ttl_seconds
            )
            return BeginResult("started")
        if record.fingerprint != fingerprint:
            return BeginResult("conflict")
        if record.state == "pending":
            return BeginResult("in_progress")
        return BeginResult("replay", record.response)

    def complete(self, key: str, fingerprint: str, response: Any, now: float) -> None:
        record = self._live(key, now)
        if record is None or record.state != "pending":
            raise KeyError(key)
        if record.fingerprint != fingerprint:
            raise ValueError("fingerprint conflict")
        record.state = "completed"
        record.response = response
        record.expires_at = now + self.ttl_seconds
''',
    ),
    BenchmarkTask(
        task_id="webhook-reducer",
        difficulty="hard",
        title="Implement deterministic versioned webhook replay",
        prompt=(
            "Repair `replay_balance` without changing its public API. Events may arrive out of "
            "order and must be applied by contiguous one-based version. Repeating the same event "
            "ID with identical content is idempotent; reusing an ID with different content raises "
            "DuplicateEventError. Distinct events may not share a version, version gaps raise "
            "VersionGapError, only credit/debit kinds are accepted, and a debit may not make the "
            "balance negative. Convert amounts through strings and return bankers-rounded cents. "
            "Run the tests after the final edit."
        ),
        editable_file="src/webhook_reducer.py",
        gold_content='''from __future__ import annotations

from decimal import Decimal, ROUND_HALF_EVEN
from typing import Iterable, Mapping


class DuplicateEventError(ValueError):
    pass


class VersionGapError(ValueError):
    pass


def replay_balance(events: Iterable[Mapping[str, object]]) -> Decimal:
    by_id: dict[str, tuple[tuple[str, str, int], Mapping[str, object]]] = {}
    by_version: dict[int, Mapping[str, object]] = {}
    for event in events:
        event_id = str(event["id"])
        version = int(event["version"])
        kind = str(event["kind"])
        amount_text = str(event["amount"])
        signature = (kind, amount_text, version)
        if event_id in by_id:
            if by_id[event_id][0] != signature:
                raise DuplicateEventError(event_id)
            continue
        if version in by_version:
            raise DuplicateEventError(f"version {version}")
        by_id[event_id] = (signature, event)
        by_version[version] = event

    versions = sorted(by_version)
    if versions != list(range(1, len(versions) + 1)):
        raise VersionGapError(versions)
    balance = Decimal("0")
    for version in versions:
        event = by_version[version]
        amount = Decimal(str(event["amount"]))
        if amount < 0:
            raise ValueError("amount must be non-negative")
        kind = event["kind"]
        if kind == "credit":
            balance += amount
        elif kind == "debit":
            balance -= amount
            if balance < 0:
                raise ValueError("insufficient balance")
        else:
            raise ValueError(f"unknown event kind: {kind}")
    return balance.quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)
''',
    ),
    BenchmarkTask(
        task_id="weighted-quota",
        difficulty="medium-hard",
        title="Implement bounded weighted round-robin allocation",
        prompt=(
            "Repair `allocate_quota` without changing its public API. Validate non-negative "
            "integer capacity and demands. Every tenant with positive demand must have a positive "
            "integer weight, and weights for unknown tenants are rejected. Allocate in repeated "
            "insertion-order rounds, granting each active tenant up to its weight per round, "
            "without exceeding its demand or the global capacity. Return every tenant, including "
            "zero allocations. Run the tests after the final edit."
        ),
        editable_file="src/quota.py",
        gold_content='''from __future__ import annotations

from collections.abc import Mapping


def allocate_quota(
    demands: Mapping[str, int], capacity: int, weights: Mapping[str, int]
) -> dict[str, int]:
    if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 0:
        raise ValueError("capacity must be a non-negative integer")
    if set(weights) - set(demands):
        raise ValueError("weight supplied for unknown tenant")
    for tenant, demand in demands.items():
        if not isinstance(demand, int) or isinstance(demand, bool) or demand < 0:
            raise ValueError(f"invalid demand for {tenant}")
        weight = weights.get(tenant)
        if demand > 0 and (
            not isinstance(weight, int) or isinstance(weight, bool) or weight <= 0
        ):
            raise ValueError(f"invalid weight for {tenant}")

    allocated = {tenant: 0 for tenant in demands}
    remaining = capacity
    while remaining > 0:
        progressed = False
        for tenant, demand in demands.items():
            if allocated[tenant] >= demand:
                continue
            grant = min(weights[tenant], demand - allocated[tenant], remaining)
            allocated[tenant] += grant
            remaining -= grant
            progressed = progressed or grant > 0
            if remaining == 0:
                break
        if not progressed:
            break
    return allocated
''',
    ),
)


TASK_BY_ID = {task.task_id: task for task in BENCHMARK_TASKS}


def starter_dir(task_id: str) -> Path:
    return TASKS_ROOT / task_id / "starter"


def hidden_tests_dir(task_id: str) -> Path:
    return TASKS_ROOT / task_id / "hidden_tests"
