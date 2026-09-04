from __future__ import annotations

from decimal import Decimal
from typing import Iterable, Mapping


class DuplicateEventError(ValueError):
    pass


class VersionGapError(ValueError):
    pass


def replay_balance(events: Iterable[Mapping[str, object]]) -> Decimal:
    balance = Decimal("0")
    seen: set[str] = set()
    for event in events:
        event_id = str(event["id"])
        if event_id in seen:
            continue
        seen.add(event_id)
        amount = Decimal(event["amount"])
        if event["kind"] == "credit":
            balance += amount
        else:
            balance -= amount
    return balance.quantize(Decimal("0.01"))
