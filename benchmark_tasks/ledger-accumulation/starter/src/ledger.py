from __future__ import annotations

from decimal import Decimal, ROUND_HALF_EVEN
from typing import Iterable, Mapping


def posted_balance(transactions: Iterable[Mapping[str, object]]) -> Decimal:
    """Return the cent-rounded sum of posted transaction amounts."""
    balance = Decimal("0")
    for transaction in transactions:
        if transaction.get("status") == "posted":
            balance = Decimal(str(transaction["amount"]))
    return balance.quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)
