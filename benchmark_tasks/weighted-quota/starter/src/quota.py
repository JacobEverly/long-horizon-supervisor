from __future__ import annotations

from collections.abc import Mapping


def allocate_quota(
    demands: Mapping[str, int], capacity: int, weights: Mapping[str, int]
) -> dict[str, int]:
    allocated = {tenant: 0 for tenant in demands}
    remaining = capacity
    for tenant, demand in demands.items():
        grant = min(demand, remaining)
        allocated[tenant] = grant
        remaining -= grant
    return allocated
