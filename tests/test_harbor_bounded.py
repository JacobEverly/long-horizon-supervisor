from __future__ import annotations

import asyncio
from types import MethodType

import pytest
from harbor.llms.base import OutputLengthExceededError

from horizon_supervisor.benchmark.harbor_bounded import (
    _bound_output_length_recursion,
    configure_harbor_retries,
)


def test_configure_harbor_retries_bounds_both_retry_layers() -> None:
    from harbor.agents.terminus_2.terminus_2 import Terminus2
    from harbor.llms.lite_llm import LiteLLM

    configure_harbor_retries(1)

    assert Terminus2._query_llm.retry.stop.max_attempt_number == 1
    assert LiteLLM.call.retry.stop.max_attempt_number == 1


def test_configure_harbor_retries_rejects_zero() -> None:
    with pytest.raises(ValueError, match="positive"):
        configure_harbor_retries(0)


def test_output_length_recursion_allows_one_correction_then_stops() -> None:
    class FakeTerminus:
        calls = 0

    async def recursively_truncated(self: FakeTerminus) -> None:
        self.calls += 1
        await self._query_llm()  # type: ignore[attr-defined]

    fake = FakeTerminus()
    bounded = _bound_output_length_recursion(recursively_truncated, 1)
    fake._query_llm = MethodType(bounded, fake)  # type: ignore[attr-defined]

    async def invoke() -> None:
        await fake._query_llm()  # type: ignore[attr-defined]

    with pytest.raises(OutputLengthExceededError, match="stopped repeated"):
        asyncio.run(invoke())

    assert fake.calls == 2


def test_output_length_recursion_rejects_negative_attempts() -> None:
    async def never_called(_self: object) -> None:
        raise AssertionError

    with pytest.raises(ValueError, match="cannot be negative"):
        _bound_output_length_recursion(never_called, -1)
