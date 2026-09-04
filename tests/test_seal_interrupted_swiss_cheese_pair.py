from __future__ import annotations

import pytest

from horizon_supervisor.training.seal_interrupted_swiss_cheese_pair import (
    interrupted_run_accounting,
)


def _manifest() -> dict:
    return {
        "config": {"authorized_model_budget_usd": 50.0},
        "dedicated_key_remaining_before_usd": 10.333925736,
    }


def test_recovers_exact_spend_from_pre_run_remaining_and_dashboard_usage() -> None:
    before, after, spend = interrupted_run_accounting(_manifest(), 39.694240884)
    assert before == pytest.approx(39.666074264)
    assert after == pytest.approx(39.694240884)
    assert spend == pytest.approx(0.02816662)


@pytest.mark.parametrize("usage", [39.0, 50.01])
def test_rejects_impossible_post_run_usage(usage: float) -> None:
    with pytest.raises(ValueError, match="outside the valid range"):
        interrupted_run_accounting(_manifest(), usage)
