from __future__ import annotations

import pytest

from horizon_supervisor.training.resume_swiss_cheese_contract import (
    accounted_key_usage,
    select_locked_and_pending,
)


def _row(task: str, status: str, outcome_id: str) -> dict:
    return {
        "task": {"source_task_name": task},
        "model": {"route_id": "route"},
        "provenance": {"replication_index": 2},
        "outcome": {"status": status},
        "outcome_id": outcome_id,
    }


def test_locks_valid_rows_and_keeps_only_invalid_or_missing_pending() -> None:
    expected = [("a", "route", 2), ("b", "route", 2), ("c", "route", 2)]
    locked, pending = select_locked_and_pending(
        [_row("a", "verified", "a"), _row("b", "infrastructure_error", "b")],
        [],
        expected,
    )
    assert list(locked) == [("a", "route", 2)]
    assert pending == [("b", "route", 2), ("c", "route", 2)]


def test_progress_row_prevents_rerunning_recovered_pair() -> None:
    expected = [("a", "route", 2), ("b", "route", 2)]
    locked, pending = select_locked_and_pending(
        [_row("a", "infrastructure_error", "old")],
        [_row("a", "verified", "replacement")],
        expected,
    )
    assert locked[("a", "route", 2)]["outcome_id"] == "replacement"
    assert pending == [("b", "route", 2)]


def test_rejects_duplicate_learning_valid_evidence() -> None:
    expected = [("a", "route", 2)]
    with pytest.raises(ValueError, match="multiple learning-valid"):
        select_locked_and_pending(
            [_row("a", "verified", "one")],
            [_row("a", "verified", "two")],
            expected,
        )


def test_accounts_across_expired_and_replacement_keys() -> None:
    assert accounted_key_usage(0.125, 39.694240884) == pytest.approx(39.819240884)


def test_rejects_negative_rollover_accounting() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        accounted_key_usage(-0.01, 39.694240884)
