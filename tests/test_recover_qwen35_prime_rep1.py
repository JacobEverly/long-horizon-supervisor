from __future__ import annotations

import pytest

from horizon_supervisor.training.recover_qwen35_prime_rep1 import (
    EXPECTED_KEY,
    consolidate_recovery_rows,
)


def _row(task: str, route: str, rep: int, status: str, outcome_id: str) -> dict:
    return {
        "task": {"source_task_name": task},
        "model": {"route_id": route},
        "provenance": {"replication_index": rep},
        "outcome_id": outcome_id,
        "outcome": {"status": status, "completed": status == "verified"},
    }


def _source_rows() -> list[dict]:
    rows = []
    for index in range(10):
        task = EXPECTED_KEY[0] if index == 5 else f"task-{index}"
        key = (task, EXPECTED_KEY[1], EXPECTED_KEY[2])
        status = "infrastructure_error" if key == EXPECTED_KEY else "verified"
        rows.append(_row(*key, status, "invalid" if key == EXPECTED_KEY else task))
    return rows


def test_replaces_only_frozen_infrastructure_pair() -> None:
    replacement = _row(*EXPECTED_KEY, "verified", "replacement")
    result = consolidate_recovery_rows(
        _source_rows(), [replacement], replaces_outcome_id="invalid"
    )

    assert len(result) == 10
    assert sum(row["outcome_id"] == "replacement" for row in result) == 1
    assert all(row["outcome"]["status"] in {"verified"} for row in result)


def test_rejects_invalid_recovery() -> None:
    replacement = _row(*EXPECTED_KEY, "infrastructure_error", "replacement")
    with pytest.raises(ValueError, match="not learning-valid"):
        consolidate_recovery_rows(
            _source_rows(), [replacement], replaces_outcome_id="invalid"
        )
