from __future__ import annotations

import pytest

from horizon_supervisor.training.recover_swiss_cheese_pair import (
    EXPECTED_KEY,
    consolidate_recovery_rows,
)

TASKS = (
    "implement-crc32-with-logic-gates",
    "improve-code-similarity-feature-extraction",
    "optimize-portfolio-allocation",
    "polyglot-text-stats-script",
    "polyglot-yaml-config-validator",
    "recover-and-sanitize-postgres-wal-secret",
    "recover-prod-db-password-from-git-history",
    "repair-broken-shell-data-pipeline",
    "xrd-two-peak-fitting",
)
ROUTES = (
    "gate7/fixed-flash",
    "gate7/fixed-qwen",
    "gate7/fixed-glm",
    "gate7/fixed-kimi",
)


def _row(task: str, route: str, status: str, outcome_id: str) -> dict:
    return {
        "task": {"source_task_name": task},
        "model": {"route_id": route},
        "provenance": {"replication_index": 3},
        "outcome_id": outcome_id,
        "outcome": {"status": status},
    }


def _source_rows() -> list[dict]:
    rows = []
    for task in TASKS:
        for route in ROUTES:
            key = (task, route, 3)
            status = "infrastructure_error" if key == EXPECTED_KEY else "verified"
            outcome_id = "invalid-source" if key == EXPECTED_KEY else f"{task}:{route}"
            rows.append(_row(task, route, status, outcome_id))
    return rows


def test_consolidates_only_the_frozen_infrastructure_invalid_pair() -> None:
    replacement = _row(EXPECTED_KEY[0], EXPECTED_KEY[1], "verified", "replacement")
    result = consolidate_recovery_rows(
        _source_rows(), [replacement], replaces_outcome_id="invalid-source"
    )

    assert len(result) == 36
    assert sum(row["outcome_id"] == "replacement" for row in result) == 1
    assert all(row["outcome"]["status"] == "verified" for row in result)


def test_rejects_rerunning_a_learning_valid_source_pair() -> None:
    rows = _source_rows()
    target = next(row for row in rows if row["outcome_id"] == "invalid-source")
    target["outcome"]["status"] = "verified"
    replacement = _row(EXPECTED_KEY[0], EXPECTED_KEY[1], "verified", "replacement")

    with pytest.raises(ValueError, match="exactly one infrastructure-invalid"):
        consolidate_recovery_rows(
            rows, [replacement], replaces_outcome_id="invalid-source"
        )


def test_rejects_an_invalid_row_outside_the_frozen_pair() -> None:
    rows = _source_rows()
    target = next(row for row in rows if row["outcome_id"] == "invalid-source")
    target["outcome"]["status"] = "verified"
    rows[0]["outcome"]["status"] = "infrastructure_error"
    replacement = _row(EXPECTED_KEY[0], EXPECTED_KEY[1], "verified", "replacement")

    with pytest.raises(ValueError, match="not the frozen Kimi xrd"):
        consolidate_recovery_rows(
            rows, [replacement], replaces_outcome_id="invalid-source"
        )


def test_rejects_an_infrastructure_invalid_recovery_outcome() -> None:
    replacement = _row(
        EXPECTED_KEY[0], EXPECTED_KEY[1], "infrastructure_error", "replacement"
    )

    with pytest.raises(ValueError, match="not learning-valid"):
        consolidate_recovery_rows(
            _source_rows(), [replacement], replaces_outcome_id="invalid-source"
        )
