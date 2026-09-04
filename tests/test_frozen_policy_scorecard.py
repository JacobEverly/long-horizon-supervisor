from __future__ import annotations

import json
from pathlib import Path

from horizon_supervisor.training.frozen_policy_scorecard import (
    EXPECTED_ROUTES,
    build_frozen_policy_scorecard,
)


def _write_outcomes(path: Path, task_count: int) -> None:
    rows = []
    for task_index in range(task_count):
        for route_index, route_id in enumerate(EXPECTED_ROUTES):
            rows.append(
                {
                    "schema_version": "matched-model-outcome.v1",
                    "matched_group_id": f"group-{task_index}",
                    "task": {"source_task_name": f"task-{task_index:02d}"},
                    "model": {"route_id": route_id},
                    "outcome": {
                        "status": "verified",
                        "completed": task_index % 4 == route_index,
                        "estimated_list_cost_usd": 0.01 * (route_index + 1),
                        "duration_seconds": 10.0 * (route_index + 1),
                    },
                }
            )
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _build_scorecard(tmp_path: Path, task_count: int) -> dict:
    outcomes_path = tmp_path / f"outcomes-{task_count}.jsonl"
    _write_outcomes(outcomes_path, task_count)
    return build_frozen_policy_scorecard(
        outcomes_path,
        tmp_path / f"scorecard-{task_count}.json",
        dedicated_key_usage_before_usd=20.0,
        dedicated_key_usage_after_usd=24.0,
        completed_run_report_spend_usd=3.9,
        expected_tasks=task_count,
    )


def test_eleven_task_scorecard_preserves_provisional_contract(tmp_path: Path) -> None:
    scorecard = _build_scorecard(tmp_path, 11)

    assert scorecard["evaluation_role"] == "provisional held-out Wave 3 checkpoint"
    assert "preliminary_cost_completion_pareto" in scorecard
    assert "cost_completion_pareto" not in scorecard
    assert scorecard["spend_audit"]["under_five_usd"] is True
    assert "under_ten_usd" not in scorecard["spend_audit"]
    assert scorecard["interpretation_guard"].startswith(
        "Eleven held-out tasks support a provisional direction"
    )


def test_eighteen_task_scorecard_uses_final_language(tmp_path: Path) -> None:
    scorecard = _build_scorecard(tmp_path, 18)
    serialized = json.dumps(scorecard).lower()

    assert scorecard["evaluation_role"] == "final held-out Wave 3 evaluation"
    assert scorecard["data"]["tasks"] == 18
    assert scorecard["data"]["records"] == 72
    assert "cost_completion_pareto" in scorecard
    assert "preliminary_cost_completion_pareto" not in scorecard
    assert scorecard["spend_audit"]["under_ten_usd"] is True
    assert "under_five_usd" not in scorecard["spend_audit"]
    assert "provisional" not in serialized
    assert "preliminary" not in serialized
