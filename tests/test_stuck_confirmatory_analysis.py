import json
from pathlib import Path

import pytest

from horizon_supervisor.training.stuck_confirmatory_analysis import (
    BASE_MODELS,
    BRANCHES,
    _load,
    analyze,
)


def _rows(*, complete: bool = True) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    groups_per_kind = 12 if complete else 2
    for kind in ("suspected_stuck", "healthy"):
        for index in range(groups_per_kind):
            task_id = f"task-{index % 8}"
            base = BASE_MODELS[index % 2]
            for action in sorted(BRANCHES):
                if kind == "suspected_stuck":
                    success = action == "switch_kimi_state"
                else:
                    success = action in {"continue_current_state", "switch_kimi_state"}
                rows.append(
                    {
                        "schema_version": "matched-stuck-branch-outcome.v0",
                        "group_id": f"{kind}-{index}",
                        "task_id": task_id,
                        "task_category": "test",
                        "checkpoint_kind": kind,
                        "checkpoint_turn": 4,
                        "base_model_id": base,
                        "destination_model_id": base,
                        "branch_action": action,
                        "preserved_state": action != "restart_kimi_clean",
                        "remaining_turns": 8,
                        "remaining_output_tokens": 32768,
                        "maximum_wall_seconds": 3000,
                        "maximum_incremental_spend_usd": 0.5,
                        "verified_completion": success,
                        "verifier_reward": float(success),
                        "cost_usd": 0.1 if action == "switch_kimi_state" else 0.02,
                        "input_tokens": 100,
                        "output_tokens": 20,
                        "cached_tokens": 10,
                        "reasoning_tokens": 5,
                        "elapsed_seconds": 10.0,
                        "state_transfer_failure": False,
                        "protocol_error": False,
                        "provider_error": False,
                        "valid": True,
                        "source_job": f"job-{kind}-{index}-{action}",
                    }
                )
    return rows


def test_complete_confirmatory_dataset_passes_both_gates() -> None:
    report = analyze(_rows())
    assert report["dataset"]["target_complete"] is True
    assert report["valid_outcomes"] == 96
    assert report["decision_gates"]["detector_gate_passed"] is True
    assert report["decision_gates"]["kimi_intervention_gate_passed"] is True
    assert report["decision"] == "CONFIRMED — proceed to training-sized collection"


def test_incomplete_pool_is_inconclusive() -> None:
    report = analyze(_rows(complete=False), pool_exhausted=True)
    assert report["dataset"]["target_complete"] is False
    assert report["decision"] == "INCONCLUSIVE — improve coverage and repeat"


def test_loader_rejects_incomplete_group(tmp_path: Path) -> None:
    rows = _rows(complete=False)
    rows.pop()
    path = tmp_path / "outcomes.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete or duplicated"):
        _load(path)
