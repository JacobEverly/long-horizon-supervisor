import json
from pathlib import Path

import pytest

from horizon_supervisor.training.stuck_pilot_analysis import BRANCHES, _load, analyze


def rows() -> list[dict]:
    output = []
    for group_index, kind in enumerate(("suspected_stuck", "healthy"), start=1):
        for action in BRANCHES:
            success = (
                action in {"switch_value_state", "switch_kimi_state"}
                if kind == "suspected_stuck"
                else action == "continue_current_state"
            )
            output.append(
                {
                    "group_id": f"g-{group_index}",
                    "task_id": f"task-{group_index}",
                    "checkpoint_kind": kind,
                    "base_model_id": "deepseek/deepseek-v4-flash-0731",
                    "branch_action": action,
                    "valid": True,
                    "remaining_turns": 6,
                    "remaining_output_tokens": 24_576,
                    "maximum_wall_seconds": 1_800,
                    "maximum_incremental_spend_usd": 0.5,
                    "verified_completion": success,
                    "verifier_reward": float(success),
                    "cost_usd": 0.1,
                    "input_tokens": 100,
                    "output_tokens": 100,
                    "cached_tokens": 0,
                    "reasoning_tokens": 0,
                    "elapsed_seconds": 10,
                    "state_transfer_failure": False,
                    "protocol_error": False,
                    "provider_error": False,
                }
            )
    return output


def test_analysis_reports_detector_and_intervention_interaction() -> None:
    report = analyze(rows())
    assert report["valid_outcomes"] == 12
    assert report["detector"]["healthy_minus_stuck_recovery_rate"] == 1.0
    assert report["detector"]["false_positive_stuck_triggers"] == 0
    assert (
        report["trigger_vs_fixed_turn_interaction"]["switch_value_state"][
            "difference_in_differences"
        ]
        == 2.0
    )


def test_loader_rejects_incomplete_matched_group(tmp_path: Path) -> None:
    path = tmp_path / "outcomes.jsonl"
    path.write_text(json.dumps(rows()[0]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        _load(path)
