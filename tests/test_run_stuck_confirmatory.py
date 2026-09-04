import json

from horizon_supervisor.training.run_stuck_confirmatory import (
    BRANCH_ACTIONS,
    _acceptance_block_reason,
    _branch_specs,
    _eligible_checkpoint,
    _selection_summary,
    _target_met,
)


def _group(
    group_id: str, task: str, kind: str, base: str
) -> list[dict[str, object]]:
    return [
        {
            "group_id": group_id,
            "task_id": task,
            "checkpoint_kind": kind,
            "base_model_id": base,
            "branch_action": action,
        }
        for action in sorted(BRANCH_ACTIONS)
    ]


def test_branch_specs_are_four_explicit_matched_snapshot_arms() -> None:
    assert _branch_specs("gate7/fixed-flash") == [
        (
            "continue_current_state",
            "gate7/fixed-flash",
            "deepseek/deepseek-v4-flash-0731",
            True,
        ),
        (
            "switch_value_state",
            "gate7/fixed-qwen",
            "qwen/qwen3.8-27b",
            True,
        ),
        ("switch_kimi_state", "gate7/fixed-kimi", "moonshotai/kimi-k3", True),
        ("restart_kimi_clean", "gate7/fixed-kimi", "moonshotai/kimi-k3", False),
    ]
    assert _branch_specs("gate7/fixed-qwen")[0][1] == "gate7/fixed-qwen"
    assert _branch_specs("gate7/fixed-qwen")[1][1] == "gate7/fixed-flash"


def test_checkpoint_eligibility_does_not_require_later_scout_verifier(tmp_path) -> None:
    record = tmp_path / "records.jsonl"
    checkpoint = {
        "schema_version": "matched-checkpoint.v0",
        "checkpoint_kind": "suspected_stuck",
        "state_transfer_eligible": True,
        "state_transfer_ineligibility_reason": None,
    }
    record.write_text(json.dumps(checkpoint) + "\n", encoding="utf-8")
    selected, error = _eligible_checkpoint(
        {"valid": False, "record_path": str(record)}, "suspected_stuck"
    )
    assert selected == checkpoint
    assert error is None


def test_checkpoint_eligibility_rejects_missing_or_duplicate_records(tmp_path) -> None:
    record = tmp_path / "records.jsonl"
    record.write_text("", encoding="utf-8")
    selected, error = _eligible_checkpoint(
        {"valid": False, "record_path": str(record)}, "healthy"
    )
    assert selected is None
    assert error == "requested checkpoint did not occur"

    checkpoint = {
        "schema_version": "matched-checkpoint.v0",
        "checkpoint_kind": "healthy",
        "state_transfer_eligible": True,
        "state_transfer_ineligibility_reason": None,
    }
    record.write_text(
        json.dumps(checkpoint) + "\n" + json.dumps(checkpoint) + "\n",
        encoding="utf-8",
    )
    selected, error = _eligible_checkpoint(
        {"valid": True, "record_path": str(record)}, "healthy"
    )
    assert selected is None
    assert error == "duplicate checkpoints were captured"


def test_predeclared_caps_block_task_and_base_concentration() -> None:
    flash = "deepseek/deepseek-v4-flash-0731"
    outcomes = []
    outcomes += _group("g1", "task-a", "suspected_stuck", flash)
    outcomes += _group("g2", "task-a", "suspected_stuck", flash)
    assert _acceptance_block_reason(
        outcomes, task_id="task-a", kind="suspected_stuck", base_model=flash
    ) == "predeclared maximum of two groups per task and kind reached"

    for index in range(8):
        outcomes += _group(f"f{index}", f"task-{index}", "healthy", flash)
    assert _acceptance_block_reason(
        outcomes, task_id="new-task", kind="healthy", base_model=flash
    ) == "predeclared maximum of eight groups for this base and kind reached"


def test_target_requires_all_representation_constraints() -> None:
    flash = "deepseek/deepseek-v4-flash-0731"
    qwen = "qwen/qwen3.8-27b"
    outcomes = []
    for kind in ("suspected_stuck", "healthy"):
        for index in range(12):
            outcomes += _group(
                f"{kind}-{index}",
                f"task-{index % 8}",
                kind,
                flash if index % 2 == 0 else qwen,
            )
    summary = _selection_summary(outcomes)
    assert summary["valid_outcome_count"] == 96
    assert _target_met(outcomes) is True

    outcomes.pop()
    assert _target_met(outcomes) is False
