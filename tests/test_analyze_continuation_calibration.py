from horizon_supervisor.training.analyze_continuation_calibration import (
    DECISION_FAIL,
    DECISION_PASS,
    analyze,
)


def _row(task: int, route: str) -> dict:
    checkpoints = [
        {
            "kind": "healthy",
            "turn": 4,
            "remaining_turns": 8,
            "state_transfer_eligible": True,
            "snapshot_fidelity_passed": True,
        }
    ]
    confirmed_task = task < 4
    review_task = task < 4 or (task >= 4 and route == "gate7/fixed-flash")
    if review_task:
        checkpoints.append(
            {
                "kind": "needs_review",
                "turn": 5,
                "remaining_turns": 7,
                "state_transfer_eligible": True,
                "snapshot_fidelity_passed": True,
            }
        )
    if confirmed_task:
        checkpoints.append(
            {
                "kind": "confirmed_stuck",
                "turn": 6,
                "remaining_turns": 6,
                "state_transfer_eligible": True,
                "snapshot_fidelity_passed": True,
            }
        )
    return {
        "schema_version": "natural-continuation-outcome.v0",
        "task_id": f"task-{task}",
        "route_id": route,
        "model_id": route,
        "valid": True,
        "structural_failure": False,
        "verifier_outcome_present": True,
        "verified_completion": not confirmed_task,
        "leakage_check_passed": True,
        "checkpoints": checkpoints,
    }


def test_analysis_passes_balanced_high_separation_evidence() -> None:
    rows = [
        _row(task, route)
        for task in range(8)
        for route in ("gate7/fixed-flash", "gate7/fixed-qwen")
    ]

    report = analyze(rows, samples=1_000)

    assert report["decision"] == DECISION_PASS
    assert report["gate_passed"] is True
    assert report["tiers"]["healthy"]["checkpoint_count"] == 16
    assert report["tiers"]["needs_review"]["checkpoint_count"] == 12
    assert report["tiers"]["confirmed_stuck"]["checkpoint_count"] == 8
    assert report["review_to_confirmation_transition_rate"] == 8 / 12
    assert all(report["gates"].values())


def test_analysis_rejects_structural_rows_and_missing_fidelity() -> None:
    row = _row(0, "gate7/fixed-flash")
    row["structural_failure"] = True
    row["checkpoints"] = []
    second = _row(4, "gate7/fixed-qwen")
    second["checkpoints"][0]["snapshot_fidelity_passed"] = False

    report = analyze([row, second], samples=100)

    assert report["decision"] == DECISION_FAIL
    assert report["structural_trajectory_count"] == 1
    assert report["tiers"]["healthy"]["checkpoint_count"] == 0
    assert report["gates"]["all_counted_snapshots_rehydrated"] is False


def test_analysis_rejects_any_row_without_leakage_attestation() -> None:
    rows = [_row(0, "gate7/fixed-flash")]
    rows[0]["leakage_check_passed"] = False

    report = analyze(rows, samples=100)

    assert report["gates"]["leakage_controls"] is False
