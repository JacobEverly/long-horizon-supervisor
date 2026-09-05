import hashlib
from pathlib import Path

from horizon_supervisor.stuck_detector import TurnObservation
from horizon_supervisor.stuck_detector_v1 import (
    ActionMode,
    StuckStatusV1,
    SuspectedStuckV1,
)

ROOT = Path(__file__).parents[1]
V0_SHA256 = "c3319c93d823455076fd294ac16e28748a2b2ebcab10e1b81760d174088f4ffe"


def turn(index: int, **updates: object) -> TurnObservation:
    values = {
        "run_id": "run-1",
        "turn": index,
        "max_turns": 12,
        "model_id": "test/model",
        "commands": (f"python build-{index}.py",),
        "terminal_tail": "work continues",
        "workspace_digest": f"digest-{index}",
        "output_token_budget": 49_152,
        "spend_budget_usd": 1.0,
    }
    values.update(updates)
    return TurnObservation(**values)


def test_v0_source_remains_byte_for_byte_frozen() -> None:
    source = ROOT / "src/horizon_supervisor/stuck_detector.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == V0_SHA256


def test_read_only_investigation_is_not_stuck_evidence() -> None:
    detector = SuspectedStuckV1()
    for index in range(1, 9):
        result = detector.observe(
            turn(
                index,
                commands=("ls -la", "find /app -maxdepth 2 -type f"),
                workspace_digest="unchanged",
            )
        )
        assert result.action_mode == ActionMode.INSPECTION
        assert result.status == StuckStatusV1.HEALTHY


def test_two_productive_error_turns_trigger_only_after_minimum_turn() -> None:
    detector = SuspectedStuckV1()
    for index in range(1, 5):
        detector.observe(turn(index))
    fifth = detector.observe(
        turn(5, commands=("pytest -q",), terminal_tail="ERROR same failure")
    )
    sixth = detector.observe(
        turn(6, commands=("pytest -q",), terminal_tail="ERROR same failure")
    )
    assert fifth.status == StuckStatusV1.HEALTHY
    assert sixth.status == StuckStatusV1.SUSPECTED_STUCK
    assert sixth.rule == "repeated_productive_failure_for_two_consecutive_turns"


def test_protocol_failure_is_structural_not_model_stuck() -> None:
    result = SuspectedStuckV1().observe(turn(1, protocol_failure=True))
    assert result.status == StuckStatusV1.STRUCTURAL_FAILURE


def test_v1_spec_forbids_task_identity_and_future_information() -> None:
    spec = SuspectedStuckV1.frozen_spec()
    assert spec["schema_version"] == "suspected-stuck.v1"
    assert "task identity" in spec["forbidden_inputs"]
    assert "future trajectory information" in spec["forbidden_inputs"]
