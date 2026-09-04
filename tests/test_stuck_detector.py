import pytest
from pydantic import ValidationError

from horizon_supervisor.stuck_detector import (
    PublicTestObservation,
    StuckStatus,
    SuspectedStuckV0,
    TurnObservation,
)


def turn(index: int, **updates: object) -> TurnObservation:
    values = {
        "run_id": "run-1",
        "turn": index,
        "max_turns": 12,
        "model_id": "test/model",
        "commands": (f"python step-{index}.py",),
        "terminal_tail": "work completed without a public milestone",
        "workspace_digest": f"digest-{index}",
        "output_token_budget": 49_152,
        "spend_budget_usd": 1.0,
    }
    values.update(updates)
    return TurnObservation(**values)


def failing_tests(fingerprint: str = "same") -> PublicTestObservation:
    return PublicTestObservation(
        command_fingerprint="pytest",
        passed=2,
        failed=1,
        failure_fingerprints=(fingerprint,),
    )


def test_changed_files_alone_are_not_meaningful_progress() -> None:
    detector = SuspectedStuckV0()
    detector.observe(turn(1, workspace_digest="a"))
    result = detector.observe(turn(2, workspace_digest="b"))
    assert result.meaningful_progress is False


def test_public_test_improvement_is_meaningful_progress() -> None:
    detector = SuspectedStuckV0()
    detector.observe(turn(1, public_tests=failing_tests()))
    result = detector.observe(
        turn(
            2,
            public_tests=PublicTestObservation(
                command_fingerprint="pytest", passed=3, failed=0
            ),
        )
    )
    assert result.status == StuckStatus.HEALTHY
    assert "additional_public_tests_passing" in result.progress_reasons
    assert "fewer_distinct_public_test_failures" in result.progress_reasons


def test_clear_repeated_action_error_loop_triggers_immediately() -> None:
    detector = SuspectedStuckV0()
    detector.observe(
        turn(
            1,
            commands=("pytest -q",),
            terminal_tail="ERROR missing module",
            workspace_digest="same",
        )
    )
    result = detector.observe(
        turn(
            2,
            commands=("  PYTEST   -q ",),
            terminal_tail="Error: missing module",
            workspace_digest="same",
        )
    )
    assert result.status == StuckStatus.SUSPECTED_STUCK
    assert result.immediate_signal == "clear_action_error_loop"


def test_two_independent_signals_must_persist() -> None:
    detector = SuspectedStuckV0()
    detector.observe(turn(1, public_tests=failing_tests(), workspace_digest="same"))
    second = detector.observe(
        turn(2, public_tests=failing_tests(), workspace_digest="same", commands=("inspect",))
    )
    third = detector.observe(
        turn(3, public_tests=failing_tests(), workspace_digest="same", commands=("inspect more",))
    )
    assert second.status == StuckStatus.HEALTHY
    assert third.status == StuckStatus.SUSPECTED_STUCK
    assert {
        "unchanged_failing_public_tests",
        "consecutive_turns_without_meaningful_progress",
    }.issubset(third.persistent_signals)


def test_protocol_failure_is_immediate() -> None:
    result = SuspectedStuckV0().observe(turn(1, protocol_failure=True))
    assert result.status == StuckStatus.SUSPECTED_STUCK


def test_detector_rejects_future_or_hidden_fields() -> None:
    payload = turn(1).model_dump()
    payload["final_task_success"] = True
    with pytest.raises(ValidationError, match="extra_forbidden"):
        TurnObservation.model_validate(payload)


def test_detector_accepts_forward_gaps_but_rejects_repeated_turns_or_mixed_runs() -> None:
    detector = SuspectedStuckV0()
    detector.observe(turn(1))
    after_gap = detector.observe(
        turn(
            3,
            commands=("python step-1.py",),
            workspace_digest="digest-1",
        )
    )
    assert after_gap.status == StuckStatus.HEALTHY
    assert after_gap.persistent_signals == ()
    assert after_gap.immediate_signal is None

    with pytest.raises(ValueError, match="strictly increasing"):
        detector.observe(turn(3))

    other = turn(4).model_copy(update={"run_id": "other"})
    with pytest.raises(ValueError, match="run ids"):
        detector.observe(other)


def test_frozen_spec_names_all_forbidden_inputs() -> None:
    spec = SuspectedStuckV0.frozen_spec()
    assert spec["schema_version"] == "suspected-stuck.v0"
    assert "private model reasoning" in spec["forbidden_inputs"]
