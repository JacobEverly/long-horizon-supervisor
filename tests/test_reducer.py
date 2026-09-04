from __future__ import annotations

import pytest

from horizon_supervisor import EventKind, StateReducer, SupervisorEvent
from horizon_supervisor.models import ProgressTrend


def event(sequence: int, kind: EventKind, **payload: object) -> SupervisorEvent:
    return SupervisorEvent(
        run_id="run-1",
        sequence=sequence,
        kind=kind,
        payload=payload,
    )


def started_event() -> SupervisorEvent:
    return event(
        1,
        EventKind.TASK_STARTED,
        objective="Fix the failing tests",
        phase="discovery",
        current_model_id="frontier",
        budget_total_usd=10,
        workspace_id="sandbox-1",
        working_directory="/workspace",
    )


def test_reducer_tracks_external_references_and_validation_trend() -> None:
    reducer = StateReducer()
    state = reducer.apply(None, started_event())
    state = reducer.apply(
        state,
        event(2, EventKind.FILES_CHANGED, changed_files=["router.py"], diff_hash="d1"),
    )
    state = reducer.apply(
        state,
        event(3, EventKind.VALIDATION_RESULT, suite="pytest", passed=8, failed=2),
    )
    state = reducer.apply(
        state,
        event(4, EventKind.VALIDATION_RESULT, suite="pytest", passed=10, failed=0),
    )

    assert state.workspace.changed_files == ["router.py"]
    assert state.workspace.diff_hash == "d1"
    assert len(state.validation_history) == 2
    assert state.progress_trend == ProgressTrend.IMPROVING
    assert state.event_count == 4


def test_reducer_marks_repeated_unproductive_steps_as_stalled() -> None:
    reducer = StateReducer()
    state = reducer.apply(None, started_event())
    state = reducer.apply(
        state,
        event(2, EventKind.TOOL_RESULT, success=False, made_progress=False),
    )
    state = reducer.apply(
        state,
        event(3, EventKind.TOOL_RESULT, success=False, made_progress=False),
    )

    assert state.progress_trend == ProgressTrend.STALLED
    assert state.consecutive_unproductive_steps == 2
    assert state.consecutive_errors == 2


def test_reducer_rejects_sequence_gaps() -> None:
    reducer = StateReducer()
    state = reducer.apply(None, started_event())

    with pytest.raises(ValueError, match="expected event sequence 2"):
        reducer.apply(
            state,
            event(3, EventKind.TOOL_RESULT, success=True, made_progress=True),
        )


def test_event_rejects_missing_required_payload() -> None:
    with pytest.raises(ValueError, match="made_progress"):
        event(2, EventKind.TOOL_RESULT, success=True)
