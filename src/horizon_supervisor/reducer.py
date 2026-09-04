from __future__ import annotations

from horizon_supervisor.models import (
    BudgetState,
    EventKind,
    ProgressTrend,
    RunPhase,
    SupervisorEvent,
    SupervisorState,
    ValidationSnapshot,
    WorkspaceRef,
)


class StateReducer:
    """Deterministically folds normalized events into portable supervisor state."""

    def apply(
        self,
        state: SupervisorState | None,
        event: SupervisorEvent,
    ) -> SupervisorState:
        if state is None:
            return self._start(event)

        self._validate_sequence(state, event)
        next_state = state.model_copy(deep=True)
        next_state.event_count = event.sequence
        next_state.steps_since_model_switch += 1

        if event.kind == EventKind.TASK_STARTED:
            raise ValueError("task_started can only be the first event")
        if event.kind == EventKind.PHASE_CHANGED:
            next_state.phase = RunPhase(event.payload["phase"])
            next_state.active_milestone = event.payload.get("active_milestone")
        elif event.kind == EventKind.PLAN_COMMITTED:
            next_state.has_committed_plan = True
            next_state.plan_summary = str(event.payload["summary"])
            next_state.progress_trend = ProgressTrend.IMPROVING
            next_state.consecutive_unproductive_steps = 0
        elif event.kind == EventKind.TOOL_RESULT:
            self._apply_tool_result(next_state, event)
        elif event.kind == EventKind.FILES_CHANGED:
            changed = set(next_state.workspace.changed_files)
            changed.update(str(path) for path in event.payload["changed_files"])
            next_state.workspace.changed_files = sorted(changed)
            next_state.workspace.diff_hash = event.payload.get(
                "diff_hash", next_state.workspace.diff_hash
            )
        elif event.kind == EventKind.VALIDATION_RESULT:
            self._apply_validation(next_state, event)
        elif event.kind == EventKind.MILESTONE_COMPLETED:
            milestone = str(event.payload["milestone"])
            if milestone not in next_state.completed_milestones:
                next_state.completed_milestones.append(milestone)
            if next_state.active_milestone == milestone:
                next_state.active_milestone = None
            next_state.progress_trend = ProgressTrend.IMPROVING
            next_state.consecutive_unproductive_steps = 0
        elif event.kind == EventKind.CONTEXT_COMPACTED:
            next_state.context_revision += 1
            next_state.context_summary = str(event.payload["summary"])
        elif event.kind == EventKind.MODEL_SELECTED:
            next_state.current_model_id = str(event.payload["model_id"])
            next_state.steps_since_model_switch = 0
        elif event.kind == EventKind.TASK_FINISHED:
            next_state.finished = True
            next_state.succeeded = bool(event.payload["success"])
            next_state.phase = RunPhase.COMPLETE

        if "spent_usd" in event.metadata:
            next_state.budget.spent_usd = float(event.metadata["spent_usd"])
        if "forecast_remaining_tokens" in event.metadata:
            next_state.forecast_remaining_tokens = int(event.metadata["forecast_remaining_tokens"])
        return next_state

    @staticmethod
    def _start(event: SupervisorEvent) -> SupervisorState:
        if event.kind != EventKind.TASK_STARTED or event.sequence != 1:
            raise ValueError("the first event must be task_started with sequence 1")
        payload = event.payload
        return SupervisorState(
            run_id=event.run_id,
            objective=str(payload["objective"]),
            phase=RunPhase(payload["phase"]),
            current_model_id=str(payload["current_model_id"]),
            workspace=WorkspaceRef(
                workspace_id=str(payload["workspace_id"]),
                working_directory=str(payload["working_directory"]),
                git_head=payload.get("git_head"),
            ),
            budget=BudgetState(
                total_usd=float(payload["budget_total_usd"]),
                reserved_usd=float(payload.get("budget_reserved_usd", 0)),
            ),
            forecast_remaining_tokens=int(payload.get("forecast_remaining_tokens", 100_000)),
            handoff_tokens=int(payload.get("handoff_tokens", 2_000)),
            event_count=1,
        )

    @staticmethod
    def _validate_sequence(state: SupervisorState, event: SupervisorEvent) -> None:
        if event.run_id != state.run_id:
            raise ValueError(f"event run_id {event.run_id!r} does not match state")
        expected = state.event_count + 1
        if event.sequence != expected:
            raise ValueError(f"expected event sequence {expected}, got {event.sequence}")
        if state.finished:
            raise ValueError("cannot apply events to a finished run")

    @staticmethod
    def _apply_tool_result(state: SupervisorState, event: SupervisorEvent) -> None:
        success = bool(event.payload["success"])
        made_progress = bool(event.payload["made_progress"])

        if success and made_progress:
            state.progress_trend = ProgressTrend.IMPROVING
            state.consecutive_unproductive_steps = 0
            state.consecutive_errors = 0
            return

        state.consecutive_unproductive_steps += 1
        if not success:
            state.consecutive_errors += 1
        if state.consecutive_unproductive_steps >= 2:
            state.progress_trend = ProgressTrend.STALLED
        else:
            state.progress_trend = ProgressTrend.STABLE

    @staticmethod
    def _apply_validation(state: SupervisorState, event: SupervisorEvent) -> None:
        snapshot = ValidationSnapshot(
            suite=str(event.payload.get("suite", "default")),
            passed=int(event.payload["passed"]),
            failed=int(event.payload["failed"]),
            skipped=int(event.payload.get("skipped", 0)),
        )

        previous = state.validation_history[-1] if state.validation_history else None
        state.validation_history.append(snapshot)

        if snapshot.failed == 0:
            state.progress_trend = ProgressTrend.IMPROVING
            state.consecutive_errors = 0
            state.consecutive_unproductive_steps = 0
        elif previous is None:
            state.progress_trend = ProgressTrend.STABLE
            state.consecutive_errors += 1
        elif snapshot.failed < previous.failed or (
            snapshot.failed == previous.failed and snapshot.passed > previous.passed
        ):
            state.progress_trend = ProgressTrend.IMPROVING
            state.consecutive_unproductive_steps = 0
        elif snapshot.failed > previous.failed:
            state.progress_trend = ProgressTrend.REGRESSING
            state.consecutive_errors += 1
            state.consecutive_unproductive_steps += 1
        else:
            state.consecutive_unproductive_steps += 1
            state.progress_trend = (
                ProgressTrend.STALLED
                if state.consecutive_unproductive_steps >= 2
                else ProgressTrend.STABLE
            )
