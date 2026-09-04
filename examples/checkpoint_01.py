"""Checkpoint 1: simulate bidirectional routing without spending API money."""

from __future__ import annotations

from horizon_supervisor import (
    CompletionConstrainedPolicy,
    EventKind,
    HeuristicCompletionEstimator,
    ModelProfile,
    StateReducer,
    SupervisorEvent,
)

MODELS = [
    ModelProfile(
        model_id="worker-8b",
        tier=0,
        capability_score=0.58,
        cost_per_million_tokens_usd=0.20,
    ),
    ModelProfile(
        model_id="middle-fast",
        tier=1,
        capability_score=0.82,
        cost_per_million_tokens_usd=1.00,
    ),
    ModelProfile(
        model_id="frontier-reasoner",
        tier=2,
        capability_score=0.96,
        cost_per_million_tokens_usd=8.00,
    ),
]


class Simulation:
    def __init__(self) -> None:
        self.reducer = StateReducer()
        self.policy = CompletionConstrainedPolicy(
            estimator=HeuristicCompletionEstimator(),
            reliability_threshold=0.70,
        )
        self.state = None
        self.sequence = 0

    def emit(self, kind: EventKind, **payload: object) -> None:
        self.sequence += 1
        event = SupervisorEvent(
            run_id="checkpoint-01",
            sequence=self.sequence,
            kind=kind,
            payload=payload,
        )
        self.state = self.reducer.apply(self.state, event)

    def route(self, checkpoint: str) -> None:
        assert self.state is not None
        decision = self.policy.decide(self.state, MODELS)
        scores = ", ".join(
            f"{estimate.model_id}={estimate.reliability_score:.2f}"
            for estimate in decision.estimates
        )
        print(
            f"{checkpoint:<29} {self.state.current_model_id:<19} "
            f"{decision.action.value:<12} -> {decision.target_model_id:<19} [{scores}]"
        )
        if decision.target_model_id != self.state.current_model_id:
            self.emit(EventKind.MODEL_SELECTED, model_id=decision.target_model_id)


def main() -> None:
    sim = Simulation()
    sim.emit(
        EventKind.TASK_STARTED,
        objective="Repair a concurrency bug and pass the repository verifier",
        phase="discovery",
        current_model_id="frontier-reasoner",
        budget_total_usd=8.00,
        budget_reserved_usd=1.50,
        workspace_id="sandbox-143",
        working_directory="/workspace",
        git_head="abc123",
        forecast_remaining_tokens=200_000,
    )

    print("checkpoint                    current model       action       target")
    print("-" * 126)
    sim.route("uncertain task start")

    sim.emit(
        EventKind.PLAN_COMMITTED,
        summary="Move invalidation inside the write lock, preserving lock ordering.",
    )
    sim.emit(
        EventKind.PHASE_CHANGED,
        phase="implementation",
        active_milestone="implement lock-order fix",
    )
    sim.route("plan is ready")

    sim.emit(EventKind.FILES_CHANGED, changed_files=["cache.py"], diff_hash="diff-001")
    sim.emit(EventKind.TOOL_RESULT, success=True, made_progress=True)
    sim.route("cheap worker progressing")

    sim.emit(EventKind.VALIDATION_RESULT, suite="pytest", passed=124, failed=2)
    sim.emit(EventKind.TOOL_RESULT, success=False, made_progress=False)
    sim.emit(EventKind.VALIDATION_RESULT, suite="pytest", passed=122, failed=4)
    sim.emit(EventKind.TOOL_RESULT, success=False, made_progress=False)
    sim.route("tests regress; worker stuck")

    sim.emit(
        EventKind.TOOL_RESULT,
        success=True,
        made_progress=True,
        summary="Frontier model identified reversed lock acquisition order.",
    )
    sim.emit(
        EventKind.PLAN_COMMITTED,
        summary="Acquire write lock before refresh lock, then re-run concurrency tests.",
    )
    sim.route("recovery plan established")

    sim.emit(EventKind.FILES_CHANGED, changed_files=["cache.py", "test_cache.py"])
    sim.emit(EventKind.MILESTONE_COMPLETED, milestone="implement lock-order fix")
    sim.emit(EventKind.PHASE_CHANGED, phase="validation")
    sim.emit(EventKind.VALIDATION_RESULT, suite="pytest", passed=128, failed=0)
    sim.route("verification is mechanical")

    sim.emit(EventKind.TASK_FINISHED, success=True)
    assert sim.state is not None
    print("-" * 126)
    print(
        f"completed={sim.state.succeeded} events={sim.state.event_count} "
        f"final_model={sim.state.current_model_id} changed={sim.state.workspace.changed_files}"
    )


if __name__ == "__main__":
    main()
