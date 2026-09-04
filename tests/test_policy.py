from __future__ import annotations

from horizon_supervisor import (
    CompletionConstrainedPolicy,
    EventKind,
    HeuristicCompletionEstimator,
    ModelProfile,
    RoutingAction,
    StateReducer,
    SupervisorEvent,
)

MODELS = [
    ModelProfile(
        model_id="worker",
        tier=0,
        capability_score=0.58,
        cost_per_million_tokens_usd=0.20,
    ),
    ModelProfile(
        model_id="middle",
        tier=1,
        capability_score=0.82,
        cost_per_million_tokens_usd=1.00,
    ),
    ModelProfile(
        model_id="frontier",
        tier=2,
        capability_score=0.96,
        cost_per_million_tokens_usd=8.00,
    ),
]


def build_state(current_model: str = "frontier"):
    reducer = StateReducer()
    state = reducer.apply(
        None,
        SupervisorEvent(
            run_id="run-1",
            sequence=1,
            kind=EventKind.TASK_STARTED,
            payload={
                "objective": "Fix a concurrency bug",
                "phase": "discovery",
                "current_model_id": current_model,
                "budget_total_usd": 10,
                "workspace_id": "sandbox-1",
                "working_directory": "/workspace",
                "forecast_remaining_tokens": 200_000,
            },
        ),
    )
    return reducer, state


def apply(reducer, state, sequence: int, kind: EventKind, **payload: object):
    return reducer.apply(
        state,
        SupervisorEvent(
            run_id=state.run_id,
            sequence=sequence,
            kind=kind,
            payload=payload,
        ),
    )


def policy(threshold: float = 0.70) -> CompletionConstrainedPolicy:
    return CompletionConstrainedPolicy(
        estimator=HeuristicCompletionEstimator(),
        reliability_threshold=threshold,
        minimum_switch_savings_usd=0,
    )


def test_uncertain_discovery_stays_on_frontier() -> None:
    _, state = build_state()

    decision = policy().decide(state, MODELS)

    assert decision.target_model_id == "frontier"
    assert decision.action == RoutingAction.STAY
    assert decision.threshold_met is True


def test_committed_plan_allows_step_down_to_worker() -> None:
    reducer, state = build_state()
    state = apply(
        reducer,
        state,
        2,
        EventKind.PLAN_COMMITTED,
        summary="Apply a localized lock-order fix",
    )
    state = apply(reducer, state, 3, EventKind.PHASE_CHANGED, phase="implementation")

    decision = policy().decide(state, MODELS)

    assert decision.target_model_id == "worker"
    assert decision.action == RoutingAction.SWITCH_DOWN


def test_regression_and_stall_escalate_worker_to_frontier() -> None:
    reducer, state = build_state(current_model="worker")
    state = apply(
        reducer,
        state,
        2,
        EventKind.PLAN_COMMITTED,
        summary="Apply a localized lock-order fix",
    )
    state = apply(reducer, state, 3, EventKind.PHASE_CHANGED, phase="implementation")
    state = apply(
        reducer,
        state,
        4,
        EventKind.VALIDATION_RESULT,
        passed=10,
        failed=2,
    )
    state = apply(
        reducer,
        state,
        5,
        EventKind.VALIDATION_RESULT,
        passed=8,
        failed=4,
    )
    state = apply(
        reducer,
        state,
        6,
        EventKind.TOOL_RESULT,
        success=False,
        made_progress=False,
    )

    decision = policy().decide(state, MODELS)

    assert decision.target_model_id == "frontier"
    assert decision.action == RoutingAction.SWITCH_UP


def test_when_nothing_meets_threshold_choose_most_reliable_affordable_model() -> None:
    _, state = build_state(current_model="worker")

    decision = policy(threshold=0.99).decide(state, MODELS)

    assert decision.target_model_id == "frontier"
    assert decision.threshold_met is False
    assert "most reliable affordable" in decision.reason


def test_budget_constraint_prevents_an_unaffordable_frontier_fallback() -> None:
    _, state = build_state(current_model="worker")
    state.budget.total_usd = 0.10
    state.budget.reserved_usd = 0.05

    decision = policy().decide(state, MODELS)

    assert decision.target_model_id == "worker"
    assert decision.action == RoutingAction.STAY
    assert decision.threshold_met is False
    assert "most reliable affordable" in decision.reason


def test_policy_halts_when_no_model_fits_the_remaining_budget() -> None:
    _, state = build_state(current_model="worker")
    state.budget.total_usd = 0.01

    decision = policy().decide(state, MODELS)

    assert decision.action == RoutingAction.HALT_BUDGET
    assert decision.target_model_id == "worker"
    assert "no model has a forecast cost" in decision.reason
