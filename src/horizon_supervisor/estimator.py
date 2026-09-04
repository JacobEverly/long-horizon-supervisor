from __future__ import annotations

import math
from typing import Protocol

from horizon_supervisor.models import (
    ModelEstimate,
    ModelProfile,
    ProgressTrend,
    RunPhase,
    SupervisorState,
)


class CompletionEstimator(Protocol):
    """Interface implemented by both heuristic and trained estimators."""

    def estimate(
        self,
        state: SupervisorState,
        models: list[ModelProfile],
    ) -> list[ModelEstimate]: ...


class HeuristicCompletionEstimator:
    """Transparent baseline estimator.

    Scores are useful for exercising the policy contract, but they are not
    calibrated completion probabilities. A trained estimator will replace this
    class without changing the policy interface.
    """

    _PHASE_REQUIREMENT = {
        RunPhase.DISCOVERY: 0.78,
        RunPhase.PLANNING: 0.82,
        RunPhase.IMPLEMENTATION: 0.54,
        RunPhase.VALIDATION: 0.52,
        RunPhase.RECOVERY: 0.88,
        RunPhase.COMPLETE: 0.0,
    }

    def estimate(
        self,
        state: SupervisorState,
        models: list[ModelProfile],
    ) -> list[ModelEstimate]:
        requirement, reasons = self._required_capability(state)
        estimates = []
        for model in models:
            reliability = self._logistic(8 * (model.capability_score - requirement))
            token_forecast = state.forecast_remaining_tokens * model.token_multiplier
            if model.model_id != state.current_model_id:
                token_forecast += state.handoff_tokens
            forecast_cost = token_forecast * model.cost_per_million_tokens_usd / 1_000_000
            estimates.append(
                ModelEstimate(
                    model_id=model.model_id,
                    reliability_score=round(reliability, 4),
                    forecast_cost_usd=round(forecast_cost, 6),
                    required_capability=round(requirement, 4),
                    reasons=reasons,
                )
            )
        return estimates

    def _required_capability(self, state: SupervisorState) -> tuple[float, list[str]]:
        requirement = self._PHASE_REQUIREMENT[state.phase]
        reasons = [f"{state.phase.value} phase starts at {requirement:.2f} capability"]

        if state.has_committed_plan and state.phase in {
            RunPhase.IMPLEMENTATION,
            RunPhase.VALIDATION,
        }:
            requirement -= 0.16
            reasons.append("committed plan lowers execution uncertainty")

        trend_adjustment = {
            ProgressTrend.UNKNOWN: 0.0,
            ProgressTrend.IMPROVING: -0.10,
            ProgressTrend.STABLE: 0.02,
            ProgressTrend.REGRESSING: 0.20,
            ProgressTrend.STALLED: 0.24,
        }[state.progress_trend]
        requirement += trend_adjustment
        if trend_adjustment:
            reasons.append(
                f"{state.progress_trend.value} progress changes requirement by "
                f"{trend_adjustment:+.2f}"
            )

        error_adjustment = min(0.15, state.consecutive_errors * 0.05)
        if error_adjustment:
            requirement += error_adjustment
            reasons.append(f"repeated errors add {error_adjustment:.2f}")

        churn_adjustment = min(0.12, state.consecutive_unproductive_steps * 0.03)
        if churn_adjustment:
            requirement += churn_adjustment
            reasons.append(f"unproductive steps add {churn_adjustment:.2f}")

        return max(0.0, min(1.0, requirement)), reasons

    @staticmethod
    def _logistic(value: float) -> float:
        return 1.0 / (1.0 + math.exp(-value))
