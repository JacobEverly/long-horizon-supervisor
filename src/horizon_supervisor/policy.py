from __future__ import annotations

from horizon_supervisor.estimator import CompletionEstimator
from horizon_supervisor.models import (
    ModelProfile,
    ProgressTrend,
    RoutingAction,
    RoutingDecision,
    SupervisorState,
)


class CompletionConstrainedPolicy:
    """Choose the cheapest model that clears a completion-reliability bar."""

    def __init__(
        self,
        estimator: CompletionEstimator,
        reliability_threshold: float = 0.70,
        minimum_switch_savings_usd: float = 0.01,
    ) -> None:
        if not 0 <= reliability_threshold <= 1:
            raise ValueError("reliability_threshold must be between 0 and 1")
        if minimum_switch_savings_usd < 0:
            raise ValueError("minimum_switch_savings_usd must be non-negative")
        self.estimator = estimator
        self.reliability_threshold = reliability_threshold
        self.minimum_switch_savings_usd = minimum_switch_savings_usd

    def decide(
        self,
        state: SupervisorState,
        models: list[ModelProfile],
    ) -> RoutingDecision:
        if not models:
            raise ValueError("at least one model profile is required")
        profiles = {profile.model_id: profile for profile in models}
        if state.current_model_id not in profiles:
            raise ValueError("current model is missing from available model profiles")

        estimates = self.estimator.estimate(state, models)
        by_id = {estimate.model_id: estimate for estimate in estimates}
        recovery_can_use_reserve = state.progress_trend in {
            ProgressTrend.REGRESSING,
            ProgressTrend.STALLED,
        }
        spend_limit = (
            state.budget.remaining_usd if recovery_can_use_reserve else state.budget.available_usd
        )
        affordable = [
            estimate for estimate in estimates if estimate.forecast_cost_usd <= spend_limit
        ]
        eligible = [
            estimate
            for estimate in affordable
            if estimate.reliability_score >= self.reliability_threshold
        ]

        if eligible:
            chosen = min(
                eligible,
                key=lambda estimate: (estimate.forecast_cost_usd, -estimate.reliability_score),
            )
            reason = "cheapest model clearing reliability threshold and available budget"
            threshold_met = True
        elif affordable:
            chosen = max(
                affordable,
                key=lambda estimate: (estimate.reliability_score, -estimate.forecast_cost_usd),
            )
            reason = (
                "no affordable model clears reliability threshold; choose most reliable affordable"
            )
            threshold_met = False
        else:
            chosen = by_id[state.current_model_id]
            return RoutingDecision(
                action=RoutingAction.HALT_BUDGET,
                current_model_id=state.current_model_id,
                target_model_id=state.current_model_id,
                reliability_threshold=self.reliability_threshold,
                threshold_met=False,
                chosen_estimate=chosen,
                estimates=sorted(estimates, key=lambda estimate: profiles[estimate.model_id].tier),
                reason="no model has a forecast cost within the remaining spend limit",
            )

        current = by_id[state.current_model_id]
        if (
            chosen.model_id != current.model_id
            and current.reliability_score >= self.reliability_threshold
            and current.forecast_cost_usd <= spend_limit
            and current.forecast_cost_usd - chosen.forecast_cost_usd
            < self.minimum_switch_savings_usd
        ):
            chosen = current
            threshold_met = True
            reason = "stay because forecast savings do not cover the switching floor"

        action = self._action(
            current=profiles[state.current_model_id],
            target=profiles[chosen.model_id],
        )
        return RoutingDecision(
            action=action,
            current_model_id=state.current_model_id,
            target_model_id=chosen.model_id,
            reliability_threshold=self.reliability_threshold,
            threshold_met=threshold_met,
            chosen_estimate=chosen,
            estimates=sorted(estimates, key=lambda estimate: profiles[estimate.model_id].tier),
            reason=reason,
        )

    @staticmethod
    def _action(current: ModelProfile, target: ModelProfile) -> RoutingAction:
        if current.model_id == target.model_id:
            return RoutingAction.STAY
        if target.tier > current.tier:
            return RoutingAction.SWITCH_UP
        if target.tier < current.tier:
            return RoutingAction.SWITCH_DOWN
        return RoutingAction.SWITCH_LATERAL
