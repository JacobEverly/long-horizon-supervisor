from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from horizon_supervisor.stuck_detector import (
    PublicTestObservation,
    SuspectedStuckV0,
    TurnObservation,
)
from horizon_supervisor.stuck_detector_v1 import ActionMode, SuspectedStuckV1


class TwoTierStatus(StrEnum):
    HEALTHY = "HEALTHY"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    CONFIRMED_STUCK = "CONFIRMED_STUCK"
    STRUCTURAL_FAILURE = "STRUCTURAL_FAILURE"


class TwoTierDetectorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    review_minimum_turn: int = Field(ge=2)
    review_signal_threshold: int = Field(ge=1)
    confirmation_minimum_turn: int = Field(ge=3)
    confirmation_window: int = Field(ge=2)
    confirmation_failure_turns: int = Field(ge=1)
    confirmation_productive_turns: int = Field(ge=1)
    minimum_remaining_turns: int = Field(ge=1)

    @model_validator(mode="after")
    def valid_thresholds(self) -> TwoTierDetectorConfig:
        if self.confirmation_minimum_turn <= self.review_minimum_turn:
            raise ValueError("confirmation must occur after review becomes eligible")
        if self.confirmation_failure_turns > self.confirmation_window:
            raise ValueError("failure-turn requirement exceeds confirmation window")
        if self.confirmation_productive_turns > self.confirmation_window:
            raise ValueError("productive-turn requirement exceeds confirmation window")
        return self


FROZEN_CANDIDATE_FAMILY = (
    TwoTierDetectorConfig(
        name="review-t5-confirm-t6-w2-e2",
        review_minimum_turn=5,
        review_signal_threshold=2,
        confirmation_minimum_turn=6,
        confirmation_window=2,
        confirmation_failure_turns=2,
        confirmation_productive_turns=1,
        minimum_remaining_turns=2,
    ),
    TwoTierDetectorConfig(
        name="review-t5-confirm-t6-w3-e2",
        review_minimum_turn=5,
        review_signal_threshold=2,
        confirmation_minimum_turn=6,
        confirmation_window=3,
        confirmation_failure_turns=2,
        confirmation_productive_turns=1,
        minimum_remaining_turns=2,
    ),
    TwoTierDetectorConfig(
        name="review-t5-confirm-t7-w3-e2",
        review_minimum_turn=5,
        review_signal_threshold=2,
        confirmation_minimum_turn=7,
        confirmation_window=3,
        confirmation_failure_turns=2,
        confirmation_productive_turns=1,
        minimum_remaining_turns=2,
    ),
    TwoTierDetectorConfig(
        name="review-t6-confirm-t7-w3-e2",
        review_minimum_turn=6,
        review_signal_threshold=2,
        confirmation_minimum_turn=7,
        confirmation_window=3,
        confirmation_failure_turns=2,
        confirmation_productive_turns=1,
        minimum_remaining_turns=2,
    ),
)


class TwoTierObservation(BaseModel):
    """Full decision-time observation for the two-tier detector.

    The schema deliberately excludes final outcomes, verifier rewards, future
    observations, private reasoning, sibling results, and task identity.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["stuck-turn-observation.v2"] = "stuck-turn-observation.v2"
    run_id: str
    turn: int = Field(ge=1)
    max_turns: int = Field(gt=0)
    model_id: str
    commands: tuple[str, ...] = ()
    terminal_tail: str = ""
    workspace_digest: str
    public_tests: PublicTestObservation | None = None
    successful_milestones: tuple[str, ...] = ()
    required_artifacts: tuple[str, ...] = ()
    protocol_failure: bool = False
    provider_failure: bool = False
    harness_failure: bool = False
    actionable_next_step: bool = True
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cached_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    output_token_budget: int = Field(gt=0)
    spent_usd: float = Field(default=0, ge=0)
    spend_budget_usd: float = Field(gt=0)
    remaining_wall_seconds: float | None = Field(default=None, ge=0)
    task_category: str | None = None
    snapshot_reproducible: bool = True
    external_state_reproducible: bool = True

    @model_validator(mode="after")
    def turn_within_limit(self) -> TwoTierObservation:
        if self.turn > self.max_turns:
            raise ValueError("turn cannot exceed max_turns")
        return self

    @classmethod
    def from_v0(cls, observation: TurnObservation) -> TwoTierObservation:
        payload = observation.model_dump()
        payload["schema_version"] = "stuck-turn-observation.v2"
        return cls.model_validate(payload)


class TwoTierAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turn: int
    status: TwoTierStatus
    action_mode: ActionMode
    meaningful_progress: bool
    progress_reasons: tuple[str, ...] = ()
    active_signals: tuple[str, ...] = ()
    persistent_review_signals: tuple[str, ...] = ()
    review_eligible: bool
    confirmation_eligible: bool
    remaining_turns: int
    rule: str


class TwoTierStuckDetectorV2:
    """Outcome-blind review/confirmation detector for long-horizon runs."""

    schema_version = "two-tier-stuck-detector.v2"
    healthy_checkpoint_turn = 4
    substantial_budget_fraction = 0.60
    workspace_cycle_lookback = 4

    def __init__(self, config: TwoTierDetectorConfig) -> None:
        if config not in FROZEN_CANDIDATE_FAMILY:
            raise ValueError("config is not part of the frozen v2 candidate family")
        self.config = config
        self._observations: list[TwoTierObservation] = []
        self._assessments: list[TwoTierAssessment] = []

    def observe(self, observation: TwoTierObservation) -> TwoTierAssessment:
        consecutive = True
        if self._observations:
            previous = self._observations[-1]
            if observation.run_id != previous.run_id:
                raise ValueError("a detector instance cannot mix run ids")
            if observation.turn <= previous.turn:
                raise ValueError("turn observations must be strictly increasing")
            consecutive = observation.turn == previous.turn + 1

        action_mode = SuspectedStuckV1.classify_actions(observation.commands)
        progress_reasons = self._progress_reasons(observation)
        meaningful_progress = bool(progress_reasons)
        active_signals = self._active_signals(
            observation,
            action_mode=action_mode,
            meaningful_progress=meaningful_progress,
            consecutive=consecutive,
        )
        persistent_review_signals: tuple[str, ...] = ()
        if self._assessments and consecutive:
            persistent_review_signals = tuple(
                sorted(
                    set(active_signals).intersection(
                        self._assessments[-1].active_signals
                    )
                )
            )

        remaining_turns = observation.max_turns - observation.turn
        enough_budget = remaining_turns >= self.config.minimum_remaining_turns
        structural_reasons = self._structural_reasons(observation)
        review_eligible = (
            not structural_reasons
            and enough_budget
            and observation.turn >= self.config.review_minimum_turn
            and not meaningful_progress
            and (
                len(persistent_review_signals)
                >= self.config.review_signal_threshold
                or self._clear_action_error_loop(observation, consecutive)
            )
        )
        confirmation_eligible = (
            review_eligible
            and observation.turn >= self.config.confirmation_minimum_turn
            and self._assessments
            and self._assessments[-1].status
            in {TwoTierStatus.NEEDS_REVIEW, TwoTierStatus.CONFIRMED_STUCK}
            and self._confirmation_window_passes(observation)
        )

        if structural_reasons:
            status = TwoTierStatus.STRUCTURAL_FAILURE
            rule = "structural_failure:" + ",".join(structural_reasons)
        elif confirmation_eligible:
            status = TwoTierStatus.CONFIRMED_STUCK
            rule = "persistent_failed_productive_work_after_review"
        elif review_eligible:
            status = TwoTierStatus.NEEDS_REVIEW
            rule = "broad_persistent_non_progress_review"
        else:
            status = TwoTierStatus.HEALTHY
            rule = "threshold_not_met"

        assessment = TwoTierAssessment(
            turn=observation.turn,
            status=status,
            action_mode=action_mode,
            meaningful_progress=meaningful_progress,
            progress_reasons=progress_reasons,
            active_signals=active_signals,
            persistent_review_signals=persistent_review_signals,
            review_eligible=review_eligible,
            confirmation_eligible=confirmation_eligible,
            remaining_turns=remaining_turns,
            rule=rule,
        )
        self._observations.append(observation)
        self._assessments.append(assessment)
        return assessment

    @staticmethod
    def _structural_reasons(observation: TwoTierObservation) -> tuple[str, ...]:
        reasons: list[str] = []
        if observation.protocol_failure:
            reasons.append("protocol")
        if observation.provider_failure:
            reasons.append("provider")
        if observation.harness_failure:
            reasons.append("harness")
        if not observation.snapshot_reproducible:
            reasons.append("snapshot_not_reproducible")
        if not observation.external_state_reproducible:
            reasons.append("external_state_not_reproducible")
        return tuple(reasons)

    def _progress_reasons(self, observation: TwoTierObservation) -> tuple[str, ...]:
        if not self._observations:
            return ()
        previous = self._observations[-1]
        reasons: list[str] = []
        current_tests = observation.public_tests
        previous_tests = previous.public_tests
        if current_tests and previous_tests:
            if current_tests.passed > previous_tests.passed:
                reasons.append("additional_public_tests_passing")
            if current_tests.failed < previous_tests.failed:
                reasons.append("fewer_distinct_public_test_failures")
        if set(observation.successful_milestones) - set(previous.successful_milestones):
            reasons.append("new_successful_execution_milestone")
        if set(observation.required_artifacts) - set(previous.required_artifacts):
            reasons.append("required_artifact_created")
        previous_errors = set(SuspectedStuckV0.error_fingerprints(previous.terminal_tail))
        current_errors = set(SuspectedStuckV0.error_fingerprints(observation.terminal_tail))
        if previous_errors and not current_errors:
            reasons.append("previously_observed_error_resolved")
        return tuple(reasons)

    @staticmethod
    def _failure_evidence(observation: TwoTierObservation) -> bool:
        errors = SuspectedStuckV0.error_fingerprints(observation.terminal_tail)
        return bool(errors or (observation.public_tests and observation.public_tests.failed))

    def _active_signals(
        self,
        observation: TwoTierObservation,
        *,
        action_mode: ActionMode,
        meaningful_progress: bool,
        consecutive: bool,
    ) -> tuple[str, ...]:
        signals: set[str] = set()
        current_errors = set(SuspectedStuckV0.error_fingerprints(observation.terminal_tail))
        if action_mode == ActionMode.PRODUCTIVE and self._failure_evidence(observation):
            signals.add("productive_failure")
        if self._observations and consecutive:
            previous = self._observations[-1]
            previous_errors = set(
                SuspectedStuckV0.error_fingerprints(previous.terminal_tail)
            )
            if current_errors.intersection(previous_errors):
                signals.add("repeated_error")
            if (
                observation.public_tests
                and previous.public_tests
                and observation.public_tests.failed > 0
                and observation.public_tests.passed == previous.public_tests.passed
                and observation.public_tests.failure_fingerprints
                == previous.public_tests.failure_fingerprints
            ):
                signals.add("unchanged_failing_public_tests")
            if observation.workspace_digest == previous.workspace_digest:
                signals.add("unchanged_execution_state")
            if (
                observation.commands
                and SuspectedStuckV0.command_fingerprint(observation.commands)
                == SuspectedStuckV0.command_fingerprint(previous.commands)
            ):
                signals.add("repeated_equivalent_commands")
            if not meaningful_progress and not self._assessments[-1].meaningful_progress:
                signals.add("consecutive_turns_without_public_progress")

        history = self._observations[-self.workspace_cycle_lookback :]
        older_digests = {item.workspace_digest for item in history[:-1]}
        if observation.workspace_digest in older_digests and not meaningful_progress:
            signals.add("workspace_state_cycle")
        output_fraction = observation.output_tokens / observation.output_token_budget
        spend_fraction = observation.spent_usd / observation.spend_budget_usd
        if (
            max(output_fraction, spend_fraction) >= self.substantial_budget_fraction
            and not meaningful_progress
        ):
            signals.add("substantial_budget_without_public_progress")
        if not observation.actionable_next_step:
            signals.add("no_actionable_next_step")
        return tuple(sorted(signals))

    def _clear_action_error_loop(
        self, observation: TwoTierObservation, consecutive: bool
    ) -> bool:
        if not self._observations or not consecutive or not observation.commands:
            return False
        previous = self._observations[-1]
        return (
            SuspectedStuckV0.command_fingerprint(observation.commands)
            == SuspectedStuckV0.command_fingerprint(previous.commands)
            and bool(
                set(SuspectedStuckV0.error_fingerprints(observation.terminal_tail))
                & set(SuspectedStuckV0.error_fingerprints(previous.terminal_tail))
            )
        )

    def _confirmation_window_passes(self, current: TwoTierObservation) -> bool:
        window_size = self.config.confirmation_window
        prior_count = window_size - 1
        if len(self._observations) < prior_count:
            return False
        observations = [*self._observations[-prior_count:], current]
        assessments = [*self._assessments[-prior_count:]]
        expected_turns = list(range(current.turn - window_size + 1, current.turn + 1))
        if [item.turn for item in observations] != expected_turns:
            return False
        failure_turns = sum(self._failure_evidence(item) for item in observations)
        action_modes = [item.action_mode for item in assessments]
        action_modes.append(SuspectedStuckV1.classify_actions(current.commands))
        productive_turns = sum(mode == ActionMode.PRODUCTIVE for mode in action_modes)
        return (
            failure_turns >= self.config.confirmation_failure_turns
            and productive_turns >= self.config.confirmation_productive_turns
        )

    @classmethod
    def frozen_spec(cls, config: TwoTierDetectorConfig) -> dict[str, object]:
        return {
            "schema_version": cls.schema_version,
            "config": config.model_dump(),
            "candidate_family": [item.model_dump() for item in FROZEN_CANDIDATE_FAMILY],
            "states": [status.value for status in TwoTierStatus],
            "healthy_checkpoint": "turn 4 while status is HEALTHY",
            "needs_review": (
                "broad persistent non-progress or repeated action/error evidence after "
                "the configured minimum turn"
            ),
            "confirmed_stuck": (
                "a later needs-review state with repeated observable failures and at "
                "least one productive action in the frozen confirmation window"
            ),
            "structural_failure": (
                "protocol, provider, harness, snapshot, and external-state failures are "
                "never model-stuck labels"
            ),
            "forbidden_inputs": [
                "hidden verifier output",
                "terminal success",
                "future observations",
                "private model reasoning",
                "sibling outcomes",
                "task identity",
                "post-hoc intervention results",
            ],
        }
