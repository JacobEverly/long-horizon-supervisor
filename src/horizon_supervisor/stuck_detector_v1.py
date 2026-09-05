from __future__ import annotations

import re
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict

from horizon_supervisor.stuck_detector import (
    PublicTestObservation,
    SuspectedStuckV0,
    TurnObservation,
)


class StuckStatusV1(StrEnum):
    HEALTHY = "HEALTHY"
    SUSPECTED_STUCK = "SUSPECTED_STUCK"
    STRUCTURAL_FAILURE = "STRUCTURAL_FAILURE"


class ActionMode(StrEnum):
    NONE = "none"
    INSPECTION = "inspection"
    PRODUCTIVE = "productive"


class TurnAssessmentV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turn: int
    status: StuckStatusV1
    action_mode: ActionMode
    meaningful_progress: bool
    progress_reasons: tuple[str, ...] = ()
    active_signals: tuple[str, ...] = ()
    persistent_signals: tuple[str, ...] = ()
    rule: str


class SuspectedStuckV1:
    """Conservative, outcome-blind detector developed after the v0 coverage audit.

    V1 corrects one specific v0 failure: an unchanged workspace during legitimate
    investigation is not itself evidence that an agent is stuck. A stuck decision
    requires repeated failure evidence from productive work. Harness/protocol
    failures are surfaced separately and must never become model-stuck labels.
    """

    schema_version = "suspected-stuck.v1"
    minimum_stuck_turn = 6
    failure_persistence_turns = 2
    healthy_checkpoint_turn = 4
    substantial_budget_fraction = 0.60

    _INSPECTION_PATTERN = re.compile(
        r"^(?:pwd|ls(?:\s|$)|find(?:\s|$)|cat(?:\s|$)|head(?:\s|$)|tail(?:\s|$)|"
        r"sed\s+-n(?:\s|$)|rg(?:\s|$)|grep(?:\s|$)|file(?:\s|$)|which(?:\s|$)|"
        r"command\s+-v(?:\s|$)|strings(?:\s|$)|readelf(?:\s|$)|objdump(?:\s|$)|"
        r"git\s+(?:status|log|diff)(?:\s|$))",
        re.IGNORECASE,
    )

    def __init__(self) -> None:
        self._observations: list[TurnObservation] = []
        self._assessments: list[TurnAssessmentV1] = []

    @classmethod
    def classify_actions(cls, commands: tuple[str, ...]) -> ActionMode:
        if not commands:
            return ActionMode.NONE
        normalized = [SuspectedStuckV0.normalize_command(command) for command in commands]
        if all(cls._INSPECTION_PATTERN.match(command) for command in normalized):
            return ActionMode.INSPECTION
        return ActionMode.PRODUCTIVE

    def observe(self, observation: TurnObservation) -> TurnAssessmentV1:
        consecutive = True
        if self._observations:
            previous = self._observations[-1]
            if observation.run_id != previous.run_id:
                raise ValueError("a detector instance cannot mix run ids")
            if observation.turn <= previous.turn:
                raise ValueError("turn observations must be strictly increasing")
            consecutive = observation.turn == previous.turn + 1

        action_mode = self.classify_actions(observation.commands)
        progress_reasons = self._progress_reasons(observation)
        meaningful_progress = bool(progress_reasons)
        active = self._active_signals(
            observation,
            action_mode=action_mode,
            meaningful_progress=meaningful_progress,
            consecutive=consecutive,
        )
        persistent: tuple[str, ...] = ()
        if self._assessments and consecutive:
            persistent = tuple(
                sorted(set(active).intersection(self._assessments[-1].active_signals))
            )

        if observation.protocol_failure:
            status = StuckStatusV1.STRUCTURAL_FAILURE
            rule = "structural_failure:protocol"
        elif (
            observation.turn >= self.minimum_stuck_turn
            and "productive_failure_without_progress" in persistent
        ):
            status = StuckStatusV1.SUSPECTED_STUCK
            rule = "repeated_productive_failure_for_two_consecutive_turns"
        else:
            status = StuckStatusV1.HEALTHY
            rule = "threshold_not_met"

        assessment = TurnAssessmentV1(
            turn=observation.turn,
            status=status,
            action_mode=action_mode,
            meaningful_progress=meaningful_progress,
            progress_reasons=progress_reasons,
            active_signals=active,
            persistent_signals=persistent,
            rule=rule,
        )
        self._observations.append(observation)
        self._assessments.append(assessment)
        return assessment

    def _progress_reasons(self, observation: TurnObservation) -> tuple[str, ...]:
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

    def _active_signals(
        self,
        observation: TurnObservation,
        *,
        action_mode: ActionMode,
        meaningful_progress: bool,
        consecutive: bool,
    ) -> tuple[str, ...]:
        signals: set[str] = set()
        current_errors = set(SuspectedStuckV0.error_fingerprints(observation.terminal_tail))
        current_tests: PublicTestObservation | None = observation.public_tests

        failure_evidence = bool(current_errors)
        if current_tests and current_tests.failed > 0:
            failure_evidence = True
        if (
            action_mode == ActionMode.PRODUCTIVE
            and failure_evidence
            and not meaningful_progress
        ):
            signals.add("productive_failure_without_progress")

        if self._observations and consecutive:
            previous = self._observations[-1]
            previous_errors = set(
                SuspectedStuckV0.error_fingerprints(previous.terminal_tail)
            )
            if current_errors.intersection(previous_errors):
                signals.add("repeated_error")
            if (
                current_tests
                and previous.public_tests
                and current_tests.failed > 0
                and current_tests.passed == previous.public_tests.passed
                and current_tests.failure_fingerprints
                == previous.public_tests.failure_fingerprints
            ):
                signals.add("unchanged_failing_public_tests")

        output_fraction = observation.output_tokens / observation.output_token_budget
        spend_fraction = observation.spent_usd / observation.spend_budget_usd
        if (
            max(output_fraction, spend_fraction) >= self.substantial_budget_fraction
            and failure_evidence
            and not meaningful_progress
        ):
            signals.add("substantial_budget_with_unresolved_failure")
        if not observation.actionable_next_step:
            signals.add("no_actionable_next_step")
        return tuple(sorted(signals))

    @classmethod
    def frozen_spec(cls) -> dict[str, object]:
        return {
            "schema_version": cls.schema_version,
            "decision_rule": (
                "SUSPECTED_STUCK at or after turn 6 only when productive failure "
                "evidence persists for two consecutive observed turns; read-only "
                "inspection and unchanged workspace state are insufficient"
            ),
            "minimum_stuck_turn": cls.minimum_stuck_turn,
            "failure_persistence_turns": cls.failure_persistence_turns,
            "healthy_checkpoint_turn": cls.healthy_checkpoint_turn,
            "substantial_budget_fraction": cls.substantial_budget_fraction,
            "structural_failure_rule": (
                "protocol failures are STRUCTURAL_FAILURE, never SUSPECTED_STUCK"
            ),
            "required_stuck_evidence": [
                "productive action",
                "observable error or failing public test",
                "no new public progress",
                "two consecutive observed turns",
            ],
            "explicitly_insufficient": [
                "unchanged workspace during inspection",
                "workspace digest cycle alone",
                "token use alone",
                "protocol failure",
            ],
            "forbidden_inputs": [
                "hidden verifier output",
                "future trajectory information",
                "final task success",
                "private model reasoning",
                "sibling branch outcomes",
                "task identity",
            ],
        }


class ProjectedCheckpointObservation(BaseModel):
    """Development-only projection used to evaluate v1 without inventing features."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["suspected-stuck-v1-projection.v0"] = (
        "suspected-stuck-v1-projection.v0"
    )
    turn: int
    error_signal_count: int
    pass_signal_count: int
