from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StuckStatus(StrEnum):
    HEALTHY = "HEALTHY"
    SUSPECTED_STUCK = "SUSPECTED_STUCK"


class PublicTestObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command_fingerprint: str
    passed: int = Field(ge=0)
    failed: int = Field(ge=0)
    failure_fingerprints: tuple[str, ...] = ()


class TurnObservation(BaseModel):
    """Information available immediately after one agent turn.

    The schema intentionally has no verifier reward, eventual success, sibling
    outcome, or private-reasoning field. ``assistant_action_summary`` may contain
    only the parsed plan/action surface returned to the harness.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["stuck-turn-observation.v0"] = "stuck-turn-observation.v0"
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
    actionable_next_step: bool = True
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cached_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    output_token_budget: int = Field(gt=0)
    spent_usd: float = Field(default=0, ge=0)
    spend_budget_usd: float = Field(gt=0)

    @model_validator(mode="after")
    def turn_within_limit(self) -> TurnObservation:
        if self.turn > self.max_turns:
            raise ValueError("turn cannot exceed max_turns")
        return self


class TurnAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turn: int
    status: StuckStatus
    meaningful_progress: bool
    progress_reasons: tuple[str, ...] = ()
    active_signals: tuple[str, ...] = ()
    persistent_signals: tuple[str, ...] = ()
    immediate_signal: str | None = None
    rule: str


class SuspectedStuckV0:
    """Frozen, outcome-blind stuck detector for the first matched-state pilot."""

    schema_version = "suspected-stuck.v0"
    persistence_turns = 2
    independent_signal_threshold = 2
    no_progress_turns = 2
    substantial_budget_fraction = 0.60
    workspace_cycle_lookback = 4

    _ERROR_PATTERN = re.compile(
        r"(?im)^.*(?:error|exception|failed|failure|traceback|segmentation fault|"
        r"command not found|permission denied|no such file).*$"
    )
    _SUCCESS_PATTERN = re.compile(
        r"(?im)(?:build succeeded|build successful|compiled successfully|"
        r"all tests passed|\b0 failed\b|server (?:started|listening)|"
        r"successfully (?:built|installed|generated|created))"
    )

    def __init__(self) -> None:
        self._observations: list[TurnObservation] = []
        self._assessments: list[TurnAssessment] = []
        self._seen_errors: Counter[str] = Counter()

    @staticmethod
    def normalize_command(command: str) -> str:
        text = re.sub(r"\s+", " ", command.strip().lower())
        text = re.sub(r"(?:/tmp|/var/tmp)/[\w./-]+", "<tmp>", text)
        text = re.sub(r"\b\d{5,}\b", "<n>", text)
        return text

    @classmethod
    def command_fingerprint(cls, commands: tuple[str, ...]) -> str:
        normalized = [cls.normalize_command(command) for command in commands]
        payload = json.dumps(normalized, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    @classmethod
    def error_fingerprints(cls, terminal_tail: str) -> tuple[str, ...]:
        normalized = []
        for match in cls._ERROR_PATTERN.findall(terminal_tail[-12_000:]):
            line = re.sub(r"\s+", " ", match.strip().lower())
            line = re.sub(r"line \d+", "line <n>", line)
            line = re.sub(r"0x[0-9a-f]+", "<addr>", line)
            line = re.sub(r"(?:/tmp|/var/tmp)/[\w./-]+", "<tmp>", line)
            line = re.sub(r"[^a-z0-9_<>./ -]+", " ", line)
            line = re.sub(r"\s+", " ", line).strip()
            if line:
                normalized.append(hashlib.sha256(line.encode()).hexdigest()[:16])
        return tuple(sorted(set(normalized)))

    @classmethod
    def looks_like_successful_milestone(cls, terminal_tail: str) -> bool:
        return bool(cls._SUCCESS_PATTERN.search(terminal_tail[-12_000:]))

    def observe(self, observation: TurnObservation) -> TurnAssessment:
        consecutive_with_prior = True
        if self._observations:
            prior = self._observations[-1]
            if observation.run_id != prior.run_id:
                raise ValueError("a detector instance cannot mix run ids")
            if observation.turn <= prior.turn:
                raise ValueError("turn observations must be strictly increasing")
            consecutive_with_prior = observation.turn == prior.turn + 1

        progress_reasons = self._meaningful_progress_reasons(observation)
        meaningful_progress = bool(progress_reasons)
        active = self._active_signals(
            observation, meaningful_progress, consecutive_with_prior
        )

        immediate = self._immediate_signal(
            observation, meaningful_progress, consecutive_with_prior
        )
        persistent: tuple[str, ...] = ()
        if self._assessments and consecutive_with_prior:
            persistent = tuple(sorted(set(active) & set(self._assessments[-1].active_signals)))

        if immediate:
            status = StuckStatus.SUSPECTED_STUCK
            rule = f"immediate:{immediate}"
        elif len(persistent) >= self.independent_signal_threshold:
            status = StuckStatus.SUSPECTED_STUCK
            rule = "two_independent_signals_for_two_consecutive_turns"
        else:
            status = StuckStatus.HEALTHY
            rule = "threshold_not_met"

        assessment = TurnAssessment(
            turn=observation.turn,
            status=status,
            meaningful_progress=meaningful_progress,
            progress_reasons=progress_reasons,
            active_signals=active,
            persistent_signals=persistent,
            immediate_signal=immediate,
            rule=rule,
        )
        self._observations.append(observation)
        self._assessments.append(assessment)
        self._seen_errors.update(self.error_fingerprints(observation.terminal_tail))
        return assessment

    def _meaningful_progress_reasons(
        self, observation: TurnObservation
    ) -> tuple[str, ...]:
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
        previous_errors = set(self.error_fingerprints(previous.terminal_tail))
        current_errors = set(self.error_fingerprints(observation.terminal_tail))
        if previous_errors and not current_errors:
            reasons.append("previously_observed_error_resolved")
        return tuple(reasons)

    def _active_signals(
        self,
        observation: TurnObservation,
        meaningful_progress: bool,
        consecutive_with_prior: bool,
    ) -> tuple[str, ...]:
        signals: set[str] = set()
        previous = self._observations[-1] if self._observations else None
        current_errors = set(self.error_fingerprints(observation.terminal_tail))

        if previous:
            previous_errors = set(self.error_fingerprints(previous.terminal_tail))
            if current_errors and current_errors.intersection(previous_errors):
                signals.add("repeated_error")

            current_tests = observation.public_tests
            previous_tests = previous.public_tests
            if (
                current_tests
                and previous_tests
                and current_tests.failed > 0
                and current_tests.failure_fingerprints
                == previous_tests.failure_fingerprints
                and current_tests.passed == previous_tests.passed
            ):
                signals.add("unchanged_failing_public_tests")

            if (
                observation.workspace_digest == previous.workspace_digest
                and not meaningful_progress
            ):
                signals.add("unchanged_execution_state")

            if (
                self.command_fingerprint(observation.commands)
                == self.command_fingerprint(previous.commands)
                and observation.commands
                and not meaningful_progress
            ):
                signals.add("repeated_equivalent_commands")

        history = self._observations[-self.workspace_cycle_lookback :]
        older_digests = {item.workspace_digest for item in history[:-1]}
        if observation.workspace_digest in older_digests and not meaningful_progress:
            signals.add("workspace_state_cycle")

        recent_progress = [item.meaningful_progress for item in self._assessments[-1:]]
        if (
            consecutive_with_prior
            and not meaningful_progress
            and recent_progress == [False]
        ):
            signals.add("consecutive_turns_without_meaningful_progress")

        output_fraction = observation.output_tokens / observation.output_token_budget
        spend_fraction = observation.spent_usd / observation.spend_budget_usd
        if (
            max(output_fraction, spend_fraction) >= self.substantial_budget_fraction
            and not meaningful_progress
        ):
            signals.add("substantial_budget_without_new_evidence")

        if not observation.actionable_next_step:
            signals.add("no_actionable_next_step")

        if observation.protocol_failure:
            signals.add("protocol_failure")

        return tuple(sorted(signals))

    def _immediate_signal(
        self,
        observation: TurnObservation,
        meaningful_progress: bool,
        consecutive_with_prior: bool,
    ) -> str | None:
        if observation.protocol_failure:
            return "protocol_failure"
        if not self._observations or meaningful_progress or not consecutive_with_prior:
            return None
        previous = self._observations[-1]
        same_commands = (
            observation.commands
            and self.command_fingerprint(observation.commands)
            == self.command_fingerprint(previous.commands)
        )
        same_errors = bool(
            set(self.error_fingerprints(observation.terminal_tail))
            & set(self.error_fingerprints(previous.terminal_tail))
        )
        unchanged_state = observation.workspace_digest == previous.workspace_digest
        if same_commands and (same_errors or unchanged_state):
            return "clear_action_error_loop"
        return None

    @classmethod
    def frozen_spec(cls) -> dict[str, object]:
        return {
            "schema_version": cls.schema_version,
            "decision_rule": (
                "SUSPECTED_STUCK when a protocol failure or clear repeated action/error "
                "loop occurs immediately, or when at least two independent signals are "
                "active on two consecutive turns"
            ),
            "persistence_turns": cls.persistence_turns,
            "independent_signal_threshold": cls.independent_signal_threshold,
            "observation_gap_handling": (
                "Command-bearing observations may skip agent-turn indices; a gap "
                "breaks consecutive persistence and immediate-loop evidence."
            ),
            "no_progress_turns": cls.no_progress_turns,
            "substantial_budget_fraction": cls.substantial_budget_fraction,
            "workspace_cycle_lookback": cls.workspace_cycle_lookback,
            "meaningful_progress": [
                "additional public tests passing",
                "fewer distinct public-test failures",
                "new successful build/execution milestone",
                "resolution of a previously observed error",
                "creation of a required artifact",
            ],
            "explicit_non_progress": [
                "file or digest change alone",
                "more commands or tokens without new public evidence",
            ],
            "forbidden_inputs": [
                "hidden verifier output",
                "future trajectory information",
                "final task success",
                "private model reasoning",
                "sibling branch outcomes",
                "post-hoc manual labels",
            ],
        }
