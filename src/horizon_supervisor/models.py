from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RunPhase(StrEnum):
    DISCOVERY = "discovery"
    PLANNING = "planning"
    IMPLEMENTATION = "implementation"
    VALIDATION = "validation"
    RECOVERY = "recovery"
    COMPLETE = "complete"


class ProgressTrend(StrEnum):
    UNKNOWN = "unknown"
    IMPROVING = "improving"
    STABLE = "stable"
    REGRESSING = "regressing"
    STALLED = "stalled"


class EventKind(StrEnum):
    TASK_STARTED = "task_started"
    PHASE_CHANGED = "phase_changed"
    PLAN_COMMITTED = "plan_committed"
    TOOL_RESULT = "tool_result"
    FILES_CHANGED = "files_changed"
    VALIDATION_RESULT = "validation_result"
    MILESTONE_COMPLETED = "milestone_completed"
    CONTEXT_COMPACTED = "context_compacted"
    MODEL_SELECTED = "model_selected"
    TASK_FINISHED = "task_finished"


class RoutingAction(StrEnum):
    STAY = "stay"
    SWITCH_UP = "switch_up"
    SWITCH_DOWN = "switch_down"
    SWITCH_LATERAL = "switch_lateral"
    HALT_BUDGET = "halt_budget"


class RecoveryAction(StrEnum):
    CONTINUE = "continue"
    ROLLBACK = "rollback"
    RESTART_CLEAN = "restart_clean"
    STOP_SUCCESS = "stop_success"


class CheckpointKind(StrEnum):
    CLEAN_BASE = "clean_base"
    VERIFIED = "verified"
    TURN = "turn"


class EvidenceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    workspace_digest: str
    passed: bool
    fresh_in_current_run: bool = False


class RecoveryCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checkpoint_id: str
    workspace_digest: str
    kind: CheckpointKind
    public_test_verified: bool = False
    sequence: int = Field(default=0, ge=0)


class RecoveryObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    current_workspace_digest: str
    current_workspace_dirty: bool
    completion_requested: bool = False
    final_verifier_failed: bool = False
    evidence: list[EvidenceRecord] = Field(default_factory=list)
    checkpoints: list[RecoveryCheckpoint] = Field(default_factory=list)


class RecoveryDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: RecoveryAction
    target_checkpoint_id: str | None = None
    current_completion_verified: bool
    reason: str


class WorkspaceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace_id: str
    working_directory: str
    git_head: str | None = None
    diff_hash: str | None = None
    changed_files: list[str] = Field(default_factory=list)


class BudgetState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_usd: float = Field(ge=0)
    spent_usd: float = Field(default=0, ge=0)
    reserved_usd: float = Field(default=0, ge=0)

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.total_usd - self.spent_usd)

    @property
    def available_usd(self) -> float:
        return max(0.0, self.remaining_usd - self.reserved_usd)


class ValidationSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    suite: str = "default"
    passed: int = Field(ge=0)
    failed: int = Field(ge=0)
    skipped: int = Field(default=0, ge=0)


class SupervisorState(BaseModel):
    """Portable state used by estimators and routing policies.

    The workspace is referenced, not copied. Framework-native objects and raw
    filesystem contents must not be placed in this model.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str
    objective: str
    phase: RunPhase
    current_model_id: str
    workspace: WorkspaceRef
    budget: BudgetState
    forecast_remaining_tokens: int = Field(default=100_000, ge=0)
    handoff_tokens: int = Field(default=2_000, ge=0)

    event_count: int = Field(default=0, ge=0)
    has_committed_plan: bool = False
    plan_summary: str | None = None
    completed_milestones: list[str] = Field(default_factory=list)
    active_milestone: str | None = None
    validation_history: list[ValidationSnapshot] = Field(default_factory=list)
    progress_trend: ProgressTrend = ProgressTrend.UNKNOWN
    consecutive_unproductive_steps: int = Field(default=0, ge=0)
    consecutive_errors: int = Field(default=0, ge=0)
    steps_since_model_switch: int = Field(default=0, ge=0)
    context_revision: int = Field(default=0, ge=0)
    context_summary: str | None = None
    finished: bool = False
    succeeded: bool | None = None


_REQUIRED_PAYLOAD_FIELDS: dict[EventKind, set[str]] = {
    EventKind.TASK_STARTED: {
        "objective",
        "phase",
        "current_model_id",
        "budget_total_usd",
        "workspace_id",
        "working_directory",
    },
    EventKind.PHASE_CHANGED: {"phase"},
    EventKind.PLAN_COMMITTED: {"summary"},
    EventKind.TOOL_RESULT: {"success", "made_progress"},
    EventKind.FILES_CHANGED: {"changed_files"},
    EventKind.VALIDATION_RESULT: {"passed", "failed"},
    EventKind.MILESTONE_COMPLETED: {"milestone"},
    EventKind.CONTEXT_COMPACTED: {"summary"},
    EventKind.MODEL_SELECTED: {"model_id"},
    EventKind.TASK_FINISHED: {"success"},
}


class SupervisorEvent(BaseModel):
    """Small live event envelope emitted by a harness adapter."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    sequence: int = Field(ge=1)
    kind: EventKind
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_required_payload(self) -> SupervisorEvent:
        missing = _REQUIRED_PAYLOAD_FIELDS[self.kind] - self.payload.keys()
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"{self.kind.value} payload is missing: {names}")
        return self


class ModelProfile(BaseModel):
    """A deployment profile, not a claim about a model family's global rank."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    tier: int = Field(ge=0)
    capability_score: float = Field(ge=0, le=1)
    cost_per_million_tokens_usd: float = Field(ge=0)
    token_multiplier: float = Field(default=1.0, gt=0)
    context_window_tokens: int = Field(default=128_000, gt=0)


class ModelCalibrationPrior(BaseModel):
    """Pre-run evidence available when a candidate model is scored.

    The prior must be computed only from trials that finished before the trial
    being predicted. This keeps the current trial's reward, cost, and latency
    on the target side of the dataset boundary.
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["catalog_only", "calibrated"] = "catalog_only"
    as_of: datetime | None = None
    task_count: int = Field(default=0, ge=0)
    verified_attempts: int = Field(default=0, ge=0)
    success_rate: float | None = Field(default=None, ge=0, le=1)
    mean_cost_usd: float | None = Field(default=None, ge=0)
    median_latency_seconds: float | None = Field(default=None, ge=0)
    source_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def catalog_only_has_no_empirical_values(self) -> ModelCalibrationPrior:
        empirical_values = (
            self.as_of,
            self.success_rate,
            self.mean_cost_usd,
            self.median_latency_seconds,
        )
        if self.status == "catalog_only" and (
            self.task_count
            or self.verified_attempts
            or self.source_ids
            or any(value is not None for value in empirical_values)
        ):
            raise ValueError("catalog_only prior cannot contain empirical evidence")
        if self.status == "calibrated" and (
            self.as_of is None or self.verified_attempts == 0 or not self.source_ids
        ):
            raise ValueError(
                "calibrated prior requires an as_of time, attempts, and sources"
            )
        return self


class CandidateModelFeatures(BaseModel):
    """Portable, pre-run features for scoring one model deployment.

    Model identity remains beside this object for provenance and optional
    deployment residuals. It is intentionally absent here so the cold-start
    scorer can evaluate a newly added model without adding a new output class.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["candidate-model-features.v0"] = (
        "candidate-model-features.v0"
    )
    context_window_tokens: int = Field(gt=0)
    input_usd_per_million_tokens: float = Field(ge=0)
    output_usd_per_million_tokens: float = Field(ge=0)
    cached_input_usd_per_million_tokens: float = Field(ge=0)
    cache_write_input_usd_per_million_tokens: float = Field(ge=0)
    supports_tool_use: bool
    reasoning_effort: str
    max_output_tokens: int = Field(gt=0)
    max_turns: int = Field(gt=0)
    request_timeout_seconds: int | None = Field(default=None, ge=60)
    request_retry_attempts: int | None = Field(default=None, ge=1)
    output_length_retry_attempts: int | None = Field(default=None, ge=0)
    catalog_source: str
    catalog_captured_at: datetime
    calibration: ModelCalibrationPrior = Field(default_factory=ModelCalibrationPrior)


class ModelEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    reliability_score: float = Field(ge=0, le=1)
    forecast_cost_usd: float = Field(ge=0)
    required_capability: float = Field(ge=0, le=1)
    reasons: list[str] = Field(default_factory=list)


class RoutingDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: RoutingAction
    current_model_id: str
    target_model_id: str
    reliability_threshold: float = Field(ge=0, le=1)
    threshold_met: bool
    chosen_estimate: ModelEstimate
    estimates: list[ModelEstimate]
    reason: str
