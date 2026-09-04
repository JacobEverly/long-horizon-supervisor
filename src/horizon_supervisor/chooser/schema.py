from __future__ import annotations

import hashlib
import re
import unicodedata
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DatasetSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    QUARANTINE = "quarantine"


class Difficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Ambiguity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TaskDomain(StrEnum):
    SOFTWARE_MAINTENANCE = "software_maintenance"
    ALGORITHMIC_PROBLEM_SOLVING = "algorithmic_problem_solving"
    DATA_PROCESSING = "data_processing"
    TESTING = "testing"
    SYSTEMS = "systems"
    ARCHITECTURE_DESIGN = "architecture_design"
    RESEARCH = "research"
    GENERAL = "general"


class LabelMethod(StrEnum):
    DATASET_METADATA = "dataset_metadata"
    NORMALIZED_RULE = "normalized_rule"
    HUMAN_ANNOTATION = "human_annotation"
    EMPIRICAL_ROLLOUT = "empirical_rollout"
    TEACHER_MODEL = "teacher_model"


class OutcomeProvenance(StrEnum):
    IMMUTABLE = "immutable"
    DATASET_DECLARED_UNVERSIONED = "dataset_declared_unversioned"


class LabelEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    method: LabelMethod
    method_version: str
    native_value: str | int | float | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    note: str | None = None


class DifficultyTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: Difficulty
    evidence: LabelEvidence


class AmbiguityTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: Ambiguity
    evidence: LabelEvidence


class DomainTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: TaskDomain
    evidence: LabelEvidence


class ModelOutcomeAggregate(BaseModel):
    """Observed outcomes for one immutable model deployment and agent configuration."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    deployment_id: str
    agent_id: str
    verifier_id: str
    attempts: int = Field(ge=1)
    successes: int = Field(ge=0)
    mean_input_tokens: float | None = Field(default=None, ge=0)
    mean_output_tokens: float | None = Field(default=None, ge=0)
    mean_cost_usd: float | None = Field(default=None, ge=0)
    mean_latency_seconds: float | None = Field(default=None, ge=0)
    source_id: str
    provenance: OutcomeProvenance = OutcomeProvenance.IMMUTABLE
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def successes_cannot_exceed_attempts(self) -> ModelOutcomeAggregate:
        if self.successes > self.attempts:
            raise ValueError("successes cannot exceed attempts")
        return self

    @property
    def observed_success_rate(self) -> float:
        return self.successes / self.attempts


class SourceReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    dataset_id: str
    revision: str
    config: str
    source_split: str
    record_id: str
    row_index: int | None = Field(default=None, ge=0)
    card_url: str
    license: str


class ChooserInput(BaseModel):
    """Fields available to the chooser before the agent begins work."""

    model_config = ConfigDict(extra="forbid")

    task_text: str = Field(min_length=1)
    repository: str | None = None
    programming_language: str | None = None
    task_family: str
    public_metadata: dict[str, str | int | float | bool | None] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def reject_hidden_answer_metadata(self) -> ChooserInput:
        forbidden_fragments = {
            "answer",
            "canonical_solution",
            "gold_patch",
            "oracle",
            "patch",
            "private_test",
            "reward",
            "solution",
            "test_patch",
            "verifier_output",
        }
        for key in self.public_metadata:
            normalized = key.lower()
            if any(fragment in normalized for fragment in forbidden_fragments):
                raise ValueError(f"public_metadata contains forbidden answer field: {key}")
        return self


class ChooserTargets(BaseModel):
    model_config = ConfigDict(extra="forbid")

    difficulty: DifficultyTarget | None = None
    ambiguity: AmbiguityTarget | None = None
    domain: DomainTarget | None = None
    model_outcomes: list[ModelOutcomeAggregate] = Field(default_factory=list)


class LeakageAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    normalized_task_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    leakage_group: str
    excluded_source_fields: list[str]
    gold_fields_used_as_model_input: Literal[False] = False
    notes: list[str] = Field(default_factory=list)


class ChooserDatasetRecord(BaseModel):
    """One auditable example for classifier or outcome-model training."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["chooser-record.v0"] = "chooser-record.v0"
    example_id: str
    record_split: DatasetSplit
    source: SourceReference
    input: ChooserInput
    targets: ChooserTargets
    leakage: LeakageAudit

    @model_validator(mode="after")
    def task_hash_must_match(self) -> ChooserDatasetRecord:
        actual = hashlib.sha256(self.input.task_text.encode("utf-8")).hexdigest()
        if actual != self.leakage.task_text_sha256:
            raise ValueError("task_text_sha256 does not match input.task_text")
        normalized = unicodedata.normalize("NFKC", self.input.task_text).lower()
        normalized = re.sub(r"\W+", " ", normalized).strip()
        normalized_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if normalized_hash != self.leakage.normalized_task_sha256:
            raise ValueError("normalized_task_sha256 does not match input.task_text")
        return self


def chooser_record_json_schema() -> dict[str, Any]:
    return ChooserDatasetRecord.model_json_schema()
