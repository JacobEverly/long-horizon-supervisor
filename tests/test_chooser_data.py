from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from horizon_supervisor.chooser.build_sample import (
    normalize_apps_difficulty,
    normalize_codeforces_rating,
)
from horizon_supervisor.chooser.schema import (
    ChooserDatasetRecord,
    ChooserInput,
    Difficulty,
    ModelOutcomeAggregate,
    OutcomeProvenance,
)

ROOT = Path(__file__).resolve().parents[1]


def test_difficulty_normalization_is_explicit() -> None:
    assert normalize_apps_difficulty("introductory") is Difficulty.EASY
    assert normalize_apps_difficulty("interview") is Difficulty.MEDIUM
    assert normalize_apps_difficulty("competition") is Difficulty.HARD
    assert normalize_codeforces_rating(1_200) is Difficulty.EASY
    assert normalize_codeforces_rating(1_300) is Difficulty.MEDIUM
    assert normalize_codeforces_rating(1_900) is Difficulty.MEDIUM
    assert normalize_codeforces_rating(2_000) is Difficulty.HARD

    with pytest.raises(ValueError, match="positive"):
        normalize_codeforces_rating(0)


def test_chooser_input_rejects_answer_fields() -> None:
    with pytest.raises(ValidationError, match="forbidden answer field"):
        ChooserInput(
            task_text="Fix the bug",
            task_family="repository_issue_resolution",
            public_metadata={"gold_patch": "do not leak this"},
        )


def test_model_outcomes_are_aggregated_and_consistent() -> None:
    outcome = ModelOutcomeAggregate(
        model_id="model-a",
        deployment_id="model-a@2026-08-25",
        agent_id="terminus-2@0.20.0",
        verifier_id="example-verifier@sha256:abc",
        attempts=4,
        successes=3,
        source_id="gate7-pilot",
    )
    assert outcome.observed_success_rate == 0.75

    with pytest.raises(ValidationError, match="cannot exceed"):
        ModelOutcomeAggregate.model_validate(
            outcome.model_dump(exclude_computed_fields=True) | {"successes": 5}
        )


def test_pinned_source_registry_has_no_unversioned_huggingface_sources() -> None:
    registry = json.loads(
        (ROOT / "data/chooser/source-registry-v0.json").read_text(encoding="utf-8")
    )
    source_ids = [source["source_id"] for source in registry["sources"]]
    assert len(source_ids) == len(set(source_ids))
    for source in registry["sources"]:
        assert source["license"]
        assert source["allowed_input_fields"]
        assert source["excluded_source_fields"]
        if source["kind"] == "huggingface":
            assert re.fullmatch(r"[0-9a-f]{40}", source["revision"])
            if source.get("collection", {}).get("reader") == "parquet":
                assert re.fullmatch(r"[0-9a-f]{40}", source["parquet_revision"])


def test_smoke_sample_validates_and_contains_no_cross_source_duplicates() -> None:
    rows = [
        ChooserDatasetRecord.model_validate_json(line)
        for line in (ROOT / "data/chooser/sample-v0.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == 36
    assert Counter(row.source.source_id for row in rows) == {
        "apps": 12,
        "deepmind-code-contests": 12,
        "nebius-swe-bench-extra": 12,
    }
    assert Counter(
        row.targets.difficulty.value.value if row.targets.difficulty else "unlabeled"
        for row in rows
    ) == {"easy": 8, "medium": 8, "hard": 8, "unlabeled": 12}
    assert all(not row.leakage.gold_fields_used_as_model_input for row in rows)

    normalized_hashes = []
    for row in rows:
        normalized = unicodedata.normalize("NFKC", row.input.task_text).lower()
        normalized = re.sub(r"\W+", " ", normalized).strip()
        normalized_hashes.append(hashlib.sha256(normalized.encode()).hexdigest())
    assert len(normalized_hashes) == len(set(normalized_hashes))


def test_task_text_mutation_invalidates_leakage_hash() -> None:
    first_line = (ROOT / "data/chooser/sample-v0.jsonl").read_text().splitlines()[0]
    payload = json.loads(first_line)
    payload["input"]["task_text"] += " changed"

    with pytest.raises(ValidationError, match="does not match"):
        ChooserDatasetRecord.model_validate(payload)


def test_full_corpus_is_group_split_deduplicated_and_outcome_audited() -> None:
    corpus_path = ROOT / "data/chooser/corpus-v0.jsonl"
    with corpus_path.open(encoding="utf-8") as handle:
        rows = [ChooserDatasetRecord.model_validate_json(line) for line in handle]

    assert len(rows) == 13_207
    assert Counter(row.source.source_id for row in rows) == {
        "apps": 4_870,
        "deepmind-code-contests": 2_107,
        "nebius-swe-bench-extra": 6_230,
    }
    assert Counter(row.record_split.value for row in rows) == {
        "train": 10_800,
        "validation": 1_138,
        "test": 1_269,
    }
    assert len({row.leakage.normalized_task_sha256 for row in rows}) == len(rows)

    group_splits: dict[str, set[str]] = {}
    for row in rows:
        group_splits.setdefault(row.leakage.leakage_group, set()).add(
            row.record_split.value
        )
    assert all(len(splits) == 1 for splits in group_splits.values())

    outcome_rows = [row for row in rows if row.targets.model_outcomes]
    assert len(outcome_rows) == 3_303
    assert sum(
        outcome.attempts
        for row in outcome_rows
        for outcome in row.targets.model_outcomes
    ) == 76_480
    assert all(
        outcome.provenance is OutcomeProvenance.DATASET_DECLARED_UNVERSIONED
        for row in outcome_rows
        for outcome in row.targets.model_outcomes
    )
    assert all(
        row.input.task_family == "repository_issue_resolution" for row in outcome_rows
    )


def test_full_corpus_summary_records_public_data_imbalance() -> None:
    summary = json.loads(
        (ROOT / "data/chooser/corpus-v0-summary.json").read_text(encoding="utf-8")
    )
    assert summary["record_count"] == 13_207
    assert summary["deduplication"]["dropped_total"] == 227
    assert summary["outcomes"]["raw_trajectory_rows"] == 80_036
    assert summary["outcomes"]["joined_trajectory_attempts"] == 76_480
    assert summary["outcomes"]["joined_task_model_sets"][
        "swe-agent-llama-405b + swe-agent-llama-70b + swe-agent-llama-8b"
    ] == 48
    assert summary["gold_fields_used_as_model_input"] is False
