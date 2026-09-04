from __future__ import annotations

import json
from pathlib import Path

import pytest

from horizon_supervisor.benchmark.merge_outcomes import merge_matched_outcomes


def _row(task: str, route: str, status: str, outcome_id: str) -> dict:
    return {
        "schema_version": "matched-model-outcome.v1",
        "outcome_id": outcome_id,
        "task": {"task_id": task, "source_task_name": task},
        "model": {"route_id": route},
        "outcome": {"status": status, "completed": status == "verified"},
        "provenance": {},
    }


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_merge_prefers_single_verified_recovery_and_preserves_audit(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    recovery = tmp_path / "recovery.jsonl"
    _write(
        first,
        [
            _row("task-a", "qwen", "verified", "q1"),
            _row("task-a", "kimi", "infrastructure_error", "k1"),
        ],
    )
    _write(recovery, [_row("task-a", "kimi", "verified", "k2")])
    output = tmp_path / "merged.jsonl"
    summary = merge_matched_outcomes(
        [first, recovery],
        output,
        tmp_path / "summary.json",
        expected_routes=("qwen", "kimi"),
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert summary["record_count"] == 2
    assert summary["superseded_status_counts"] == {"infrastructure_error": 1}
    kimi = next(row for row in rows if row["model"]["route_id"] == "kimi")
    assert kimi["outcome_id"] == "k2"
    assert kimi["provenance"]["superseded_outcome_ids"] == ["k1"]


def test_merge_rejects_multiple_verified_attempts(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _write(first, [_row("task-a", "qwen", "verified", "q1")])
    _write(second, [_row("task-a", "qwen", "verified", "q2")])
    with pytest.raises(ValueError, match="multiple learning-valid trials"):
        merge_matched_outcomes(
            [first, second],
            tmp_path / "merged.jsonl",
            tmp_path / "summary.json",
            expected_routes=("qwen",),
        )


def test_merge_accepts_attributable_agent_protocol_failure(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write(
        source,
        [
            _row("task-a", "qwen", "agent_protocol_failure", "q1"),
            _row("task-a", "kimi", "verified", "k1"),
        ],
    )
    output = tmp_path / "merged.jsonl"
    summary = merge_matched_outcomes(
        [source],
        output,
        tmp_path / "summary.json",
        expected_routes=("qwen", "kimi"),
    )

    assert summary["all_pairs_present_once_and_learning_valid"] is True
    assert summary["all_pairs_present_once_and_verified"] is False
    assert summary["learning_status_counts"] == {
        "agent_protocol_failure": 1,
        "verified": 1,
    }


def test_merge_can_accept_provider_failure_for_deployment_screen(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.jsonl"
    recovery = tmp_path / "recovery.jsonl"
    _write(
        first,
        [_row("task-a", "qwen", "infrastructure_error", "q1")],
    )
    _write(recovery, [_row("task-a", "qwen", "provider_error", "q2")])

    summary = merge_matched_outcomes(
        [first, recovery],
        tmp_path / "merged.jsonl",
        tmp_path / "summary.json",
        expected_routes=("qwen",),
        accepted_statuses=frozenset(
            {"verified", "agent_protocol_failure", "provider_error"}
        ),
    )

    assert summary["all_pairs_present_once_and_accepted"] is True
    assert summary["all_pairs_present_once_and_learning_valid"] is False
    assert summary["learning_status_counts"] == {"provider_error": 1}
    assert summary["superseded_status_counts"] == {"infrastructure_error": 1}


def test_merge_excludes_declared_task_before_panel_validation(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write(
        source,
        [
            _row("task-a", "qwen", "verified", "q1"),
            _row("task-a", "kimi", "verified", "k1"),
            _row("task-b", "qwen", "infrastructure_error", "q2"),
        ],
    )

    output = tmp_path / "merged.jsonl"
    summary = merge_matched_outcomes(
        [source],
        output,
        tmp_path / "summary.json",
        expected_routes=("qwen", "kimi"),
        exclude_tasks=frozenset({"task-b"}),
    )

    assert summary["record_count"] == 2
    assert summary["task_count"] == 1
    assert summary["excluded_task_names"] == ["task-b"]
    assert summary["excluded_record_count"] == 1
    retained_tasks = {
        json.loads(line)["task"]["source_task_name"]
        for line in output.read_text().splitlines()
    }
    assert retained_tasks == {"task-a"}
