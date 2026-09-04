from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from horizon_supervisor.benchmark.sentinel_screen import analyze_sentinel_screen


def _row(
    task: str,
    task_id: str,
    route: str,
    completed: bool,
    status: str = "verified",
) -> dict:
    return {
        "schema_version": "matched-model-outcome.v1",
        "matched_group_id": f"group-{task}",
        "task": {"source_task_name": task, "task_id": task_id},
        "model": {"route_id": route},
        "outcome": {
            "status": status,
            "completed": completed,
            "estimated_list_cost_usd": 0.1 if route == "glm" else 0.01,
        },
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    manifest = {
        "sentinel_routes": ["qwen", "glm"],
        "tranches": [
            {"id": "t1", "task_names": ["task-a", "task-b", "task-c"]}
        ],
        "expansion_rule": {
            "agreement_audit_seed": "seed",
            "agreement_audits_per_tranche": 1,
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    rows = [
        _row("task-a", "id-a", "qwen", False),
        _row("task-a", "id-a", "glm", True),
        _row("task-b", "id-b", "qwen", True),
        _row("task-b", "id-b", "glm", True),
        _row("task-c", "id-c", "qwen", False),
        _row("task-c", "id-c", "glm", False),
    ]
    outcomes_path = tmp_path / "outcomes.jsonl"
    outcomes_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return manifest_path, outcomes_path


def test_sentinel_analysis_expands_disagreements_and_hashed_audit(
    tmp_path: Path,
) -> None:
    manifest_path, outcomes_path = _fixture(tmp_path)
    report = analyze_sentinel_screen(
        outcomes_path, manifest_path, "t1", tmp_path / "report.json"
    )
    expected_audit = min(
        ("task-b", "task-c"),
        key=lambda task: hashlib.sha256(
            f"seed|{'id-b' if task == 'task-b' else 'id-c'}|{task}".encode()
        ).hexdigest(),
    )

    assert report["disagreement_task_names"] == ["task-a"]
    assert report["agreement_audit_task_names"] == [expected_audit]
    assert report["expansion_task_names"] == ["task-a", expected_audit]
    assert report["trial_accounting"] == {
        "sentinel_trials": 6,
        "expansion_trials": 4,
        "total_trials_after_expansion": 10,
        "full_matrix_trials": 12,
        "saved_trials": 2,
    }
    assert report["best_completion_first_static"]["route_id"] == "glm"
    assert report["sentinel_oracle_headroom"] == 0


def test_sentinel_analysis_rejects_incomplete_tranche(tmp_path: Path) -> None:
    manifest_path, outcomes_path = _fixture(tmp_path)
    rows = outcomes_path.read_text().splitlines()
    outcomes_path.write_text("\n".join(rows[:-1]) + "\n")

    with pytest.raises(ValueError, match="not complete"):
        analyze_sentinel_screen(
            outcomes_path, manifest_path, "t1", tmp_path / "report.json"
        )


def test_sentinel_analysis_counts_provider_failure_as_deployment_outcome(
    tmp_path: Path,
) -> None:
    manifest_path, outcomes_path = _fixture(tmp_path)
    rows = [json.loads(line) for line in outcomes_path.read_text().splitlines()]
    rows[0] = _row(
        "task-a", "id-a", "qwen", False, status="provider_error"
    )
    outcomes_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    report = analyze_sentinel_screen(
        outcomes_path, manifest_path, "t1", tmp_path / "report.json"
    )

    observation = report["deployment_observation"]
    assert observation["status_counts"] == {
        "provider_error": 1,
        "verified": 5,
    }
    assert observation["provider_failure_pairs"] == ["task-a|qwen"]
    assert observation["capability_learning_valid_pair_count"] == 5


def test_sentinel_analysis_rejects_environment_failure(tmp_path: Path) -> None:
    manifest_path, outcomes_path = _fixture(tmp_path)
    rows = [json.loads(line) for line in outcomes_path.read_text().splitlines()]
    rows[0] = _row(
        "task-a", "id-a", "qwen", False, status="infrastructure_error"
    )
    outcomes_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    with pytest.raises(ValueError, match="not deployment-observable"):
        analyze_sentinel_screen(
            outcomes_path, manifest_path, "t1", tmp_path / "report.json"
        )
