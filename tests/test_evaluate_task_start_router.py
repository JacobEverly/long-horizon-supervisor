from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from horizon_supervisor.training.evaluate_task_start_router import (
    evaluate_frozen_task_start_router,
)
from horizon_supervisor.training.task_start_router import (
    DEFAULT_ROUTES,
    ConstantProbabilityEstimator,
    TaskStartRouter,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path: Path) -> Path:
    vectorizer = TfidfVectorizer().fit(["synthetic held out task"])
    costs = {
        "gate7/fixed-flash": 0.01,
        "gate7/fixed-qwen": 0.02,
        "gate7/fixed-glm": 0.03,
        "gate7/fixed-kimi": 0.04,
    }
    artifact = TaskStartRouter(
        schema_version="task-start-router.v0",
        routes=DEFAULT_ROUTES,
        vectorizer=vectorizer,
        route_estimators={
            route_id: ConstantProbabilityEstimator(0.5) for route_id in DEFAULT_ROUTES
        },
        median_route_costs_usd=costs,
        success_first_margins={1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
        random_seed=17,
        development_task_count=35,
        training_dataset_sha256="development-dataset-hash",
    )
    joblib.dump(artifact, path)
    return path


def _task(tasks_dir: Path, task_name: str) -> None:
    root = tasks_dir / task_name
    root.mkdir(parents=True)
    (root / "instruction.md").write_text(f"Implement synthetic held out task {task_name}")
    (root / "task.toml").write_text(
        "[metadata]\n"
        'difficulty = "medium"\n'
        'category = "software-engineering"\n'
        'tags = ["synthetic"]\n'
    )


def _outcome(
    task_name: str,
    route_id: str,
    *,
    completed: bool,
    record_split: str = "held_out",
) -> dict:
    route_costs = {
        "gate7/fixed-flash": 0.01,
        "gate7/fixed-qwen": 0.02,
        "gate7/fixed-glm": 0.03,
        "gate7/fixed-kimi": 0.04,
    }
    return {
        "schema_version": "matched-model-outcome.v1",
        "outcome_id": f"{task_name}-{route_id}",
        "matched_group_id": f"group-{task_name}",
        "initial_state": {"kind": "clean_task_start", "digest": f"state-{task_name}"},
        "task": {
            "task_id": f"id-{task_name}",
            "source_task_name": task_name,
            "difficulty": "medium",
            "category": "software-engineering",
            "record_split": record_split,
        },
        "model": {"route_id": route_id},
        "outcome": {
            "status": "verified",
            "completed": completed,
            "duration_seconds": 10.0,
            "estimated_list_cost_usd": route_costs[route_id],
            "allocated_provider_cost_usd": None,
        },
    }


def _held_out_fixture(
    tmp_path: Path,
    *,
    record_split: str = "held_out",
) -> tuple[Path, Path]:
    task_success_route = {
        "task-a": "gate7/fixed-flash",
        "task-b": "gate7/fixed-qwen",
        "task-c": "gate7/fixed-glm",
    }
    tasks_dir = tmp_path / "tasks"
    rows = []
    for task_name, successful_route in task_success_route.items():
        _task(tasks_dir, task_name)
        for route_id in DEFAULT_ROUTES:
            rows.append(
                _outcome(
                    task_name,
                    route_id,
                    completed=route_id == successful_route,
                    record_split=record_split,
                )
            )
    outcomes = tmp_path / "synthetic-outcomes.jsonl"
    outcomes.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return outcomes, tasks_dir


def test_frozen_evaluator_is_fit_free_replays_all_capacities_and_preserves_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_path = _artifact(tmp_path / "router.joblib")
    outcomes, tasks_dir = _held_out_fixture(tmp_path)
    artifact_hash_before = _sha256(artifact_path)

    def forbidden_fit(*_args, **_kwargs):
        raise AssertionError("held-out evaluation must never fit")

    monkeypatch.setattr(TfidfVectorizer, "fit", forbidden_fit)
    report = evaluate_frozen_task_start_router(
        outcomes_path=outcomes,
        artifact_path=artifact_path,
        task_dir=tasks_dir,
        report_path=tmp_path / "scorecard.json",
        expected_tasks=3,
    )

    learned = report["learned_task_start_policies"]
    assert [row["successes"] for row in learned] == [1, 2, 3, 3]
    assert [row["total_attempts"] for row in learned] == [3, 5, 6, 6]
    assert [row["successes"] for row in report["fixed_cascade_baselines"]] == [2, 3, 3]
    assert report["evaluation_contract"]["fit_calls"] == 0
    assert report["evaluation_contract"]["tuning_calls"] == 0
    assert report["frozen_artifact"]["hash_preserved"] is True
    assert report["frozen_artifact"]["sha256_before_evaluation"] == artifact_hash_before
    assert _sha256(artifact_path) == artifact_hash_before


def test_frozen_evaluator_rejects_non_held_out_rows(tmp_path: Path) -> None:
    artifact_path = _artifact(tmp_path / "router.joblib")
    outcomes, tasks_dir = _held_out_fixture(tmp_path, record_split="development")

    with pytest.raises(ValueError, match="held_out rows only"):
        evaluate_frozen_task_start_router(
            outcomes_path=outcomes,
            artifact_path=artifact_path,
            task_dir=tasks_dir,
            report_path=tmp_path / "scorecard.json",
            expected_tasks=3,
        )


def test_frozen_evaluator_rejects_non_rectangular_coverage(tmp_path: Path) -> None:
    artifact_path = _artifact(tmp_path / "router.joblib")
    outcomes, tasks_dir = _held_out_fixture(tmp_path)
    rows = outcomes.read_text().splitlines()
    outcomes.write_text("\n".join(rows[:-1]) + "\n")

    with pytest.raises(ValueError, match="not rectangular"):
        evaluate_frozen_task_start_router(
            outcomes_path=outcomes,
            artifact_path=artifact_path,
            task_dir=tasks_dir,
            report_path=tmp_path / "scorecard.json",
            expected_tasks=3,
        )
