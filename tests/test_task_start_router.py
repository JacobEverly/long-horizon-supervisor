from __future__ import annotations

import json
from pathlib import Path

import pytest

from horizon_supervisor.training.task_start_router import (
    load_task_start_router,
    train_task_start_router,
)


def _row(
    task_index: int,
    route_id: str,
    *,
    completed: bool,
    cost: float,
    record_split: str = "development",
) -> dict:
    task_name = f"task-{task_index}"
    return {
        "schema_version": "supervisor-task-route.v0",
        "example_id": f"example-{task_index}-{route_id}",
        "record_split": record_split,
        "leakage_group": f"leakage-{task_index}",
        "matched_group_id": f"matched-{task_index}",
        "initial_state": {"kind": "clean_task_start", "digest": f"state-{task_index}"},
        "input": {
            "task_id": f"id-{task_index}",
            "source_task_name": task_name,
            "instruction": f"Implement deterministic feature number {task_index}",
            "instruction_sha256": f"instruction-{task_index}",
            "difficulty": "medium" if task_index < 3 else "hard",
            "category": "software-engineering",
            "tags": ["example", f"group-{task_index % 2}"],
        },
        "available_actions": [
            {"action": "start_model", "target_route_id": "cheap"},
            {"action": "start_model", "target_route_id": "strong"},
        ],
        "logged_action": {"action": "start_model", "target_route_id": route_id},
        "candidate": {
            "route_id": route_id,
            "endpoint": f"example/{route_id}",
            "agent": "example-agent",
            "features": {},
        },
        "target": {
            "completed": completed,
            "reward": float(completed),
            "status": "verified",
            "exception_type": None,
            "cost_usd": cost,
            "cost_basis": "cache-aware-list-price",
            "duration_seconds": 10.0,
            "input_tokens": 100,
            "cache_tokens": 10,
            "output_tokens": 20,
            "model_calls": 2,
        },
        "provenance": {
            "source_result_path": f"runs/development/task-{task_index}/{route_id}/result.json"
        },
    }


def _dataset(path: Path, *, record_split: str = "development") -> Path:
    rows = []
    for task_index in range(4):
        rows.extend(
            (
                _row(
                    task_index,
                    "cheap",
                    completed=task_index in {0, 1},
                    cost=0.01,
                    record_split=record_split,
                ),
                _row(
                    task_index,
                    "strong",
                    completed=task_index in {0, 2, 3},
                    cost=0.10,
                    record_split=record_split,
                ),
            )
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _train(tmp_path: Path, stem: str = "run"):
    return train_task_start_router(
        dataset_path=_dataset(tmp_path / f"{stem}-development.jsonl"),
        artifact_path=tmp_path / f"{stem}.joblib",
        report_path=tmp_path / f"{stem}.json",
        expected_routes=("cheap", "strong"),
        expected_tasks=4,
    )


def test_nested_loocv_keeps_task_and_leakage_groups_isolated(tmp_path: Path) -> None:
    _artifact, report = _train(tmp_path)

    assert report["nested_loocv"]["group_isolation_verified"] is True
    assert report["nested_loocv"]["outer_folds"] == 4
    assert report["nested_loocv"]["inner_folds_per_outer_fold"] == 3
    assert all(
        not fold["training_contains_held_out_task"]
        and not fold["training_contains_held_out_leakage_group"]
        and fold["training_task_count"] == 3
        for fold in report["nested_loocv"]["fold_audit"]
    )
    assert {
        policy["max_routes"]
        for policy in report["learned_route_policies_nested_loocv"]
    } == {1, 2}


def test_joblib_artifact_reload_preserves_predictions(tmp_path: Path) -> None:
    artifact, _report = _train(tmp_path)
    artifact_path = tmp_path / "run.joblib"
    restored = load_task_start_router(artifact_path)

    kwargs = {
        "instruction": "Implement deterministic feature number 8",
        "difficulty": "hard",
        "category": "software-engineering",
        "tags": ["example"],
        "max_routes": 2,
    }
    assert restored.predict_route_order(**kwargs) == artifact.predict_route_order(**kwargs)
    assert restored.training_dataset_sha256 == artifact.training_dataset_sha256


def test_fixed_seed_produces_identical_policy_results(tmp_path: Path) -> None:
    first_artifact, first_report = _train(tmp_path, "first")
    second_artifact, second_report = _train(tmp_path, "second")

    assert first_report["comparison"] == second_report["comparison"]
    assert (
        first_report["final_artifact_fit"]["selected_margin_by_max_routes"]
        == second_report["final_artifact_fit"]["selected_margin_by_max_routes"]
    )
    assert first_artifact.predict_route_order("Implement feature 10", max_routes=2) == (
        second_artifact.predict_route_order("Implement feature 10", max_routes=2)
    )


def test_router_rejects_held_out_rows(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "development.jsonl", record_split="held_out")

    with pytest.raises(ValueError, match="development rows only"):
        train_task_start_router(
            dataset_path=dataset,
            artifact_path=tmp_path / "artifact.joblib",
            report_path=tmp_path / "report.json",
            expected_routes=("cheap", "strong"),
            expected_tasks=4,
        )


def test_router_rejects_held_out_paths_before_reading(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "wave3-results.jsonl")

    with pytest.raises(ValueError, match="held-out/Wave 3 path"):
        train_task_start_router(
            dataset_path=dataset,
            artifact_path=tmp_path / "artifact.joblib",
            report_path=tmp_path / "report.json",
            expected_routes=("cheap", "strong"),
            expected_tasks=4,
        )
