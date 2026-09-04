from __future__ import annotations

import json
from pathlib import Path

import pytest

from horizon_supervisor.training.route_baseline import analyze_route_baseline


def _row(
    task: str,
    route: str,
    completed: bool,
    cost: float,
    *,
    status: str = "verified",
) -> dict:
    return {
        "schema_version": "matched-model-outcome.v1",
        "matched_group_id": f"group-{task}",
        "task": {
            "task_id": task,
            "source_task_name": task,
            "difficulty": "medium",
            "category": "software-engineering",
        },
        "model": {"route_id": route},
        "outcome": {
            "status": status,
            "completed": completed,
            "allocated_provider_cost_usd": cost,
            "estimated_list_cost_usd": cost,
            "duration_seconds": 10.0,
        },
    }


def test_route_baseline_uses_task_held_out_predictions(tmp_path: Path) -> None:
    tasks = ["task-a", "task-b", "task-c", "task-d"]
    outcomes = tmp_path / "outcomes.jsonl"
    rows = []
    for index, task in enumerate(tasks):
        rows.extend(
            (
                _row(task, "cheap", index < 2, 0.01),
                _row(
                    task,
                    "strong",
                    index in {0, 2},
                    0.10,
                    status=("agent_protocol_failure" if index == 3 else "verified"),
                ),
            )
        )
        root = tmp_path / "tasks" / task
        root.mkdir(parents=True)
        (root / "instruction.md").write_text(f"Implement example number {index}", encoding="utf-8")
        (root / "task.toml").write_text(
            "[metadata]\n"
            'difficulty = "medium"\n'
            'category = "software-engineering"\n'
            'tags = ["example"]\n',
            encoding="utf-8",
        )
    outcomes.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    panel = tmp_path / "panel.jsonl"
    panel.write_text(
        "".join(
            json.dumps(
                {
                    "wave": 1,
                    "source_task_name": task,
                    "difficulty": "medium",
                    "category": "software-engineering",
                }
            )
            + "\n"
            for task in tasks
        ),
        encoding="utf-8",
    )

    report = analyze_route_baseline(
        outcomes_path=outcomes,
        task_dir=tmp_path / "tasks",
        panel_path=panel,
        report_path=tmp_path / "report.json",
    )

    assert report["data"]["records"] == 8
    assert report["data"]["task_pattern_counts"] == {
        "all_failure": 1,
        "all_success": 1,
        "discriminating": 2,
    }
    assert report["hindsight_upper_bound"]["successes"] == 3
    assert report["evaluation_contract"]["cross_validation"] == "leave-one-task-out"
    assert all(
        policy["uses_held_out_task_outcome"] is False for policy in report["task_held_out_policies"]
    )
    assert report["benchmark_representation"]["is_full_frozen_wave"] is True
    assert report["benchmark_representation"]["observed_waves"] == [1]
    assert report["data_sufficiency"]["ready_for_general_learned_router_claim"] is False
    assert report["data"]["cost_basis_counts"] == {"cache-aware-list-price": 8}
    cascade = report["fixed_cascade_completion_first_frontier"][1]
    assert cascade["route_order"] == ["cheap", "strong"]
    assert cascade["successes"] == 3
    assert cascade["total_attempts"] == 6
    assert cascade["total_cost_usd"] == pytest.approx(0.24)
    assert report["nested_task_held_out_cascades"][1][
        "nested_evaluation_available"
    ] is True
    assert report["nested_task_held_out_cascades"][1][
        "uses_held_out_task_outcome"
    ] is False
    assert (
        report["evaluation_contract"]["cascade_stop_signal"]
        == "observed verifier-confirmed success"
    )


def test_route_baseline_accepts_unallocated_list_price_costs(tmp_path: Path) -> None:
    tasks = ["task-a", "task-b"]
    rows = []
    for task in tasks:
        for route, cost in (("cheap", 0.01), ("strong", 0.10)):
            row = _row(task, route, route == "strong", cost)
            row["outcome"]["allocated_provider_cost_usd"] = None
            rows.append(row)
        root = tmp_path / "tasks" / task
        root.mkdir(parents=True)
        (root / "instruction.md").write_text(f"Implement {task}")
        (root / "task.toml").write_text(
            '[metadata]\ndifficulty = "medium"\ncategory = "software-engineering"\n'
        )
    outcomes = tmp_path / "outcomes.jsonl"
    outcomes.write_text("".join(json.dumps(row) + "\n" for row in rows))
    panel = tmp_path / "panel.jsonl"
    panel.write_text(
        "".join(
            json.dumps(
                {
                    "wave": 1,
                    "source_task_name": task,
                    "difficulty": "medium",
                    "category": "software-engineering",
                }
            )
            + "\n"
            for task in tasks
        )
    )
    report = analyze_route_baseline(
        outcomes_path=outcomes,
        task_dir=tmp_path / "tasks",
        panel_path=panel,
        report_path=tmp_path / "report.json",
    )
    assert report["best_completion_first_static"]["route_id"] == "strong"
    assert report["data"]["cost_basis_counts"] == {"cache-aware-list-price": 4}


def test_route_baseline_rejects_non_rectangular_data(tmp_path: Path) -> None:
    outcomes = tmp_path / "outcomes.jsonl"
    outcomes.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                _row("task-a", "cheap", True, 0.01),
                _row("task-a", "strong", True, 0.10),
                _row("task-b", "cheap", False, 0.01),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not rectangular"):
        analyze_route_baseline(
            outcomes_path=outcomes,
            task_dir=tmp_path,
            panel_path=tmp_path / "panel.jsonl",
            report_path=tmp_path / "report.json",
        )


def test_route_baseline_resolves_tasks_across_multiple_waves(tmp_path: Path) -> None:
    tasks = (("task-a", 1), ("task-b", 2))
    outcomes = tmp_path / "outcomes.jsonl"
    outcomes.write_text(
        "".join(
            json.dumps(_row(task, route, route == "cheap", cost)) + "\n"
            for task, _wave in tasks
            for route, cost in (("cheap", 0.01), ("strong", 0.10))
        )
    )
    task_dirs = (tmp_path / "wave-1", tmp_path / "wave-2")
    for (task, _wave), task_dir in zip(tasks, task_dirs, strict=True):
        root = task_dir / task
        root.mkdir(parents=True)
        (root / "instruction.md").write_text(f"Implement {task}")
        (root / "task.toml").write_text(
            '[metadata]\ndifficulty = "medium"\ncategory = "software-engineering"\n'
        )
    panel = tmp_path / "panel.jsonl"
    panel.write_text(
        "".join(
            json.dumps(
                {
                    "wave": wave,
                    "source_task_name": task,
                    "difficulty": "medium",
                    "category": "software-engineering",
                }
            )
            + "\n"
            for task, wave in tasks
        )
    )
    report = analyze_route_baseline(
        outcomes_path=outcomes,
        task_dir=task_dirs,
        panel_path=panel,
        report_path=tmp_path / "report.json",
    )
    assert report["benchmark_representation"]["observed_waves"] == [1, 2]
    assert report["benchmark_representation"]["is_full_frozen_wave"] is True


def test_route_baseline_combines_sources_and_projects_routes(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                _row("task-a", "qwen", False, 0.01),
                _row("task-a", "glm", True, 0.10),
                _row("task-a", "extra", True, 0.05),
            )
        )
    )
    second.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                _row("task-b", "qwen", True, 0.01),
                _row("task-b", "glm", False, 0.10),
            )
        )
    )
    for task in ("task-a", "task-b"):
        root = tmp_path / "tasks" / task
        root.mkdir(parents=True)
        (root / "instruction.md").write_text(f"Implement {task}")
        (root / "task.toml").write_text(
            '[metadata]\ndifficulty = "medium"\ncategory = "software-engineering"\n'
        )
    panel = tmp_path / "panel.jsonl"
    panel.write_text(
        "".join(
            json.dumps(
                {
                    "wave": 1,
                    "source_task_name": task,
                    "difficulty": "medium",
                    "category": "software-engineering",
                }
            )
            + "\n"
            for task in ("task-a", "task-b")
        )
    )

    report = analyze_route_baseline(
        outcomes_path=(first, second),
        task_dir=tmp_path / "tasks",
        panel_path=panel,
        report_path=tmp_path / "report.json",
        include_routes=("qwen", "glm"),
        include_tasks=("task-a", "task-b"),
    )

    assert report["data"]["paths"] == [str(first), str(second)]
    assert report["data"]["records"] == 4
    assert report["data"]["tasks"] == 2
    assert report["data"]["routes"] == 2
    assert report["data"]["included_task_names"] == ["task-a", "task-b"]


def test_route_baseline_records_excluded_tasks_before_panel_validation(
    tmp_path: Path,
) -> None:
    outcomes = tmp_path / "outcomes.jsonl"
    outcomes.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                _row("task-a", "qwen", False, 0.01),
                _row("task-a", "glm", True, 0.10),
                _row("task-b", "qwen", True, 0.01),
                _row("task-b", "glm", False, 0.10),
                _row("incompatible", "qwen", False, 0.01),
            )
        )
    )
    for task in ("task-a", "task-b"):
        root = tmp_path / "tasks" / task
        root.mkdir(parents=True)
        (root / "instruction.md").write_text(f"Implement {task}")
        (root / "task.toml").write_text(
            '[metadata]\ndifficulty = "medium"\ncategory = "software-engineering"\n'
        )
    panel = tmp_path / "panel.jsonl"
    panel.write_text(
        "".join(
            json.dumps(
                {
                    "wave": 1,
                    "source_task_name": task,
                    "difficulty": "medium",
                    "category": "software-engineering",
                }
            )
            + "\n"
            for task in ("task-a", "task-b", "incompatible")
        )
    )

    report = analyze_route_baseline(
        outcomes_path=outcomes,
        task_dir=tmp_path / "tasks",
        panel_path=panel,
        report_path=tmp_path / "report.json",
        exclude_tasks=("incompatible",),
    )

    assert report["data"]["records"] == 4
    assert report["data"]["excluded_task_names"] == ["incompatible"]
    assert report["benchmark_representation"]["frozen_wave_tasks"] == 2
    assert report["benchmark_representation"]["excluded_frozen_task_names"] == ["incompatible"]
    assert report["benchmark_representation"]["is_full_frozen_wave"] is True


def test_route_baseline_rejects_conflicting_task_filters(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot be used together"):
        analyze_route_baseline(
            outcomes_path=tmp_path / "unused.jsonl",
            task_dir=tmp_path / "tasks",
            panel_path=tmp_path / "panel.jsonl",
            report_path=tmp_path / "report.json",
            include_tasks=("task-a",),
            exclude_tasks=("task-b",),
        )
