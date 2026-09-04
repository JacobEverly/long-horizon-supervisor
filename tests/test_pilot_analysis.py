from __future__ import annotations

import json
from pathlib import Path

from horizon_supervisor.benchmark.pilot_analysis import analyze_pilot


def _row(task: str, route: str, completed: bool, cost: float, latency: float) -> dict:
    return {
        "task": {
            "source_task_name": task,
            "difficulty": "hard" if task in {"a", "b"} else "medium",
            "category": f"category-{task}",
        },
        "model": {"route_id": route},
        "outcome": {
            "status": "verified",
            "completed": completed,
            "allocated_provider_cost_usd": cost,
            "duration_seconds": latency,
        },
    }


def test_pilot_analysis_validates_balance_and_finds_static_pareto(tmp_path: Path) -> None:
    rows = []
    for task in "abcdef":
        rows.extend(
            [
                _row(task, "cheap", task in "abc", 1.0, 10.0),
                _row(task, "slow", task in "ab", 2.0, 20.0),
            ]
        )
    input_path = tmp_path / "pilot.jsonl"
    input_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = analyze_pilot(input_path, tmp_path / "report.json")
    assert report["all_quality_checks_passed"] is True
    assert report["data"]["successes"] == 5
    assert report["static_pareto_route_ids"] == ["cheap"]
    assert report["decision"]["continue_to_full_18_task_development_wave"] is True
    assert report["decision"]["enough_for_learned_generalization_claim"] is False
