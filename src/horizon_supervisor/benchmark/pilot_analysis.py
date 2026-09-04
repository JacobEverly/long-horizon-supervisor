from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    no_worse = (
        left["successes"] >= right["successes"]
        and left["total_cost_usd"] <= right["total_cost_usd"]
        and left["median_latency_seconds"] <= right["median_latency_seconds"]
    )
    strictly_better = (
        left["successes"] > right["successes"]
        or left["total_cost_usd"] < right["total_cost_usd"]
        or left["median_latency_seconds"] < right["median_latency_seconds"]
    )
    return no_worse and strictly_better


def analyze_pilot(input_path: Path, output_path: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
    ]
    if not rows:
        raise ValueError("pilot dataset is empty")
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_route: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["task"]["source_task_name"]].append(row)
        by_route[row["model"]["route_id"]].append(row)
    route_ids = sorted(by_route)
    expected_pairs = {
        (task_name, route_id) for task_name in by_task for route_id in route_ids
    }
    observed_pairs = {
        (row["task"]["source_task_name"], row["model"]["route_id"])
        for row in rows
    }
    if observed_pairs != expected_pairs or len(rows) != len(expected_pairs):
        raise ValueError("pilot dataset is not rectangular")

    route_summaries = []
    for route_id, route_rows in sorted(by_route.items()):
        summary = {
            "route_id": route_id,
            "tasks": len(route_rows),
            "successes": sum(row["outcome"]["completed"] for row in route_rows),
            "success_rate": statistics.mean(
                int(row["outcome"]["completed"]) for row in route_rows
            ),
            "total_cost_usd": sum(
                row["outcome"]["allocated_provider_cost_usd"]
                for row in route_rows
            ),
            "median_latency_seconds": statistics.median(
                row["outcome"]["duration_seconds"] for row in route_rows
            ),
            "mean_latency_seconds": statistics.mean(
                row["outcome"]["duration_seconds"] for row in route_rows
            ),
        }
        route_summaries.append(summary)
    pareto_routes = [
        row["route_id"]
        for row in route_summaries
        if not any(
            _dominates(other, row)
            for other in route_summaries
            if other["route_id"] != row["route_id"]
        )
    ]

    cheapest_oracle_rows = []
    fastest_oracle_rows = []
    task_patterns: Counter[str] = Counter()
    for _task_name, task_rows in sorted(by_task.items()):
        successes = [row for row in task_rows if row["outcome"]["completed"]]
        task_patterns[
            "all_success"
            if len(successes) == len(task_rows)
            else "all_failure"
            if not successes
            else "discriminating"
        ] += 1
        candidate_rows = successes or task_rows
        cheapest_oracle_rows.append(
            min(candidate_rows, key=lambda row: row["outcome"]["allocated_provider_cost_usd"])
        )
        fastest_oracle_rows.append(
            min(candidate_rows, key=lambda row: row["outcome"]["duration_seconds"])
        )

    best_static = max(
        route_summaries,
        key=lambda row: (row["successes"], -row["total_cost_usd"]),
    )
    cheapest_oracle_cost = sum(
        row["outcome"]["allocated_provider_cost_usd"]
        for row in cheapest_oracle_rows
    )
    fastest_oracle_latency = sum(
        row["outcome"]["duration_seconds"] for row in fastest_oracle_rows
    )
    best_static_latency = sum(
        row["outcome"]["duration_seconds"]
        for row in by_route[best_static["route_id"]]
    )
    difficulty_counts = Counter(
        task_rows[0]["task"]["difficulty"] for task_rows in by_task.values()
    )
    category_counts = Counter(
        task_rows[0]["task"]["category"] for task_rows in by_task.values()
    )
    statuses = Counter(row["outcome"]["status"] for row in rows)
    successes = sum(row["outcome"]["completed"] for row in rows)
    checks = {
        "rectangular_task_model_pairs": True,
        "all_outcomes_verified": statuses == {"verified": len(rows)},
        "all_cost_targets_present": all(
            row["outcome"]["allocated_provider_cost_usd"] is not None for row in rows
        ),
        "all_latency_targets_present": all(
            row["outcome"]["duration_seconds"] is not None for row in rows
        ),
        "positive_and_negative_completion_labels": 0 < successes < len(rows),
        "contains_discriminating_task": task_patterns["discriminating"] > 0,
        "difficulty_ratio_matches_wave_one": difficulty_counts
        == {"hard": 2, "medium": 4},
        "six_distinct_categories": len(category_counts) == 6,
    }
    report = {
        "schema_version": "gate8-six-task-pilot-analysis.v0",
        "data": {
            "path": str(input_path),
            "records": len(rows),
            "tasks": len(by_task),
            "routes": len(by_route),
            "successes": successes,
            "failures": len(rows) - successes,
            "success_prevalence": successes / len(rows),
            "status_counts": dict(sorted(statuses.items())),
            "difficulty_counts": dict(sorted(difficulty_counts.items())),
            "category_counts": dict(sorted(category_counts.items())),
            "task_pattern_counts": dict(sorted(task_patterns.items())),
            "clean_allocated_cost_usd": sum(
                row["outcome"]["allocated_provider_cost_usd"] for row in rows
            ),
        },
        "static_route_baselines": route_summaries,
        "static_pareto_route_ids": pareto_routes,
        "completion_first_best_static": best_static,
        "hindsight_upper_bounds": {
            "maximum_solved_tasks": sum(bool(rows_) for rows_ in (
                [row for row in task_rows if row["outcome"]["completed"]]
                for task_rows in by_task.values()
            )),
            "cheapest_success_or_cheapest_failure_cost_usd": cheapest_oracle_cost,
            "cost_savings_vs_best_static": (
                1 - cheapest_oracle_cost / best_static["total_cost_usd"]
                if best_static["total_cost_usd"]
                else 0.0
            ),
            "fastest_success_or_fastest_failure_latency_seconds": (
                fastest_oracle_latency
            ),
            "latency_savings_vs_best_static": (
                1 - fastest_oracle_latency / best_static_latency
                if best_static_latency
                else 0.0
            ),
            "uses_current_trial_outcomes": True,
        },
        "quality_checks": checks,
        "all_quality_checks_passed": all(checks.values()),
        "decision": {
            "enough_for_descriptive_policy_comparison": all(checks.values()),
            "enough_for_learned_generalization_claim": False,
            "continue_to_full_18_task_development_wave": all(checks.values()),
        },
        "interpretation_guard": (
            "Six tasks can validate plumbing and reveal routing opportunity, but cannot "
            "support a general learned-router claim. Oracle rows use hindsight and are "
            "upper bounds, not deployable policies."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the Gate 8 six-task pilot")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze_pilot(args.input, args.output), indent=2))


if __name__ == "__main__":
    main()
