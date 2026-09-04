from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from horizon_supervisor.training.route_baseline import (
    _cascade_summary,
    _load_rows,
    _pareto_strategies,
    _rectangular_panel,
    _route_summary,
)

EXPECTED_ROUTES = (
    "gate7/fixed-flash",
    "gate7/fixed-glm",
    "gate7/fixed-kimi",
    "gate7/fixed-qwen",
)
FROZEN_CASCADES = {
    "frozen-cascade:flash>qwen": (
        "gate7/fixed-flash",
        "gate7/fixed-qwen",
    ),
    "frozen-cascade:flash>qwen>glm": (
        "gate7/fixed-flash",
        "gate7/fixed-qwen",
        "gate7/fixed-glm",
    ),
    "frozen-cascade:flash>qwen>glm>kimi": (
        "gate7/fixed-flash",
        "gate7/fixed-qwen",
        "gate7/fixed-glm",
        "gate7/fixed-kimi",
    ),
}


def build_frozen_policy_scorecard(
    outcomes_path: Path,
    output_path: Path,
    *,
    dedicated_key_usage_before_usd: float,
    dedicated_key_usage_after_usd: float,
    completed_run_report_spend_usd: float,
    expected_tasks: int = 11,
) -> dict[str, Any]:
    rows = _load_rows(outcomes_path)
    by_task, routes = _rectangular_panel(rows)
    if tuple(routes) != EXPECTED_ROUTES:
        raise ValueError(f"unexpected route set: {routes}")
    if len(by_task) != expected_tasks:
        raise ValueError(
            f"expected {expected_tasks} task groups, found {len(by_task)}"
        )

    statics = _route_summary(by_task, routes)
    cascades = [
        {
            **_cascade_summary(
                name,
                by_task,
                {task_name: order for task_name in by_task},
            ),
            "route_order": list(order),
            "max_routes": len(order),
            "policy_selected_on": "35-task development set",
            "held_out_policy_tuning": False,
        }
        for name, order in FROZEN_CASCADES.items()
    ]
    strategies = statics + cascades
    pareto_names = set(_pareto_strategies(strategies))
    pareto = [
        {
            "strategy": row["strategy"],
            "successes": row["successes"],
            "success_rate": row["success_rate"],
            "total_cost_usd": row["total_cost_usd"],
        }
        for row in strategies
        if row["strategy"] in pareto_names
    ]
    pareto.sort(key=lambda row: (row["successes"], row["total_cost_usd"]))

    task_outcomes = []
    for task_name in sorted(by_task):
        route_results = {
            route_id: bool(by_task[task_name][route_id]["outcome"]["completed"])
            for route_id in routes
        }
        task_outcomes.append(
            {
                "task": task_name,
                "route_completed": route_results,
                "success_count": sum(route_results.values()),
            }
        )
    patterns = Counter(
        "all-success"
        if row["success_count"] == len(routes)
        else "all-failure"
        if row["success_count"] == 0
        else "discriminating"
        for row in task_outcomes
    )

    best_static = max(
        statics,
        key=lambda row: (row["successes"], -row["total_cost_usd"]),
    )
    two_route, three_route, four_route = cascades
    key_delta = dedicated_key_usage_after_usd - dedicated_key_usage_before_usd
    report_delta = key_delta - completed_run_report_spend_usd
    final_evaluation = expected_tasks == 18
    if final_evaluation:
        evaluation_role = "final held-out Wave 3 evaluation"
        pareto_field = "cost_completion_pareto"
        spend_threshold_field = "under_ten_usd"
        spend_threshold = 10.0
        interpretation_guard = (
            "The complete 18-task held-out Wave 3 supports a final sealed "
            "evaluation result, not a general production guarantee. Costs are "
            "cache-aware catalog replay costs; exact experiment spend is reported "
            "separately from the dedicated key."
        )
    else:
        evaluation_role = "provisional held-out Wave 3 checkpoint"
        pareto_field = "preliminary_cost_completion_pareto"
        spend_threshold_field = "under_five_usd"
        spend_threshold = 5.0
        interpretation_guard = (
            "Eleven held-out tasks support a provisional direction, not a general "
            "production guarantee. Costs are cache-aware catalog replay costs; exact "
            "experiment spend is reported separately from the dedicated key."
            if expected_tasks == 11
            else (
                f"{expected_tasks} held-out tasks support a provisional direction, "
                "not a general production guarantee. Costs are cache-aware catalog "
                "replay costs; exact experiment spend is reported separately from "
                "the dedicated key."
            )
        )
    scorecard = {
        "schema_version": "frozen-held-out-policy-scorecard.v0",
        "evaluation_role": evaluation_role,
        "data": {
            "outcomes_path": str(outcomes_path),
            "tasks": len(by_task),
            "routes": len(routes),
            "records": len(rows),
            "all_pairs_learning_valid_and_rectangular": True,
            "task_pattern_counts": dict(sorted(patterns.items())),
        },
        "policy_guard": {
            "route_order_source": "35-task development report",
            "held_out_policy_tuning": False,
            "cascade_stop_signal": "verifier-confirmed success",
            "replay_assumption": "clean-start restart-and-escalate",
        },
        "static_models": statics,
        "frozen_cascades": cascades,
        pareto_field: pareto,
        "task_outcomes": task_outcomes,
        "spend_audit": {
            "basis": "dedicated OpenRouter key usage delta",
            "dedicated_key_usage_before_usd": dedicated_key_usage_before_usd,
            "dedicated_key_usage_after_usd": dedicated_key_usage_after_usd,
            "exact_incremental_spend_usd": key_delta,
            "completed_run_report_spend_usd": completed_run_report_spend_usd,
            "key_delta_minus_run_reports_usd": report_delta,
            spend_threshold_field: key_delta < spend_threshold,
        },
        "headline": {
            "best_static_strategy": best_static["strategy"],
            "best_static_successes": best_static["successes"],
            "best_static_success_rate": best_static["success_rate"],
            "two_route_successes": two_route["successes"],
            "two_route_total_cost_usd": two_route["total_cost_usd"],
            "three_route_successes": three_route["successes"],
            "three_route_total_cost_usd": three_route["total_cost_usd"],
            "four_route_successes": four_route["successes"],
            "four_route_success_rate": four_route["success_rate"],
            "four_route_total_cost_usd": four_route["total_cost_usd"],
            "four_route_beats_best_static_completion": (
                four_route["successes"] > best_static["successes"]
            ),
        },
        "interpretation_guard": interpretation_guard,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(scorecard, indent=2) + "\n", encoding="utf-8")
    return scorecard


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score frozen static and cascade policies on held-out outcomes"
    )
    parser.add_argument("--outcomes", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--key-usage-before", required=True, type=float)
    parser.add_argument("--key-usage-after", required=True, type=float)
    parser.add_argument("--run-report-spend", required=True, type=float)
    parser.add_argument("--expected-tasks", default=11, type=int)
    args = parser.parse_args()
    scorecard = build_frozen_policy_scorecard(
        args.outcomes,
        args.output,
        dedicated_key_usage_before_usd=args.key_usage_before,
        dedicated_key_usage_after_usd=args.key_usage_after,
        completed_run_report_spend_usd=args.run_report_spend,
        expected_tasks=args.expected_tasks,
    )
    print(json.dumps(scorecard, indent=2))


if __name__ == "__main__":
    main()
