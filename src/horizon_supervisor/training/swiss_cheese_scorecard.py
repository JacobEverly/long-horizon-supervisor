from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from horizon_supervisor.benchmark.matched_outcomes import LEARNING_VALID_STATUSES


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cost(row: dict[str, Any]) -> float:
    value = row["outcome"].get("estimated_list_cost_usd")
    if value is None:
        raise ValueError("replication outcome is missing comparable list cost")
    return float(value)


def _tokens(row: dict[str, Any]) -> int:
    outcome = row["outcome"]
    return int(outcome.get("input_tokens") or 0) + int(outcome.get("output_tokens") or 0)


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot take a percentile of an empty sample")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _task_bootstrap_interval(
    values_by_task: dict[str, list[float]],
    *,
    seed: int,
    samples: int,
) -> list[float]:
    tasks = sorted(values_by_task)
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        selected = [rng.choice(tasks) for _ in tasks]
        values = [value for task in selected for value in values_by_task[task]]
        estimates.append(statistics.mean(values))
    return [_percentile(estimates, 0.025), _percentile(estimates, 0.975)]


def _sequence_result(
    task: str,
    replication: int,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    attempted = []
    cost = 0.0
    tokens = 0
    duration = 0.0
    completed = False
    for row in rows:
        attempted.append(row["model"]["route_id"])
        cost += _cost(row)
        tokens += _tokens(row)
        duration += float(row["outcome"].get("duration_seconds") or 0.0)
        if row["outcome"]["completed"]:
            completed = True
            break
    return {
        "task": task,
        "replication": replication,
        "attempted_routes": attempted,
        "completed": completed,
        "attempts": len(attempted),
        "cost_usd": cost,
        "tokens": tokens,
        "duration_seconds": duration,
    }


def _policy_summary(
    strategy: str,
    family: str,
    sequences: list[dict[str, Any]],
    *,
    route_order: list[str],
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    by_task: dict[str, list[float]] = defaultdict(list)
    for row in sequences:
        by_task[row["task"]].append(float(row["completed"]))
    successes = sum(bool(row["completed"]) for row in sequences)
    return {
        "strategy": strategy,
        "family": family,
        "route_order": route_order,
        "evaluation_units": len(sequences),
        "task_clusters": len(by_task),
        "successes": successes,
        "success_rate": successes / len(sequences),
        "success_rate_task_bootstrap_95": _task_bootstrap_interval(
            by_task, seed=bootstrap_seed, samples=bootstrap_samples
        ),
        "total_cost_usd": sum(row["cost_usd"] for row in sequences),
        "total_tokens": sum(row["tokens"] for row in sequences),
        "total_duration_seconds": sum(row["duration_seconds"] for row in sequences),
        "total_attempts": sum(row["attempts"] for row in sequences),
        "mean_attempts": statistics.mean(row["attempts"] for row in sequences),
        "stops_after_first_verified_success": True,
    }


def _phi(a: list[int], b: list[int]) -> float | None:
    n11 = sum(x == 1 and y == 1 for x, y in zip(a, b, strict=True))
    n10 = sum(x == 1 and y == 0 for x, y in zip(a, b, strict=True))
    n01 = sum(x == 0 and y == 1 for x, y in zip(a, b, strict=True))
    n00 = sum(x == 0 and y == 0 for x, y in zip(a, b, strict=True))
    denominator = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    return (n11 * n00 - n10 * n01) / denominator if denominator else None


def _pareto(policies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frontier = []
    for candidate in policies:
        dominated = any(
            other["successes"] >= candidate["successes"]
            and other["total_cost_usd"] <= candidate["total_cost_usd"]
            and (
                other["successes"] > candidate["successes"]
                or other["total_cost_usd"] < candidate["total_cost_usd"]
            )
            for other in policies
            if other["strategy"] != candidate["strategy"]
        )
        if not dominated:
            frontier.append(
                {
                    key: candidate[key]
                    for key in (
                        "strategy",
                        "family",
                        "route_order",
                        "successes",
                        "success_rate",
                        "total_cost_usd",
                        "total_attempts",
                    )
                }
            )
    return sorted(frontier, key=lambda row: (row["successes"], row["total_cost_usd"]))


def build_swiss_cheese_scorecard(
    manifest_path: Path,
    matrix_path: Path,
    output_path: Path,
    *,
    key_usage_before_usd: float,
    key_usage_after_usd: float,
    completed_run_report_spend_usd: float,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in matrix_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tasks = tuple(manifest["design"]["tasks"])
    routes = tuple(manifest["design"]["routes"])
    confirmatory = tuple(manifest["design"]["confirmatory_replication_indices"])
    bootstrap_seed = int(manifest["analysis"]["bootstrap_seed"])
    bootstrap_samples = int(manifest["analysis"]["bootstrap_samples"])
    if len(rows) != len(tasks) * len(routes) * 3:
        raise ValueError("replication matrix does not have exact 10x5x3 coverage")

    by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        key = (
            row["task"]["source_task_name"],
            row["model"]["route_id"],
            int(row["provenance"]["replication_index"]),
        )
        if key in by_key:
            raise ValueError(f"duplicate replication outcome: {key}")
        if row["task"].get("record_split") != "development":
            raise ValueError("scorecard rejects held-out replication rows")
        if row["outcome"]["status"] not in LEARNING_VALID_STATUSES:
            raise ValueError("scorecard requires learning-valid outcomes")
        by_key[key] = row

    policies: list[dict[str, Any]] = []
    static_by_route: dict[str, dict[str, Any]] = {}
    same_retry_by_route: dict[str, dict[str, Any]] = {}
    heterogeneous_by_order: dict[tuple[str, str], dict[str, Any]] = {}

    for route in routes:
        sequences = [
            _sequence_result(task, rep, [by_key[(task, route, rep)]])
            for task in tasks
            for rep in confirmatory
        ]
        summary = _policy_summary(
            f"static:{route}",
            "static-pass-at-1",
            sequences,
            route_order=[route],
            bootstrap_seed=bootstrap_seed,
            bootstrap_samples=bootstrap_samples,
        )
        static_by_route[route] = summary
        policies.append(summary)

        retry_sequences = []
        for task in tasks:
            for first_rep, second_rep in zip(
                confirmatory, reversed(confirmatory), strict=True
            ):
                retry_sequences.append(
                    _sequence_result(
                        task,
                        first_rep,
                        [
                            by_key[(task, route, first_rep)],
                            by_key[(task, route, second_rep)],
                        ],
                    )
                )
        retry = _policy_summary(
            f"same-model-retry:{route}>{route}",
            "same-model-pass-at-2",
            retry_sequences,
            route_order=[route, route],
            bootstrap_seed=bootstrap_seed,
            bootstrap_samples=bootstrap_samples,
        )
        same_retry_by_route[route] = retry
        policies.append(retry)

    for first, second in itertools.permutations(routes, 2):
        sequences = [
            _sequence_result(
                task,
                rep,
                [by_key[(task, first, rep)], by_key[(task, second, rep)]],
            )
            for task in tasks
            for rep in confirmatory
        ]
        summary = _policy_summary(
            f"heterogeneous:{first}>{second}",
            "heterogeneous-pass-at-2",
            sequences,
            route_order=[first, second],
            bootstrap_seed=bootstrap_seed,
            bootstrap_samples=bootstrap_samples,
        )
        heterogeneous_by_order[(first, second)] = summary
        policies.append(summary)

    named_cascades = manifest["analysis"]["named_cascades"]
    cascade_summaries = []
    for name, order in named_cascades.items():
        sequences = [
            _sequence_result(
                task,
                rep,
                [by_key[(task, route, rep)] for route in order],
            )
            for task in tasks
            for rep in confirmatory
        ]
        summary = _policy_summary(
            name,
            "multi-model-cascade",
            sequences,
            route_order=list(order),
            bootstrap_seed=bootstrap_seed,
            bootstrap_samples=bootstrap_samples,
        )
        cascade_summaries.append(summary)
        policies.append(summary)

    contrasts = []
    for first, second in manifest["analysis"]["predeclared_heterogeneous_contrasts"]:
        heterogeneous = heterogeneous_by_order[(first, second)]
        same = same_retry_by_route[first]
        hetero_by_task: dict[str, list[float]] = defaultdict(list)
        same_by_task: dict[str, list[float]] = defaultdict(list)
        distinct_rescue_tasks = set()
        rescue_opportunities = 0
        rescues = 0
        for task in tasks:
            for first_rep, retry_rep in zip(
                confirmatory, reversed(confirmatory), strict=True
            ):
                first_row = by_key[(task, first, first_rep)]
                second_row = by_key[(task, second, first_rep)]
                retry_row = by_key[(task, first, retry_rep)]
                hetero_value = int(
                    first_row["outcome"]["completed"]
                    or second_row["outcome"]["completed"]
                )
                same_value = int(
                    first_row["outcome"]["completed"]
                    or retry_row["outcome"]["completed"]
                )
                hetero_by_task[task].append(float(hetero_value - same_value))
                same_by_task[task].append(float(same_value))
                if not first_row["outcome"]["completed"]:
                    rescue_opportunities += 1
                    if second_row["outcome"]["completed"]:
                        rescues += 1
                        distinct_rescue_tasks.add(task)
        delta = statistics.mean(
            value for values in hetero_by_task.values() for value in values
        )
        contrasts.append(
            {
                "heterogeneous_strategy": heterogeneous["strategy"],
                "same_model_control": same["strategy"],
                "completion_rate_delta": delta,
                "completion_rate_delta_task_bootstrap_95": (
                    _task_bootstrap_interval(
                        hetero_by_task,
                        seed=bootstrap_seed,
                        samples=bootstrap_samples,
                    )
                ),
                "cost_delta_usd": (
                    heterogeneous["total_cost_usd"] - same["total_cost_usd"]
                ),
                "rescue_opportunities_after_first_failure": rescue_opportunities,
                "heterogeneous_rescues": rescues,
                "distinct_rescue_tasks": sorted(distinct_rescue_tasks),
                "comparison_units_share_identical_first_attempts": True,
            }
        )

    pairwise = []
    for route_a, route_b in itertools.combinations(routes, 2):
        a = [
            int(by_key[(task, route_a, rep)]["outcome"]["completed"])
            for task in tasks
            for rep in confirmatory
        ]
        b = [
            int(by_key[(task, route_b, rep)]["outcome"]["completed"])
            for task in tasks
            for rep in confirmatory
        ]
        both = sum(x and y for x, y in zip(a, b, strict=True))
        either = sum(x or y for x, y in zip(a, b, strict=True))
        pairwise.append(
            {
                "routes": [route_a, route_b],
                "both_success": both,
                "first_only_success": sum(
                    x and not y for x, y in zip(a, b, strict=True)
                ),
                "second_only_success": sum(
                    y and not x for x, y in zip(a, b, strict=True)
                ),
                "both_failure": sum(
                    not x and not y for x, y in zip(a, b, strict=True)
                ),
                "success_jaccard": both / either if either else None,
                "failure_phi": _phi([1 - x for x in a], [1 - y for y in b]),
            }
        )

    coverage = []
    successful_tasks_by_route: dict[str, set[str]] = defaultdict(set)
    for task in tasks:
        row = {"task": task, "models": {}}
        for route in routes:
            results = [
                int(by_key[(task, route, rep)]["outcome"]["completed"])
                for rep in (1, 2, 3)
            ]
            if any(results[rep - 1] for rep in confirmatory):
                successful_tasks_by_route[route].add(task)
            row["models"][route] = {
                "discovery_replication": results[0],
                "confirmatory_replications": results[1:],
                "confirmatory_pass_count": sum(results[1:]),
            }
        coverage.append(row)

    unique_tasks = {}
    for route in routes:
        others = set().union(
            *(successful_tasks_by_route[other] for other in routes if other != route)
        )
        unique_tasks[route] = sorted(successful_tasks_by_route[route] - others)

    stability = []
    for rep in (1, 2, 3):
        counts = {
            route: sum(
                by_key[(task, route, rep)]["outcome"]["completed"] for task in tasks
            )
            for route in routes
        }
        stability.append(
            {
                "replication": rep,
                "role": "outcome-selected discovery" if rep == 1 else "confirmatory",
                "success_counts": counts,
                "completion_order": sorted(routes, key=lambda route: (-counts[route], route)),
            }
        )

    frontier = _pareto(policies)
    predeclared_names = {
        f"heterogeneous:{first}>{second}"
        for first, second in manifest["analysis"]["predeclared_heterogeneous_contrasts"]
    }
    supported = [
        contrast
        for contrast in contrasts
        if contrast["heterogeneous_strategy"] in predeclared_names
        and contrast["completion_rate_delta_task_bootstrap_95"][0] > 0
    ]
    suggestive = [
        contrast
        for contrast in contrasts
        if contrast["completion_rate_delta"] > 0
        and len(contrast["distinct_rescue_tasks"]) >= 2
    ]
    complementarity_decision = (
        "supported"
        if supported
        else "suggestive_not_confirmed"
        if suggestive
        else "not_supported_beyond_retry"
    )

    small_route = manifest["design"]["small_route"]
    small_frontier = [row for row in frontier if small_route in row["route_order"]]
    small_rescue_tasks = set()
    for contrast in contrasts:
        if contrast["heterogeneous_strategy"].endswith(">" + small_route):
            small_rescue_tasks.update(contrast["distinct_rescue_tasks"])
    small_decision = (
        "earned_portfolio_place"
        if small_frontier and small_rescue_tasks
        else "did_not_earn_portfolio_place"
    )

    oracle_units = [
        any(
            by_key[(task, route, rep)]["outcome"]["completed"]
            for route in routes
        )
        for task in tasks
        for rep in confirmatory
    ]
    oracle_tasks = [
        any(
            by_key[(task, route, rep)]["outcome"]["completed"]
            for route in routes
            for rep in confirmatory
        )
        for task in tasks
    ]

    key_delta = key_usage_after_usd - key_usage_before_usd
    report = {
        "schema_version": "swiss-cheese-replication-scorecard.v0",
        "experiment": {
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "matrix_path": str(matrix_path),
            "matrix_sha256": _sha256(matrix_path),
            "tasks": len(tasks),
            "routes": len(routes),
            "replications_per_pair": 3,
            "records": len(rows),
            "confirmatory_replication_indices": list(confirmatory),
            "discovery_replication_excluded_from_confirmatory_claims": True,
        },
        "static_pass_at_1": list(static_by_route.values()),
        "same_model_pass_at_2": list(same_retry_by_route.values()),
        "ordered_heterogeneous_pass_at_2": list(heterogeneous_by_order.values()),
        "named_cascades": cascade_summaries,
        "predeclared_same_vs_different_model_contrasts": contrasts,
        "pairwise_success_overlap_and_failure_correlation": pairwise,
        "strict_unique_confirmatory_tasks_by_model": unique_tasks,
        "swiss_cheese_coverage_matrix": coverage,
        "model_order_stability": stability,
        "success_cost_pareto": frontier,
        "empirical_oracle": {
            "deployable": False,
            "same_replication_any_model_successes": sum(oracle_units),
            "same_replication_units": len(oracle_units),
            "any_confirmatory_replication_task_successes": sum(oracle_tasks),
            "tasks": len(oracle_tasks),
        },
        "decision": {
            "repeatable_cross_model_complementarity": complementarity_decision,
            "statistically_supported_predeclared_contrasts": [
                row["heterogeneous_strategy"] for row in supported
            ],
            "suggestive_contrasts": [
                row["heterogeneous_strategy"] for row in suggestive
            ],
            "small_model": small_decision,
            "small_model_pareto_strategies": [
                row["strategy"] for row in small_frontier
            ],
            "small_model_distinct_rescue_tasks": sorted(small_rescue_tasks),
            "claim_guard": (
                "The task panel was selected for first-replication disagreement. "
                "Only replications 2 and 3 support repeatability claims; the "
                "experiment remains small and task-cluster uncertainty is required."
            ),
        },
        "spend_audit": {
            "basis": "dedicated OpenRouter key usage delta",
            "dedicated_key_usage_before_usd": key_usage_before_usd,
            "dedicated_key_usage_after_usd": key_usage_after_usd,
            "exact_incremental_spend_usd": key_delta,
            "completed_run_report_spend_usd": completed_run_report_spend_usd,
            "key_delta_minus_run_reports_usd": (
                key_delta - completed_run_report_spend_usd
            ),
            "under_twenty_usd": key_delta < 20.0,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the Swiss-cheese replication")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--key-usage-before", required=True, type=float)
    parser.add_argument("--key-usage-after", required=True, type=float)
    parser.add_argument("--run-report-spend", required=True, type=float)
    args = parser.parse_args()
    report = build_swiss_cheese_scorecard(
        args.manifest,
        args.matrix,
        args.output,
        key_usage_before_usd=args.key_usage_before,
        key_usage_after_usd=args.key_usage_after,
        completed_run_report_spend_usd=args.run_report_spend,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
