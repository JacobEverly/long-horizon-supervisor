from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import tomllib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from horizon_supervisor.benchmark.matched_outcomes import LEARNING_VALID_STATUSES

DEFAULT_OUTCOMES = Path(
    "artifacts/official/gate8-twelve-task-development/matched-outcomes-48-v1.jsonl"
)
DEFAULT_TASKS = Path("data/supervisor/terminal-bench-pro-wave-1/tasks")
DEFAULT_PANEL = Path("data/supervisor/terminal-bench-pro-panel-v0.jsonl")
DEFAULT_REPORT = Path("artifacts/official/gate8-twelve-task-development/route-baseline-v0.json")
SUCCESS_FIRST_MARGINS = (0.0, 0.02, 0.05, 0.1, 0.2, 1.0)


def _comparable_cost(row: dict[str, Any]) -> float:
    """Use the portable catalog price, with provider allocation as legacy fallback."""
    outcome = row["outcome"]
    value = outcome.get("estimated_list_cost_usd")
    if value is not None:
        return float(value)
    value = outcome.get("allocated_provider_cost_usd")
    if value is not None:
        return float(value)
    raise ValueError("outcome has neither estimated nor allocated cost")


def _cost_basis(row: dict[str, Any]) -> str:
    return (
        "cache-aware-list-price"
        if row["outcome"].get("estimated_list_cost_usd") is not None
        else "allocated-provider-spend"
    )


def _load_rows(paths: Path | tuple[Path, ...]) -> list[dict[str, Any]]:
    source_paths = (paths,) if isinstance(paths, Path) else paths
    rows = [
        json.loads(line)
        for path in source_paths
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    if not rows:
        raise ValueError("matched-outcome dataset is empty")
    return rows


def _rectangular_panel(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, dict[str, Any]]], list[str]]:
    by_task: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("schema_version") != "matched-model-outcome.v1":
            raise ValueError("unsupported matched-outcome schema")
        if row["outcome"]["status"] not in LEARNING_VALID_STATUSES:
            raise ValueError("routing baselines require learning-valid outcomes only")
        task_name = row["task"]["source_task_name"]
        route_id = row["model"]["route_id"]
        if route_id in by_task[task_name]:
            raise ValueError(f"duplicate task-route pair: {task_name}|{route_id}")
        by_task[task_name][route_id] = row

    route_sets = {tuple(sorted(route_rows)) for route_rows in by_task.values()}
    if len(route_sets) != 1:
        raise ValueError("matched-outcome dataset is not rectangular")
    routes = list(next(iter(route_sets)))
    if len(rows) != len(by_task) * len(routes):
        raise ValueError("matched-outcome dataset has missing or extra pairs")
    for task_name, route_rows in by_task.items():
        group_ids = {row["matched_group_id"] for row in route_rows.values()}
        if len(group_ids) != 1:
            raise ValueError(f"task {task_name} does not share one matched group")
    return dict(by_task), routes


def _task_document(task_name: str, task_dirs: tuple[Path, ...]) -> str:
    matches = [task_dir / task_name for task_dir in task_dirs if (task_dir / task_name).is_dir()]
    if len(matches) != 1:
        raise ValueError(f"expected one task directory for {task_name}, found {len(matches)}")
    root = matches[0]
    instruction_path = root / "instruction.md"
    config_path = root / "task.toml"
    if not instruction_path.exists() or not config_path.exists():
        raise ValueError(f"task inputs are missing for {task_name}")
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    metadata = config.get("metadata", {})
    tags = " ".join(f"tag_{str(tag).replace('-', '_')}" for tag in metadata.get("tags", []))
    difficulty = str(metadata.get("difficulty", "unknown")).replace("-", "_")
    category = str(metadata.get("category", "unknown")).replace("-", "_")
    return "\n".join(
        (
            f"difficulty_{difficulty} category_{category} {tags}",
            instruction_path.read_text(encoding="utf-8"),
        )
    )


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float]:
    if total == 0:
        return [0.0, 1.0]
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    radius = (
        z * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2)) / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _route_summary(
    by_task: dict[str, dict[str, dict[str, Any]]], routes: list[str]
) -> list[dict[str, Any]]:
    summaries = []
    for route_id in routes:
        route_rows = [rows[route_id] for rows in by_task.values()]
        summaries.append(
            {
                "strategy": f"always:{route_id}",
                "route_id": route_id,
                "tasks": len(route_rows),
                "successes": sum(row["outcome"]["completed"] for row in route_rows),
                "success_rate": statistics.mean(
                    int(row["outcome"]["completed"]) for row in route_rows
                ),
                "total_cost_usd": sum(_comparable_cost(row) for row in route_rows),
                "median_latency_seconds": statistics.median(
                    float(row["outcome"]["duration_seconds"]) for row in route_rows
                ),
            }
        )
    return summaries


def _policy_summary(
    strategy: str,
    choices: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "strategy": strategy,
        "tasks": len(choices),
        "successes": sum(choice["outcome"]["completed"] for choice in choices),
        "success_rate": statistics.mean(int(choice["outcome"]["completed"]) for choice in choices),
        "total_cost_usd": sum(_comparable_cost(choice) for choice in choices),
        "selected_route_counts": dict(
            sorted(Counter(choice["model"]["route_id"] for choice in choices).items())
        ),
    }


def _cascade_summary(
    strategy: str,
    by_task: dict[str, dict[str, dict[str, Any]]],
    route_orders: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    decisions = []
    route_attempt_counts: Counter[str] = Counter()
    for task_name in sorted(by_task):
        attempted_routes = []
        task_cost = 0.0
        task_latency = 0.0
        completed = False
        for route_id in route_orders[task_name]:
            row = by_task[task_name][route_id]
            attempted_routes.append(route_id)
            route_attempt_counts[route_id] += 1
            task_cost += _comparable_cost(row)
            task_latency += float(row["outcome"]["duration_seconds"])
            if row["outcome"]["completed"]:
                completed = True
                break
        decisions.append(
            {
                "task": task_name,
                "attempted_routes": attempted_routes,
                "completed": completed,
                "attempt_count": len(attempted_routes),
                "replayed_cost_usd": task_cost,
                "replayed_latency_seconds": task_latency,
            }
        )
    return {
        "strategy": strategy,
        "tasks": len(decisions),
        "successes": sum(decision["completed"] for decision in decisions),
        "success_rate": statistics.mean(
            int(decision["completed"]) for decision in decisions
        ),
        "total_cost_usd": sum(decision["replayed_cost_usd"] for decision in decisions),
        "total_latency_seconds": sum(
            decision["replayed_latency_seconds"] for decision in decisions
        ),
        "total_attempts": sum(decision["attempt_count"] for decision in decisions),
        "mean_attempts_per_task": statistics.mean(
            decision["attempt_count"] for decision in decisions
        ),
        "route_attempt_counts": dict(sorted(route_attempt_counts.items())),
        "stops_after_observed_success": True,
        "decisions": decisions,
    }


def _fixed_cascade_baselines(
    by_task: dict[str, dict[str, dict[str, Any]]],
    routes: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_cascades = []
    for max_routes in range(1, len(routes) + 1):
        for order in itertools.permutations(routes, max_routes):
            summary = _cascade_summary(
                "fixed-cascade:" + ">".join(order),
                by_task,
                {task_name: order for task_name in by_task},
            )
            summary["max_routes"] = max_routes
            summary["route_order"] = list(order)
            all_cascades.append(summary)

    completion_first_frontier = []
    for max_routes in range(1, len(routes) + 1):
        candidates = [
            summary
            for summary in all_cascades
            if summary["max_routes"] == max_routes
        ]
        completion_first_frontier.append(
            max(
                candidates,
                key=lambda row: (
                    row["successes"],
                    -row["total_cost_usd"],
                    -row["total_latency_seconds"],
                    row["strategy"],
                ),
            )
        )

    pareto_frontier = [
        candidate
        for candidate in all_cascades
        if not any(
            other["successes"] >= candidate["successes"]
            and other["total_cost_usd"] <= candidate["total_cost_usd"]
            and (
                other["successes"] > candidate["successes"]
                or other["total_cost_usd"] < candidate["total_cost_usd"]
            )
            for other in all_cascades
            if other["strategy"] != candidate["strategy"]
        )
    ]
    pareto_frontier.sort(key=lambda row: (row["successes"], row["total_cost_usd"]))
    return completion_first_frontier, pareto_frontier


def _oracle_summary(
    by_task: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    choices = []
    for task_rows in by_task.values():
        successful = [row for row in task_rows.values() if row["outcome"]["completed"]]
        candidates = successful or list(task_rows.values())
        choices.append(
            min(
                candidates,
                key=_comparable_cost,
            )
        )
    summary = _policy_summary("hindsight:cheapest-success", choices)
    summary["deployable"] = False
    return summary


def _task_documents(
    task_names: list[str], task_dirs: tuple[Path, ...]
) -> dict[str, str]:
    return {name: _task_document(name, task_dirs) for name in task_names}


def _fit_task_route_probabilities(
    by_task: dict[str, dict[str, dict[str, Any]]],
    routes: list[str],
    documents: dict[str, str],
    train_tasks: list[str],
    target_task: str,
) -> dict[str, dict[str, float]]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        max_features=1_024,
        sublinear_tf=True,
    )
    train_matrix = vectorizer.fit_transform([documents[name] for name in train_tasks])
    target_matrix = vectorizer.transform([documents[target_task]])
    predictions = {}
    for route_id in routes:
        labels = np.asarray(
            [
                int(by_task[name][route_id]["outcome"]["completed"])
                for name in train_tasks
            ],
            dtype=np.int64,
        )
        if len(set(labels.tolist())) == 1:
            probability = float(labels[0])
        else:
            model = LogisticRegression(
                C=0.25,
                solver="liblinear",
                max_iter=1_000,
                random_state=17,
            )
            model.fit(train_matrix, labels)
            probability = float(model.predict_proba(target_matrix)[0, 1])
        train_costs = [_comparable_cost(by_task[name][route_id]) for name in train_tasks]
        predictions[route_id] = {
            "success_probability": probability,
            "forecast_cost_usd": statistics.median(train_costs),
        }
    return predictions


def _task_held_out_probabilities(
    by_task: dict[str, dict[str, dict[str, Any]]],
    routes: list[str],
    documents: dict[str, str],
) -> dict[str, dict[str, dict[str, float]]]:
    task_names = sorted(by_task)
    predictions = {}
    for held_out in task_names:
        predictions[held_out] = _fit_task_route_probabilities(
            by_task,
            routes,
            documents,
            [name for name in task_names if name != held_out],
            held_out,
        )
    return predictions


def _route_order(
    scores: dict[str, dict[str, float]],
    routes: list[str],
    margin: float,
) -> tuple[str, ...]:
    remaining = set(routes)
    ordered = []
    while remaining:
        best_probability = max(
            scores[route_id]["success_probability"] for route_id in remaining
        )
        eligible = [
            route_id
            for route_id in remaining
            if scores[route_id]["success_probability"] >= best_probability - margin
        ]
        selected = min(
            eligible,
            key=lambda route_id: (
                scores[route_id]["forecast_cost_usd"],
                route_id,
            ),
        )
        ordered.append(selected)
        remaining.remove(selected)
    return tuple(ordered)


def _nested_task_held_out_cascades(
    by_task: dict[str, dict[str, dict[str, Any]]],
    routes: list[str],
    documents: dict[str, str],
) -> list[dict[str, Any]]:
    task_names = sorted(by_task)
    if len(task_names) < 4:
        fixed_frontier, _ = _fixed_cascade_baselines(by_task, routes)
        unavailable = []
        for summary in fixed_frontier:
            summary = json.loads(json.dumps(summary))
            summary.update(
                {
                    "strategy": (
                        "nested-task-held-out-cascade-unavailable:"
                        f"max-routes={summary['max_routes']}"
                    ),
                    "nested_evaluation_available": False,
                    "uses_held_out_task_outcome": None,
                    "hyperparameter_selection": (
                        "unavailable with fewer than four task groups"
                    ),
                    "selected_margin_counts": {},
                    "selected_order_counts": {},
                }
            )
            unavailable.append(summary)
        return unavailable

    prediction_cache: dict[
        tuple[tuple[str, ...], str], dict[str, dict[str, float]]
    ] = {}

    def predictions(train_tasks: list[str], target_task: str):
        key = (tuple(train_tasks), target_task)
        if key not in prediction_cache:
            prediction_cache[key] = _fit_task_route_probabilities(
                by_task,
                routes,
                documents,
                train_tasks,
                target_task,
            )
        return prediction_cache[key]

    policies = []
    for max_routes in range(1, len(routes) + 1):
        outer_orders = {}
        selected_margins = {}
        for outer_task in task_names:
            outer_train = [name for name in task_names if name != outer_task]
            margin_scores = []
            for margin in SUCCESS_FIRST_MARGINS:
                inner_orders = {}
                for inner_task in outer_train:
                    inner_train = [
                        name for name in outer_train if name != inner_task
                    ]
                    inner_orders[inner_task] = _route_order(
                        predictions(inner_train, inner_task),
                        routes,
                        margin,
                    )[:max_routes]
                inner_summary = _cascade_summary(
                    f"inner:margin={margin:g}",
                    {name: by_task[name] for name in outer_train},
                    inner_orders,
                )
                margin_scores.append(
                    (
                        inner_summary["successes"],
                        -inner_summary["total_cost_usd"],
                        -inner_summary["total_attempts"],
                        -margin,
                        margin,
                    )
                )
            selected_margin = max(margin_scores)[-1]
            selected_margins[outer_task] = selected_margin
            outer_orders[outer_task] = _route_order(
                predictions(outer_train, outer_task),
                routes,
                selected_margin,
            )[:max_routes]

        summary = _cascade_summary(
            f"nested-task-held-out-cascade:max-routes={max_routes}",
            by_task,
            outer_orders,
        )
        summary.update(
            {
                "max_routes": max_routes,
                "nested_evaluation_available": True,
                "uses_held_out_task_outcome": False,
                "hyperparameter_selection": "nested leave-one-task-out",
                "selected_margin_counts": dict(
                    sorted(Counter(selected_margins.values()).items())
                ),
                "selected_order_counts": dict(
                    sorted(
                        Counter(">".join(order) for order in outer_orders.values()).items()
                    )
                ),
            }
        )
        policies.append(summary)
    return policies


def _held_out_policies(
    by_task: dict[str, dict[str, dict[str, Any]]],
    routes: list[str],
    predictions: dict[str, dict[str, dict[str, float]]],
) -> list[dict[str, Any]]:
    policies = []
    for margin in SUCCESS_FIRST_MARGINS:
        choices = []
        decisions = []
        for task_name in sorted(by_task):
            scores = predictions[task_name]
            best_probability = max(row["success_probability"] for row in scores.values())
            eligible = [
                route_id
                for route_id in routes
                if scores[route_id]["success_probability"] >= best_probability - margin
            ]
            selected = min(
                eligible,
                key=lambda route_id: (
                    scores[route_id]["forecast_cost_usd"],
                    route_id,
                ),
            )
            choices.append(by_task[task_name][selected])
            decisions.append(
                {
                    "task": task_name,
                    "selected_route": selected,
                    "success": by_task[task_name][selected]["outcome"]["completed"],
                    "observed_cost_usd": _comparable_cost(by_task[task_name][selected]),
                    "predicted_success_probability": scores[selected]["success_probability"],
                    "eligible_route_count": len(eligible),
                }
            )
        summary = _policy_summary(f"task-held-out:margin={margin:g}", choices)
        summary.update(
            {
                "success_probability_margin": margin,
                "uses_held_out_task_outcome": False,
                "decisions": decisions,
            }
        )
        policies.append(summary)
    return policies


def _pareto_strategies(strategies: list[dict[str, Any]]) -> list[str]:
    efficient = []
    for candidate in strategies:
        dominated = any(
            other["successes"] >= candidate["successes"]
            and other["total_cost_usd"] <= candidate["total_cost_usd"]
            and (
                other["successes"] > candidate["successes"]
                or other["total_cost_usd"] < candidate["total_cost_usd"]
            )
            for other in strategies
            if other["strategy"] != candidate["strategy"]
        )
        if not dominated:
            efficient.append(candidate["strategy"])
    return efficient


def _wave_representation(
    by_task: dict[str, dict[str, dict[str, Any]]],
    panel_path: Path,
    excluded_task_names: tuple[str, ...] = (),
) -> dict[str, Any]:
    panel_rows = [json.loads(line) for line in panel_path.read_text(encoding="utf-8").splitlines()]
    observed_names = set(by_task)
    observed_waves = sorted(
        {int(row["wave"]) for row in panel_rows if row["source_task_name"] in observed_names}
    )
    excluded_set = set(excluded_task_names)
    excluded_frozen_rows = [
        row
        for row in panel_rows
        if int(row["wave"]) in observed_waves and row["source_task_name"] in excluded_set
    ]
    wave_rows = [
        row
        for row in panel_rows
        if int(row["wave"]) in observed_waves and row["source_task_name"] not in excluded_set
    ]
    observed_rows = [row for row in wave_rows if row["source_task_name"] in observed_names]

    def counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
        return dict(sorted(Counter(str(row[key]) for row in rows).items()))

    all_categories = {row["category"] for row in wave_rows}
    total_variation = 0.5 * sum(
        abs(
            sum(row["category"] == category for row in observed_rows) / len(observed_rows)
            - sum(row["category"] == category for row in wave_rows) / len(wave_rows)
        )
        for category in all_categories
    )
    return {
        "frozen_wave_tasks": len(wave_rows),
        "excluded_frozen_task_names": sorted(
            row["source_task_name"] for row in excluded_frozen_rows
        ),
        "observed_waves": observed_waves,
        "completed_matched_tasks": len(observed_rows),
        "remaining_matched_tasks": len(wave_rows) - len(observed_rows),
        "missing_task_names": sorted(
            row["source_task_name"]
            for row in wave_rows
            if row["source_task_name"] not in observed_names
        ),
        "observed_difficulty_counts": counts(observed_rows, "difficulty"),
        "wave_difficulty_counts": counts(wave_rows, "difficulty"),
        "observed_category_counts": counts(observed_rows, "category"),
        "wave_category_counts": counts(wave_rows, "category"),
        "category_total_variation_distance": total_variation,
        "is_full_frozen_wave": len(observed_rows) == len(wave_rows),
    }


def analyze_route_baseline(
    outcomes_path: Path | tuple[Path, ...] = DEFAULT_OUTCOMES,
    task_dir: Path | tuple[Path, ...] = DEFAULT_TASKS,
    panel_path: Path = DEFAULT_PANEL,
    report_path: Path = DEFAULT_REPORT,
    include_routes: tuple[str, ...] = (),
    include_tasks: tuple[str, ...] = (),
    exclude_tasks: tuple[str, ...] = (),
) -> dict[str, Any]:
    if include_tasks and exclude_tasks:
        raise ValueError("include_tasks and exclude_tasks cannot be used together")
    rows = _load_rows(outcomes_path)
    if include_tasks:
        include_task_set = set(include_tasks)
        rows = [row for row in rows if row["task"]["source_task_name"] in include_task_set]
        observed_tasks = {row["task"]["source_task_name"] for row in rows}
        if observed_tasks != include_task_set:
            raise ValueError(
                "included tasks are missing from matched outcomes: "
                f"{sorted(include_task_set - observed_tasks)}"
            )
    if exclude_tasks:
        exclude_task_set = set(exclude_tasks)
        observed_tasks = {row["task"]["source_task_name"] for row in rows}
        if not exclude_task_set <= observed_tasks:
            raise ValueError(
                "excluded tasks are missing from matched outcomes: "
                f"{sorted(exclude_task_set - observed_tasks)}"
            )
        rows = [row for row in rows if row["task"]["source_task_name"] not in exclude_task_set]
    if include_routes:
        include_route_set = set(include_routes)
        rows = [row for row in rows if row["model"]["route_id"] in include_route_set]
        observed_routes = {row["model"]["route_id"] for row in rows}
        if observed_routes != include_route_set:
            raise ValueError(
                "included routes are missing from matched outcomes: "
                f"{sorted(include_route_set - observed_routes)}"
            )
    by_task, routes = _rectangular_panel(rows)
    task_dirs = (task_dir,) if isinstance(task_dir, Path) else task_dir
    route_baselines = _route_summary(by_task, routes)
    best_static = max(
        route_baselines,
        key=lambda row: (row["successes"], -row["total_cost_usd"]),
    )
    oracle = _oracle_summary(by_task)
    documents = _task_documents(sorted(by_task), task_dirs)
    predictions = _task_held_out_probabilities(by_task, routes, documents)
    learned_policies = _held_out_policies(by_task, routes, predictions)
    fixed_cascade_frontier, fixed_cascade_pareto = _fixed_cascade_baselines(
        by_task,
        routes,
    )
    nested_cascades = _nested_task_held_out_cascades(by_task, routes, documents)
    deployable_strategies = route_baselines + learned_policies
    best_learned = max(
        learned_policies,
        key=lambda row: (row["successes"], -row["total_cost_usd"]),
    )

    patterns: Counter[str] = Counter()
    for task_rows in by_task.values():
        successes = sum(row["outcome"]["completed"] for row in task_rows.values())
        patterns[
            "all_success"
            if successes == len(routes)
            else "all_failure"
            if successes == 0
            else "discriminating"
        ] += 1
    discriminating = patterns["discriminating"]
    task_count = len(by_task)
    representation = _wave_representation(by_task, panel_path, exclude_tasks)
    route_label_balance = {
        route_id: {
            "successes": sum(
                by_task[task_name][route_id]["outcome"]["completed"] for task_name in by_task
            ),
            "failures": sum(
                not by_task[task_name][route_id]["outcome"]["completed"] for task_name in by_task
            ),
        }
        for route_id in routes
    }
    sufficiency_checks = {
        "at_least_50_representative_matched_tasks": task_count >= 50,
        "at_least_20_discriminating_tasks": discriminating >= 20,
        "every_route_has_positive_and_negative_labels": all(
            counts["successes"] > 0 and counts["failures"] > 0
            for counts in route_label_balance.values()
        ),
        "oracle_shows_routing_headroom": oracle["successes"] > best_static["successes"],
        "task_held_out_policy_beats_best_static_success": (
            best_learned["successes"] > best_static["successes"]
        ),
        "two_route_fixed_cascade_beats_best_static_success": (
            fixed_cascade_frontier[1]["successes"] > best_static["successes"]
        ),
        "nested_cascade_reaches_oracle_success": (
            nested_cascades[-1]["nested_evaluation_available"]
            and nested_cascades[-1]["successes"] == oracle["successes"]
        ),
        "all_observed_waves_complete": representation["is_full_frozen_wave"],
    }
    report = {
        "schema_version": "task-route-baseline.v0",
        "data": {
            "path": (str(outcomes_path) if isinstance(outcomes_path, Path) else None),
            "paths": [
                str(path)
                for path in ((outcomes_path,) if isinstance(outcomes_path, Path) else outcomes_path)
            ],
            "included_task_names": sorted(include_tasks),
            "excluded_task_names": sorted(exclude_tasks),
            "records": len(rows),
            "tasks": task_count,
            "routes": len(routes),
            "successes": sum(row["outcome"]["completed"] for row in rows),
            "failures": sum(not row["outcome"]["completed"] for row in rows),
            "task_pattern_counts": dict(sorted(patterns.items())),
            "discriminating_task_rate": discriminating / task_count,
            "discriminating_task_rate_wilson_95": _wilson_interval(discriminating, task_count),
            "route_label_balance": route_label_balance,
            "learning_status_counts": dict(
                sorted(Counter(row["outcome"]["status"] for row in rows).items())
            ),
            "cost_basis_counts": dict(sorted(Counter(_cost_basis(row) for row in rows).items())),
            "cost_comparison_basis": (
                "cache-aware catalog list price when present; allocated provider "
                "spend only as a legacy fallback"
            ),
            "all_rows_learning_valid_and_rectangular": True,
        },
        "benchmark_representation": representation,
        "static_route_baselines": route_baselines,
        "best_completion_first_static": best_static,
        "task_held_out_policies": learned_policies,
        "best_task_held_out_policy": best_learned,
        "fixed_cascade_completion_first_frontier": fixed_cascade_frontier,
        "fixed_cascade_cost_completion_pareto_frontier": fixed_cascade_pareto,
        "nested_task_held_out_cascades": nested_cascades,
        "hindsight_upper_bound": oracle,
        "deployable_pareto_strategies": _pareto_strategies(deployable_strategies),
        "data_sufficiency": {
            "checks": sufficiency_checks,
            "ready_for_general_learned_router_claim": all(sufficiency_checks.values()),
            "representative_matched_task_target": 50,
            "representative_task_gap": max(0, 50 - task_count),
            "discriminating_task_target": 20,
            "discriminating_task_gap": max(0, 20 - discriminating),
            "estimated_tasks_for_20_discriminating_at_observed_rate": (
                math.ceil(20 / (discriminating / task_count)) if discriminating else None
            ),
            "next_collection_gate": (
                f"Complete the {representation['remaining_matched_tasks']} missing tasks "
                f"across observed waves {representation['observed_waves']}, then use "
                "outcome-blind proportional expansion plus a separately "
                "reported sentinel-screened contrast set."
            ),
        },
        "evaluation_contract": {
            "split_unit": "task",
            "cross_validation": "leave-one-task-out",
            "model_input": "public task instruction and task.toml metadata only",
            "current_task_outcomes_used_as_features": False,
            "agent_protocol_failures_are_negative_labels": True,
            "current_task_costs_used_for_selection": False,
            "cost_forecast": "training-fold median by route",
            "cascade_stop_signal": "observed verifier-confirmed success",
            "cascade_replay_assumption": (
                "Each fallback attempt replays its matched clean-start outcome. "
                "This estimates restart-and-escalate behavior; it does not prove "
                "that a mid-run state handoff would reproduce the same outcome."
            ),
            "nested_cascade_hyperparameter_selection": (
                "For each outer held-out task, the probability margin is selected "
                "using leave-one-task-out evaluation inside the remaining tasks."
            ),
            "identity_cold_start_guard": (
                "This first scorer estimates each calibrated deployment separately. A "
                "new deployment can enter immediately through the cheapest/static fallback, "
                "but needs a small matched calibration panel before receiving task-specific "
                "scores; the full task model does not need to be retrained."
            ),
        },
        "interpretation_guard": (
            f"The current {task_count} task groups support an honest plumbing and "
            "headroom check, not a "
            "general routing claim. The hindsight strategy is an upper bound. The task-held-"
            "out policies are deployable simulations, but their estimates have high variance "
            "until the representative and discriminating-task targets are met."
        ),
        "supervisor_result": {
            "best_single_model_successes": best_static["successes"],
            "two_route_fixed_cascade_successes": fixed_cascade_frontier[1][
                "successes"
            ],
            "two_route_fixed_cascade_cost_usd": fixed_cascade_frontier[1][
                "total_cost_usd"
            ],
            "full_fixed_cascade_successes": fixed_cascade_frontier[-1][
                "successes"
            ],
            "full_fixed_cascade_cost_usd": fixed_cascade_frontier[-1][
                "total_cost_usd"
            ],
            "full_fixed_cascade_beats_single_on_completion": (
                fixed_cascade_frontier[-1]["successes"] > best_static["successes"]
            ),
            "full_fixed_cascade_beats_single_on_model_cost": (
                fixed_cascade_frontier[-1]["total_cost_usd"]
                < best_static["total_cost_usd"]
            ),
            "learned_ordering_beats_fixed_ordering": any(
                learned["successes"] > fixed["successes"]
                or (
                    learned["successes"] == fixed["successes"]
                    and learned["total_cost_usd"] < fixed["total_cost_usd"]
                )
                for learned, fixed in zip(
                    nested_cascades,
                    fixed_cascade_frontier,
                    strict=True,
                )
            ),
            "claim_boundary": (
                "This is clean-start fallback replay on one attempt per task-model "
                "pair, not proof of replicated production reliability or mid-run "
                "state transfer."
            ),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate task-held-out routing baselines on matched outcomes"
    )
    parser.add_argument("--outcomes", type=Path, action="append")
    parser.add_argument("--include-route", action="append")
    parser.add_argument("--include-task", action="append")
    parser.add_argument("--exclude-task", action="append")
    parser.add_argument("--tasks", type=Path, action="append")
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = analyze_route_baseline(
        outcomes_path=(tuple(args.outcomes) if args.outcomes else DEFAULT_OUTCOMES),
        task_dir=tuple(args.tasks) if args.tasks else DEFAULT_TASKS,
        panel_path=args.panel,
        report_path=args.report,
        include_routes=tuple(args.include_route or ()),
        include_tasks=tuple(args.include_task or ()),
        exclude_tasks=tuple(args.exclude_task or ()),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
