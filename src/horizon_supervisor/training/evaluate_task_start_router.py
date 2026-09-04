from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import tomllib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from horizon_supervisor.benchmark.matched_outcomes import LEARNING_VALID_STATUSES
from horizon_supervisor.training.task_start_router import (
    DEFAULT_ARTIFACT,
    DEFAULT_ROUTES,
    FROZEN_CASCADES,
    TaskStartRouter,
    load_task_start_router,
)

DEFAULT_OUTCOMES = Path(
    "artifacts/official/gate8-wave3-18-task-checkpoint/matched-outcomes-72-v1.jsonl"
)
DEFAULT_TASK_DIR = Path("data/supervisor/terminal-bench-pro-wave-3/tasks")
DEFAULT_REPORT = Path(
    "artifacts/official/gate8-wave3-18-task-checkpoint/"
    "task-start-router-heldout-scorecard-v0.json"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cost(row: dict[str, Any]) -> tuple[float, str]:
    outcome = row["outcome"]
    if outcome.get("estimated_list_cost_usd") is not None:
        return float(outcome["estimated_list_cost_usd"]), "cache-aware-list-price"
    if outcome.get("allocated_provider_cost_usd") is not None:
        return float(outcome["allocated_provider_cost_usd"]), "allocated-provider-spend"
    raise ValueError("held-out outcome is missing an exact comparable cost")


def _load_held_out_panel(
    path: Path,
    *,
    expected_routes: tuple[str, ...],
    expected_tasks: int,
) -> tuple[dict[str, dict[str, dict[str, Any]]], list[str], list[dict[str, Any]]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("held-out outcome dataset is empty")
    expected_route_set = set(expected_routes)
    if len(expected_route_set) != len(expected_routes) or len(expected_routes) != 4:
        raise ValueError("held-out evaluation requires exactly four unique artifact routes")

    by_task: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    task_identity: dict[str, tuple[str, str, str]] = {}
    group_owner: dict[str, str] = {}
    for row in rows:
        if row.get("schema_version") != "matched-model-outcome.v1":
            raise ValueError("unsupported matched-outcome schema")
        task = row.get("task", {})
        if task.get("record_split") != "held_out":
            raise ValueError("frozen evaluator accepts held_out rows only")
        outcome = row.get("outcome", {})
        if outcome.get("status") not in LEARNING_VALID_STATUSES:
            raise ValueError("held-out evaluator requires learning-valid outcomes")
        if row.get("initial_state", {}).get("kind") != "clean_task_start":
            raise ValueError("held-out evaluator requires clean-start outcomes")
        _cost(row)

        task_name = str(task.get("source_task_name", ""))
        task_id = str(task.get("task_id", ""))
        route_id = str(row.get("model", {}).get("route_id", ""))
        matched_group = str(row.get("matched_group_id", ""))
        initial_digest = str(row.get("initial_state", {}).get("digest", ""))
        if not all((task_name, task_id, route_id, matched_group, initial_digest)):
            raise ValueError("held-out row is missing identity fields")
        if route_id not in expected_route_set:
            raise ValueError(f"unexpected held-out route: {route_id}")
        if route_id in by_task[task_name]:
            raise ValueError(f"duplicate held-out task-route pair: {task_name}|{route_id}")
        identity = (task_id, matched_group, initial_digest)
        if task_identity.setdefault(task_name, identity) != identity:
            raise ValueError(f"held-out task routes do not share one clean start: {task_name}")
        if group_owner.setdefault(matched_group, task_name) != task_name:
            raise ValueError("matched group crosses held-out task identities")
        by_task[task_name][route_id] = row

    if len(by_task) != expected_tasks:
        raise ValueError(f"expected {expected_tasks} held-out tasks, found {len(by_task)}")
    for task_name, route_rows in by_task.items():
        if set(route_rows) != expected_route_set:
            raise ValueError(f"held-out panel is not rectangular for {task_name}")
    if len(rows) != expected_tasks * len(expected_routes):
        raise ValueError("held-out panel has missing or extra task-route rows")
    return dict(by_task), sorted(expected_routes), rows


def _task_input(task_name: str, task_dirs: tuple[Path, ...]) -> dict[str, Any]:
    roots = [root / task_name for root in task_dirs if (root / task_name).is_dir()]
    if len(roots) != 1:
        raise ValueError(f"expected one public task directory for {task_name}, found {len(roots)}")
    instruction_path = roots[0] / "instruction.md"
    config_path = roots[0] / "task.toml"
    if not instruction_path.is_file() or not config_path.is_file():
        raise ValueError(f"public task inputs are missing for {task_name}")
    metadata = tomllib.loads(config_path.read_text(encoding="utf-8")).get("metadata", {})
    return {
        "instruction": instruction_path.read_text(encoding="utf-8"),
        "difficulty": str(metadata.get("difficulty", "unknown")),
        "category": str(metadata.get("category", "unknown")),
        "tags": [str(tag) for tag in metadata.get("tags", [])],
    }


def _replay(
    strategy: str,
    by_task: dict[str, dict[str, dict[str, Any]]],
    route_orders: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    decisions = []
    attempt_counts: Counter[str] = Counter()
    for task_name in sorted(by_task):
        attempted_routes = []
        replayed_cost = 0.0
        cost_bases: Counter[str] = Counter()
        completed = False
        for route_id in route_orders[task_name]:
            row = by_task[task_name][route_id]
            cost, basis = _cost(row)
            attempted_routes.append(route_id)
            attempt_counts[route_id] += 1
            replayed_cost += cost
            cost_bases[basis] += 1
            if row["outcome"]["completed"]:
                completed = True
                break
        decisions.append(
            {
                "task": task_name,
                "attempted_routes": attempted_routes,
                "completed": completed,
                "attempt_count": len(attempted_routes),
                "replayed_cost_usd": replayed_cost,
                "cost_basis_counts": dict(sorted(cost_bases.items())),
            }
        )
    return {
        "strategy": strategy,
        "tasks": len(decisions),
        "successes": sum(decision["completed"] for decision in decisions),
        "success_rate": statistics.mean(int(decision["completed"]) for decision in decisions),
        "total_cost_usd": sum(decision["replayed_cost_usd"] for decision in decisions),
        "total_attempts": sum(decision["attempt_count"] for decision in decisions),
        "mean_attempts_per_task": statistics.mean(
            decision["attempt_count"] for decision in decisions
        ),
        "route_attempt_counts": dict(sorted(attempt_counts.items())),
        "stops_after_first_observed_success": True,
        "decisions": decisions,
    }


def _comparison_row(family: str, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "family": family,
        "strategy": summary["strategy"],
        "max_routes": summary.get("max_routes", 1),
        "successes": summary["successes"],
        "success_rate": summary["success_rate"],
        "total_cost_usd": summary["total_cost_usd"],
        "total_attempts": summary["total_attempts"],
    }


def evaluate_frozen_task_start_router(
    outcomes_path: Path,
    artifact_path: Path,
    task_dir: Path | tuple[Path, ...],
    report_path: Path,
    *,
    expected_tasks: int = 18,
) -> dict[str, Any]:
    if artifact_path.resolve() == report_path.resolve():
        raise ValueError("report path must not overwrite the frozen artifact")
    artifact_sha256_before = _sha256_file(artifact_path)
    artifact: TaskStartRouter = load_task_start_router(artifact_path)
    if artifact.development_task_count != 35:
        raise ValueError("frozen held-out evaluation requires the 35-task development artifact")
    if tuple(artifact.routes) != DEFAULT_ROUTES:
        raise ValueError(f"unexpected frozen artifact routes: {artifact.routes}")
    if set(artifact.success_first_margins) != {1, 2, 3, 4}:
        raise ValueError("frozen artifact must contain 1/2/3/4-route policy margins")

    by_task, routes, rows = _load_held_out_panel(
        outcomes_path,
        expected_routes=artifact.routes,
        expected_tasks=expected_tasks,
    )
    task_dirs = (task_dir,) if isinstance(task_dir, Path) else task_dir
    public_inputs = {name: _task_input(name, task_dirs) for name in sorted(by_task)}
    for task_name, task_rows in by_task.items():
        example = next(iter(task_rows.values()))["task"]
        task_input = public_inputs[task_name]
        if str(example.get("difficulty")) != task_input["difficulty"]:
            raise ValueError(f"difficulty mismatch for held-out task {task_name}")
        if str(example.get("category")) != task_input["category"]:
            raise ValueError(f"category mismatch for held-out task {task_name}")

    static_models = []
    for route_id in routes:
        summary = _replay(
            f"always:{route_id}",
            by_task,
            {task_name: (route_id,) for task_name in by_task},
        )
        summary.update({"route_id": route_id, "max_routes": 1})
        static_models.append(summary)
    best_static = max(
        static_models,
        key=lambda row: (row["successes"], -row["total_cost_usd"]),
    )

    fixed_cascades = []
    for max_routes, order in sorted(FROZEN_CASCADES.items()):
        summary = _replay(
            f"frozen-cascade:max-routes={max_routes}",
            by_task,
            {task_name: order for task_name in by_task},
        )
        summary.update(
            {
                "max_routes": max_routes,
                "route_order": list(order),
                "policy_selected_on": "35-task development set",
            }
        )
        fixed_cascades.append(summary)

    learned_policies = []
    for max_routes in range(1, 5):
        orders = {}
        forecasts = {}
        for task_name, task_input in public_inputs.items():
            prediction = artifact.predict_route_order(
                **task_input,
                max_routes=max_routes,
            )
            orders[task_name] = tuple(row["route_id"] for row in prediction)
            forecasts[task_name] = prediction
        summary = _replay(
            f"frozen-learned-task-start-router:max-routes={max_routes}",
            by_task,
            orders,
        )
        summary.update(
            {
                "max_routes": max_routes,
                "success_first_margin": artifact.success_first_margins[max_routes],
                "task_route_forecasts": forecasts,
                "fit_or_tuning_during_evaluation": False,
            }
        )
        learned_policies.append(summary)

    artifact_sha256_after = _sha256_file(artifact_path)
    if artifact_sha256_after != artifact_sha256_before:
        raise RuntimeError("frozen task-start router artifact changed during evaluation")
    comparison = [
        *(_comparison_row("static-model", row) for row in static_models),
        *(_comparison_row("frozen-fixed-cascade", row) for row in fixed_cascades),
        *(_comparison_row("frozen-learned-task-start-router", row) for row in learned_policies),
    ]
    report = {
        "schema_version": "frozen-task-start-router-heldout-scorecard.v0",
        "held_out_data": {
            "path": str(outcomes_path),
            "sha256": _sha256_file(outcomes_path),
            "record_split": "held_out",
            "records": len(rows),
            "tasks": len(by_task),
            "routes": routes,
            "rectangular_and_learning_valid": True,
            "cost_basis_counts": dict(
                sorted(Counter(_cost(row)[1] for row in rows).items())
            ),
        },
        "frozen_artifact": {
            "path": str(artifact_path),
            "sha256_before_evaluation": artifact_sha256_before,
            "sha256_after_evaluation": artifact_sha256_after,
            "hash_preserved": True,
            "training_dataset_sha256": artifact.training_dataset_sha256,
            "development_task_count": artifact.development_task_count,
            "random_seed": artifact.random_seed,
            "success_first_margins": {
                str(key): value for key, value in artifact.success_first_margins.items()
            },
        },
        "static_model_baselines": static_models,
        "best_completion_first_static": best_static,
        "fixed_cascade_baselines": fixed_cascades,
        "learned_task_start_policies": learned_policies,
        "comparison": comparison,
        "evaluation_contract": {
            "fit_calls": 0,
            "tuning_calls": 0,
            "held_out_policy_changes": 0,
            "held_out_rows_required": True,
            "exact_task_route_coverage_required": True,
            "costs": "exact observed comparable cost from each attempted outcome",
            "cascade_stop_signal": "first verifier-confirmed success",
            "policy_inputs": "public task instruction and metadata only",
            "clean_start_replay_only": True,
            "claim_boundary": (
                "This evaluates frozen task-start and clean-restart ordering. It does not "
                "evaluate mid-run state transfer or live continue/switch/stop decisions."
            ),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit-free held-out evaluation of a frozen task-start router"
    )
    parser.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--tasks", type=Path, action="append")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--expected-tasks", type=int, default=18)
    args = parser.parse_args()
    report = evaluate_frozen_task_start_router(
        outcomes_path=args.outcomes,
        artifact_path=args.artifact,
        task_dir=tuple(args.tasks) if args.tasks else DEFAULT_TASK_DIR,
        report_path=args.report,
        expected_tasks=args.expected_tasks,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
