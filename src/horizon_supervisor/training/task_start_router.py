from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from horizon_supervisor.benchmark.matched_outcomes import LEARNING_VALID_STATUSES
from horizon_supervisor.training.route_baseline import SUCCESS_FIRST_MARGINS

DEFAULT_DATASET = Path("data/supervisor/gate8-development-task-route-v0.jsonl")
DEFAULT_ARTIFACT = Path(
    "artifacts/official/task-start-router-development-v0/task-start-router-v0.joblib"
)
DEFAULT_REPORT = Path(
    "artifacts/official/task-start-router-development-v0/nested-loocv-report-v0.json"
)
DEFAULT_ROUTES = (
    "gate7/fixed-flash",
    "gate7/fixed-glm",
    "gate7/fixed-kimi",
    "gate7/fixed-qwen",
)
FROZEN_CASCADES = {
    2: ("gate7/fixed-flash", "gate7/fixed-qwen"),
    3: ("gate7/fixed-flash", "gate7/fixed-qwen", "gate7/fixed-glm"),
    4: (
        "gate7/fixed-flash",
        "gate7/fixed-qwen",
        "gate7/fixed-glm",
        "gate7/fixed-kimi",
    ),
}
RANDOM_SEED = 17
_FORBIDDEN_PATH_PATTERN = re.compile(
    r"^(?:gate8[-_]wave3|terminal[-_]bench[-_]pro[-_]wave[-_]3|wave[-_]?3|held[-_]?out)",
    re.IGNORECASE,
)

# ``python -m`` executes this file as ``__main__``. Register the canonical module
# name so the persisted classes remain importable in a fresh Python process.
if __name__ == "__main__":
    sys.modules["horizon_supervisor.training.task_start_router"] = sys.modules[__name__]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _guard_development_path(value: str | Path, *, label: str) -> None:
    if any(_FORBIDDEN_PATH_PATTERN.search(part) for part in Path(value).parts):
        raise ValueError(f"{label} references a held-out/Wave 3 path: {value}")


def task_document(task_input: dict[str, Any]) -> str:
    """Create the public task-only text representation used at train and inference time."""
    tags = " ".join(
        f"tag_{str(tag).replace('-', '_')}" for tag in task_input.get("tags", [])
    )
    difficulty = str(task_input.get("difficulty", "unknown")).replace("-", "_")
    category = str(task_input.get("category", "unknown")).replace("-", "_")
    instruction = str(task_input["instruction"])
    return "\n".join(
        (
            f"difficulty_{difficulty} category_{category} {tags}",
            instruction,
        )
    )


def _read_development_rows(
    dataset_path: Path,
    *,
    expected_routes: tuple[str, ...],
    expected_tasks: int | None,
) -> tuple[dict[str, dict[str, dict[str, Any]]], list[str], dict[str, str]]:
    _guard_development_path(dataset_path, label="training dataset")
    rows = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("task-route training dataset is empty")
    if not expected_routes or len(expected_routes) != len(set(expected_routes)):
        raise ValueError("expected routes must be non-empty and unique")

    expected_route_set = set(expected_routes)
    by_task: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    task_identity: dict[str, tuple[str, str, str, str]] = {}
    matched_group_owner: dict[str, str] = {}
    leakage_group_owner: dict[str, str] = {}
    documents: dict[str, str] = {}

    for row in rows:
        if row.get("schema_version") != "supervisor-task-route.v0":
            raise ValueError("unsupported task-route schema")
        if row.get("record_split") != "development":
            raise ValueError("router training accepts development rows only")
        target = row.get("target", {})
        if target.get("status") not in LEARNING_VALID_STATUSES:
            raise ValueError("router training requires learning-valid targets")
        if target.get("cost_usd") is None:
            raise ValueError("router training target is missing comparable cost")
        if row.get("initial_state", {}).get("kind") != "clean_task_start":
            raise ValueError("task-start router requires clean-start examples")

        task_input = row.get("input", {})
        task_name = str(task_input.get("source_task_name", ""))
        task_id = str(task_input.get("task_id", ""))
        route_id = str(row.get("candidate", {}).get("route_id", ""))
        matched_group = str(row.get("matched_group_id", ""))
        leakage_group = str(row.get("leakage_group", ""))
        initial_digest = str(row["initial_state"].get("digest", ""))
        if not all((task_name, task_id, route_id, matched_group, leakage_group, initial_digest)):
            raise ValueError("task-route row is missing identity fields")
        if route_id not in expected_route_set:
            raise ValueError(f"unexpected route in training data: {route_id}")
        if route_id in by_task[task_name]:
            raise ValueError(f"duplicate task-route pair: {task_name}|{route_id}")
        if row.get("logged_action") != {
            "action": "start_model",
            "target_route_id": route_id,
        }:
            raise ValueError(f"logged action does not match candidate: {task_name}|{route_id}")
        available_routes = {
            action.get("target_route_id")
            for action in row.get("available_actions", [])
            if action.get("action") == "start_model"
        }
        if available_routes != expected_route_set:
            raise ValueError(f"available actions are incomplete for {task_name}|{route_id}")

        identity = (task_id, matched_group, leakage_group, initial_digest)
        previous = task_identity.setdefault(task_name, identity)
        if previous != identity:
            raise ValueError(f"task routes do not share one leakage-safe group: {task_name}")
        for group, owners, label in (
            (matched_group, matched_group_owner, "matched group"),
            (leakage_group, leakage_group_owner, "leakage group"),
        ):
            owner = owners.setdefault(group, task_name)
            if owner != task_name:
                raise ValueError(f"{label} crosses task identities: {owner} and {task_name}")

        provenance_path = row.get("provenance", {}).get("source_result_path")
        if provenance_path:
            _guard_development_path(provenance_path, label="row provenance")
        document = task_document(task_input)
        previous_document = documents.setdefault(task_name, document)
        if previous_document != document:
            raise ValueError(f"public task input differs across routes: {task_name}")
        by_task[task_name][route_id] = row

    if expected_tasks is not None and len(by_task) != expected_tasks:
        raise ValueError(f"expected {expected_tasks} task groups, found {len(by_task)}")
    for task_name, route_rows in by_task.items():
        if set(route_rows) != expected_route_set:
            raise ValueError(f"training dataset is not rectangular for {task_name}")
    if len(rows) != len(by_task) * len(expected_routes):
        raise ValueError("training dataset has missing or extra task-route rows")
    return dict(by_task), sorted(expected_routes), documents


@dataclass(frozen=True)
class ConstantProbabilityEstimator:
    probability: float

    def predict_proba(self, matrix: Any) -> np.ndarray:
        count = int(matrix.shape[0])
        return np.tile([1.0 - self.probability, self.probability], (count, 1))


@dataclass
class TaskStartRouter:
    """Reloadable task-start routing policy trained only on public task information."""

    schema_version: str
    routes: tuple[str, ...]
    vectorizer: TfidfVectorizer
    route_estimators: dict[str, Any]
    median_route_costs_usd: dict[str, float]
    success_first_margins: dict[int, float]
    random_seed: int
    development_task_count: int
    training_dataset_sha256: str

    def score_routes(
        self,
        instruction: str,
        *,
        difficulty: str = "unknown",
        category: str = "unknown",
        tags: tuple[str, ...] | list[str] = (),
    ) -> dict[str, dict[str, float]]:
        document = task_document(
            {
                "instruction": instruction,
                "difficulty": difficulty,
                "category": category,
                "tags": list(tags),
            }
        )
        matrix = self.vectorizer.transform([document])
        return {
            route_id: {
                "success_probability": float(
                    self.route_estimators[route_id].predict_proba(matrix)[0, 1]
                ),
                "forecast_cost_usd": self.median_route_costs_usd[route_id],
            }
            for route_id in self.routes
        }

    def predict_route_order(
        self,
        instruction: str,
        *,
        max_routes: int = 1,
        difficulty: str = "unknown",
        category: str = "unknown",
        tags: tuple[str, ...] | list[str] = (),
    ) -> list[dict[str, Any]]:
        if max_routes not in self.success_first_margins:
            raise ValueError(f"max_routes must be in {sorted(self.success_first_margins)}")
        scores = self.score_routes(
            instruction,
            difficulty=difficulty,
            category=category,
            tags=tags,
        )
        order = _route_order(
            scores,
            list(self.routes),
            self.success_first_margins[max_routes],
        )[:max_routes]
        return [
            {
                "rank": rank,
                "route_id": route_id,
                **scores[route_id],
            }
            for rank, route_id in enumerate(order, start=1)
        ]


ConstantProbabilityEstimator.__module__ = "horizon_supervisor.training.task_start_router"
TaskStartRouter.__module__ = "horizon_supervisor.training.task_start_router"


def load_task_start_router(path: Path) -> TaskStartRouter:
    artifact = joblib.load(path)
    if not isinstance(artifact, TaskStartRouter):
        raise ValueError("joblib file is not a TaskStartRouter artifact")
    if artifact.schema_version != "task-start-router.v0":
        raise ValueError(f"unsupported router artifact: {artifact.schema_version}")
    return artifact


def _fit_components(
    by_task: dict[str, dict[str, dict[str, Any]]],
    routes: list[str],
    documents: dict[str, str],
    train_tasks: list[str],
    *,
    random_seed: int,
) -> tuple[TfidfVectorizer, dict[str, Any], dict[str, float]]:
    if not train_tasks:
        raise ValueError("cannot fit router without training task groups")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        max_features=1_024,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform([documents[name] for name in train_tasks])
    estimators: dict[str, Any] = {}
    costs = {}
    for route_id in routes:
        labels = np.asarray(
            [int(by_task[name][route_id]["target"]["completed"]) for name in train_tasks],
            dtype=np.int64,
        )
        if len(set(labels.tolist())) == 1:
            estimator: Any = ConstantProbabilityEstimator(float(labels[0]))
        else:
            estimator = LogisticRegression(
                C=0.25,
                solver="liblinear",
                max_iter=1_000,
                random_state=random_seed,
            )
            estimator.fit(matrix, labels)
        estimators[route_id] = estimator
        costs[route_id] = statistics.median(
            float(by_task[name][route_id]["target"]["cost_usd"])
            for name in train_tasks
        )
    return vectorizer, estimators, costs


def _predict_scores(
    by_task: dict[str, dict[str, dict[str, Any]]],
    routes: list[str],
    documents: dict[str, str],
    train_tasks: list[str],
    target_task: str,
    *,
    random_seed: int,
) -> dict[str, dict[str, float]]:
    vectorizer, estimators, costs = _fit_components(
        by_task,
        routes,
        documents,
        train_tasks,
        random_seed=random_seed,
    )
    matrix = vectorizer.transform([documents[target_task]])
    return {
        route_id: {
            "success_probability": float(estimators[route_id].predict_proba(matrix)[0, 1]),
            "forecast_cost_usd": costs[route_id],
        }
        for route_id in routes
    }


def _route_order(
    scores: dict[str, dict[str, float]],
    routes: list[str],
    margin: float,
) -> tuple[str, ...]:
    remaining = set(routes)
    ordered = []
    while remaining:
        best_probability = max(scores[route]["success_probability"] for route in remaining)
        eligible = [
            route
            for route in remaining
            if scores[route]["success_probability"] >= best_probability - margin
        ]
        selected = min(
            eligible,
            key=lambda route: (scores[route]["forecast_cost_usd"], route),
        )
        ordered.append(selected)
        remaining.remove(selected)
    return tuple(ordered)


def _cascade_summary(
    name: str,
    by_task: dict[str, dict[str, dict[str, Any]]],
    orders: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    decisions = []
    route_attempt_counts: Counter[str] = Counter()
    for task_name in sorted(by_task):
        attempted = []
        total_cost = 0.0
        completed = False
        for route_id in orders[task_name]:
            attempted.append(route_id)
            route_attempt_counts[route_id] += 1
            row = by_task[task_name][route_id]
            total_cost += float(row["target"]["cost_usd"])
            if row["target"]["completed"]:
                completed = True
                break
        decisions.append(
            {
                "task": task_name,
                "attempted_routes": attempted,
                "completed": completed,
                "attempt_count": len(attempted),
                "replayed_cost_usd": total_cost,
            }
        )
    return {
        "strategy": name,
        "tasks": len(decisions),
        "successes": sum(decision["completed"] for decision in decisions),
        "success_rate": statistics.mean(int(decision["completed"]) for decision in decisions),
        "total_cost_usd": sum(decision["replayed_cost_usd"] for decision in decisions),
        "total_attempts": sum(decision["attempt_count"] for decision in decisions),
        "mean_attempts_per_task": statistics.mean(
            decision["attempt_count"] for decision in decisions
        ),
        "route_attempt_counts": dict(sorted(route_attempt_counts.items())),
        "decisions": decisions,
    }


def _static_summaries(
    by_task: dict[str, dict[str, dict[str, Any]]], routes: list[str]
) -> list[dict[str, Any]]:
    return [
        _cascade_summary(
            f"always:{route_id}",
            by_task,
            {task_name: (route_id,) for task_name in by_task},
        )
        for route_id in routes
    ]


def _select_margin(
    by_task: dict[str, dict[str, dict[str, Any]]],
    routes: list[str],
    validation_scores: dict[str, dict[str, dict[str, float]]],
    *,
    max_routes: int,
) -> tuple[float, list[dict[str, Any]]]:
    candidates = []
    for margin in SUCCESS_FIRST_MARGINS:
        orders = {
            task_name: _route_order(validation_scores[task_name], routes, margin)[
                :max_routes
            ]
            for task_name in by_task
        }
        summary = _cascade_summary(f"margin={margin:g}", by_task, orders)
        candidates.append(
            {
                "margin": margin,
                "successes": summary["successes"],
                "total_cost_usd": summary["total_cost_usd"],
                "total_attempts": summary["total_attempts"],
            }
        )
    selected = max(
        candidates,
        key=lambda row: (
            row["successes"],
            -row["total_cost_usd"],
            -row["total_attempts"],
            -row["margin"],
        ),
    )
    return float(selected["margin"]), candidates


def _leave_one_task_out_scores(
    by_task: dict[str, dict[str, dict[str, Any]]],
    routes: list[str],
    documents: dict[str, str],
    tasks: list[str],
    *,
    random_seed: int,
) -> dict[str, dict[str, dict[str, float]]]:
    return {
        held_out: _predict_scores(
            by_task,
            routes,
            documents,
            [name for name in tasks if name != held_out],
            held_out,
            random_seed=random_seed,
        )
        for held_out in tasks
    }


def _nested_loocv_policies(
    by_task: dict[str, dict[str, dict[str, Any]]],
    routes: list[str],
    documents: dict[str, str],
    *,
    random_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tasks = sorted(by_task)
    if len(tasks) < 4:
        raise ValueError("nested leave-one-task-out requires at least four task groups")
    outer_orders: dict[int, dict[str, tuple[str, ...]]] = {
        max_routes: {} for max_routes in range(1, len(routes) + 1)
    }
    outer_margins: dict[int, dict[str, float]] = {
        max_routes: {} for max_routes in range(1, len(routes) + 1)
    }
    fold_audit = []

    for outer_task in tasks:
        outer_train = [name for name in tasks if name != outer_task]
        inner_scores = _leave_one_task_out_scores(
            by_task,
            routes,
            documents,
            outer_train,
            random_seed=random_seed,
        )
        outer_scores = _predict_scores(
            by_task,
            routes,
            documents,
            outer_train,
            outer_task,
            random_seed=random_seed,
        )
        selected_by_capacity = {}
        for max_routes in range(1, len(routes) + 1):
            margin, _candidates = _select_margin(
                {name: by_task[name] for name in outer_train},
                routes,
                inner_scores,
                max_routes=max_routes,
            )
            outer_margins[max_routes][outer_task] = margin
            outer_orders[max_routes][outer_task] = _route_order(
                outer_scores, routes, margin
            )[:max_routes]
            selected_by_capacity[str(max_routes)] = margin
        train_leakage_groups = sorted(
            by_task[name][routes[0]]["leakage_group"] for name in outer_train
        )
        held_out_leakage_group = by_task[outer_task][routes[0]]["leakage_group"]
        fold_audit.append(
            {
                "held_out_task": outer_task,
                "held_out_leakage_group": held_out_leakage_group,
                "training_task_count": len(outer_train),
                "training_contains_held_out_task": outer_task in outer_train,
                "training_contains_held_out_leakage_group": (
                    held_out_leakage_group in train_leakage_groups
                ),
                "training_leakage_groups_sha256": hashlib.sha256(
                    "\n".join(train_leakage_groups).encode()
                ).hexdigest(),
                "selected_margin_by_max_routes": selected_by_capacity,
            }
        )

    policies = []
    for max_routes in range(1, len(routes) + 1):
        summary = _cascade_summary(
            f"learned:nested-loocv:max-routes={max_routes}",
            by_task,
            outer_orders[max_routes],
        )
        summary.update(
            {
                "max_routes": max_routes,
                "evaluation": "nested leave-one-task-out",
                "hyperparameter_selection": "inner leave-one-task-out",
                "uses_outer_task_outcome_for_training_or_margin_selection": False,
                "selected_margin_counts": dict(
                    sorted(Counter(outer_margins[max_routes].values()).items())
                ),
                "selected_order_counts": dict(
                    sorted(
                        Counter(
                            ">".join(order)
                            for order in outer_orders[max_routes].values()
                        ).items()
                    )
                ),
            }
        )
        policies.append(summary)
    return policies, fold_audit


def _best_fixed_cascades(
    by_task: dict[str, dict[str, dict[str, Any]]], routes: list[str]
) -> list[dict[str, Any]]:
    frontier = []
    for max_routes in range(1, len(routes) + 1):
        candidates = [
            {
                **_cascade_summary(
                    "fixed-cascade:" + ">".join(order),
                    by_task,
                    {task_name: order for task_name in by_task},
                ),
                "max_routes": max_routes,
                "route_order": list(order),
            }
            for order in itertools.permutations(routes, max_routes)
        ]
        frontier.append(
            max(
                candidates,
                key=lambda row: (
                    row["successes"],
                    -row["total_cost_usd"],
                    -row["total_attempts"],
                    row["strategy"],
                ),
            )
        )
    return frontier


def train_task_start_router(
    dataset_path: Path = DEFAULT_DATASET,
    artifact_path: Path = DEFAULT_ARTIFACT,
    report_path: Path = DEFAULT_REPORT,
    *,
    expected_routes: tuple[str, ...] = DEFAULT_ROUTES,
    expected_tasks: int | None = 35,
    random_seed: int = RANDOM_SEED,
) -> tuple[TaskStartRouter, dict[str, Any]]:
    by_task, routes, documents = _read_development_rows(
        dataset_path,
        expected_routes=expected_routes,
        expected_tasks=expected_tasks,
    )
    tasks = sorted(by_task)
    nested_policies, fold_audit = _nested_loocv_policies(
        by_task,
        routes,
        documents,
        random_seed=random_seed,
    )

    full_loocv_scores = _leave_one_task_out_scores(
        by_task,
        routes,
        documents,
        tasks,
        random_seed=random_seed,
    )
    final_margins = {}
    final_margin_search = {}
    for max_routes in range(1, len(routes) + 1):
        margin, candidates = _select_margin(
            by_task,
            routes,
            full_loocv_scores,
            max_routes=max_routes,
        )
        final_margins[max_routes] = margin
        final_margin_search[str(max_routes)] = candidates

    vectorizer, estimators, costs = _fit_components(
        by_task,
        routes,
        documents,
        tasks,
        random_seed=random_seed,
    )
    dataset_sha256 = _sha256_file(dataset_path)
    artifact = TaskStartRouter(
        schema_version="task-start-router.v0",
        routes=tuple(routes),
        vectorizer=vectorizer,
        route_estimators=estimators,
        median_route_costs_usd=costs,
        success_first_margins=final_margins,
        random_seed=random_seed,
        development_task_count=len(tasks),
        training_dataset_sha256=dataset_sha256,
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path)

    static_models = _static_summaries(by_task, routes)
    optimized_fixed = _best_fixed_cascades(by_task, routes)
    frozen_orders = (
        FROZEN_CASCADES
        if set(routes) == set(DEFAULT_ROUTES)
        else {
            row["max_routes"]: tuple(row["route_order"])
            for row in optimized_fixed
            if row["max_routes"] > 1
        }
    )
    frozen_cascades = [
        {
            **_cascade_summary(
                f"frozen-development-cascade:max-routes={max_routes}",
                by_task,
                {task_name: order for task_name in by_task},
            ),
            "max_routes": max_routes,
            "route_order": list(order),
            "policy_selection": "frozen before Wave 3 evaluation",
        }
        for max_routes, order in sorted(frozen_orders.items())
    ]
    comparison = []
    for family, rows in (
        ("static-model", static_models),
        ("frozen-fixed-cascade", frozen_cascades),
        ("learned-task-start-policy", nested_policies),
    ):
        comparison.extend(
            {
                "family": family,
                "strategy": row["strategy"],
                "max_routes": row.get("max_routes", 1),
                "successes": row["successes"],
                "success_rate": row["success_rate"],
                "total_cost_usd": row["total_cost_usd"],
                "total_attempts": row["total_attempts"],
            }
            for row in rows
        )

    matching_baselines = {
        1: max(
            static_models,
            key=lambda row: (row["successes"], -row["total_cost_usd"]),
        )
    }
    matching_baselines.update({row["max_routes"]: row for row in frozen_cascades})
    learned_beats_matching_baseline = {
        str(row["max_routes"]): (
            row["successes"] > matching_baselines[row["max_routes"]]["successes"]
            or (
                row["successes"] == matching_baselines[row["max_routes"]]["successes"]
                and row["total_cost_usd"]
                < matching_baselines[row["max_routes"]]["total_cost_usd"]
            )
        )
        for row in nested_policies
    }

    report = {
        "schema_version": "task-start-router-training-report.v0",
        "training_data": {
            "path": str(dataset_path),
            "sha256": dataset_sha256,
            "record_split": "development",
            "records": len(tasks) * len(routes),
            "task_groups": len(tasks),
            "routes": routes,
            "all_rows_learning_valid": True,
            "all_rows_clean_start": True,
            "held_out_or_wave3_rows": 0,
            "held_out_or_wave3_paths_used": 0,
        },
        "recipe": {
            "feature_input": "public instruction, difficulty, category, and tags only",
            "vectorizer": {
                "kind": "TF-IDF",
                "lowercase": True,
                "ngram_range": [1, 2],
                "min_df": 1,
                "max_features": 1_024,
                "sublinear_tf": True,
            },
            "estimator": {
                "kind": "one logistic regression per route",
                "C": 0.25,
                "solver": "liblinear",
                "max_iter": 1_000,
                "random_seed": random_seed,
                "constant_label_folds": "constant probability estimator",
            },
            "cost_forecast": "training-fold median observed route cost",
            "margin_candidates": list(SUCCESS_FIRST_MARGINS),
            "selection_objective": (
                "lexicographic: successes, then lower cost, then fewer attempts, "
                "then smaller margin"
            ),
        },
        "nested_loocv": {
            "split_unit": "task/leakage group",
            "outer_folds": len(tasks),
            "inner_folds_per_outer_fold": len(tasks) - 1,
            "group_isolation_verified": all(
                not fold["training_contains_held_out_task"]
                and not fold["training_contains_held_out_leakage_group"]
                for fold in fold_audit
            ),
            "fold_audit": fold_audit,
        },
        "static_model_baselines": static_models,
        "fixed_cascade_baselines": frozen_cascades,
        "development_optimized_fixed_cascade_frontier": optimized_fixed,
        "learned_route_policies_nested_loocv": nested_policies,
        "comparison": comparison,
        "result": {
            "learned_beats_matching_baseline_by_max_routes": (
                learned_beats_matching_baseline
            ),
            "learned_task_start_router_improvement_supported": any(
                learned_beats_matching_baseline.values()
            ),
            "decision": (
                "Keep the artifact as a reproducible baseline, but do not claim that "
                "the learned task-start router beats the frozen rules on 35 development "
                "tasks. More task groups or richer live-run observations are required."
            ),
        },
        "final_artifact_fit": {
            "path": str(artifact_path),
            "sha256": _sha256_file(artifact_path),
            "fit_task_groups": len(tasks),
            "fit_records": len(tasks) * len(routes),
            "median_route_costs_usd": costs,
            "selected_margin_by_max_routes": {
                str(key): value for key, value in final_margins.items()
            },
            "margin_search_leave_one_task_out": final_margin_search,
            "reloadable_with": "load_task_start_router",
        },
        "evaluation_contract": {
            "development_only": True,
            "wave3_evaluated": False,
            "held_out_labels_used": False,
            "outer_task_outcome_used_for_training": False,
            "outer_task_outcome_used_for_margin_selection": False,
            "current_task_cost_used_for_ordering": False,
            "cascade_stop_signal": "observed verifier-confirmed success",
            "claim_boundary": (
                "This is a task-start and clean-restart routing policy. It does not yet "
                "learn mid-run continue, switch, recovery, or stop decisions."
            ),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return artifact, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and nested-LOOCV evaluate the development-only task-start router"
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--expected-tasks", type=int, default=35)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    _artifact, report = train_task_start_router(
        dataset_path=args.dataset,
        artifact_path=args.artifact,
        report_path=args.report,
        expected_tasks=args.expected_tasks,
        random_seed=args.random_seed,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
