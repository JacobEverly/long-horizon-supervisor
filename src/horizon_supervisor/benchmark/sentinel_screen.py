from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from horizon_supervisor.benchmark.matched_outcomes import LEARNING_VALID_STATUSES

SENTINEL_OBSERVABLE_STATUSES = LEARNING_VALID_STATUSES | {"provider_error"}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _cost(row: dict[str, Any]) -> float:
    outcome = row["outcome"]
    value = outcome.get("estimated_list_cost_usd")
    if value is None:
        value = outcome.get("allocated_provider_cost_usd")
    if value is None:
        raise ValueError("sentinel outcome has no comparable cost")
    return float(value)


def _wilson(successes: int, total: int, z: float = 1.96) -> list[float]:
    if total == 0:
        return [0.0, 1.0]
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z**2 / (4 * total**2)
        )
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def analyze_sentinel_screen(
    outcomes_path: Path,
    manifest_path: Path,
    tranche_id: str,
    report_path: Path,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tranche = next(
        (row for row in manifest["tranches"] if row["id"] == tranche_id), None
    )
    if tranche is None:
        raise ValueError(f"unknown sentinel tranche {tranche_id!r}")
    task_names = list(tranche["task_names"])
    task_set = set(task_names)
    routes = tuple(manifest["sentinel_routes"])
    by_task: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in _load_jsonl(outcomes_path):
        task_name = row["task"]["source_task_name"]
        route_id = row["model"]["route_id"]
        if task_name not in task_set or route_id not in routes:
            raise ValueError(f"unexpected sentinel pair {task_name}|{route_id}")
        if row["outcome"]["status"] not in SENTINEL_OBSERVABLE_STATUSES:
            raise ValueError(
                f"sentinel pair is not deployment-observable: {task_name}|{route_id}"
            )
        if route_id in by_task[task_name]:
            raise ValueError(f"duplicate sentinel pair {task_name}|{route_id}")
        by_task[task_name][route_id] = row
    expected_pairs = {(task_name, route) for task_name in task_names for route in routes}
    observed_pairs = {
        (task_name, route)
        for task_name, task_rows in by_task.items()
        for route in task_rows
    }
    if observed_pairs != expected_pairs:
        missing = sorted(expected_pairs - observed_pairs)
        raise ValueError(f"sentinel tranche is not complete; missing={missing}")

    disagreements: list[str] = []
    agreements: list[str] = []
    outcomes_by_task: dict[str, dict[str, bool]] = {}
    for task_name in task_names:
        route_outcomes = {
            route: bool(by_task[task_name][route]["outcome"]["completed"])
            for route in routes
        }
        outcomes_by_task[task_name] = route_outcomes
        target = disagreements if len(set(route_outcomes.values())) > 1 else agreements
        target.append(task_name)

    seed = manifest["expansion_rule"]["agreement_audit_seed"]
    audit_count = int(manifest["expansion_rule"]["agreement_audits_per_tranche"])
    ranked_agreements = sorted(
        agreements,
        key=lambda task_name: hashlib.sha256(
            (
                f"{seed}|{by_task[task_name][routes[0]]['task']['task_id']}|"
                f"{task_name}"
            ).encode()
        ).hexdigest(),
    )
    agreement_audits = ranked_agreements[:audit_count]
    expansion_set = set(disagreements) | set(agreement_audits)
    expansion_tasks = [task for task in task_names if task in expansion_set]

    static = []
    for route in routes:
        rows = [by_task[task][route] for task in task_names]
        static.append(
            {
                "route_id": route,
                "successes": sum(row["outcome"]["completed"] for row in rows),
                "total_cost_usd": sum(_cost(row) for row in rows),
            }
        )
    best_static = max(
        static, key=lambda row: (row["successes"], -row["total_cost_usd"])
    )
    oracle_successes = sum(
        any(by_task[task][route]["outcome"]["completed"] for route in routes)
        for task in task_names
    )
    screen_trials = len(task_names) * len(routes)
    expansion_trials = len(expansion_tasks) * 2
    full_matrix_trials = len(task_names) * 4
    all_rows = [by_task[task][route] for task in task_names for route in routes]
    status_counts = Counter(row["outcome"]["status"] for row in all_rows)
    provider_failure_pairs = [
        f"{row['task']['source_task_name']}|{row['model']['route_id']}"
        for row in all_rows
        if row["outcome"]["status"] == "provider_error"
    ]
    report = {
        "schema_version": "sentinel-screen-analysis.v0",
        "manifest_path": str(manifest_path),
        "outcomes_path": str(outcomes_path),
        "tranche_id": tranche_id,
        "task_count": len(task_names),
        "sentinel_routes": list(routes),
        "outcomes_by_task": outcomes_by_task,
        "disagreement_task_names": disagreements,
        "agreement_task_names": agreements,
        "disagreement_count": len(disagreements),
        "disagreement_rate": len(disagreements) / len(task_names),
        "disagreement_rate_wilson_95": _wilson(
            len(disagreements), len(task_names)
        ),
        "agreement_audit_task_names": agreement_audits,
        "expansion_task_names": expansion_tasks,
        "expansion_route_ids": ["gate7/fixed-flash", "gate7/fixed-kimi"],
        "trial_accounting": {
            "sentinel_trials": screen_trials,
            "expansion_trials": expansion_trials,
            "total_trials_after_expansion": screen_trials + expansion_trials,
            "full_matrix_trials": full_matrix_trials,
            "saved_trials": full_matrix_trials - screen_trials - expansion_trials,
        },
        "sentinel_static_routes": static,
        "best_completion_first_static": best_static,
        "sentinel_hindsight_successes": oracle_successes,
        "sentinel_oracle_headroom": oracle_successes - best_static["successes"],
        "deployment_observation": {
            "status_counts": dict(sorted(status_counts.items())),
            "provider_failure_pairs": provider_failure_pairs,
            "screen_observed_pair_count": len(all_rows),
            "capability_learning_valid_pair_count": sum(
                row["outcome"]["status"] in LEARNING_VALID_STATUSES
                for row in all_rows
            ),
            "interpretation": (
                "Provider failures count as unsuccessful observed deployments in the "
                "completion-first screen, but remain excluded from capability training. "
                "Environment and harness failures are never accepted as observations."
            ),
        },
        "selection_bias_guard": (
            "All sentinel disagreements are expanded. Agreement audits are selected "
            "only by the frozen hash rule; screen and expansion results remain separate."
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a frozen sentinel screen")
    parser.add_argument("--outcomes", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--tranche", required=True)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    report = analyze_sentinel_screen(
        args.outcomes, args.manifest, args.tranche, args.report
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
