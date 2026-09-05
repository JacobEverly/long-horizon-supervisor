from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = ROOT / "artifacts/official/two-tier-continuation-calibration-v0"
OUTCOMES_PATH = OUTPUT_ROOT / "natural-continuation-outcomes-v0.jsonl"
REPORT_PATH = OUTPUT_ROOT / "calibration-report-v0.json"
ROUTES = ("gate7/fixed-flash", "gate7/fixed-qwen")
KINDS = ("healthy", "needs_review", "confirmed_stuck")
DECISION_PASS = "READY FOR MATCHED INTERVENTION PILOT — continuation calibration passed"
DECISION_FEASIBLE = "CHECKPOINT FEASIBILITY PROVEN — detector effect unresolved"
DECISION_FAIL = "NOT READY — continuation calibration gate failed"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _usable_checkpoints(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    checkpoints = []
    for row in rows:
        if (
            row.get("schema_version") != "natural-continuation-outcome.v0"
            or row.get("valid") is not True
            or row.get("structural_failure") is True
            or row.get("verifier_outcome_present") is not True
        ):
            continue
        for checkpoint in row.get("checkpoints", []):
            if (
                checkpoint.get("kind") in KINDS
                and checkpoint.get("state_transfer_eligible") is True
                and checkpoint.get("snapshot_fidelity_passed") is True
            ):
                checkpoints.append(
                    {
                        "task_id": row["task_id"],
                        "route_id": row["route_id"],
                        "model_id": row["model_id"],
                        "completed": bool(row["verified_completion"]),
                        "kind": checkpoint["kind"],
                        "turn": int(checkpoint["turn"]),
                        "remaining_turns": int(checkpoint["remaining_turns"]),
                    }
                )
    return checkpoints


def _rate(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return sum(bool(row["completed"]) for row in rows) / len(rows)


def _difference(rows: list[dict[str, Any]]) -> float | None:
    healthy = [row for row in rows if row["kind"] == "healthy"]
    confirmed = [row for row in rows if row["kind"] == "confirmed_stuck"]
    healthy_rate = _rate(healthy)
    confirmed_rate = _rate(confirmed)
    if healthy_rate is None or confirmed_rate is None:
        return None
    return healthy_rate - confirmed_rate


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _clustered_interval(
    checkpoints: list[dict[str, Any]], *, seed: int, samples: int
) -> list[float] | None:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for checkpoint in checkpoints:
        by_task[str(checkpoint["task_id"])].append(checkpoint)
    tasks = sorted(by_task)
    if len(tasks) < 2:
        return None
    randomizer = random.Random(seed)
    differences = []
    for _ in range(samples):
        sample = []
        for task in randomizer.choices(tasks, k=len(tasks)):
            sample.extend(by_task[task])
        difference = _difference(sample)
        if difference is not None:
            differences.append(difference)
    if not differences:
        return None
    return [_percentile(differences, 0.025), _percentile(differences, 0.975)]


def _tier_summary(checkpoints: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    rows = [row for row in checkpoints if row["kind"] == kind]
    rate = _rate(rows)
    by_route = Counter(str(row["route_id"]) for row in rows)
    return {
        "checkpoint_count": len(rows),
        "task_count": len({str(row["task_id"]) for row in rows}),
        "by_route": {route: by_route[route] for route in ROUTES},
        "completed": sum(bool(row["completed"]) for row in rows),
        "natural_recovery_rate": rate,
        "minimum_remaining_turns": min(
            (int(row["remaining_turns"]) for row in rows), default=None
        ),
    }


def analyze(
    rows: list[dict[str, Any]], *, seed: int = 20260905, samples: int = 10_000
) -> dict[str, Any]:
    checkpoints = _usable_checkpoints(rows)
    tiers = {kind: _tier_summary(checkpoints, kind) for kind in KINDS}
    difference = _difference(checkpoints)
    interval = _clustered_interval(checkpoints, seed=seed, samples=samples)

    differences_by_route = {}
    for route in ROUTES:
        route_rows = [row for row in checkpoints if row["route_id"] == route]
        differences_by_route[route] = _difference(route_rows)

    tasks = sorted({str(row["task_id"]) for row in checkpoints})
    leave_one_out = []
    for task in tasks:
        value = _difference(
            [row for row in checkpoints if str(row["task_id"]) != task]
        )
        if value is not None:
            leave_one_out.append(value)
    confirmed_by_task = Counter(
        str(row["task_id"])
        for row in checkpoints
        if row["kind"] == "confirmed_stuck"
    )
    confirmed_total = tiers["confirmed_stuck"]["checkpoint_count"]
    max_confirmed_share = (
        max(confirmed_by_task.values(), default=0) / confirmed_total
        if confirmed_total
        else None
    )

    review_runs = {
        (str(row["task_id"]), str(row["route_id"]))
        for row in checkpoints
        if row["kind"] == "needs_review"
    }
    confirmed_runs = {
        (str(row["task_id"]), str(row["route_id"]))
        for row in checkpoints
        if row["kind"] == "confirmed_stuck"
    }
    transition_rate = (
        len(review_runs & confirmed_runs) / len(review_runs) if review_runs else None
    )

    all_candidate_checkpoints = [
        checkpoint
        for row in rows
        if row.get("structural_failure") is not True
        for checkpoint in row.get("checkpoints", [])
        if checkpoint.get("state_transfer_eligible") is True
    ]
    all_fidelity = bool(all_candidate_checkpoints) and all(
        checkpoint.get("snapshot_fidelity_passed") is True
        for checkpoint in all_candidate_checkpoints
    )
    both_models = {
        kind: all(tiers[kind]["by_route"][route] > 0 for route in ROUTES)
        for kind in KINDS
    }
    coverage = {
        "needs_review": (
            tiers["needs_review"]["checkpoint_count"] >= 12
            and tiers["needs_review"]["task_count"] >= 8
            and both_models["needs_review"]
        ),
        "confirmed_stuck": (
            tiers["confirmed_stuck"]["checkpoint_count"] >= 6
            and tiers["confirmed_stuck"]["task_count"] >= 4
            and both_models["confirmed_stuck"]
        ),
        "healthy": (
            tiers["healthy"]["checkpoint_count"] >= 12
            and tiers["healthy"]["task_count"] >= 8
            and both_models["healthy"]
        ),
    }
    gates = {
        "healthy_minus_confirmed_at_least_20pp": (
            difference is not None and difference >= 0.20
        ),
        "task_clustered_interval_excludes_zero": (
            interval is not None and interval[0] > 0
        ),
        "direction_positive_both_models": all(
            value is not None and value > 0 for value in differences_by_route.values()
        ),
        "not_driven_by_one_task": (
            bool(leave_one_out)
            and min(leave_one_out) > 0
            and max_confirmed_share is not None
            and max_confirmed_share <= 0.25
        ),
        "needs_review_coverage": coverage["needs_review"],
        "confirmed_stuck_coverage": coverage["confirmed_stuck"],
        "healthy_coverage": coverage["healthy"],
        "confirmed_has_two_remaining_turns": (
            tiers["confirmed_stuck"]["minimum_remaining_turns"] is not None
            and tiers["confirmed_stuck"]["minimum_remaining_turns"] >= 2
        ),
        "all_counted_snapshots_rehydrated": all_fidelity,
        "structural_failures_separate": all(
            not row.get("checkpoints")
            for row in rows
            if row.get("structural_failure") is True
        ),
        "leakage_controls": all(
            row.get("leakage_check_passed") is True for row in rows
        ),
    }
    passed = all(gates.values())
    feasibility_gates = [
        "needs_review_coverage",
        "confirmed_stuck_coverage",
        "healthy_coverage",
        "confirmed_has_two_remaining_turns",
        "all_counted_snapshots_rehydrated",
        "structural_failures_separate",
        "leakage_controls",
    ]
    feasible = all(gates[name] for name in feasibility_gates)
    decision = DECISION_PASS if passed else DECISION_FEASIBLE if feasible else DECISION_FAIL
    return {
        "schema_version": "two-tier-continuation-calibration-report.v0",
        "decision": decision,
        "gate_passed": passed,
        "checkpoint_feasibility_passed": feasible,
        "trajectory_count": len(rows),
        "valid_natural_trajectory_count": sum(
            row.get("valid") is True and row.get("structural_failure") is not True
            for row in rows
        ),
        "structural_trajectory_count": sum(
            row.get("structural_failure") is True for row in rows
        ),
        "tiers": tiers,
        "review_to_confirmation_transition_rate": transition_rate,
        "healthy_minus_confirmed_recovery": difference,
        "task_clustered_interval_95": interval,
        "differences_by_route": differences_by_route,
        "leave_one_task_out_min_difference": min(leave_one_out, default=None),
        "maximum_single_task_confirmed_share": max_confirmed_share,
        "gates": gates,
        "bootstrap": {"unit": "task", "seed": seed, "samples": samples},
    }


def run(
    outcomes_path: Path = OUTCOMES_PATH, report_path: Path = REPORT_PATH
) -> dict[str, Any]:
    rows = _rows(outcomes_path)
    report = analyze(rows)
    report["outcomes_path"] = str(outcomes_path.relative_to(ROOT))
    report["outcomes_sha256"] = _sha256(outcomes_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcomes", type=Path, default=OUTCOMES_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    print(json.dumps(run(args.outcomes, args.report), indent=2))


if __name__ == "__main__":
    main()
