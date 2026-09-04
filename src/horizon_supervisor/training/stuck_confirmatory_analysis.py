from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from statistics import mean
from typing import Any

BRANCHES = {
    "continue_current_state",
    "switch_value_state",
    "switch_kimi_state",
    "restart_kimi_clean",
}
BASE_MODELS = (
    "deepseek/deepseek-v4-flash-0731",
    "qwen/qwen3.8-27b",
)
KINDS = ("suspected_stuck", "healthy")
BOOTSTRAP_SEED = 20260904
BOOTSTRAP_SAMPLES = 10_000


def _load(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("confirmatory outcome table is empty")
    if any(row.get("valid") is not True for row in rows):
        raise ValueError("confirmatory analysis accepts valid outcomes only")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("checkpoint_kind") not in KINDS:
            raise ValueError("unexpected checkpoint kind")
        if row.get("base_model_id") not in BASE_MODELS:
            raise ValueError("unexpected base model")
        grouped[str(row["group_id"])].append(row)

    for group_id, group in grouped.items():
        actions = {str(row.get("branch_action")) for row in group}
        if len(group) != len(BRANCHES) or actions != BRANCHES:
            raise ValueError(f"matched group {group_id} is incomplete or duplicated")
        invariants = {
            (
                row["task_id"],
                row["checkpoint_kind"],
                row["checkpoint_turn"],
                row["base_model_id"],
                row["remaining_turns"],
                row["remaining_output_tokens"],
                row["maximum_wall_seconds"],
                row["maximum_incremental_spend_usd"],
            )
            for row in group
        }
        if len(invariants) != 1:
            raise ValueError(f"matched group {group_id} does not share one state and budget")
    return rows


def _by_group(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[str(row["group_id"])][str(row["branch_action"])] = row
    return dict(grouped)


def _cluster_interval(
    items: list[dict[str, Any]],
    statistic: Callable[[list[dict[str, Any]]], float | None],
    *,
    seed: int = BOOTSTRAP_SEED,
    samples: int = BOOTSTRAP_SAMPLES,
) -> list[float] | None:
    tasks = sorted({str(item["task_id"]) for item in items})
    if len(tasks) < 2:
        return None
    by_task = {task: [item for item in items if item["task_id"] == task] for task in tasks}
    rng = random.Random(seed)
    values: list[float] = []
    attempts = 0
    while len(values) < samples and attempts < samples * 20:
        attempts += 1
        sampled = [rng.choice(tasks) for _ in tasks]
        sample_items = [item for task in sampled for item in by_task[task]]
        value = statistic(sample_items)
        if value is not None:
            values.append(value)
    if len(values) < samples:
        return None
    values.sort()
    return [values[int(samples * 0.025)], values[int(samples * 0.975) - 1]]


def _completion_rate(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return mean(bool(row["verified_completion"]) for row in rows)


def _arm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "outcomes": 0,
            "verified_completions": 0,
            "completion_rate": None,
            "completion_rate_task_bootstrap_95": None,
            "mean_verifier_reward": None,
            "verifier_confirmed_progress": 0,
            "total_cost_usd": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_tokens": 0,
            "reasoning_tokens": 0,
            "elapsed_seconds": 0.0,
            "state_transfer_failures": 0,
            "protocol_errors": 0,
            "provider_errors": 0,
        }
    return {
        "outcomes": len(rows),
        "verified_completions": sum(bool(row["verified_completion"]) for row in rows),
        "completion_rate": _completion_rate(rows),
        "completion_rate_task_bootstrap_95": _cluster_interval(
            rows, _completion_rate
        ),
        "mean_verifier_reward": mean(float(row["verifier_reward"]) for row in rows),
        "verifier_confirmed_progress": sum(
            float(row["verifier_reward"]) > 0 for row in rows
        ),
        "total_cost_usd": sum(float(row["cost_usd"]) for row in rows),
        "input_tokens": sum(int(row["input_tokens"]) for row in rows),
        "output_tokens": sum(int(row["output_tokens"]) for row in rows),
        "cached_tokens": sum(int(row["cached_tokens"]) for row in rows),
        "reasoning_tokens": sum(int(row["reasoning_tokens"]) for row in rows),
        "elapsed_seconds": sum(float(row["elapsed_seconds"]) for row in rows),
        "state_transfer_failures": sum(
            bool(row["state_transfer_failure"]) for row in rows
        ),
        "protocol_errors": sum(bool(row["protocol_error"]) for row in rows),
        "provider_errors": sum(bool(row["provider_error"]) for row in rows),
    }


def _paired_contrast(
    groups: list[dict[str, dict[str, Any]]], left: str, right: str
) -> dict[str, Any]:
    pairs = [
        {
            "task_id": group[left]["task_id"],
            "left": group[left],
            "right": group[right],
        }
        for group in groups
    ]

    def delta(sample: list[dict[str, Any]]) -> float | None:
        if not sample:
            return None
        return mean(
            bool(item["left"]["verified_completion"])
            - bool(item["right"]["verified_completion"])
            for item in sample
        )

    return {
        "left": left,
        "right": right,
        "matched_groups": len(pairs),
        "left_summary": _arm_summary([item["left"] for item in pairs]),
        "right_summary": _arm_summary([item["right"] for item in pairs]),
        "completion_rate_difference": delta(pairs),
        "completion_rate_difference_task_bootstrap_95": _cluster_interval(
            pairs, delta
        ),
        "rescue_count": sum(
            bool(item["left"]["verified_completion"])
            and not bool(item["right"]["verified_completion"])
            for item in pairs
        ),
        "harm_count": sum(
            not bool(item["left"]["verified_completion"])
            and bool(item["right"]["verified_completion"])
            for item in pairs
        ),
    }


def _interaction(
    stuck: list[dict[str, dict[str, Any]]],
    healthy: list[dict[str, dict[str, Any]]],
    action: str,
) -> dict[str, Any]:
    items = [
        {
            "task_id": group[action]["task_id"],
            "kind": kind,
            "effect": (
                bool(group[action]["verified_completion"])
                - bool(group["continue_current_state"]["verified_completion"])
            ),
        }
        for kind, groups in (("suspected_stuck", stuck), ("healthy", healthy))
        for group in groups
    ]

    def difference_in_differences(sample: list[dict[str, Any]]) -> float | None:
        stuck_effects = [item["effect"] for item in sample if item["kind"] == "suspected_stuck"]
        healthy_effects = [item["effect"] for item in sample if item["kind"] == "healthy"]
        if not stuck_effects or not healthy_effects:
            return None
        return mean(stuck_effects) - mean(healthy_effects)

    stuck_effect = _paired_contrast(stuck, action, "continue_current_state")
    healthy_effect = _paired_contrast(healthy, action, "continue_current_state")
    return {
        "action": action,
        "stuck_effect": stuck_effect,
        "healthy_effect": healthy_effect,
        "difference_in_differences": difference_in_differences(items),
        "difference_in_differences_task_bootstrap_95": _cluster_interval(
            items, difference_in_differences
        ),
    }


def _dataset_checks(
    grouped: dict[str, dict[str, dict[str, Any]]]
) -> dict[str, Any]:
    group_rows = [next(iter(group.values())) for group in grouped.values()]
    kind_counts = Counter(str(row["checkpoint_kind"]) for row in group_rows)
    kind_tasks = {
        kind: sorted(
            {str(row["task_id"]) for row in group_rows if row["checkpoint_kind"] == kind}
        )
        for kind in KINDS
    }
    kind_base_counts = {
        kind: {
            base: sum(
                row["checkpoint_kind"] == kind and row["base_model_id"] == base
                for row in group_rows
            )
            for base in BASE_MODELS
        }
        for kind in KINDS
    }
    task_kind_counts = Counter(
        (str(row["task_id"]), str(row["checkpoint_kind"])) for row in group_rows
    )
    unique_tasks = sorted({str(row["task_id"]) for row in group_rows})
    checks = {
        "exactly_12_groups_per_kind": all(kind_counts[kind] == 12 for kind in KINDS),
        "exactly_96_valid_outcomes": len(grouped) * len(BRANCHES) == 96,
        "at_least_8_unique_tasks": len(unique_tasks) >= 8,
        "at_least_4_unique_tasks_per_kind": all(
            len(kind_tasks[kind]) >= 4 for kind in KINDS
        ),
        "at_least_4_groups_per_base_and_kind": all(
            kind_base_counts[kind][base] >= 4 for kind in KINDS for base in BASE_MODELS
        ),
        "no_more_than_2_groups_per_task_and_kind": max(
            task_kind_counts.values(), default=0
        )
        <= 2,
    }
    return {
        "group_counts": dict(kind_counts),
        "unique_tasks": unique_tasks,
        "unique_task_count": len(unique_tasks),
        "unique_tasks_by_kind": kind_tasks,
        "base_counts_by_kind": kind_base_counts,
        "maximum_groups_per_task_and_kind": max(task_kind_counts.values(), default=0),
        "checks": checks,
        "target_complete": all(checks.values()),
    }


def _detector_gap(
    stuck: list[dict[str, dict[str, Any]]],
    healthy: list[dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    items = [
        {
            "task_id": group["continue_current_state"]["task_id"],
            "kind": kind,
            "base_model_id": group["continue_current_state"]["base_model_id"],
            "success": bool(group["continue_current_state"]["verified_completion"]),
        }
        for kind, groups in (("suspected_stuck", stuck), ("healthy", healthy))
        for group in groups
    ]

    def gap(sample: list[dict[str, Any]]) -> float | None:
        stuck_values = [item["success"] for item in sample if item["kind"] == "suspected_stuck"]
        healthy_values = [item["success"] for item in sample if item["kind"] == "healthy"]
        if not stuck_values or not healthy_values:
            return None
        return mean(healthy_values) - mean(stuck_values)

    tasks = sorted({str(item["task_id"]) for item in items})
    leave_one_task_out = {
        task: gap([item for item in items if item["task_id"] != task]) for task in tasks
    }
    by_base = {
        base: gap([item for item in items if item["base_model_id"] == base])
        for base in BASE_MODELS
    }
    return {
        "continue_at_stuck": _arm_summary(
            [group["continue_current_state"] for group in stuck]
        ),
        "continue_at_healthy": _arm_summary(
            [group["continue_current_state"] for group in healthy]
        ),
        "healthy_minus_stuck_recovery_rate": gap(items),
        "healthy_minus_stuck_task_bootstrap_95": _cluster_interval(items, gap),
        "leave_one_task_out_gaps": leave_one_task_out,
        "base_specific_gaps": by_base,
        "false_positive_stuck_triggers": sum(
            bool(group["continue_current_state"]["verified_completion"])
            for group in stuck
        ),
        "false_positive_stuck_trigger_rate": _completion_rate(
            [group["continue_current_state"] for group in stuck]
        ),
    }


def _pareto_at_stuck(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    summaries = {
        action: _arm_summary(
            [
                row
                for row in rows
                if row["checkpoint_kind"] == "suspected_stuck"
                and row["branch_action"] == action
            ]
        )
        for action in sorted(BRANCHES)
    }
    frontier = []
    for action, summary in summaries.items():
        success = int(summary["verified_completions"])
        cost = float(summary["total_cost_usd"])
        dominated = any(
            int(other["verified_completions"]) >= success
            and float(other["total_cost_usd"]) <= cost
            and (
                int(other["verified_completions"]) > success
                or float(other["total_cost_usd"]) < cost
            )
            for other_action, other in summaries.items()
            if other_action != action
        )
        if not dominated:
            frontier.append(action)
    return {"arms": summaries, "non_dominated_actions": sorted(frontier)}


def analyze(rows: list[dict[str, Any]], *, pool_exhausted: bool = False) -> dict[str, Any]:
    grouped = _by_group(rows)
    stuck = [
        group
        for group in grouped.values()
        if next(iter(group.values()))["checkpoint_kind"] == "suspected_stuck"
    ]
    healthy = [
        group
        for group in grouped.values()
        if next(iter(group.values()))["checkpoint_kind"] == "healthy"
    ]
    dataset = _dataset_checks(grouped)
    detector = _detector_gap(stuck, healthy)
    kimi_interaction = _interaction(stuck, healthy, "switch_kimi_state")
    value_interaction = _interaction(stuck, healthy, "switch_value_state")
    comparisons = {
        "kimi_state_vs_continue_at_stuck": _paired_contrast(
            stuck, "switch_kimi_state", "continue_current_state"
        ),
        "kimi_state_vs_continue_at_healthy": _paired_contrast(
            healthy, "switch_kimi_state", "continue_current_state"
        ),
        "kimi_state_vs_kimi_clean_at_stuck": _paired_contrast(
            stuck, "switch_kimi_state", "restart_kimi_clean"
        ),
        "kimi_state_vs_value_state_at_stuck": _paired_contrast(
            stuck, "switch_kimi_state", "switch_value_state"
        ),
        "value_state_vs_continue_at_stuck": _paired_contrast(
            stuck, "switch_value_state", "continue_current_state"
        ),
    }
    directionality = {
        base: _paired_contrast(
            [
                group
                for group in stuck
                if group["continue_current_state"]["base_model_id"] == base
            ],
            "switch_value_state",
            "continue_current_state",
        )
        for base in BASE_MODELS
    }
    pareto = _pareto_at_stuck(rows)

    gap = detector["healthy_minus_stuck_recovery_rate"]
    gap_interval = detector["healthy_minus_stuck_task_bootstrap_95"]
    task_robust = bool(detector["leave_one_task_out_gaps"]) and all(
        value is not None and value > 0
        for value in detector["leave_one_task_out_gaps"].values()
    )
    base_robust = all(
        value is not None and value > 0 for value in detector["base_specific_gaps"].values()
    )
    detector_gate = bool(
        dataset["target_complete"]
        and gap is not None
        and gap >= 0.20
        and gap_interval is not None
        and gap_interval[0] > 0
        and task_robust
        and base_robust
    )

    kimi_stuck = comparisons["kimi_state_vs_continue_at_stuck"]
    kimi_healthy = comparisons["kimi_state_vs_continue_at_healthy"]
    kimi_stuck_effect = kimi_stuck["completion_rate_difference"]
    kimi_healthy_effect = kimi_healthy["completion_rate_difference"]
    did = kimi_interaction["difference_in_differences"]
    kimi_gate = bool(
        dataset["target_complete"]
        and kimi_stuck_effect is not None
        and kimi_stuck_effect >= 0.15
        and kimi_stuck["rescue_count"] >= 2
        and kimi_healthy_effect is not None
        and kimi_stuck_effect > kimi_healthy_effect
        and did is not None
        and did > 0
        and "switch_kimi_state" in pareto["non_dominated_actions"]
    )
    jaggedness_routes = [
        base
        for base, comparison in directionality.items()
        if comparison["rescue_count"] >= 2
    ]
    if not dataset["target_complete"]:
        decision = "INCONCLUSIVE — improve coverage and repeat"
    elif detector_gate and kimi_gate:
        decision = "CONFIRMED — proceed to training-sized collection"
    else:
        decision = "REJECTED — do not build the learned intervention policy"

    return {
        "schema_version": "stuck-confirmatory-analysis.v0",
        "analysis_is_frozen_and_fit_free": True,
        "pool_exhausted": pool_exhausted,
        "valid_outcomes": len(rows),
        "matched_groups": len(grouped),
        "dataset": dataset,
        "detector": detector,
        "comparisons": comparisons,
        "directionality": directionality,
        "interactions": {
            "kimi": kimi_interaction,
            "value_switch": value_interaction,
        },
        "success_cost_pareto_at_stuck": pareto,
        "decision_gates": {
            "detector_gate_passed": detector_gate,
            "kimi_intervention_gate_passed": kimi_gate,
            "training_sized_collection_justified": detector_gate and kimi_gate,
            "jaggedness_routes_with_at_least_two_unique_rescues": jaggedness_routes,
        },
        "decision": decision,
        "all_arms": {
            f"{kind}:{action}": _arm_summary(
                [
                    row
                    for row in rows
                    if row["checkpoint_kind"] == kind
                    and row["branch_action"] == action
                ]
            )
            for kind in KINDS
            for action in sorted(BRANCHES)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pool-exhausted", action="store_true")
    args = parser.parse_args()
    rows = _load(args.outcomes)
    report = analyze(rows, pool_exhausted=args.pool_exhausted)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
