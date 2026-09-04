from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from statistics import mean
from typing import Any

BRANCHES = {
    "continue_current_state",
    "restart_current_clean",
    "switch_value_state",
    "restart_value_clean",
    "switch_kimi_state",
    "restart_kimi_clean",
}


def _load(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise ValueError("outcome table is empty")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("valid") is not True:
            continue
        grouped[row["group_id"]].append(row)
    for group_id, group in grouped.items():
        observed = {row["branch_action"] for row in group}
        if observed != BRANCHES or len(group) != len(BRANCHES):
            raise ValueError(f"matched group {group_id} is incomplete or duplicated")
        limits = {
            (
                row["remaining_turns"],
                row["remaining_output_tokens"],
                row["maximum_wall_seconds"],
                row["maximum_incremental_spend_usd"],
            )
            for row in group
        }
        if len(limits) != 1:
            raise ValueError(f"matched group {group_id} has unequal branch limits")
    return [row for group in grouped.values() for row in group]


def _by_group(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[row["group_id"]][row["branch_action"]] = row
    return dict(grouped)


def _cluster_interval(
    items: list[dict[str, Any]],
    statistic: Callable[[list[dict[str, Any]]], float],
    *,
    seed: int = 20260902,
    samples: int = 10_000,
) -> list[float] | None:
    tasks = sorted({item["task_id"] for item in items})
    if len(tasks) < 2:
        return None
    by_task = {task: [item for item in items if item["task_id"] == task] for task in tasks}
    rng = random.Random(seed)
    values = []
    for _ in range(samples):
        sampled = [rng.choice(tasks) for _ in tasks]
        sample_items = [item for task in sampled for item in by_task[task]]
        values.append(statistic(sample_items))
    values.sort()
    return [values[int(samples * 0.025)], values[int(samples * 0.975) - 1]]


def _arm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "outcomes": len(rows),
        "verified_completions": sum(bool(row["verified_completion"]) for row in rows),
        "completion_rate": mean(bool(row["verified_completion"]) for row in rows),
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
        {"task_id": group[left]["task_id"], "left": group[left], "right": group[right]}
        for group in groups
    ]

    def delta(sample: list[dict[str, Any]]) -> float:
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
        "rescue_rate": mean(
            bool(item["left"]["verified_completion"])
            and not bool(item["right"]["verified_completion"])
            for item in pairs
        ),
        "harm_count": sum(
            not bool(item["left"]["verified_completion"])
            and bool(item["right"]["verified_completion"])
            for item in pairs
        ),
        "harm_rate": mean(
            not bool(item["left"]["verified_completion"])
            and bool(item["right"]["verified_completion"])
            for item in pairs
        ),
    }


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
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

    continue_stuck = [group["continue_current_state"] for group in stuck]
    continue_healthy = [group["continue_current_state"] for group in healthy]
    detector_gap_items = [
        {
            "task_id": row["task_id"],
            "checkpoint_kind": "suspected_stuck",
            "success": bool(row["verified_completion"]),
        }
        for row in continue_stuck
    ] + [
        {
            "task_id": row["task_id"],
            "checkpoint_kind": "healthy",
            "success": bool(row["verified_completion"]),
        }
        for row in continue_healthy
    ]

    def detector_gap(sample: list[dict[str, Any]]) -> float:
        stuck_values = [
            item["success"]
            for item in sample
            if item["checkpoint_kind"] == "suspected_stuck"
        ]
        healthy_values = [
            item["success"]
            for item in sample
            if item["checkpoint_kind"] == "healthy"
        ]
        if not stuck_values or not healthy_values:
            return 0.0
        return mean(healthy_values) - mean(stuck_values)

    directionality = {}
    for base in (
        "deepseek/deepseek-v4-flash-0731",
        "qwen/qwen3.8-27b",
    ):
        subset = [
            group
            for group in stuck
            if next(iter(group.values()))["base_model_id"] == base
        ]
        directionality[base] = _paired_contrast(
            subset, "switch_value_state", "continue_current_state"
        ) if subset else None

    interaction = {}
    for action in ("switch_value_state", "switch_kimi_state"):
        stuck_contrast = _paired_contrast(stuck, action, "continue_current_state")
        healthy_contrast = _paired_contrast(healthy, action, "continue_current_state")
        interaction[action] = {
            "triggered_intervention_effect": stuck_contrast[
                "completion_rate_difference"
            ],
            "fixed_turn_intervention_effect": healthy_contrast[
                "completion_rate_difference"
            ],
            "difference_in_differences": (
                stuck_contrast["completion_rate_difference"]
                - healthy_contrast["completion_rate_difference"]
            ),
        }

    comparisons = {
        "cross_model_state_vs_continue_at_stuck": _paired_contrast(
            stuck, "switch_value_state", "continue_current_state"
        ),
        "kimi_state_vs_continue_at_stuck": _paired_contrast(
            stuck, "switch_kimi_state", "continue_current_state"
        ),
        "kimi_state_vs_cross_model_state_at_stuck": _paired_contrast(
            stuck, "switch_kimi_state", "switch_value_state"
        ),
        "cross_model_state_vs_destination_clean_at_stuck": _paired_contrast(
            stuck, "switch_value_state", "restart_value_clean"
        ),
        "kimi_state_vs_kimi_clean_at_stuck": _paired_contrast(
            stuck, "switch_kimi_state", "restart_kimi_clean"
        ),
        "continue_vs_current_clean_at_stuck": _paired_contrast(
            stuck, "continue_current_state", "restart_current_clean"
        ),
    }
    return {
        "schema_version": "matched-stuck-intervention-analysis.v0",
        "analysis_is_frozen_and_fit_free": True,
        "valid_outcomes": len(rows),
        "matched_groups": len(grouped),
        "stuck_groups": len(stuck),
        "healthy_groups": len(healthy),
        "detector": {
            "continue_at_stuck": _arm_summary(continue_stuck),
            "continue_at_healthy": _arm_summary(continue_healthy),
            "healthy_minus_stuck_recovery_rate": detector_gap(detector_gap_items),
            "healthy_minus_stuck_task_bootstrap_95": _cluster_interval(
                detector_gap_items, detector_gap
            ),
            "false_positive_stuck_triggers": sum(
                bool(row["verified_completion"]) for row in continue_stuck
            ),
            "false_positive_stuck_trigger_rate": mean(
                bool(row["verified_completion"]) for row in continue_stuck
            ),
        },
        "comparisons": comparisons,
        "directionality": directionality,
        "trigger_vs_fixed_turn_interaction": interaction,
        "all_arms": {
            f"{kind}:{action}": _arm_summary(
                [
                    row
                    for row in rows
                    if row["checkpoint_kind"] == kind
                    and row["branch_action"] == action
                ]
            )
            for kind in ("suspected_stuck", "healthy")
            for action in sorted(BRANCHES)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = _load(args.outcomes)
    report = analyze(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
