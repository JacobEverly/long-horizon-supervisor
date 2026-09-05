from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from horizon_supervisor.stuck_detector import TurnObservation
from horizon_supervisor.stuck_detector_v2 import (
    FROZEN_CANDIDATE_FAMILY,
    TwoTierDetectorConfig,
    TwoTierObservation,
    TwoTierStatus,
    TwoTierStuckDetectorV2,
)
from horizon_supervisor.training.develop_stuck_detector_v1 import (
    DEFAULT_CHECKPOINTS,
    DEFAULT_MANIFEST,
    DEFAULT_OUTCOMES,
    DEFAULT_STATE,
    FLASH_ROUTE,
    ROUTES,
    Trajectory,
    _read_json,
    _read_jsonl,
    load_development_trajectories,
    sha256_file,
)

DEFAULT_V0_SOURCE = Path("src/horizon_supervisor/stuck_detector.py")
DEFAULT_V1_SOURCE = Path("src/horizon_supervisor/stuck_detector_v1.py")
DEFAULT_V2_SOURCE = Path("src/horizon_supervisor/stuck_detector_v2.py")
DEFAULT_EVALUATOR_SOURCE = Path(
    "src/horizon_supervisor/training/develop_two_tier_detector_v2.py"
)
DEFAULT_REPORT = Path(
    "artifacts/official/two-tier-detector-v2/development-report-v0.json"
)
V0_EXPECTED_SHA256 = "c3319c93d823455076fd294ac16e28748a2b2ebcab10e1b81760d174088f4ffe"
V1_EXPECTED_SHA256 = "220f5faade081cd22d283ee68c23647b37e843458afb72dec3a2833ab8b0475c"
MAX_TURNS = 12
FOLD_COUNT = 5
BOOTSTRAP_SAMPLES = 10_000
SEED = 23


@dataclass(frozen=True)
class TierCheckpoint:
    tier: str
    task_id: str
    route_id: str
    trajectory_id: str
    turn: int
    remaining_turns: int
    completed: int


def _task_folds(task_ids: Iterable[str], fold_count: int = FOLD_COUNT) -> dict[str, int]:
    return {
        task_id: index % fold_count
        for index, task_id in enumerate(sorted(set(task_ids)))
    }


def _projected_checkpoint(
    trajectory: Trajectory, tier: str, row: dict[str, Any]
) -> TierCheckpoint:
    turn = int(row["observation"]["turn_index"])
    return TierCheckpoint(
        tier=tier,
        task_id=trajectory.task_id,
        route_id=trajectory.route_id,
        trajectory_id=trajectory.trajectory_id,
        turn=turn,
        remaining_turns=MAX_TURNS - turn,
        completed=trajectory.completed,
    )


def select_projected_tiers(
    trajectories: Sequence[Trajectory], config: TwoTierDetectorConfig
) -> dict[str, list[TierCheckpoint]]:
    """Project v2 onto the online-safe fields shared by natural continuations."""

    selected: dict[str, list[TierCheckpoint]] = {
        "healthy": [],
        "needs_review": [],
        "confirmed_stuck": [],
    }
    for trajectory in trajectories:
        if trajectory.status != "verified":
            continue
        healthy_row = next(
            (
                row
                for row in trajectory.rows
                if int(row["observation"]["turn_index"])
                >= TwoTierStuckDetectorV2.healthy_checkpoint_turn
            ),
            None,
        )
        review_index: int | None = None
        confirm_index: int | None = None
        for index, row in enumerate(trajectory.rows):
            observation = row["observation"]
            turn = int(observation["turn_index"])
            if turn > MAX_TURNS - config.minimum_remaining_turns:
                continue
            review_window = trajectory.rows[max(0, index - 1) : index + 1]
            review_error_turns = sum(
                int(item["observation"]["error_signal_count"] > 0)
                for item in review_window
            )
            review_no_pass = all(
                item["observation"]["pass_signal_count"] == 0
                for item in review_window
            )
            if (
                review_index is None
                and turn >= config.review_minimum_turn
                and review_error_turns >= 1
                and review_no_pass
            ):
                review_index = index

            confirmation_window = trajectory.rows[
                max(0, index - config.confirmation_window + 1) : index + 1
            ]
            confirmation_error_turns = sum(
                int(item["observation"]["error_signal_count"] > 0)
                for item in confirmation_window
            )
            confirmation_no_pass = all(
                item["observation"]["pass_signal_count"] == 0
                for item in confirmation_window
            )
            if (
                review_index is not None
                and index > review_index
                and confirm_index is None
                and turn >= config.confirmation_minimum_turn
                and len(confirmation_window) == config.confirmation_window
                and confirmation_error_turns >= config.confirmation_failure_turns
                and confirmation_no_pass
            ):
                confirm_index = index
                break

        if healthy_row is not None:
            selected["healthy"].append(
                _projected_checkpoint(trajectory, "healthy", healthy_row)
            )
        if review_index is not None:
            selected["needs_review"].append(
                _projected_checkpoint(
                    trajectory, "needs_review", trajectory.rows[review_index]
                )
            )
        if confirm_index is not None:
            selected["confirmed_stuck"].append(
                _projected_checkpoint(
                    trajectory, "confirmed_stuck", trajectory.rows[confirm_index]
                )
            )
    return selected


def _rate(items: Sequence[TierCheckpoint]) -> float | None:
    if not items:
        return None
    return sum(item.completed for item in items) / len(items)


def _gap(healthy: Sequence[TierCheckpoint], confirmed: Sequence[TierCheckpoint]) -> float:
    healthy_rate = _rate(healthy)
    confirmed_rate = _rate(confirmed)
    if healthy_rate is None or confirmed_rate is None:
        return float("-inf")
    return healthy_rate - confirmed_rate


def _development_candidate_eligible(
    selected: dict[str, list[TierCheckpoint]],
) -> bool:
    review = selected["needs_review"]
    confirmed = selected["confirmed_stuck"]
    return (
        len(selected["healthy"]) >= 20
        and len(review) >= 10
        and len({item.task_id for item in review}) >= 6
        and len(confirmed) >= 6
        and len({item.task_id for item in confirmed}) >= 4
        and all(any(item.route_id == route for item in confirmed) for route in ROUTES)
        and all(item.remaining_turns >= 2 for item in confirmed)
    )


def choose_candidate(trajectories: Sequence[Trajectory]) -> TwoTierDetectorConfig:
    scored: list[tuple[float, int, int, str, TwoTierDetectorConfig]] = []
    for candidate in FROZEN_CANDIDATE_FAMILY:
        selected = select_projected_tiers(trajectories, candidate)
        if not _development_candidate_eligible(selected):
            continue
        scored.append(
            (
                _gap(selected["healthy"], selected["confirmed_stuck"]),
                len(selected["confirmed_stuck"]),
                len(selected["needs_review"]),
                candidate.name,
                candidate,
            )
        )
    if not scored:
        raise ValueError("no frozen v2 candidate has minimum development coverage")
    return max(scored)[-1]


def cross_fitted_evaluation(
    trajectories: Sequence[Trajectory], fold_count: int = FOLD_COUNT
) -> tuple[dict[str, list[TierCheckpoint]], list[dict[str, Any]]]:
    folds = _task_folds((trajectory.task_id for trajectory in trajectories), fold_count)
    combined = {"healthy": [], "needs_review": [], "confirmed_stuck": []}
    reports: list[dict[str, Any]] = []
    for fold in range(fold_count):
        development = [item for item in trajectories if folds[item.task_id] != fold]
        evaluation = [item for item in trajectories if folds[item.task_id] == fold]
        candidate = choose_candidate(development)
        selected = select_projected_tiers(evaluation, candidate)
        for tier in combined:
            combined[tier].extend(selected[tier])
        reports.append(
            {
                "fold": fold,
                "development_task_count": len({item.task_id for item in development}),
                "evaluation_task_count": len({item.task_id for item in evaluation}),
                "selected_candidate": candidate.name,
                "tier_counts": {tier: len(items) for tier, items in selected.items()},
                "task_overlap": False,
            }
        )
    return combined, reports


def clustered_interval(
    healthy: Sequence[TierCheckpoint],
    confirmed: Sequence[TierCheckpoint],
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = SEED,
) -> tuple[float, float]:
    healthy_by_task: defaultdict[str, list[TierCheckpoint]] = defaultdict(list)
    confirmed_by_task: defaultdict[str, list[TierCheckpoint]] = defaultdict(list)
    for item in healthy:
        healthy_by_task[item.task_id].append(item)
    for item in confirmed:
        confirmed_by_task[item.task_id].append(item)
    tasks = sorted(set(healthy_by_task) | set(confirmed_by_task))
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(samples):
        task_sample = [rng.choice(tasks) for _ in tasks]
        sampled_healthy = [
            item for task_id in task_sample for item in healthy_by_task[task_id]
        ]
        sampled_confirmed = [
            item for task_id in task_sample for item in confirmed_by_task[task_id]
        ]
        if sampled_healthy and sampled_confirmed:
            values.append(_gap(sampled_healthy, sampled_confirmed))
    if not values:
        raise ValueError("clustered interval has no valid task resamples")
    values.sort()
    return (
        values[int(0.025 * len(values))],
        values[min(len(values) - 1, int(0.975 * len(values)))],
    )


def _tier_metrics(items: Sequence[TierCheckpoint]) -> dict[str, Any]:
    return {
        "checkpoint_count": len(items),
        "task_count": len({item.task_id for item in items}),
        "completed": sum(item.completed for item in items),
        "continuation_recovery_rate": _rate(items),
        "minimum_remaining_turns": min(
            (item.remaining_turns for item in items), default=None
        ),
        "turn_distribution": dict(
            sorted(Counter(str(item.turn) for item in items).items())
        ),
        "by_base_route": {
            route: {
                "checkpoint_count": sum(item.route_id == route for item in items),
                "task_count": len(
                    {item.task_id for item in items if item.route_id == route}
                ),
                "completed": sum(
                    item.completed for item in items if item.route_id == route
                ),
                "continuation_recovery_rate": _rate(
                    [item for item in items if item.route_id == route]
                ),
            }
            for route in ROUTES
        },
    }


def _robustness(
    healthy: Sequence[TierCheckpoint], confirmed: Sequence[TierCheckpoint]
) -> dict[str, Any]:
    by_route: dict[str, Any] = {}
    for route in ROUTES:
        route_healthy = [item for item in healthy if item.route_id == route]
        route_confirmed = [item for item in confirmed if item.route_id == route]
        by_route[route] = {
            "healthy_rate": _rate(route_healthy),
            "confirmed_stuck_rate": _rate(route_confirmed),
            "difference": _gap(route_healthy, route_confirmed),
        }
    tasks = sorted({item.task_id for item in healthy} | {item.task_id for item in confirmed})
    leave_one_out = [
        _gap(
            [item for item in healthy if item.task_id != task_id],
            [item for item in confirmed if item.task_id != task_id],
        )
        for task_id in tasks
        if any(item.task_id != task_id for item in confirmed)
    ]
    maximum_share = max(
        (
            count / len(confirmed)
            for count in Counter(item.task_id for item in confirmed).values()
        ),
        default=1.0,
    )
    not_driven = (
        bool(leave_one_out)
        and min(leave_one_out) > 0
        and maximum_share <= 0.25
        and all(value["difference"] > 0 for value in by_route.values())
    )
    return {
        "by_base_route": by_route,
        "leave_one_task_out_min_difference": min(leave_one_out, default=None),
        "leave_one_task_out_max_difference": max(leave_one_out, default=None),
        "maximum_single_task_share": maximum_share,
        "not_driven_by_one_task_or_base": not_driven,
    }


def replay_full_scout_observations(
    config: TwoTierDetectorConfig,
    state_path: Path = DEFAULT_STATE,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    state = _read_json(state_path)
    manifest = _read_json(manifest_path)
    task_by_position = {
        int(item["position"]): item
        for item in manifest["task_selection"]["ordered_pool"]
    }
    ineligible_by_schedule: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in state["ineligible"]:
        if item.get("schedule_item"):
            ineligible_by_schedule[str(item["schedule_item"])].append(item)

    hits: dict[str, list[dict[str, Any]]] = {
        "healthy": [],
        "needs_review": [],
        "confirmed_stuck": [],
    }
    schedule_audit: list[dict[str, Any]] = []
    for schedule_item in state["completed_schedule_items"]:
        position_text, route_id, requested_kind = str(schedule_item).split(":", 2)
        position = int(position_text)
        route_name = "flash" if route_id == FLASH_ROUTE else "qwen"
        prefix = f"base-{requested_kind}-{position:02d}-{route_name}-"
        matching = [
            attempt
            for attempt in state["attempts"]
            if str(attempt["job_name"]).startswith(prefix)
        ]
        task = task_by_position[position]
        unmanaged = any(
            item.get("reason")
            == "unmanaged process state has no frozen rehydration recipe"
            for item in ineligible_by_schedule.get(str(schedule_item), [])
        )
        if not matching or not Path(matching[-1]["record_path"]).exists():
            schedule_audit.append(
                {
                    "schedule_item": schedule_item,
                    "record_available": False,
                    "structural": unmanaged,
                }
            )
            continue
        attempt = matching[-1]
        events = [
            item
            for item in _read_jsonl(Path(attempt["record_path"]))
            if item.get("schema_version") == "stuck-observation-event.v0"
        ]
        detector = TwoTierStuckDetectorV2(config)
        assessments = []
        for event in events:
            observation = TwoTierObservation.from_v0(
                TurnObservation.model_validate(event["observation"])
            ).model_copy(
                update={
                    "provider_failure": bool(attempt.get("provider_error")),
                    "external_state_reproducible": not unmanaged,
                    "task_category": task["category"],
                }
            )
            assessments.append(detector.observe(observation))

        first_by_tier = {
            "healthy": next(
                (
                    item
                    for item in assessments
                    if item.turn == TwoTierStuckDetectorV2.healthy_checkpoint_turn
                    and item.status == TwoTierStatus.HEALTHY
                ),
                None,
            ),
            "needs_review": next(
                (item for item in assessments if item.status == TwoTierStatus.NEEDS_REVIEW),
                None,
            ),
            "confirmed_stuck": next(
                (
                    item
                    for item in assessments
                    if item.status == TwoTierStatus.CONFIRMED_STUCK
                ),
                None,
            ),
        }
        for tier, assessment in first_by_tier.items():
            if assessment is not None:
                hits[tier].append(
                    {
                        "task_id": task["task_id"],
                        "task_category": task["category"],
                        "route_id": route_id,
                        "turn": assessment.turn,
                        "remaining_turns": assessment.remaining_turns,
                        "action_mode": assessment.action_mode.value,
                    }
                )
        schedule_audit.append(
            {
                "schedule_item": schedule_item,
                "record_available": True,
                "structural": any(
                    item.status == TwoTierStatus.STRUCTURAL_FAILURE
                    for item in assessments
                ),
                "healthy_hit": first_by_tier["healthy"] is not None,
                "needs_review_hit": first_by_tier["needs_review"] is not None,
                "confirmed_stuck_hit": first_by_tier["confirmed_stuck"] is not None,
            }
        )

    def summarize(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
        return {
            "checkpoint_count": len(items),
            "task_count": len({item["task_id"] for item in items}),
            "by_route": dict(sorted(Counter(item["route_id"] for item in items).items())),
            "by_action_mode": dict(
                sorted(Counter(item["action_mode"] for item in items).items())
            ),
            "minimum_remaining_turns": min(
                (item["remaining_turns"] for item in items), default=None
            ),
            "turn_distribution": dict(
                sorted(Counter(str(item["turn"]) for item in items).items())
            ),
        }

    return {
        "scope": "84 frozen prior scout slots; v2 replayed on full pre-outcome observations",
        "planned_schedule_items": len(state["completed_schedule_items"]),
        "records_available": sum(item["record_available"] for item in schedule_audit),
        "structural_schedule_items": sum(item["structural"] for item in schedule_audit),
        "tiers": {tier: summarize(items) for tier, items in hits.items()},
        "task_and_model_rows": hits,
    }


def build_report(
    checkpoints_path: Path = DEFAULT_CHECKPOINTS,
    state_path: Path = DEFAULT_STATE,
    manifest_path: Path = DEFAULT_MANIFEST,
    outcomes_path: Path = DEFAULT_OUTCOMES,
    v0_source: Path = DEFAULT_V0_SOURCE,
    v1_source: Path = DEFAULT_V1_SOURCE,
    v2_source: Path = DEFAULT_V2_SOURCE,
    evaluator_source: Path = DEFAULT_EVALUATOR_SOURCE,
) -> dict[str, Any]:
    if sha256_file(v0_source) != V0_EXPECTED_SHA256:
        raise ValueError("v0 source changed")
    if sha256_file(v1_source) != V1_EXPECTED_SHA256:
        raise ValueError("v1 source changed")
    trajectories = load_development_trajectories(checkpoints_path)
    verified = [item for item in trajectories if item.status == "verified"]
    structural = [item for item in trajectories if item.status != "verified"]
    cross_fitted, folds = cross_fitted_evaluation(trajectories)
    healthy = cross_fitted["healthy"]
    confirmed = cross_fitted["confirmed_stuck"]
    difference = _gap(healthy, confirmed)
    interval = clustered_interval(healthy, confirmed)
    robustness = _robustness(healthy, confirmed)
    final_candidate = choose_candidate(verified)
    final_projection = select_projected_tiers(verified, final_candidate)
    replay = replay_full_scout_observations(final_candidate, state_path, manifest_path)

    replay_review = replay["tiers"]["needs_review"]
    replay_confirmed = replay["tiers"]["confirmed_stuck"]
    replay_healthy = replay["tiers"]["healthy"]
    gates = {
        "healthy_minus_confirmed_at_least_20_points": {
            "passed": difference >= 0.20,
            "observed_difference": difference,
            "threshold": 0.20,
        },
        "task_clustered_interval_excludes_zero": {
            "passed": interval[0] > 0,
            "interval_95": list(interval),
        },
        "directionally_consistent_both_models": {
            "passed": all(
                item["difference"] > 0
                for item in robustness["by_base_route"].values()
            ),
            "by_base_route": robustness["by_base_route"],
        },
        "not_driven_by_one_task": {
            "passed": robustness["not_driven_by_one_task_or_base"],
            "leave_one_task_out_min_difference": robustness[
                "leave_one_task_out_min_difference"
            ],
            "maximum_single_task_share": robustness["maximum_single_task_share"],
        },
        "needs_review_replay_coverage": {
            "passed": (
                replay_review["checkpoint_count"] >= 12
                and replay_review["task_count"] >= 8
            ),
            "checkpoint_threshold": 12,
            "task_threshold": 8,
            **replay_review,
        },
        "confirmed_stuck_replay_coverage": {
            "passed": (
                replay_confirmed["checkpoint_count"] >= 6
                and replay_confirmed["task_count"] >= 4
                and all(replay_confirmed["by_route"].get(route, 0) > 0 for route in ROUTES)
            ),
            "checkpoint_threshold": 6,
            "task_threshold": 4,
            **replay_confirmed,
        },
        "healthy_replay_coverage": {
            "passed": replay_healthy["checkpoint_count"] >= 12,
            "checkpoint_threshold": 12,
            **replay_healthy,
        },
        "confirmed_checkpoints_have_two_remaining_turns": {
            "passed": (
                replay_confirmed["minimum_remaining_turns"] is not None
                and replay_confirmed["minimum_remaining_turns"] >= 2
            ),
            "minimum_remaining_turns": replay_confirmed["minimum_remaining_turns"],
        },
        "structural_failures_separate": {
            "passed": True,
            "structural_schedule_items": replay["structural_schedule_items"],
            "rule": "structural status overrides all detector tiers",
        },
        "leakage_controls": {
            "passed": True,
            "controls": [
                "development rows only",
                "task-grouped candidate selection and evaluation",
                "online-safe shared-feature projection",
                "full replay uses only pre-outcome observations",
                "task identity used for grouping and aggregate diversity only",
                "structural outcomes excluded from recovery labels",
                "no intervention outcome used",
            ],
        },
    }
    gate_passed = all(bool(item["passed"]) for item in gates.values())
    return {
        "schema_version": "two-tier-detector-development-report.v0",
        "decision": (
            "READY_FOR_FRESH_SHADOW_PILOT"
            if gate_passed
            else "NOT READY — two-tier detector gate failed"
        ),
        "gate_passed": gate_passed,
        "frozen_contract": {
            "candidate_family": [
                candidate.model_dump() for candidate in FROZEN_CANDIDATE_FAMILY
            ],
            "fold_count": FOLD_COUNT,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "seed": SEED,
            "candidate_selection": (
                "Within each development fold, maximize healthy-minus-confirmed "
                "continuation recovery among candidates meeting minimum tier, task, "
                "base-model, and remaining-turn coverage."
            ),
            "gate_thresholds": {
                "recovery_difference": 0.20,
                "needs_review_replay_checkpoints": 12,
                "needs_review_replay_tasks": 8,
                "confirmed_replay_checkpoints": 6,
                "confirmed_replay_tasks": 4,
                "healthy_replay_checkpoints": 12,
                "minimum_remaining_turns": 2,
            },
        },
        "integrity": {
            "v0_source_sha256": sha256_file(v0_source),
            "v1_source_sha256": sha256_file(v1_source),
            "v2_source_sha256": sha256_file(v2_source),
            "evaluator_source_sha256": sha256_file(evaluator_source),
            "development_checkpoints_sha256": sha256_file(checkpoints_path),
            "prior_state_sha256": sha256_file(state_path),
            "prior_manifest_sha256": sha256_file(manifest_path),
            "prior_outcomes_sha256": sha256_file(outcomes_path),
        },
        "development_data": {
            "flash_qwen_trajectories": len(trajectories),
            "natural_verified_trajectories": len(verified),
            "structural_or_protocol_trajectories": len(structural),
            "task_count": len({item.task_id for item in trajectories}),
            "structural_excluded_from_recovery_labels": True,
        },
        "task_grouped_evaluation": {
            "folds": folds,
            "tiers": {
                tier: _tier_metrics(items) for tier, items in cross_fitted.items()
            },
            "healthy_minus_confirmed_difference": difference,
            "task_clustered_interval_95": list(interval),
            "robustness": robustness,
            "interpretation": (
                "Development evidence only; candidate design and evaluation reuse the "
                "same development corpus and are not independent confirmation."
            ),
        },
        "final_development_candidate": {
            "config": final_candidate.model_dump(),
            "projected_tiers_in_sample_not_used_for_gate": {
                tier: _tier_metrics(items) for tier, items in final_projection.items()
            },
            "full_observation_84_scout_replay": replay,
        },
        "offline_gate": gates,
        "next_phase": {
            "fresh_task_source": "permitted" if gate_passed else "not_run",
            "paid_shadow_pilot": "permitted" if gate_passed else "not_run",
            "intervention_outcomes": "forbidden",
            "intervention_training": "forbidden",
            "reason": "offline_gate_passed" if gate_passed else "offline_gate_failed",
        },
        "limitations": [
            (
                "Natural continuation trajectories expose an online-safe summary, not "
                "the complete live v2 observation, so tier recovery is a projection."
            ),
            (
                "The exact 84-scout replay has full observations but most scouts were "
                "stopped at checkpoints and cannot supply unbiased recovery labels."
            ),
            "No intervention action is labeled or evaluated in this milestone.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    report = build_report(args.checkpoints, args.state, args.manifest, args.outcomes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"decision": report["decision"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
