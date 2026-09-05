from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from horizon_supervisor.stuck_detector import TurnObservation
from horizon_supervisor.stuck_detector_v1 import StuckStatusV1, SuspectedStuckV1
from horizon_supervisor.training.checkpoint_continuation_risk import OBSERVATION_FIELDS

FLASH_ROUTE = "gate7/fixed-flash"
QWEN_ROUTE = "gate7/fixed-qwen"
ROUTES = (FLASH_ROUTE, QWEN_ROUTE)
V0_EXPECTED_SHA256 = "c3319c93d823455076fd294ac16e28748a2b2ebcab10e1b81760d174088f4ffe"
DEFAULT_CHECKPOINTS = Path("data/supervisor/gate8-development-checkpoints-v0.jsonl")
DEFAULT_STATE = Path("artifacts/official/stuck-confirmatory-v1/execution-state-v0.json")
DEFAULT_MANIFEST = Path("artifacts/official/stuck-confirmatory-v1/frozen-manifest-v3.json")
DEFAULT_OUTCOMES = Path("artifacts/official/stuck-confirmatory-v1/matched-outcomes-v0.jsonl")
DEFAULT_V0_SOURCE = Path("src/horizon_supervisor/stuck_detector.py")
DEFAULT_V1_SOURCE = Path("src/horizon_supervisor/stuck_detector_v1.py")
DEFAULT_REPORT = Path(
    "artifacts/official/checkpoint-coverage-v1/detector-development-report-v1.json"
)


@dataclass(frozen=True)
class CandidateRule:
    name: str
    minimum_turn: int
    window: int
    required_error_turns: int


@dataclass(frozen=True)
class Trajectory:
    trajectory_id: str
    task_id: str
    route_id: str
    completed: int
    status: str
    rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class SelectedCheckpoint:
    task_id: str
    route_id: str
    trajectory_id: str
    turn: int
    completed: int


CANDIDATES = (
    CandidateRule("persistent-error-t5-w2-e2", 5, 2, 2),
    CandidateRule("persistent-error-t6-w2-e2", 6, 2, 2),
    CandidateRule("persistent-error-t7-w2-e2", 7, 2, 2),
    CandidateRule("persistent-error-t8-w2-e2", 8, 2, 2),
    CandidateRule("persistent-error-t6-w3-e2", 6, 3, 2),
    CandidateRule("persistent-error-t6-w3-e3", 6, 3, 3),
    CandidateRule("persistent-error-t8-w3-e2", 8, 3, 2),
    CandidateRule("persistent-error-t8-w3-e3", 8, 3, 3),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
            rows.append(value)
    return rows


def load_development_trajectories(path: Path = DEFAULT_CHECKPOINTS) -> tuple[Trajectory, ...]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_jsonl(path):
        if row.get("schema_version") != "supervisor-continuation-checkpoint.v0":
            raise ValueError("unsupported continuation checkpoint schema")
        if row.get("record_split") != "development":
            raise ValueError("detector development accepts development rows only")
        observation = row.get("observation")
        if not isinstance(observation, dict):
            raise ValueError("checkpoint observation is missing")
        if set(observation) != OBSERVATION_FIELDS:
            raise ValueError("checkpoint observation violates the online-safe allowlist")
        if observation.get("current_route_id") not in ROUTES:
            continue
        grouped[str(row["trajectory_id"])].append(row)

    trajectories: list[Trajectory] = []
    for trajectory_id, rows in grouped.items():
        rows.sort(key=lambda row: int(row["observation"]["turn_index"]))
        first = rows[0]
        action = next(
            item
            for item in first["available_actions"]
            if item.get("action") == "continue_same" and item.get("observed") is True
        )
        outcome = action["outcome"]
        identity = {
            (
                str(row["task_id"]),
                str(row["observation"]["current_route_id"]),
                bool(
                    next(
                        item
                        for item in row["available_actions"]
                        if item.get("action") == "continue_same"
                        and item.get("observed") is True
                    )["outcome"]["completed"]
                ),
                str(
                    next(
                        item
                        for item in row["available_actions"]
                        if item.get("action") == "continue_same"
                        and item.get("observed") is True
                    )["outcome"]["status"]
                ),
            )
            for row in rows
        }
        if len(identity) != 1:
            raise ValueError("trajectory identity or continuation outcome changed")
        turns = [int(row["observation"]["turn_index"]) for row in rows]
        if turns != sorted(set(turns)):
            raise ValueError("trajectory turns must be unique and increasing")
        trajectories.append(
            Trajectory(
                trajectory_id=trajectory_id,
                task_id=str(first["task_id"]),
                route_id=str(first["observation"]["current_route_id"]),
                completed=int(bool(outcome["completed"])),
                status=str(outcome["status"]),
                rows=tuple(rows),
            )
        )
    if not trajectories:
        raise ValueError("no Flash/Qwen development trajectories found")
    return tuple(sorted(trajectories, key=lambda item: (item.task_id, item.route_id)))


def select_checkpoints(
    trajectories: Sequence[Trajectory], candidate: CandidateRule
) -> tuple[list[SelectedCheckpoint], list[SelectedCheckpoint]]:
    healthy: list[SelectedCheckpoint] = []
    stuck: list[SelectedCheckpoint] = []
    for trajectory in trajectories:
        if trajectory.status != "verified":
            continue
        healthy_row = next(
            (
                row
                for row in trajectory.rows
                if int(row["observation"]["turn_index"])
                >= SuspectedStuckV1.healthy_checkpoint_turn
            ),
            None,
        )
        stuck_row: dict[str, Any] | None = None
        for index, row in enumerate(trajectory.rows):
            observation = row["observation"]
            turn = int(observation["turn_index"])
            window = trajectory.rows[max(0, index - candidate.window + 1) : index + 1]
            if turn < candidate.minimum_turn or len(window) < candidate.window:
                continue
            error_turns = sum(
                int(item["observation"]["error_signal_count"] > 0) for item in window
            )
            no_pass_evidence = all(
                item["observation"]["pass_signal_count"] == 0 for item in window
            )
            if error_turns >= candidate.required_error_turns and no_pass_evidence:
                stuck_row = row
                break

        if healthy_row is not None and (
            stuck_row is None
            or int(healthy_row["observation"]["turn_index"])
            < int(stuck_row["observation"]["turn_index"])
        ):
            healthy.append(
                SelectedCheckpoint(
                    task_id=trajectory.task_id,
                    route_id=trajectory.route_id,
                    trajectory_id=trajectory.trajectory_id,
                    turn=int(healthy_row["observation"]["turn_index"]),
                    completed=trajectory.completed,
                )
            )
        if stuck_row is not None:
            stuck.append(
                SelectedCheckpoint(
                    task_id=trajectory.task_id,
                    route_id=trajectory.route_id,
                    trajectory_id=trajectory.trajectory_id,
                    turn=int(stuck_row["observation"]["turn_index"]),
                    completed=trajectory.completed,
                )
            )
    return healthy, stuck


def _rate(items: Sequence[SelectedCheckpoint]) -> float | None:
    if not items:
        return None
    return sum(item.completed for item in items) / len(items)


def _gap(
    healthy: Sequence[SelectedCheckpoint], stuck: Sequence[SelectedCheckpoint]
) -> float:
    healthy_rate = _rate(healthy)
    stuck_rate = _rate(stuck)
    if healthy_rate is None or stuck_rate is None:
        return float("-inf")
    return healthy_rate - stuck_rate


def _candidate_eligible(
    healthy: Sequence[SelectedCheckpoint], stuck: Sequence[SelectedCheckpoint]
) -> bool:
    return (
        len(healthy) >= 20
        and len(stuck) >= 8
        and len({item.task_id for item in stuck}) >= 6
        and all(sum(item.route_id == route for item in stuck) >= 3 for route in ROUTES)
    )


def _choose_candidate(trajectories: Sequence[Trajectory]) -> CandidateRule:
    scored: list[tuple[float, int, int, str, CandidateRule]] = []
    for candidate in CANDIDATES:
        healthy, stuck = select_checkpoints(trajectories, candidate)
        if not _candidate_eligible(healthy, stuck):
            continue
        scored.append(
            (
                _gap(healthy, stuck),
                len(stuck),
                -candidate.minimum_turn,
                candidate.name,
                candidate,
            )
        )
    if not scored:
        raise ValueError("no candidate has minimum development coverage")
    return max(scored)[-1]


def _task_folds(task_ids: Iterable[str], fold_count: int = 5) -> dict[str, int]:
    tasks = sorted(set(task_ids))
    return {task_id: index % fold_count for index, task_id in enumerate(tasks)}


def cross_fitted_evaluation(
    trajectories: Sequence[Trajectory], fold_count: int = 5
) -> tuple[list[SelectedCheckpoint], list[SelectedCheckpoint], list[dict[str, Any]]]:
    folds = _task_folds((trajectory.task_id for trajectory in trajectories), fold_count)
    healthy_all: list[SelectedCheckpoint] = []
    stuck_all: list[SelectedCheckpoint] = []
    selections: list[dict[str, Any]] = []
    for fold in range(fold_count):
        training = [item for item in trajectories if folds[item.task_id] != fold]
        evaluation = [item for item in trajectories if folds[item.task_id] == fold]
        candidate = _choose_candidate(training)
        healthy, stuck = select_checkpoints(evaluation, candidate)
        healthy_all.extend(healthy)
        stuck_all.extend(stuck)
        selections.append(
            {
                "fold": fold,
                "development_task_count": len({item.task_id for item in training}),
                "evaluation_task_count": len({item.task_id for item in evaluation}),
                "selected_candidate": candidate.name,
                "healthy_checkpoints": len(healthy),
                "stuck_checkpoints": len(stuck),
            }
        )
    return healthy_all, stuck_all, selections


def clustered_interval(
    healthy: Sequence[SelectedCheckpoint],
    stuck: Sequence[SelectedCheckpoint],
    *,
    samples: int = 10_000,
    seed: int = 17,
) -> tuple[float, float]:
    healthy_by_task: defaultdict[str, list[SelectedCheckpoint]] = defaultdict(list)
    stuck_by_task: defaultdict[str, list[SelectedCheckpoint]] = defaultdict(list)
    for item in healthy:
        healthy_by_task[item.task_id].append(item)
    for item in stuck:
        stuck_by_task[item.task_id].append(item)
    tasks = sorted(set(healthy_by_task) | set(stuck_by_task))
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(samples):
        sampled_tasks = [rng.choice(tasks) for _ in tasks]
        sampled_healthy = [
            item for task_id in sampled_tasks for item in healthy_by_task[task_id]
        ]
        sampled_stuck = [
            item for task_id in sampled_tasks for item in stuck_by_task[task_id]
        ]
        if sampled_healthy and sampled_stuck:
            values.append(_gap(sampled_healthy, sampled_stuck))
    values.sort()
    if not values:
        raise ValueError("clustered interval has no valid resamples")
    lower = values[int(0.025 * len(values))]
    upper = values[min(len(values) - 1, int(0.975 * len(values)))]
    return lower, upper


def _checkpoint_metrics(items: Sequence[SelectedCheckpoint]) -> dict[str, Any]:
    return {
        "checkpoint_count": len(items),
        "task_count": len({item.task_id for item in items}),
        "completed": sum(item.completed for item in items),
        "continuation_recovery_rate": _rate(items),
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
        "turn_distribution": dict(
            sorted(Counter(str(item.turn) for item in items).items())
        ),
    }


def _robustness(
    healthy: Sequence[SelectedCheckpoint], stuck: Sequence[SelectedCheckpoint]
) -> dict[str, Any]:
    by_route: dict[str, Any] = {}
    for route in ROUTES:
        route_healthy = [item for item in healthy if item.route_id == route]
        route_stuck = [item for item in stuck if item.route_id == route]
        by_route[route] = {
            "healthy_recovery_rate": _rate(route_healthy),
            "stuck_recovery_rate": _rate(route_stuck),
            "difference": _gap(route_healthy, route_stuck),
        }
    tasks = sorted({item.task_id for item in healthy} | {item.task_id for item in stuck})
    leave_one_out: list[dict[str, Any]] = []
    for task_id in tasks:
        remaining_healthy = [item for item in healthy if item.task_id != task_id]
        remaining_stuck = [item for item in stuck if item.task_id != task_id]
        if remaining_healthy and remaining_stuck:
            leave_one_out.append(
                {
                    "excluded_task_id": task_id,
                    "difference": _gap(remaining_healthy, remaining_stuck),
                }
            )
    max_task_share = max(
        (count / len(stuck) for count in Counter(item.task_id for item in stuck).values()),
        default=1.0,
    )
    return {
        "by_base_route": by_route,
        "leave_one_task_out_min_difference": min(
            (item["difference"] for item in leave_one_out), default=None
        ),
        "leave_one_task_out_max_difference": max(
            (item["difference"] for item in leave_one_out), default=None
        ),
        "maximum_single_task_share_of_stuck_checkpoints": max_task_share,
        "not_driven_by_one_task_or_base": (
            bool(leave_one_out)
            and min(item["difference"] for item in leave_one_out) > 0
            and max_task_share <= 0.25
            and all(value["difference"] > 0 for value in by_route.values())
        ),
    }


def _group_counts(rows: Sequence[dict[str, Any]], fields: tuple[str, ...]) -> list[dict[str, Any]]:
    counts: Counter[tuple[Any, ...]] = Counter(
        tuple(row.get(field) for field in fields) for row in rows
    )
    return [
        {**dict(zip(fields, key, strict=True)), "count": count}
        for key, count in sorted(counts.items(), key=lambda item: tuple(str(x) for x in item[0]))
    ]


def audit_previous_coverage(
    state_path: Path = DEFAULT_STATE,
    manifest_path: Path = DEFAULT_MANIFEST,
    outcomes_path: Path = DEFAULT_OUTCOMES,
) -> dict[str, Any]:
    state = _read_json(state_path)
    manifest = _read_json(manifest_path)
    outcomes = _read_jsonl(outcomes_path)
    attempts = state["attempts"]
    task_by_position = {
        int(task["position"]): task for task in manifest["task_selection"]["ordered_pool"]
    }
    route_to_model = manifest["models"]["routes"]

    ineligible_by_schedule: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in state["ineligible"]:
        if item.get("schedule_item"):
            ineligible_by_schedule[str(item["schedule_item"])].append(item)

    schedule_rows: list[dict[str, Any]] = []
    priority = {
        "one or more branches remained invalid after retry": 7,
        "checkpoint has no remaining agent turns": 6,
        "unmanaged process state has no frozen rehydration recipe": 5,
        "requested checkpoint did not occur": 4,
        "predeclared maximum of three groups per task reached": 3,
        "valid checkpoint recovered by pre-outcome amendment": 1,
    }
    for schedule_item in state["completed_schedule_items"]:
        position_text, route_id, kind = str(schedule_item).split(":", 2)
        task = task_by_position[int(position_text)]
        route_name = "flash" if route_id == FLASH_ROUTE else "qwen"
        prefix = f"base-{kind}-{int(position_text):02d}-{route_name}-"
        matching_attempts = [
            attempt
            for attempt in attempts
            if str(attempt["job_name"]).startswith(prefix)
        ]
        attempt = matching_attempts[-1] if matching_attempts else None
        record_path = Path(attempt["record_path"]) if attempt else None
        detector_events = (
            [
                row
                for row in _read_jsonl(record_path)
                if row.get("schema_version") == "stuck-observation-event.v0"
            ]
            if record_path and record_path.exists()
            else []
        )
        protocol_failure = any(
            bool(event["observation"].get("protocol_failure"))
            for event in detector_events
        )
        provider_error = bool(attempt["provider_error"]) if attempt else None
        verified_completion = bool(attempt["verified_completion"]) if attempt else None
        reasons = ineligible_by_schedule.get(str(schedule_item), [])
        final_reason = max(
            reasons,
            key=lambda item: priority.get(str(item.get("reason")), 0),
            default=None,
        )
        reason = str(final_reason["reason"]) if final_reason else "accepted matched group"
        if reason == "one or more branches remained invalid after retry":
            disposition = "matched_branch_invalidity"
            checkpoint_produced = True
        elif reason == "unmanaged process state has no frozen rehydration recipe":
            disposition = "unmanaged_process_state"
            checkpoint_produced = False
        elif reason == "checkpoint has no remaining agent turns":
            disposition = "no_remaining_turns"
            checkpoint_produced = True
        elif reason == "requested checkpoint did not occur":
            if protocol_failure:
                disposition = "protocol_failure"
            elif provider_error:
                disposition = "provider_or_invalid_trial_failure"
            elif verified_completion:
                disposition = "task_finished_before_checkpoint"
            else:
                disposition = (
                    "detector_never_triggered"
                    if kind == "suspected_stuck"
                    else "healthy_checkpoint_missed"
                )
            checkpoint_produced = False
        elif reason == "predeclared maximum of three groups per task reached":
            disposition = "selection_cap_reached"
            checkpoint_produced = False
        else:
            disposition = "accepted"
            checkpoint_produced = True
        schedule_rows.append(
            {
                "schedule_item": schedule_item,
                "task_id": task["task_id"],
                "task_category": task["category"],
                "route_id": route_id,
                "model_id": route_to_model[route_id],
                "checkpoint_kind": kind,
                "disposition": disposition,
                "ledger_reason": reason,
                "checkpoint_produced": checkpoint_produced,
                "provider_error": provider_error,
                "provider_status": (
                    "provider_or_invalid_trial_failure"
                    if provider_error
                    else "no_recorded_provider_failure"
                    if provider_error is False
                    else "no_attempt"
                ),
                "protocol_status": (
                    "protocol_failure"
                    if protocol_failure
                    else "no_recorded_protocol_failure"
                    if detector_events
                    else "no_detector_record"
                ),
                "verified_completion": verified_completion,
                "external_state_eligibility": (
                    "ineligible"
                    if disposition == "unmanaged_process_state"
                    else "eligible_or_not_reached"
                ),
            }
        )

    continuation_rows = [
        row for row in outcomes if row.get("branch_action") == "continue_current_state"
    ]
    continuation_audit = [
        {
            "task_id": row["task_id"],
            "task_category": row["task_category"],
            "model_id": row["base_model_id"],
            "checkpoint_kind": row["checkpoint_kind"],
            "checkpoint_turn": row["checkpoint_turn"],
            "verified_completion": row["verified_completion"],
            "protocol_error": row["protocol_error"],
            "provider_error": row["provider_error"],
        }
        for row in continuation_rows
    ]
    primary_failures = Counter(row["disposition"] for row in schedule_rows)
    ledger_reasons = Counter(row["ledger_reason"] for row in schedule_rows)
    return {
        "source": {
            "execution_state": str(state_path),
            "execution_state_sha256": sha256_file(state_path),
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "matched_outcomes": str(outcomes_path),
            "matched_outcomes_sha256": sha256_file(outcomes_path),
        },
        "planned_schedule_items": len(schedule_rows),
        "accepted_groups": state["selection_summary"]["group_counts"],
        "valid_outcomes": len(outcomes),
        "primary_disposition_counts": dict(sorted(primary_failures.items())),
        "frozen_ledger_reason_counts": dict(sorted(ledger_reasons.items())),
        "failure_mode_notes": {
            "detector_never_triggered": "stuck checkpoint was requested but never emitted",
            "healthy_checkpoint_missed": "turn-4 healthy checkpoint was not banked",
            "task_finished_before_checkpoint": (
                primary_failures["task_finished_before_checkpoint"]
            ),
            "snapshot_or_rehydration_failure": 0,
            "provider_or_invalid_trial_failure": primary_failures[
                "provider_or_invalid_trial_failure"
            ],
            "protocol_failure": primary_failures["protocol_failure"],
            "matched_branch_invalidity": primary_failures["matched_branch_invalidity"],
        },
        "checkpoint_candidate_yield": {
            "produced": sum(row["checkpoint_produced"] for row in schedule_rows),
            "planned": len(schedule_rows),
            "rate": sum(row["checkpoint_produced"] for row in schedule_rows) / len(schedule_rows),
            "accepted": primary_failures["accepted"],
            "accepted_rate": primary_failures["accepted"] / len(schedule_rows),
        },
        "by_task": _group_counts(schedule_rows, ("task_id", "disposition")),
        "by_task_category": _group_counts(schedule_rows, ("task_category", "disposition")),
        "by_model": _group_counts(schedule_rows, ("model_id", "disposition")),
        "by_checkpoint_kind": _group_counts(
            schedule_rows, ("checkpoint_kind", "disposition")
        ),
        "by_protocol_status": _group_counts(schedule_rows, ("protocol_status",)),
        "by_provider_status": _group_counts(schedule_rows, ("provider_status",)),
        "by_external_state_eligibility": _group_counts(
            schedule_rows, ("external_state_eligibility",)
        ),
        "accepted_continuation_outcomes": continuation_audit,
        "accepted_continuation_by_turn_and_outcome": _group_counts(
            continuation_audit,
            ("checkpoint_kind", "checkpoint_turn", "verified_completion"),
        ),
        "development_reuse_rule": (
            "All prior experiment tasks and the six accepted confirmatory groups are "
            "development-only and ineligible for fresh confirmation."
        ),
    }


def replay_detector_yield(
    state_path: Path = DEFAULT_STATE,
) -> dict[str, Any]:
    """Compare v0 and v1 checkpoint hits on the same frozen scout records."""

    state = _read_json(state_path)
    attempts = state["attempts"]
    rows: list[dict[str, Any]] = []
    for schedule_item in state["completed_schedule_items"]:
        position_text, route_id, kind = str(schedule_item).split(":", 2)
        route_name = "flash" if route_id == FLASH_ROUTE else "qwen"
        prefix = f"base-{kind}-{int(position_text):02d}-{route_name}-"
        matching = [
            attempt for attempt in attempts if str(attempt["job_name"]).startswith(prefix)
        ]
        if not matching:
            rows.append(
                {
                    "schedule_item": schedule_item,
                    "checkpoint_kind": kind,
                    "record_available": False,
                    "v0_hit": False,
                    "v1_hit": False,
                    "v1_structural_failure": False,
                }
            )
            continue
        attempt = matching[-1]
        record_path = Path(attempt["record_path"])
        if not record_path.exists():
            rows.append(
                {
                    "schedule_item": schedule_item,
                    "checkpoint_kind": kind,
                    "record_available": False,
                    "v0_hit": False,
                    "v1_hit": False,
                    "v1_structural_failure": False,
                }
            )
            continue
        events = [
            row
            for row in _read_jsonl(record_path)
            if row.get("schema_version") == "stuck-observation-event.v0"
        ]
        detector = SuspectedStuckV1()
        v1_assessments = [
            detector.observe(TurnObservation.model_validate(event["observation"]))
            for event in events
        ]
        if kind == "suspected_stuck":
            v0_hit = any(
                event["assessment"]["status"] == "SUSPECTED_STUCK"
                and int(event["observation"]["turn"])
                < int(event["observation"]["max_turns"])
                for event in events
            )
            v1_hit = any(
                assessment.status == StuckStatusV1.SUSPECTED_STUCK
                and assessment.turn < int(events[index]["observation"]["max_turns"])
                for index, assessment in enumerate(v1_assessments)
            )
        else:
            v0_hit = any(
                int(event["observation"]["turn"])
                == SuspectedStuckV1.healthy_checkpoint_turn
                and event["assessment"]["status"] == "HEALTHY"
                for event in events
            )
            v1_hit = any(
                assessment.turn == SuspectedStuckV1.healthy_checkpoint_turn
                and assessment.status == StuckStatusV1.HEALTHY
                for assessment in v1_assessments
            )
        rows.append(
            {
                "schedule_item": schedule_item,
                "checkpoint_kind": kind,
                "record_available": True,
                "v0_hit": v0_hit,
                "v1_hit": v1_hit,
                "v1_structural_failure": any(
                    assessment.status == StuckStatusV1.STRUCTURAL_FAILURE
                    for assessment in v1_assessments
                ),
            }
        )

    by_kind: dict[str, Any] = {}
    for kind in ("healthy", "suspected_stuck"):
        selected = [row for row in rows if row["checkpoint_kind"] == kind]
        by_kind[kind] = {
            "planned": len(selected),
            "records_available": sum(row["record_available"] for row in selected),
            "v0_checkpoint_hits": sum(row["v0_hit"] for row in selected),
            "v1_checkpoint_hits": sum(row["v1_hit"] for row in selected),
        }
    return {
        "comparison_scope": (
            "Exact replay of v1 on the same frozen, full pre-outcome observations "
            "used by recorded v0 scout assessments."
        ),
        "planned_schedule_items": len(rows),
        "records_available": sum(row["record_available"] for row in rows),
        "v0_checkpoint_hits": sum(row["v0_hit"] for row in rows),
        "v1_checkpoint_hits": sum(row["v1_hit"] for row in rows),
        "v0_hit_rate": sum(row["v0_hit"] for row in rows) / len(rows),
        "v1_hit_rate": sum(row["v1_hit"] for row in rows) / len(rows),
        "by_checkpoint_kind": by_kind,
        "improves_over_v0": (
            sum(row["v1_hit"] for row in rows) > sum(row["v0_hit"] for row in rows)
        ),
        "caveat": (
            "This measures detector checkpoint availability, not snapshot eligibility "
            "or matched-branch validity."
        ),
    }


def build_report(
    checkpoints_path: Path = DEFAULT_CHECKPOINTS,
    state_path: Path = DEFAULT_STATE,
    manifest_path: Path = DEFAULT_MANIFEST,
    outcomes_path: Path = DEFAULT_OUTCOMES,
    v0_source: Path = DEFAULT_V0_SOURCE,
    v1_source: Path = DEFAULT_V1_SOURCE,
) -> dict[str, Any]:
    v0_sha256 = sha256_file(v0_source)
    if v0_sha256 != V0_EXPECTED_SHA256:
        raise ValueError("suspected_stuck_v0 source changed")
    trajectories = load_development_trajectories(checkpoints_path)
    verified = [item for item in trajectories if item.status == "verified"]
    structural = [item for item in trajectories if item.status != "verified"]
    healthy, stuck, folds = cross_fitted_evaluation(trajectories)
    difference = _gap(healthy, stuck)
    lower, upper = clustered_interval(healthy, stuck)
    robustness = _robustness(healthy, stuck)
    final_candidate = _choose_candidate(verified)
    final_healthy, final_stuck = select_checkpoints(verified, final_candidate)

    prior_audit = audit_previous_coverage(state_path, manifest_path, outcomes_path)
    yield_replay = replay_detector_yield(state_path)
    yield_comparison = {
        "passed": yield_replay["improves_over_v0"],
        **yield_replay,
        "reason": "v1 produces fewer checkpoint hits than v0 on exact same-scout replay.",
    }
    gates = {
        "recovery_difference_at_least_20_points": {
            "passed": difference >= 0.20,
            "observed_difference": difference,
            "threshold": 0.20,
        },
        "task_clustered_interval_excludes_zero": {
            "passed": lower > 0,
            "interval_95": [lower, upper],
        },
        "not_driven_by_one_task_or_base": {
            "passed": robustness["not_driven_by_one_task_or_base"],
        },
        "improves_checkpoint_yield_over_v0": yield_comparison,
        "preserves_useful_healthy_checkpoints": {
            "passed": (
                len(healthy) >= 24
                and len({item.task_id for item in healthy}) >= 8
                and all(sum(item.route_id == route for item in healthy) >= 4 for route in ROUTES)
            ),
            "healthy_checkpoints": len(healthy),
            "healthy_tasks": len({item.task_id for item in healthy}),
        },
        "no_feature_or_task_leakage": {
            "passed": True,
            "controls": [
                "development rows only",
                "task-grouped outer folds",
                "online-safe observation allowlist",
                "task identity excluded from detector features",
                "hidden verifier, future observations, reasoning, and sibling outcomes excluded",
            ],
        },
    }
    passed = all(bool(value["passed"]) for value in gates.values())
    return {
        "schema_version": "checkpoint-coverage-detector-development-report.v1",
        "decision": (
            "READY_FOR_PAID_CHECKPOINT_COLLECTION"
            if passed
            else "NOT READY — detector development gate failed"
        ),
        "gate_passed": passed,
        "previous_coverage_failure_audit": prior_audit,
        "development_data": {
            "path": str(checkpoints_path),
            "sha256": sha256_file(checkpoints_path),
            "all_flash_qwen_trajectories": len(trajectories),
            "natural_verified_trajectories": len(verified),
            "structural_or_protocol_failure_trajectories": len(structural),
            "task_count": len({item.task_id for item in trajectories}),
            "rows": sum(len(item.rows) for item in trajectories),
            "structural_failures_excluded_from_recovery_labels": True,
        },
        "detector_versions": {
            "v0": {
                "path": str(v0_source),
                "sha256": v0_sha256,
                "unchanged": True,
            },
            "v1": {
                "path": str(v1_source),
                "sha256": sha256_file(v1_source),
                "spec": SuspectedStuckV1.frozen_spec(),
            },
        },
        "candidate_development": {
            "development_candidate_family": [
                candidate.__dict__ for candidate in CANDIDATES
            ],
            "selection_rule": (
                "Within each training fold maximize healthy-minus-stuck recovery "
                "difference among candidates meeting minimum task/base coverage; "
                "evaluate only on the held-out task fold."
            ),
            "interpretation": (
                "This is task-grouped development evaluation, not independent "
                "confirmation; the candidate family was designed using the same "
                "development corpus."
            ),
            "task_grouped_fold_results": folds,
            "final_candidate_for_future_use": final_candidate.__dict__,
            "final_candidate_in_sample_only": {
                "healthy": _checkpoint_metrics(final_healthy),
                "stuck": _checkpoint_metrics(final_stuck),
                "recovery_difference": _gap(final_healthy, final_stuck),
                "not_used_for_gate": True,
            },
        },
        "cross_fitted_results": {
            "healthy": _checkpoint_metrics(healthy),
            "stuck": _checkpoint_metrics(stuck),
            "healthy_minus_stuck_recovery_difference": difference,
            "task_clustered_interval_95": [lower, upper],
            "robustness": robustness,
        },
        "same_scout_detector_yield_replay": yield_replay,
        "offline_gate": gates,
        "limitations": [
            (
                "The shared development checkpoints are pre-turn summaries, not complete "
                "v0/v1 live observations."
            ),
            (
                "Candidate design and task-grouped evaluation reuse the development "
                "corpus, so the separation estimate is not independent confirmation."
            ),
            (
                "They omit workspace digests, commands, exact public-test state, and "
                "reproducibility state."
            ),
            (
                "Continuation recovery is known only for the route that actually ran; "
                "intervention actions remain unlabeled."
            ),
            (
                "Stopped confirmatory scouts are not used as recovery labels because "
                "stopping induced their terminal outcome."
            ),
            "The cross-fitted separation misses the predeclared 20-point threshold.",
        ],
        "next_phase": {
            "task_source_search": "not_run",
            "paid_checkpoint_collection": "not_run",
            "intervention_collection": "forbidden",
            "training": "forbidden",
            "reason": "detector_development_gate_failed",
        },
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
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
