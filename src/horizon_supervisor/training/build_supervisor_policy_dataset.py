from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from horizon_supervisor.benchmark.matched_outcomes import LEARNING_VALID_STATUSES
from horizon_supervisor.supervisor_data.pivot_checkpoints import (
    ERROR_PATTERN,
    PASS_PATTERN,
    PROMPT_PATTERN,
    TERMINAL_TAIL_CHARS,
    TEST_PATTERN,
    split_prompt,
    task_leakage_sha256,
)

DEFAULT_OUTCOMES = Path(
    "artifacts/official/gate8-proportional-30-task-checkpoint/"
    "matched-outcomes-140-v1.jsonl"
)
DEFAULT_TASK_DIRS = (
    Path("data/supervisor/terminal-bench-pro-wave-1/tasks"),
    Path("data/supervisor/terminal-bench-pro-wave-2/tasks"),
)
DEFAULT_HELDOUT_TASK_DIR = Path("data/supervisor/terminal-bench-pro-wave-3/tasks")
DEFAULT_TASK_ROUTE_OUTPUT = Path(
    "data/supervisor/gate8-development-task-route-v0.jsonl"
)
DEFAULT_CHECKPOINT_OUTPUT = Path(
    "data/supervisor/gate8-development-checkpoints-v0.jsonl"
)
DEFAULT_SUMMARY_OUTPUT = Path(
    "data/supervisor/gate8-development-policy-dataset-v0-summary.json"
)
DEFAULT_ROUTES = (
    "gate7/fixed-flash",
    "gate7/fixed-glm",
    "gate7/fixed-kimi",
    "gate7/fixed-qwen",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on {path}:{line_number}") from exc
    if not rows:
        raise ValueError(f"JSONL input is empty: {path}")
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _task_root(task_name: str, task_dirs: tuple[Path, ...]) -> Path:
    matches = [task_dir / task_name for task_dir in task_dirs if (task_dir / task_name).is_dir()]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one public task directory for {task_name!r}, found {len(matches)}"
        )
    return matches[0]


def _public_task(task_name: str, task_dirs: tuple[Path, ...]) -> dict[str, Any]:
    root = _task_root(task_name, task_dirs)
    instruction_path = root / "instruction.md"
    config_path = root / "task.toml"
    if not instruction_path.is_file() or not config_path.is_file():
        raise ValueError(f"public task files are missing for {task_name!r}")
    instruction = instruction_path.read_text(encoding="utf-8")
    metadata = tomllib.loads(config_path.read_text(encoding="utf-8")).get(
        "metadata", {}
    )
    return {
        "source_task_name": task_name,
        "instruction": instruction,
        "instruction_sha256": _sha256_bytes(instruction.encode()),
        "leakage_group": task_leakage_sha256(instruction),
        "difficulty": str(metadata.get("difficulty", "unknown")),
        "category": str(metadata.get("category", "unknown")),
        "tags": sorted(str(tag) for tag in metadata.get("tags", [])),
        "source_root": str(root),
    }


def _heldout_prompt_groups(
    heldout_task_dir: Path,
    *,
    expected_task_count: int | None,
) -> dict[str, str]:
    instruction_paths = sorted(heldout_task_dir.glob("*/instruction.md"))
    if not instruction_paths:
        raise ValueError(f"held-out task directory has no instructions: {heldout_task_dir}")
    if expected_task_count is not None and len(instruction_paths) != expected_task_count:
        raise ValueError(
            f"expected {expected_task_count} held-out tasks, found {len(instruction_paths)}"
        )
    groups: dict[str, str] = {}
    for path in instruction_paths:
        group = task_leakage_sha256(path.read_text(encoding="utf-8"))
        if group in groups:
            raise ValueError(
                "duplicate normalized prompt in held-out tasks: "
                f"{groups[group]!r} and {path.parent.name!r}"
            )
        groups[group] = path.parent.name
    return groups


def _rectangular_development_panel(
    rows: list[dict[str, Any]],
    *,
    expected_routes: tuple[str, ...],
    expected_task_count: int | None,
) -> tuple[dict[str, dict[str, dict[str, Any]]], list[str]]:
    if len(expected_routes) != len(set(expected_routes)) or not expected_routes:
        raise ValueError("expected routes must be non-empty and unique")
    expected_route_set = set(expected_routes)
    by_task: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    task_identity: dict[str, tuple[str, str, str]] = {}

    for row in rows:
        if row.get("schema_version") != "matched-model-outcome.v1":
            raise ValueError("unsupported matched-outcome schema")
        if row.get("task", {}).get("record_split") != "development":
            raise ValueError("training data must contain development rows only")
        if row.get("outcome", {}).get("status") not in LEARNING_VALID_STATUSES:
            raise ValueError("training data contains a non-learning-valid outcome")
        task = row["task"]
        task_name = str(task["source_task_name"])
        route_id = str(row["model"]["route_id"])
        if route_id not in expected_route_set:
            raise ValueError(f"unexpected route in development data: {route_id}")
        if route_id in by_task[task_name]:
            raise ValueError(f"duplicate task-route pair: {task_name}|{route_id}")
        identity = (
            str(task["task_id"]),
            str(row["matched_group_id"]),
            str(row["initial_state"]["digest"]),
        )
        previous_identity = task_identity.setdefault(task_name, identity)
        if identity != previous_identity:
            raise ValueError(f"task routes do not share one matched clean start: {task_name}")
        if row["initial_state"].get("kind") != "clean_task_start":
            raise ValueError(f"task-route row is not a clean start: {task_name}|{route_id}")
        by_task[task_name][route_id] = row

    if expected_task_count is not None and len(by_task) != expected_task_count:
        raise ValueError(f"expected {expected_task_count} tasks, found {len(by_task)}")
    for task_name, route_rows in by_task.items():
        if set(route_rows) != expected_route_set:
            missing = sorted(expected_route_set - set(route_rows))
            extra = sorted(set(route_rows) - expected_route_set)
            raise ValueError(
                f"non-rectangular task {task_name!r}; missing={missing}, extra={extra}"
            )
    if len(rows) != len(by_task) * len(expected_routes):
        raise ValueError("development outcomes contain missing or extra task-route rows")
    return dict(by_task), sorted(expected_routes)


def _comparable_cost(row: dict[str, Any]) -> tuple[float, str]:
    outcome = row["outcome"]
    if outcome.get("estimated_list_cost_usd") is not None:
        return float(outcome["estimated_list_cost_usd"]), "cache-aware-list-price"
    if outcome.get("allocated_provider_cost_usd") is not None:
        return float(outcome["allocated_provider_cost_usd"]), "allocated-provider-spend"
    raise ValueError("outcome has no comparable cost")


def _task_route_target(row: dict[str, Any]) -> dict[str, Any]:
    outcome = row["outcome"]
    cost, cost_basis = _comparable_cost(row)
    return {
        "completed": bool(outcome["completed"]),
        "reward": outcome.get("reward"),
        "status": str(outcome["status"]),
        "exception_type": outcome.get("exception_type"),
        "cost_usd": cost,
        "cost_basis": cost_basis,
        "duration_seconds": float(outcome["duration_seconds"]),
        "input_tokens": int(outcome["input_tokens"]),
        "cache_tokens": int(outcome["cache_tokens"]),
        "output_tokens": int(outcome["output_tokens"]),
        "model_calls": int(outcome["model_calls"]),
    }


def _task_route_row(
    row: dict[str, Any],
    public_task: dict[str, Any],
    routes: list[str],
) -> dict[str, Any]:
    task = row["task"]
    if str(task["difficulty"]) != public_task["difficulty"]:
        raise ValueError(f"difficulty mismatch for {task['source_task_name']}")
    if str(task["category"]) != public_task["category"]:
        raise ValueError(f"category mismatch for {task['source_task_name']}")
    route_id = str(row["model"]["route_id"])
    return {
        "schema_version": "supervisor-task-route.v0",
        "example_id": _sha256_bytes(
            f"{row['matched_group_id']}|{route_id}".encode()
        ),
        "record_split": "development",
        "leakage_group": public_task["leakage_group"],
        "matched_group_id": row["matched_group_id"],
        "initial_state": {
            "kind": "clean_task_start",
            "digest": row["initial_state"]["digest"],
        },
        "input": {
            "task_id": task["task_id"],
            "source_task_name": task["source_task_name"],
            "instruction": public_task["instruction"],
            "instruction_sha256": public_task["instruction_sha256"],
            "difficulty": public_task["difficulty"],
            "category": public_task["category"],
            "tags": public_task["tags"],
        },
        "available_actions": [
            {"action": "start_model", "target_route_id": candidate}
            for candidate in routes
        ],
        "logged_action": {
            "action": "start_model",
            "target_route_id": route_id,
        },
        "candidate": {
            "route_id": route_id,
            "endpoint": row["model"]["endpoint"],
            "agent": row["model"]["agent"],
            "features": row["model"]["candidate_features"],
        },
        "target": _task_route_target(row),
        "provenance": {
            "source_outcome_id": row["outcome_id"],
            "source_result_path": row["provenance"]["result_path"],
            "task_source_revision": row["provenance"]["task_source_revision"],
        },
    }


def _observation_text(step: dict[str, Any]) -> str:
    observation = step.get("observation")
    if observation is None:
        return ""
    if isinstance(observation, str):
        return observation
    if isinstance(observation, dict):
        results = observation.get("results", [])
        if isinstance(results, list):
            return "\n".join(
                str(result.get("content", ""))
                for result in results
                if isinstance(result, dict) and result.get("content") is not None
            )
    raise ValueError("unsupported ATIF observation structure")


def _online_observation(
    terminal_state: str,
    *,
    turn_index: int,
    prior_tool_call_count: int,
    prior_observation_chars: int,
    cumulative_input_tokens: int,
    cumulative_cache_tokens: int,
    cumulative_output_tokens: int,
    route_id: str,
) -> dict[str, Any]:
    return {
        "turn_index": turn_index,
        "current_route_id": route_id,
        "prior_agent_turn_count": turn_index - 1,
        "prior_tool_call_count": prior_tool_call_count,
        "prior_observation_chars": prior_observation_chars,
        "cumulative_input_tokens": cumulative_input_tokens,
        "cumulative_cache_tokens": cumulative_cache_tokens,
        "cumulative_output_tokens": cumulative_output_tokens,
        "terminal_chars": len(terminal_state),
        "terminal_lines": terminal_state.count("\n") + bool(terminal_state),
        "error_signal_count": len(ERROR_PATTERN.findall(terminal_state)),
        "pass_signal_count": len(PASS_PATTERN.findall(terminal_state)),
        "test_signal_count": len(TEST_PATTERN.findall(terminal_state)),
        "shell_prompt_count": len(PROMPT_PATTERN.findall(terminal_state)),
        "terminal_tail": terminal_state[-TERMINAL_TAIL_CHARS:],
        "terminal_tail_truncated": len(terminal_state) > TERMINAL_TAIL_CHARS,
    }


def _unobserved_action(action: str, target_route_id: str | None) -> dict[str, Any]:
    return {
        "action": action,
        "target_route_id": target_route_id,
        "observed": False,
        "outcome": None,
    }


def _checkpoint_actions(
    row: dict[str, Any],
    routes: list[str],
    *,
    remaining_tokens: dict[str, int],
) -> list[dict[str, Any]]:
    route_id = str(row["model"]["route_id"])
    continue_outcome = _task_route_target(row) | remaining_tokens
    actions = [
        {
            "action": "continue_same",
            "target_route_id": route_id,
            "observed": True,
            "outcome": continue_outcome,
        }
    ]
    actions.extend(
        _unobserved_action("switch_model", candidate)
        for candidate in routes
        if candidate != route_id
    )
    actions.extend(
        _unobserved_action("restart_clean", candidate) for candidate in routes
    )
    actions.append(_unobserved_action("stop", None))
    return actions


def _trajectory_checkpoints(
    row: dict[str, Any],
    public_task: dict[str, Any],
    routes: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result_path = Path(row["provenance"]["result_path"])
    trajectory_path = result_path.parent / "agent" / "trajectory.json"
    if not trajectory_path.is_file():
        raise ValueError(f"ATIF trajectory is missing: {trajectory_path}")
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    if trajectory.get("schema_version") != "ATIF-v1.7":
        raise ValueError(f"unsupported trajectory schema: {trajectory_path}")
    user_steps = [step for step in trajectory["steps"] if step.get("source") == "user"]
    if not user_steps:
        raise ValueError(f"trajectory has no initial user state: {trajectory_path}")
    task_description, initial_terminal = split_prompt(str(user_steps[0]["message"]))
    if task_leakage_sha256(task_description) != public_task["leakage_group"]:
        raise ValueError(
            f"trajectory task prompt does not match public instruction: {trajectory_path}"
        )

    agent_steps = [
        step for step in trajectory["steps"] if step.get("source") == "agent"
    ]
    total = trajectory.get("final_metrics", {})
    total_input = int(total.get("total_prompt_tokens", 0))
    total_cache = int(total.get("total_cached_tokens", 0))
    total_output = int(total.get("total_completion_tokens", 0))
    outcome = row["outcome"]
    if (total_input, total_cache, total_output) != (
        int(outcome["input_tokens"]),
        int(outcome["cache_tokens"]),
        int(outcome["output_tokens"]),
    ):
        raise ValueError(f"ATIF and matched-outcome token totals differ: {trajectory_path}")

    terminal_state = initial_terminal
    cumulative_input = 0
    cumulative_cache = 0
    cumulative_output = 0
    prior_tool_calls = 0
    prior_observation_chars = len(initial_terminal)
    checkpoints = []
    trajectory_weight = 1.0 / len(agent_steps) if agent_steps else 0.0

    for turn_index, step in enumerate(agent_steps, start=1):
        metrics = step.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError(f"agent step lacks metrics: {trajectory_path}")
        remaining_tokens = {
            "remaining_input_tokens": total_input - cumulative_input,
            "remaining_cache_tokens": total_cache - cumulative_cache,
            "remaining_output_tokens": total_output - cumulative_output,
        }
        if any(value < 0 for value in remaining_tokens.values()):
            raise ValueError(f"cumulative tokens exceed final totals: {trajectory_path}")
        route_id = str(row["model"]["route_id"])
        checkpoint_id = _sha256_bytes(
            f"{row['outcome_id']}|turn={turn_index}".encode()
        )
        checkpoints.append(
            {
                "schema_version": "supervisor-continuation-checkpoint.v0",
                "checkpoint_id": checkpoint_id,
                "record_split": "development",
                "leakage_group": public_task["leakage_group"],
                "matched_group_id": row["matched_group_id"],
                "task_id": row["task"]["task_id"],
                "source_task_name": row["task"]["source_task_name"],
                "trajectory_id": trajectory["session_id"],
                "source_outcome_id": row["outcome_id"],
                "observation": _online_observation(
                    terminal_state,
                    turn_index=turn_index,
                    prior_tool_call_count=prior_tool_calls,
                    prior_observation_chars=prior_observation_chars,
                    cumulative_input_tokens=cumulative_input,
                    cumulative_cache_tokens=cumulative_cache,
                    cumulative_output_tokens=cumulative_output,
                    route_id=route_id,
                ),
                "available_actions": _checkpoint_actions(
                    row,
                    routes,
                    remaining_tokens=remaining_tokens,
                ),
                "logged_action": {
                    "action": "continue_same",
                    "target_route_id": route_id,
                },
                "training": {
                    "trajectory_weight": trajectory_weight,
                    "label_scope": (
                        "Observed continuation on a fixed-model trajectory; switch, "
                        "restart, and stop outcomes are intentionally unlabeled."
                    ),
                },
                "provenance": {
                    "trajectory_path": str(trajectory_path),
                    "source_step_id": step["step_id"],
                },
            }
        )

        cumulative_input += int(metrics.get("prompt_tokens", 0))
        cumulative_cache += int(metrics.get("cached_tokens", 0))
        cumulative_output += int(metrics.get("completion_tokens", 0))
        prior_tool_calls += len(step.get("tool_calls") or [])
        new_observation = _observation_text(step)
        if new_observation:
            terminal_state = new_observation
            prior_observation_chars += len(new_observation)

    if (cumulative_input, cumulative_cache, cumulative_output) != (
        total_input,
        total_cache,
        total_output,
    ):
        raise ValueError(f"ATIF step metrics do not sum to final totals: {trajectory_path}")
    return checkpoints, {
        "trajectory_id": trajectory["session_id"],
        "turns": len(agent_steps),
        "zero_turn": not agent_steps,
    }


def build_development_policy_dataset(
    *,
    outcomes_path: Path = DEFAULT_OUTCOMES,
    task_dirs: tuple[Path, ...] = DEFAULT_TASK_DIRS,
    heldout_task_dir: Path = DEFAULT_HELDOUT_TASK_DIR,
    task_route_output_path: Path = DEFAULT_TASK_ROUTE_OUTPUT,
    checkpoint_output_path: Path = DEFAULT_CHECKPOINT_OUTPUT,
    summary_output_path: Path = DEFAULT_SUMMARY_OUTPUT,
    expected_routes: tuple[str, ...] = DEFAULT_ROUTES,
    expected_task_count: int | None = 35,
    expected_heldout_task_count: int | None = 18,
) -> dict[str, Any]:
    """Build leakage-safe development rows for the first supervisor policy.

    The clean-start table has counterfactual outcomes for every task-route pair.
    The turn table has only the actually observed fixed-model continuation label.
    It deliberately leaves switch, restart, and stop outcomes null.
    """

    source_rows = _read_jsonl(outcomes_path)
    by_task, routes = _rectangular_development_panel(
        source_rows,
        expected_routes=expected_routes,
        expected_task_count=expected_task_count,
    )
    public_tasks = {
        task_name: _public_task(task_name, task_dirs) for task_name in sorted(by_task)
    }
    heldout_groups = _heldout_prompt_groups(
        heldout_task_dir,
        expected_task_count=expected_heldout_task_count,
    )
    overlap = {
        public_task["leakage_group"]: task_name
        for task_name, public_task in public_tasks.items()
        if public_task["leakage_group"] in heldout_groups
    }
    if overlap:
        collisions = [
            f"{task_name}|{heldout_groups[group]}"
            for group, task_name in sorted(overlap.items())
        ]
        raise ValueError(
            "development prompt overlaps Wave 3 held-out prompt: " + ", ".join(collisions)
        )

    task_route_rows = []
    checkpoint_rows = []
    trajectory_summaries = []
    action_counts: Counter[str] = Counter()
    for task_name in sorted(by_task):
        public_task = public_tasks[task_name]
        for route_id in routes:
            row = by_task[task_name][route_id]
            task_route_rows.append(_task_route_row(row, public_task, routes))
            checkpoints, trajectory_summary = _trajectory_checkpoints(
                row, public_task, routes
            )
            checkpoint_rows.extend(checkpoints)
            trajectory_summaries.append(trajectory_summary)
            for checkpoint in checkpoints:
                action_counts.update(
                    action["action"] for action in checkpoint["available_actions"]
                )

    _write_jsonl(task_route_output_path, task_route_rows)
    _write_jsonl(checkpoint_output_path, checkpoint_rows)
    weight_by_trajectory: defaultdict[str, float] = defaultdict(float)
    for checkpoint in checkpoint_rows:
        weight_by_trajectory[checkpoint["trajectory_id"]] += checkpoint["training"][
            "trajectory_weight"
        ]
    non_unit_weights = {
        trajectory_id: weight
        for trajectory_id, weight in weight_by_trajectory.items()
        if abs(weight - 1.0) > 1e-9
    }
    if non_unit_weights:
        raise RuntimeError(f"trajectory weights do not sum to one: {non_unit_weights}")

    summary = {
        "schema_version": "supervisor-development-policy-dataset-summary.v0",
        "source": {
            "outcomes_path": str(outcomes_path),
            "outcomes_sha256": _sha256_file(outcomes_path),
            "task_dirs": [str(path) for path in task_dirs],
            "heldout_task_dir": str(heldout_task_dir),
        },
        "clean_start": {
            "rows": len(task_route_rows),
            "tasks": len(by_task),
            "routes": len(routes),
            "route_ids": routes,
            "all_pairs_present_once": len(task_route_rows) == len(by_task) * len(routes),
        },
        "continuation": {
            "rows": len(checkpoint_rows),
            "trajectories": len(trajectory_summaries),
            "trajectories_with_turns": len(weight_by_trajectory),
            "zero_turn_trajectories": sum(
                int(item["zero_turn"]) for item in trajectory_summaries
            ),
            "action_candidate_counts": dict(sorted(action_counts.items())),
            "observed_action": "continue_same",
            "unobserved_actions_have_null_outcomes": True,
            "all_nonempty_trajectory_weights_sum_to_one": True,
        },
        "leakage_guards": {
            "record_split": "development",
            "split_group": "normalized public task instruction SHA-256",
            "heldout_prompt_count": len(heldout_groups),
            "development_heldout_prompt_overlap_count": 0,
            "agent_messages_in_observation": False,
            "agent_reasoning_in_observation": False,
            "current_or_future_tool_observation_in_input": False,
            "verifier_artifacts_read": False,
            "unobserved_action_labels_fabricated": False,
        },
        "outputs": {
            "task_route_path": str(task_route_output_path),
            "task_route_sha256": _sha256_file(task_route_output_path),
            "checkpoint_path": str(checkpoint_output_path),
            "checkpoint_sha256": _sha256_file(checkpoint_output_path),
        },
    }
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the development-only initial supervisor policy dataset"
    )
    parser.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES)
    parser.add_argument("--task-dir", type=Path, action="append")
    parser.add_argument(
        "--heldout-task-dir", type=Path, default=DEFAULT_HELDOUT_TASK_DIR
    )
    parser.add_argument(
        "--task-route-output", type=Path, default=DEFAULT_TASK_ROUTE_OUTPUT
    )
    parser.add_argument(
        "--checkpoint-output", type=Path, default=DEFAULT_CHECKPOINT_OUTPUT
    )
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--expected-route", action="append")
    parser.add_argument("--expected-task-count", type=int, default=35)
    parser.add_argument("--expected-heldout-task-count", type=int, default=18)
    args = parser.parse_args()
    summary = build_development_policy_dataset(
        outcomes_path=args.outcomes,
        task_dirs=tuple(args.task_dir) if args.task_dir else DEFAULT_TASK_DIRS,
        heldout_task_dir=args.heldout_task_dir,
        task_route_output_path=args.task_route_output,
        checkpoint_output_path=args.checkpoint_output,
        summary_output_path=args.summary,
        expected_routes=(
            tuple(args.expected_route) if args.expected_route else DEFAULT_ROUTES
        ),
        expected_task_count=args.expected_task_count,
        expected_heldout_task_count=args.expected_heldout_task_count,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
