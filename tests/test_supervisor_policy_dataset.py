from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from horizon_supervisor.training.build_supervisor_policy_dataset import (
    DEFAULT_ROUTES,
    build_development_policy_dataset,
)

ROOT = Path(__file__).resolve().parents[1]


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _write_task(
    tasks_dir: Path,
    task_name: str,
    instruction: str,
    *,
    difficulty: str = "medium",
    category: str = "software-engineering",
) -> None:
    task_dir = tasks_dir / task_name
    task_dir.mkdir(parents=True)
    (task_dir / "instruction.md").write_text(instruction)
    (task_dir / "task.toml").write_text(
        "[metadata]\n"
        f'difficulty = "{difficulty}"\n'
        f'category = "{category}"\n'
        'tags = ["example"]\n'
    )


def _outcome_row(
    task_name: str,
    route_id: str,
    result_path: Path,
    *,
    record_split: str = "development",
) -> dict:
    return {
        "schema_version": "matched-model-outcome.v1",
        "outcome_id": f"outcome-{task_name}-{route_id}",
        "matched_group_id": f"group-{task_name}",
        "initial_state": {"kind": "clean_task_start", "digest": "clean-digest"},
        "task": {
            "task_id": f"task-id-{task_name}",
            "source_task_name": task_name,
            "difficulty": "medium",
            "category": "software-engineering",
            "record_split": record_split,
        },
        "model": {
            "route_id": route_id,
            "endpoint": "example/model",
            "agent": "terminus-2",
            "candidate_features": {
                "schema_version": "candidate-model-features.v0",
                "context_window_tokens": 100_000,
                "input_usd_per_million_tokens": 1.0,
                "output_usd_per_million_tokens": 2.0,
                "cached_input_usd_per_million_tokens": 0.2,
            },
        },
        "outcome": {
            "completed": True,
            "reward": 1.0,
            "status": "verified",
            "exception_type": None,
            "estimated_list_cost_usd": 0.25,
            "allocated_provider_cost_usd": None,
            "duration_seconds": 12.0,
            "input_tokens": 30,
            "cache_tokens": 2,
            "output_tokens": 7,
            "model_calls": 2,
        },
        "provenance": {
            "result_path": str(result_path),
            "task_source_revision": "revision",
        },
    }


def _write_two_turn_trajectory(result_path: Path, instruction: str) -> None:
    trajectory_path = result_path.parent / "agent" / "trajectory.json"
    trajectory_path.parent.mkdir(parents=True)
    prompt = (
        "Harness wrapper\nTask Description:\n"
        f"{instruction}\n\nCurrent terminal state:\n"
        "Current Terminal Screen:\n$ initial visible state\n"
    )
    trajectory = {
        "schema_version": "ATIF-v1.7",
        "session_id": "trajectory-1",
        "agent": {"name": "terminus-2", "model_name": "example/model"},
        "steps": [
            {"step_id": 1, "source": "user", "message": prompt},
            {
                "step_id": 2,
                "source": "agent",
                "message": "AGENT MESSAGE MUST NOT ENTER INPUT",
                "reasoning_content": "PRIVATE FUTURE REASONING",
                "tool_calls": [{"tool_call_id": "one"}],
                "observation": {
                    "results": [{"content": "PREVIOUS VISIBLE OBSERVATION"}]
                },
                "metrics": {
                    "prompt_tokens": 10,
                    "cached_tokens": 1,
                    "completion_tokens": 3,
                },
            },
            {
                "step_id": 3,
                "source": "agent",
                "message": "SECOND AGENT MESSAGE MUST NOT ENTER INPUT",
                "reasoning_content": "SECOND PRIVATE FUTURE REASONING",
                "tool_calls": [],
                "observation": {
                    "results": [{"content": "CURRENT FUTURE OBSERVATION"}]
                },
                "metrics": {
                    "prompt_tokens": 20,
                    "cached_tokens": 1,
                    "completion_tokens": 4,
                },
            },
        ],
        "final_metrics": {
            "total_prompt_tokens": 30,
            "total_cached_tokens": 2,
            "total_completion_tokens": 7,
        },
    }
    trajectory_path.write_text(json.dumps(trajectory))


def test_authoritative_development_build_is_rectangular_and_leakage_safe(
    tmp_path: Path,
) -> None:
    task_route_path = tmp_path / "task-route.jsonl"
    checkpoint_path = tmp_path / "checkpoints.jsonl"
    summary_path = tmp_path / "summary.json"
    summary = build_development_policy_dataset(
        outcomes_path=(
            ROOT
            / "artifacts/official/gate8-proportional-30-task-checkpoint/"
            "matched-outcomes-140-v1.jsonl"
        ),
        task_dirs=(
            ROOT / "data/supervisor/terminal-bench-pro-wave-1/tasks",
            ROOT / "data/supervisor/terminal-bench-pro-wave-2/tasks",
        ),
        heldout_task_dir=ROOT / "data/supervisor/terminal-bench-pro-wave-3/tasks",
        task_route_output_path=task_route_path,
        checkpoint_output_path=checkpoint_path,
        summary_output_path=summary_path,
    )

    task_routes = _jsonl(task_route_path)
    checkpoints = _jsonl(checkpoint_path)
    assert len(task_routes) == 140
    assert len(checkpoints) == 1_154
    assert summary["clean_start"]["tasks"] == 35
    assert summary["clean_start"]["routes"] == 4
    assert summary["continuation"]["trajectories"] == 140
    assert summary["continuation"]["zero_turn_trajectories"] == 1
    assert summary["leakage_guards"][
        "development_heldout_prompt_overlap_count"
    ] == 0
    assert {row["record_split"] for row in task_routes + checkpoints} == {
        "development"
    }

    routes_by_group: defaultdict[str, set[str]] = defaultdict(set)
    leakage_by_task: defaultdict[str, set[str]] = defaultdict(set)
    for row in task_routes:
        routes_by_group[row["matched_group_id"]].add(row["candidate"]["route_id"])
        leakage_by_task[row["input"]["task_id"]].add(row["leakage_group"])
        assert [
            action["target_route_id"] for action in row["available_actions"]
        ] == list(DEFAULT_ROUTES)
    assert len(routes_by_group) == 35
    assert all(routes == set(DEFAULT_ROUTES) for routes in routes_by_group.values())
    assert all(len(groups) == 1 for groups in leakage_by_task.values())

    weight_by_trajectory: defaultdict[str, float] = defaultdict(float)
    for row in checkpoints:
        weight_by_trajectory[row["trajectory_id"]] += row["training"][
            "trajectory_weight"
        ]
        action_counts = Counter(
            action["action"] for action in row["available_actions"]
        )
        assert action_counts == {
            "continue_same": 1,
            "switch_model": 3,
            "restart_clean": 4,
            "stop": 1,
        }
        observed = [
            action for action in row["available_actions"] if action["observed"]
        ]
        unobserved = [
            action for action in row["available_actions"] if not action["observed"]
        ]
        assert len(observed) == 1
        assert observed[0]["action"] == "continue_same"
        assert observed[0]["outcome"] is not None
        assert all(action["outcome"] is None for action in unobserved)
        observation = json.dumps(row["observation"]).lower()
        assert "reasoning_content" not in observation
        assert "final_metrics" not in observation
        assert "verifier_result" not in observation
        assert "test-stdout.txt" not in observation
        assert set(row["observation"]) == {
            "turn_index",
            "current_route_id",
            "prior_agent_turn_count",
            "prior_tool_call_count",
            "prior_observation_chars",
            "cumulative_input_tokens",
            "cumulative_cache_tokens",
            "cumulative_output_tokens",
            "terminal_chars",
            "terminal_lines",
            "error_signal_count",
            "pass_signal_count",
            "test_signal_count",
            "shell_prompt_count",
            "terminal_tail",
            "terminal_tail_truncated",
        }
    assert len(weight_by_trajectory) == 139
    assert all(weight == pytest.approx(1.0) for weight in weight_by_trajectory.values())


def test_checkpoint_input_uses_only_state_visible_before_the_current_turn(
    tmp_path: Path,
) -> None:
    task_name = "development-task"
    instruction = "Repair the example safely."
    tasks_dir = tmp_path / "development-tasks"
    heldout_dir = tmp_path / "heldout-tasks"
    _write_task(tasks_dir, task_name, instruction)
    _write_task(heldout_dir, "heldout-task", "Solve a different held-out task.")
    result_path = tmp_path / "trial" / "result.json"
    result_path.parent.mkdir(parents=True)
    _write_two_turn_trajectory(result_path, instruction)
    outcomes_path = tmp_path / "outcomes.jsonl"
    outcomes_path.write_text(
        json.dumps(_outcome_row(task_name, "route/one", result_path)) + "\n"
    )
    task_route_path = tmp_path / "task-route.jsonl"
    checkpoint_path = tmp_path / "checkpoints.jsonl"

    build_development_policy_dataset(
        outcomes_path=outcomes_path,
        task_dirs=(tasks_dir,),
        heldout_task_dir=heldout_dir,
        task_route_output_path=task_route_path,
        checkpoint_output_path=checkpoint_path,
        summary_output_path=tmp_path / "summary.json",
        expected_routes=("route/one",),
        expected_task_count=1,
        expected_heldout_task_count=1,
    )

    checkpoints = _jsonl(checkpoint_path)
    assert len(checkpoints) == 2
    assert checkpoints[0]["observation"]["terminal_tail"] == "$ initial visible state"
    assert (
        checkpoints[1]["observation"]["terminal_tail"]
        == "PREVIOUS VISIBLE OBSERVATION"
    )
    serialized_inputs = json.dumps([row["observation"] for row in checkpoints])
    assert "AGENT MESSAGE MUST NOT ENTER INPUT" not in serialized_inputs
    assert "PRIVATE FUTURE REASONING" not in serialized_inputs
    assert "CURRENT FUTURE OBSERVATION" not in serialized_inputs
    assert [row["training"]["trajectory_weight"] for row in checkpoints] == [
        0.5,
        0.5,
    ]


def test_builder_rejects_non_development_outcomes(tmp_path: Path) -> None:
    outcomes_path = tmp_path / "outcomes.jsonl"
    outcomes_path.write_text(
        json.dumps(
            _outcome_row(
                "task", "route/one", tmp_path / "trial" / "result.json",
                record_split="held_out",
            )
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="development rows only"):
        build_development_policy_dataset(
            outcomes_path=outcomes_path,
            task_dirs=(tmp_path / "tasks",),
            heldout_task_dir=tmp_path / "heldout",
            task_route_output_path=tmp_path / "task-route.jsonl",
            checkpoint_output_path=tmp_path / "checkpoints.jsonl",
            summary_output_path=tmp_path / "summary.json",
            expected_routes=("route/one",),
            expected_task_count=1,
            expected_heldout_task_count=1,
        )


def test_builder_rejects_normalized_wave3_prompt_overlap(tmp_path: Path) -> None:
    task_name = "development-task"
    tasks_dir = tmp_path / "development-tasks"
    heldout_dir = tmp_path / "heldout-tasks"
    _write_task(tasks_dir, task_name, "Repair   the example.\n")
    _write_task(heldout_dir, "heldout-task", "repair the EXAMPLE.")
    outcomes_path = tmp_path / "outcomes.jsonl"
    outcomes_path.write_text(
        json.dumps(
            _outcome_row(task_name, "route/one", tmp_path / "trial" / "result.json")
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="overlaps Wave 3"):
        build_development_policy_dataset(
            outcomes_path=outcomes_path,
            task_dirs=(tasks_dir,),
            heldout_task_dir=heldout_dir,
            task_route_output_path=tmp_path / "task-route.jsonl",
            checkpoint_output_path=tmp_path / "checkpoints.jsonl",
            summary_output_path=tmp_path / "summary.json",
            expected_routes=("route/one",),
            expected_task_count=1,
            expected_heldout_task_count=1,
        )
