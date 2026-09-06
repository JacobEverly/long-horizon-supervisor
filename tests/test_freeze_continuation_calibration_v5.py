import json
from collections import Counter

from horizon_supervisor.training.freeze_continuation_calibration_v5 import (
    DIFFICULTY_QUOTAS,
    TRANCHE_DIFFICULTY_COUNTS,
    V3_MANIFEST,
    _normalized_instruction_sha256,
    _prior_task_ids,
    build_task_pool,
    static_checkpoint_compatibility,
)


def _config(*, difficulty: str = "hard", tags=()) -> str:
    tags_toml = ", ".join(repr(tag) for tag in tags)
    return f"""
version = "1.0"
[metadata]
difficulty = "{difficulty}"
category = "debugging"
tags = [{tags_toml}]
[environment]
cpus = 1
memory_mb = 2048
gpus = 0
"""


def test_v5_static_filter_rejects_service_state_and_non_app_workdirs() -> None:
    assert static_checkpoint_compatibility(
        "ordinary-task", _config(), "FROM python:3.13\nWORKDIR /app\n"
    ) == (True, [])

    compatible, reasons = static_checkpoint_compatibility(
        "launch-server-task",
        _config(tags=("networking",)),
        "FROM python:3.13\nWORKDIR /root\n",
    )
    assert compatible is False
    assert reasons == [
        "service_or_interactive_runtime_tag",
        "service_or_interactive_task_id",
        "final_workdir_not_app",
    ]


def test_v5_pool_freezes_balanced_difficulty_tranches() -> None:
    locks = []
    index = 0
    for tranche, mix in TRANCHE_DIFFICULTY_COUNTS.items():
        for difficulty, count in mix.items():
            for _ in range(count):
                index += 1
                locks.append(
                    {
                        "task_id": f"task-{index:02d}",
                        "category": "debugging",
                        "difficulty": difficulty,
                        "tranche": tranche,
                        "instruction_sha256": f"{index:064x}",
                        "task_root": f"private/tasks/task-{index:02d}",
                        "task_tree_sha256": f"{index + 100:064x}",
                        "outcome_blind_rank": f"{index + 200:064x}",
                    }
                )

    tasks = build_task_pool({"tasks": locks})

    assert len(tasks) == 24
    assert [task["position"] for task in tasks] == list(range(1, 25))
    assert [task["tranche"] for task in tasks] == [1] * 8 + [2] * 8 + [3] * 8
    assert Counter(task["difficulty"] for task in tasks) == Counter(
        DIFFICULTY_QUOTAS
    )
    assert all(task["prior_terminal_outcome_count"] == 0 for task in tasks)


def test_v5_prior_exposure_includes_every_v3_task() -> None:
    v3 = json.loads(V3_MANIFEST.read_text(encoding="utf-8"))
    v3_ids = {
        task["task_id"] for task in v3["task_selection"]["ordered_pool"]
    }
    prior = _prior_task_ids({"task_selection": {"ordered_pool": []}}, set())

    assert v3_ids
    assert v3_ids <= prior


def test_v5_normalizes_instruction_before_overlap_hashing() -> None:
    assert _normalized_instruction_sha256("Solve—THIS\n task!") == (
        _normalized_instruction_sha256("solve this task")
    )
