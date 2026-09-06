from horizon_supervisor.training.freeze_continuation_calibration_v4 import (
    CATEGORY_QUOTAS,
    build_task_pool,
    static_checkpoint_compatibility,
)


def _config(*, difficulty: str = "hard", category: str = "debugging", tags=()) -> str:
    tags_toml = ", ".join(repr(tag) for tag in tags)
    return f"""
version = "1.0"
[metadata]
difficulty = "{difficulty}"
category = "{category}"
tags = [{tags_toml}]
[environment]
cpus = 1
memory_mb = 2048
gpus = 0
"""


def test_v4_static_filter_is_public_and_checkpoint_focused() -> None:
    assert static_checkpoint_compatibility(_config()) == (True, [])

    compatible, reasons = static_checkpoint_compatibility(
        _config(category="system-administration", tags=("qemu",))
    )
    assert compatible is False
    assert reasons == [
        "category_not_targeted",
        "persistent_or_distributed_runtime_tag",
    ]


def test_v4_pool_contains_three_frozen_tranches() -> None:
    fresh = {
        "tasks": [
            {
                "task_id": f"task-{index:02d}",
                "category": "debugging",
                "difficulty": "hard",
                "instruction_sha256": f"{index + 1:064x}",
                "task_root": f"private/tasks/task-{index:02d}",
                "task_tree_sha256": f"{index + 101:064x}",
                "outcome_blind_rank": f"{index + 201:064x}",
            }
            for index in range(sum(CATEGORY_QUOTAS.values()))
        ]
    }

    tasks = build_task_pool(fresh)

    assert len(tasks) == 24
    assert [task["position"] for task in tasks] == list(range(1, 25))
    assert [task["tranche"] for task in tasks] == [1] * 8 + [2] * 8 + [3] * 8
    assert all(task["difficulty"] == "hard" for task in tasks)
    assert all(task["prior_terminal_outcome_count"] == 0 for task in tasks)
