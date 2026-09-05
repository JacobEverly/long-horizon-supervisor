import json

from horizon_supervisor.training.freeze_continuation_calibration_v2 import (
    EXPOSED_V1_TASK_ID,
    V1_MANIFEST,
    build_task_pool,
)


def test_v2_pool_excludes_exposed_task_and_adds_fresh_replacement() -> None:
    manifest = json.loads(V1_MANIFEST.read_text(encoding="utf-8"))
    replacement = {
        "task_id": "fresh-replacement",
        "category": "data-processing",
        "difficulty": "medium",
        "instruction_sha256": "i" * 64,
        "task_root": "private/tasks/fresh-replacement",
        "task_tree_sha256": "t" * 64,
        "outcome_blind_rank": "r" * 64,
    }

    tasks = build_task_pool(manifest, replacement)

    assert len(tasks) == 16
    assert tasks[0]["task_id"] == "fresh-replacement"
    assert EXPOSED_V1_TASK_ID not in {task["task_id"] for task in tasks}
    assert [task["position"] for task in tasks] == list(range(1, 17))
    assert [task["tranche"] for task in tasks] == [1] * 8 + [2] * 8
    assert all(task["prior_v1_terminal_outcome_count"] == 0 for task in tasks)
