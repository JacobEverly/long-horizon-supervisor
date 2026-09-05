import json
from pathlib import Path

from horizon_supervisor.training.freeze_continuation_calibration_v1 import (
    ATTEMPTED_V0_TASK_ID,
    EXPECTED_V0_MANIFEST_SHA256,
    V0_MANIFEST,
    build_task_pool,
)


def test_v1_pool_reuses_only_unattempted_tasks_and_adds_fresh_replacement() -> None:
    assert EXPECTED_V0_MANIFEST_SHA256 == (
        "6b542b96882bd95611548fade83b55075a4c636e83f4be227af705bdba50b2d6"
    )
    manifest = json.loads(Path(V0_MANIFEST).read_text(encoding="utf-8"))
    replacement = {
        "task_id": "replacement-task",
        "category": "data-processing",
        "difficulty": "medium",
        "instruction_sha256": "i" * 64,
        "task_root": "private/tasks/replacement-task",
        "task_tree_sha256": "t" * 64,
        "outcome_blind_rank": "r" * 64,
    }

    tasks = build_task_pool(manifest, replacement)

    assert len(tasks) == 16
    assert tasks[0]["task_id"] == "replacement-task"
    assert ATTEMPTED_V0_TASK_ID not in {task["task_id"] for task in tasks}
    assert [task["position"] for task in tasks] == list(range(1, 17))
    assert all(task["prior_terminal_outcome_count"] == 0 for task in tasks)
