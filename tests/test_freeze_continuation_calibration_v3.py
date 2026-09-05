import json

from horizon_supervisor.training.freeze_continuation_calibration_v3 import (
    EXPANSION_SPECS,
    EXPOSED_TASK_IDS,
    REPLACEMENT_SPEC,
    V2_MANIFEST,
    build_task_pool,
)


def test_v3_pool_freezes_three_outcome_blind_tranches() -> None:
    manifest = json.loads(V2_MANIFEST.read_text(encoding="utf-8"))
    specs = (REPLACEMENT_SPEC, *EXPANSION_SPECS)
    fresh = {
        "tasks": [
            {
                "task_id": task_id,
                "role": "replacement" if index == 0 else "expansion",
                "category": category,
                "difficulty": difficulty,
                "instruction_sha256": f"{index + 1:064x}",
                "task_root": f"private/tasks/{task_id}",
                "task_tree_sha256": f"{index + 101:064x}",
                "outcome_blind_rank": f"{index + 201:064x}",
            }
            for index, (task_id, difficulty, category) in enumerate(specs)
        ]
    }

    tasks = build_task_pool(manifest, fresh)

    assert len(tasks) == 24
    assert tasks[0]["task_id"] == REPLACEMENT_SPEC[0]
    assert [task["task_id"] for task in tasks[16:]] == [
        spec[0] for spec in EXPANSION_SPECS
    ]
    assert [task["tranche"] for task in tasks] == [1] * 8 + [2] * 8 + [3] * 8
    assert not EXPOSED_TASK_IDS.intersection(task["task_id"] for task in tasks)
    assert len({task["task_id"] for task in tasks}) == 24
