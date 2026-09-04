from __future__ import annotations

import json
from pathlib import Path

from horizon_supervisor.supervisor_data.swe_transfer_checkpoints import (
    _item_text,
    _primary_action,
)

ROOT = Path(__file__).resolve().parents[1]


def test_swe_item_text_extracts_only_observed_content() -> None:
    assert _item_text({"type": "function_call_output", "output": "2 passed"}) == "2 passed"
    assert _item_text(
        {"type": "message", "content": [{"type": "output_text", "text": "done"}]}
    ) == "done"


def test_swe_action_mapping_is_explicit() -> None:
    assert _primary_action("finish") == "finish"
    assert _primary_action("read_file") == "inspect"
    assert _primary_action("apply_patch") == "edit"
    assert _primary_action("execute_bash") == "execute"
    assert _primary_action("update_plan") == "plan"


def test_swe_transfer_dataset_is_sealed_and_matches_pinned_counts() -> None:
    summary = json.loads(
        (ROOT / "data/supervisor/swe-pivot-transfer-checkpoints-v0-summary.json").read_text()
    )
    assert summary["raw_row_count"] == 50_308
    assert summary["checkpoint_count"] == 50_308
    assert summary["unique_checkpoint_id_count"] == 50_308
    assert summary["source_task_id_count"] == 2_484
    assert summary["trajectory_count"] == 3_219
    assert summary["task_complete_counts"]["true"] == 1_042
    assert summary["parse_failure_count"] == 0
    assert "high-confidence" in summary["intended_use"].lower()


def test_swe_training_source_does_not_overlap_verified_holdout() -> None:
    audit = json.loads(
        (ROOT / "data/supervisor/swe-pivot-overlap-audit-v0.json").read_text()
    )
    assert audit["exact_instance_id_overlap_count"] == 0
    assert audit["normalized_problem_text_overlap_count"] == 0
    assert audit["training_allowed"] is True
