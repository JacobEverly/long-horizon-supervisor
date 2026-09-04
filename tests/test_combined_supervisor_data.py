from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_combined_checkpoint_dataset_has_unique_group_safe_ids() -> None:
    summary = json.loads(
        (ROOT / "data/supervisor/supervisor-checkpoints-v1-summary.json").read_text()
    )
    assert summary["checkpoint_count"] == 145_914
    assert summary["unique_checkpoint_id_count"] == 145_914
    assert summary["task_record_count"] == 15_832
    assert summary["checkpoint_source_counts"] == {
        "nemotron-terminal-pivot-v1": 31_111,
        "openthoughts-agent-v1-sft": 114_803,
    }
    assert sum(summary["checkpoint_split_counts"].values()) == 145_914
    assert sum(summary["task_split_counts"].values()) == 15_832
    assert all(summary["leakage_guards"].values())


def test_three_source_dataset_keeps_only_high_confidence_swe_tasks() -> None:
    summary = json.loads(
        (ROOT / "data/supervisor/supervisor-checkpoints-v2-summary.json").read_text()
    )
    assert summary["checkpoint_count"] == 160_279
    assert summary["unique_checkpoint_id_count"] == 160_279
    assert summary["checkpoint_source_counts"] == {
        "nemotron-swe-pivot-v1": 14_365,
        "nemotron-terminal-pivot-v1": 31_111,
        "openthoughts-agent-v1-sft": 114_803,
    }
    assert summary["filtered_checkpoint_counts"] == {
        "nemotron-swe-pivot-v1": 35_943
    }
    assert summary["source_filters"]["nemotron-swe-pivot-v1"] == {
        "pass_rate_min": 0.625
    }
    assert sum(summary["checkpoint_split_counts"].values()) == 160_279
    assert sum(summary["task_split_counts"].values()) == summary["task_record_count"]
    assert summary["cross_source_overlap_group_count"] == 0
    assert all(summary["leakage_guards"].values())
