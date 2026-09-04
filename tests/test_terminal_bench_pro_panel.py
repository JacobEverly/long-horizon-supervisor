from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_gate8_six_task_pilot_is_frozen_and_difficulty_proportional() -> None:
    pilot = json.loads(
        (ROOT / "benchmarks/gate8-six-task-pilot-v0.json").read_text()
    )
    panel = [
        json.loads(line)
        for line in (
            ROOT / "data/supervisor/terminal-bench-pro-panel-v0.jsonl"
        ).read_text().splitlines()
        if json.loads(line)["wave"] == 1
    ]
    by_name = {row["source_task_name"]: row for row in panel}
    selected = pilot["tasks"]
    assert len(selected) == 6
    assert pilot["trial_count"] == 24
    assert Counter(row["difficulty"] for row in selected) == {"hard": 2, "medium": 4}
    assert len({row["category"] for row in selected}) == 6
    assert all(
        by_name[row["source_task_name"]]["task_id"] == row["task_id"]
        for row in selected
    )


def test_terminal_bench_pro_panel_is_wave_stratified_and_separate_from_final() -> None:
    summary = json.loads(
        (ROOT / "data/supervisor/terminal-bench-pro-panel-v0-summary.json").read_text()
    )
    assert summary["source"]["source_row_count"] == 200
    assert summary["record_count"] == 72
    assert summary["waves"] == 4
    assert summary["wave_counts"] == {"1": 18, "2": 18, "3": 18, "4": 18}
    assert summary["difficulty_counts"] == {"hard": 24, "medium": 48}
    assert summary["final_benchmark"]["covered_seats"] == 18
    assert summary["final_benchmark"]["coverage_rate"] == 0.6
    assert summary["exact_final_task_name_overlap_count"] == 0
