from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_terminal_bench_pro_wave_one_is_materialized_for_harbor() -> None:
    root = ROOT / "data/supervisor/terminal-bench-pro-wave-1"
    summary = json.loads((root / "summary.json").read_text())
    assert summary["wave"] == 1
    assert summary["task_count"] == 18
    assert summary["all_execution_locks_verified"] is True
    assert summary["all_archive_paths_safe"] is True
    task_dirs = [path for path in (root / "tasks").iterdir() if path.is_dir()]
    assert len(task_dirs) == 18
    assert all((path / "instruction.md").is_file() for path in task_dirs)
    assert all((path / "task.toml").is_file() for path in task_dirs)


def test_terminal_bench_pro_wave_two_is_materialized_for_harbor() -> None:
    root = ROOT / "data/supervisor/terminal-bench-pro-wave-2"
    summary = json.loads((root / "summary.json").read_text())
    assert summary["wave"] == 2
    assert summary["task_count"] == 18
    assert summary["all_execution_locks_verified"] is True
    assert summary["all_archive_paths_safe"] is True
    assert len([path for path in (root / "tasks").iterdir() if path.is_dir()]) == 18


def test_terminal_bench_pro_wave_three_is_materialized_for_harbor() -> None:
    root = ROOT / "data/supervisor/terminal-bench-pro-wave-3"
    summary = json.loads((root / "summary.json").read_text())
    assert summary["wave"] == 3
    assert summary["task_count"] == 18
    assert summary["all_execution_locks_verified"] is True
    assert summary["all_archive_paths_safe"] is True
    assert len([path for path in (root / "tasks").iterdir() if path.is_dir()]) == 18


def test_terminal_bench_pro_wave_four_is_materialized_for_harbor() -> None:
    root = ROOT / "data/supervisor/terminal-bench-pro-wave-4"
    summary = json.loads((root / "summary.json").read_text())
    assert summary["wave"] == 4
    assert summary["task_count"] == 18
    assert summary["all_execution_locks_verified"] is True
    assert summary["all_archive_paths_safe"] is True
    assert len([path for path in (root / "tasks").iterdir() if path.is_dir()]) == 18
