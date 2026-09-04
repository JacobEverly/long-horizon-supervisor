from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from horizon_supervisor.supervisor_data.pivot_checkpoints import (
    classify_action,
    split_conversation,
    split_prompt,
    transform_pivot_row,
)
from horizon_supervisor.supervisor_data.terminal_catalog import _file_labels, _split

ROOT = Path(__file__).resolve().parents[1]


def test_supervisor_registry_pins_sources_and_quarantines_unclear_licenses() -> None:
    registry = json.loads(
        (ROOT / "data/supervisor/source-registry-v0.json").read_text(encoding="utf-8")
    )
    source_ids = [source["source_id"] for source in registry["sources"]]
    assert len(source_ids) == len(set(source_ids))
    for source in registry["sources"]:
        assert source["license"]
        assert source["intended_use"]
        assert source["allowed_fields"]
        if source["kind"] == "huggingface":
            assert re.fullmatch(r"[0-9a-f]{40}", source["revision"])
        if source["license"] in {"not-declared", "clarification-required"}:
            assert source["quality_tier"] == "Q"


def test_skill_file_labels_are_normalized_to_benchmark_categories() -> None:
    assert _file_labels(
        "synthetic_tasks/skill_based/medium/software_engineering/data_filtered.parquet"
    ) == ("medium", "software-engineering")


def test_task_split_is_deterministic_and_group_safe() -> None:
    assert _split("medium|debugging|task-1") == _split("medium|debugging|task-1")
    assert _split("medium|debugging|task-1") in {
        "train",
        "validation",
        "internal_test",
        "sealed_test",
    }


def test_terminal_catalog_is_metadata_only_and_matches_pinned_counts() -> None:
    catalog_path = ROOT / "data/supervisor/terminal-corpus-catalog-v0.jsonl"
    summary_path = ROOT / "data/supervisor/terminal-corpus-catalog-v0-summary.json"
    rows = [json.loads(line) for line in catalog_path.read_text().splitlines()]
    summary = json.loads(summary_path.read_text())

    assert len(rows) == 356_199
    assert summary["trajectory_count"] == 366_154
    assert summary["unique_task_count"] == 356_199
    assert summary["benchmark_coverage"]["directly_covered_benchmark_tasks"] == 25
    assert summary["benchmark_coverage"]["direct_coverage_rate"] == 0.833333
    assert all("conversations" not in row for row in rows)
    assert all("solution" not in row for row in rows)
    assert sum(Counter(row["recommended_split"] for row in rows).values()) == 356_199


def test_companion_catalog_fills_exact_terminal_benchmark_category_gaps() -> None:
    rows = [
        json.loads(line)
        for line in (ROOT / "data/supervisor/companion-task-catalog-v0.jsonl")
        .read_text()
        .splitlines()
    ]
    summary = json.loads(
        (ROOT / "data/supervisor/companion-task-catalog-v0-summary.json").read_text()
    )
    assert len(rows) == 2_406
    assert summary["category_counts"] == {
        "games": 78,
        "machine-learning": 817,
        "optimization": 256,
        "personal-assistant": 1_255,
    }
    assert all("instruction" not in row for row in rows)
    assert all("solution" not in row for row in rows)
    assert all("ground_truth" not in row for row in rows)


def test_aligned_panel_exactly_matches_frozen_benchmark_proportions() -> None:
    panel = [
        json.loads(line)
        for line in (ROOT / "data/supervisor/aligned-task-panel-v0.jsonl")
        .read_text()
        .splitlines()
    ]
    summary = json.loads(
        (ROOT / "data/supervisor/aligned-task-panel-v0-summary.json").read_text()
    )
    benchmark = json.loads(
        (ROOT / "benchmarks/terminal-bench-2.1-gate7.json").read_text()
    )
    benchmark_counts = Counter(task["category"] for task in benchmark["tasks"])
    panel_counts = Counter(row["category"] for row in panel)
    assert len(panel) == 1_950
    assert panel_counts == {
        category: count * 65 for category, count in benchmark_counts.items()
    }
    assert all(row["recommended_split"] == "train" for row in panel)
    assert len({row["task_id"] for row in panel}) == len(panel)
    assert all(check["exact_match"] for check in summary["distribution_check"].values())


def test_pivot_parser_separates_online_state_from_future_information() -> None:
    prompt = (
        "wrapper\nTask Description:\nFix the package.\n\nCurrent terminal state:\n"
        "Current Terminal Screen:\n$ pytest\n1 failed\n"
    )
    task, state = split_prompt(prompt)
    assert task == "Fix the package."
    assert state == "$ pytest\n1 failed"

    task, state, observed = split_conversation(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "previous action"},
            {"role": "user", "content": "New Terminal Output:\n2 passed"},
        ]
    )
    assert task == "Fix the package."
    assert state == "New Terminal Output:\n2 passed"
    assert observed["prior_assistant_turn_count"] == 1

    source = {
        "source_id": "nemotron-terminal-pivot-v1",
        "dataset_id": "nvidia/Nemotron-RL-Agentic-Terminal-Pivot-v1",
        "revision": "df75a0134ab603d6926f5b6efb9eacd3603b2049",
    }
    row = {
        "uuid": "checkpoint-1",
        "task_name": "task-1",
        "responses_create_params": {"input": [{"role": "user", "content": prompt}]},
        "expected_answer": json.dumps(
            {
                "analysis": "private reference reasoning",
                "plan": "private future plan",
                "commands": [{"keystrokes": "pytest -q\n", "duration": 1.0}],
                "task_complete": False,
            }
        ),
        "agent_ref": "terminus-2",
        "metadata": {
            "harness": "terminus_2",
            "teacher_model": "zai-org/GLM-5.1",
            "source_trajectory_uid": "trajectory-1",
            "pivot_agent_turn_index": 3,
            "total_source_agent_turns": 20,
        },
    }
    task_row, checkpoint = transform_pivot_row(row, source)
    assert task_row["task_description"] == "Fix the package."
    assert checkpoint["target"]["primary_action"] == "validate"
    assert checkpoint["input"]["turn_index"] == 3
    assert checkpoint["leakage_group"] == task_row["leakage_group"]
    assert "total_source_agent_turns" not in checkpoint["input"]
    assert checkpoint["audit_only"]["total_source_agent_turns"] == 20
    serialized = json.dumps(checkpoint)
    assert "pytest -q" not in serialized
    assert "private reference" not in serialized


def test_action_classification_handles_mixed_batches_and_completion() -> None:
    assert classify_action([], True) == ("finish", [])
    primary, classes = classify_action(
        [{"keystrokes": "sed -i 's/a/b/' app.py\npytest -q\n"}], False
    )
    assert primary == "mixed"
    assert classes == ["edit", "validate"]
