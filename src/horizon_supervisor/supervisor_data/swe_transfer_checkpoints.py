from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import HfApi, snapshot_download

from horizon_supervisor.supervisor_data.pivot_checkpoints import (
    _online_features,
    _split,
    task_leakage_sha256,
)

DEFAULT_REGISTRY = Path("data/supervisor/source-registry-v0.json")
DEFAULT_CHECKPOINTS = Path("data/supervisor/swe-pivot-transfer-checkpoints-v0.jsonl")
DEFAULT_TASKS = Path("data/supervisor/swe-pivot-transfer-tasks-v0.jsonl")
DEFAULT_SUMMARY = Path("data/supervisor/swe-pivot-transfer-checkpoints-v0-summary.json")
SOURCE_ID = "nemotron-swe-pivot-v1"
PARQUET_PATTERN = "default/partial-train/*.parquet"
COLUMNS = [
    "trajectory_id",
    "info",
    "responses_create_params",
    "expected_action",
    "metadata",
    "agent_ref",
    "pass_rate",
    "pass_rate_total",
    "pass_rate_passed",
]

INSPECT_ACTIONS = {"glob", "grep", "grep_files", "list_dir", "read", "read_file"}
EDIT_ACTIONS = {"apply_patch", "edit", "str_replace_editor", "write"}
EXECUTE_ACTIONS = {"bash", "execute_bash", "shell_command"}
PLAN_ACTIONS = {"task_tracker", "think", "todo_write", "update_plan"}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source(registry: dict[str, Any]) -> dict[str, Any]:
    for item in registry["sources"]:
        if item["source_id"] == SOURCE_ID:
            return item
    raise ValueError(f"missing source {SOURCE_ID!r}")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _parse_item(value: str | dict[str, Any]) -> dict[str, Any]:
    return json.loads(value) if isinstance(value, str) else value


def _item_text(item: dict[str, Any]) -> str:
    if isinstance(item.get("content"), str):
        return item["content"]
    if item.get("type") == "function_call_output":
        return str(item.get("output", ""))
    content = item.get("content")
    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("text"):
                texts.append(str(block["text"]))
        return "\n".join(texts)
    summary = item.get("summary")
    if isinstance(summary, list):
        return "\n".join(
            str(block.get("text", "")) for block in summary if isinstance(block, dict)
        )
    if item.get("type") == "function_call":
        return f"{item.get('name', '')} {item.get('arguments', '')}"
    return ""


def _task_description(items: list[dict[str, Any]]) -> str:
    for item in items:
        if item.get("role") == "user":
            text = _item_text(item).strip()
            if text:
                return text
    raise ValueError("no user task message found")


def _observed_terminal(items: list[dict[str, Any]]) -> str:
    for item in reversed(items):
        if item.get("type") == "function_call_output":
            return _item_text(item).rstrip()
    for item in reversed(items[1:]):
        text = _item_text(item).strip()
        if text:
            return text
    return _item_text(items[0]).rstrip()


def _observed_history(items: list[dict[str, Any]]) -> dict[str, int]:
    texts = [_item_text(item) for item in items]
    return {
        "history_message_count": max(0, len(items) - 1),
        "prior_assistant_turn_count": sum(
            item.get("role") == "assistant" or item.get("type") == "function_call"
            for item in items
        ),
        "prior_user_update_count": sum(
            item.get("type") == "function_call_output" for item in items
        ),
        "observed_history_chars": sum(len(text) for text in texts),
    }


def _primary_action(action_name: str) -> str:
    if action_name == "finish":
        return "finish"
    if action_name in INSPECT_ACTIONS:
        return "inspect"
    if action_name in EDIT_ACTIONS:
        return "edit"
    if action_name in EXECUTE_ACTIONS:
        return "execute"
    if action_name in PLAN_ACTIONS:
        return "plan"
    return "other"


def build_swe_transfer_checkpoints(
    registry_path: Path = DEFAULT_REGISTRY,
    checkpoints_path: Path = DEFAULT_CHECKPOINTS,
    tasks_path: Path = DEFAULT_TASKS,
    summary_path: Path = DEFAULT_SUMMARY,
    *,
    snapshot_path: Path | None = None,
) -> dict[str, Any]:
    source = _source(_load_json(registry_path))
    parquet_revision = source["parquet_revision"]
    parquet_sha = HfApi().dataset_info(
        source["dataset_id"], revision=parquet_revision
    ).sha
    if parquet_sha != parquet_revision:
        raise RuntimeError(
            f"parquet revision mismatch: expected {parquet_revision}, got {parquet_sha}"
        )
    if snapshot_path is None:
        snapshot_path = Path(
            snapshot_download(
                repo_id=source["dataset_id"],
                repo_type="dataset",
                revision=parquet_revision,
                allow_patterns=[PARQUET_PATTERN],
            )
        )
    parquet_files = sorted((snapshot_path / "default/partial-train").glob("*.parquet"))
    if len(parquet_files) != 10:
        raise RuntimeError(f"expected 10 Parquet shards, found {len(parquet_files)}")

    tasks: dict[str, dict[str, Any]] = {}
    checkpoint_ids: set[str] = set()
    trajectory_ids: set[int] = set()
    source_task_ids: set[str] = set()
    action_counts: Counter[str] = Counter()
    finish_counts: Counter[str] = Counter()
    confidence_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    parse_failures = 0
    row_count = 0

    checkpoints_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoints_path.open("w", encoding="utf-8") as checkpoint_handle:
        for parquet_file in parquet_files:
            parquet = pq.ParquetFile(parquet_file)
            for row_group in range(parquet.num_row_groups):
                rows = parquet.read_row_group(row_group, columns=COLUMNS).to_pylist()
                for row_index, row in enumerate(rows):
                    row_count += 1
                    try:
                        items = [
                            _parse_item(value)
                            for value in row["responses_create_params"]["input"]
                        ]
                        task_description = _task_description(items)
                        terminal_state = _observed_terminal(items)
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                        parse_failures += 1
                        continue
                    source_task_id = row["metadata"]["instance_id"]
                    source_task_ids.add(source_task_id)
                    task_description_sha256 = _sha256(task_description)
                    leakage_group = task_leakage_sha256(task_description)
                    task_id = _sha256(
                        f"{source['dataset_id']}@{source['revision']}|{source_task_id}|"
                        f"{task_description_sha256}"
                    )
                    task = {
                        "schema_version": "swe-pivot-transfer-task.v0",
                        "task_id": task_id,
                        "source_task_name": source_task_id,
                        "record_split": _split(leakage_group),
                        "leakage_group": leakage_group,
                        "task_description": task_description,
                        "task_description_sha256": task_description_sha256,
                        "task_chars": len(task_description),
                        "source": {
                            "source_id": source["source_id"],
                            "dataset_id": source["dataset_id"],
                            "revision": source["revision"],
                            "parquet_revision": parquet_revision,
                        },
                    }
                    existing = tasks.setdefault(task_id, task)
                    if existing["task_description_sha256"] != task_description_sha256:
                        raise ValueError(f"task identity collision for {source_task_id}")

                    action_name = row["expected_action"]["name"]
                    task_complete = action_name == "finish"
                    primary_action = _primary_action(action_name)
                    trajectory_ids.add(int(row["trajectory_id"]))
                    checkpoint_id = _sha256(
                        f"{parquet_revision}|{parquet_file.name}|{row_group}|{row_index}"
                    )
                    if checkpoint_id in checkpoint_ids:
                        raise ValueError(f"duplicate checkpoint id {checkpoint_id}")
                    checkpoint_ids.add(checkpoint_id)
                    info = row["info"]
                    checkpoint = {
                        "schema_version": "swe-pivot-transfer-checkpoint.v0",
                        "checkpoint_id": checkpoint_id,
                        "task_id": task_id,
                        "leakage_group": leakage_group,
                        "record_split": task["record_split"],
                        "input": _online_features(
                            terminal_state,
                            max(0, int(info["step"]) - 1),
                            _observed_history(items),
                        ),
                        "target": {
                            "primary_action": primary_action,
                            "action_classes": [primary_action],
                            "task_complete": task_complete,
                            "command_count": None,
                            "command_chars": None,
                        },
                        "provenance": {
                            "source_id": source["source_id"],
                            "source_trajectory_uid": str(row["trajectory_id"]),
                            "teacher_model": "not-declared",
                            "harness": row["metadata"]["agent_cls"],
                            "agent_ref": row["agent_ref"]["name"],
                        },
                        "audit_only": {
                            "expected_action_name": action_name,
                            "pass_rate": row["pass_rate"],
                            "pass_rate_total": row["pass_rate_total"],
                            "pass_rate_passed": row["pass_rate_passed"],
                            "note": "Post-generation confidence; excluded from model input.",
                        },
                    }
                    checkpoint_handle.write(json.dumps(checkpoint, sort_keys=True) + "\n")
                    action_counts[action_name] += 1
                    finish_counts[str(task_complete).lower()] += 1
                    confidence_counts[f"{float(row['pass_rate']):.10g}"] += 1
                    split_counts[checkpoint["record_split"]] += 1

    with tasks_path.open("w", encoding="utf-8") as task_handle:
        for task_id in sorted(tasks):
            task_handle.write(json.dumps(tasks[task_id], sort_keys=True) + "\n")

    summary = {
        "schema_version": "swe-pivot-transfer-checkpoint-summary.v0",
        "source": {
            "source_id": source["source_id"],
            "dataset_id": source["dataset_id"],
            "revision": source["revision"],
            "parquet_revision": parquet_revision,
            "license": source["license"],
        },
        "raw_row_count": row_count,
        "checkpoint_count": len(checkpoint_ids),
        "unique_checkpoint_id_count": len(checkpoint_ids),
        "task_record_count": len(tasks),
        "source_task_id_count": len(source_task_ids),
        "trajectory_count": len(trajectory_ids),
        "parse_failure_count": parse_failures,
        "expected_action_counts": dict(sorted(action_counts.items())),
        "task_complete_counts": dict(sorted(finish_counts.items())),
        "pass_rate_counts": dict(sorted(confidence_counts.items())),
        "checkpoint_split_counts": dict(sorted(split_counts.items())),
        "intended_use": (
            "Completed external transfer diagnostic; high-confidence rows may now be used "
            "for stage pretraining after overlap filtering."
        ),
        "leakage_guards": {
            "expected_action_arguments_retained": False,
            "reference_message_retained": False,
            "pass_rate_in_model_input": False,
            "all_checkpoint_ids_unique": True,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build sealed SWE pivot transfer data")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    summary = build_swe_transfer_checkpoints(
        registry_path=args.registry,
        snapshot_path=args.snapshot,
        checkpoints_path=args.checkpoints,
        tasks_path=args.tasks,
        summary_path=args.summary,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
