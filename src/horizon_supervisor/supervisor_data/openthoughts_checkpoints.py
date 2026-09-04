from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

from horizon_supervisor.supervisor_data.pivot_checkpoints import (
    _online_features,
    _split,
    classify_action,
    split_conversation,
    task_leakage_sha256,
)

DEFAULT_REGISTRY = Path("data/supervisor/source-registry-v0.json")
DEFAULT_CHECKPOINTS = Path("data/supervisor/openthoughts-checkpoints-v0.jsonl")
DEFAULT_TASKS = Path("data/supervisor/openthoughts-tasks-v0.jsonl")
DEFAULT_SUMMARY = Path("data/supervisor/openthoughts-checkpoints-v0-summary.json")
SOURCE_ID = "openthoughts-agent-v1-sft"
SOURCE_FILE = "data/train-00000-of-00001.parquet"
COLUMNS = ["conversations", "task", "model", "agent", "run_id", "episode"]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source(registry: dict[str, Any]) -> dict[str, Any]:
    for item in registry["sources"]:
        if item["source_id"] == SOURCE_ID:
            return item
    raise ValueError(f"missing source {SOURCE_ID!r}")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def extract_action_json(content: str) -> tuple[dict[str, Any] | None, str]:
    content = content.strip()
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict) and "commands" in value:
        return value, "direct"

    decoder = json.JSONDecoder()
    for position, character in enumerate(content):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(content[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "commands" in value:
            return value, "embedded"
    return None, "failed"


def _checkpoint(
    row: dict[str, Any],
    source: dict[str, Any],
    task: dict[str, Any],
    history: list[dict[str, Any]],
    answer: dict[str, Any],
    assistant_turn_index: int,
    total_assistant_turns: int,
    parse_mode: str,
    trajectory_uid: str,
) -> dict[str, Any]:
    _, terminal_state, observed_history = split_conversation(history)
    commands = answer.get("commands") or []
    task_complete = bool(answer.get("task_complete", False))
    primary_action, action_classes = classify_action(commands, task_complete)
    checkpoint_id = _sha256(
        f"{source['dataset_id']}@{source['revision']}|{trajectory_uid}|"
        f"{assistant_turn_index}"
    )
    return {
        "schema_version": "openthoughts-checkpoint.v0",
        "checkpoint_id": checkpoint_id,
        "task_id": task["task_id"],
        "leakage_group": task["leakage_group"],
        "record_split": task["record_split"],
        "input": _online_features(
            terminal_state,
            assistant_turn_index,
            observed_history,
        ),
        "target": {
            "primary_action": primary_action,
            "action_classes": action_classes,
            "task_complete": task_complete,
            "command_count": len(commands),
            "command_chars": sum(
                len(str(command.get("keystrokes", ""))) for command in commands
            ),
        },
        "provenance": {
            "source_id": source["source_id"],
            "source_trajectory_uid": trajectory_uid,
            "teacher_model": row["model"],
            "harness": row["agent"],
            "agent_ref": row["agent"],
            "action_parse_mode": parse_mode,
        },
        "audit_only": {
            "total_source_agent_turns": total_assistant_turns,
            "note": "Post-run value; excluded from model input to prevent future leakage.",
        },
    }


def build_openthoughts_checkpoints(
    registry_path: Path = DEFAULT_REGISTRY,
    checkpoints_path: Path = DEFAULT_CHECKPOINTS,
    tasks_path: Path = DEFAULT_TASKS,
    summary_path: Path = DEFAULT_SUMMARY,
    *,
    api: HfApi | None = None,
    fs: HfFileSystem | None = None,
) -> dict[str, Any]:
    source = _source(_load_json(registry_path))
    api = api or HfApi()
    fs = fs or HfFileSystem()
    info = api.dataset_info(source["dataset_id"], revision=source["revision"])
    if info.sha != source["revision"]:
        raise RuntimeError(
            f"revision mismatch: expected {source['revision']}, received {info.sha}"
        )
    remote_path = f"datasets/{source['dataset_id']}@{source['revision']}/{SOURCE_FILE}"

    tasks: dict[str, dict[str, Any]] = {}
    trajectory_ids: set[str] = set()
    checkpoint_ids: set[str] = set()
    checkpoint_count = 0
    raw_trajectory_count = 0
    raw_assistant_turns = 0
    dropped_task_parse = 0
    dropped_action_parse = 0
    parse_modes: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    action_class_counts: Counter[str] = Counter()
    teacher_counts: Counter[str] = Counter()

    checkpoints_path.parent.mkdir(parents=True, exist_ok=True)
    with fs.open(remote_path, "rb") as remote_handle, checkpoints_path.open(
        "w", encoding="utf-8"
    ) as checkpoint_handle:
        parquet = pq.ParquetFile(remote_handle)
        for row_group in range(parquet.num_row_groups):
            rows = parquet.read_row_group(row_group, columns=COLUMNS).to_pylist()
            for row in rows:
                raw_trajectory_count += 1
                conversations = row["conversations"]
                assistant_positions = [
                    index
                    for index, message in enumerate(conversations)
                    if message["role"] == "assistant"
                ]
                raw_assistant_turns += len(assistant_positions)
                try:
                    task_description, _, _ = split_conversation(conversations[:1])
                except ValueError:
                    dropped_task_parse += 1
                    continue
                task_description_sha256 = _sha256(task_description)
                leakage_group = task_leakage_sha256(task_description)
                source_task_name = row["task"]
                task_id = _sha256(
                    f"{source['dataset_id']}@{source['revision']}|{source_task_name}|"
                    f"{task_description_sha256}"
                )
                task = {
                    "schema_version": "openthoughts-task.v0",
                    "task_id": task_id,
                    "source_task_name": source_task_name,
                    "record_split": _split(leakage_group),
                    "leakage_group": leakage_group,
                    "task_description": task_description,
                    "task_description_sha256": task_description_sha256,
                    "task_chars": len(task_description),
                    "source": {
                        "source_id": source["source_id"],
                        "dataset_id": source["dataset_id"],
                        "revision": source["revision"],
                    },
                }
                existing = tasks.setdefault(task_id, task)
                if existing["task_description_sha256"] != task_description_sha256:
                    raise ValueError(f"task collision for {source_task_name!r}")
                trajectory_uid = _sha256(
                    f"{source['dataset_id']}@{source['revision']}|{task_id}|"
                    f"{row['run_id']}|{row['episode']}"
                )
                if trajectory_uid in trajectory_ids:
                    raise ValueError(f"duplicate trajectory identity for {source_task_name!r}")
                trajectory_ids.add(trajectory_uid)

                for assistant_turn_index, message_position in enumerate(
                    assistant_positions
                ):
                    answer, parse_mode = extract_action_json(
                        str(conversations[message_position]["content"])
                    )
                    parse_modes[parse_mode] += 1
                    if answer is None:
                        dropped_action_parse += 1
                        continue
                    history = conversations[:message_position]
                    checkpoint = _checkpoint(
                        row,
                        source,
                        task,
                        history,
                        answer,
                        assistant_turn_index,
                        len(assistant_positions),
                        parse_mode,
                        trajectory_uid,
                    )
                    if checkpoint["checkpoint_id"] in checkpoint_ids:
                        raise ValueError(
                            f"duplicate checkpoint identity for {source_task_name!r}"
                        )
                    checkpoint_ids.add(checkpoint["checkpoint_id"])
                    checkpoint_handle.write(json.dumps(checkpoint, sort_keys=True) + "\n")
                    checkpoint_count += 1
                    split_counts[checkpoint["record_split"]] += 1
                    action_counts[checkpoint["target"]["primary_action"]] += 1
                    action_class_counts.update(checkpoint["target"]["action_classes"])
                    teacher_counts[checkpoint["provenance"]["teacher_model"]] += 1

    with tasks_path.open("w", encoding="utf-8") as task_handle:
        for task_id in sorted(tasks):
            task_handle.write(json.dumps(tasks[task_id], sort_keys=True) + "\n")

    task_split_counts = Counter(task["record_split"] for task in tasks.values())
    summary = {
        "schema_version": "openthoughts-checkpoint-summary.v0",
        "source": {
            "source_id": source["source_id"],
            "dataset_id": source["dataset_id"],
            "revision": source["revision"],
            "license": source["license"],
        },
        "raw_trajectory_count": raw_trajectory_count,
        "trajectory_count": len(trajectory_ids),
        "task_count": len(tasks),
        "raw_assistant_turn_count": raw_assistant_turns,
        "checkpoint_count": checkpoint_count,
        "unique_checkpoint_id_count": len(checkpoint_ids),
        "dropped": {
            "trajectory_task_parse_failures": dropped_task_parse,
            "assistant_action_parse_failures": dropped_action_parse,
        },
        "action_parse_mode_counts": dict(sorted(parse_modes.items())),
        "checkpoint_split_counts": dict(sorted(split_counts.items())),
        "task_split_counts": dict(sorted(task_split_counts.items())),
        "primary_action_counts": dict(sorted(action_counts.items())),
        "action_class_counts": dict(sorted(action_class_counts.items())),
        "teacher_model_counts": dict(sorted(teacher_counts.items())),
        "leakage_guards": {
            "split_group": "normalized task-description SHA-256",
            "reference_commands_retained": False,
            "reference_analysis_retained": False,
            "reference_plan_retained": False,
            "future_total_turns_in_input": False,
        },
        "label_scope": (
            "Stage/action supervision from released successful trajectories; not a "
            "counterfactual model-switch or failure-recovery label."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build OpenThoughts checkpoint data")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    summary = build_openthoughts_checkpoints(
        registry_path=args.registry,
        checkpoints_path=args.checkpoints,
        tasks_path=args.tasks,
        summary_path=args.summary,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
