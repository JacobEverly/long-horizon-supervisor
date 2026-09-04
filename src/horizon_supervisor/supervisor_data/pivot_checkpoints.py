from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, TextIO

from huggingface_hub import hf_hub_download

DEFAULT_REGISTRY = Path("data/supervisor/source-registry-v0.json")
DEFAULT_CHECKPOINTS = Path("data/supervisor/terminal-pivot-checkpoints-v0.jsonl")
DEFAULT_TASKS = Path("data/supervisor/terminal-pivot-tasks-v0.jsonl")
DEFAULT_SUMMARY = Path("data/supervisor/terminal-pivot-checkpoints-v0-summary.json")
SOURCE_ID = "nemotron-terminal-pivot-v1"

TASK_MARKER = "Task Description:\n"
STATE_MARKER = "\n\nCurrent terminal state:\nCurrent Terminal Screen:\n"
TERMINAL_TAIL_CHARS = 4_096

ERROR_PATTERN = re.compile(
    r"\b(error|failed|failure|exception|traceback|timeout|timed out|segfault)\b",
    re.IGNORECASE,
)
PASS_PATTERN = re.compile(
    r"\b(passed|success|successful|ok|all tests pass)\b", re.IGNORECASE
)
TEST_PATTERN = re.compile(r"\b(pytest|unittest|test|tests|verify|verification)\b", re.IGNORECASE)
PROMPT_PATTERN = re.compile(r"(?m)^[^\n]{0,80}(?:[$#>]|[❯➜])\s*$")

VALIDATE_PATTERN = re.compile(
    r"(^|[;&|\n]\s*)(pytest|python\s+-m\s+pytest|cargo\s+test|go\s+test|"
    r"npm\s+(run\s+)?test|pnpm\s+test|yarn\s+test|make\s+(test|check)|"
    r"ruff|mypy|eslint|check|verify)\b",
    re.IGNORECASE,
)
INSPECT_PATTERN = re.compile(
    r"(^|[;&|\n]\s*)(ls|find|rg|grep|cat|head|tail|pwd|tree|wc|stat|"
    r"git\s+(status|diff|log|show)|sed\s+-n)\b",
    re.IGNORECASE,
)
EDIT_PATTERN = re.compile(
    r"(apply_patch|sed\s+-i|perl\s+-pi|\btee\b|(^|\s)(mkdir|touch|cp|mv|rm|chmod)\b|"
    r">{1,2}\s*[^&])",
    re.IGNORECASE,
)
BUILD_PATTERN = re.compile(
    r"(^|[;&|\n]\s*)(apt(-get)?|pip|uv|poetry|npm|pnpm|yarn)\s+(install|add)|"
    r"(^|[;&|\n]\s*)(make|cmake|ninja|gcc|g\+\+|clang|cargo\s+build|go\s+build)\b",
    re.IGNORECASE,
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source(registry: dict[str, Any], source_id: str) -> dict[str, Any]:
    for item in registry["sources"]:
        if item["source_id"] == source_id:
            return item
    raise ValueError(f"missing source {source_id!r}")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def task_leakage_sha256(task_description: str) -> str:
    normalized = unicodedata.normalize("NFKC", task_description)
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    return _sha256(normalized)


def _split(task_id: str) -> str:
    bucket = int(task_id[:8], 16) % 100
    if bucket < 75:
        return "train"
    if bucket < 85:
        return "validation"
    if bucket < 95:
        return "internal_test"
    return "sealed_test"


def split_prompt(prompt: str) -> tuple[str, str]:
    if TASK_MARKER not in prompt or STATE_MARKER not in prompt:
        raise ValueError("pivot prompt does not contain the expected task/state markers")
    _, task_and_state = prompt.split(TASK_MARKER, maxsplit=1)
    task_description, terminal_state = task_and_state.split(STATE_MARKER, maxsplit=1)
    return task_description.strip(), terminal_state.rstrip()


def split_conversation(messages: list[dict[str, Any]]) -> tuple[str, str, dict[str, int]]:
    if not messages or messages[0].get("role") != "user":
        raise ValueError("pivot conversation must start with a user task message")
    task_description, initial_terminal = split_prompt(str(messages[0]["content"]))
    user_messages = [message for message in messages if message.get("role") == "user"]
    assistant_messages = [
        message for message in messages if message.get("role") == "assistant"
    ]
    terminal_state = (
        initial_terminal if len(user_messages) == 1 else str(user_messages[-1]["content"]).rstrip()
    )
    observed = {
        "history_message_count": len(messages) - 1,
        "prior_assistant_turn_count": len(assistant_messages),
        "prior_user_update_count": max(0, len(user_messages) - 1),
        "observed_history_chars": sum(len(str(message.get("content", ""))) for message in messages),
    }
    return task_description, terminal_state, observed


def _classify_command_classes(command: str) -> set[str]:
    command = command.strip()
    if not command:
        return {"wait"}
    labels = set()
    if VALIDATE_PATTERN.search(command):
        labels.add("validate")
    if INSPECT_PATTERN.search(command):
        labels.add("inspect")
    if EDIT_PATTERN.search(command):
        labels.add("edit")
    if BUILD_PATTERN.search(command):
        labels.add("build_or_install")
    if not labels:
        labels.add("execute")
    return labels


def classify_command(command: str) -> str:
    labels = _classify_command_classes(command)
    return next(iter(labels)) if len(labels) == 1 else "mixed"


def classify_action(commands: list[dict[str, Any]], task_complete: bool) -> tuple[str, list[str]]:
    classes = [
        _classify_command_classes(str(command.get("keystrokes", ""))) for command in commands
    ]
    unique = sorted(set().union(*classes)) if classes else []
    if not commands:
        return ("finish" if task_complete else "wait"), unique
    if len(unique) == 1:
        return unique[0], unique
    return "mixed", unique


def _online_features(
    terminal_state: str, turn_index: int, observed_history: dict[str, int]
) -> dict[str, Any]:
    return observed_history | {
        "turn_index": turn_index,
        "terminal_chars": len(terminal_state),
        "terminal_lines": terminal_state.count("\n") + bool(terminal_state),
        "error_signal_count": len(ERROR_PATTERN.findall(terminal_state)),
        "pass_signal_count": len(PASS_PATTERN.findall(terminal_state)),
        "test_signal_count": len(TEST_PATTERN.findall(terminal_state)),
        "shell_prompt_count": len(PROMPT_PATTERN.findall(terminal_state)),
        "terminal_tail": terminal_state[-TERMINAL_TAIL_CHARS:],
        "terminal_tail_truncated": len(terminal_state) > TERMINAL_TAIL_CHARS,
    }


def transform_pivot_row(
    row: dict[str, Any], source: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    input_messages = row["responses_create_params"]["input"]
    task_description, terminal_state, observed_history = split_conversation(input_messages)
    answer = json.loads(row["expected_answer"])
    metadata = row["metadata"]
    task_name = row["task_name"]
    task_description_sha256 = _sha256(task_description)
    leakage_group = task_leakage_sha256(task_description)
    task_id = _sha256(
        f"{source['dataset_id']}@{source['revision']}|{task_name}|"
        f"{task_description_sha256}"
    )
    record_split = _split(leakage_group)
    commands = answer.get("commands") or []
    task_complete = bool(answer.get("task_complete", False))
    primary_action, action_classes = classify_action(commands, task_complete)

    task = {
        "schema_version": "terminal-pivot-task.v0",
        "task_id": task_id,
        "source_task_name": task_name,
        "record_split": record_split,
        "task_description": task_description,
        "task_description_sha256": task_description_sha256,
        "leakage_group": leakage_group,
        "task_chars": len(task_description),
        "source": {
            "source_id": source["source_id"],
            "dataset_id": source["dataset_id"],
            "revision": source["revision"],
        },
    }
    checkpoint = {
        "schema_version": "terminal-pivot-checkpoint.v0",
        "checkpoint_id": row["uuid"],
        "task_id": task_id,
        "leakage_group": leakage_group,
        "record_split": record_split,
        "input": _online_features(
            terminal_state,
            int(metadata["pivot_agent_turn_index"]),
            observed_history,
        ),
        "target": {
            "primary_action": primary_action,
            "action_classes": action_classes,
            "task_complete": task_complete,
            "command_count": len(commands),
            "command_chars": sum(len(str(command.get("keystrokes", ""))) for command in commands),
        },
        "provenance": {
            "source_id": source["source_id"],
            "source_uuid": row["uuid"],
            "source_trajectory_uid": metadata["source_trajectory_uid"],
            "teacher_model": metadata["teacher_model"],
            "harness": metadata["harness"],
            "agent_ref": row["agent_ref"],
        },
        "audit_only": {
            "total_source_agent_turns": int(metadata["total_source_agent_turns"]),
            "note": "Post-run value; excluded from model input to prevent future leakage.",
        },
    }
    return task, checkpoint


def _iter_rows(handle: TextIO) -> Any:
    for line_number, line in enumerate(handle, start=1):
        if line.strip():
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on source line {line_number}") from exc


def build_pivot_checkpoints(
    registry_path: Path = DEFAULT_REGISTRY,
    checkpoints_path: Path = DEFAULT_CHECKPOINTS,
    tasks_path: Path = DEFAULT_TASKS,
    summary_path: Path = DEFAULT_SUMMARY,
    *,
    source_path: Path | None = None,
) -> dict[str, Any]:
    source = _source(_load_json(registry_path), SOURCE_ID)
    if source_path is None:
        source_path = Path(
            hf_hub_download(
                repo_id=source["dataset_id"],
                repo_type="dataset",
                filename="atcb_terminal_pivot_release_final_v2.jsonl",
                revision=source["revision"],
            )
        )

    tasks: dict[str, dict[str, Any]] = {}
    checkpoint_count = 0
    split_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    action_class_counts: Counter[str] = Counter()
    teacher_counts: Counter[str] = Counter()
    trajectory_ids: set[str] = set()
    source_task_names: set[str] = set()
    terminal_chars = 0
    truncated_count = 0

    checkpoints_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open(encoding="utf-8") as source_handle, checkpoints_path.open(
        "w", encoding="utf-8"
    ) as checkpoint_handle:
        for row in _iter_rows(source_handle):
            task, checkpoint = transform_pivot_row(row, source)
            existing = tasks.setdefault(task["task_id"], task)
            if existing["task_description_sha256"] != task["task_description_sha256"]:
                raise ValueError(f"task text changed within source task {row['task_name']!r}")
            checkpoint_handle.write(json.dumps(checkpoint, sort_keys=True) + "\n")
            checkpoint_count += 1
            split_counts[checkpoint["record_split"]] += 1
            action_counts[checkpoint["target"]["primary_action"]] += 1
            action_class_counts.update(checkpoint["target"]["action_classes"])
            teacher_counts[checkpoint["provenance"]["teacher_model"]] += 1
            trajectory_ids.add(checkpoint["provenance"]["source_trajectory_uid"])
            source_task_names.add(task["source_task_name"])
            terminal_chars += checkpoint["input"]["terminal_chars"]
            truncated_count += int(checkpoint["input"]["terminal_tail_truncated"])

    with tasks_path.open("w", encoding="utf-8") as task_handle:
        for task_id in sorted(tasks):
            task_handle.write(json.dumps(tasks[task_id], sort_keys=True) + "\n")

    task_split_counts = Counter(task["record_split"] for task in tasks.values())
    summary = {
        "schema_version": "terminal-pivot-checkpoint-summary.v0",
        "source": {
            "source_id": source["source_id"],
            "dataset_id": source["dataset_id"],
            "revision": source["revision"],
            "license": source["license"],
        },
        "checkpoint_count": checkpoint_count,
        "task_count": len(tasks),
        "source_task_name_count": len(source_task_names),
        "trajectory_count": len(trajectory_ids),
        "checkpoint_split_counts": dict(sorted(split_counts.items())),
        "task_split_counts": dict(sorted(task_split_counts.items())),
        "primary_action_counts": dict(sorted(action_counts.items())),
        "action_class_counts": dict(sorted(action_class_counts.items())),
        "teacher_model_counts": dict(sorted(teacher_counts.items())),
        "terminal_state": {
            "total_source_chars": terminal_chars,
            "tail_chars_retained_per_checkpoint": TERMINAL_TAIL_CHARS,
            "truncated_checkpoint_count": truncated_count,
        },
        "leakage_guards": {
            "split_group": "normalized task-description SHA-256",
            "reference_commands_retained": False,
            "reference_analysis_retained": False,
            "reference_plan_retained": False,
            "future_total_turns_in_input": False,
            "post_run_total_turns_location": "audit_only.total_source_agent_turns",
        },
        "label_scope": (
            "Stage/action supervision from verifier-passing trajectories; not a "
            "counterfactual model-switch or failure-recovery label."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build online-safe terminal pivot checkpoints")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    summary = build_pivot_checkpoints(
        registry_path=args.registry,
        source_path=args.source,
        checkpoints_path=args.checkpoints,
        tasks_path=args.tasks,
        summary_path=args.summary,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
