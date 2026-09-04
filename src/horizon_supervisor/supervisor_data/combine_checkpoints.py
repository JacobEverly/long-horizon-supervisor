from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DEFAULT_PIVOT_CHECKPOINTS = Path(
    "data/supervisor/terminal-pivot-checkpoints-v0.jsonl"
)
DEFAULT_PIVOT_TASKS = Path("data/supervisor/terminal-pivot-tasks-v0.jsonl")
DEFAULT_OPENTHOUGHTS_CHECKPOINTS = Path(
    "data/supervisor/openthoughts-checkpoints-v0.jsonl"
)
DEFAULT_OPENTHOUGHTS_TASKS = Path("data/supervisor/openthoughts-tasks-v0.jsonl")
DEFAULT_SWE_CHECKPOINTS = Path(
    "data/supervisor/swe-pivot-transfer-checkpoints-v0.jsonl"
)
DEFAULT_SWE_TASKS = Path("data/supervisor/swe-pivot-transfer-tasks-v0.jsonl")
DEFAULT_CHECKPOINTS = Path("data/supervisor/supervisor-checkpoints-v1.jsonl")
DEFAULT_TASKS = Path("data/supervisor/supervisor-tasks-v1.jsonl")
DEFAULT_SUMMARY = Path("data/supervisor/supervisor-checkpoints-v1-summary.json")


def _iter_jsonl(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def combine_checkpoints(
    pivot_checkpoints_path: Path = DEFAULT_PIVOT_CHECKPOINTS,
    pivot_tasks_path: Path = DEFAULT_PIVOT_TASKS,
    openthoughts_checkpoints_path: Path = DEFAULT_OPENTHOUGHTS_CHECKPOINTS,
    openthoughts_tasks_path: Path = DEFAULT_OPENTHOUGHTS_TASKS,
    checkpoints_path: Path = DEFAULT_CHECKPOINTS,
    tasks_path: Path = DEFAULT_TASKS,
    summary_path: Path = DEFAULT_SUMMARY,
    swe_checkpoints_path: Path | None = None,
    swe_tasks_path: Path | None = None,
    swe_min_pass_rate: float = 0.625,
) -> dict[str, Any]:
    task_paths = [pivot_tasks_path, openthoughts_tasks_path]
    checkpoint_sources: list[tuple[Path, float | None]] = [
        (pivot_checkpoints_path, None),
        (openthoughts_checkpoints_path, None),
    ]
    if (swe_checkpoints_path is None) != (swe_tasks_path is None):
        raise ValueError("SWE checkpoint and task paths must be supplied together")
    if swe_checkpoints_path is not None and swe_tasks_path is not None:
        task_paths.append(swe_tasks_path)
        checkpoint_sources.append((swe_checkpoints_path, swe_min_pass_rate))

    eligible_task_ids: set[str] = set()
    filtered_source_ids: set[str] = set()
    filtered_checkpoint_counts: Counter[str] = Counter()
    for path, min_pass_rate in checkpoint_sources:
        for checkpoint in _iter_jsonl(path):
            source_id = checkpoint["provenance"]["source_id"]
            if min_pass_rate is not None:
                filtered_source_ids.add(source_id)
                pass_rate = checkpoint.get("audit_only", {}).get("pass_rate")
                if pass_rate is None:
                    raise ValueError(
                        f"checkpoint source {source_id} lacks required pass_rate"
                    )
                if float(pass_rate) < min_pass_rate:
                    filtered_checkpoint_counts[source_id] += 1
                    continue
            eligible_task_ids.add(checkpoint["task_id"])

    task_ids: set[str] = set()
    checkpoint_ids: set[str] = set()
    leakage_splits: dict[str, str] = {}
    leakage_sources: dict[str, set[str]] = defaultdict(set)
    task_source_counts: Counter[str] = Counter()
    task_split_counts: Counter[str] = Counter()
    filtered_task_counts: Counter[str] = Counter()

    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    with tasks_path.open("w", encoding="utf-8") as output:
        for path in task_paths:
            for task in _iter_jsonl(path):
                task_id = task["task_id"]
                source_id = task["source"]["source_id"]
                if source_id in filtered_source_ids and task_id not in eligible_task_ids:
                    filtered_task_counts[source_id] += 1
                    continue
                if task_id in task_ids:
                    raise ValueError(f"duplicate task id {task_id}")
                task_ids.add(task_id)
                leakage_group = task["leakage_group"]
                split = task["record_split"]
                existing_split = leakage_splits.setdefault(leakage_group, split)
                if existing_split != split:
                    raise ValueError(
                        f"leakage group {leakage_group} appears in multiple splits"
                    )
                leakage_sources[leakage_group].add(source_id)
                task_source_counts[source_id] += 1
                task_split_counts[split] += 1
                output.write(json.dumps(task, sort_keys=True) + "\n")

    missing_task_ids = eligible_task_ids - task_ids
    if missing_task_ids:
        raise ValueError(
            f"{len(missing_task_ids)} eligible checkpoint tasks are missing task records"
        )

    checkpoint_source_counts: Counter[str] = Counter()
    checkpoint_split_counts: Counter[str] = Counter()
    teacher_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    with checkpoints_path.open("w", encoding="utf-8") as output:
        for path, min_pass_rate in checkpoint_sources:
            for checkpoint in _iter_jsonl(path):
                source_id = checkpoint["provenance"]["source_id"]
                if min_pass_rate is not None:
                    pass_rate = checkpoint.get("audit_only", {}).get("pass_rate")
                    if pass_rate is None:
                        raise ValueError(
                            f"checkpoint source {source_id} lacks required pass_rate"
                        )
                    if float(pass_rate) < min_pass_rate:
                        continue
                checkpoint_id = checkpoint["checkpoint_id"]
                if checkpoint_id in checkpoint_ids:
                    raise ValueError(f"duplicate checkpoint id {checkpoint_id}")
                checkpoint_ids.add(checkpoint_id)
                if checkpoint["task_id"] not in task_ids:
                    raise ValueError(
                        f"checkpoint {checkpoint_id} references a missing task"
                    )
                leakage_group = checkpoint["leakage_group"]
                split = checkpoint["record_split"]
                if leakage_splits.get(leakage_group) != split:
                    raise ValueError(
                        f"checkpoint {checkpoint_id} violates leakage-group split"
                    )
                checkpoint_source_counts[source_id] += 1
                checkpoint_split_counts[split] += 1
                teacher_counts[checkpoint["provenance"]["teacher_model"]] += 1
                target_counts[str(checkpoint["target"]["task_complete"]).lower()] += 1
                output.write(json.dumps(checkpoint, sort_keys=True) + "\n")

    overlap_groups = {
        leakage_group: sorted(sources)
        for leakage_group, sources in leakage_sources.items()
        if len(sources) > 1
    }
    summary = {
        "schema_version": "supervisor-checkpoint-summary.v1",
        "checkpoint_count": len(checkpoint_ids),
        "unique_checkpoint_id_count": len(checkpoint_ids),
        "task_record_count": len(task_ids),
        "unique_leakage_group_count": len(leakage_splits),
        "cross_source_overlap_group_count": len(overlap_groups),
        "cross_source_overlap_groups": overlap_groups,
        "checkpoint_source_counts": dict(sorted(checkpoint_source_counts.items())),
        "task_source_counts": dict(sorted(task_source_counts.items())),
        "checkpoint_split_counts": dict(sorted(checkpoint_split_counts.items())),
        "task_split_counts": dict(sorted(task_split_counts.items())),
        "teacher_model_counts": dict(sorted(teacher_counts.items())),
        "task_complete_counts": dict(sorted(target_counts.items())),
        "filtered_checkpoint_counts": dict(sorted(filtered_checkpoint_counts.items())),
        "filtered_task_counts": dict(sorted(filtered_task_counts.items())),
        "source_filters": {
            "nemotron-swe-pivot-v1": (
                {"pass_rate_min": swe_min_pass_rate}
                if swe_checkpoints_path is not None
                else None
            )
        },
        "leakage_guards": {
            "split_group": "normalized task-description SHA-256 across all sources",
            "all_checkpoint_task_references_resolved": True,
            "all_checkpoint_ids_unique": True,
            "all_task_ids_unique": True,
        },
        "input_files": {
            "checkpoints": [str(path) for path, _ in checkpoint_sources],
            "tasks": [str(path) for path in task_paths],
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine leakage-safe checkpoint sources")
    parser.add_argument("--pivot-checkpoints", type=Path, default=DEFAULT_PIVOT_CHECKPOINTS)
    parser.add_argument("--pivot-tasks", type=Path, default=DEFAULT_PIVOT_TASKS)
    parser.add_argument(
        "--openthoughts-checkpoints",
        type=Path,
        default=DEFAULT_OPENTHOUGHTS_CHECKPOINTS,
    )
    parser.add_argument("--openthoughts-tasks", type=Path, default=DEFAULT_OPENTHOUGHTS_TASKS)
    parser.add_argument("--swe-checkpoints", type=Path)
    parser.add_argument("--swe-tasks", type=Path)
    parser.add_argument("--swe-min-pass-rate", type=float, default=0.625)
    parser.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    summary = combine_checkpoints(
        pivot_checkpoints_path=args.pivot_checkpoints,
        pivot_tasks_path=args.pivot_tasks,
        openthoughts_checkpoints_path=args.openthoughts_checkpoints,
        openthoughts_tasks_path=args.openthoughts_tasks,
        checkpoints_path=args.checkpoints,
        tasks_path=args.tasks,
        summary_path=args.summary,
        swe_checkpoints_path=args.swe_checkpoints,
        swe_tasks_path=args.swe_tasks,
        swe_min_pass_rate=args.swe_min_pass_rate,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
