from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DEFAULT_BENCHMARK = Path("benchmarks/terminal-bench-2.1-gate7.json")
DEFAULT_TERMINAL_CATALOG = Path("data/supervisor/terminal-corpus-catalog-v0.jsonl")
DEFAULT_COMPANION_CATALOG = Path("data/supervisor/companion-task-catalog-v0.jsonl")
DEFAULT_OUTPUT = Path("data/supervisor/aligned-task-panel-v0.jsonl")
DEFAULT_SUMMARY = Path("data/supervisor/aligned-task-panel-v0-summary.json")
PANEL_SEED = "aligned-task-panel-v0|2026-08-25"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_jsonl(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _rank(task_id: str) -> str:
    return hashlib.sha256(f"{PANEL_SEED}|{task_id}".encode()).hexdigest()


def build_proportional_panel(
    benchmark_path: Path = DEFAULT_BENCHMARK,
    terminal_catalog_path: Path = DEFAULT_TERMINAL_CATALOG,
    companion_catalog_path: Path = DEFAULT_COMPANION_CATALOG,
    output_path: Path = DEFAULT_OUTPUT,
    summary_path: Path = DEFAULT_SUMMARY,
    *,
    examples_per_benchmark_seat: int = 65,
) -> dict[str, Any]:
    if examples_per_benchmark_seat <= 0:
        raise ValueError("examples_per_benchmark_seat must be positive")
    benchmark = _load_json(benchmark_path)
    benchmark_counts = Counter(task["category"] for task in benchmark["tasks"])

    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in (terminal_catalog_path, companion_catalog_path):
        for row in _iter_jsonl(path):
            if row["recommended_split"] != "train":
                continue
            if row["category"] in benchmark_counts:
                candidates[row["category"]].append(row)

    selected = []
    availability = {}
    for category, benchmark_seats in sorted(benchmark_counts.items()):
        required = benchmark_seats * examples_per_benchmark_seat
        pool = candidates[category]
        availability[category] = len(pool)
        if len(pool) < required:
            raise ValueError(
                f"{category} has {len(pool)} train tasks but panel requires {required}"
            )
        ranked = sorted(pool, key=lambda row: (_rank(row["task_id"]), row["task_id"]))
        for row in ranked[:required]:
            selected.append(
                row
                | {
                    "panel": {
                        "panel_id": "aligned-task-panel-v0",
                        "benchmark_category": category,
                        "benchmark_seats": benchmark_seats,
                        "examples_per_benchmark_seat": examples_per_benchmark_seat,
                        "selection_rank": _rank(row["task_id"]),
                    }
                }
            )

    selected.sort(key=lambda row: (row["category"], row["panel"]["selection_rank"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    category_counts = Counter(row["category"] for row in selected)
    source_counts = Counter(row["source"]["source_id"] for row in selected)
    total_benchmark = sum(benchmark_counts.values())
    total_panel = len(selected)
    distribution_check = {
        category: {
            "benchmark_share": benchmark_counts[category] / total_benchmark,
            "panel_share": category_counts[category] / total_panel,
            "exact_match": (
                benchmark_counts[category] * total_panel
                == category_counts[category] * total_benchmark
            ),
        }
        for category in sorted(benchmark_counts)
    }
    summary = {
        "schema_version": "aligned-task-panel-summary.v0",
        "panel_id": "aligned-task-panel-v0",
        "selection_seed": PANEL_SEED,
        "benchmark": benchmark["benchmark"],
        "benchmark_selection_id": benchmark["selection_id"],
        "examples_per_benchmark_seat": examples_per_benchmark_seat,
        "record_count": total_panel,
        "category_counts": dict(sorted(category_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "train_pool_availability": dict(sorted(availability.items())),
        "distribution_check": distribution_check,
        "all_tasks_from_source_train_splits": True,
        "task_content_materialized": False,
        "intended_use": (
            "Leakage-safe task manifest for proportional prompt materialization and matched "
            "rollout sampling; it is not itself a text training file."
        ),
        "scale_decision": (
            "65 examples per benchmark seat is the largest exact-proportion panel available "
            "without consuming held-out game tasks or procedurally generating more games."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a benchmark-proportional task panel")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--terminal-catalog", type=Path, default=DEFAULT_TERMINAL_CATALOG)
    parser.add_argument("--companion-catalog", type=Path, default=DEFAULT_COMPANION_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--examples-per-seat", type=int, default=65)
    args = parser.parse_args()
    summary = build_proportional_panel(
        benchmark_path=args.benchmark,
        terminal_catalog_path=args.terminal_catalog,
        companion_catalog_path=args.companion_catalog,
        output_path=args.output,
        summary_path=args.summary,
        examples_per_benchmark_seat=args.examples_per_seat,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
