from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

DEFAULT_REGISTRY = Path("data/supervisor/source-registry-v0.json")
DEFAULT_BENCHMARK = Path("benchmarks/terminal-bench-2.1-gate7.json")
DEFAULT_OUTPUT = Path("data/supervisor/terminal-corpus-catalog-v0.jsonl")
DEFAULT_SUMMARY = Path("data/supervisor/terminal-corpus-catalog-v0-summary.json")

SOURCE_ID = "nemotron-terminal-corpus"
CATALOG_COLUMNS = ["task", "model", "agent", "enable_thinking"]

ADAPTER_LABELS = {
    "dataset_adapters/code.parquet": ("adapter", "software-engineering"),
    "dataset_adapters/math.parquet": ("adapter", "mathematics"),
    "dataset_adapters/swe.parquet": ("adapter", "software-engineering"),
}

ADJACENT_SOURCES = {
    "games": ["reasoning-gym-harbor-easy", "reasoning-gym-harbor-hard"],
    "machine-learning": [
        "nemotron-terminal-corpus:data-science",
        "ml-dev-bench-harbor",
    ],
    "mathematics": ["nemotron-terminal-corpus:dataset_adapters/math", "aime"],
    "optimization": ["algotune-harbor", "gso-harbor"],
    "personal-assistant": ["nemotron-workplace-assistant"],
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source(registry: dict[str, Any], source_id: str) -> dict[str, Any]:
    for item in registry["sources"]:
        if item["source_id"] == source_id:
            return item
    raise ValueError(f"missing source {source_id!r}")


def _catalog_files(source: dict[str, Any], api: HfApi) -> list[str]:
    info = api.dataset_info(source["dataset_id"], revision=source["revision"])
    if info.sha != source["revision"]:
        raise RuntimeError(
            f"revision mismatch: expected {source['revision']}, received {info.sha}"
        )
    files = sorted(
        sibling.rfilename
        for sibling in info.siblings
        if (
            sibling.rfilename in ADAPTER_LABELS
            or (
                sibling.rfilename.startswith("synthetic_tasks/skill_based/")
                and sibling.rfilename.endswith("/data_filtered.parquet")
            )
        )
    )
    if not files:
        raise RuntimeError("no terminal corpus Parquet files found")
    return files


def _file_labels(filename: str) -> tuple[str, str]:
    if filename in ADAPTER_LABELS:
        return ADAPTER_LABELS[filename]
    parts = PurePosixPath(filename).parts
    if len(parts) != 5 or parts[:2] != ("synthetic_tasks", "skill_based"):
        raise ValueError(f"unexpected skill file path: {filename}")
    return parts[2], parts[3].replace("_", "-")


def _split(task_key: str) -> str:
    bucket = int(hashlib.sha256(task_key.encode()).hexdigest()[:8], 16) % 100
    if bucket < 85:
        return "train"
    if bucket < 95:
        return "validation"
    return "internal_test"


def _task_id(source: dict[str, Any], filename: str, task_name: str) -> str:
    key = f"{source['dataset_id']}@{source['revision']}|{filename}|{task_name}"
    return hashlib.sha256(key.encode()).hexdigest()


def _coverage(
    catalog_counts: Counter[str], benchmark: dict[str, Any]
) -> dict[str, Any]:
    benchmark_counts = Counter(task["category"] for task in benchmark["tasks"])
    total_benchmark = sum(benchmark_counts.values())
    total_catalog = sum(catalog_counts.values())
    rows = []
    for category, benchmark_tasks in sorted(benchmark_counts.items()):
        source_tasks = catalog_counts[category]
        target_share = benchmark_tasks / total_benchmark
        source_share = source_tasks / total_catalog if total_catalog else 0.0
        status = "direct" if source_tasks else "adjacent_source_required"
        rows.append(
            {
                "benchmark_category": category,
                "benchmark_tasks": benchmark_tasks,
                "benchmark_share": round(target_share, 6),
                "direct_unique_source_tasks": source_tasks,
                "direct_source_share": round(source_share, 6),
                "importance_weight_if_direct": (
                    round(target_share / source_share, 6) if source_share else None
                ),
                "coverage_status": status,
                "companion_sources": ADJACENT_SOURCES.get(category, []),
            }
        )
    directly_covered_seats = sum(
        row["benchmark_tasks"]
        for row in rows
        if row["coverage_status"] == "direct"
    )
    return {
        "benchmark": benchmark["benchmark"],
        "selection_id": benchmark["selection_id"],
        "benchmark_task_count": total_benchmark,
        "directly_covered_benchmark_tasks": directly_covered_seats,
        "direct_coverage_rate": round(directly_covered_seats / total_benchmark, 6),
        "categories": rows,
    }


def build_terminal_catalog(
    registry_path: Path = DEFAULT_REGISTRY,
    benchmark_path: Path = DEFAULT_BENCHMARK,
    output_path: Path = DEFAULT_OUTPUT,
    summary_path: Path = DEFAULT_SUMMARY,
    *,
    api: HfApi | None = None,
    fs: HfFileSystem | None = None,
) -> dict[str, Any]:
    registry = _load_json(registry_path)
    benchmark = _load_json(benchmark_path)
    source = _source(registry, SOURCE_ID)
    api = api or HfApi()
    fs = fs or HfFileSystem()

    catalog: dict[tuple[str, str, str], dict[str, Any]] = {}
    trajectory_counts: Counter[tuple[str, str]] = Counter()
    unique_counts: Counter[tuple[str, str]] = Counter()

    files = _catalog_files(source, api)
    for filename in files:
        difficulty, category = _file_labels(filename)
        remote_path = f"datasets/{source['dataset_id']}@{source['revision']}/{filename}"
        with fs.open(remote_path, "rb") as handle:
            table = pq.ParquetFile(handle).read(columns=CATALOG_COLUMNS)
        trajectory_counts[(difficulty, category)] += table.num_rows
        for row in table.to_pylist():
            task_name = row["task"]
            key = (difficulty, category, task_name)
            if key not in catalog:
                task_key = f"{difficulty}|{category}|{task_name}"
                catalog[key] = {
                    "schema_version": "terminal-corpus-catalog.v0",
                    "task_id": _task_id(source, filename, task_name),
                    "source_task_name": task_name,
                    "difficulty": difficulty,
                    "category": category,
                    "recommended_split": _split(task_key),
                    "trajectory_count": 0,
                    "models": set(),
                    "agents": set(),
                    "enable_thinking_values": set(),
                    "source": {
                        "source_id": SOURCE_ID,
                        "dataset_id": source["dataset_id"],
                        "revision": source["revision"],
                        "file": filename,
                    },
                }
            item = catalog[key]
            item["trajectory_count"] += 1
            item["models"].add(row["model"])
            item["agents"].add(row["agent"])
            item["enable_thinking_values"].add(row["enable_thinking"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    with output_path.open("w", encoding="utf-8") as handle:
        for key in sorted(catalog):
            item = catalog[key]
            split_counts[item["recommended_split"]] += 1
            category_counts[item["category"]] += 1
            unique_counts[(item["difficulty"], item["category"])] += 1
            serializable = item | {
                "models": sorted(item["models"]),
                "agents": sorted(item["agents"]),
                "enable_thinking_values": sorted(item["enable_thinking_values"]),
            }
            handle.write(json.dumps(serializable, sort_keys=True) + "\n")

    by_slice = []
    for difficulty, category in sorted(trajectory_counts):
        by_slice.append(
            {
                "difficulty": difficulty,
                "category": category,
                "trajectory_count": trajectory_counts[(difficulty, category)],
                "unique_task_count": unique_counts[(difficulty, category)],
            }
        )
    summary = {
        "schema_version": "terminal-corpus-catalog-summary.v0",
        "source": {
            "source_id": SOURCE_ID,
            "dataset_id": source["dataset_id"],
            "revision": source["revision"],
            "license": source["license"],
        },
        "scope": "all terminal corpus trajectories; conversation content was not read",
        "trajectory_count": sum(trajectory_counts.values()),
        "unique_task_count": len(catalog),
        "split_counts": dict(sorted(split_counts.items())),
        "slices": by_slice,
        "benchmark_coverage": _coverage(category_counts, benchmark),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Catalog pinned Nemotron terminal tasks")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    summary = build_terminal_catalog(
        registry_path=args.registry,
        benchmark_path=args.benchmark,
        output_path=args.output,
        summary_path=args.summary,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
