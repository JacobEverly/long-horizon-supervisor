from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

DEFAULT_REGISTRY = Path("data/supervisor/source-registry-v0.json")
DEFAULT_BENCHMARK = Path("benchmarks/terminal-bench-2.1-gate7.json")
DEFAULT_OUTPUT = Path("data/supervisor/terminal-bench-pro-panel-v0.jsonl")
DEFAULT_SUMMARY = Path("data/supervisor/terminal-bench-pro-panel-v0-summary.json")
SOURCE_ID = "terminal-bench-pro-public"
SOURCE_FILE = "Terminal_Bench_Pro_Public-200.parquet"
SELECTION_SEED = "terminal-bench-pro-panel-v0|2026-08-25"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source(registry: dict[str, Any]) -> dict[str, Any]:
    for source in registry["sources"]:
        if source["source_id"] == SOURCE_ID:
            return source
    raise ValueError(f"missing source {SOURCE_ID!r}")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode())


def _rank(source_task_name: str) -> str:
    return _sha256_text(f"{SELECTION_SEED}|{source_task_name}")


def build_terminal_bench_pro_panel(
    registry_path: Path = DEFAULT_REGISTRY,
    benchmark_path: Path = DEFAULT_BENCHMARK,
    output_path: Path = DEFAULT_OUTPUT,
    summary_path: Path = DEFAULT_SUMMARY,
    *,
    waves: int = 4,
) -> dict[str, Any]:
    if waves <= 0:
        raise ValueError("waves must be positive")
    source = _source(_load_json(registry_path))
    benchmark = _load_json(benchmark_path)
    info = HfApi().dataset_info(source["dataset_id"], revision=source["revision"])
    if info.sha != source["revision"]:
        raise RuntimeError(
            f"revision mismatch: expected {source['revision']}, got {info.sha}"
        )
    parquet_path = Path(
        hf_hub_download(
            repo_id=source["dataset_id"],
            repo_type="dataset",
            filename=SOURCE_FILE,
            revision=source["revision"],
        )
    )
    table = pq.read_table(
        parquet_path, columns=["task_id", "instruction", "config", "archive"]
    )

    source_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in table.to_pylist():
        metadata = tomllib.loads(row["config"])["metadata"]
        difficulty = metadata["difficulty"]
        category = metadata["category"]
        source_rows[(difficulty, category)].append(
            {
                "source_task_name": row["task_id"],
                "instruction": row["instruction"],
                "difficulty": difficulty,
                "category": category,
                "instruction_sha256": _sha256_text(row["instruction"]),
                "config_sha256": _sha256_text(row["config"]),
                "archive_sha256": _sha256_bytes(row["archive"]),
            }
        )

    benchmark_strata = Counter(
        (task["difficulty"], task["category"]) for task in benchmark["tasks"]
    )
    selected = []
    covered_benchmark_seats = 0
    missing_strata = []
    for (difficulty, category), seats in sorted(benchmark_strata.items()):
        pool = source_rows.get((difficulty, category), [])
        if not pool:
            missing_strata.append(
                {
                    "difficulty": difficulty,
                    "category": category,
                    "benchmark_seats": seats,
                }
            )
            continue
        required = seats * waves
        if len(pool) < required:
            raise ValueError(
                f"{difficulty}/{category} has {len(pool)} tasks but needs {required}"
            )
        covered_benchmark_seats += seats
        ranked = sorted(
            pool,
            key=lambda row: (
                _rank(row["source_task_name"]),
                row["source_task_name"],
            ),
        )
        for index, row in enumerate(ranked[:required]):
            wave = index % waves + 1
            task_id = _sha256_text(
                f"{source['dataset_id']}@{source['revision']}|{row['source_task_name']}"
            )
            selected.append(
                {
                    "schema_version": "terminal-bench-pro-panel.v0",
                    "task_id": task_id,
                    "source_task_name": row["source_task_name"],
                    "instruction": row["instruction"],
                    "instruction_sha256": row["instruction_sha256"],
                    "difficulty": difficulty,
                    "category": category,
                    "wave": wave,
                    "selection_rank": _rank(row["source_task_name"]),
                    "execution_lock": {
                        "config_sha256": row["config_sha256"],
                        "archive_sha256": row["archive_sha256"],
                    },
                    "source": {
                        "source_id": source["source_id"],
                        "dataset_id": source["dataset_id"],
                        "revision": source["revision"],
                        "file": SOURCE_FILE,
                        "license": source["license"],
                    },
                    "intended_use": "matched_rollout_development_only",
                }
            )

    selected.sort(
        key=lambda row: (
            row["wave"],
            row["difficulty"],
            row["category"],
            row["selection_rank"],
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    final_task_names = {task["name"] for task in benchmark["tasks"]}
    selected_names = {row["source_task_name"] for row in selected}
    wave_counts = Counter(row["wave"] for row in selected)
    category_counts = Counter(row["category"] for row in selected)
    difficulty_counts = Counter(row["difficulty"] for row in selected)
    wave_strata = Counter(
        (row["wave"], row["difficulty"], row["category"]) for row in selected
    )
    summary = {
        "schema_version": "terminal-bench-pro-panel-summary.v0",
        "selection_seed": SELECTION_SEED,
        "source": {
            "source_id": source["source_id"],
            "dataset_id": source["dataset_id"],
            "revision": source["revision"],
            "source_row_count": table.num_rows,
        },
        "final_benchmark": {
            "selection_id": benchmark["selection_id"],
            "task_count": len(benchmark["tasks"]),
            "covered_seats": covered_benchmark_seats,
            "coverage_rate": covered_benchmark_seats / len(benchmark["tasks"]),
            "missing_strata": missing_strata,
        },
        "waves": waves,
        "record_count": len(selected),
        "wave_counts": {str(key): value for key, value in sorted(wave_counts.items())},
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "wave_stratum_counts": {
            f"wave-{wave}|{difficulty}|{category}": count
            for (wave, difficulty, category), count in sorted(wave_strata.items())
        },
        "exact_final_task_name_overlap_count": len(selected_names & final_task_names),
        "exact_final_task_name_overlaps": sorted(selected_names & final_task_names),
        "selection_contract": (
            "Each wave contains one development task per covered frozen benchmark "
            "seat, preserving covered difficulty/category proportions."
        ),
        "interpretation_guard": (
            "This public development panel covers 18 of 30 frozen benchmark seats. "
            "It may select a routing recipe but cannot replace final Terminal-Bench 2.1."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build matched rollout development panel")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--waves", type=int, default=4)
    args = parser.parse_args()
    summary = build_terminal_bench_pro_panel(
        registry_path=args.registry,
        benchmark_path=args.benchmark,
        output_path=args.output,
        summary_path=args.summary,
        waves=args.waves,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
