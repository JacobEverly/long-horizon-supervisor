from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from horizon_supervisor.supervisor_data.pivot_checkpoints import task_leakage_sha256

DEFAULT_REGISTRY = Path("data/supervisor/source-registry-v0.json")
DEFAULT_TASKS = Path("data/supervisor/swe-pivot-transfer-tasks-v0.jsonl")
DEFAULT_OUTPUT = Path("data/supervisor/swe-pivot-overlap-audit-v0.json")
VERIFIED_FILE = "data/test-00000-of-00001.parquet"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source(registry: dict[str, Any], source_id: str) -> dict[str, Any]:
    for item in registry["sources"]:
        if item["source_id"] == source_id:
            return item
    raise ValueError(f"missing source {source_id!r}")


def build_swe_overlap_audit(
    registry_path: Path = DEFAULT_REGISTRY,
    tasks_path: Path = DEFAULT_TASKS,
    output_path: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    registry = _load_json(registry_path)
    verified_source = _source(registry, "swe-bench-verified")
    verified_path = Path(
        hf_hub_download(
            repo_id=verified_source["dataset_id"],
            repo_type="dataset",
            filename=VERIFIED_FILE,
            revision=verified_source["revision"],
        )
    )
    verified_rows = pq.read_table(
        verified_path, columns=["instance_id", "problem_statement"]
    ).to_pylist()
    verified_ids = {row["instance_id"] for row in verified_rows}
    verified_by_leakage = {
        task_leakage_sha256(row["problem_statement"]): row["instance_id"]
        for row in verified_rows
    }

    source_rows = []
    with tasks_path.open(encoding="utf-8") as handle:
        for line in handle:
            source_rows.append(json.loads(line))
    source_task_names = {row["source_task_name"] for row in source_rows}
    exact_id_overlap = sorted(source_task_names & verified_ids)
    normalized_text_overlap = sorted(
        {
            (row["source_task_name"], verified_by_leakage[row["leakage_group"]])
            for row in source_rows
            if row["leakage_group"] in verified_by_leakage
        }
    )
    report = {
        "schema_version": "swe-pivot-overlap-audit.v0",
        "created_at": "2026-08-25",
        "training_source": {
            "source_id": "nemotron-swe-pivot-v1",
            "task_record_count": len(source_rows),
            "source_task_id_count": len(source_task_names),
        },
        "held_out_source": {
            "source_id": "swe-bench-verified",
            "revision": verified_source["revision"],
            "task_count": len(verified_rows),
        },
        "exact_instance_id_overlap_count": len(exact_id_overlap),
        "exact_instance_id_overlaps": exact_id_overlap,
        "normalized_problem_text_overlap_count": len(normalized_text_overlap),
        "normalized_problem_text_overlaps": normalized_text_overlap,
        "training_allowed": not exact_id_overlap and not normalized_text_overlap,
        "note": (
            "Repository overlap alone is expected in SWE data; exact instance and "
            "normalized problem text are the exclusion keys."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit SWE pivot against Verified")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_swe_overlap_audit(args.registry, args.tasks, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
