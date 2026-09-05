from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from horizon_supervisor.supervisor_data.materialize_terminal_bench_pro import (
    _safe_members,
)
from horizon_supervisor.training.freeze_continuation_calibration import (
    EXACT_MODELS,
    ROOT,
    SOURCE_PANEL,
    SWITCHYARD,
    _sha256,
    _tree_sha256,
)
from horizon_supervisor.training.freeze_continuation_calibration_v1 import (
    _portable_path,
    _sha256_bytes,
)

V1_ROOT = ROOT / "artifacts/official/two-tier-continuation-calibration-v1"
V1_MANIFEST = V1_ROOT / "frozen-manifest-v1.json"
V1_FAILURE = V1_ROOT / "execution-failure-report-v1.json"
EXPECTED_V1_MANIFEST_SHA256 = (
    "a5a915b6ea6b988ce73ae7b4e9ccf1da5b9c751f1450f1df39772b7b5ad08f15"
)
EXPOSED_V1_TASK_ID = "normalize-invoice-pdfs-to-csv"
REPLACEMENT_TASK_ID = "parse-bitcoin-tx-to-json"
REPLACEMENT_SELECTION_SEED = "continuation-calibration-v2|2026-09-05"
REPLACEMENT_ROOT = ROOT / "data/supervisor/terminal-bench-pro-continuation-v2"
OUTPUT_ROOT = ROOT / "artifacts/official/two-tier-continuation-calibration-v2"


def _v1_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    if _sha256(V1_MANIFEST) != EXPECTED_V1_MANIFEST_SHA256:
        raise RuntimeError("immutable v1 continuation manifest changed")
    failure = json.loads(V1_FAILURE.read_text(encoding="utf-8"))
    if (
        failure["valid_evaluation_outcome_count"] != 0
        or failure["diagnosis"]["model_comparison_allowed"] is not False
    ):
        raise RuntimeError("v2 reuse requires zero valid v1 evaluation outcomes")
    return json.loads(V1_MANIFEST.read_text(encoding="utf-8")), failure


def _replacement_source_row() -> tuple[dict[str, Any], dict[str, Any]]:
    panel_rows = [
        json.loads(line)
        for line in SOURCE_PANEL.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    source = panel_rows[0]["source"]
    info = HfApi().dataset_info(source["dataset_id"], revision=source["revision"])
    if info.sha != source["revision"]:
        raise RuntimeError("replacement source revision changed")
    parquet_path = Path(
        hf_hub_download(
            repo_id=source["dataset_id"],
            repo_type="dataset",
            filename=source["file"],
            revision=source["revision"],
        )
    )
    source_rows = {
        row["task_id"]: row
        for row in pq.read_table(
            parquet_path,
            columns=["task_id", "instruction", "config", "archive"],
        ).to_pylist()
    }
    excluded = {row["source_task_name"] for row in panel_rows} | {
        EXPOSED_V1_TASK_ID
    }
    if REPLACEMENT_TASK_ID in excluded:
        raise RuntimeError("v2 replacement is not fresh")
    row = source_rows[REPLACEMENT_TASK_ID]
    metadata = tomllib.loads(row["config"])["metadata"]
    if (metadata.get("difficulty"), metadata.get("category")) != (
        "medium",
        "data-processing",
    ):
        raise RuntimeError("v2 replacement changed stratum")
    candidates = []
    for task_id, candidate in source_rows.items():
        candidate_metadata = tomllib.loads(candidate["config"])["metadata"]
        if task_id in excluded or (
            candidate_metadata.get("difficulty"),
            candidate_metadata.get("category"),
        ) != ("medium", "data-processing"):
            continue
        candidates.append(
            (
                hashlib.sha256(
                    f"{REPLACEMENT_SELECTION_SEED}|{task_id}".encode()
                ).hexdigest(),
                task_id,
            )
        )
    expected_rank = hashlib.sha256(
        f"{REPLACEMENT_SELECTION_SEED}|{REPLACEMENT_TASK_ID}".encode()
    ).hexdigest()
    if min(candidates) != (expected_rank, REPLACEMENT_TASK_ID):
        raise RuntimeError("v2 replacement is not the outcome-blind first candidate")
    return row, source


def materialize_replacement(output_root: Path = REPLACEMENT_ROOT) -> dict[str, Any]:
    row, source = _replacement_source_row()
    task_root = output_root / "tasks" / REPLACEMENT_TASK_ID
    lock_path = output_root / "replacement-source-lock-v2.json"
    if task_root.exists():
        if not lock_path.is_file():
            raise FileExistsError(f"unlocked v2 replacement exists: {task_root}")
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if existing.get("task_tree_sha256") != _tree_sha256(task_root):
            raise RuntimeError("existing v2 replacement tree changed")
        return existing | {"lock_path": _portable_path(lock_path)}
    output_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(row["archive"]), mode="r:gz") as archive:
        members = _safe_members(archive, REPLACEMENT_TASK_ID)
        archive.extractall(output_root / "tasks", members=members, filter="data")
    for required in (
        "instruction.md",
        "task.toml",
        "environment/Dockerfile",
        "tests/test.sh",
    ):
        if not (task_root / required).is_file():
            raise RuntimeError(f"v2 replacement is missing {required}")
    lock = {
        "schema_version": "continuation-calibration-replacement-lock.v2",
        "selection_seed": REPLACEMENT_SELECTION_SEED,
        "outcome_blind_rank": hashlib.sha256(
            f"{REPLACEMENT_SELECTION_SEED}|{REPLACEMENT_TASK_ID}".encode()
        ).hexdigest(),
        "source": source,
        "task_id": REPLACEMENT_TASK_ID,
        "difficulty": "medium",
        "category": "data-processing",
        "instruction_sha256": _sha256_bytes(row["instruction"].encode()),
        "config_sha256": _sha256_bytes(row["config"].encode()),
        "archive_sha256": _sha256_bytes(row["archive"]),
        "task_tree_sha256": _tree_sha256(task_root),
        "task_root": _portable_path(task_root),
        "prior_72_task_panel_overlap": False,
        "prior_continuation_exposure": False,
    }
    lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return lock | {"lock_path": _portable_path(lock_path)}


def build_task_pool(
    v1_manifest: dict[str, Any], replacement: dict[str, Any]
) -> list[dict[str, Any]]:
    prior = [
        dict(task)
        for task in v1_manifest["task_selection"]["ordered_pool"]
        if task["task_id"] != EXPOSED_V1_TASK_ID
    ]
    if len(prior) != 15:
        raise RuntimeError("v1 pool must leave exactly 15 unexposed tasks")
    for task in prior:
        task["prior_v1_manifest_reference_count"] = 1
        task["prior_v1_terminal_outcome_count"] = 0
    replacement_task = {
        "task_id": replacement["task_id"],
        "task_category": replacement["category"],
        "difficulty": replacement["difficulty"],
        "wave": "continuation-v2-replacement",
        "instruction_sha256": replacement["instruction_sha256"],
        "task_root": replacement["task_root"],
        "task_tree_sha256": replacement["task_tree_sha256"],
        "selection_rank": replacement["outcome_blind_rank"],
        "prior_official_reference_count": 0,
        "prior_terminal_outcome_count": 0,
        "prior_v1_terminal_outcome_count": 0,
        "snapshot_compatibility": (
            "task-workdir state plus static protected inputs; live task processes "
            "are rejected by action-scoped process deltas"
        ),
    }
    ordered = [replacement_task, *prior]
    for position, task in enumerate(ordered, start=1):
        task["position"] = position
        task["tranche"] = 1 if position <= 8 else 2
    if len(ordered) != 16 or len({task["task_id"] for task in ordered}) != 16:
        raise RuntimeError("v2 task pool is not 16 unique tasks")
    if EXPOSED_V1_TASK_ID in {task["task_id"] for task in ordered}:
        raise RuntimeError("v2 reused the v1-exposed task")
    return ordered


def freeze(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    output_root = output_root.resolve()
    manifest_path = output_root / "frozen-manifest-v2.json"
    if manifest_path.exists():
        raise FileExistsError(f"continuation calibration v2 is frozen: {manifest_path}")
    v1_manifest, _ = _v1_inputs()
    if v1_manifest["models"]["routes"] != EXACT_MODELS:
        raise RuntimeError("v1 routes no longer match exact models")
    replacement = materialize_replacement()
    tasks = build_task_pool(v1_manifest, replacement)
    for task in tasks:
        task_root = ROOT / task["task_root"]
        if _tree_sha256(task_root) != task["task_tree_sha256"]:
            raise RuntimeError(f"v2 task tree changed: {task['task_id']}")

    output_root.mkdir(parents=True, exist_ok=True)
    eligibility_path = output_root / "task-eligibility-report-v2.json"
    eligibility = {
        "schema_version": "continuation-task-eligibility.v2",
        "created_at": datetime.now(UTC).isoformat(),
        "outcome_blind": True,
        "v1_manifest_sha256": EXPECTED_V1_MANIFEST_SHA256,
        "v1_failure_sha256": _sha256(V1_FAILURE),
        "exposed_v1_task_excluded": EXPOSED_V1_TASK_ID,
        "prior_valid_evaluation_outcome_count": 0,
        "reused_unexposed_frozen_tasks": 15,
        "fresh_replacement_tasks": 1,
        "tasks": tasks,
    }
    eligibility_path.write_text(json.dumps(eligibility, indent=2) + "\n")
    v1_catalog_path = ROOT / v1_manifest["models"]["catalog_path"]
    if _sha256(v1_catalog_path) != v1_manifest["models"]["catalog_sha256"]:
        raise RuntimeError("v1 frozen model catalog changed")
    catalog_path = output_root / "frozen-model-catalog-v2.json"
    catalog = json.loads(v1_catalog_path.read_text(encoding="utf-8"))
    catalog["schema_version"] = "continuation-model-catalog.v2"
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n")

    code_paths = [
        "src/horizon_supervisor/stuck_detector_v2.py",
        "src/horizon_supervisor/benchmark/pilot_harbor.py",
        "src/horizon_supervisor/benchmark/continuation_harbor.py",
        "src/horizon_supervisor/benchmark/continuation_harbor_v1.py",
        "src/horizon_supervisor/training/analyze_continuation_calibration.py",
        "src/horizon_supervisor/training/run_stuck_pilot.py",
        "src/horizon_supervisor/training/run_continuation_calibration.py",
        "src/horizon_supervisor/training/run_continuation_calibration_v1.py",
        "src/horizon_supervisor/training/freeze_continuation_calibration_v2.py",
        "src/horizon_supervisor/training/run_continuation_calibration_v2.py",
    ]
    for relative in code_paths:
        if not (ROOT / relative).is_file():
            raise FileNotFoundError(ROOT / relative)
    manifest = {
        "schema_version": "two-tier-continuation-calibration-manifest.v2",
        "frozen_at": datetime.now(UTC).isoformat(),
        "frozen_before_model_outcomes": True,
        "lineage": {
            "v1_manifest_sha256": EXPECTED_V1_MANIFEST_SHA256,
            "v1_failure_sha256": _sha256(V1_FAILURE),
            "v1_valid_evaluation_outcome_count": 0,
            "v1_exposed_task_excluded": EXPOSED_V1_TASK_ID,
            "minimal_revision": (
                "pass an explicit provider-free checkpoint-replay environment, "
                "journal completed trials before replay, and select one fresh "
                "outcome-blind same-stratum replacement"
            ),
            "detector_thresholds_changed": False,
            "analysis_gates_changed": False,
            "models_changed": False,
        },
        "objective": v1_manifest["objective"],
        "models": {
            "routes": EXACT_MODELS,
            "catalog_path": _portable_path(catalog_path),
            "catalog_sha256": _sha256(catalog_path),
        },
        "detector": v1_manifest["detector"],
        "process_reproducibility": v1_manifest["process_reproducibility"],
        "task_selection": {
            "eligibility_path": _portable_path(eligibility_path),
            "eligibility_sha256": _sha256(eligibility_path),
            "replacement_lock_path": replacement["lock_path"],
            "replacement_lock_sha256": _sha256(ROOT / replacement["lock_path"]),
            "ordered_pool": tasks,
            "maximum_tasks": 16,
        },
        "execution": v1_manifest["execution"]
        | {
            "checkpoint_replay_provider_calls": 0,
            "completed_trial_journal_before_replay": True,
        },
        "sampling": v1_manifest["sampling"],
        "analysis": v1_manifest["analysis"],
        "budget": {
            "original_project_baseline_usd": 49.257980597,
            "project_openrouter_spend_before_usd": 49.308541986,
            "project_cumulative_ceiling_usd": 200.0,
            "phase_a_incremental_ceiling_usd": 4.93,
            "tranche_1_incremental_ceiling_usd": 2.45,
            "per_trial_incremental_ceiling_usd": 0.5,
            "request_reserve_usd": 0.05,
            "provider_key_baseline": "record immediately before first paid v2 call",
            "provider_hard_limit_rule": (
                "hard key limit must be no more than v2 baseline usage plus $4.95"
            ),
        },
        "integrity": {
            "switchyard_sha256": _sha256(SWITCHYARD),
            "source_panel_sha256": _sha256(SOURCE_PANEL),
            "code_sha256": {
                relative: _sha256(ROOT / relative) for relative in code_paths
            },
        },
        "forbidden": v1_manifest["forbidden"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    digest = _sha256(manifest_path)
    manifest_path.with_suffix(".sha256").write_text(
        f"{digest}  {manifest_path.name}\n"
    )
    return {"manifest_path": str(manifest_path), "manifest_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    print(json.dumps(freeze(args.output_root), indent=2))


if __name__ == "__main__":
    main()
