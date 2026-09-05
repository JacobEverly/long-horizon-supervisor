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
    MODEL_CATALOG_URL,
    ROOT,
    SOURCE_PANEL,
    SWITCHYARD,
    _route_endpoints,
    _sha256,
    _tree_sha256,
    fetch_model_catalog,
)

V0_ROOT = ROOT / "artifacts/official/two-tier-continuation-calibration-v0"
V0_MANIFEST = V0_ROOT / "frozen-manifest-v0.json"
V0_FAILURE = V0_ROOT / "structural-failure-report-v0.json"
OUTPUT_ROOT = ROOT / "artifacts/official/two-tier-continuation-calibration-v1"
REPLACEMENT_ROOT = ROOT / "data/supervisor/terminal-bench-pro-continuation-v1"
REPLACEMENT_TASK_ID = "normalize-invoice-pdfs-to-csv"
REPLACEMENT_SELECTION_SEED = "continuation-calibration-v1|2026-09-05"
ATTEMPTED_V0_TASK_ID = "jq-github-contributor-report"
EXPECTED_V0_MANIFEST_SHA256 = (
    "6b542b96882bd95611548fade83b55075a4c636e83f4be227af705bdba50b2d6"
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    return (
        str(resolved.relative_to(ROOT.resolve()))
        if resolved.is_relative_to(ROOT.resolve())
        else str(resolved)
    )


def _v0_manifest() -> dict[str, Any]:
    if _sha256(V0_MANIFEST) != EXPECTED_V0_MANIFEST_SHA256:
        raise RuntimeError("immutable v0 continuation manifest changed")
    failure = json.loads(V0_FAILURE.read_text(encoding="utf-8"))
    if failure["diagnosis"]["model_outcome_observed"]:
        raise RuntimeError("v1 task reuse contract requires zero v0 terminal outcomes")
    return json.loads(V0_MANIFEST.read_text(encoding="utf-8"))


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
    selected_names = {row["source_task_name"] for row in panel_rows}
    if REPLACEMENT_TASK_ID in selected_names:
        raise RuntimeError("replacement task is not fresh relative to the 72-task panel")
    row = source_rows[REPLACEMENT_TASK_ID]
    metadata = tomllib.loads(row["config"])["metadata"]
    intended_stratum = {"difficulty": "medium", "category": "data-processing"}
    if {key: metadata[key] for key in intended_stratum} != intended_stratum:
        raise RuntimeError("replacement task no longer fills the intended stratum")
    rank = hashlib.sha256(
        f"{REPLACEMENT_SELECTION_SEED}|{REPLACEMENT_TASK_ID}".encode()
    ).hexdigest()
    candidates = []
    for task_id, candidate in source_rows.items():
        if task_id in selected_names:
            continue
        candidate_metadata = tomllib.loads(candidate["config"])["metadata"]
        if (
            candidate_metadata.get("difficulty"),
            candidate_metadata.get("category"),
        ) != (metadata["difficulty"], metadata["category"]):
            continue
        candidate_rank = hashlib.sha256(
            f"{REPLACEMENT_SELECTION_SEED}|{task_id}".encode()
        ).hexdigest()
        candidates.append((candidate_rank, task_id))
    if min(candidates) != (rank, REPLACEMENT_TASK_ID):
        raise RuntimeError("replacement is not the first outcome-blind fresh stratum task")
    return row, source


def materialize_replacement(
    output_root: Path = REPLACEMENT_ROOT,
) -> dict[str, Any]:
    row, source = _replacement_source_row()
    task_root = output_root / "tasks" / REPLACEMENT_TASK_ID
    lock_path = output_root / "replacement-source-lock-v1.json"
    if task_root.exists():
        if not lock_path.is_file():
            raise FileExistsError(f"unlocked replacement task exists: {task_root}")
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        expected_source_hashes = {
            "instruction_sha256": _sha256_bytes(row["instruction"].encode()),
            "config_sha256": _sha256_bytes(row["config"].encode()),
            "archive_sha256": _sha256_bytes(row["archive"]),
        }
        if any(existing.get(key) != value for key, value in expected_source_hashes.items()):
            raise RuntimeError("existing replacement source lock changed")
        if existing.get("task_tree_sha256") != _tree_sha256(task_root):
            raise RuntimeError("existing replacement task tree changed")
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
            raise RuntimeError(f"replacement task is missing {required}")
    lock = {
        "schema_version": "continuation-calibration-replacement-lock.v1",
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
        "prior_official_outcome_overlap": False,
    }
    lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return lock | {"lock_path": _portable_path(lock_path)}


def build_task_pool(
    v0_manifest: dict[str, Any], replacement: dict[str, Any]
) -> list[dict[str, Any]]:
    prior = [
        dict(task)
        for task in v0_manifest["task_selection"]["ordered_pool"]
        if task["task_id"] != ATTEMPTED_V0_TASK_ID
    ]
    if len(prior) != 15:
        raise RuntimeError("v0 pool must leave exactly 15 unattempted tasks")
    for task in prior:
        task["prior_manifest_reference_count"] = 1
        task["prior_terminal_outcome_count"] = 0
        task["snapshot_compatibility"] = (
            "task-workdir state plus static protected inputs; live task processes "
            "are rejected by action-scoped process deltas"
        )
    replacement_task = {
        "task_id": replacement["task_id"],
        "task_category": replacement["category"],
        "difficulty": replacement["difficulty"],
        "wave": "continuation-v1-replacement",
        "instruction_sha256": replacement["instruction_sha256"],
        "task_root": replacement["task_root"],
        "task_tree_sha256": replacement["task_tree_sha256"],
        "selection_rank": replacement["outcome_blind_rank"],
        "prior_official_reference_count": 0,
        "prior_terminal_outcome_count": 0,
        "snapshot_compatibility": (
            "task-workdir state plus static protected inputs; live task processes "
            "are rejected by action-scoped process deltas"
        ),
    }
    ordered = [replacement_task, *prior]
    for position, task in enumerate(ordered, start=1):
        task["position"] = position
        task["tranche"] = 1 if position <= 8 else 2
        task["prior_terminal_outcome_count"] = 0
    if len(ordered) != 16 or len({task["task_id"] for task in ordered}) != 16:
        raise RuntimeError("v1 task pool is not 16 unique tasks")
    if ATTEMPTED_V0_TASK_ID in {task["task_id"] for task in ordered}:
        raise RuntimeError("v1 task pool reused the attempted structural task")
    return ordered


def freeze(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    output_root = output_root.resolve()
    manifest_path = output_root / "frozen-manifest-v1.json"
    if manifest_path.exists():
        raise FileExistsError(f"continuation calibration v1 is frozen: {manifest_path}")
    if _route_endpoints() != EXACT_MODELS:
        raise RuntimeError("the Flash/Qwen Switchyard routes changed")
    v0_manifest = _v0_manifest()
    replacement = materialize_replacement()
    tasks = build_task_pool(v0_manifest, replacement)
    for task in tasks:
        task_root = ROOT / task["task_root"]
        if _tree_sha256(task_root) != task["task_tree_sha256"]:
            raise RuntimeError(f"task tree changed: {task['task_id']}")

    output_root.mkdir(parents=True, exist_ok=True)
    eligibility_path = output_root / "task-eligibility-report-v1.json"
    eligibility = {
        "schema_version": "continuation-task-eligibility.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "outcome_blind": True,
        "v0_manifest_sha256": EXPECTED_V0_MANIFEST_SHA256,
        "v0_failure_sha256": _sha256(V0_FAILURE),
        "attempted_v0_task_excluded": ATTEMPTED_V0_TASK_ID,
        "prior_terminal_outcome_count": 0,
        "reused_unattempted_frozen_tasks": 15,
        "fresh_replacement_tasks": 1,
        "tasks": tasks,
    }
    eligibility_path.write_text(json.dumps(eligibility, indent=2) + "\n", encoding="utf-8")
    catalog_path = output_root / "frozen-model-catalog-v1.json"
    catalog = fetch_model_catalog(MODEL_CATALOG_URL)
    catalog["schema_version"] = "continuation-model-catalog.v1"
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")

    code_paths = [
        "src/horizon_supervisor/stuck_detector_v2.py",
        "src/horizon_supervisor/benchmark/pilot_harbor.py",
        "src/horizon_supervisor/benchmark/continuation_harbor.py",
        "src/horizon_supervisor/benchmark/continuation_harbor_v1.py",
        "src/horizon_supervisor/training/freeze_continuation_calibration_v1.py",
        "src/horizon_supervisor/training/run_continuation_calibration_v1.py",
        "src/horizon_supervisor/training/analyze_continuation_calibration.py",
    ]
    for relative in code_paths:
        if not (ROOT / relative).is_file():
            raise FileNotFoundError(ROOT / relative)
    manifest = {
        "schema_version": "two-tier-continuation-calibration-manifest.v1",
        "frozen_at": datetime.now(UTC).isoformat(),
        "frozen_before_model_outcomes": True,
        "lineage": {
            "v0_manifest_sha256": EXPECTED_V0_MANIFEST_SHA256,
            "v0_failure_sha256": _sha256(V0_FAILURE),
            "v0_terminal_outcome_count": 0,
            "minimal_revision": (
                "replace absolute workspace-process rejection with pre/post "
                "agent-action process-delta tracking"
            ),
            "detector_thresholds_changed": False,
        },
        "objective": (
            "Measure natural review-to-confirmation transitions and continuation "
            "recovery without any intervention."
        ),
        "models": {
            "routes": EXACT_MODELS,
            "catalog_path": _portable_path(catalog_path),
            "catalog_sha256": _sha256(catalog_path),
        },
        "detector": v0_manifest["detector"],
        "process_reproducibility": {
            "schema_version": "action-process-delta.v1",
            "baseline": "process identities observed immediately before each action",
            "unmanaged": (
                "new non-shell process identities rooted in the task workdir, "
                "carried until exit"
            ),
            "platform_processes_observed_before_action": "reproducible harness state",
            "threshold_or_signal_changes": False,
        },
        "task_selection": {
            "eligibility_path": _portable_path(eligibility_path),
            "eligibility_sha256": _sha256(eligibility_path),
            "replacement_lock_path": replacement["lock_path"],
            "replacement_lock_sha256": _sha256(
                ROOT / replacement["lock_path"]
            ),
            "ordered_pool": tasks,
            "maximum_tasks": 16,
        },
        "execution": {
            "agent": "ProcessDeltaContinuationTerminus2",
            "max_turns": 12,
            "healthy_checkpoint_turn": 4,
            "natural_continuation_only": True,
            "interventions_forbidden": True,
            "infrastructure_retries_per_trial": 1,
            "state_fidelity": "fresh SeededDaytonaEnvironment rehydration per checkpoint",
        },
        "sampling": v0_manifest["sampling"],
        "analysis": v0_manifest["analysis"],
        "budget": {
            "original_project_baseline_usd": 49.257980597,
            "project_openrouter_spend_before_usd": 49.259005378,
            "project_cumulative_ceiling_usd": 200.0,
            "phase_a_incremental_ceiling_usd": 4.99,
            "tranche_1_incremental_ceiling_usd": 2.5,
            "per_trial_incremental_ceiling_usd": 0.5,
            "request_reserve_usd": 0.05,
            "provider_key_baseline": "record immediately before first paid v1 call",
            "provider_hard_limit_rule": (
                "hard key limit must be no more than v1 baseline usage plus $5.01"
            ),
        },
        "integrity": {
            "switchyard_sha256": _sha256(SWITCHYARD),
            "source_panel_sha256": _sha256(SOURCE_PANEL),
            "code_sha256": {
                relative: _sha256(ROOT / relative) for relative in code_paths
            },
        },
        "forbidden": v0_manifest["forbidden"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    digest = _sha256(manifest_path)
    manifest_path.with_suffix(".sha256").write_text(
        f"{digest}  {manifest_path.name}\n", encoding="utf-8"
    )
    return {"manifest_path": str(manifest_path), "manifest_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    print(json.dumps(freeze(args.output_root), indent=2))


if __name__ == "__main__":
    main()
