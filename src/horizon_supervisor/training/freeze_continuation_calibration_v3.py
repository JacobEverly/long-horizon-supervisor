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

V2_ROOT = ROOT / "artifacts/official/two-tier-continuation-calibration-v2"
V2_MANIFEST = V2_ROOT / "frozen-manifest-v2.json"
V2_FAILURE = V2_ROOT / "structural-failure-report-v2.json"
EXPECTED_V2_MANIFEST_SHA256 = (
    "0ad0576bcde8abc6ae372f187c19202d8cc889cb21ea1fabe2ef5d4661d04fef"
)
EXPOSED_TASK_IDS = {
    "normalize-invoice-pdfs-to-csv",
    "parse-bitcoin-tx-to-json",
}
REPLACEMENT_SELECTION_SEED = "continuation-calibration-v3|2026-09-05"
EXPANSION_SELECTION_SEED = "continuation-calibration-v3-tranche3|2026-09-05"
REPLACEMENT_SPEC = (
    "summarize-api-log-status-metrics",
    "medium",
    "data-processing",
)
EXPANSION_SPECS = (
    ("craft-binary-message-file", "medium", "data-processing"),
    ("mcts-tictactoe-ai-implementation", "medium", "games"),
    ("stabilize-neural-network-training", "medium", "machine-learning"),
    ("compute-symbolic-eigenpairs-3x3-matrix", "medium", "scientific-computing"),
    ("fix-xor-neural-network-instability", "hard", "debugging"),
    ("implement-connect-four-mcts", "hard", "games"),
    ("retrieve-vault-root-token", "hard", "security"),
    ("setup-ubuntu-vm-ssh-key-auth", "hard", "system-administration"),
)
FRESH_ROOT = ROOT / "data/supervisor/terminal-bench-pro-continuation-v3"
OUTPUT_ROOT = ROOT / "artifacts/official/two-tier-continuation-calibration-v3"


def _v2_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    if _sha256(V2_MANIFEST) != EXPECTED_V2_MANIFEST_SHA256:
        raise RuntimeError("immutable v2 continuation manifest changed")
    failure = json.loads(V2_FAILURE.read_text(encoding="utf-8"))
    if (
        failure["terminal_outcome_count"] != 0
        or failure["diagnosis"]["model_outcome_observed"] is not False
    ):
        raise RuntimeError("v3 reuse requires zero terminal v2 outcomes")
    return json.loads(V2_MANIFEST.read_text(encoding="utf-8")), failure


def _pinned_source() -> tuple[dict[str, dict[str, Any]], dict[str, Any], set[str]]:
    panel_rows = [
        json.loads(line)
        for line in SOURCE_PANEL.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    source = panel_rows[0]["source"]
    info = HfApi().dataset_info(source["dataset_id"], revision=source["revision"])
    if info.sha != source["revision"]:
        raise RuntimeError("v3 source revision changed")
    parquet_path = Path(
        hf_hub_download(
            repo_id=source["dataset_id"],
            repo_type="dataset",
            filename=source["file"],
            revision=source["revision"],
        )
    )
    rows = {
        row["task_id"]: row
        for row in pq.read_table(
            parquet_path,
            columns=["task_id", "instruction", "config", "archive"],
        ).to_pylist()
    }
    panel_names = {row["source_task_name"] for row in panel_rows}
    return rows, source, panel_names


def _rank(seed: str, task_id: str, stratum: tuple[str, str] | None = None) -> str:
    pieces = [seed]
    if stratum is not None:
        pieces.extend(stratum)
    pieces.append(task_id)
    return hashlib.sha256("|".join(pieces).encode()).hexdigest()


def _validate_selection(
    rows: dict[str, dict[str, Any]], panel_names: set[str]
) -> list[tuple[dict[str, Any], str, str]]:
    excluded = panel_names | EXPOSED_TASK_IDS
    task_id, difficulty, category = REPLACEMENT_SPEC
    candidates = []
    for candidate_id, row in rows.items():
        metadata = tomllib.loads(row["config"])["metadata"]
        if candidate_id in excluded or (
            metadata.get("difficulty"), metadata.get("category")
        ) != (difficulty, category):
            continue
        candidates.append((_rank(REPLACEMENT_SELECTION_SEED, candidate_id), candidate_id))
    selected_rank, selected_id = min(candidates)
    if selected_id != task_id:
        raise RuntimeError("v3 replacement is not outcome-blind first candidate")
    selected = [(rows[task_id], selected_rank, "replacement")]
    excluded.add(task_id)

    for expected_id, expected_difficulty, expected_category in EXPANSION_SPECS:
        stratum = (expected_difficulty, expected_category)
        candidates = []
        for candidate_id, row in rows.items():
            metadata = tomllib.loads(row["config"])["metadata"]
            if candidate_id in excluded or (
                metadata.get("difficulty"), metadata.get("category")
            ) != stratum:
                continue
            candidates.append(
                (
                    _rank(EXPANSION_SELECTION_SEED, candidate_id, stratum),
                    candidate_id,
                )
            )
        selected_rank, selected_id = min(candidates)
        if selected_id != expected_id:
            raise RuntimeError(
                f"v3 expansion is not outcome-blind first for {stratum}"
            )
        selected.append((rows[selected_id], selected_rank, "expansion"))
        excluded.add(selected_id)
    if len(selected) != 9 or len({row[0]["task_id"] for row in selected}) != 9:
        raise RuntimeError("v3 fresh selection is not nine unique tasks")
    return selected


def materialize_fresh_tasks(output_root: Path = FRESH_ROOT) -> dict[str, Any]:
    rows, source, panel_names = _pinned_source()
    selected = _validate_selection(rows, panel_names)
    lock_path = output_root / "task-source-lock-v3.json"
    if lock_path.is_file():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        for task in lock["tasks"]:
            if _tree_sha256(ROOT / task["task_root"]) != task["task_tree_sha256"]:
                raise RuntimeError(f"existing v3 task tree changed: {task['task_id']}")
        return lock | {"lock_path": _portable_path(lock_path)}

    output_root.mkdir(parents=True, exist_ok=True)
    task_locks = []
    for row, selection_rank, role in selected:
        task_id = row["task_id"]
        task_root = output_root / "tasks" / task_id
        if task_root.exists():
            raise FileExistsError(f"unlocked v3 task exists: {task_root}")
        with tarfile.open(fileobj=io.BytesIO(row["archive"]), mode="r:gz") as archive:
            members = _safe_members(archive, task_id)
            archive.extractall(output_root / "tasks", members=members, filter="data")
        for required in (
            "instruction.md",
            "task.toml",
            "environment/Dockerfile",
            "tests/test.sh",
        ):
            if not (task_root / required).is_file():
                raise RuntimeError(f"v3 task {task_id} is missing {required}")
        metadata = tomllib.loads(row["config"])["metadata"]
        task_locks.append(
            {
                "task_id": task_id,
                "role": role,
                "difficulty": metadata["difficulty"],
                "category": metadata["category"],
                "outcome_blind_rank": selection_rank,
                "instruction_sha256": _sha256_bytes(row["instruction"].encode()),
                "config_sha256": _sha256_bytes(row["config"].encode()),
                "archive_sha256": _sha256_bytes(row["archive"]),
                "task_tree_sha256": _tree_sha256(task_root),
                "task_root": _portable_path(task_root),
                "prior_72_task_panel_overlap": False,
                "prior_continuation_exposure": False,
            }
        )
    lock = {
        "schema_version": "continuation-calibration-task-source-lock.v3",
        "replacement_selection_seed": REPLACEMENT_SELECTION_SEED,
        "expansion_selection_seed": EXPANSION_SELECTION_SEED,
        "source": source,
        "tasks": task_locks,
    }
    lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return lock | {"lock_path": _portable_path(lock_path)}


def _fresh_task(lock: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": lock["task_id"],
        "task_category": lock["category"],
        "difficulty": lock["difficulty"],
        "wave": f"continuation-v3-{lock['role']}",
        "instruction_sha256": lock["instruction_sha256"],
        "task_root": lock["task_root"],
        "task_tree_sha256": lock["task_tree_sha256"],
        "selection_rank": lock["outcome_blind_rank"],
        "prior_official_reference_count": 0,
        "prior_terminal_outcome_count": 0,
        "snapshot_compatibility": (
            "task-workdir state plus static protected inputs; live task processes "
            "are rejected by action-scoped process deltas after frozen "
            "terminal-service exclusions"
        ),
    }


def build_task_pool(v2_manifest: dict[str, Any], fresh: dict[str, Any]) -> list[dict[str, Any]]:
    prior = [
        dict(task)
        for task in v2_manifest["task_selection"]["ordered_pool"]
        if task["task_id"] not in EXPOSED_TASK_IDS
    ]
    if len(prior) != 15:
        raise RuntimeError("v2 pool must leave exactly 15 unexposed tasks")
    for task in prior:
        task["prior_v2_manifest_reference_count"] = 1
        task["prior_v2_terminal_outcome_count"] = 0
    locked = {task["task_id"]: task for task in fresh["tasks"]}
    replacement = _fresh_task(locked[REPLACEMENT_SPEC[0]])
    expansion = [_fresh_task(locked[spec[0]]) for spec in EXPANSION_SPECS]
    ordered = [replacement, *prior, *expansion]
    for position, task in enumerate(ordered, start=1):
        task["position"] = position
        task["tranche"] = 1 if position <= 8 else 2 if position <= 16 else 3
    if len(ordered) != 24 or len({task["task_id"] for task in ordered}) != 24:
        raise RuntimeError("v3 pool is not 24 unique tasks")
    if EXPOSED_TASK_IDS.intersection(task["task_id"] for task in ordered):
        raise RuntimeError("v3 reused an exposed continuation task")
    return ordered


def freeze(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    output_root = output_root.resolve()
    manifest_path = output_root / "frozen-manifest-v3.json"
    if manifest_path.exists():
        raise FileExistsError(f"continuation calibration v3 is frozen: {manifest_path}")
    v2_manifest, _ = _v2_inputs()
    if v2_manifest["models"]["routes"] != EXACT_MODELS:
        raise RuntimeError("v2 routes no longer match exact models")
    fresh = materialize_fresh_tasks()
    tasks = build_task_pool(v2_manifest, fresh)
    for task in tasks:
        if _tree_sha256(ROOT / task["task_root"]) != task["task_tree_sha256"]:
            raise RuntimeError(f"v3 task tree changed: {task['task_id']}")

    output_root.mkdir(parents=True, exist_ok=True)
    eligibility_path = output_root / "task-eligibility-report-v3.json"
    eligibility = {
        "schema_version": "continuation-task-eligibility.v3",
        "created_at": datetime.now(UTC).isoformat(),
        "outcome_blind": True,
        "v2_manifest_sha256": EXPECTED_V2_MANIFEST_SHA256,
        "v2_failure_sha256": _sha256(V2_FAILURE),
        "exposed_tasks_excluded": sorted(EXPOSED_TASK_IDS),
        "prior_terminal_outcome_count": 0,
        "reused_unexposed_frozen_tasks": 15,
        "fresh_replacement_tasks": 1,
        "fresh_prefrozen_expansion_tasks": 8,
        "tasks": tasks,
    }
    eligibility_path.write_text(json.dumps(eligibility, indent=2) + "\n")
    v2_catalog_path = ROOT / v2_manifest["models"]["catalog_path"]
    if _sha256(v2_catalog_path) != v2_manifest["models"]["catalog_sha256"]:
        raise RuntimeError("v2 frozen model catalog changed")
    catalog_path = output_root / "frozen-model-catalog-v3.json"
    catalog = json.loads(v2_catalog_path.read_text(encoding="utf-8"))
    catalog["schema_version"] = "continuation-model-catalog.v3"
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n")

    code_paths = [
        "src/horizon_supervisor/stuck_detector_v2.py",
        "src/horizon_supervisor/benchmark/pilot_harbor.py",
        "src/horizon_supervisor/benchmark/continuation_harbor.py",
        "src/horizon_supervisor/benchmark/continuation_harbor_v1.py",
        "src/horizon_supervisor/benchmark/continuation_harbor_v2.py",
        "src/horizon_supervisor/training/analyze_continuation_calibration.py",
        "src/horizon_supervisor/training/run_stuck_pilot.py",
        "src/horizon_supervisor/training/run_continuation_calibration.py",
        "src/horizon_supervisor/training/run_continuation_calibration_v1.py",
        "src/horizon_supervisor/training/run_continuation_calibration_v2.py",
        "src/horizon_supervisor/training/freeze_continuation_calibration_v3.py",
        "src/horizon_supervisor/training/run_continuation_calibration_v3.py",
    ]
    for relative in code_paths:
        if not (ROOT / relative).is_file():
            raise FileNotFoundError(ROOT / relative)
    manifest = {
        "schema_version": "two-tier-continuation-calibration-manifest.v3",
        "frozen_at": datetime.now(UTC).isoformat(),
        "frozen_before_model_outcomes": True,
        "lineage": {
            "v2_manifest_sha256": EXPECTED_V2_MANIFEST_SHA256,
            "v2_failure_sha256": _sha256(V2_FAILURE),
            "v2_terminal_outcome_count": 0,
            "v2_exposed_task_excluded": "parse-bitcoin-tx-to-json",
            "minimal_revision": "add only tail to the harness-only process allowlist",
            "detector_thresholds_changed": False,
            "analysis_gates_changed": False,
            "models_changed": False,
        },
        "objective": v2_manifest["objective"],
        "models": {
            "routes": EXACT_MODELS,
            "catalog_path": _portable_path(catalog_path),
            "catalog_sha256": _sha256(catalog_path),
        },
        "detector": v2_manifest["detector"],
        "process_reproducibility": v2_manifest["process_reproducibility"]
        | {
            "schema_version": "action-process-delta.v2",
            "frozen_harness_only_additions": ["tail"],
        },
        "task_selection": {
            "eligibility_path": _portable_path(eligibility_path),
            "eligibility_sha256": _sha256(eligibility_path),
            "task_source_lock_path": fresh["lock_path"],
            "task_source_lock_sha256": _sha256(ROOT / fresh["lock_path"]),
            "ordered_pool": tasks,
            "maximum_tasks": 24,
        },
        "execution": v2_manifest["execution"]
        | {"agent": "HarnessFilteredContinuationTerminus2"},
        "sampling": {
            "tranches": [
                {"tranche": 1, "task_positions": list(range(1, 9))},
                {"tranche": 2, "task_positions": list(range(9, 17))},
                {"tranche": 3, "task_positions": list(range(17, 25))},
            ],
            "rule": (
                "Run tranche 1; stop only if every readiness gate passes. "
                "Otherwise run tranche 2, then the frozen difficulty-balanced "
                "tranche 3, subject to frozen budget ceilings."
            ),
            "outcome_blind_task_order": True,
            "tranche_3_difficulty_mix": {"medium": 4, "hard": 4},
            "tranche_3_category_balance": True,
        },
        "analysis": v2_manifest["analysis"],
        "budget": {
            "original_project_baseline_usd": 49.257980597,
            "project_openrouter_spend_before_usd": 49.308781292,
            "project_cumulative_ceiling_usd": 200.0,
            "phase_a_incremental_ceiling_usd": 4.93,
            "tranche_1_incremental_ceiling_usd": 1.6,
            "tranche_2_incremental_ceiling_usd": 3.25,
            "per_trial_incremental_ceiling_usd": 0.5,
            "request_reserve_usd": 0.05,
            "provider_key_baseline": "record immediately before first paid v3 call",
            "provider_hard_limit_rule": (
                "hard key limit must be no more than v3 baseline usage plus $4.95"
            ),
        },
        "integrity": {
            "switchyard_sha256": _sha256(SWITCHYARD),
            "source_panel_sha256": _sha256(SOURCE_PANEL),
            "code_sha256": {
                relative: _sha256(ROOT / relative) for relative in code_paths
            },
        },
        "forbidden": v2_manifest["forbidden"],
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
