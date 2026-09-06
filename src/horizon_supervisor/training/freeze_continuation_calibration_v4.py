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
from horizon_supervisor.training.freeze_continuation_calibration_v3 import (
    EXPOSED_TASK_IDS,
    _pinned_source,
)

V3_ROOT = ROOT / "artifacts/official/two-tier-continuation-calibration-v3"
V3_MANIFEST = V3_ROOT / "frozen-manifest-v3.json"
V3_REPORT = V3_ROOT / "calibration-report-v3.json"
V3_LEDGER = V3_ROOT / "execution-ledger-v3.json"
V3_OUTCOMES = V3_ROOT / "natural-continuation-outcomes-v3.jsonl"
V3_CHECKPOINTS = V3_ROOT / "checkpoint-bank-index-v3.jsonl"
V3_CENSORING = V3_ROOT / "censoring-analysis-v3.json"

EXPECTED_V3_MANIFEST_SHA256 = (
    "ed056a2a9fbaad3354a43ee2113a49c877b0d5c9bbbe35069426787dc996964e"
)
EXPECTED_V3_REPORT_SHA256 = (
    "d7d75808837c90d7ecfe4a8bda0393cf6bae139c24425c5aac52e226e261f6c0"
)
EXPECTED_V3_LEDGER_SHA256 = (
    "51def4f861b81e7d5c1e4c0d8c1cd5f01d1e00f7de78939fdcfd17fac703dc9f"
)
EXPECTED_V3_OUTCOMES_SHA256 = (
    "44df1f96136adbd17be79a3cec8aa73ddf7fd43e8d195d9828971d8d43f999f0"
)
EXPECTED_V3_CHECKPOINTS_SHA256 = (
    "9e6737a353a3e785562e1cd046ef312e483bc93b7938d04ee53bd7aac8157739"
)
EXPECTED_V3_CENSORING_SHA256 = (
    "c83dbb9690f56907c514db36e8f226d2c100743164a68a014363d2f67ea63213"
)

SELECTION_SEED = "continuation-calibration-v4|2026-09-05"
CATEGORY_QUOTAS = {
    "data-processing": 5,
    "debugging": 5,
    "games": 5,
    "machine-learning": 2,
    "scientific-computing": 4,
    "security": 3,
}
DISALLOWED_TAGS = frozenset({"distributed-training", "mongodb", "qemu"})
MAX_CPUS = 2
MAX_MEMORY_MB = 4_096
FRESH_ROOT = ROOT / "data/supervisor/terminal-bench-pro-continuation-v4"
OUTPUT_ROOT = ROOT / "artifacts/official/two-tier-continuation-calibration-v4"


def _v3_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    expected = {
        V3_MANIFEST: EXPECTED_V3_MANIFEST_SHA256,
        V3_REPORT: EXPECTED_V3_REPORT_SHA256,
        V3_LEDGER: EXPECTED_V3_LEDGER_SHA256,
        V3_OUTCOMES: EXPECTED_V3_OUTCOMES_SHA256,
        V3_CHECKPOINTS: EXPECTED_V3_CHECKPOINTS_SHA256,
        V3_CENSORING: EXPECTED_V3_CENSORING_SHA256,
    }
    for path, digest in expected.items():
        if not path.is_file() or _sha256(path) != digest:
            raise RuntimeError(f"immutable v3 input changed: {path}")
    manifest = json.loads(V3_MANIFEST.read_text(encoding="utf-8"))
    report = json.loads(V3_REPORT.read_text(encoding="utf-8"))
    ledger = json.loads(V3_LEDGER.read_text(encoding="utf-8"))
    censoring = json.loads(V3_CENSORING.read_text(encoding="utf-8"))
    if (
        report["trajectory_count"] != 48
        or report["gate_passed"] is not False
        or report["tiers"]["confirmed_stuck"]["task_count"] != 2
        or ledger["status"] != "complete"
        or ledger["openrouter"]["project_spend_after_usd"] != 50.736977388999996
    ):
        raise RuntimeError("v3 terminal result no longer matches v4 lineage")
    if (
        censoring["invalid_exception_types"].get("OutputLengthExceeded") != 25
        or censoring["diagnosis"]["detector_threshold_change_justified"] is not False
    ):
        raise RuntimeError("v3 censoring diagnosis no longer supports v4")
    return manifest, report, censoring


def _rank(task_id: str, category: str) -> str:
    return hashlib.sha256(
        f"{SELECTION_SEED}|hard|{category}|{task_id}".encode()
    ).hexdigest()


def static_checkpoint_compatibility(config: str) -> tuple[bool, list[str]]:
    parsed = tomllib.loads(config)
    metadata = parsed["metadata"]
    environment = parsed["environment"]
    reasons: list[str] = []
    if metadata.get("difficulty") != "hard":
        reasons.append("not_hard")
    if metadata.get("category") not in CATEGORY_QUOTAS:
        reasons.append("category_not_targeted")
    tags = {str(tag).lower() for tag in metadata.get("tags", [])}
    if tags.intersection(DISALLOWED_TAGS):
        reasons.append("persistent_or_distributed_runtime_tag")
    if int(environment.get("cpus", 0)) > MAX_CPUS:
        reasons.append("cpu_limit_exceeds_static_cap")
    if int(environment.get("memory_mb", 0)) > MAX_MEMORY_MB:
        reasons.append("memory_limit_exceeds_static_cap")
    if int(environment.get("gpus", 0)) != 0:
        reasons.append("gpu_required")
    return not reasons, reasons


def select_fresh_rows() -> tuple[
    list[tuple[dict[str, Any], str]], list[dict[str, Any]], dict[str, Any]
]:
    rows, source, panel_names = _pinned_source()
    v3_manifest, _, _ = _v3_inputs()
    prior_continuation = {
        task["task_id"] for task in v3_manifest["task_selection"]["ordered_pool"]
    } | EXPOSED_TASK_IDS
    excluded_names = panel_names | prior_continuation
    eligible_by_category: dict[str, list[tuple[str, str, dict[str, Any]]]] = {
        category: [] for category in CATEGORY_QUOTAS
    }
    exclusions = []
    for task_id, row in rows.items():
        if task_id in excluded_names:
            exclusions.append({"task_id": task_id, "reasons": ["prior_exposure"]})
            continue
        compatible, reasons = static_checkpoint_compatibility(row["config"])
        if not compatible:
            exclusions.append({"task_id": task_id, "reasons": reasons})
            continue
        category = tomllib.loads(row["config"])["metadata"]["category"]
        eligible_by_category[category].append(
            (_rank(task_id, category), task_id, row)
        )

    selected: list[tuple[dict[str, Any], str]] = []
    for category, quota in CATEGORY_QUOTAS.items():
        candidates = sorted(eligible_by_category[category])
        if len(candidates) < quota:
            raise RuntimeError(
                f"v4 static pool has {len(candidates)} {category} tasks; needs {quota}"
            )
        selected.extend((row, rank) for rank, _, row in candidates[:quota])
    selected.sort(
        key=lambda item: (
            _rank(
                item[0]["task_id"],
                tomllib.loads(item[0]["config"])["metadata"]["category"],
            ),
            item[0]["task_id"],
        )
    )
    if len(selected) != 24 or len({row["task_id"] for row, _ in selected}) != 24:
        raise RuntimeError("v4 selection must contain 24 unique fresh tasks")
    if excluded_names.intersection(row["task_id"] for row, _ in selected):
        raise RuntimeError("v4 selection reused an exposed task")
    audit = {
        "source_task_count": len(rows),
        "prior_exposure_count": len(excluded_names),
        "static_eligible_count": sum(map(len, eligible_by_category.values())),
        "selected_count": len(selected),
    }
    return selected, exclusions, source | audit


def materialize_fresh_tasks(output_root: Path = FRESH_ROOT) -> dict[str, Any]:
    lock_path = output_root / "task-source-lock-v4.json"
    if lock_path.is_file():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        for task in lock["tasks"]:
            if _tree_sha256(ROOT / task["task_root"]) != task["task_tree_sha256"]:
                raise RuntimeError(f"existing v4 task tree changed: {task['task_id']}")
        return lock | {"lock_path": _portable_path(lock_path)}

    selected, exclusions, source = select_fresh_rows()
    output_root.mkdir(parents=True, exist_ok=True)
    task_locks = []
    for row, selection_rank in selected:
        task_id = row["task_id"]
        task_root = output_root / "tasks" / task_id
        if task_root.exists():
            raise FileExistsError(f"unlocked v4 task exists: {task_root}")
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
                raise RuntimeError(f"v4 task {task_id} is missing {required}")
        metadata = tomllib.loads(row["config"])["metadata"]
        task_locks.append(
            {
                "task_id": task_id,
                "difficulty": metadata["difficulty"],
                "category": metadata["category"],
                "tags": metadata.get("tags", []),
                "outcome_blind_rank": selection_rank,
                "instruction_sha256": _sha256_bytes(row["instruction"].encode()),
                "config_sha256": _sha256_bytes(row["config"].encode()),
                "archive_sha256": _sha256_bytes(row["archive"]),
                "task_tree_sha256": _tree_sha256(task_root),
                "task_root": _portable_path(task_root),
                "prior_public_panel_overlap": False,
                "prior_continuation_exposure": False,
                "static_checkpoint_compatible": True,
            }
        )
    lock = {
        "schema_version": "continuation-calibration-task-source-lock.v4",
        "selection_seed": SELECTION_SEED,
        "category_quotas": CATEGORY_QUOTAS,
        "static_compatibility": {
            "maximum_cpus": MAX_CPUS,
            "maximum_memory_mb": MAX_MEMORY_MB,
            "gpus": 0,
            "disallowed_tags": sorted(DISALLOWED_TAGS),
        },
        "source": source,
        "tasks": task_locks,
        "excluded_task_count": len(exclusions),
    }
    lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return lock | {"lock_path": _portable_path(lock_path)}


def build_task_pool(fresh: dict[str, Any]) -> list[dict[str, Any]]:
    ordered = []
    for position, lock in enumerate(fresh["tasks"], start=1):
        ordered.append(
            {
                "task_id": lock["task_id"],
                "task_category": lock["category"],
                "difficulty": lock["difficulty"],
                "wave": "continuation-v4-targeted-hard",
                "instruction_sha256": lock["instruction_sha256"],
                "task_root": lock["task_root"],
                "task_tree_sha256": lock["task_tree_sha256"],
                "selection_rank": lock["outcome_blind_rank"],
                "static_checkpoint_compatible": True,
                "prior_terminal_outcome_count": 0,
                "position": position,
                "tranche": 1 if position <= 8 else 2 if position <= 16 else 3,
            }
        )
    if len(ordered) != 24:
        raise RuntimeError("v4 pool must contain 24 tasks")
    return ordered


def freeze(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    output_root = output_root.resolve()
    manifest_path = output_root / "frozen-manifest-v4.json"
    if manifest_path.exists():
        raise FileExistsError(f"continuation calibration v4 is frozen: {manifest_path}")
    v3_manifest, v3_report, censoring = _v3_inputs()
    if v3_manifest["models"]["routes"] != EXACT_MODELS:
        raise RuntimeError("v3 routes no longer match exact models")
    fresh = materialize_fresh_tasks()
    tasks = build_task_pool(fresh)

    output_root.mkdir(parents=True, exist_ok=True)
    eligibility_path = output_root / "task-eligibility-report-v4.json"
    eligibility = {
        "schema_version": "continuation-task-eligibility.v4",
        "created_at": datetime.now(UTC).isoformat(),
        "outcome_blind": True,
        "selection_seed": SELECTION_SEED,
        "category_quotas": CATEGORY_QUOTAS,
        "difficulty": "hard",
        "static_compatibility_only": True,
        "prior_outcomes_used_to_select_task_ids": False,
        "tasks": tasks,
    }
    eligibility_path.write_text(json.dumps(eligibility, indent=2) + "\n")

    v3_catalog_path = ROOT / v3_manifest["models"]["catalog_path"]
    catalog_path = output_root / "frozen-model-catalog-v4.json"
    catalog = json.loads(v3_catalog_path.read_text(encoding="utf-8"))
    catalog["schema_version"] = "continuation-model-catalog.v4"
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
        "src/horizon_supervisor/training/run_continuation_calibration_v3.py",
        "src/horizon_supervisor/training/freeze_continuation_calibration_v4.py",
        "src/horizon_supervisor/training/run_continuation_calibration_v4.py",
    ]
    for relative in code_paths:
        if not (ROOT / relative).is_file():
            raise FileNotFoundError(ROOT / relative)
    manifest = {
        "schema_version": "two-tier-continuation-calibration-manifest.v4",
        "frozen_at": datetime.now(UTC).isoformat(),
        "frozen_before_model_outcomes": True,
        "lineage": {
            "v3_manifest_sha256": EXPECTED_V3_MANIFEST_SHA256,
            "v3_report_sha256": EXPECTED_V3_REPORT_SHA256,
            "v3_ledger_sha256": EXPECTED_V3_LEDGER_SHA256,
            "v3_outcomes_sha256": EXPECTED_V3_OUTCOMES_SHA256,
            "v3_checkpoint_bank_sha256": EXPECTED_V3_CHECKPOINTS_SHA256,
            "v3_censoring_sha256": EXPECTED_V3_CENSORING_SHA256,
            "v3_trajectory_count": v3_report["trajectory_count"],
            "v3_confirmed_stuck_task_count": v3_report["tiers"][
                "confirmed_stuck"
            ]["task_count"],
            "minimal_revision": (
                "raise only the per-response output cap from 4096 to 8192 tokens; "
                "preserve the 49152-token total run budget"
            ),
            "detector_thresholds_changed": False,
            "analysis_gates_changed": False,
            "models_changed": False,
            "max_turns_changed": False,
            "total_output_token_budget_changed": False,
        },
        "objective": v3_manifest["objective"],
        "models": {
            "routes": EXACT_MODELS,
            "catalog_path": _portable_path(catalog_path),
            "catalog_sha256": _sha256(catalog_path),
        },
        "detector": v3_manifest["detector"],
        "process_reproducibility": v3_manifest["process_reproducibility"],
        "task_selection": {
            "eligibility_path": _portable_path(eligibility_path),
            "eligibility_sha256": _sha256(eligibility_path),
            "task_source_lock_path": fresh["lock_path"],
            "task_source_lock_sha256": _sha256(ROOT / fresh["lock_path"]),
            "ordered_pool": tasks,
            "maximum_new_tasks": 24,
        },
        "execution": v3_manifest["execution"]
        | {
            "per_response_output_tokens": 8_192,
            "total_output_token_budget": 49_152,
            "output_length_corrective_retries": 1,
        },
        "sampling": {
            "tranches": [
                {"tranche": 1, "task_positions": list(range(1, 9))},
                {"tranche": 2, "task_positions": list(range(9, 17))},
                {"tranche": 3, "task_positions": list(range(17, 25))},
            ],
            "rule": (
                "At each boundary, analyze immutable v3 outcomes plus all completed "
                "v4 outcomes. Stop only if every unchanged readiness gate passes."
            ),
            "fresh_hard_tasks_only": True,
            "outcome_blind_task_order": True,
            "static_checkpoint_compatibility_filter": True,
        },
        "analysis": v3_manifest["analysis"]
        | {
            "base_outcomes_path": _portable_path(V3_OUTCOMES),
            "base_outcomes_sha256": EXPECTED_V3_OUTCOMES_SHA256,
            "aggregate_v3_and_v4": True,
        },
        "budget": {
            "original_project_baseline_usd": 49.257980597,
            "project_openrouter_spend_before_usd": 50.736977389,
            "project_cumulative_ceiling_usd": 200.0,
            "phase_a_incremental_ceiling_usd": 3.5,
            "tranche_1_incremental_ceiling_usd": 1.4,
            "tranche_2_incremental_ceiling_usd": 2.5,
            "per_trial_incremental_ceiling_usd": 0.5,
            "request_reserve_usd": 0.05,
            "provider_key_baseline": "record immediately before first paid v4 call",
            "provider_hard_limit_rule": (
                "hard key limit must be no more than v4 baseline usage plus $3.52"
            ),
        },
        "integrity": {
            "switchyard_sha256": _sha256(SWITCHYARD),
            "source_panel_sha256": _sha256(SOURCE_PANEL),
            "code_sha256": {
                relative: _sha256(ROOT / relative) for relative in code_paths
            },
        },
        "forbidden": v3_manifest["forbidden"],
        "diagnosis": censoring["diagnosis"],
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
