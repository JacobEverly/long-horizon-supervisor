from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import tarfile
import tomllib
import unicodedata
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
from horizon_supervisor.training.freeze_continuation_calibration_v4 import (
    EXPECTED_V3_MANIFEST_SHA256,
    V3_MANIFEST,
)

V4_ROOT = ROOT / "artifacts/official/two-tier-continuation-calibration-v4"
V4_MANIFEST = V4_ROOT / "frozen-manifest-v4.json"
V4_REPORT = V4_ROOT / "calibration-report-v4.json"
V4_LEDGER = V4_ROOT / "execution-ledger-v4.json"
V4_FIDELITY = V4_ROOT / "snapshot-fidelity-v4.json"
V4_PUBLIC = V4_ROOT / "public-summary-v4.json"

EXPECTED_V4_MANIFEST_SHA256 = (
    "7a3b661a296f2b8ceaa3a9234b6b76d8fc0e0d161a73163fe21575c25903b260"
)
EXPECTED_V4_REPORT_SHA256 = (
    "56505afe6dbb22789855f8f153ef388957e484f38e2c8dbb394bfe184f4cd7b9"
)
EXPECTED_V4_LEDGER_SHA256 = (
    "d020de59ae3318dc0997e982ca25709d15e84f82f6f3d55c5819282112314ac5"
)
EXPECTED_V4_FIDELITY_SHA256 = (
    "128526e01fc2f690e16e8d6f9150a6fe88bd7d8b3a774a363348e6a6c1972534"
)
EXPECTED_V4_PUBLIC_SHA256 = (
    "f06f583d2133d25e0f9fc7b9534941f7c31606f0001f9ca03a103146d73efef5"
)

SELECTION_SEED = "continuation-calibration-v5|2026-09-06"
DIFFICULTY_QUOTAS = {"hard": 7, "medium": 6, "easy": 11}
TRANCHE_DIFFICULTY_COUNTS = {
    1: {"hard": 3, "medium": 2, "easy": 3},
    2: {"hard": 2, "medium": 2, "easy": 4},
    3: {"hard": 2, "medium": 2, "easy": 4},
}
DISALLOWED_TAGS = frozenset(
    {
        "apache",
        "distributed-training",
        "interactive",
        "mlflow",
        "mongodb",
        "networking",
        "postgres",
        "postgresql",
        "qemu",
        "server",
        "ssh",
        "vault",
    }
)
DISALLOWED_TASK_ID_TOKENS = (
    "apache",
    "mlflow",
    "mongodb",
    "postgres",
    "qemu",
    "server",
    "ssh",
    "vault",
)
MAX_CPUS = 4
MAX_MEMORY_MB = 8_192
REQUIRED_FINAL_WORKDIR = "/app"
FRESH_ROOT = ROOT / "data/supervisor/terminal-bench-pro-continuation-v5"
OUTPUT_ROOT = ROOT / "artifacts/official/two-tier-continuation-calibration-v5"


def _v4_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    expected = {
        V4_MANIFEST: EXPECTED_V4_MANIFEST_SHA256,
        V4_REPORT: EXPECTED_V4_REPORT_SHA256,
        V4_LEDGER: EXPECTED_V4_LEDGER_SHA256,
        V4_FIDELITY: EXPECTED_V4_FIDELITY_SHA256,
        V4_PUBLIC: EXPECTED_V4_PUBLIC_SHA256,
    }
    for path, digest in expected.items():
        if not path.is_file() or _sha256(path) != digest:
            raise RuntimeError(f"immutable v4 input changed: {path}")
    manifest = json.loads(V4_MANIFEST.read_text(encoding="utf-8"))
    report = json.loads(V4_REPORT.read_text(encoding="utf-8"))
    ledger = json.loads(V4_LEDGER.read_text(encoding="utf-8"))
    fidelity = json.loads(V4_FIDELITY.read_text(encoding="utf-8"))
    public = json.loads(V4_PUBLIC.read_text(encoding="utf-8"))
    if (
        report["trajectory_count"] != 96
        or report["gate_passed"] is not False
        or report["gates"]["all_counted_snapshots_rehydrated"] is not False
        or not all(
            value
            for name, value in report["gates"].items()
            if name != "all_counted_snapshots_rehydrated"
        )
    ):
        raise RuntimeError("v4 terminal calibration verdict changed")
    if (
        fidelity["attempt_count"] != 65
        or fidelity["pass_count"] != 63
        or fidelity["all_passed"] is not False
    ):
        raise RuntimeError("v4 fidelity failure changed")
    if (
        ledger["status"] != "complete"
        or ledger["stop_reason"] != "frozen_task_pool_exhausted"
        or ledger["openrouter"]["project_spend_after_usd"] != 53.269355918
        or ledger["cleanup"]["remaining_new_sandbox_ids"]
        or public["gate_passed"] is not False
    ):
        raise RuntimeError("v4 execution evidence changed")
    return manifest, report


def _rank(task_id: str, difficulty: str) -> str:
    return hashlib.sha256(
        f"{SELECTION_SEED}|{difficulty}|{task_id}".encode()
    ).hexdigest()


def _tranche_rank(task_id: str, tranche: int) -> str:
    return hashlib.sha256(
        f"{SELECTION_SEED}|tranche-{tranche}|{task_id}".encode()
    ).hexdigest()


def _normalized_instruction_sha256(instruction: str) -> str:
    normalized = unicodedata.normalize("NFKC", instruction).lower()
    normalized = re.sub(r"\W+", " ", normalized).strip()
    return hashlib.sha256(normalized.encode()).hexdigest()


def _dockerfile_from_archive(task_id: str, archive_bytes: bytes) -> str:
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
        candidates = [
            member
            for member in archive.getmembers()
            if member.isfile()
            and (
                member.name == "environment/Dockerfile"
                or member.name.endswith(f"/{task_id}/environment/Dockerfile")
                or member.name.endswith("/environment/Dockerfile")
            )
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"expected one Dockerfile for {task_id}, found {len(candidates)}"
            )
        handle = archive.extractfile(candidates[0])
        if handle is None:
            raise RuntimeError(f"could not read Dockerfile for {task_id}")
        return handle.read().decode("utf-8")


def _final_workdir(dockerfile: str) -> str | None:
    matches = re.findall(
        r"^\s*WORKDIR\s+([^\s#]+)", dockerfile, flags=re.IGNORECASE | re.MULTILINE
    )
    return matches[-1] if matches else None


def static_checkpoint_compatibility(
    task_id: str, config: str, dockerfile: str
) -> tuple[bool, list[str]]:
    parsed = tomllib.loads(config)
    metadata = parsed["metadata"]
    environment = parsed["environment"]
    reasons: list[str] = []
    if metadata.get("difficulty") not in DIFFICULTY_QUOTAS:
        reasons.append("difficulty_not_targeted")
    tags = {str(tag).lower() for tag in metadata.get("tags", [])}
    if tags.intersection(DISALLOWED_TAGS):
        reasons.append("service_or_interactive_runtime_tag")
    lowered_id = task_id.lower()
    if any(token in lowered_id for token in DISALLOWED_TASK_ID_TOKENS):
        reasons.append("service_or_interactive_task_id")
    if int(environment.get("cpus", 0)) > MAX_CPUS:
        reasons.append("cpu_limit_exceeds_static_cap")
    if int(environment.get("memory_mb", 0)) > MAX_MEMORY_MB:
        reasons.append("memory_limit_exceeds_static_cap")
    if int(environment.get("gpus", 0)) != 0:
        reasons.append("gpu_required")
    if _final_workdir(dockerfile) != REQUIRED_FINAL_WORKDIR:
        reasons.append("final_workdir_not_app")
    return not reasons, reasons


def _prior_task_ids(v4_manifest: dict[str, Any], panel_names: set[str]) -> set[str]:
    if not V3_MANIFEST.is_file() or _sha256(V3_MANIFEST) != EXPECTED_V3_MANIFEST_SHA256:
        raise RuntimeError("immutable v3 continuation manifest changed")
    v3_manifest = json.loads(V3_MANIFEST.read_text(encoding="utf-8"))
    prior = set(panel_names) | set(EXPOSED_TASK_IDS)
    prior.update(
        task["task_id"] for task in v3_manifest["task_selection"]["ordered_pool"]
    )
    prior.update(
        task["task_id"] for task in v4_manifest["task_selection"]["ordered_pool"]
    )
    return prior


def select_fresh_rows() -> tuple[
    list[tuple[dict[str, Any], str, int]], list[dict[str, Any]], dict[str, Any]
]:
    rows, source, panel_names = _pinned_source()
    v4_manifest, _ = _v4_inputs()
    excluded_names = _prior_task_ids(v4_manifest, panel_names)
    prior_instruction_hashes = {
        _normalized_instruction_sha256(row["instruction"])
        for task_id, row in rows.items()
        if task_id in excluded_names
    }
    eligible_by_difficulty: dict[str, list[tuple[str, str, dict[str, Any]]]] = {
        difficulty: [] for difficulty in DIFFICULTY_QUOTAS
    }
    exclusions = []
    for task_id, row in rows.items():
        if task_id in excluded_names:
            exclusions.append({"task_id": task_id, "reasons": ["prior_exposure"]})
            continue
        normalized_instruction_sha256 = _normalized_instruction_sha256(
            row["instruction"]
        )
        if normalized_instruction_sha256 in prior_instruction_hashes:
            exclusions.append(
                {"task_id": task_id, "reasons": ["prior_instruction_overlap"]}
            )
            continue
        dockerfile = _dockerfile_from_archive(task_id, row["archive"])
        compatible, reasons = static_checkpoint_compatibility(
            task_id, row["config"], dockerfile
        )
        if not compatible:
            exclusions.append({"task_id": task_id, "reasons": reasons})
            continue
        difficulty = tomllib.loads(row["config"])["metadata"]["difficulty"]
        eligible_by_difficulty[difficulty].append(
            (_rank(task_id, difficulty), task_id, row)
        )

    selected_by_difficulty: dict[str, list[tuple[str, str, dict[str, Any]]]] = {}
    for difficulty, quota in DIFFICULTY_QUOTAS.items():
        candidates = sorted(eligible_by_difficulty[difficulty])
        if len(candidates) < quota:
            raise RuntimeError(
                f"v5 static pool has {len(candidates)} {difficulty} tasks; needs {quota}"
            )
        selected_by_difficulty[difficulty] = candidates[:quota]

    offsets = {difficulty: 0 for difficulty in DIFFICULTY_QUOTAS}
    selected: list[tuple[dict[str, Any], str, int]] = []
    for tranche in (1, 2, 3):
        tranche_rows = []
        for difficulty, count in TRANCHE_DIFFICULTY_COUNTS[tranche].items():
            start = offsets[difficulty]
            chosen = selected_by_difficulty[difficulty][start : start + count]
            offsets[difficulty] += count
            tranche_rows.extend((row, rank, tranche) for rank, _, row in chosen)
        tranche_rows.sort(
            key=lambda item: (_tranche_rank(item[0]["task_id"], tranche), item[0]["task_id"])
        )
        selected.extend(tranche_rows)

    selected_ids = [row["task_id"] for row, _, _ in selected]
    if len(selected) != 24 or len(set(selected_ids)) != 24:
        raise RuntimeError("v5 selection must contain 24 unique tasks")
    if excluded_names.intersection(selected_ids):
        raise RuntimeError("v5 selection reused an exposed task")
    selected_instruction_hashes = [
        _normalized_instruction_sha256(row["instruction"])
        for row, _, _ in selected
    ]
    if prior_instruction_hashes.intersection(selected_instruction_hashes):
        raise RuntimeError("v5 selection reused a prior normalized instruction")
    if len(set(selected_instruction_hashes)) != len(selected_instruction_hashes):
        raise RuntimeError("v5 selection contains duplicate normalized instructions")
    audit = {
        "source_task_count": len(rows),
        "prior_exposure_count": len(excluded_names),
        "static_eligible_by_difficulty": {
            key: len(value) for key, value in eligible_by_difficulty.items()
        },
        "selected_count": len(selected),
        "normalized_prior_instruction_overlap_count": 0,
        "normalized_selected_instruction_duplicate_count": 0,
    }
    return selected, exclusions, source | audit


def materialize_fresh_tasks(output_root: Path = FRESH_ROOT) -> dict[str, Any]:
    lock_path = output_root / "task-source-lock-v5.json"
    if lock_path.is_file():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        for task in lock["tasks"]:
            if _tree_sha256(ROOT / task["task_root"]) != task["task_tree_sha256"]:
                raise RuntimeError(f"existing v5 task tree changed: {task['task_id']}")
        return lock | {"lock_path": _portable_path(lock_path)}

    selected, exclusions, source = select_fresh_rows()
    output_root.mkdir(parents=True, exist_ok=True)
    task_locks = []
    for row, selection_rank, tranche in selected:
        task_id = row["task_id"]
        task_root = output_root / "tasks" / task_id
        if task_root.exists():
            raise FileExistsError(f"unlocked v5 task exists: {task_root}")
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
                raise RuntimeError(f"v5 task {task_id} is missing {required}")
        metadata = tomllib.loads(row["config"])["metadata"]
        task_locks.append(
            {
                "task_id": task_id,
                "difficulty": metadata["difficulty"],
                "category": metadata["category"],
                "tags": metadata.get("tags", []),
                "tranche": tranche,
                "outcome_blind_rank": selection_rank,
                "instruction_sha256": _sha256_bytes(row["instruction"].encode()),
                "normalized_instruction_sha256": _normalized_instruction_sha256(
                    row["instruction"]
                ),
                "config_sha256": _sha256_bytes(row["config"].encode()),
                "archive_sha256": _sha256_bytes(row["archive"]),
                "task_tree_sha256": _tree_sha256(task_root),
                "task_root": _portable_path(task_root),
                "final_docker_workdir": REQUIRED_FINAL_WORKDIR,
                "prior_public_panel_overlap": False,
                "prior_continuation_exposure": False,
                "static_checkpoint_compatible": True,
            }
        )
    lock = {
        "schema_version": "continuation-calibration-task-source-lock.v5",
        "selection_seed": SELECTION_SEED,
        "difficulty_quotas": DIFFICULTY_QUOTAS,
        "tranche_difficulty_counts": TRANCHE_DIFFICULTY_COUNTS,
        "static_compatibility": {
            "maximum_cpus": MAX_CPUS,
            "maximum_memory_mb": MAX_MEMORY_MB,
            "gpus": 0,
            "required_final_workdir": REQUIRED_FINAL_WORKDIR,
            "disallowed_tags": sorted(DISALLOWED_TAGS),
            "disallowed_task_id_tokens": list(DISALLOWED_TASK_ID_TOKENS),
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
                "wave": "continuation-v5-independent-compatible",
                "instruction_sha256": lock["instruction_sha256"],
                "task_root": lock["task_root"],
                "task_tree_sha256": lock["task_tree_sha256"],
                "selection_rank": lock["outcome_blind_rank"],
                "static_checkpoint_compatible": True,
                "prior_terminal_outcome_count": 0,
                "position": position,
                "tranche": lock["tranche"],
            }
        )
    if len(ordered) != 24:
        raise RuntimeError("v5 pool must contain 24 tasks")
    if [row["tranche"] for row in ordered] != [1] * 8 + [2] * 8 + [3] * 8:
        raise RuntimeError("v5 tranche order changed")
    return ordered


def freeze(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    output_root = output_root.resolve()
    manifest_path = output_root / "frozen-manifest-v5.json"
    if manifest_path.exists():
        raise FileExistsError(f"continuation calibration v5 is frozen: {manifest_path}")
    v4_manifest, _ = _v4_inputs()
    if v4_manifest["models"]["routes"] != EXACT_MODELS:
        raise RuntimeError("v4 routes no longer match exact models")
    fresh = materialize_fresh_tasks()
    tasks = build_task_pool(fresh)

    output_root.mkdir(parents=True, exist_ok=True)
    eligibility_path = output_root / "task-eligibility-report-v5.json"
    eligibility = {
        "schema_version": "continuation-task-eligibility.v5",
        "created_at": datetime.now(UTC).isoformat(),
        "outcome_blind": True,
        "selection_seed": SELECTION_SEED,
        "difficulty_quotas": DIFFICULTY_QUOTAS,
        "tranche_difficulty_counts": TRANCHE_DIFFICULTY_COUNTS,
        "static_compatibility_only": True,
        "prior_outcomes_used_to_select_task_ids": False,
        "tasks": tasks,
    }
    eligibility_path.write_text(json.dumps(eligibility, indent=2) + "\n")

    v4_catalog_path = ROOT / v4_manifest["models"]["catalog_path"]
    catalog_path = output_root / "frozen-model-catalog-v5.json"
    catalog = json.loads(v4_catalog_path.read_text(encoding="utf-8"))
    catalog["schema_version"] = "continuation-model-catalog.v5"
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n")

    code_paths = [
        "src/horizon_supervisor/stuck_detector_v2.py",
        "src/horizon_supervisor/benchmark/pilot_harbor.py",
        "src/horizon_supervisor/benchmark/continuation_harbor.py",
        "src/horizon_supervisor/benchmark/continuation_harbor_v1.py",
        "src/horizon_supervisor/benchmark/continuation_harbor_v2.py",
        "src/horizon_supervisor/benchmark/permission_preserving_daytona.py",
        "src/horizon_supervisor/training/analyze_continuation_calibration.py",
        "src/horizon_supervisor/training/run_stuck_pilot.py",
        "src/horizon_supervisor/training/run_continuation_calibration.py",
        "src/horizon_supervisor/training/run_continuation_calibration_v1.py",
        "src/horizon_supervisor/training/run_continuation_calibration_v2.py",
        "src/horizon_supervisor/training/run_continuation_calibration_v3.py",
        "src/horizon_supervisor/training/run_continuation_calibration_v4.py",
        "src/horizon_supervisor/training/freeze_continuation_calibration_v5.py",
        "src/horizon_supervisor/training/run_continuation_calibration_v5.py",
        "src/horizon_supervisor/training/run_permission_transport_smoke_v5.py",
    ]
    for relative in code_paths:
        if not (ROOT / relative).is_file():
            raise FileNotFoundError(ROOT / relative)

    process_reproducibility = v4_manifest["process_reproducibility"] | {
        "checkpoint_transport": (
            "PermissionPreservingDaytonaEnvironment with safe tar data checks, "
            "ordinary POSIX mode preservation, and end-to-end digest verification"
        )
    }
    manifest = {
        "schema_version": "two-tier-continuation-calibration-manifest.v5",
        "frozen_at": datetime.now(UTC).isoformat(),
        "frozen_before_model_outcomes": True,
        "lineage": {
            "v3_manifest_sha256": EXPECTED_V3_MANIFEST_SHA256,
            "v4_manifest_sha256": EXPECTED_V4_MANIFEST_SHA256,
            "v4_report_sha256": EXPECTED_V4_REPORT_SHA256,
            "v4_ledger_sha256": EXPECTED_V4_LEDGER_SHA256,
            "v4_fidelity_sha256": EXPECTED_V4_FIDELITY_SHA256,
            "v4_public_summary_sha256": EXPECTED_V4_PUBLIC_SHA256,
            "v4_trajectory_count": 48,
            "v4_failed_fidelity_replays": 2,
            "minimal_revision": (
                "use safe permission-preserving checkpoint download transport; "
                "exclude final workdirs and public tags/task IDs that imply "
                "harness-owned or service process state; use all 13 remaining "
                "fresh compatible hard/medium tasks plus 11 easy tasks because "
                "fewer than 24 compatible hard/medium tasks remain; "
                "score only the independent v5 cohort"
            ),
            "detector_thresholds_changed": False,
            "analysis_gate_thresholds_changed": False,
            "models_changed": False,
            "max_turns_changed": False,
            "token_limits_changed": False,
            "prior_outcomes_reused_for_scoring": False,
        },
        "objective": v4_manifest["objective"],
        "models": {
            "routes": EXACT_MODELS,
            "catalog_path": _portable_path(catalog_path),
            "catalog_sha256": _sha256(catalog_path),
        },
        "detector": v4_manifest["detector"],
        "process_reproducibility": process_reproducibility,
        "task_selection": {
            "eligibility_path": _portable_path(eligibility_path),
            "eligibility_sha256": _sha256(eligibility_path),
            "task_source_lock_path": fresh["lock_path"],
            "task_source_lock_sha256": _sha256(ROOT / fresh["lock_path"]),
            "ordered_pool": tasks,
            "maximum_new_tasks": 24,
        },
        "execution": v4_manifest["execution"]
        | {
            "checkpoint_environment": (
                "horizon_supervisor.benchmark.permission_preserving_daytona:"
                "PermissionPreservingDaytonaEnvironment"
            ),
            "transport_smoke_required": True,
            "transport_smoke_path": (
                "artifacts/official/two-tier-continuation-calibration-v5/"
                "permission-transport-smoke-v5.json"
            ),
            "transport_smoke_schema": "permission-transport-smoke.v5",
            "transport_smoke_provider_model_calls": 0,
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
                "At each boundary, analyze only immutable completed v5 outcomes. "
                "Stop only if every unchanged readiness gate passes; otherwise "
                "expand outcome-blind to the next frozen tranche."
            ),
            "difficulty_quotas": DIFFICULTY_QUOTAS,
            "tranche_difficulty_counts": TRANCHE_DIFFICULTY_COUNTS,
            "outcome_blind_task_order": True,
            "static_checkpoint_compatibility_filter": True,
        },
        "analysis": v4_manifest["analysis"]
        | {
            "cohort": "fresh_v5_only",
            "aggregate_v3_and_v4": False,
            "prior_outcomes_used_for_fit_or_tuning": False,
        },
        "budget": {
            "original_project_baseline_usd": 49.257980597,
            "project_openrouter_spend_before_usd": 53.269355918,
            "project_cumulative_ceiling_usd": 200.0,
            "phase_a_incremental_ceiling_usd": 3.5,
            "tranche_1_incremental_ceiling_usd": 1.4,
            "tranche_2_incremental_ceiling_usd": 2.5,
            "per_trial_incremental_ceiling_usd": 0.5,
            "request_reserve_usd": 0.05,
            "provider_key_baseline": "record immediately before first paid v5 call",
            "provider_hard_limit_rule": (
                "hard key limit must be no more than v5 baseline usage plus $3.52"
            ),
        },
        "integrity": {
            "switchyard_sha256": _sha256(SWITCHYARD),
            "source_panel_sha256": _sha256(SOURCE_PANEL),
            "code_sha256": {
                relative: _sha256(ROOT / relative) for relative in code_paths
            },
        },
        "forbidden": v4_manifest["forbidden"],
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
