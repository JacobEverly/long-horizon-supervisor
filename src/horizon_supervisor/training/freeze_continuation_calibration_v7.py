"""Freeze a fresh, outcome-blind continuation-calibration v7 cohort."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
import tomllib
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any

from horizon_supervisor.supervisor_data.materialize_terminal_bench_pro import (
    _safe_members,
)
from horizon_supervisor.training.freeze_continuation_calibration import (
    ROOT,
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
from horizon_supervisor.training.freeze_continuation_calibration_v5 import (
    _dockerfile_from_archive,
    _normalized_instruction_sha256,
    static_checkpoint_compatibility,
)
from horizon_supervisor.training.freeze_continuation_calibration_v6 import (
    OUTPUT_ROOT as V6_ROOT,
)

V6_MANIFEST = V6_ROOT / "frozen-manifest-v6.json"
V6_SMOKE = V6_ROOT / "permission-transport-smoke-v6.json"
V6_MANIFEST_SHA256 = "0f412bee4bef70d3e65a80961c52d87955f03d9f61955b59b2f17b101331b5cc"
V6_SMOKE_SHA256 = "6508b9d8329278144ac1b8a9c890c5eba3db7ed5f2cc5f5d28c51bb3b08f4f80"

SELECTION_SEED = "continuation-calibration-v7|2026-09-07"
TASK_COUNT = 16
CATEGORY_QUOTA = {
    "data-processing": 2,
    "debugging": 2,
    "games": 2,
    "machine-learning": 2,
    "scientific-computing": 2,
    "security": 2,
    "software-engineering": 2,
    "system-administration": 2,
}
FRESH_ROOT = ROOT / "data/supervisor/terminal-bench-pro-continuation-v7"
OUTPUT_ROOT = ROOT / "artifacts/official/two-tier-continuation-calibration-v7"


def _rank(task_id: str, category: str) -> str:
    return hashlib.sha256(f"{SELECTION_SEED}|{category}|{task_id}".encode()).hexdigest()


def _prior_ids() -> set[str]:
    prior = set(EXPOSED_TASK_IDS)
    for path in (
        ROOT / "data/supervisor/terminal-bench-pro-panel-v0.jsonl",
        ROOT / "data/supervisor/terminal-bench-pro-continuation-v5/task-source-lock-v5.json",
    ):
        if path.suffix == ".jsonl":
            prior.update(
                json.loads(line)["source_task_name"]
                for line in path.read_text().splitlines()
                if line.strip()
            )
        else:
            prior.update(
                row["task_id"] for row in json.loads(path.read_text())["tasks"]
            )
    for path in (
        ROOT / "artifacts/official/two-tier-continuation-calibration-v3/frozen-manifest-v3.json",
        ROOT / "artifacts/official/two-tier-continuation-calibration-v4/frozen-manifest-v4.json",
        V6_MANIFEST,
    ):
        prior.update(
            row["task_id"]
            for row in json.loads(path.read_text())["task_selection"]["ordered_pool"]
        )
    return prior


def select_tasks() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, source, _ = _pinned_source()
    prior = _prior_ids()
    candidates: dict[str, list[tuple[str, str, dict[str, Any]]]] = {
        category: [] for category in CATEGORY_QUOTA
    }
    for task_id, row in rows.items():
        if task_id in prior:
            continue
        config = tomllib.loads(row["config"])
        metadata = config["metadata"]
        compatible, _ = static_checkpoint_compatibility(
            task_id, row["config"], _dockerfile_from_archive(task_id, row["archive"])
        )
        category = metadata.get("category")
        if compatible and metadata.get("difficulty") == "easy" and category in candidates:
            candidates[category].append((_rank(task_id, category), task_id, row))
    selected: list[dict[str, Any]] = []
    for category, quota in CATEGORY_QUOTA.items():
        options = sorted(candidates[category])
        if len(options) < quota:
            raise RuntimeError(f"v7 lacks {quota} fresh {category} tasks")
        selected.extend(row for _, _, row in options[:quota])
    selected.sort(
        key=lambda row: hashlib.sha256(
            f"{SELECTION_SEED}|order|{row['task_id']}".encode()
        ).hexdigest()
    )
    if len(selected) != TASK_COUNT or len({row["task_id"] for row in selected}) != TASK_COUNT:
        raise RuntimeError("v7 selection is not rectangular")
    instructions = [
        _normalized_instruction_sha256(row["instruction"]) for row in selected
    ]
    if len(set(instructions)) != len(instructions):
        raise RuntimeError("v7 selected instructions are duplicated")
    return selected, {
        "dataset_id": source["dataset_id"],
        "file": source["file"],
        "revision": source["revision"],
        "selection_seed": SELECTION_SEED,
        "prior_task_count": len(prior),
        "selected_count": len(selected),
        "category_quota": CATEGORY_QUOTA,
        "static_eligible_remaining": {category: len(rows) for category, rows in candidates.items()},
    }


def materialize() -> dict[str, Any]:
    lock_path = FRESH_ROOT / "task-source-lock-v7.json"
    if lock_path.exists():
        return json.loads(lock_path.read_text())
    selected, source = select_tasks()
    task_locks = []
    for position, row in enumerate(selected, start=1):
        task_id = row["task_id"]
        task_root = FRESH_ROOT / "tasks" / task_id
        task_root.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(row["archive"]), mode="r:gz") as archive:
            archive.extractall(
                task_root.parent,
                members=_safe_members(archive, task_id),
                filter="data",
            )
        required = (
            "instruction.md",
            "task.toml",
            "environment/Dockerfile",
            "tests/test.sh",
        )
        if any(not (task_root / item).is_file() for item in required):
            raise RuntimeError(f"v7 task {task_id} is incomplete")
        metadata = tomllib.loads(row["config"])["metadata"]
        task_locks.append(
            {
                "task_id": task_id,
                "difficulty": metadata["difficulty"],
                "category": metadata["category"],
                "instruction_sha256": _sha256_bytes(row["instruction"].encode()),
                "normalized_instruction_sha256": _normalized_instruction_sha256(row["instruction"]),
                "config_sha256": _sha256_bytes(row["config"].encode()),
                "archive_sha256": _sha256_bytes(row["archive"]),
                "task_tree_sha256": _tree_sha256(task_root),
                "task_root": _portable_path(task_root),
                "position": position,
                "static_checkpoint_compatible": True,
                "prior_public_panel_overlap": False,
                "prior_continuation_exposure": False,
            }
        )
    lock = {
        "schema_version": "continuation-calibration-task-source-lock.v7",
        "source": source,
        "tasks": task_locks,
        "selected_instruction_duplicate_count": 0,
    }
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")
    return lock


def freeze() -> dict[str, Any]:
    lock = materialize()
    if OUTPUT_ROOT.exists() and (OUTPUT_ROOT / "frozen-manifest-v7.json").exists():
        raise FileExistsError("v7 manifest is already frozen")
    base = json.loads(V6_MANIFEST.read_text())
    manifest = deepcopy(base)
    manifest["schema_version"] = "two-tier-continuation-calibration-manifest.v7"
    manifest["frozen_at"] = datetime.now(UTC).isoformat()
    manifest["lineage"] = {
        "v6_manifest_sha256": V6_MANIFEST_SHA256,
        "v6_transport_smoke_sha256": V6_SMOKE_SHA256,
        "minimal_revision": (
            "fresh 16-task cohort plus bounded 420-second Harbor child timeout "
            "after repeated provider hangs"
        ),
        "task_selection_changed": True,
        "detector_thresholds_changed": False,
        "analysis_gate_thresholds_changed": False,
        "models_changed": False,
        "max_turns_changed": False,
        "token_limits_changed": False,
        "natural_continuation_protocol_changed": False,
        "prior_outcomes_reused_for_scoring": False,
    }
    manifest["task_selection"] = {
        "ordered_pool": [
            {
                "task_id": row["task_id"],
                "task_category": row["category"],
                "difficulty": row["difficulty"],
                "task_root": row["task_root"],
                "task_tree_sha256": row["task_tree_sha256"],
                "instruction_sha256": row["instruction_sha256"],
                "static_checkpoint_compatible": True,
                "prior_terminal_outcome_count": 0,
                "position": row["position"],
                "tranche": 1 if row["position"] <= 8 else 2,
            }
            for row in lock["tasks"]
        ],
        "source_task_lock": (
            "data/supervisor/terminal-bench-pro-continuation-v7/"
            "task-source-lock-v7.json"
        ),
        "source_task_lock_sha256": _sha256(FRESH_ROOT / "task-source-lock-v7.json"),
    }
    manifest["task_selection"]["ordered_pool"] = manifest["task_selection"].pop("ordered_pool")
    manifest["execution"] |= {
        "transport_smoke_path": (
            "artifacts/official/two-tier-continuation-calibration-v7/"
            "permission-transport-smoke-v7.json"
        ),
        "transport_smoke_schema": "permission-transport-smoke.v7",
        "child_timeout_seconds": 420,
    }
    manifest["analysis"] |= {
        "cohort": "fresh_v7_only",
        "prior_outcomes_used_for_fit_or_tuning": False,
        "aggregate_v6": False,
    }
    manifest["budget"] |= {
        "project_openrouter_spend_before_usd": 54.738412354,
        "phase_a_incremental_ceiling_usd": 2.03,
        "tranche_1_incremental_ceiling_usd": 1.0,
        "tranche_2_incremental_ceiling_usd": 2.03,
        "per_trial_incremental_ceiling_usd": 0.35,
        "provider_hard_limit_rule": "hard key limit no more than v7 baseline plus $1.82",
    }
    manifest["integrity"]["code_sha256"] = {
        "src/horizon_supervisor/training/freeze_continuation_calibration_v7.py": _sha256(
            ROOT / "src/horizon_supervisor/training/freeze_continuation_calibration_v7.py"
        ),
        "src/horizon_supervisor/training/run_continuation_calibration_v7.py": _sha256(
            ROOT / "src/horizon_supervisor/training/run_continuation_calibration_v7.py"
        ),
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    smoke = json.loads(V6_SMOKE.read_text())
    smoke["schema_version"] = "permission-transport-smoke.v7"
    smoke["lineage_v6_smoke_sha256"] = V6_SMOKE_SHA256
    smoke["remaining_daytona_environments"] = 0
    (OUTPUT_ROOT / "permission-transport-smoke-v7.json").write_text(
        json.dumps(smoke, indent=2) + "\n"
    )
    manifest["execution"]["transport_smoke_sha256"] = _sha256(
        OUTPUT_ROOT / "permission-transport-smoke-v7.json"
    )
    path = OUTPUT_ROOT / "frozen-manifest-v7.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    path.with_suffix(".sha256").write_text(f"{_sha256(path)}  {path.name}\n")
    return {
        "manifest_path": str(path),
        "manifest_sha256": _sha256(path),
        "task_count": len(lock["tasks"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    print(json.dumps(freeze() if args.freeze else materialize(), indent=2))


if __name__ == "__main__":
    main()
