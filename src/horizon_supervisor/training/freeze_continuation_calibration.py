from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
import urllib.request
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from horizon_supervisor.stuck_detector_v2 import (
    FROZEN_CANDIDATE_FAMILY,
    TwoTierStuckDetectorV2,
)

ROOT = Path(__file__).resolve().parents[3]
SOURCE_PANEL = ROOT / "data/supervisor/terminal-bench-pro-panel-v0.jsonl"
TASKS_ROOT = ROOT / "data/supervisor/terminal-bench-pro-wave-4/tasks"
SWITCHYARD = ROOT / "benchmarks/switchyard-gate7.toml"
OUTPUT_ROOT = ROOT / "artifacts/official/two-tier-continuation-calibration-v0"
MODEL_CATALOG_URL = "https://openrouter.ai/api/v1/models"
EXACT_MODELS = {
    "gate7/fixed-flash": "deepseek/deepseek-v4-flash-0731",
    "gate7/fixed-qwen": "qwen/qwen3.8-27b",
}
SELECTED_CONFIG = FROZEN_CANDIDATE_FAMILY[0]
SELECTION_SEED = "two-tier-continuation-calibration-v0|2026-09-05"
STATIC_EXCLUSIONS = {
    "build-arm64-qemu-linux-with-custom-message": (
        "required state and outputs live under /workspace and include a live QEMU guest"
    ),
    "build-coq-from-source": (
        "required source and installed outputs live under /tmp outside the task workdir"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(item.read_bytes()).digest())
    return digest.hexdigest()


def _rank(task_id: str) -> str:
    return hashlib.sha256(f"{SELECTION_SEED}|{task_id}".encode()).hexdigest()


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    return (
        str(resolved.relative_to(ROOT.resolve()))
        if resolved.is_relative_to(ROOT.resolve())
        else str(resolved)
    )


def _panel_rows(path: Path = SOURCE_PANEL) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _route_endpoints(path: Path = SWITCHYARD) -> dict[str, str]:
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    targets = config["targets"]
    return {
        route["id"]: targets[route["target"]]["id"]
        for route in config["routes"].values()
        if route["id"] in EXACT_MODELS
    }


def _prior_reference_index(
    task_ids: set[str], official_root: Path
) -> dict[str, list[str]]:
    references = {task_id: [] for task_id in task_ids}
    if not official_root.exists():
        return references
    for path in official_root.rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".jsonl", ".md"}:
            continue
        if path.resolve().is_relative_to(OUTPUT_ROOT.resolve()):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        portable = _portable_path(path)
        for task_id in task_ids:
            if task_id in text:
                references[task_id].append(portable)
    return {task_id: sorted(paths) for task_id, paths in references.items()}


def select_task_pool(
    panel_path: Path = SOURCE_PANEL,
    tasks_root: Path = TASKS_ROOT,
    official_root: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    official_root = official_root or ROOT / "artifacts/official"
    rows = [row for row in _panel_rows(panel_path) if int(row["wave"]) == 4]
    if len(rows) != 18:
        raise RuntimeError("the frozen Wave 4 source must contain exactly 18 tasks")
    reference_index = _prior_reference_index(
        {str(row["source_task_name"]) for row in rows}, official_root
    )

    excluded = []
    eligible_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task_id = str(row["source_task_name"])
        references = reference_index[task_id]
        reason = STATIC_EXCLUSIONS.get(task_id)
        task_root = tasks_root / task_id
        if reason:
            excluded.append(
                {
                    "task_id": task_id,
                    "reason": reason,
                    "decision_source": "public instruction and environment contract only",
                }
            )
            continue
        if references:
            raise RuntimeError(
                f"fresh Wave 4 task {task_id} appears in prior official evidence: "
                f"{references}"
            )
        for required in (
            "instruction.md",
            "task.toml",
            "environment/Dockerfile",
            "tests/test.sh",
        ):
            if not (task_root / required).is_file():
                raise FileNotFoundError(task_root / required)
        eligible_by_category[str(row["category"])].append(
            {
                "task_id": task_id,
                "task_category": row["category"],
                "difficulty": row["difficulty"],
                "wave": 4,
                "instruction_sha256": row["instruction_sha256"],
                "task_root": str(task_root.relative_to(ROOT)),
                "task_tree_sha256": _tree_sha256(task_root),
                "selection_rank": _rank(task_id),
                "prior_official_reference_count": 0,
                "snapshot_compatibility": (
                    "task-workdir state only; live relevant processes are rejected online"
                ),
            }
        )

    for category_rows in eligible_by_category.values():
        category_rows.sort(key=lambda row: (row["selection_rank"], row["task_id"]))
    ordered = []
    round_index = 0
    while True:
        added = False
        for category in sorted(eligible_by_category):
            category_rows = eligible_by_category[category]
            if round_index < len(category_rows):
                ordered.append(category_rows[round_index])
                added = True
        if not added:
            break
        round_index += 1
    if len(ordered) != 16:
        raise RuntimeError("Wave 4 must yield exactly 16 statically eligible tasks")
    for position, task in enumerate(ordered, start=1):
        task["position"] = position
        task["tranche"] = 1 if position <= 8 else 2
    return ordered, sorted(excluded, key=lambda row: row["task_id"])


def fetch_model_catalog(url: str = MODEL_CATALOG_URL) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.load(response)
    by_id = {row["id"]: row for row in payload["data"]}
    missing = set(EXACT_MODELS.values()) - set(by_id)
    if missing:
        raise RuntimeError(f"OpenRouter catalog is missing frozen models: {sorted(missing)}")
    models = []
    for model_id in EXACT_MODELS.values():
        row = by_id[model_id]
        if "tools" not in row.get("supported_parameters", []):
            raise RuntimeError(f"model no longer advertises tool use: {model_id}")
        models.append(
            {
                "model_id": model_id,
                "canonical_slug": row.get("canonical_slug"),
                "created": row.get("created"),
                "context_length": row.get("context_length"),
                "max_completion_tokens": (row.get("top_provider") or {}).get(
                    "max_completion_tokens"
                ),
                "pricing": row.get("pricing"),
                "supported_parameters": row.get("supported_parameters"),
            }
        )
    return {
        "schema_version": "continuation-model-catalog.v0",
        "captured_at": datetime.now(UTC).isoformat(),
        "source": MODEL_CATALOG_URL,
        "models": models,
    }


def freeze(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    output_root = output_root.resolve()
    manifest_path = output_root / "frozen-manifest-v0.json"
    if manifest_path.exists():
        raise FileExistsError(f"continuation calibration is already frozen: {manifest_path}")
    if _route_endpoints() != EXACT_MODELS:
        raise RuntimeError("the Flash/Qwen Switchyard routes changed")

    tasks, excluded = select_task_pool()
    output_root.mkdir(parents=True, exist_ok=True)
    eligibility_path = output_root / "task-eligibility-report-v0.json"
    eligibility = {
        "schema_version": "continuation-task-eligibility.v0",
        "created_at": datetime.now(UTC).isoformat(),
        "source_panel": str(SOURCE_PANEL.relative_to(ROOT)),
        "source_panel_sha256": _sha256(SOURCE_PANEL),
        "selection_seed": SELECTION_SEED,
        "outcome_blind": True,
        "tasks_inspected": 18,
        "eligible_task_count": len(tasks),
        "excluded_task_count": len(excluded),
        "eligible_tasks": tasks,
        "excluded_tasks": excluded,
    }
    eligibility_path.write_text(json.dumps(eligibility, indent=2) + "\n", encoding="utf-8")

    catalog_path = output_root / "frozen-model-catalog-v0.json"
    catalog = fetch_model_catalog()
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")

    code_paths = [
        "src/horizon_supervisor/stuck_detector_v2.py",
        "src/horizon_supervisor/benchmark/continuation_harbor.py",
        "src/horizon_supervisor/benchmark/pilot_harbor.py",
        "src/horizon_supervisor/training/freeze_continuation_calibration.py",
        "src/horizon_supervisor/training/run_continuation_calibration.py",
        "src/horizon_supervisor/training/analyze_continuation_calibration.py",
    ]
    for relative in code_paths:
        if not (ROOT / relative).is_file():
            raise FileNotFoundError(ROOT / relative)

    manifest = {
        "schema_version": "two-tier-continuation-calibration-manifest.v0",
        "frozen_at": datetime.now(UTC).isoformat(),
        "frozen_before_model_outcomes": True,
        "objective": (
            "Measure natural review-to-confirmation transitions and continuation "
            "recovery without any intervention."
        ),
        "models": {
            "routes": EXACT_MODELS,
            "catalog_path": _portable_path(catalog_path),
            "catalog_sha256": _sha256(catalog_path),
        },
        "detector": TwoTierStuckDetectorV2.frozen_spec(SELECTED_CONFIG),
        "task_selection": {
            "eligibility_path": _portable_path(eligibility_path),
            "eligibility_sha256": _sha256(eligibility_path),
            "ordered_pool": tasks,
            "maximum_tasks": 16,
        },
        "execution": {
            "agent": "ContinuationTerminus2",
            "max_turns": 12,
            "healthy_checkpoint_turn": 4,
            "natural_continuation_only": True,
            "interventions_forbidden": True,
            "infrastructure_retries_per_trial": 1,
            "state_fidelity": "fresh SeededDaytonaEnvironment rehydration per checkpoint",
        },
        "sampling": {
            "tranches": [
                {"tranche": 1, "task_positions": list(range(1, 9))},
                {"tranche": 2, "task_positions": list(range(9, 17))},
            ],
            "rule": (
                "Run tranche 1, stop early only if every readiness gate passes; "
                "otherwise run tranche 2 if the frozen budget remains."
            ),
            "outcome_blind_task_order": True,
        },
        "analysis": {
            "bootstrap_unit": "task",
            "bootstrap_seed": 20260905,
            "bootstrap_samples": 10_000,
            "gates": {
                "healthy_minus_confirmed_recovery": 0.20,
                "clustered_interval_excludes_zero": True,
                "direction_positive_both_models": True,
                "leave_one_task_out_difference_positive": True,
                "maximum_single_task_confirmed_share": 0.25,
                "needs_review_checkpoints": 12,
                "needs_review_tasks": 8,
                "confirmed_checkpoints": 6,
                "confirmed_tasks": 4,
                "healthy_checkpoints": 12,
                "healthy_tasks": 8,
                "both_models_each_tier": True,
                "minimum_confirmed_remaining_turns": 2,
                "all_counted_snapshots_rehydrated": True,
                "structural_failures_separate": True,
                "leakage_controls": True,
            },
        },
        "budget": {
            "project_openrouter_spend_before_usd": 49.257980597,
            "project_cumulative_ceiling_usd": 200.0,
            "phase_a_incremental_ceiling_usd": 5.0,
            "tranche_1_incremental_ceiling_usd": 2.5,
            "per_trial_incremental_ceiling_usd": 0.5,
            "request_reserve_usd": 0.05,
            "provider_key_baseline": "record immediately before first paid call",
            "provider_hard_limit_rule": (
                "hard key limit must be no more than baseline usage plus $5.02"
            ),
        },
        "integrity": {
            "switchyard_sha256": _sha256(SWITCHYARD),
            "code_sha256": {
                relative: _sha256(ROOT / relative) for relative in code_paths
            },
        },
        "forbidden": [
            "model intervention",
            "restart",
            "post-outcome threshold tuning",
            "hidden verifier input",
            "private reasoning input",
            "sibling outcome input",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    digest = _sha256(manifest_path)
    manifest_path.with_suffix(".sha256").write_text(
        f"{digest}  {manifest_path.name}\n", encoding="utf-8"
    )
    return {"manifest_path": str(manifest_path), "manifest_sha256": digest, **manifest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    print(json.dumps(freeze(args.output_root), indent=2))


if __name__ == "__main__":
    main()
