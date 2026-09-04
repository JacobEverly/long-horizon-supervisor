from __future__ import annotations

import argparse
import hashlib
import json
import os
import tomllib
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from horizon_supervisor.benchmark.gate7 import query_openrouter_key
from horizon_supervisor.benchmark.model_catalog import load_model_catalog
from horizon_supervisor.stuck_detector import SuspectedStuckV0

ROOT = Path(__file__).resolve().parents[3]
PANEL = ROOT / "data/supervisor/terminal-bench-pro-panel-v0.jsonl"
SWITCHYARD = ROOT / "benchmarks/switchyard-gate7.toml"
SWISS_MANIFEST = (
    ROOT
    / "artifacts/official/swiss-cheese-replication-v0/frozen-experiment-manifest-v0.json"
)
OUTPUT_ROOT = ROOT / "artifacts/official/stuck-intervention-pilot-v0"
FIDELITY_REPORT = OUTPUT_ROOT / "daytona-fork-fidelity-v0.json"
EXACT_MODELS = {
    "gate7/fixed-flash": "deepseek/deepseek-v4-flash-0731",
    "gate7/fixed-qwen": "qwen/qwen3.8-27b",
    "gate7/fixed-kimi": "moonshotai/kimi-k3",
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


def _panel_rows(panel_path: Path = PANEL) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in panel_path.read_text(encoding="utf-8").splitlines()
    ]


def select_ordered_task_pool(
    panel_path: Path = PANEL, swiss_manifest_path: Path = SWISS_MANIFEST
) -> list[dict[str, Any]]:
    rows = _panel_rows(panel_path)
    posthoc = set(
        json.loads(swiss_manifest_path.read_text(encoding="utf-8"))["design"]["tasks"]
    )
    eligible = [
        row
        for row in rows
        if row["wave"] in {1, 2}
        and row["difficulty"] == "hard"
        and row["source_task_name"] not in posthoc
    ]
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_category[row["category"]].append(row)
    for category in by_category:
        by_category[category].sort(key=lambda row: row["source_task_name"])

    selected: list[dict[str, Any]] = []
    rank = 0
    while len(selected) < 8:
        added = False
        for category in sorted(by_category):
            if rank < len(by_category[category]):
                selected.append(by_category[category][rank])
                added = True
                if len(selected) == 8:
                    break
        if not added:
            raise ValueError("fewer than eight eligible development tasks")
        rank += 1
    return selected


def _route_endpoints(path: Path) -> dict[str, str]:
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    targets = config["targets"]
    endpoints = {}
    for route in config["routes"].values():
        route_id = route["id"]
        if route_id in EXACT_MODELS:
            endpoints[route_id] = targets[route["target"]]["id"]
    return endpoints


def freeze(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "frozen-pilot-manifest-v0.json"
    if manifest_path.exists():
        raise FileExistsError(f"pilot is already frozen: {manifest_path}")
    fidelity_path = output_root / FIDELITY_REPORT.name
    fidelity = json.loads(fidelity_path.read_text(encoding="utf-8"))
    if fidelity.get("passed") is not True:
        raise RuntimeError("Daytona fork fidelity must pass before freezing the pilot")
    endpoints = _route_endpoints(SWITCHYARD)
    if endpoints != EXACT_MODELS:
        raise RuntimeError(f"model endpoints changed: {endpoints}")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required to freeze spend baseline")
    key_info = query_openrouter_key(api_key)
    usage = float(key_info["usage"])
    hard_limit = float(key_info["limit"])

    selected = select_ordered_task_pool()
    tasks = []
    for index, row in enumerate(selected, start=1):
        task_root = (
            ROOT
            / f"data/supervisor/terminal-bench-pro-wave-{row['wave']}/tasks"
            / row["source_task_name"]
        )
        if not task_root.is_dir():
            raise FileNotFoundError(task_root)
        tasks.append(
            {
                "position": index,
                "task_id": row["source_task_name"],
                "wave": row["wave"],
                "category": row["category"],
                "difficulty": row["difficulty"],
                "task_root": str(task_root.relative_to(ROOT)),
                "task_tree_sha256": _tree_sha256(task_root),
            }
        )

    catalog_path = output_root / "frozen-model-catalog-v0.json"
    catalog = [
        model
        for model in load_model_catalog(catalog_path, roster="gate4")
        if model.model_id in EXACT_MODELS.values()
    ]
    catalog_path.write_text(
        json.dumps(
            {
                "captured_at": datetime.now(UTC).isoformat(),
                "models": [asdict(model) for model in catalog],
                "allowed_model_ids": list(EXACT_MODELS.values()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    code_files = [
        ROOT / "src/horizon_supervisor/stuck_detector.py",
        ROOT / "src/horizon_supervisor/snapshot.py",
        ROOT / "src/horizon_supervisor/benchmark/pilot_harbor.py",
        ROOT / "src/horizon_supervisor/training/stuck_pilot_analysis.py",
        ROOT / "src/horizon_supervisor/training/run_stuck_pilot.py",
    ]
    manifest = {
        "schema_version": "matched-stuck-intervention-pilot.v0",
        "frozen_at": datetime.now(UTC).isoformat(),
        "objective": (
            "Test whether suspected_stuck_v0 predicts low recovery and whether "
            "matched state-preserving intervention improves verified completion."
        ),
        "detector": SuspectedStuckV0.frozen_spec(),
        "task_selection": {
            "eligible_split": "Terminal-Bench Pro development Waves 1 and 2 only",
            "rule": (
                "Filter to difficulty=hard, exclude every Swiss-cheese post-hoc task, "
                "sort task names within category, then round-robin sorted categories "
                "by within-category rank; take the first eight."
            ),
            "outcome_blind": True,
            "ordered_pool": tasks,
            "initial_positions": [1, 2, 3, 4],
            "expansion_positions": [5, 6, 7, 8],
            "expansion_depends_only_on_trigger_eligibility": True,
        },
        "models": {
            "routes": EXACT_MODELS,
            "base_routes": ["gate7/fixed-flash", "gate7/fixed-qwen"],
            "escalation_route": "gate7/fixed-kimi",
            "forbidden_routes": [
                "gate7/fixed-glm",
                "swiss/fixed-qwen35-9b",
            ],
            "catalog_path": str(catalog_path.relative_to(ROOT)),
            "catalog_sha256": _sha256(catalog_path),
        },
        "execution": {
            "agent": "horizon_supervisor.benchmark.pilot_harbor:PilotTerminus2",
            "harness": "Harbor 0.21.0",
            "environment": "Daytona direct sandbox",
            "snapshot_adapter": (
                "permission-preserving workspace archive rehydrated into a fresh "
                "Daytona sandbox"
            ),
            "snapshot_fidelity_path": str(fidelity_path.relative_to(ROOT)),
            "snapshot_fidelity_sha256": _sha256(fidelity_path),
            "process_fallback": (
                "No process-memory claim. Checkpoints with unmanaged relevant processes "
                "are structurally ineligible; declared public services must be restarted "
                "from a frozen recipe."
            ),
            "max_turns": 12,
            "max_output_tokens_per_turn": 4096,
            "reasoning_effort": "high",
            "temperature": "provider_default; omitted",
            "request_timeout_seconds": 1200,
            "provider_attempts": 1,
            "model_retry_attempts": 0,
            "healthy_checkpoint": "turn 4 when detector status is HEALTHY",
            "stuck_checkpoint": "first turn where detector status is SUSPECTED_STUCK",
            "base_schedule": (
                "task-pool order; Flash then Qwen within each task; separate "
                "suspected-stuck and healthy-target trajectories so each continuation "
                "has one unambiguous checkpoint budget"
            ),
        },
        "branch_contract": {
            "branches": [
                "continue_current_state",
                "restart_current_clean",
                "switch_value_state",
                "restart_value_clean",
                "switch_kimi_state",
                "restart_kimi_clean",
            ],
            "remaining_turns": "12 minus checkpoint turn",
            "remaining_output_tokens": "remaining_turns times 4096",
            "maximum_wall_seconds": (
                "3600 minus source-agent elapsed seconds at completed snapshot, with "
                "a 60-second floor; enforced on every new branch with Harbor's agent "
                "timeout multiplier"
            ),
            "maximum_incremental_spend_usd": 0.5,
            "same_limits_within_group": True,
            "completion": "Harbor external verifier reward >= 1.0; stop on completion",
            "preserved_state": (
                "permission-preserving workspace archive plus public, reasoning-free "
                "handoff; no process-memory claim"
            ),
            "clean_restart": "fresh task environment with no prior-state handoff",
        },
        "sampling_and_stopping": {
            "target_stuck_groups": 4,
            "target_healthy_groups": 4,
            "maximum_valid_branch_outcomes": 48,
            "selection": (
                "First four structurally eligible checkpoints of each kind in the "
                "predeclared base schedule."
            ),
            "stop_expansion": (
                "Stop after both targets, otherwise exhaust the ordered eight-task pool."
            ),
            "base_representation": (
                "Interleaving Flash then Qwen ensures both bases are represented whenever "
                "both produce eligible checkpoints before a target fills."
            ),
        },
        "retry_policy": {
            "valid_model_outcome_rerun": False,
            "infrastructure_failure": (
                "Retry the invalid route once from the same frozen anchor or clean state; "
                "never rerun valid siblings."
            ),
            "provider_or_protocol_errors_are_structured": True,
        },
        "analysis": {
            "module": "horizon_supervisor.training.stuck_pilot_analysis",
            "task_cluster_bootstrap_seed": 20260902,
            "task_cluster_bootstrap_samples": 10_000,
            "primary_interaction": (
                "intervention-minus-continue at suspected_stuck minus the same contrast "
                "at healthy turn-4 checkpoints"
            ),
            "policy_tuning_on_pilot_outcomes": False,
        },
        "budget": {
            "additional_openrouter_cap_usd": 15.0,
            "usage_before_usd": usage,
            "usage_ceiling_usd": usage + 15.0,
            "dedicated_key_hard_limit_usd": hard_limit,
            "effective_current_key_ceiling_usd": min(usage + 15.0, hard_limit),
            "daytona_charges_reported_separately": True,
        },
        "integrity": {
            "switchyard_sha256": _sha256(SWITCHYARD),
            "panel_sha256": _sha256(PANEL),
            "swiss_manifest_sha256": _sha256(SWISS_MANIFEST),
            "code_sha256": {
                str(path.relative_to(ROOT)): _sha256(path) for path in code_files
            },
            "forbidden_observation_fields": [
                "hidden verifier output",
                "future trajectory",
                "final task success",
                "private reasoning",
                "sibling outcomes",
            ],
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_root / "frozen-pilot-manifest-v0.sha256").write_text(
        f"{_sha256(manifest_path)}  {manifest_path.name}\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    result = freeze(args.output_root)
    print(
        json.dumps(
            {
                "tasks": [
                    row["task_id"]
                    for row in result["task_selection"]["ordered_pool"]
                ],
                "usage_before_usd": result["budget"]["usage_before_usd"],
            }
        )
    )


if __name__ == "__main__":
    main()
