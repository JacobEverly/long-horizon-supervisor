from __future__ import annotations

import argparse
import hashlib
import json
import os
import tomllib
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from horizon_supervisor.benchmark.gate7 import query_openrouter_key
from horizon_supervisor.stuck_detector import SuspectedStuckV0

ROOT = Path(__file__).resolve().parents[3]
PANEL = ROOT / "data/supervisor/terminal-bench-pro-panel-v0.jsonl"
SWITCHYARD = ROOT / "benchmarks/switchyard-gate7.toml"
PILOT_MANIFEST = (
    ROOT / "artifacts/official/stuck-intervention-pilot-v0/frozen-pilot-manifest-v5.json"
)
SWISS_MANIFEST = (
    ROOT
    / "artifacts/official/swiss-cheese-replication-v0/frozen-experiment-manifest-v0.json"
)
PRIOR_MODEL_CATALOG = (
    ROOT / "artifacts/official/stuck-intervention-pilot-v0/frozen-model-catalog-v0.json"
)
OUTPUT_ROOT = ROOT / "artifacts/official/stuck-confirmatory-v1"
FIDELITY_REPORT = OUTPUT_ROOT / "snapshot-fidelity-v0.json"
EXPECTED_DETECTOR_SHA256 = (
    "c3319c93d823455076fd294ac16e28748a2b2ebcab10e1b81760d174088f4ffe"
)
EXACT_MODELS = {
    "gate7/fixed-flash": "deepseek/deepseek-v4-flash-0731",
    "gate7/fixed-qwen": "qwen/qwen3.8-27b",
    "gate7/fixed-kimi": "moonshotai/kimi-k3",
}
BRANCHES = [
    "continue_current_state",
    "switch_value_state",
    "switch_kimi_state",
    "restart_kimi_clean",
]

# These exclusions are based only on the public instruction and environment
# contract. Their required state lives outside the task workdir or in a live
# service that the archive adapter cannot preserve or deterministically restart.
STATIC_SNAPSHOT_EXCLUSIONS = {
    "find-invalid-blockchain-transactions": (
        "requires a pre-existing localhost HTTP service started by the image"
    ),
    "build-prime-factorization-http-api": (
        "requires an agent-created persistent HTTP server"
    ),
    "build-nginx-1-24-production-server": (
        "requires a daemon plus state under /usr/local, /etc, and /var"
    ),
    "reverse-engineer-kvstore-binary-protocol": (
        "requires a pre-existing localhost TCP service outside the task workdir"
    ),
    "diagnose-and-repair-broken-pip-installation": (
        "requires mutations to system pip configuration outside the task workdir"
    ),
    "make-ascii-fits-keywords-case-insensitive": (
        "requires mutation of an installed library outside the task workdir"
    ),
    "restore-broken-pip-installation": (
        "testing the required repair mutates system executables outside the task workdir"
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


def _panel_rows(panel_path: Path = PANEL) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in panel_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _excluded_task_ids(
    pilot_manifest_path: Path = PILOT_MANIFEST,
    swiss_manifest_path: Path = SWISS_MANIFEST,
) -> tuple[set[str], set[str]]:
    pilot = json.loads(pilot_manifest_path.read_text(encoding="utf-8"))
    prior_pilot = {
        str(row["task_id"]) for row in pilot["task_selection"]["ordered_pool"]
    }
    swiss = json.loads(swiss_manifest_path.read_text(encoding="utf-8"))
    posthoc = {str(task) for task in swiss["design"]["tasks"]}
    return prior_pilot, posthoc


def select_ordered_task_pool(
    panel_path: Path = PANEL,
    pilot_manifest_path: Path = PILOT_MANIFEST,
    swiss_manifest_path: Path = SWISS_MANIFEST,
) -> list[dict[str, Any]]:
    prior_pilot, posthoc = _excluded_task_ids(
        pilot_manifest_path, swiss_manifest_path
    )
    eligible = [
        row
        for row in _panel_rows(panel_path)
        if row["wave"] in {1, 2}
        and row["source_task_name"] not in prior_pilot
        and row["source_task_name"] not in posthoc
        and row["source_task_name"] not in STATIC_SNAPSHOT_EXCLUSIONS
    ]
    if not eligible:
        raise ValueError("no untouched snapshot-compatible development tasks")

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_category[str(row["category"])].append(row)
    for rows in by_category.values():
        rows.sort(
            key=lambda row: (
                0 if row["difficulty"] == "hard" else 1,
                -float(row.get("expert_time_estimate_min") or 0),
                str(row["source_task_name"]),
            )
        )

    ordered: list[dict[str, Any]] = []
    rank = 0
    while True:
        added = False
        for category in sorted(by_category):
            if rank < len(by_category[category]):
                ordered.append(by_category[category][rank])
                added = True
        if not added:
            break
        rank += 1
    return ordered


def _route_endpoints(path: Path = SWITCHYARD) -> dict[str, str]:
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    targets = config["targets"]
    return {
        route["id"]: targets[route["target"]]["id"]
        for route in config["routes"].values()
        if route["id"] in EXACT_MODELS
    }


def _write_model_catalog(output_path: Path) -> None:
    source = json.loads(PRIOR_MODEL_CATALOG.read_text(encoding="utf-8"))
    models = [
        row for row in source["models"] if row.get("model_id") in EXACT_MODELS.values()
    ]
    if {row.get("model_id") for row in models} != set(EXACT_MODELS.values()):
        raise RuntimeError("prior frozen catalog does not contain the exact model roster")
    output_path.write_text(
        json.dumps(
            {
                "captured_at": datetime.now(UTC).isoformat(),
                "models": models,
                "allowed_model_ids": list(EXACT_MODELS.values()),
                "source_catalog_path": str(PRIOR_MODEL_CATALOG.relative_to(ROOT)),
                "source_catalog_sha256": _sha256(PRIOR_MODEL_CATALOG),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def freeze(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "frozen-manifest-v0.json"
    if manifest_path.exists():
        raise FileExistsError(f"confirmatory experiment is already frozen: {manifest_path}")

    detector_path = ROOT / "src/horizon_supervisor/stuck_detector.py"
    if _sha256(detector_path) != EXPECTED_DETECTOR_SHA256:
        raise RuntimeError("suspected_stuck_v0 source hash changed")
    if _route_endpoints() != EXACT_MODELS:
        raise RuntimeError("the exact confirmatory route/model mapping changed")

    fidelity_path = output_root / FIDELITY_REPORT.name
    fidelity = json.loads(fidelity_path.read_text(encoding="utf-8"))
    if fidelity.get("passed") is not True:
        raise RuntimeError("snapshot fidelity must pass before freezing")

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required to freeze the spend baseline")
    key_info = query_openrouter_key(api_key)
    usage = float(key_info["usage"])
    hard_limit = float(key_info["limit"])
    remaining = float(key_info.get("limit_remaining", hard_limit - usage))
    if remaining < 20.0:
        raise RuntimeError(
            f"dedicated OpenRouter key has only ${remaining:.2f} remaining; "
            "the frozen experiment requires a $20 ceiling"
        )

    catalog_path = output_root / "frozen-model-catalog-v0.json"
    _write_model_catalog(catalog_path)

    selected = select_ordered_task_pool()
    tasks = []
    for position, row in enumerate(selected, start=1):
        task_root = (
            ROOT
            / f"data/supervisor/terminal-bench-pro-wave-{row['wave']}/tasks"
            / row["source_task_name"]
        )
        if not task_root.is_dir():
            raise FileNotFoundError(task_root)
        tasks.append(
            {
                "position": position,
                "task_id": row["source_task_name"],
                "wave": row["wave"],
                "category": row["category"],
                "difficulty": row["difficulty"],
                "expert_time_estimate_min": row.get("expert_time_estimate_min"),
                "task_root": str(task_root.relative_to(ROOT)),
                "task_tree_sha256": _tree_sha256(task_root),
                "snapshot_compatibility": "task-workdir state only",
            }
        )

    code_files = [
        detector_path,
        ROOT / "src/horizon_supervisor/snapshot.py",
        ROOT / "src/horizon_supervisor/benchmark/pilot_harbor.py",
        ROOT / "src/horizon_supervisor/benchmark/daytona_snapshot_fidelity.py",
        ROOT / "src/horizon_supervisor/training/run_stuck_pilot.py",
        ROOT / "src/horizon_supervisor/training/run_stuck_confirmatory.py",
        ROOT / "src/horizon_supervisor/training/stuck_confirmatory_analysis.py",
        ROOT / "src/horizon_supervisor/training/freeze_stuck_confirmatory.py",
    ]
    for path in code_files:
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = {
        "schema_version": "stuck-confirmatory-manifest.v0",
        "frozen_at": datetime.now(UTC).isoformat(),
        "objective": (
            "Confirm or reject suspected_stuck_v0 and preserved-state Kimi "
            "escalation using broader independent matched checkpoints."
        ),
        "detector": {
            **SuspectedStuckV0.frozen_spec(),
            "source_sha256": EXPECTED_DETECTOR_SHA256,
            "healthy_checkpoint": "turn 4 when detector remains HEALTHY",
            "stuck_checkpoint": "first valid SUSPECTED_STUCK trigger",
            "tuning_permitted": False,
        },
        "task_selection": {
            "eligible_split": "Terminal-Bench Pro development Waves 1 and 2 only",
            "rule": (
                "Exclude the prior stuck-pilot tasks, Swiss-cheese post-hoc tasks, "
                "held-out Wave 3, and public-contract snapshot incompatibilities; "
                "within category sort hard before medium, longer public expert estimate "
                "before shorter, then task id; round-robin sorted categories by rank."
            ),
            "outcome_blind": True,
            "ordered_pool": tasks,
            "pool_size": len(tasks),
            "static_snapshot_exclusions": [
                {"task_id": task_id, "reason": reason}
                for task_id, reason in sorted(STATIC_SNAPSHOT_EXCLUSIONS.items())
            ],
            "pool_expansion_after_outcomes": False,
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
                "permission-preserving task-workdir archive rehydrated into a fresh "
                "Daytona sandbox"
            ),
            "snapshot_fidelity_path": str(fidelity_path.relative_to(ROOT)),
            "snapshot_fidelity_sha256": _sha256(fidelity_path),
            "process_fallback": (
                "No process-memory claim; unmanaged relevant processes make a "
                "checkpoint structurally ineligible."
            ),
            "max_turns": 12,
            "max_output_tokens_per_turn": 4096,
            "reasoning_effort": "high",
            "temperature": "provider_default; omitted",
            "request_timeout_seconds": 1200,
            "provider_attempts": 1,
            "model_retry_attempts": 0,
            "base_schedule": (
                "frozen task order; Flash then Qwen; suspected-stuck then healthy "
                "target trajectories"
            ),
            "concurrency": 1,
            "concurrency_reason": (
                "The current adapter uses shared routing counters and sequential key "
                "deltas for exact arm-level provider cost attribution; concurrent paid "
                "arms are therefore not safely attributable."
            ),
        },
        "branch_contract": {
            "branches": BRANCHES,
            "flash_destinations": {
                "continue_current_state": "gate7/fixed-flash",
                "switch_value_state": "gate7/fixed-qwen",
                "switch_kimi_state": "gate7/fixed-kimi",
                "restart_kimi_clean": "gate7/fixed-kimi",
            },
            "qwen_destinations": {
                "continue_current_state": "gate7/fixed-qwen",
                "switch_value_state": "gate7/fixed-flash",
                "switch_kimi_state": "gate7/fixed-kimi",
                "restart_kimi_clean": "gate7/fixed-kimi",
            },
            "remaining_turns": "12 minus checkpoint turn",
            "remaining_output_tokens": "remaining_turns times 4096",
            "maximum_wall_seconds": (
                "3600 minus source-agent elapsed seconds, with a 60-second floor"
            ),
            "maximum_incremental_spend_usd": 0.5,
            "same_limits_within_group": True,
            "completion": "Harbor external verifier reward >= 1.0",
            "preserved_state": (
                "permission-preserving task-workdir archive plus public, "
                "reasoning-free handoff"
            ),
            "clean_restart": "fresh task environment without prior-state handoff",
        },
        "sampling_and_stopping": {
            "target_stuck_groups": 12,
            "target_healthy_groups": 12,
            "target_valid_outcomes": 96,
            "minimum_unique_tasks_overall": 8,
            "minimum_unique_tasks_per_kind": 4,
            "minimum_groups_per_base_and_kind": 4,
            "maximum_groups_per_base_and_kind": 8,
            "maximum_groups_per_task_and_kind": 2,
            "maximum_groups_per_task_overall": 3,
            "selection": (
                "First structurally eligible checkpoints in frozen schedule order "
                "that do not violate predeclared representation caps."
            ),
            "stop": (
                "Stop at all targets or after the complete frozen pool is exhausted; "
                "never add post-hoc tasks."
            ),
        },
        "retry_and_resume": {
            "valid_model_outcome_rerun": False,
            "infrastructure_failure": (
                "Retry once route-for-route; never rerun a sealed valid sibling."
            ),
            "provider_and_protocol_errors_are_structured": True,
            "partial_state_path": (
                "artifacts/official/stuck-confirmatory-v1/execution-state-v0.json"
            ),
            "sealed_outcomes_path": (
                "artifacts/official/stuck-confirmatory-v1/matched-outcomes-v0.jsonl"
            ),
        },
        "analysis": {
            "module": "horizon_supervisor.training.stuck_confirmatory_analysis",
            "task_cluster_bootstrap_seed": 20260904,
            "task_cluster_bootstrap_samples": 10_000,
            "policy_tuning_on_outcomes": False,
            "detector_gate": {
                "minimum_healthy_minus_stuck_completion": 0.20,
                "task_bootstrap_95_lower_bound_must_exceed_zero": True,
                "leave_one_task_out_gaps_must_be_positive": True,
                "both_base_specific_gaps_must_be_positive": True,
            },
            "kimi_gate": {
                "minimum_stuck_completion_gain_over_continue": 0.15,
                "minimum_unique_rescues": 2,
                "stuck_gain_must_exceed_healthy_gain": True,
                "difference_in_differences_must_be_positive": True,
                "must_be_non_dominated_on_success_and_cost": True,
            },
            "training_requires_both_gates": True,
        },
        "budget": {
            "additional_openrouter_cap_usd": 20.0,
            "usage_before_usd": usage,
            "usage_ceiling_usd": usage + 20.0,
            "dedicated_key_hard_limit_usd": hard_limit,
            "effective_current_key_ceiling_usd": min(usage + 20.0, hard_limit),
            "project_openrouter_spend_before_usd": 45.887060203,
            "daytona_charges_reported_separately": True,
        },
        "integrity": {
            "switchyard_sha256": _sha256(SWITCHYARD),
            "panel_sha256": _sha256(PANEL),
            "pilot_manifest_sha256": _sha256(PILOT_MANIFEST),
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
    (output_root / "frozen-manifest-v0.sha256").write_text(
        f"{_sha256(manifest_path)}  {manifest_path.name}\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    manifest = freeze(args.output_root)
    print(
        json.dumps(
            {
                "task_count": manifest["task_selection"]["pool_size"],
                "usage_before_usd": manifest["budget"]["usage_before_usd"],
                "usage_ceiling_usd": manifest["budget"]["usage_ceiling_usd"],
            }
        )
    )


if __name__ == "__main__":
    main()
