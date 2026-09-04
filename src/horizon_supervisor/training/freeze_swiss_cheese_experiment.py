from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from horizon_supervisor.benchmark.gate8 import _tree_sha256
from horizon_supervisor.benchmark.model_catalog import (
    SWISS_CHEESE_SMALL_MODEL_ID,
    load_model_catalog,
)

BASELINE_USAGE_USD = 27.653569393
INCREMENTAL_CAP_USD = 20.0
USAGE_CEILING_USD = BASELINE_USAGE_USD + INCREMENTAL_CAP_USD
MINIMUM_RESERVE_USD = 1.5
KEY_HARD_CAP_USD = 50.0
SMALL_ROUTE = "swiss/fixed-qwen35-9b"
EXISTING_ROUTES = (
    "gate7/fixed-flash",
    "gate7/fixed-qwen",
    "gate7/fixed-glm",
    "gate7/fixed-kimi",
)
ALL_ROUTES = (*EXISTING_ROUTES, SMALL_ROUTE)
PRIME_TASK = "python-prime-http-server"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _route_endpoints(path: Path) -> dict[str, str]:
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    targets = {name: row["id"] for name, row in config["targets"].items()}
    return {
        row["id"]: targets[row["target"]]
        for row in config["routes"].values()
        if row.get("type") == "passthrough"
    }


def _budget_contract(
    *,
    tasks: list[str],
    routes: list[str],
    n_concurrent: int,
    frozen_inputs: dict[str, str],
    replication_index: int,
    run_label: str,
) -> dict[str, Any]:
    return {
        "schema_version": "swiss-cheese-clean-start-replication-budget.v0",
        "wave": 3,
        "frozen_at": datetime.now(UTC).date().isoformat(),
        "scope": (
            "Outcome-blind post-hoc clean-start replication for the Swiss-cheese "
            "same-model versus heterogeneous-model experiment"
        ),
        "trial_count": len(tasks) * len(routes),
        "no_harbor_retries": True,
        "replication_design": {
            "experiment": "swiss-cheese-replication-v0",
            "replication_index": replication_index,
            "run_label": run_label,
            "record_split": "development",
            "evaluation_role": "posthoc_clean_start_replication",
            "outcome_blind": True,
            "policy_tuning_during_collection": False,
            "clean_start_only": True,
        },
        "execution_contract": {
            "route_ids": routes,
            "selected_task_names": tasks,
            "run_controls": {
                "n_concurrent": n_concurrent,
                "max_turns": 12,
                "max_output_tokens": 8192,
                "reasoning_effort": "high",
                "request_timeout_seconds": 1200,
                "request_retry_attempts": 1,
                "output_length_retry_attempts": 1,
                "wall_timeout_seconds": 7200,
                "model_roster": "swiss_cheese",
            },
        },
        "model_budget": {
            "total_key_spend_at_freeze_usd": BASELINE_USAGE_USD,
            "checkpoint_incremental_spend_cap_usd": INCREMENTAL_CAP_USD,
            "dedicated_key_usage_ceiling_usd": USAGE_CEILING_USD,
            "minimum_next_task_reserve_usd": MINIMUM_RESERVE_USD,
            "dedicated_openrouter_key_remaining_at_freeze_usd": (
                KEY_HARD_CAP_USD - BASELINE_USAGE_USD
            ),
            "dedicated_openrouter_key_hard_cap_usd": KEY_HARD_CAP_USD,
        },
        "sandbox_budget": {
            "environment": "daytona",
            "heavy_task_isolated": PRIME_TASK in tasks and len(tasks) == 1,
            "stale_sandbox_preflight_required": True,
            "stale_sandbox_count_at_freeze": 0,
        },
        "frozen_inputs": frozen_inputs,
        "interpretation_guard": (
            "The ten tasks were selected because replication 1 disagreed across "
            "models. Replication 1 is discovery-only. Only frozen replications 2 "
            "and 3 support repeatability claims. Infrastructure-invalid trials may "
            "be recovered route-for-route; learning-valid outcomes may not be rerun."
        ),
    }


def freeze_experiment(root: Path, output_root: Path, contract_root: Path) -> dict[str, Any]:
    baseline = root / (
        "artifacts/official/gate8-wave3-18-task-checkpoint/"
        "matched-outcomes-72-v1.jsonl"
    )
    scorecard_path = root / (
        "artifacts/official/gate8-wave3-18-task-checkpoint/"
        "frozen-policy-scorecard-18-task-v0.json"
    )
    panel_path = root / "data/supervisor/terminal-bench-pro-panel-v0.jsonl"
    tasks_path = root / "data/supervisor/terminal-bench-pro-wave-3/tasks"
    switchyard_path = root / "benchmarks/switchyard-swiss-cheese-v0.toml"
    analysis_path = root / "src/horizon_supervisor/training/swiss_cheese_scorecard.py"
    builder_path = root / "src/horizon_supervisor/training/build_swiss_cheese_matrix.py"
    extractor_path = root / "src/horizon_supervisor/benchmark/matched_outcomes.py"
    runner_path = root / "src/horizon_supervisor/benchmark/gate8.py"
    cli_path = root / "src/horizon_supervisor/benchmark/cli.py"
    model_catalog_path = root / "src/horizon_supervisor/benchmark/model_catalog.py"
    orchestrator_path = root / (
        "src/horizon_supervisor/training/run_swiss_cheese_experiment.py"
    )
    manifest_path = output_root / "frozen-experiment-manifest-v0.json"
    catalog_path = output_root / "frozen-model-catalog-v0.json"
    if manifest_path.exists() or catalog_path.exists():
        raise FileExistsError("Swiss-cheese experiment is already frozen")

    scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
    tasks = sorted(
        row["task"]
        for row in scorecard["task_outcomes"]
        if 0 < int(row["success_count"]) < 4
    )
    if len(tasks) != 10:
        raise RuntimeError(f"expected 10 discriminating tasks, found {len(tasks)}")
    light_tasks = [task for task in tasks if task != PRIME_TASK]
    if len(light_tasks) != 9 or PRIME_TASK not in tasks:
        raise RuntimeError("unexpected light/heavy discriminating-task split")

    route_endpoints = _route_endpoints(switchyard_path)
    if set(route_endpoints) != set(ALL_ROUTES):
        raise RuntimeError("Swiss-cheese Switchyard routes do not match the design")
    output_root.mkdir(parents=True, exist_ok=False)
    contract_root.mkdir(parents=True, exist_ok=False)
    catalog = load_model_catalog(catalog_path, roster="swiss_cheese")
    catalog_payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    if catalog_payload["source"] != "openrouter-public-catalog":
        raise RuntimeError("live OpenRouter catalog is required at experiment freeze")
    small = next(model for model in catalog if model.model_id == SWISS_CHEESE_SMALL_MODEL_ID)
    if small.context_length < 262_144:
        raise RuntimeError("small model context window is below the frozen minimum")
    if small.input_usd_per_million > 0.1 or small.output_usd_per_million > 0.15:
        raise RuntimeError("small model pricing changed above the selection ceiling")

    frozen_inputs = {
        "panel_sha256": _sha256(panel_path),
        "tasks_tree_sha256": _tree_sha256(tasks_path),
        "switchyard_config_sha256": _sha256(switchyard_path),
    }
    run_specs = []
    for replication in (2, 3):
        run_specs.append(
            {
                "label": f"existing-light-rep{replication}",
                "replication_index": replication,
                "tasks": light_tasks,
                "routes": list(EXISTING_ROUTES),
                "n_concurrent": 3,
            }
        )
        for route in EXISTING_ROUTES:
            route_stem = route.rsplit("-", maxsplit=1)[-1]
            run_specs.append(
                {
                    "label": f"prime-{route_stem}-rep{replication}",
                    "replication_index": replication,
                    "tasks": [PRIME_TASK],
                    "routes": [route],
                    "n_concurrent": 1,
                }
            )
    for replication in (1, 2, 3):
        run_specs.append(
            {
                "label": f"small-all-rep{replication}",
                "replication_index": replication,
                "tasks": tasks,
                "routes": [SMALL_ROUTE],
                "n_concurrent": 1,
            }
        )

    contracts = []
    for spec in run_specs:
        contract_path = contract_root / f"{spec['label']}.json"
        contract = _budget_contract(
            tasks=spec["tasks"],
            routes=spec["routes"],
            n_concurrent=spec["n_concurrent"],
            frozen_inputs=frozen_inputs,
            replication_index=spec["replication_index"],
            run_label=spec["label"],
        )
        contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
        contracts.append(
            {
                **spec,
                "contract_path": str(contract_path.relative_to(root)),
                "contract_sha256": _sha256(contract_path),
            }
        )

    manifest = {
        "schema_version": "swiss-cheese-replication-experiment.v0",
        "frozen_at": datetime.now(UTC).isoformat(),
        "hypothesis": (
            "Different-model retries provide repeatable completion or cost gains "
            "beyond a second independent attempt from the same model."
        ),
        "design": {
            "tasks": tasks,
            "task_selection": (
                "All and only Wave 3 tasks with 1-3 successes across the four "
                "original routes in discovery replication 1"
            ),
            "routes": list(ALL_ROUTES),
            "existing_routes": list(EXISTING_ROUTES),
            "small_route": SMALL_ROUTE,
            "route_endpoints": route_endpoints,
            "replication_indices": [1, 2, 3],
            "confirmatory_replication_indices": [2, 3],
            "discovery_replication_index": 1,
            "new_trial_count": sum(
                len(spec["tasks"]) * len(spec["routes"]) for spec in run_specs
            ),
            "final_matrix_record_count": len(tasks) * len(ALL_ROUTES) * 3,
            "clean_start_only": True,
            "same_model_retry_pairing": "2->3 and 3->2",
            "heterogeneous_pairing": "same replication index (2->2 and 3->3)",
        },
        "model_selection": {
            "small_model_id": SWISS_CHEESE_SMALL_MODEL_ID,
            "small_model_label": small.label,
            "selection_reason": (
                "Current 9B tool-capable model with 262k context; smaller and "
                "cheaper than the existing Qwen3.8-27B route."
            ),
            "selection_observed_task_outcomes": False,
            "catalog_path": str(catalog_path.relative_to(root)),
            "catalog_sha256": _sha256(catalog_path),
        },
        "execution": {
            "contracts": contracts,
            "infrastructure_recovery_only": True,
            "learning_valid_reruns_forbidden": True,
            "record_split": "development",
            "evaluation_role": "posthoc_clean_start_replication",
        },
        "analysis": {
            "confirmatory_data": "replications 2 and 3 only",
            "descriptive_discovery_data": "replication 1 only",
            "bootstrap_unit": "original task",
            "bootstrap_seed": 20260831,
            "bootstrap_samples": 10000,
            "predeclared_heterogeneous_contrasts": [
                ["gate7/fixed-flash", "gate7/fixed-qwen"],
                ["gate7/fixed-kimi", "gate7/fixed-flash"],
                ["gate7/fixed-kimi", "gate7/fixed-qwen"],
                ["gate7/fixed-kimi", "gate7/fixed-glm"],
                ["gate7/fixed-flash", SMALL_ROUTE],
            ],
            "named_cascades": {
                "existing-frozen:flash>qwen>glm>kimi": list(EXISTING_ROUTES),
                "small-overlay:flash>small>qwen>glm>kimi": [
                    "gate7/fixed-flash",
                    SMALL_ROUTE,
                    "gate7/fixed-qwen",
                    "gate7/fixed-glm",
                    "gate7/fixed-kimi",
                ],
            },
            "support_rule": (
                "Supported only if a predeclared heterogeneous-vs-same-model "
                "completion delta has a task-cluster bootstrap 95% lower bound "
                "above zero. Positive point estimates with rescues on at least "
                "two tasks are suggestive, not confirmed."
            ),
            "small_model_rule": (
                "The small route earns a place only if it rescues at least one "
                "distinct confirmatory task and appears in a success-cost Pareto "
                "strategy."
            ),
        },
        "budget": {
            "dedicated_key_usage_before_usd": BASELINE_USAGE_USD,
            "incremental_spend_cap_usd": INCREMENTAL_CAP_USD,
            "dedicated_key_usage_ceiling_usd": USAGE_CEILING_USD,
            "minimum_next_task_reserve_usd": MINIMUM_RESERVE_USD,
            "historical_two_existing_replications_estimate_usd": 10.910506764,
        },
        "frozen_inputs": {
            "baseline_outcomes_path": str(baseline.relative_to(root)),
            "baseline_outcomes_sha256": _sha256(baseline),
            "discovery_scorecard_path": str(scorecard_path.relative_to(root)),
            "discovery_scorecard_sha256": _sha256(scorecard_path),
            **frozen_inputs,
            "analysis_code_sha256": _sha256(analysis_path),
            "matrix_builder_code_sha256": _sha256(builder_path),
            "outcome_extractor_code_sha256": _sha256(extractor_path),
            "benchmark_runner_code_sha256": _sha256(runner_path),
            "benchmark_cli_code_sha256": _sha256(cli_path),
            "model_catalog_code_sha256": _sha256(model_catalog_path),
            "experiment_orchestrator_code_sha256": _sha256(orchestrator_path),
        },
        "interpretation_guard": (
            "This is a post-hoc replication of an outcome-selected task panel. "
            "It estimates repeatability on these tasks and does not restore their "
            "held-out status or establish general production performance."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the Swiss-cheese experiment")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/official/swiss-cheese-replication-v0"),
    )
    parser.add_argument(
        "--contract-root",
        type=Path,
        default=Path("benchmarks/swiss-cheese-replication-v0"),
    )
    args = parser.parse_args()
    manifest = freeze_experiment(
        args.root.resolve(), args.output_root.resolve(), args.contract_root.resolve()
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
