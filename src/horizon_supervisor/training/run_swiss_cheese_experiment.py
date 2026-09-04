from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from horizon_supervisor.benchmark.gate7 import query_openrouter_key
from horizon_supervisor.benchmark.gate8 import Gate8PilotConfig, run_gate8_pilot
from horizon_supervisor.benchmark.matched_outcomes import (
    LEARNING_VALID_STATUSES,
    build_matched_outcomes,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _key_usage() -> float:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required")
    payload = query_openrouter_key(api_key)
    if payload.get("usage") is None:
        raise RuntimeError("OpenRouter did not return exact key usage")
    return float(payload["usage"])


def _validate_frozen_code(root: Path, manifest: dict[str, Any]) -> None:
    frozen = manifest["frozen_inputs"]
    paths = {
        "analysis_code_sha256": (
            root / "src/horizon_supervisor/training/swiss_cheese_scorecard.py"
        ),
        "matrix_builder_code_sha256": (
            root / "src/horizon_supervisor/training/build_swiss_cheese_matrix.py"
        ),
        "outcome_extractor_code_sha256": (
            root / "src/horizon_supervisor/benchmark/matched_outcomes.py"
        ),
        "benchmark_runner_code_sha256": (
            root / "src/horizon_supervisor/benchmark/gate8.py"
        ),
        "benchmark_cli_code_sha256": root / "src/horizon_supervisor/benchmark/cli.py",
        "model_catalog_code_sha256": (
            root / "src/horizon_supervisor/benchmark/model_catalog.py"
        ),
        "experiment_orchestrator_code_sha256": Path(__file__),
    }
    for field, path in paths.items():
        if _sha256(path) != frozen[field]:
            raise RuntimeError(f"frozen code changed: {path}")


def _config_for_contract(
    root: Path,
    output_root: Path,
    contract_path: Path,
    contract: dict[str, Any],
) -> Gate8PilotConfig:
    execution = contract["execution_contract"]
    controls = execution["run_controls"]
    return Gate8PilotConfig(
        artifacts_root=output_root / "runs",
        wave=3,
        panel_path=root / "data/supervisor/terminal-bench-pro-panel-v0.jsonl",
        tasks_path=root / "data/supervisor/terminal-bench-pro-wave-3/tasks",
        switchyard_config_path=root / "benchmarks/switchyard-swiss-cheese-v0.toml",
        budget_contract_path=contract_path,
        route_ids=tuple(execution["route_ids"]),
        include_task_names=tuple(execution["selected_task_names"]),
        environment="daytona",
        n_concurrent=int(controls["n_concurrent"]),
        max_turns=int(controls["max_turns"]),
        max_output_tokens=int(controls["max_output_tokens"]),
        reasoning_effort=str(controls["reasoning_effort"]),
        request_timeout_seconds=int(controls["request_timeout_seconds"]),
        request_retry_attempts=int(controls["request_retry_attempts"]),
        output_length_retry_attempts=int(
            controls["output_length_retry_attempts"]
        ),
        authorized_model_budget_usd=float(
            contract["model_budget"]["dedicated_openrouter_key_hard_cap_usd"]
        ),
        wall_timeout_seconds=int(controls["wall_timeout_seconds"]),
        model_roster=str(controls["model_roster"]),
    )


def _extract_run(
    root: Path,
    output_root: Path,
    label: str,
    replication_index: int,
    report: dict[str, Any],
    routes: tuple[str, ...],
) -> tuple[dict[str, Any], Path, Path]:
    run_root = Path(report["pricing_snapshot"]).parent
    outcomes_path = output_root / "replication-inputs" / f"{label}.jsonl"
    summary_path = output_root / "replication-inputs" / f"{label}-summary.json"
    summary = build_matched_outcomes(
        run_root,
        outcomes_path,
        summary_path,
        panel_path=root / "data/supervisor/terminal-bench-pro-panel-v0.jsonl",
        switchyard_path=root / "benchmarks/switchyard-swiss-cheese-v0.toml",
        expected_routes=routes,
        record_split_override="development",
        evaluation_role="posthoc_clean_start_replication",
        replication_index=replication_index,
    )
    return summary, outcomes_path, summary_path


def run_frozen_experiment(root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_root = manifest_path.parent
    _validate_frozen_code(root, manifest)
    (output_root / "runs").mkdir(parents=True, exist_ok=True)
    (output_root / "replication-inputs").mkdir(parents=True, exist_ok=True)
    ledger_path = output_root / "execution-ledger-v0.json"
    ledger = (
        json.loads(ledger_path.read_text(encoding="utf-8"))
        if ledger_path.exists()
        else {
            "schema_version": "swiss-cheese-execution-ledger.v0",
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "started_at": datetime.now(UTC).isoformat(),
            "key_usage_before_usd": _key_usage(),
            "runs": [],
        }
    )
    if ledger["manifest_sha256"] != _sha256(manifest_path):
        raise RuntimeError("manifest changed after execution began")
    if not ledger["runs"]:
        expected_usage = float(
            manifest["budget"]["dedicated_key_usage_before_usd"]
        )
        if abs(float(ledger["key_usage_before_usd"]) - expected_usage) > 0.001:
            raise RuntimeError(
                "dedicated-key usage no longer matches the frozen baseline"
            )

    completed = {
        row["label"] for row in ledger["runs"] if row["status"] == "sealed"
    }
    for spec in manifest["execution"]["contracts"]:
        label = spec["label"]
        if label in completed:
            continue
        contract_path = root / spec["contract_path"]
        if _sha256(contract_path) != spec["contract_sha256"]:
            raise RuntimeError(f"frozen execution contract changed: {label}")
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        config = _config_for_contract(root, output_root, contract_path, contract)
        report = run_gate8_pilot(config)
        summary, outcomes_path, summary_path = _extract_run(
            root,
            output_root,
            label,
            int(spec["replication_index"]),
            report,
            tuple(spec["routes"]),
        )
        rows = [
            json.loads(line)
            for line in outcomes_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        invalid = [
            row
            for row in rows
            if row["outcome"]["status"] not in LEARNING_VALID_STATUSES
        ]
        sealed = (
            report["status"] == "complete"
            and summary["all_pairs_present_once"]
            and not invalid
        )
        ledger["runs"].append(
            {
                "label": label,
                "replication_index": spec["replication_index"],
                "status": "sealed" if sealed else "needs_infrastructure_recovery",
                "report_path": str(Path(report["pricing_snapshot"]).parent / "report.json"),
                "outcomes_path": str(outcomes_path),
                "summary_path": str(summary_path),
                "provider_spend_usd": report.get("provider_spend_usd"),
                "outcome_status_counts": summary["status_counts"],
                "missing_pairs": summary["missing_pairs"],
            }
        )
        ledger["key_usage_latest_usd"] = _key_usage()
        _write_json(ledger_path, ledger)
        if not sealed:
            raise RuntimeError(
                f"{label} needs infrastructure-only recovery; no learning-valid "
                "pair was rerun"
            )

    ledger["status"] = "collection_complete"
    ledger["completed_at"] = datetime.now(UTC).isoformat()
    ledger["key_usage_after_usd"] = _key_usage()
    ledger["exact_incremental_spend_usd"] = (
        float(ledger["key_usage_after_usd"])
        - float(ledger["key_usage_before_usd"])
    )
    _write_json(ledger_path, ledger)
    return ledger


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen Swiss-cheese matrix")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "artifacts/official/swiss-cheese-replication-v0/"
            "frozen-experiment-manifest-v0.json"
        ),
    )
    args = parser.parse_args()
    result = run_frozen_experiment(args.root.resolve(), args.manifest.resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
