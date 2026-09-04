from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from horizon_supervisor.benchmark.matched_outcomes import LEARNING_VALID_STATUSES
from horizon_supervisor.training.resume_swiss_cheese_contract import (
    _read_rows,
    _row_key,
    _write_rows,
)
from horizon_supervisor.training.run_swiss_cheese_experiment import (
    _extract_run,
    _write_json,
)


def interrupted_run_accounting(
    run_manifest: dict[str, Any], exact_key_usage_after_usd: float
) -> tuple[float, float, float]:
    hard_cap = float(run_manifest["config"]["authorized_model_budget_usd"])
    remaining_before = float(run_manifest["dedicated_key_remaining_before_usd"])
    usage_before = hard_cap - remaining_before
    usage_after = float(exact_key_usage_after_usd)
    spend = usage_after - usage_before
    if not 0 <= usage_before <= hard_cap:
        raise ValueError("interrupted run has invalid pre-run key accounting")
    if not usage_before <= usage_after <= hard_cap:
        raise ValueError("exact post-run key usage is outside the valid range")
    return usage_before, usage_after, spend


def _single_path(paths: list[Path], description: str) -> Path:
    if len(paths) != 1:
        raise RuntimeError(f"expected exactly one {description}, found {len(paths)}")
    return paths[0]


def seal_interrupted_pair(
    *,
    root: Path,
    output_root: Path,
    run_root: Path,
    source_label: str,
    recovery_label: str,
    replication_index: int,
    exact_key_usage_after_usd: float,
    accounting_source: str,
) -> dict[str, Any]:
    report_path = run_root / "report.json"
    if report_path.exists():
        raise RuntimeError("interrupted run already has a final report")
    manifest = json.loads((run_root / "run-manifest.json").read_text(encoding="utf-8"))
    selected_tasks = manifest["frozen_inputs"]["selected_task_names"]
    routes = tuple(manifest["config"]["route_ids"])
    if len(selected_tasks) != 1 or len(routes) != 1:
        raise RuntimeError("offline sealing is limited to one completed task-route pair")
    task_name = str(selected_tasks[0])
    route = str(routes[0])
    usage_before, usage_after, provider_spend = interrupted_run_accounting(
        manifest, exact_key_usage_after_usd
    )
    task_stats_path = _single_path(
        sorted((run_root / "task-stats").glob("*.json")), "routing-stats file"
    )
    job_results = sorted((run_root / "jobs").glob("*/result.json"))
    harbor_result_path = _single_path(job_results, "Harbor job result")
    harbor_result = json.loads(harbor_result_path.read_text(encoding="utf-8"))
    stats = harbor_result.get("stats") or {}
    if (
        int(harbor_result.get("n_total_trials") or 0) != 1
        or int(stats.get("n_completed_trials") or 0) != 1
        or int(stats.get("n_errored_trials") or 0) != 0
        or not harbor_result.get("finished_at")
    ):
        raise RuntimeError("Harbor result is not one cleanly completed trial")
    routing_stats = json.loads(task_stats_path.read_text(encoding="utf-8"))
    report = {
        **manifest,
        "status": "complete",
        "return_code": 0,
        "wall_timeout_reached": False,
        "provider_spend_usd": provider_spend,
        "dedicated_key_usage_before_usd": usage_before,
        "dedicated_key_usage_after_usd": usage_after,
        "dedicated_key_remaining_after_usd": (
            float(manifest["config"]["authorized_model_budget_usd"]) - usage_after
        ),
        "task_tree_unchanged": True,
        "completed_task_batches": 1,
        "task_runs": [
            {
                "source_task_name": task_name,
                "job_name": harbor_result_path.parent.name,
                "return_code": 0,
                "wall_timeout_reached": False,
                "provider_spend_usd": provider_spend,
                "dedicated_key_usage_before_usd": usage_before,
                "dedicated_key_usage_after_usd": usage_after,
                "routing_stats_path": str(task_stats_path),
                "routing_stats": routing_stats,
                "harbor_result": harbor_result,
            }
        ],
        "stop_reason": None,
        "accounting_recovery": {
            "reason": "post-task OpenRouter key query returned HTTP 401 after the key expired",
            "source": accounting_source,
            "recovered_at": datetime.now(UTC).isoformat(),
            "model_execution_was_not_repeated": True,
        },
    }
    _write_json(report_path, report)

    summary, outcomes_path, summary_path = _extract_run(
        root,
        output_root,
        recovery_label,
        replication_index,
        report,
        routes,
    )
    rows = _read_rows(outcomes_path)
    expected_key = (task_name, route, replication_index)
    if len(rows) != 1 or _row_key(rows[0]) != expected_key:
        raise RuntimeError("recovered run produced the wrong matched outcome")
    row = rows[0]
    if row["outcome"]["status"] not in LEARNING_VALID_STATUSES:
        raise RuntimeError("completed interrupted run is not learning-valid")

    ledger_path = output_root / "execution-ledger-v0.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if any(item["label"] == recovery_label for item in ledger["runs"]):
        raise RuntimeError("recovery label already exists in execution ledger")
    contract_path = Path(manifest["config"]["budget_contract_path"])
    ledger["runs"].append(
        {
            "label": recovery_label,
            "replication_index": replication_index,
            "status": "sealed",
            "infrastructure_recovery_or_continuation_only": True,
            "contract_path": str(contract_path),
            "report_path": str(report_path),
            "outcomes_path": str(outcomes_path),
            "summary_path": str(summary_path),
            "provider_spend_usd": provider_spend,
            "outcome_status_counts": summary["status_counts"],
            "missing_pairs": [],
            "replaces_outcome_id": None,
            "replacement_outcome_id": row["outcome_id"],
            "accounting_recovered_without_model_rerun": True,
        }
    )
    ledger["key_usage_latest_usd"] = usage_after
    ledger.setdefault("accounting_recoveries", []).append(
        {
            "label": recovery_label,
            "run_root": str(run_root),
            "key_usage_before_usd": usage_before,
            "key_usage_after_usd": usage_after,
            "provider_spend_usd": provider_spend,
            "source": accounting_source,
        }
    )
    _write_json(ledger_path, ledger)

    progress_path = output_root / "resume-progress" / f"{source_label}.jsonl"
    progress_rows = _read_rows(progress_path)
    if any(_row_key(item) == expected_key for item in progress_rows):
        raise RuntimeError("recovered pair is already present in resume progress")
    progress_rows.append(row)
    _write_rows(progress_path, progress_rows)
    return {
        "status": "sealed",
        "source_label": source_label,
        "recovery_label": recovery_label,
        "task": task_name,
        "route": route,
        "provider_spend_usd": provider_spend,
        "exact_key_usage_after_usd": usage_after,
        "outcomes_path": str(outcomes_path),
        "progress_path": str(progress_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seal a completed single-pair Gate 8 run after accounting failed"
    )
    parser.add_argument("run_root", type=Path)
    parser.add_argument("source_label")
    parser.add_argument("recovery_label")
    parser.add_argument("--replication-index", type=int, required=True)
    parser.add_argument("--exact-key-usage-after-usd", type=float, required=True)
    parser.add_argument("--accounting-source", required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/official/swiss-cheese-replication-v0"),
    )
    args = parser.parse_args()
    result = seal_interrupted_pair(
        root=args.root.resolve(),
        output_root=args.output_root.resolve(),
        run_root=args.run_root.resolve(),
        source_label=args.source_label,
        recovery_label=args.recovery_label,
        replication_index=args.replication_index,
        exact_key_usage_after_usd=args.exact_key_usage_after_usd,
        accounting_source=args.accounting_source,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
