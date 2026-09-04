from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from horizon_supervisor.benchmark.gate7 import query_openrouter_key
from horizon_supervisor.training.run_swiss_cheese_experiment import _write_json


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_execution_audit(
    *,
    ledger: dict[str, Any],
    matrix_summary: dict[str, Any],
    scorecard: dict[str, Any],
    prior_key_fingerprint: str,
    prior_key_final_usage_usd: float,
    active_key_fingerprint: str,
    active_key_initial_usage_usd: float,
    active_key_final_usage_usd: float,
    active_key_limit_usd: float,
    spend_ceiling_usd: float,
) -> dict[str, Any]:
    baseline = float(ledger["key_usage_before_usd"])
    combined_final = prior_key_final_usage_usd + (
        active_key_final_usage_usd - active_key_initial_usage_usd
    )
    exact_spend = combined_final - baseline
    run_report_spend = sum(
        float(row.get("provider_spend_usd") or 0.0) for row in ledger["runs"]
    )
    if not matrix_summary.get("all_pairs_present_once_and_learning_valid"):
        raise ValueError("replication matrix is not complete and learning-valid")
    if int(matrix_summary.get("record_count") or 0) != 150:
        raise ValueError("replication matrix does not contain 150 records")
    if exact_spend < 0 or combined_final > spend_ceiling_usd:
        raise ValueError("exact experiment spend is outside the frozen budget")
    scorecard_spend = float(scorecard["spend_audit"]["exact_incremental_spend_usd"])
    if abs(scorecard_spend - exact_spend) > 1e-9:
        raise ValueError("scorecard spend does not match rollover accounting")
    return {
        "schema_version": "swiss-cheese-execution-audit.v0",
        "status": "complete",
        "accounting_basis": (
            "sum of exact usage deltas across two dedicated OpenRouter keys"
        ),
        "experiment_baseline_usd": baseline,
        "conceptual_combined_final_usage_usd": combined_final,
        "exact_incremental_provider_spend_usd": exact_spend,
        "frozen_incremental_spend_ceiling_usd": spend_ceiling_usd - baseline,
        "under_frozen_ceiling": combined_final <= spend_ceiling_usd,
        "run_report_spend_usd": run_report_spend,
        "key_delta_minus_run_reports_usd": exact_spend - run_report_spend,
        "keys": [
            {
                "role": "original_dedicated_key",
                "fingerprint_sha256": prior_key_fingerprint,
                "final_usage_usd": prior_key_final_usage_usd,
                "rollover_reason": "key expired during post-task accounting",
            },
            {
                "role": "capped_continuation_key",
                "fingerprint_sha256": active_key_fingerprint,
                "initial_usage_usd": active_key_initial_usage_usd,
                "final_usage_usd": active_key_final_usage_usd,
                "usage_delta_usd": (
                    active_key_final_usage_usd - active_key_initial_usage_usd
                ),
                "hard_limit_usd": active_key_limit_usd,
            },
        ],
        "matrix": {
            "records": matrix_summary["record_count"],
            "tasks": matrix_summary["task_count"],
            "routes": matrix_summary["route_count"],
            "replications_per_pair": matrix_summary["replications_per_pair"],
            "complete_and_learning_valid": True,
        },
    }


def finalize_execution(
    *,
    ledger_path: Path,
    matrix_summary_path: Path,
    matrix_path: Path,
    scorecard_path: Path,
    memo_path: Path,
    output_path: Path,
    prior_key_fingerprint: str,
    prior_key_final_usage_usd: float,
    active_key_initial_usage_usd: float,
    spend_ceiling_usd: float,
) -> dict[str, Any]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required")
    key_info = query_openrouter_key(api_key)
    if key_info.get("usage") is None or key_info.get("limit") is None:
        raise RuntimeError("OpenRouter did not return exact active-key accounting")
    active_fingerprint = hashlib.sha256(api_key.encode()).hexdigest()
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    matrix_summary = json.loads(matrix_summary_path.read_text(encoding="utf-8"))
    scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
    audit = build_execution_audit(
        ledger=ledger,
        matrix_summary=matrix_summary,
        scorecard=scorecard,
        prior_key_fingerprint=prior_key_fingerprint,
        prior_key_final_usage_usd=prior_key_final_usage_usd,
        active_key_fingerprint=active_fingerprint,
        active_key_initial_usage_usd=active_key_initial_usage_usd,
        active_key_final_usage_usd=float(key_info["usage"]),
        active_key_limit_usd=float(key_info["limit"]),
        spend_ceiling_usd=spend_ceiling_usd,
    )
    audit["completed_at"] = datetime.now(UTC).isoformat()
    audit["artifacts"] = {
        "matrix_path": str(matrix_path),
        "matrix_sha256": _sha256(matrix_path),
        "matrix_summary_path": str(matrix_summary_path),
        "matrix_summary_sha256": _sha256(matrix_summary_path),
        "scorecard_path": str(scorecard_path),
        "scorecard_sha256": _sha256(scorecard_path),
        "memo_path": str(memo_path),
        "memo_sha256": _sha256(memo_path),
    }
    _write_json(output_path, audit)

    ledger["status"] = "collection_and_analysis_complete"
    ledger["completed_at"] = audit["completed_at"]
    ledger["key_usage_after_usd"] = audit["conceptual_combined_final_usage_usd"]
    ledger["exact_incremental_spend_usd"] = audit[
        "exact_incremental_provider_spend_usd"
    ]
    ledger["run_report_spend_usd"] = audit["run_report_spend_usd"]
    ledger["key_delta_minus_run_reports_usd"] = audit[
        "key_delta_minus_run_reports_usd"
    ]
    ledger["key_rollover"] = audit["keys"]
    ledger["final_artifacts"] = audit["artifacts"]
    ledger["execution_audit_path"] = str(output_path)
    _write_json(ledger_path, ledger)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finalize exact Swiss-cheese execution and rollover accounting"
    )
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--matrix-summary", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--scorecard", type=Path, required=True)
    parser.add_argument("--memo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior-key-fingerprint", required=True)
    parser.add_argument("--prior-key-final-usage-usd", type=float, required=True)
    parser.add_argument("--active-key-initial-usage-usd", type=float, default=0.0)
    parser.add_argument("--spend-ceiling-usd", type=float, required=True)
    args = parser.parse_args()
    result = finalize_execution(
        ledger_path=args.ledger,
        matrix_summary_path=args.matrix_summary,
        matrix_path=args.matrix,
        scorecard_path=args.scorecard,
        memo_path=args.memo,
        output_path=args.output,
        prior_key_fingerprint=args.prior_key_fingerprint,
        prior_key_final_usage_usd=args.prior_key_final_usage_usd,
        active_key_initial_usage_usd=args.active_key_initial_usage_usd,
        spend_ceiling_usd=args.spend_ceiling_usd,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
