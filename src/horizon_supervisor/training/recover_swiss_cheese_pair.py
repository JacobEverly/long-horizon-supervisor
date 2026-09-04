from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from horizon_supervisor.benchmark.gate8 import run_gate8_pilot
from horizon_supervisor.benchmark.matched_outcomes import LEARNING_VALID_STATUSES
from horizon_supervisor.training.run_swiss_cheese_experiment import (
    _config_for_contract,
    _extract_run,
    _key_usage,
    _validate_frozen_code,
    _write_json,
)

EXPECTED_CONTRACT_SHA256 = (
    "8e4e1d21ff4506be7490851bbbe6340a291160a1770da1de3bfec1a746b98e9b"
)
SOURCE_LABEL = "existing-light-rep3"
RECOVERY_LABEL = "recovery-kimi-xrd-rep3"
EXPECTED_KEY = ("xrd-two-peak-fitting", "gate7/fixed-kimi", 3)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _row_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(row["task"]["source_task_name"]),
        str(row["model"]["route_id"]),
        int(row["provenance"]["replication_index"]),
    )


def consolidate_recovery_rows(
    source_rows: list[dict[str, Any]],
    recovery_rows: list[dict[str, Any]],
    *,
    replaces_outcome_id: str,
) -> list[dict[str, Any]]:
    if len(source_rows) != 36:
        raise ValueError("source replication must contain exactly 36 outcomes")
    if len(recovery_rows) != 1:
        raise ValueError("recovery run must contain exactly one outcome")

    invalid = [
        row
        for row in source_rows
        if row["outcome"]["status"] not in LEARNING_VALID_STATUSES
    ]
    if len(invalid) != 1:
        raise ValueError("source must contain exactly one infrastructure-invalid row")
    invalid_row = invalid[0]
    if _row_key(invalid_row) != EXPECTED_KEY:
        raise ValueError("recovery target is not the frozen Kimi xrd replication-3 pair")
    if invalid_row["outcome_id"] != replaces_outcome_id:
        raise ValueError("recovery contract does not name the invalid source outcome")
    if invalid_row["outcome"]["status"] != "infrastructure_error":
        raise ValueError("only an infrastructure_error may be recovered")

    replacement = recovery_rows[0]
    if _row_key(replacement) != EXPECTED_KEY:
        raise ValueError("recovery outcome does not match the frozen pair")
    if replacement["outcome"]["status"] not in LEARNING_VALID_STATUSES:
        raise ValueError("recovery outcome is not learning-valid")

    consolidated = [
        replacement if row["outcome_id"] == replaces_outcome_id else row
        for row in source_rows
    ]
    keys = [_row_key(row) for row in consolidated]
    if len(keys) != len(set(keys)) or len(consolidated) != 36:
        raise ValueError("recovered replication is not a unique 9x4 matrix")
    if any(
        row["outcome"]["status"] not in LEARNING_VALID_STATUSES
        for row in consolidated
    ):
        raise ValueError("recovered replication still contains invalid outcomes")
    return consolidated


def _summary(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    source_path: Path,
    recovery_path: Path,
) -> dict[str, Any]:
    route_counts = Counter(row["model"]["route_id"] for row in rows)
    status_counts = Counter(row["outcome"]["status"] for row in rows)
    return {
        "schema_version": "matched-model-outcome-summary.v1",
        "wave": 3,
        "record_count": len(rows),
        "expected_record_count": 36,
        "task_count": len({row["task"]["source_task_name"] for row in rows}),
        "route_counts": dict(sorted(route_counts.items())),
        "record_split_counts": {"development": len(rows)},
        "status_counts": dict(sorted(status_counts.items())),
        "verified_completion_count": sum(
            bool(row["outcome"]["completed"]) for row in rows
        ),
        "cost_attributed_record_count": sum(
            row["outcome"].get("allocated_provider_cost_usd") is not None
            for row in rows
        ),
        "allocated_provider_cost_total_usd": sum(
            float(row["outcome"].get("allocated_provider_cost_usd") or 0.0)
            for row in rows
        ),
        "provider_spend_trusted": True,
        "record_split_override": "development",
        "evaluation_role": "posthoc_clean_start_replication",
        "replication_index": 3,
        "missing_pair_count": 0,
        "missing_pairs": [],
        "all_pairs_present_once": True,
        "all_pairs_present_once_and_learning_valid": True,
        "infrastructure_errors_are_not_failures": True,
        "infrastructure_recovery": {
            "source_path": str(source_path),
            "recovery_path": str(recovery_path),
            "replaced_key": list(EXPECTED_KEY),
        },
        "outcome_path": str(output_path),
    }


def _daytona_sandbox_ids() -> list[str]:
    from daytona import Daytona

    return [str(sandbox.id) for sandbox in Daytona().list()]


def _existing_learning_valid_recovery(
    recovery_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]] | None:
    if not recovery_path.exists():
        return None
    rows = _read_rows(recovery_path)
    if (
        len(rows) != 1
        or _row_key(rows[0]) != EXPECTED_KEY
        or rows[0]["outcome"]["status"] not in LEARNING_VALID_STATUSES
    ):
        return None
    result_path = Path(rows[0]["provenance"]["result_path"])
    run_root = result_path.parents[3]
    report_path = run_root / "report.json"
    summary_path = recovery_path.with_name(f"{RECOVERY_LABEL}-summary.json")
    if not report_path.exists() or not summary_path.exists():
        raise RuntimeError("sealed recovery outcome is missing its report or summary")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if report.get("status") != "complete" or not summary.get(
        "all_pairs_present_once"
    ):
        raise RuntimeError("existing recovery artifacts are not sealed")
    return rows, report, summary


def run_recovery(root: Path, manifest_path: Path, contract_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    output_root = manifest_path.parent
    ledger_path = output_root / "execution-ledger-v0.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))

    _validate_frozen_code(root, manifest)
    if _sha256(contract_path) != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError("infrastructure-recovery contract changed")
    if not contract["replication_design"].get("infrastructure_recovery_only"):
        raise RuntimeError("contract is not restricted to infrastructure recovery")
    if contract["replication_design"]["run_label"] != RECOVERY_LABEL:
        raise RuntimeError("unexpected recovery label")

    source_runs = [row for row in ledger["runs"] if row["label"] == SOURCE_LABEL]
    if len(source_runs) != 1 or source_runs[0]["status"] != "needs_infrastructure_recovery":
        raise RuntimeError("source run is not awaiting exactly one recovery")
    if any(row["label"] == RECOVERY_LABEL for row in ledger["runs"]):
        raise RuntimeError("recovery label already exists in the execution ledger")

    source_path = Path(source_runs[0]["outcomes_path"])
    source_rows = _read_rows(source_path)
    replaces_outcome_id = str(
        contract["replication_design"]["replaces_outcome_id"]
    )
    # Validate the source before any paid call. A synthetic learning-valid row is
    # intentionally rejected here so a successful pair can never be rerun.
    consolidate_recovery_rows(
        source_rows,
        [
            {
                **next(row for row in source_rows if row["outcome_id"] == replaces_outcome_id),
                "outcome": {
                    **next(
                        row
                        for row in source_rows
                        if row["outcome_id"] == replaces_outcome_id
                    )["outcome"],
                    "status": "verified",
                },
            }
        ],
        replaces_outcome_id=replaces_outcome_id,
    )

    recovery_path = output_root / "replication-inputs" / f"{RECOVERY_LABEL}.jsonl"
    recovery_summary_path = recovery_path.with_name(f"{RECOVERY_LABEL}-summary.json")
    existing = _existing_learning_valid_recovery(recovery_path)
    if existing is None:
        stale_ids = _daytona_sandbox_ids()
        if stale_ids:
            raise RuntimeError(
                "Daytona preflight requires zero sandboxes; clean these exact ids first: "
                + ", ".join(stale_ids)
            )
        config = _config_for_contract(root, output_root, contract_path, contract)
        report = run_gate8_pilot(config)
        recovery_summary, recovery_path, recovery_summary_path = _extract_run(
            root,
            output_root,
            RECOVERY_LABEL,
            3,
            report,
            ("gate7/fixed-kimi",),
        )
        recovery_rows = _read_rows(recovery_path)
    else:
        recovery_rows, report, recovery_summary = existing
    consolidated = consolidate_recovery_rows(
        source_rows,
        recovery_rows,
        replaces_outcome_id=replaces_outcome_id,
    )
    if report["status"] != "complete" or not recovery_summary["all_pairs_present_once"]:
        raise RuntimeError("recovery run did not produce one complete pair")

    remaining_ids = _daytona_sandbox_ids()
    if remaining_ids:
        raise RuntimeError(
            "recovery finished but Daytona cleanup is incomplete: "
            + ", ".join(remaining_ids)
        )

    backup_path = source_path.with_name(f"{SOURCE_LABEL}-pre-recovery.jsonl")
    if backup_path.exists() and _sha256(backup_path) != _sha256(source_path):
        raise RuntimeError("existing pre-recovery backup does not match source")
    if not backup_path.exists():
        shutil.copy2(source_path, backup_path)
    temporary = source_path.with_suffix(source_path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in consolidated),
        encoding="utf-8",
    )
    temporary.replace(source_path)
    summary_path = source_path.with_name(f"{SOURCE_LABEL}-summary.json")
    consolidated_summary = _summary(
        consolidated,
        source_path,
        source_path=backup_path,
        recovery_path=recovery_path,
    )
    _write_json(summary_path, consolidated_summary)

    source_run = source_runs[0]
    source_run["status"] = "sealed"
    source_run["outcome_status_counts"] = consolidated_summary["status_counts"]
    source_run["original_outcomes_path"] = str(backup_path)
    source_run["infrastructure_recovery_label"] = RECOVERY_LABEL
    source_run["replaced_outcome_id"] = replaces_outcome_id
    source_run["replacement_outcome_id"] = recovery_rows[0]["outcome_id"]
    ledger["runs"].append(
        {
            "label": RECOVERY_LABEL,
            "replication_index": 3,
            "status": "sealed",
            "recovery_only": True,
            "contract_path": str(contract_path),
            "contract_sha256": _sha256(contract_path),
            "report_path": str(Path(report["pricing_snapshot"]).parent / "report.json"),
            "outcomes_path": str(recovery_path),
            "summary_path": str(recovery_summary_path),
            "provider_spend_usd": report.get("provider_spend_usd"),
            "outcome_status_counts": recovery_summary["status_counts"],
            "missing_pairs": recovery_summary["missing_pairs"],
            "replaces_outcome_id": replaces_outcome_id,
            "replacement_outcome_id": recovery_rows[0]["outcome_id"],
        }
    )
    ledger["key_usage_latest_usd"] = _key_usage()
    ledger["last_recovery_at"] = datetime.now(UTC).isoformat()
    _write_json(ledger_path, ledger)
    return {
        "status": "sealed",
        "recovery_label": RECOVERY_LABEL,
        "replacement_outcome_id": recovery_rows[0]["outcome_id"],
        "consolidated_outcomes_path": str(source_path),
        "ledger_path": str(ledger_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recover the one frozen infrastructure-invalid Swiss-cheese pair"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "artifacts/official/swiss-cheese-replication-v0/"
            "frozen-experiment-manifest-v0.json"
        ),
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(
            "benchmarks/swiss-cheese-replication-v0/"
            "recovery-kimi-xrd-rep3.json"
        ),
    )
    args = parser.parse_args()
    result = run_recovery(
        args.root.resolve(), args.manifest.resolve(), args.contract.resolve()
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
