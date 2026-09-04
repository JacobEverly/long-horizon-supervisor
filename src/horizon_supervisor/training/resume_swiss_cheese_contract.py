from __future__ import annotations

import argparse
import copy
import json
import time
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


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _row_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(row["task"]["source_task_name"]),
        str(row["model"]["route_id"]),
        int(row["provenance"]["replication_index"]),
    )


def accounted_key_usage(current_key_usage_usd: float, prior_key_usage_usd: float) -> float:
    if current_key_usage_usd < 0 or prior_key_usage_usd < 0:
        raise ValueError("key usage accounting values must be non-negative")
    return current_key_usage_usd + prior_key_usage_usd


def select_locked_and_pending(
    source_rows: list[dict[str, Any]],
    progress_rows: list[dict[str, Any]],
    expected_keys: list[tuple[str, str, int]],
) -> tuple[dict[tuple[str, str, int], dict[str, Any]], list[tuple[str, str, int]]]:
    expected = set(expected_keys)
    if len(expected) != len(expected_keys):
        raise ValueError("expected task-route-replication keys are not unique")
    locked: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in [*source_rows, *progress_rows]:
        key = _row_key(row)
        if key not in expected:
            raise ValueError(f"outcome is outside the frozen contract: {key}")
        if row["outcome"]["status"] not in LEARNING_VALID_STATUSES:
            continue
        previous = locked.get(key)
        if previous is not None and previous["outcome_id"] != row["outcome_id"]:
            raise ValueError(f"multiple learning-valid outcomes for frozen pair: {key}")
        locked[key] = row
    pending = [key for key in expected_keys if key not in locked]
    return locked, pending


def _summary(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    source_path: Path,
    progress_path: Path,
    replication_index: int,
) -> dict[str, Any]:
    route_counts = Counter(row["model"]["route_id"] for row in rows)
    status_counts = Counter(row["outcome"]["status"] for row in rows)
    return {
        "schema_version": "matched-model-outcome-summary.v1",
        "wave": 3,
        "record_count": len(rows),
        "expected_record_count": len(rows),
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
        "replication_index": replication_index,
        "missing_pair_count": 0,
        "missing_pairs": [],
        "all_pairs_present_once": True,
        "all_pairs_present_once_and_learning_valid": True,
        "infrastructure_errors_are_not_failures": True,
        "resumed_collection": {
            "source_path": str(source_path),
            "progress_path": str(progress_path),
            "valid_pairs_were_never_rerun": True,
        },
        "outcome_path": str(output_path),
    }


def _daytona_sandbox_ids() -> list[str]:
    from daytona import Daytona

    return [str(sandbox.id) for sandbox in Daytona().list()]


def _wait_for_daytona_cleanup(timeout_seconds: int = 180) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        sandbox_ids = _daytona_sandbox_ids()
        if not sandbox_ids:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "Daytona cleanup did not finish for exact sandbox ids: "
                + ", ".join(sandbox_ids)
            )
        time.sleep(5)


def _pair_contract(
    original: dict[str, Any],
    *,
    label: str,
    source_label: str,
    task: str,
    route: str,
    replication_index: int,
    replaces_outcome_id: str | None,
) -> dict[str, Any]:
    contract = copy.deepcopy(original)
    contract["frozen_at"] = datetime.now(UTC).date().isoformat()
    contract["scope"] = (
        "Infrastructure-only recovery or missing-pair continuation for one frozen "
        "Swiss-cheese task-route-replication pair"
    )
    contract["trial_count"] = 1
    contract["replication_design"].update(
        {
            "run_label": label,
            "source_run_label": source_label,
            "replication_index": replication_index,
            "source_record_count": 1,
            "outcome_blind": True,
            "policy_tuning_during_collection": False,
            "clean_start_only": True,
            "infrastructure_recovery_only": replaces_outcome_id is not None,
            "missing_pair_continuation_only": replaces_outcome_id is None,
            "replaces_outcome_id": replaces_outcome_id,
        }
    )
    contract["execution_contract"]["route_ids"] = [route]
    contract["execution_contract"]["selected_task_names"] = [task]
    contract["execution_contract"]["run_controls"]["n_concurrent"] = 1
    contract["interpretation_guard"] = (
        "This generated contract may collect only the named frozen pair. A learning-"
        "valid pair is locked immediately and may never be rerun."
    )
    return contract


def _append_attempt_to_ledger(
    ledger: dict[str, Any],
    ledger_path: Path,
    *,
    label: str,
    replication_index: int,
    contract_path: Path,
    report: dict[str, Any],
    outcomes_path: Path,
    summary_path: Path,
    row: dict[str, Any],
    replaces_outcome_id: str | None,
    current_key_usage_usd: float,
    prior_key_usage_usd: float,
) -> None:
    ledger["runs"].append(
        {
            "label": label,
            "replication_index": replication_index,
            "status": (
                "sealed"
                if row["outcome"]["status"] in LEARNING_VALID_STATUSES
                else "needs_infrastructure_recovery"
            ),
            "infrastructure_recovery_or_continuation_only": True,
            "contract_path": str(contract_path),
            "report_path": str(Path(report["pricing_snapshot"]).parent / "report.json"),
            "outcomes_path": str(outcomes_path),
            "summary_path": str(summary_path),
            "provider_spend_usd": report.get("provider_spend_usd"),
            "outcome_status_counts": {row["outcome"]["status"]: 1},
            "missing_pairs": [],
            "replaces_outcome_id": replaces_outcome_id,
            "replacement_outcome_id": row["outcome_id"],
        }
    )
    ledger["active_key_usage_usd"] = current_key_usage_usd
    ledger["active_key_usage_offset_usd"] = prior_key_usage_usd
    ledger["key_usage_latest_usd"] = accounted_key_usage(
        current_key_usage_usd, prior_key_usage_usd
    )
    _write_json(ledger_path, ledger)


def run_resume(
    root: Path,
    manifest_path: Path,
    source_label: str,
    *,
    max_infrastructure_attempts: int = 3,
    prior_key_usage_usd: float = 0.0,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_frozen_code(root, manifest)
    output_root = manifest_path.parent
    ledger_path = output_root / "execution-ledger-v0.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    specs = [row for row in manifest["execution"]["contracts"] if row["label"] == source_label]
    if len(specs) != 1:
        raise RuntimeError("source label is not exactly one frozen contract")
    spec = specs[0]
    source_runs = [row for row in ledger["runs"] if row["label"] == source_label]
    if len(source_runs) != 1 or source_runs[0]["status"] != "needs_infrastructure_recovery":
        raise RuntimeError("source run is not awaiting recovery/continuation")
    original_contract_path = root / spec["contract_path"]
    original = json.loads(original_contract_path.read_text(encoding="utf-8"))
    replication_index = int(spec["replication_index"])
    expected_keys = [
        (task, route, replication_index)
        for task in spec["tasks"]
        for route in spec["routes"]
    ]
    source_path = Path(source_runs[0]["outcomes_path"])
    progress_path = output_root / "resume-progress" / f"{source_label}.jsonl"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    generated_root = output_root / "generated-contracts"
    generated_root.mkdir(parents=True, exist_ok=True)
    source_rows = _read_rows(source_path)
    progress_rows = _read_rows(progress_path)
    locked, pending = select_locked_and_pending(source_rows, progress_rows, expected_keys)
    invalid_by_key = {
        _row_key(row): row
        for row in source_rows
        if row["outcome"]["status"] not in LEARNING_VALID_STATUSES
    }
    spend_ceiling = float(manifest["budget"]["dedicated_key_usage_ceiling_usd"])
    reserve = float(original["model_budget"]["minimum_next_task_reserve_usd"])

    for key in pending:
        pair_index = expected_keys.index(key) + 1
        task, route, _ = key
        replaces = invalid_by_key.get(key, {}).get("outcome_id")
        label_prefix = f"resume-{source_label}-{pair_index:02d}-a"
        previous_attempts = [
            int(row["label"].removeprefix(label_prefix))
            for row in ledger["runs"]
            if row["label"].startswith(label_prefix)
            and row["label"].removeprefix(label_prefix).isdigit()
        ]
        first_attempt = max(previous_attempts, default=0) + 1
        for attempt in range(
            first_attempt, first_attempt + max_infrastructure_attempts
        ):
            current_usage = _key_usage()
            usage = accounted_key_usage(current_usage, prior_key_usage_usd)
            if usage + reserve > spend_ceiling:
                raise RuntimeError("experiment reserve would exceed the frozen spend ceiling")
            if _daytona_sandbox_ids():
                raise RuntimeError("Daytona preflight requires zero sandboxes")
            label = f"resume-{source_label}-{pair_index:02d}-a{attempt}"
            contract_path = generated_root / f"{label}.json"
            contract = _pair_contract(
                original,
                label=label,
                source_label=source_label,
                task=task,
                route=route,
                replication_index=replication_index,
                replaces_outcome_id=str(replaces) if replaces else None,
            )
            _write_json(contract_path, contract)
            report = run_gate8_pilot(
                _config_for_contract(root, output_root, contract_path, contract)
            )
            summary, outcomes_path, summary_path = _extract_run(
                root,
                output_root,
                label,
                replication_index,
                report,
                (route,),
            )
            rows = _read_rows(outcomes_path)
            if len(rows) != 1 or _row_key(rows[0]) != key:
                raise RuntimeError("single-pair continuation produced the wrong outcome")
            row = rows[0]
            current_usage = _key_usage()
            _append_attempt_to_ledger(
                ledger,
                ledger_path,
                label=label,
                replication_index=replication_index,
                contract_path=contract_path,
                report=report,
                outcomes_path=outcomes_path,
                summary_path=summary_path,
                row=row,
                replaces_outcome_id=str(replaces) if replaces else None,
                current_key_usage_usd=current_usage,
                prior_key_usage_usd=prior_key_usage_usd,
            )
            _wait_for_daytona_cleanup()
            if accounted_key_usage(_key_usage(), prior_key_usage_usd) > spend_ceiling:
                raise RuntimeError("frozen Swiss-cheese spend ceiling was exceeded")
            if row["outcome"]["status"] in LEARNING_VALID_STATUSES:
                locked[key] = row
                _write_rows(progress_path, [locked[k] for k in expected_keys if k in locked])
                break
            replaces = row["outcome_id"]
        else:
            raise RuntimeError(f"infrastructure retry limit reached for {key}")

    if set(locked) != set(expected_keys):
        raise RuntimeError("resume did not produce every frozen pair")
    consolidated = [locked[key] for key in expected_keys]
    backup_path = source_path.with_name(f"{source_label}-pre-resume.jsonl")
    if not backup_path.exists():
        _write_rows(backup_path, source_rows)
    _write_rows(source_path, consolidated)
    summary_path = Path(source_runs[0]["summary_path"])
    consolidated_summary = _summary(
        consolidated,
        source_path,
        source_path=backup_path,
        progress_path=progress_path,
        replication_index=replication_index,
    )
    _write_json(summary_path, consolidated_summary)
    source_run = source_runs[0]
    source_run["status"] = "sealed"
    source_run["outcome_status_counts"] = consolidated_summary["status_counts"]
    source_run["missing_pairs"] = []
    source_run["original_outcomes_path"] = str(backup_path)
    source_run["resume_progress_path"] = str(progress_path)
    source_run["resumed_without_rerunning_valid_pairs"] = True
    current_usage = _key_usage()
    ledger["active_key_usage_usd"] = current_usage
    ledger["active_key_usage_offset_usd"] = prior_key_usage_usd
    ledger["key_usage_latest_usd"] = accounted_key_usage(
        current_usage, prior_key_usage_usd
    )
    ledger["last_recovery_at"] = datetime.now(UTC).isoformat()
    _write_json(ledger_path, ledger)
    return {
        "status": "sealed",
        "source_label": source_label,
        "outcome_count": len(consolidated),
        "completed_count": sum(bool(row["outcome"]["completed"]) for row in consolidated),
        "outcomes_path": str(source_path),
        "ledger_path": str(ledger_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume one interrupted frozen Swiss-cheese execution contract"
    )
    parser.add_argument("source_label")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "artifacts/official/swiss-cheese-replication-v0/"
            "frozen-experiment-manifest-v0.json"
        ),
    )
    parser.add_argument("--max-infrastructure-attempts", type=int, default=3)
    parser.add_argument(
        "--prior-key-usage-usd",
        type=float,
        default=0.0,
        help="frozen usage on expired earlier experiment keys",
    )
    args = parser.parse_args()
    result = run_resume(
        args.root.resolve(),
        args.manifest.resolve(),
        args.source_label,
        max_infrastructure_attempts=args.max_infrastructure_attempts,
        prior_key_usage_usd=args.prior_key_usage_usd,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
