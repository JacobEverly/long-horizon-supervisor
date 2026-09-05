from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from daytona import Daytona
from switchyard.cli.launchers.native_server import NativeServer

from horizon_supervisor.benchmark.continuation_harbor import record_rows
from horizon_supervisor.benchmark.gate7 import query_openrouter_key
from horizon_supervisor.training.analyze_continuation_calibration import analyze
from horizon_supervisor.training.freeze_continuation_calibration import (
    EXACT_MODELS,
    ROOT,
    SELECTED_CONFIG,
    SWITCHYARD,
    _sha256,
    _tree_sha256,
)
from horizon_supervisor.training.freeze_continuation_calibration_v1 import (
    EXPECTED_V0_MANIFEST_SHA256,
    OUTPUT_ROOT,
    V0_FAILURE,
    V0_MANIFEST,
)
from horizon_supervisor.training.run_continuation_calibration import (
    _attempt_usage_record,
    _cleanup_new_sandboxes,
    _get_json,
    _harbor_command,
    _outcome_row,
    _post_json,
    _provider_error,
    _retryable_infrastructure_failure,
    _run_command,
    _runtime_ceiling,
    _task_parent,
    _trial_result,
    _valid_trial,
    _validate_checkpoint,
    _write_json,
    _write_jsonl,
    validate_key_budget,
)

MANIFEST = OUTPUT_ROOT / "frozen-manifest-v1.json"
STATE_PATH = OUTPUT_ROOT / "execution-state-v1.json"
OUTCOMES_PATH = OUTPUT_ROOT / "natural-continuation-outcomes-v1.jsonl"
USAGE_PATH = OUTPUT_ROOT / "trial-usage-ledger-v1.jsonl"
CHECKPOINT_INDEX = OUTPUT_ROOT / "checkpoint-bank-index-v1.jsonl"
FIDELITY_REPORT = OUTPUT_ROOT / "snapshot-fidelity-v1.json"
LEDGER_PATH = OUTPUT_ROOT / "execution-ledger-v1.json"
EXPECTED_SCHEMA = "two-tier-continuation-calibration-manifest.v1"


def validate_manifest(
    manifest_path: Path = MANIFEST,
) -> tuple[dict[str, Any], str]:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sidecar = manifest_path.with_suffix(".sha256")
    if not sidecar.is_file():
        raise RuntimeError("frozen v1 manifest sidecar is missing")
    expected_hash = sidecar.read_text(encoding="utf-8").split()[0]
    if _sha256(manifest_path) != expected_hash:
        raise RuntimeError("frozen continuation v1 manifest hash mismatch")
    if manifest.get("schema_version") != EXPECTED_SCHEMA:
        raise RuntimeError("unexpected continuation v1 manifest schema")
    if manifest.get("frozen_before_model_outcomes") is not True:
        raise RuntimeError("v1 manifest was not frozen before outcomes")
    if manifest["lineage"] != {
        "v0_manifest_sha256": EXPECTED_V0_MANIFEST_SHA256,
        "v0_failure_sha256": _sha256(V0_FAILURE),
        "v0_terminal_outcome_count": 0,
        "minimal_revision": (
            "replace absolute workspace-process rejection with pre/post "
            "agent-action process-delta tracking"
        ),
        "detector_thresholds_changed": False,
    }:
        raise RuntimeError("v1 lineage or minimal revision changed")
    if manifest["models"]["routes"] != EXACT_MODELS:
        raise RuntimeError("frozen route/model mapping changed")
    if manifest["detector"]["config"] != SELECTED_CONFIG.model_dump():
        raise RuntimeError("frozen detector thresholds changed")
    if manifest["process_reproducibility"] != {
        "schema_version": "action-process-delta.v1",
        "baseline": "process identities observed immediately before each action",
        "unmanaged": (
            "new non-shell process identities rooted in the task workdir, "
            "carried until exit"
        ),
        "platform_processes_observed_before_action": "reproducible harness state",
        "threshold_or_signal_changes": False,
    }:
        raise RuntimeError("v1 process-reproducibility contract changed")
    if manifest["execution"] != {
        "agent": "ProcessDeltaContinuationTerminus2",
        "max_turns": 12,
        "healthy_checkpoint_turn": 4,
        "natural_continuation_only": True,
        "interventions_forbidden": True,
        "infrastructure_retries_per_trial": 1,
        "state_fidelity": "fresh SeededDaytonaEnvironment rehydration per checkpoint",
    }:
        raise RuntimeError("natural-continuation v1 execution contract changed")
    budget = manifest["budget"]
    if (
        float(budget["original_project_baseline_usd"]) != 49.257980597
        or float(budget["project_openrouter_spend_before_usd"]) != 49.259005378
        or float(budget["project_cumulative_ceiling_usd"]) != 200.0
        or float(budget["phase_a_incremental_ceiling_usd"]) != 4.99
        or float(budget["tranche_1_incremental_ceiling_usd"]) != 2.5
        or float(budget["per_trial_incremental_ceiling_usd"]) != 0.5
    ):
        raise RuntimeError("frozen v1 budget changed")
    tasks = manifest["task_selection"]["ordered_pool"]
    if len(tasks) != 16 or [row["position"] for row in tasks] != list(range(1, 17)):
        raise RuntimeError("frozen v1 task pool changed")
    if any(row.get("prior_terminal_outcome_count") != 0 for row in tasks):
        raise RuntimeError("v1 pool includes a task with a prior terminal outcome")
    for task in tasks:
        task_root = ROOT / task["task_root"]
        if not task_root.is_dir() or _tree_sha256(task_root) != task["task_tree_sha256"]:
            raise RuntimeError(f"frozen task tree changed: {task['task_id']}")
    fixed_files = {
        SWITCHYARD: manifest["integrity"]["switchyard_sha256"],
        V0_MANIFEST: EXPECTED_V0_MANIFEST_SHA256,
        V0_FAILURE: manifest["lineage"]["v0_failure_sha256"],
        ROOT / manifest["models"]["catalog_path"]: manifest["models"][
            "catalog_sha256"
        ],
        ROOT / manifest["task_selection"]["eligibility_path"]: manifest[
            "task_selection"
        ]["eligibility_sha256"],
        ROOT / manifest["task_selection"]["replacement_lock_path"]: manifest[
            "task_selection"
        ]["replacement_lock_sha256"],
    }
    fixed_files.update(
        {
            ROOT / relative: digest
            for relative, digest in manifest["integrity"]["code_sha256"].items()
        }
    )
    for path, digest in fixed_files.items():
        if not path.is_file() or _sha256(path) != digest:
            raise RuntimeError(f"frozen v1 input hash mismatch: {path}")
    return manifest, expected_hash


def _continuation_command(
    *,
    manifest: dict[str, Any],
    server: NativeServer,
    run_root: Path,
    task: dict[str, Any],
    route_id: str,
    model_id: str,
    job_name: str,
    record_path: Path,
    provider_usage_start: float,
    provider_usage_ceiling: float,
) -> list[str]:
    command = _harbor_command(
        task_parent=_task_parent(task),
        task_id=task["task_id"],
        route_id=route_id,
        model_id=model_id,
        job_name=job_name,
        jobs_dir=run_root / "jobs",
        record_path=record_path,
        max_turns=int(manifest["execution"]["max_turns"]),
        agent_timeout_seconds=3_600,
        provider_usage_start=provider_usage_start,
        stats_url=f"{server.base_url}/v1/stats",
        capture_healthy=False,
        capture_stuck=False,
        provider_usage_ceiling=provider_usage_ceiling,
        stop_after_checkpoint=False,
        stop_after_healthy_window=False,
    )
    command[command.index("--agent") + 1] = (
        "horizon_supervisor.benchmark.continuation_harbor_v1:"
        "ProcessDeltaContinuationTerminus2"
    )
    include_index = command.index("--include-task-name")
    command[include_index:include_index] = [
        "--agent-kwarg",
        f"continuation_detector_config={json.dumps(SELECTED_CONFIG.model_dump())}",
        "--agent-kwarg",
        f"continuation_task_category={task['task_category']}",
    ]
    return command


def _run_trial(
    *,
    manifest: dict[str, Any],
    server: NativeServer,
    run_root: Path,
    task: dict[str, Any],
    route_id: str,
    model_id: str,
    baseline: float,
) -> dict[str, Any]:
    api_key = os.environ["OPENROUTER_API_KEY"]
    before = query_openrouter_key(api_key)
    validate_key_budget(manifest, before, baseline=baseline)
    usage_before = float(before["usage"])
    ceiling = _runtime_ceiling(manifest, baseline, int(task["tranche"]))
    if ceiling - usage_before < float(
        manifest["budget"]["per_trial_incremental_ceiling_usd"]
    ):
        raise RuntimeError("frozen OpenRouter v1 tranche ceiling lacks one reserve")
    _post_json(f"{server.base_url}/v1/stats/reset")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    label = route_id.rsplit("-", 1)[-1]
    job_name = f"continuation-v1-{task['position']:02d}-{label}-{timestamp}"
    record_path = run_root / "records" / f"{job_name}.jsonl"
    command = _continuation_command(
        manifest=manifest,
        server=server,
        run_root=run_root,
        task=task,
        route_id=route_id,
        model_id=model_id,
        job_name=job_name,
        record_path=record_path,
        provider_usage_start=usage_before,
        provider_usage_ceiling=ceiling,
    )
    environment = os.environ.copy()
    environment["OPENAI_API_KEY"] = "switchyard-local"
    environment["OPENAI_BASE_URL"] = f"{server.base_url}/v1"
    environment["HORIZON_HARBOR_LLM_ATTEMPTS"] = "1"
    environment["HORIZON_HARBOR_OUTPUT_LENGTH_RETRIES"] = "1"
    return_code, output, timed_out = _run_command(
        command, environment=environment, timeout=5_100
    )
    log_path = run_root / "logs" / f"{job_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(output, encoding="utf-8")
    stats = _get_json(f"{server.base_url}/v1/stats")
    after = query_openrouter_key(api_key)
    usage_after = float(after["usage"])
    result = _trial_result(run_root / "jobs" / job_name)
    return {
        "job_name": job_name,
        "task_id": task["task_id"],
        "route_id": route_id,
        "model_id": model_id,
        "return_code": return_code,
        "timed_out": timed_out,
        "provider_usage_before_usd": usage_before,
        "provider_usage_after_usd": usage_after,
        "provider_spend_usd": max(0.0, usage_after - usage_before),
        "key_query_error": None,
        "stats": stats,
        "record_path": str(record_path),
        "job_dir": str(run_root / "jobs" / job_name),
        "result": result,
        "valid": _valid_trial(
            return_code=return_code, timed_out=timed_out, result=result
        ),
    }


def _run_with_retry(**kwargs: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    attempts = []
    for _ in range(2):
        trial = _run_trial(**kwargs)
        attempts.append(trial)
        if trial["valid"] or not _retryable_infrastructure_failure(trial):
            break
    return attempts[-1], attempts


def _structural_runtime_failure(
    trial: dict[str, Any], records: list[dict[str, Any]]
) -> bool:
    return bool(
        _provider_error(trial.get("result"))
        or any(
            row.get("schema_version") == "pilot-budget-halt.v0" for row in records
        )
        or any(
            row.get("assessment", {}).get("status") == "STRUCTURAL_FAILURE"
            for row in records
        )
    )


def _v1_outcome_row(
    *,
    task: dict[str, Any],
    trial: dict[str, Any],
    fidelity_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    row = _outcome_row(task=task, trial=trial, fidelity_rows=fidelity_rows)
    records = record_rows(Path(trial["record_path"]))
    if not _structural_runtime_failure(trial, records):
        return row
    row["structural_failure"] = True
    row["discarded_checkpoint_count"] += len(row["checkpoints"])
    row["checkpoints"] = []
    return row


def _new_state(manifest_hash: str, usage: float, hard_limit: float) -> dict[str, Any]:
    return {
        "schema_version": "continuation-calibration-execution-state.v1",
        "manifest_sha256": manifest_hash,
        "status": "in_progress",
        "started_at": datetime.now(UTC).isoformat(),
        "provider_usage_baseline_usd": usage,
        "provider_hard_limit_usd": hard_limit,
        "completed_schedule_items": [],
        "outcomes": [],
        "attempts": [],
        "fidelity_rows": [],
        "tranche_reports": [],
    }


def _persist(state: dict[str, Any]) -> None:
    state["updated_at"] = datetime.now(UTC).isoformat()
    _write_json(STATE_PATH, state)
    _write_jsonl(OUTCOMES_PATH, state["outcomes"])
    _write_jsonl(USAGE_PATH, state["attempts"])
    checkpoint_rows = [
        checkpoint
        | {
            "task_id": row["task_id"],
            "route_id": row["route_id"],
            "model_id": row["model_id"],
        }
        for row in state["outcomes"]
        for checkpoint in row["checkpoints"]
        if checkpoint["state_transfer_eligible"]
        and checkpoint["snapshot_fidelity_passed"]
    ]
    _write_jsonl(CHECKPOINT_INDEX, checkpoint_rows)
    _write_json(
        FIDELITY_REPORT,
        {
            "schema_version": "continuation-snapshot-fidelity-report.v1",
            "attempt_count": len(state["fidelity_rows"]),
            "pass_count": sum(row["passed"] for row in state["fidelity_rows"]),
            "all_passed": bool(state["fidelity_rows"])
            and all(row["passed"] for row in state["fidelity_rows"]),
            "rows": state["fidelity_rows"],
        },
    )


def _wait_for_cleanup(initial_sandboxes: set[str]) -> tuple[list[str], list[str]]:
    errors = []
    remaining: list[str] = []
    for _ in range(12):
        try:
            remaining = sorted(
                {sandbox.id for sandbox in Daytona().list()} - initial_sandboxes
            )
        except Exception as error:  # pragma: no cover - live recovery path
            errors.append(f"final list: {type(error).__name__}: {error}")
            break
        if not remaining:
            break
        time.sleep(2)
    return remaining, errors


def run(manifest_path: Path = MANIFEST) -> dict[str, Any]:
    manifest, manifest_hash = validate_manifest(manifest_path)
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or not os.getenv("DAYTONA_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY and DAYTONA_API_KEY are required")
    initial_sandboxes = {sandbox.id for sandbox in Daytona().list()}
    if initial_sandboxes:
        raise RuntimeError("continuation calibration v1 requires zero sandboxes")
    key_info = query_openrouter_key(api_key)
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if state["manifest_sha256"] != manifest_hash:
            raise RuntimeError("v1 execution state belongs to another manifest")
    else:
        state = _new_state(
            manifest_hash, float(key_info["usage"]), float(key_info["limit"])
        )
        _persist(state)
    baseline = float(state["provider_usage_baseline_usd"])
    validate_key_budget(manifest, key_info, baseline=baseline)

    server = NativeServer(SWITCHYARD)
    status = "in_progress"
    stop_reason = None
    execution_error = None
    completed = set(state["completed_schedule_items"])
    try:
        for tranche in (1, 2):
            for task in manifest["task_selection"]["ordered_pool"]:
                if int(task["tranche"]) != tranche:
                    continue
                for route_id, model_id in EXACT_MODELS.items():
                    schedule_item = f"{task['position']}:{route_id}"
                    if schedule_item in completed:
                        continue
                    trial, attempts = _run_with_retry(
                        manifest=manifest,
                        server=server,
                        run_root=OUTPUT_ROOT,
                        task=task,
                        route_id=route_id,
                        model_id=model_id,
                        baseline=baseline,
                    )
                    state["attempts"].extend(
                        _attempt_usage_record(attempt) for attempt in attempts
                    )
                    records = record_rows(Path(trial["record_path"]))
                    checkpoints = [
                        row
                        for row in records
                        if row.get("schema_version") == "matched-checkpoint.v0"
                        and row.get("state_transfer_eligible") is True
                    ]
                    fidelity_rows = []
                    if trial["valid"] and not _structural_runtime_failure(
                        trial, records
                    ):
                        for checkpoint in checkpoints:
                            fidelity = _validate_checkpoint(
                                task=task,
                                checkpoint=checkpoint,
                                run_root=OUTPUT_ROOT,
                            )
                            fidelity_rows.append(fidelity)
                            state["fidelity_rows"].append(fidelity)
                    state["outcomes"].append(
                        _v1_outcome_row(
                            task=task, trial=trial, fidelity_rows=fidelity_rows
                        )
                    )
                    state["completed_schedule_items"].append(schedule_item)
                    completed.add(schedule_item)
                    _persist(state)
            tranche_report = analyze(state["outcomes"])
            tranche_report["schema_version"] = (
                "two-tier-continuation-calibration-report.v1"
            )
            tranche_report["tranche"] = tranche
            state["tranche_reports"].append(tranche_report)
            _persist(state)
            if tranche_report["gate_passed"]:
                status = "complete"
                stop_reason = "continuation_calibration_gate_passed"
                break
        else:
            status = "complete"
            stop_reason = "frozen_task_pool_exhausted"
    except RuntimeError as error:
        status = "stopped"
        execution_error = f"{type(error).__name__}: {error}"
        stop_reason = (
            "frozen_openrouter_spend_ceiling_reached"
            if "ceiling" in str(error) or "budget" in str(error)
            else "execution_runtime_error"
        )
    except (urllib.error.HTTPError, urllib.error.URLError) as error:
        status = "stopped"
        stop_reason = "provider_key_unavailable"
        execution_error = f"{type(error).__name__}: {error}"
    except KeyboardInterrupt:
        status = "interrupted"
        stop_reason = "operator_interrupt"
    finally:
        server.close()
        cleanup = _cleanup_new_sandboxes(initial_sandboxes)

    try:
        final_info = query_openrouter_key(api_key)
        usage_after = float(final_info["usage"])
        final_query_error = None
    except Exception as error:  # pragma: no cover - live recovery path
        usage_after = baseline
        final_query_error = f"{type(error).__name__}: {error}"
    remaining, final_cleanup_errors = _wait_for_cleanup(initial_sandboxes)
    cleanup["errors"].extend(final_cleanup_errors)
    state["status"] = status
    state["stop_reason"] = stop_reason
    state["execution_error"] = execution_error
    _persist(state)
    incremental = max(0.0, usage_after - baseline)
    report = analyze(state["outcomes"])
    report["schema_version"] = "two-tier-continuation-calibration-report.v1"
    ledger = {
        "schema_version": "continuation-calibration-execution-ledger.v1",
        "status": status,
        "stop_reason": stop_reason,
        "execution_error": execution_error,
        "manifest_sha256": manifest_hash,
        "completed_schedule_items": len(completed),
        "trajectory_count": len(state["outcomes"]),
        "checkpoint_count": sum(
            len(row["checkpoints"]) for row in state["outcomes"]
        ),
        "decision": report["decision"],
        "gate_passed": report["gate_passed"],
        "openrouter": {
            "usage_before_usd": baseline,
            "usage_after_usd": usage_after,
            "exact_incremental_spend_usd": incremental,
            "project_spend_before_usd": manifest["budget"][
                "project_openrouter_spend_before_usd"
            ],
            "project_spend_after_usd": (
                float(manifest["budget"]["project_openrouter_spend_before_usd"])
                + incremental
            ),
            "final_query_error": final_query_error,
        },
        "cleanup": {
            "initial_sandbox_count": len(initial_sandboxes),
            "removed_new_sandbox_ids": cleanup["removed"],
            "cleanup_errors": cleanup["errors"],
            "remaining_new_sandbox_ids": remaining,
        },
    }
    _write_json(LEDGER_PATH, ledger)
    _write_json(OUTPUT_ROOT / "calibration-report-v1.json", report)
    return ledger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    print(json.dumps(run(args.manifest), indent=2))


if __name__ == "__main__":
    main()
