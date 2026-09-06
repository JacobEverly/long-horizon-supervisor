from __future__ import annotations

import argparse
import json
import os
from collections import Counter
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
from horizon_supervisor.training.freeze_continuation_calibration_v4 import (
    EXPECTED_V3_MANIFEST_SHA256,
    V3_MANIFEST,
)
from horizon_supervisor.training.freeze_continuation_calibration_v5 import (
    DIFFICULTY_QUOTAS,
    EXPECTED_V4_FIDELITY_SHA256,
    EXPECTED_V4_LEDGER_SHA256,
    EXPECTED_V4_MANIFEST_SHA256,
    EXPECTED_V4_PUBLIC_SHA256,
    EXPECTED_V4_REPORT_SHA256,
    TRANCHE_DIFFICULTY_COUNTS,
    V4_FIDELITY,
    V4_LEDGER,
    V4_MANIFEST,
    V4_PUBLIC,
    V4_REPORT,
)
from horizon_supervisor.training.freeze_continuation_calibration_v6 import (
    EXPECTED_V5_FAILURE_SHA256,
    EXPECTED_V5_MANIFEST_SHA256,
    EXPECTED_V5_SMOKE_SHA256,
    OUTPUT_ROOT,
    V5_FAILURE,
    V5_MANIFEST,
    V5_SMOKE,
)
from horizon_supervisor.training.run_continuation_calibration import (
    _get_json,
    _post_json,
    _trial_result,
    _write_json,
    _write_jsonl,
    validate_key_budget,
)
from horizon_supervisor.training.run_continuation_calibration_v1 import (
    _structural_runtime_failure,
    _v1_outcome_row,
    _wait_for_cleanup,
)
from horizon_supervisor.training.run_continuation_calibration_v2 import (
    _validate_checkpoint,
)
from horizon_supervisor.training.run_continuation_calibration_v4 import (
    _continuation_command as _v4_continuation_command,
)
from horizon_supervisor.training.run_stuck_pilot import (
    _attempt_usage_record,
    _cleanup_new_sandboxes,
    _retryable_infrastructure_failure,
    _run_command,
    _valid_trial,
)

MANIFEST = OUTPUT_ROOT / "frozen-manifest-v6.json"
STATE_PATH = OUTPUT_ROOT / "execution-state-v6.json"
OUTCOMES_PATH = OUTPUT_ROOT / "natural-continuation-outcomes-v6.jsonl"
USAGE_PATH = OUTPUT_ROOT / "trial-usage-ledger-v6.jsonl"
CHECKPOINT_INDEX = OUTPUT_ROOT / "checkpoint-bank-index-v6.jsonl"
FIDELITY_REPORT = OUTPUT_ROOT / "snapshot-fidelity-v6.json"
LEDGER_PATH = OUTPUT_ROOT / "execution-ledger-v6.json"
REPORT_PATH = OUTPUT_ROOT / "calibration-report-v6.json"
SMOKE_REPORT = OUTPUT_ROOT / "permission-transport-smoke-v6.json"
EXPECTED_SCHEMA = "two-tier-continuation-calibration-manifest.v6"
EXPECTED_ENVIRONMENT = (
    "horizon_supervisor.benchmark.permission_preserving_daytona:"
    "PermissionPreservingDaytonaEnvironment"
)


def validate_manifest(
    manifest_path: Path = MANIFEST,
) -> tuple[dict[str, Any], str]:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sidecar = manifest_path.with_suffix(".sha256")
    if not sidecar.is_file():
        raise RuntimeError("frozen v6 manifest sidecar is missing")
    expected_hash = sidecar.read_text(encoding="utf-8").split()[0]
    if _sha256(manifest_path) != expected_hash:
        raise RuntimeError("frozen continuation v6 manifest hash mismatch")
    if manifest.get("schema_version") != EXPECTED_SCHEMA:
        raise RuntimeError("unexpected continuation v6 manifest schema")
    if manifest.get("frozen_before_model_outcomes") is not True:
        raise RuntimeError("v6 manifest was not frozen before outcomes")

    lineage = manifest["lineage"]
    expected_lineage = {
        "v3_manifest_sha256": EXPECTED_V3_MANIFEST_SHA256,
        "v4_manifest_sha256": EXPECTED_V4_MANIFEST_SHA256,
        "v4_report_sha256": EXPECTED_V4_REPORT_SHA256,
        "v4_ledger_sha256": EXPECTED_V4_LEDGER_SHA256,
        "v4_fidelity_sha256": EXPECTED_V4_FIDELITY_SHA256,
        "v4_public_summary_sha256": EXPECTED_V4_PUBLIC_SHA256,
        "v5_manifest_sha256": EXPECTED_V5_MANIFEST_SHA256,
        "v5_failed_smoke_sha256": EXPECTED_V5_SMOKE_SHA256,
        "v5_failure_summary_sha256": EXPECTED_V5_FAILURE_SHA256,
    }
    if any(lineage.get(key) != value for key, value in expected_lineage.items()):
        raise RuntimeError("v6 lineage hashes changed")
    if (
        lineage["detector_thresholds_changed"] is not False
        or lineage["analysis_gate_thresholds_changed"] is not False
        or lineage["models_changed"] is not False
        or lineage["max_turns_changed"] is not False
        or lineage["token_limits_changed"] is not False
        or lineage["prior_outcomes_reused_for_scoring"] is not False
        or lineage["task_selection_changed"] is not False
        or lineage["natural_continuation_protocol_changed"] is not False
        or lineage["v5_provider_model_calls"] != 0
        or lineage["v5_model_outcomes_observed"] != 0
        or lineage["v5_failed_only_cleanup_finalization"] is not True
    ):
        raise RuntimeError("v6 minimal revision contract changed")
    immutable_inputs = {
        V3_MANIFEST: EXPECTED_V3_MANIFEST_SHA256,
        V4_MANIFEST: EXPECTED_V4_MANIFEST_SHA256,
        V4_REPORT: EXPECTED_V4_REPORT_SHA256,
        V4_LEDGER: EXPECTED_V4_LEDGER_SHA256,
        V4_FIDELITY: EXPECTED_V4_FIDELITY_SHA256,
        V4_PUBLIC: EXPECTED_V4_PUBLIC_SHA256,
        V5_MANIFEST: EXPECTED_V5_MANIFEST_SHA256,
        V5_SMOKE: EXPECTED_V5_SMOKE_SHA256,
        V5_FAILURE: EXPECTED_V5_FAILURE_SHA256,
    }
    for path, digest in immutable_inputs.items():
        if not path.is_file() or _sha256(path) != digest:
            raise RuntimeError(f"immutable v4 input changed: {path}")

    if manifest["models"]["routes"] != EXACT_MODELS:
        raise RuntimeError("frozen v6 route/model mapping changed")
    if manifest["detector"]["config"] != SELECTED_CONFIG.model_dump():
        raise RuntimeError("frozen v6 detector config changed")
    execution = manifest["execution"]
    if (
        execution["agent"] != "HarnessFilteredContinuationTerminus2"
        or execution["checkpoint_environment"] != EXPECTED_ENVIRONMENT
        or execution["max_turns"] != 12
        or execution["per_response_output_tokens"] != 8_192
        or execution["total_output_token_budget"] != 49_152
        or execution["output_length_corrective_retries"] != 1
        or execution["natural_continuation_only"] is not True
        or execution["interventions_forbidden"] is not True
        or execution["checkpoint_replay_provider_calls"] != 0
        or execution["transport_smoke_required"] is not True
        or execution["transport_smoke_schema"] != "permission-transport-smoke.v6"
        or execution["transport_smoke_cleanup_wait_seconds"] != 24
    ):
        raise RuntimeError("frozen v6 execution contract changed")
    expected_smoke_path = (
        "artifacts/official/two-tier-continuation-calibration-v6/"
        "permission-transport-smoke-v6.json"
    )
    if execution["transport_smoke_path"] != expected_smoke_path:
        raise RuntimeError("v6 transport smoke path changed")

    budget = manifest["budget"]
    if (
        float(budget["project_openrouter_spend_before_usd"]) != 53.269355918
        or float(budget["project_cumulative_ceiling_usd"]) != 200.0
        or float(budget["phase_a_incremental_ceiling_usd"]) != 3.5
        or float(budget["tranche_1_incremental_ceiling_usd"]) != 1.4
        or float(budget["tranche_2_incremental_ceiling_usd"]) != 2.5
        or float(budget["per_trial_incremental_ceiling_usd"]) != 0.5
    ):
        raise RuntimeError("frozen v6 budget changed")

    tasks = manifest["task_selection"]["ordered_pool"]
    if len(tasks) != 24 or [row["position"] for row in tasks] != list(range(1, 25)):
        raise RuntimeError("frozen v6 task pool changed")
    if [row["tranche"] for row in tasks] != [1] * 8 + [2] * 8 + [3] * 8:
        raise RuntimeError("frozen v6 tranches changed")
    if Counter(row["difficulty"] for row in tasks) != Counter(DIFFICULTY_QUOTAS):
        raise RuntimeError("frozen v6 difficulty quotas changed")
    for tranche, expected in TRANCHE_DIFFICULTY_COUNTS.items():
        actual = Counter(
            row["difficulty"] for row in tasks if row["tranche"] == tranche
        )
        if actual != Counter(expected):
            raise RuntimeError(f"frozen v6 tranche {tranche} mix changed")
    if any(
        row["static_checkpoint_compatible"] is not True
        or row["prior_terminal_outcome_count"] != 0
        for row in tasks
    ):
        raise RuntimeError("v6 task eligibility changed")

    analysis = manifest["analysis"]
    if (
        analysis["cohort"] != "fresh_v6_only"
        or analysis["aggregate_v3_and_v4"] is not False
        or analysis["prior_outcomes_used_for_fit_or_tuning"] is not False
    ):
        raise RuntimeError("v6 independent analysis contract changed")
    fixed_files = {
        SWITCHYARD: manifest["integrity"]["switchyard_sha256"],
        ROOT / manifest["models"]["catalog_path"]: manifest["models"][
            "catalog_sha256"
        ],
        ROOT / manifest["task_selection"]["eligibility_path"]: manifest[
            "task_selection"
        ]["eligibility_sha256"],
        ROOT / manifest["task_selection"]["task_source_lock_path"]: manifest[
            "task_selection"
        ]["task_source_lock_sha256"],
    }
    fixed_files.update(
        {
            ROOT / relative: digest
            for relative, digest in manifest["integrity"]["code_sha256"].items()
        }
    )
    for path, digest in fixed_files.items():
        if not path.is_file() or _sha256(path) != digest:
            raise RuntimeError(f"frozen v6 input hash mismatch: {path}")
    for task in tasks:
        task_root = ROOT / task["task_root"]
        if not task_root.is_dir() or _tree_sha256(task_root) != task["task_tree_sha256"]:
            raise RuntimeError(f"frozen v6 task tree changed: {task['task_id']}")
    return manifest, expected_hash


def validate_transport_smoke(manifest: dict[str, Any]) -> tuple[dict[str, Any], str]:
    path = ROOT / manifest["execution"]["transport_smoke_path"]
    if not path.is_file():
        raise RuntimeError("v6 permission-preserving transport smoke is missing")
    smoke = json.loads(path.read_text(encoding="utf-8"))
    if (
        smoke.get("schema_version") != "permission-transport-smoke.v6"
        or smoke.get("passed") is not True
        or smoke.get("provider_model_calls") != 0
        or smoke.get("remote_to_local_digest_match") is not True
        or smoke.get("local_to_remote_digest_match") is not True
        or smoke.get("read_only_git_object_mode_preserved") is not True
        or smoke.get("remaining_daytona_environments") != 0
    ):
        raise RuntimeError("v6 permission-preserving transport smoke did not pass")
    return smoke, _sha256(path)


def _runtime_ceiling(manifest: dict[str, Any], baseline: float, tranche: int) -> float:
    budget = manifest["budget"]
    increments = {
        1: float(budget["tranche_1_incremental_ceiling_usd"]),
        2: float(budget["tranche_2_incremental_ceiling_usd"]),
        3: float(budget["phase_a_incremental_ceiling_usd"]),
    }
    return baseline + increments[tranche]


def _replace_environment(command: list[str], environment: str) -> None:
    matches = [index for index, value in enumerate(command) if value == "--env"]
    if len(matches) != 1 or matches[0] + 1 >= len(command):
        raise RuntimeError("expected one Harbor --env argument")
    command[matches[0] + 1] = environment


def _continuation_command(**kwargs: Any) -> list[str]:
    command = _v4_continuation_command(**kwargs)
    _replace_environment(command, EXPECTED_ENVIRONMENT)
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
        raise RuntimeError("frozen OpenRouter v6 tranche ceiling lacks one reserve")
    _post_json(f"{server.base_url}/v1/stats/reset")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    label = route_id.rsplit("-", 1)[-1]
    job_name = f"continuation-v6-{task['position']:02d}-{label}-{timestamp}"
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
    environment["HORIZON_HARBOR_OUTPUT_LENGTH_RETRIES"] = str(
        manifest["execution"]["output_length_corrective_retries"]
    )
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


def _new_state(
    manifest_hash: str, smoke_hash: str, usage: float, hard_limit: float
) -> dict[str, Any]:
    return {
        "schema_version": "continuation-calibration-execution-state.v6",
        "manifest_sha256": manifest_hash,
        "transport_smoke_sha256": smoke_hash,
        "status": "in_progress",
        "started_at": datetime.now(UTC).isoformat(),
        "provider_usage_baseline_usd": usage,
        "provider_hard_limit_usd": hard_limit,
        "completed_schedule_items": [],
        "outcomes": [],
        "attempts": [],
        "fidelity_rows": [],
        "tranche_reports": [],
        "pending_trial": None,
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
            "schema_version": "continuation-snapshot-fidelity-report.v6",
            "attempt_count": len(state["fidelity_rows"]),
            "pass_count": sum(row["passed"] for row in state["fidelity_rows"]),
            "all_passed": bool(state["fidelity_rows"])
            and all(row["passed"] for row in state["fidelity_rows"]),
            "rows": state["fidelity_rows"],
        },
    )


def _finalize_pending(state: dict[str, Any], manifest: dict[str, Any]) -> None:
    pending = state.get("pending_trial")
    if not pending:
        return
    task = next(
        task
        for task in manifest["task_selection"]["ordered_pool"]
        if int(task["position"]) == int(pending["task_position"])
    )
    trial = pending["trial"]
    records = record_rows(Path(trial["record_path"]))
    checkpoints = [
        row
        for row in records
        if row.get("schema_version") == "matched-checkpoint.v0"
        and row.get("state_transfer_eligible") is True
    ]
    fidelity_rows = pending.setdefault("fidelity_rows", [])
    completed_snapshots = {row["snapshot_id"] for row in fidelity_rows}
    if trial["valid"] and not _structural_runtime_failure(trial, records):
        for checkpoint in checkpoints:
            snapshot_id = (
                f"{checkpoint['run_id']}-{checkpoint['checkpoint_kind']}-"
                f"t{int(checkpoint['observation']['turn']):02d}"
            )
            if snapshot_id in completed_snapshots:
                continue
            fidelity = _validate_checkpoint(
                task=task, checkpoint=checkpoint, run_root=OUTPUT_ROOT
            )
            fidelity_rows.append(fidelity)
            state["fidelity_rows"].append(fidelity)
            completed_snapshots.add(snapshot_id)
            _persist(state)
    state["outcomes"].append(
        _v1_outcome_row(task=task, trial=trial, fidelity_rows=fidelity_rows)
    )
    schedule_item = pending["schedule_item"]
    if schedule_item not in state["completed_schedule_items"]:
        state["completed_schedule_items"].append(schedule_item)
    state["pending_trial"] = None
    _persist(state)


def _report(
    state: dict[str, Any], *, tranche: int | None = None
) -> dict[str, Any]:
    report = analyze(state["outcomes"])
    report["schema_version"] = "two-tier-continuation-calibration-report.v6"
    report["cohort"] = "fresh_v6_only"
    report["new_v6_trajectory_count"] = len(state["outcomes"])
    report["prior_outcome_count_used"] = 0
    if tranche is not None:
        report["tranche"] = tranche
    return report


def run(manifest_path: Path = MANIFEST) -> dict[str, Any]:
    manifest, manifest_hash = validate_manifest(manifest_path)
    _, smoke_hash = validate_transport_smoke(manifest)
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or not os.getenv("DAYTONA_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY and DAYTONA_API_KEY are required")
    initial_sandboxes = {sandbox.id for sandbox in Daytona().list()}
    if initial_sandboxes:
        raise RuntimeError("continuation calibration v6 requires zero sandboxes")
    key_info = query_openrouter_key(api_key)
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if state["manifest_sha256"] != manifest_hash:
            raise RuntimeError("v6 execution state belongs to another manifest")
        if state["transport_smoke_sha256"] != smoke_hash:
            raise RuntimeError("v6 transport smoke changed after execution began")
    else:
        state = _new_state(
            manifest_hash,
            smoke_hash,
            float(key_info["usage"]),
            float(key_info["limit"]),
        )
        _persist(state)
    baseline = float(state["provider_usage_baseline_usd"])
    validate_key_budget(manifest, key_info, baseline=baseline)
    state["status"] = "in_progress"
    state["execution_error"] = None
    _persist(state)

    server: NativeServer | None = None
    status = "in_progress"
    stop_reason = None
    execution_error = None
    try:
        server = NativeServer(SWITCHYARD)
        _finalize_pending(state, manifest)
        completed = set(state["completed_schedule_items"])
        for tranche in (1, 2, 3):
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
                    state["pending_trial"] = {
                        "schema_version": "continuation-pending-trial.v6",
                        "schedule_item": schedule_item,
                        "task_position": task["position"],
                        "route_id": route_id,
                        "model_id": model_id,
                        "trial": trial,
                        "fidelity_rows": [],
                    }
                    _persist(state)
                    _finalize_pending(state, manifest)
                    completed = set(state["completed_schedule_items"])
            tranche_report = _report(state, tranche=tranche)
            state["tranche_reports"].append(tranche_report)
            _persist(state)
            if tranche_report["gate_passed"]:
                status = "complete"
                stop_reason = "continuation_calibration_gate_passed"
                break
        else:
            status = "complete"
            stop_reason = "frozen_task_pool_exhausted"
    except KeyboardInterrupt:
        status = "interrupted"
        stop_reason = "operator_interrupt"
    except Exception as error:  # preserve an auditable, resumable pending trial
        status = "stopped"
        stop_reason = (
            "frozen_openrouter_spend_ceiling_reached"
            if "ceiling" in str(error) or "budget" in str(error)
            else "execution_runtime_error"
        )
        execution_error = f"{type(error).__name__}: {error}"
    finally:
        if server is not None:
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
    report = _report(state)
    ledger = {
        "schema_version": "continuation-calibration-execution-ledger.v6",
        "status": status,
        "stop_reason": stop_reason,
        "execution_error": execution_error,
        "manifest_sha256": manifest_hash,
        "transport_smoke_sha256": smoke_hash,
        "completed_schedule_items": len(state["completed_schedule_items"]),
        "new_v6_trajectory_count": len(state["outcomes"]),
        "analysis_cohort": "fresh_v6_only",
        "prior_outcome_count_used": 0,
        "new_v6_checkpoint_count": sum(
            len(row["checkpoints"]) for row in state["outcomes"]
        ),
        "pending_trial": state["pending_trial"] is not None,
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
    _write_json(REPORT_PATH, report)
    return ledger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    print(json.dumps(run(args.manifest), indent=2))


if __name__ == "__main__":
    main()
