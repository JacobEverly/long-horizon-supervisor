from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from daytona import Daytona
from switchyard.cli.launchers.native_server import NativeServer

from horizon_supervisor.benchmark.gate7 import query_openrouter_key
from horizon_supervisor.training.run_stuck_pilot import (
    BRANCH_ACTIONS,
    KIMI_MODEL,
    KIMI_ROUTE,
    ROOT,
    SWITCHYARD,
    _attempt_usage_record,
    _branch_outcome,
    _checkpoint_records,
    _cleanup_new_sandboxes,
    _continue_outcome,
    _limits,
    _run_with_infrastructure_retry,
    _sha256,
    _trial_result,
    _validate_frozen_inputs,
)

OUTPUT_ROOT = ROOT / "artifacts/official/stuck-intervention-pilot-v0"
MANIFEST = OUTPUT_ROOT / "frozen-pilot-manifest-v3.json"


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(item.read_bytes()).digest())
    return digest.hexdigest()


def _record_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _stats_from_record(path: Path, model_id: str) -> dict[str, Any]:
    observations = [
        row["observation"]
        for row in _record_rows(path)
        if row.get("schema_version") == "stuck-observation-event.v0"
    ]
    if not observations:
        raise RuntimeError(f"source attempt lacks observable turn records: {path}")
    final = observations[-1]
    if final.get("model_id") != model_id:
        raise RuntimeError("source attempt model does not match its observation")
    return {
        "models": {
            model_id: {
                "prompt_tokens": int(final.get("input_tokens", 0)),
                "completion_tokens": int(final.get("output_tokens", 0)),
                "cached_tokens": int(final.get("cached_tokens", 0)),
                "reasoning_tokens": int(final.get("reasoning_tokens", 0)),
            }
        }
    }


def _reconstruct_trial(attempt: dict[str, Any]) -> dict[str, Any]:
    record_path = Path(attempt["record_path"])
    job_dir = Path(attempt["job_dir"])
    result = _trial_result(job_dir)
    if not attempt.get("valid") or result is None:
        raise RuntimeError("a sealed sibling attempt is no longer valid")
    return {
        **attempt,
        "result": result,
        "stats": _stats_from_record(record_path, str(attempt["model_id"])),
    }


def _find_attempt(
    attempts: list[dict[str, Any]], *, contains: str
) -> dict[str, Any]:
    matches = [row for row in attempts if contains in str(row.get("job_name"))]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one sealed attempt containing {contains!r}")
    return matches[0]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def run_recovery(manifest_path: Path = MANIFEST) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    expected_manifest_hash = manifest_path.with_suffix(".sha256").read_text().split()[0]
    if _sha256(manifest_path) != expected_manifest_hash:
        raise RuntimeError("frozen v3 recovery manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_frozen_inputs(manifest)
    if not os.getenv("OPENROUTER_API_KEY") or not os.getenv("DAYTONA_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY and DAYTONA_API_KEY are required")

    recovery = manifest["recovery"]
    output_path = ROOT / recovery["output_path"]
    if output_path.exists() or output_path.with_suffix(".sha256").exists():
        raise FileExistsError("the recovered matched-group output already exists")
    source_root = ROOT / recovery["source_execution_root"]
    if _tree_sha256(source_root) != recovery["source_execution_tree_sha256"]:
        raise RuntimeError("the sealed source execution tree changed")
    source_ledger = json.loads(
        (ROOT / recovery["source_ledger_path"]).read_text(encoding="utf-8")
    )
    attempts = list(source_ledger["attempts"])
    if len(attempts) != int(recovery["source_attempt_count"]):
        raise RuntimeError("the sealed source attempt count changed")

    tasks = [
        row
        for row in manifest["task_selection"]["ordered_pool"]
        if row["position"] == recovery["task_position"]
        and row["task_id"] == recovery["task_id"]
    ]
    if len(tasks) != 1:
        raise RuntimeError("the frozen recovery task is missing")
    task = tasks[0]
    base_attempt = _find_attempt(attempts, contains="base-suspected_stuck-01-qwen-")
    base_trial = _reconstruct_trial(base_attempt)
    checkpoints = [
        row
        for row in _checkpoint_records(Path(base_attempt["record_path"]))
        if row.get("checkpoint_kind") == "suspected_stuck"
        and int(row["observation"]["turn"]) == 10
    ]
    if len(checkpoints) != 1:
        raise RuntimeError("the sealed turn-10 checkpoint changed")
    checkpoint = checkpoints[0]
    limits = _limits(checkpoint)
    group_id = str(recovery["group_id"])
    group_rows = [
        _continue_outcome(
            group_id=group_id,
            task=task,
            base_trial=base_trial,
            checkpoint=checkpoint,
            limits=limits,
        )
    ]
    for action in recovery["sealed_valid_actions"]:
        if action == "continue_current_state":
            continue
        attempt = _find_attempt(attempts, contains=f"-{action}-")
        trial = _reconstruct_trial(attempt)
        group_rows.append(
            _branch_outcome(
                group_id=group_id,
                task=task,
                base_model_id=recovery["base_model_id"],
                checkpoint=checkpoint,
                action=action,
                trial=trial,
                limits=limits,
            )
        )
    if len(group_rows) != 5 or not all(row["valid"] for row in group_rows):
        raise RuntimeError("the five sealed sibling outcomes could not be reconstructed")

    initial_sandboxes = {sandbox.id for sandbox in Daytona().list()}
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_root = OUTPUT_ROOT / f"recovery-{timestamp}"
    run_root.mkdir(parents=True)
    server = NativeServer(SWITCHYARD)
    recovery_attempts: list[dict[str, Any]] = []
    cleanup: dict[str, list[str]] = {"removed": [], "errors": []}
    try:
        recovered_trial, recovery_attempts = _run_with_infrastructure_retry(
            server=server,
            manifest=manifest,
            run_root=run_root,
            task=task,
            route_id=KIMI_ROUTE,
            model_id=KIMI_MODEL,
            label=f"{group_id}-restart_kimi_clean-recovery",
            max_turns=limits["remaining_turns"],
            capture_healthy=False,
            capture_stuck=False,
            agent_timeout_seconds=limits["maximum_wall_seconds"],
        )
        recovered_row = _branch_outcome(
            group_id=group_id,
            task=task,
            base_model_id=recovery["base_model_id"],
            checkpoint=checkpoint,
            action="restart_kimi_clean",
            trial=recovered_trial,
            limits=limits,
        )
        group_rows.append(recovered_row)
    finally:
        server.close()
        cleanup = _cleanup_new_sandboxes(initial_sandboxes)

    actions = {row["branch_action"] for row in group_rows}
    complete = (
        len(group_rows) == 6
        and actions == BRANCH_ACTIONS
        and all(row["valid"] for row in group_rows)
    )
    usage_before = float(manifest["budget"]["usage_before_usd"])
    key_after = query_openrouter_key(os.environ["OPENROUTER_API_KEY"])
    usage_after = float(key_after["usage"])
    prior_spend = float(manifest["budget"]["prior_key_spend_usd"])
    usage_path = run_root / "trial-usage-ledger-v0.jsonl"
    _write_jsonl(
        usage_path,
        [_attempt_usage_record(attempt) for attempt in recovery_attempts],
    )
    report = {
        "schema_version": "stuck-pilot-group-recovery.v0",
        "status": "complete" if complete else "stopped",
        "manifest_path": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": expected_manifest_hash,
        "group_id": group_id,
        "recovered_action": "restart_kimi_clean",
        "valid_outcome_count": len(group_rows) if complete else 0,
        "current_key_usage_before_usd": usage_before,
        "current_key_usage_after_usd": usage_after,
        "current_key_spend_usd": max(0.0, usage_after - usage_before),
        "prior_key_spend_usd": prior_spend,
        "exact_incremental_openrouter_spend_usd": (
            prior_spend + max(0.0, usage_after - usage_before)
        ),
        "cleanup": {
            "initial_sandbox_ids": sorted(initial_sandboxes),
            "removed_new_sandbox_ids": cleanup["removed"],
            "cleanup_errors": cleanup["errors"],
            "remaining_new_sandbox_ids": sorted(
                {sandbox.id for sandbox in Daytona().list()} - initial_sandboxes
            ),
        },
        "trial_usage_ledger_path": str(usage_path.relative_to(ROOT)),
    }
    if complete:
        _write_jsonl(output_path, group_rows)
        output_path.with_suffix(".sha256").write_text(
            f"{_sha256(output_path)}  {output_path.name}\n", encoding="utf-8"
        )
        report["outcomes_path"] = str(output_path.relative_to(ROOT))
        report["outcomes_sha256"] = _sha256(output_path)
    (run_root / "recovery-ledger.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    if not complete:
        raise RuntimeError("the sole recovered Kimi-clean arm remained invalid")
    return {"run_root": str(run_root), **report}


def main() -> None:
    print(json.dumps(run_recovery(), indent=2))


if __name__ == "__main__":
    main()
