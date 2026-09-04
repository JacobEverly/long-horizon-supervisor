from __future__ import annotations

import argparse
import json
import os
import urllib.error
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from daytona import Daytona
from switchyard.cli.launchers.native_server import NativeServer

from horizon_supervisor.benchmark.gate7 import query_openrouter_key
from horizon_supervisor.training.run_stuck_pilot import (
    BASE_ROUTE_TO_MODEL,
    KIMI_MODEL,
    KIMI_ROUTE,
    SWITCHYARD,
    _attempt_usage_record,
    _branch_outcome,
    _checkpoint_records,
    _cleanup_new_sandboxes,
    _limits,
    _run_with_infrastructure_retry,
    _sha256,
    _tree_sha256,
)

ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = ROOT / "artifacts/official/stuck-confirmatory-v1"
MANIFEST = OUTPUT_ROOT / "frozen-manifest-v2.json"
STATE_PATH = OUTPUT_ROOT / "execution-state-v0.json"
OUTCOMES_PATH = OUTPUT_ROOT / "matched-outcomes-v0.jsonl"
LEDGER_PATH = OUTPUT_ROOT / "execution-ledger-v0.json"
USAGE_PATH = OUTPUT_ROOT / "trial-usage-ledger-v0.jsonl"
BRANCH_ACTIONS = {
    "continue_current_state",
    "switch_value_state",
    "switch_kimi_state",
    "restart_kimi_clean",
}
KINDS = ("suspected_stuck", "healthy")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _validate_frozen_inputs(manifest_path: Path, manifest: dict[str, Any]) -> str:
    sidecar = manifest_path.with_suffix(".sha256")
    if not sidecar.is_file():
        raise RuntimeError("frozen manifest hash sidecar is missing")
    expected_manifest_hash = sidecar.read_text(encoding="utf-8").split()[0]
    if _sha256(manifest_path) != expected_manifest_hash:
        raise RuntimeError("frozen confirmatory manifest hash mismatch")
    if manifest.get("schema_version") != "stuck-confirmatory-manifest.v2":
        raise RuntimeError("unexpected confirmatory manifest schema")
    amendment = manifest.get("amendment") or {}
    if amendment.get("id") != "explicit-four-way-snapshot-branching-v1":
        raise RuntimeError("required pre-outcome orchestration amendment is missing")
    if amendment.get("scope") != "orchestration_only":
        raise RuntimeError("unexpected confirmatory amendment scope")
    if amendment.get("accepted_outcomes_before_amendment") != 0:
        raise RuntimeError("orchestration amendment was not frozen before outcomes")
    efficiency_amendment = manifest.get("execution_efficiency_amendment") or {}
    if efficiency_amendment.get("id") != "stop-scout-after-sealed-checkpoint-v2":
        raise RuntimeError("required scout-efficiency amendment is missing")
    if efficiency_amendment.get("scope") != "orchestration_only":
        raise RuntimeError("unexpected scout-efficiency amendment scope")
    if efficiency_amendment.get("accepted_outcomes_before_amendment") != 4:
        raise RuntimeError("scout-efficiency amendment outcome boundary changed")
    if manifest["models"]["routes"] != {
        **BASE_ROUTE_TO_MODEL,
        KIMI_ROUTE: KIMI_MODEL,
    }:
        raise RuntimeError("frozen route/model roster changed")
    if set(manifest["branch_contract"]["branches"]) != BRANCH_ACTIONS:
        raise RuntimeError("frozen four-arm contract changed")
    if manifest["sampling_and_stopping"] != {
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
    }:
        raise RuntimeError("frozen sampling and stopping contract changed")

    integrity = manifest["integrity"]
    fixed_files = {
        SWITCHYARD: integrity["switchyard_sha256"],
        ROOT / manifest["models"]["catalog_path"]: manifest["models"][
            "catalog_sha256"
        ],
        ROOT / manifest["execution"]["snapshot_fidelity_path"]: manifest[
            "execution"
        ]["snapshot_fidelity_sha256"],
    }
    fixed_files.update(
        {ROOT / relative: expected for relative, expected in integrity["code_sha256"].items()}
    )
    fixed_files[
        ROOT / "src/horizon_supervisor/training/amend_stuck_confirmatory.py"
    ] = integrity["amendment_script_sha256"]
    fixed_files[
        ROOT / "src/horizon_supervisor/training/amend_stuck_confirmatory_efficiency.py"
    ] = integrity["efficiency_amendment_script_sha256"]
    for path, expected in fixed_files.items():
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"frozen input hash mismatch: {path}")
    tasks = manifest["task_selection"]["ordered_pool"]
    if len(tasks) != int(manifest["task_selection"]["pool_size"]):
        raise RuntimeError("frozen pool size does not match the ordered task list")
    if len({task["task_id"] for task in tasks}) != len(tasks):
        raise RuntimeError("frozen task pool contains duplicates")
    for task in tasks:
        task_root = ROOT / task["task_root"]
        if not task_root.is_dir() or _tree_sha256(task_root) != task["task_tree_sha256"]:
            raise RuntimeError(f"frozen task tree changed: {task['task_id']}")
    if float(manifest["budget"]["additional_openrouter_cap_usd"]) != 20.0:
        raise RuntimeError("frozen additional OpenRouter cap changed")
    return expected_manifest_hash


def _branch_specs(base_route: str) -> list[tuple[str, str, str, bool]]:
    if base_route not in BASE_ROUTE_TO_MODEL:
        raise ValueError(f"unexpected base route: {base_route}")
    other = (
        "gate7/fixed-qwen" if base_route == "gate7/fixed-flash" else "gate7/fixed-flash"
    )
    return [
        (
            "continue_current_state",
            base_route,
            BASE_ROUTE_TO_MODEL[base_route],
            True,
        ),
        ("switch_value_state", other, BASE_ROUTE_TO_MODEL[other], True),
        ("switch_kimi_state", KIMI_ROUTE, KIMI_MODEL, True),
        ("restart_kimi_clean", KIMI_ROUTE, KIMI_MODEL, False),
    ]


def _group_rows(outcomes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    groups = []
    for row in outcomes:
        group_id = str(row["group_id"])
        if group_id not in seen:
            groups.append(row)
            seen.add(group_id)
    return groups


def _selection_summary(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    groups = _group_rows(outcomes)
    kind_counts = Counter(str(row["checkpoint_kind"]) for row in groups)
    base_counts = {
        kind: {
            model: sum(
                row["checkpoint_kind"] == kind and row["base_model_id"] == model
                for row in groups
            )
            for model in BASE_ROUTE_TO_MODEL.values()
        }
        for kind in KINDS
    }
    task_kind_counts = Counter(
        (str(row["task_id"]), str(row["checkpoint_kind"])) for row in groups
    )
    task_counts = Counter(str(row["task_id"]) for row in groups)
    tasks_by_kind = {
        kind: sorted(
            {str(row["task_id"]) for row in groups if row["checkpoint_kind"] == kind}
        )
        for kind in KINDS
    }
    unique_tasks = sorted(task_counts)
    return {
        "group_counts": {kind: kind_counts[kind] for kind in KINDS},
        "base_counts_by_kind": base_counts,
        "task_kind_counts": {
            f"{task}|{kind}": count
            for (task, kind), count in sorted(task_kind_counts.items())
        },
        "task_counts": dict(sorted(task_counts.items())),
        "unique_tasks": unique_tasks,
        "unique_tasks_by_kind": tasks_by_kind,
        "valid_outcome_count": len(outcomes),
    }


def _target_met(outcomes: list[dict[str, Any]]) -> bool:
    summary = _selection_summary(outcomes)
    return bool(
        summary["valid_outcome_count"] == 96
        and all(summary["group_counts"][kind] == 12 for kind in KINDS)
        and len(summary["unique_tasks"]) >= 8
        and all(len(summary["unique_tasks_by_kind"][kind]) >= 4 for kind in KINDS)
        and all(
            summary["base_counts_by_kind"][kind][base] >= 4
            for kind in KINDS
            for base in BASE_ROUTE_TO_MODEL.values()
        )
        and max(summary["task_kind_counts"].values(), default=0) <= 2
    )


def _acceptance_block_reason(
    outcomes: list[dict[str, Any]], *, task_id: str, kind: str, base_model: str
) -> str | None:
    summary = _selection_summary(outcomes)
    if summary["group_counts"][kind] >= 12:
        return "checkpoint-kind target already filled"
    if summary["base_counts_by_kind"][kind][base_model] >= 8:
        return "predeclared maximum of eight groups for this base and kind reached"
    if summary["task_kind_counts"].get(f"{task_id}|{kind}", 0) >= 2:
        return "predeclared maximum of two groups per task and kind reached"
    if summary["task_counts"].get(task_id, 0) >= 3:
        return "predeclared maximum of three groups per task reached"
    return None


def _eligible_checkpoint(
    base_trial: dict[str, Any], kind: str
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate the scouting checkpoint independently of the later scout result."""
    checkpoints = [
        checkpoint
        for checkpoint in _checkpoint_records(Path(base_trial["record_path"]))
        if checkpoint["checkpoint_kind"] == kind
    ]
    if not checkpoints:
        return None, "requested checkpoint did not occur"
    if len(checkpoints) != 1:
        return None, "duplicate checkpoints were captured"
    checkpoint = checkpoints[0]
    if not checkpoint["state_transfer_eligible"]:
        return None, str(checkpoint["state_transfer_ineligibility_reason"])
    return checkpoint, None


def _new_state(manifest_hash: str, usage_before: float) -> dict[str, Any]:
    return {
        "schema_version": "stuck-confirmatory-execution-state.v0",
        "manifest_sha256": manifest_hash,
        "status": "in_progress",
        "started_at": datetime.now(UTC).isoformat(),
        "completed_schedule_items": [],
        "outcomes": [],
        "ineligible": [],
        "attempts": [],
        "pending_group": None,
        "spend_checkpoints": [
            {
                "kind": "baseline",
                "accepted_groups": 0,
                "usage_usd": usage_before,
                "recorded_at": datetime.now(UTC).isoformat(),
            }
        ],
        "reported_group_milestones": [],
    }


def _load_or_create_state(manifest_hash: str, usage_before: float) -> dict[str, Any]:
    if not STATE_PATH.exists():
        state = _new_state(manifest_hash, usage_before)
        _persist_state(state)
        return state
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    if state.get("schema_version") != "stuck-confirmatory-execution-state.v0":
        raise RuntimeError("unexpected resumable state schema")
    if state.get("manifest_sha256") != manifest_hash:
        raise RuntimeError("resumable state belongs to a different frozen manifest")
    return state


def _persist_state(state: dict[str, Any]) -> None:
    state["selection_summary"] = _selection_summary(state["outcomes"])
    state["updated_at"] = datetime.now(UTC).isoformat()
    _write_json(STATE_PATH, state)
    _write_jsonl(OUTCOMES_PATH, state["outcomes"])
    _write_jsonl(USAGE_PATH, state["attempts"])


def _record_attempts(state: dict[str, Any], attempts: list[dict[str, Any]]) -> None:
    state["attempts"].extend(_attempt_usage_record(attempt) for attempt in attempts)


def _record_progress_if_due(state: dict[str, Any], api_key: str) -> None:
    accepted_groups = len(_group_rows(state["outcomes"]))
    if accepted_groups == 0 or accepted_groups % 4:
        return
    if accepted_groups in state["reported_group_milestones"]:
        return
    key_info = query_openrouter_key(api_key)
    state["spend_checkpoints"].append(
        {
            "kind": "four-group-tranche",
            "accepted_groups": accepted_groups,
            "usage_usd": float(key_info["usage"]),
            "recorded_at": datetime.now(UTC).isoformat(),
        }
    )
    state["reported_group_milestones"].append(accepted_groups)
    summary = _selection_summary(state["outcomes"])
    progress = {
        "accepted_groups": accepted_groups,
        "group_counts": summary["group_counts"],
        "unique_task_count": len(summary["unique_tasks"]),
        "base_counts_by_kind": summary["base_counts_by_kind"],
        "valid_outcomes": len(state["outcomes"]),
        "ineligible_count": len(state["ineligible"]),
        "incremental_spend_usd": (
            float(key_info["usage"]) - state["spend_checkpoints"][0]["usage_usd"]
        ),
    }
    print(json.dumps({"progress": progress}), flush=True)


def _finish_pending_group(
    *,
    state: dict[str, Any],
    manifest: dict[str, Any],
    server: NativeServer,
    api_key: str,
) -> None:
    pending = state.get("pending_group")
    if not pending:
        return
    existing_actions = {row["branch_action"] for row in pending["rows"]}
    checkpoint = pending["checkpoint"]
    task = pending["task"]
    for action, route_id, model_id, state_transfer in _branch_specs(
        pending["base_route"]
    ):
        if action in existing_actions:
            continue
        seed = Path(checkpoint["anchor_workspace_path"]) if state_transfer else None
        trial, attempts = _run_with_infrastructure_retry(
            server=server,
            manifest=manifest,
            run_root=OUTPUT_ROOT,
            task=task,
            route_id=route_id,
            model_id=model_id,
            label=f"{pending['group_id']}-{action}",
            max_turns=pending["limits"]["remaining_turns"],
            capture_healthy=False,
            capture_stuck=False,
            workspace_seed=seed,
            expected_workspace_digest=(
                checkpoint["observation"]["workspace_digest"]
                if state_transfer
                else None
            ),
            handoff_path=Path(pending["handoff_path"]) if state_transfer else None,
            agent_timeout_seconds=pending["limits"]["maximum_wall_seconds"],
        )
        _record_attempts(state, attempts)
        row = _branch_outcome(
            group_id=pending["group_id"],
            task=task,
            base_model_id=pending["base_model"],
            checkpoint=checkpoint,
            action=action,
            trial=trial,
            limits=pending["limits"],
        )
        if action == "continue_current_state":
            # The confirmatory amendment launches continuation from the same
            # rehydrated snapshot as the two preserved-state switch arms.
            row["preserved_state"] = True
            exception_text = json.dumps(
                (trial.get("result") or {}).get("exception_info") or {}
            ).lower()
            row["state_transfer_failure"] = any(
                marker in exception_text
                for marker in ("rehydrated workspace", "workspace_seed", "environment")
            )
        pending["rows"].append(row)
        _persist_state(state)
        if not row["valid"]:
            state["ineligible"].append(
                {
                    "group_id": pending["group_id"],
                    "schedule_item": pending["schedule_item"],
                    "reason": "one or more branches remained invalid after retry",
                    "invalid_action": action,
                    "unexecuted_actions": sorted(
                        BRANCH_ACTIONS
                        - {branch["branch_action"] for branch in pending["rows"]}
                    ),
                }
            )
            state["completed_schedule_items"].append(pending["schedule_item"])
            state["pending_group"] = None
            _persist_state(state)
            return

    rows = pending["rows"]
    if len(rows) != 4 or {row["branch_action"] for row in rows} != BRANCH_ACTIONS:
        raise RuntimeError("pending group did not resolve to the frozen four arms")
    if any(row.get("valid") is not True for row in rows):
        raise RuntimeError("invalid row reached confirmatory acceptance")
    if {row["group_id"] for row in state["outcomes"]} & {pending["group_id"]}:
        raise RuntimeError("refusing to duplicate a sealed matched group")
    state["outcomes"].extend(rows)
    state["completed_schedule_items"].append(pending["schedule_item"])
    state["pending_group"] = None
    _record_progress_if_due(state, api_key)
    _persist_state(state)


def _final_ledger(
    *,
    state: dict[str, Any],
    manifest: dict[str, Any],
    manifest_path: Path,
    manifest_hash: str,
    status: str,
    stop_reason: str,
    execution_error: str | None,
    usage_after: float,
    final_key_query_error: str | None,
    cleanup: dict[str, list[str]],
    initial_sandbox_ids: set[str],
    remaining_sandbox_ids: list[str],
) -> dict[str, Any]:
    usage_before = float(manifest["budget"]["usage_before_usd"])
    exact_spend = max(0.0, usage_after - usage_before)
    return {
        "schema_version": "stuck-confirmatory-execution-ledger.v0",
        "status": status,
        "stop_reason": stop_reason,
        "execution_error": execution_error,
        "manifest_path": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": manifest_hash,
        "selection_summary": _selection_summary(state["outcomes"]),
        "valid_outcome_count": len(state["outcomes"]),
        "outcomes_path": str(OUTCOMES_PATH.relative_to(ROOT)),
        "trial_usage_ledger_path": str(USAGE_PATH.relative_to(ROOT)),
        "ineligible": state["ineligible"],
        "completed_schedule_item_count": len(state["completed_schedule_items"]),
        "attempt_count": len(state["attempts"]),
        "spend_checkpoints": state["spend_checkpoints"],
        "openrouter_usage_before_usd": usage_before,
        "openrouter_usage_after_usd": usage_after,
        "exact_incremental_openrouter_spend_usd": exact_spend,
        "project_openrouter_spend_before_usd": manifest["budget"][
            "project_openrouter_spend_before_usd"
        ],
        "project_openrouter_spend_after_usd": (
            float(manifest["budget"]["project_openrouter_spend_before_usd"])
            + exact_spend
        ),
        "additional_openrouter_cap_usd": manifest["budget"][
            "additional_openrouter_cap_usd"
        ],
        "final_key_query_error": final_key_query_error,
        "exact_spend_reconciled": final_key_query_error is None,
        "daytona_charge_usd": None,
        "daytona_charge_availability": (
            "The installed Daytona SDK exposes sandbox lifecycle, not account charges."
        ),
        "cleanup": {
            "initial_sandbox_ids": sorted(initial_sandbox_ids),
            "removed_new_sandbox_ids": cleanup["removed"],
            "cleanup_errors": cleanup["errors"],
            "remaining_new_sandbox_ids": remaining_sandbox_ids,
        },
    }


def run(manifest_path: Path = MANIFEST) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_hash = _validate_frozen_inputs(manifest_path, manifest)
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or not os.getenv("DAYTONA_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY and DAYTONA_API_KEY are required")

    key_info = query_openrouter_key(api_key)
    current_usage = float(key_info["usage"])
    if current_usage >= float(manifest["budget"]["usage_ceiling_usd"]) - 0.5:
        raise RuntimeError("frozen OpenRouter spend ceiling lacks one-trial reserve")

    initial_sandboxes = {sandbox.id for sandbox in Daytona().list()}
    if initial_sandboxes:
        raise RuntimeError("confirmatory execution requires zero pre-existing sandboxes")
    state = _load_or_create_state(
        manifest_hash, float(manifest["budget"]["usage_before_usd"])
    )
    if _target_met(state["outcomes"]):
        raise RuntimeError("confirmatory target is already sealed")

    server = NativeServer(SWITCHYARD)
    stop_reason = "frozen_pool_exhausted_before_dataset_target"
    status = "stopped"
    execution_error: str | None = None
    try:
        _finish_pending_group(
            state=state, manifest=manifest, server=server, api_key=api_key
        )
        if _target_met(state["outcomes"]):
            stop_reason = "confirmatory_dataset_target_reached"
            status = "complete"
        else:
            completed = set(state["completed_schedule_items"])
            for task in manifest["task_selection"]["ordered_pool"]:
                if _target_met(state["outcomes"]):
                    break
                for base_route, base_model in BASE_ROUTE_TO_MODEL.items():
                    if _target_met(state["outcomes"]):
                        break
                    for kind in KINDS:
                        if _target_met(state["outcomes"]):
                            break
                        schedule_item = f"{task['position']}:{base_route}:{kind}"
                        if schedule_item in completed:
                            continue
                        blocked = _acceptance_block_reason(
                            state["outcomes"],
                            task_id=task["task_id"],
                            kind=kind,
                            base_model=base_model,
                        )
                        if blocked:
                            state["ineligible"].append(
                                {
                                    "schedule_item": schedule_item,
                                    "task_id": task["task_id"],
                                    "base_model_id": base_model,
                                    "checkpoint_kind": kind,
                                    "reason": blocked,
                                    "selection_skip": True,
                                }
                            )
                            state["completed_schedule_items"].append(schedule_item)
                            completed.add(schedule_item)
                            _persist_state(state)
                            continue

                        label = base_route.rsplit("-", 1)[-1]
                        base, attempts = _run_with_infrastructure_retry(
                            server=server,
                            manifest=manifest,
                            run_root=OUTPUT_ROOT,
                            task=task,
                            route_id=base_route,
                            model_id=base_model,
                            label=f"base-{kind}-{task['position']:02d}-{label}",
                            max_turns=12,
                            capture_healthy=kind == "healthy",
                            capture_stuck=kind == "suspected_stuck",
                            stop_after_checkpoint=True,
                            enforce_branch_budget=False,
                        )
                        _record_attempts(state, attempts)
                        checkpoint, checkpoint_error = _eligible_checkpoint(base, kind)
                        if checkpoint_error:
                            state["ineligible"].append(
                                {
                                    "schedule_item": schedule_item,
                                    "task_id": task["task_id"],
                                    "base_model_id": base_model,
                                    "checkpoint_kind": kind,
                                    "reason": checkpoint_error,
                                    "scout_trial_valid": bool(base["valid"]),
                                    "provider_error": attempts[-1]["result"] is not None
                                    and bool(
                                        attempts[-1]["result"].get("exception_info") or {}
                                    ),
                                }
                            )
                            state["completed_schedule_items"].append(schedule_item)
                            completed.add(schedule_item)
                            _persist_state(state)
                            continue
                        assert checkpoint is not None

                        group_id = (
                            f"{kind}-{task['position']:02d}-{label}-"
                            f"t{checkpoint['observation']['turn']:02d}"
                        )
                        limits = _limits(checkpoint)
                        handoff_path = OUTPUT_ROOT / "handoffs" / f"{group_id}.md"
                        handoff_path.parent.mkdir(parents=True, exist_ok=True)
                        handoff_path.write_text(checkpoint["handoff"], encoding="utf-8")
                        state["pending_group"] = {
                            "schedule_item": schedule_item,
                            "group_id": group_id,
                            "task": task,
                            "base_route": base_route,
                            "base_model": base_model,
                            "checkpoint": checkpoint,
                            "limits": limits,
                            "handoff_path": str(handoff_path),
                            "rows": [],
                        }
                        _persist_state(state)
                        _finish_pending_group(
                            state=state,
                            manifest=manifest,
                            server=server,
                            api_key=api_key,
                        )
                        completed = set(state["completed_schedule_items"])
            if _target_met(state["outcomes"]):
                stop_reason = "confirmatory_dataset_target_reached"
                status = "complete"
    except RuntimeError as error:
        status = "stopped"
        execution_error = f"{type(error).__name__}: {error}"
        stop_reason = (
            "frozen_openrouter_spend_ceiling_reached"
            if "spend ceiling" in str(error)
            else "execution_runtime_error"
        )
    except (urllib.error.HTTPError, urllib.error.URLError) as error:
        status = "stopped"
        stop_reason = "provider_key_unavailable"
        execution_error = f"{type(error).__name__}: {error}"
    except KeyboardInterrupt:
        status = "interrupted"
        stop_reason = "operator_interrupt"
        execution_error = None
    except Exception as error:  # pragma: no cover - live safety net
        status = "stopped"
        stop_reason = "unexpected_execution_error"
        execution_error = f"{type(error).__name__}: {error}"
    finally:
        server.close()
        cleanup = _cleanup_new_sandboxes(initial_sandboxes)

    final_key_query_error = None
    try:
        usage_after = float(query_openrouter_key(api_key)["usage"])
    except (urllib.error.HTTPError, urllib.error.URLError) as error:
        usage_after = current_usage
        final_key_query_error = f"{type(error).__name__}: {error}"
    try:
        remaining_sandboxes = sorted(
            {sandbox.id for sandbox in Daytona().list()} - initial_sandboxes
        )
    except Exception as error:  # pragma: no cover - live safety net
        remaining_sandboxes = []
        cleanup["errors"].append(f"final list: {type(error).__name__}: {error}")

    state["status"] = status
    state["stop_reason"] = stop_reason
    state["execution_error"] = execution_error
    state["spend_checkpoints"].append(
        {
            "kind": "final",
            "accepted_groups": len(_group_rows(state["outcomes"])),
            "usage_usd": usage_after,
            "recorded_at": datetime.now(UTC).isoformat(),
        }
    )
    _persist_state(state)
    ledger = _final_ledger(
        state=state,
        manifest=manifest,
        manifest_path=manifest_path,
        manifest_hash=manifest_hash,
        status=status,
        stop_reason=stop_reason,
        execution_error=execution_error,
        usage_after=usage_after,
        final_key_query_error=final_key_query_error,
        cleanup=cleanup,
        initial_sandbox_ids=initial_sandboxes,
        remaining_sandbox_ids=remaining_sandboxes,
    )
    _write_json(LEDGER_PATH, ledger)
    return ledger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    print(json.dumps(run(args.manifest), indent=2))


if __name__ == "__main__":
    main()
