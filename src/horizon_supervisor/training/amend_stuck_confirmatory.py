from __future__ import annotations

import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from horizon_supervisor.training.run_stuck_pilot import _checkpoint_records, _limits

ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = ROOT / "artifacts/official/stuck-confirmatory-v1"
SOURCE_MANIFEST = OUTPUT_ROOT / "frozen-manifest-v0.json"
AMENDED_MANIFEST = OUTPUT_ROOT / "frozen-manifest-v1.json"
STATE_PATH = OUTPUT_ROOT / "execution-state-v0.json"
OUTCOMES_PATH = OUTPUT_ROOT / "matched-outcomes-v0.jsonl"
LEDGER_PATH = OUTPUT_ROOT / "execution-ledger-v0.json"
AMENDMENT_ID = "explicit-four-way-snapshot-branching-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _source_manifest() -> tuple[dict[str, Any], str]:
    sidecar = SOURCE_MANIFEST.with_suffix(".sha256")
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    actual = _sha256(SOURCE_MANIFEST)
    if actual != expected:
        raise RuntimeError("source manifest hash mismatch")
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "stuck-confirmatory-manifest.v0":
        raise RuntimeError("unexpected source manifest schema")
    return manifest, actual


def _matched_checkpoint(record_path: Path, kind: str) -> dict[str, Any] | None:
    checkpoints = [
        row
        for row in _checkpoint_records(record_path)
        if row["checkpoint_kind"] == kind
    ]
    if len(checkpoints) > 1:
        raise RuntimeError("pre-amendment attempt contains duplicate checkpoints")
    return checkpoints[0] if checkpoints else None


def _migrate_state(
    *, manifest: dict[str, Any], amended_manifest_hash: str, source_manifest_hash: str
) -> dict[str, Any]:
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    if state.get("manifest_sha256") != source_manifest_hash:
        raise RuntimeError("execution state does not belong to the source manifest")
    if state.get("outcomes") or OUTCOMES_PATH.read_text(encoding="utf-8").strip():
        raise RuntimeError("cannot amend after a matched outcome has been accepted")

    backup_path = OUTPUT_ROOT / "execution-state-pre-amendment-v0.json"
    if backup_path.exists():
        raise FileExistsError(backup_path)
    _write_json(backup_path, state)
    backup_hash = _sha256(backup_path)
    backup_path.with_suffix(".sha256").write_text(
        f"{backup_hash}  {backup_path.name}\n", encoding="utf-8"
    )

    task_by_position = {
        int(task["position"]): task
        for task in manifest["task_selection"]["ordered_pool"]
    }
    pending: dict[str, Any] | None = None
    reopened_items: list[str] = []
    for entry in state.get("ineligible", []):
        if entry.get("reason") != "base trial did not produce a valid verifier outcome":
            continue
        item = str(entry["schedule_item"])
        position_text, base_route, kind = item.split(":")
        task = task_by_position[int(position_text)]
        matching = [
            attempt
            for attempt in state.get("attempts", [])
            if attempt["task_id"] == task["task_id"]
            and attempt["route_id"] == base_route
            and f"base-{kind}-" in attempt["job_name"]
        ]
        if len(matching) != 1:
            raise RuntimeError(f"cannot identify pre-amendment scout attempt: {item}")
        attempt = matching[0]
        checkpoint = _matched_checkpoint(Path(attempt["record_path"]), kind)
        entry["pre_amendment_reason"] = entry["reason"]
        entry["amendment_id"] = AMENDMENT_ID
        if checkpoint is None:
            entry["reason"] = "requested checkpoint did not occur"
            entry["amendment_disposition"] = "remains structurally ineligible"
            continue
        if checkpoint.get("state_transfer_eligible") is not True:
            entry["reason"] = str(checkpoint["state_transfer_ineligibility_reason"])
            entry["amendment_disposition"] = "remains structurally ineligible"
            continue
        if pending is not None:
            raise RuntimeError("more than one pre-amendment checkpoint requires recovery")

        base_model = manifest["models"]["routes"][base_route]
        label = base_route.rsplit("-", 1)[-1]
        group_id = (
            f"{kind}-{int(position_text):02d}-{label}-"
            f"t{checkpoint['observation']['turn']:02d}"
        )
        handoff_path = OUTPUT_ROOT / "handoffs" / f"{group_id}.md"
        handoff_path.parent.mkdir(parents=True, exist_ok=True)
        handoff_path.write_text(checkpoint["handoff"], encoding="utf-8")
        pending = {
            "schedule_item": item,
            "group_id": group_id,
            "task": task,
            "base_route": base_route,
            "base_model": base_model,
            "checkpoint": checkpoint,
            "limits": _limits(checkpoint),
            "handoff_path": str(handoff_path),
            "rows": [],
        }
        entry["reason"] = "valid checkpoint recovered by pre-outcome amendment"
        entry["amendment_disposition"] = "recovered as pending matched group"
        reopened_items.append(item)

    if pending is None:
        raise RuntimeError("no valid pre-amendment checkpoint was available to recover")
    completed = list(state.get("completed_schedule_items", []))
    for item in reopened_items:
        if item not in completed:
            raise RuntimeError(f"recovered schedule item was not marked complete: {item}")
        completed.remove(item)
    state["completed_schedule_items"] = completed
    state["pending_group"] = pending
    state["manifest_sha256"] = amended_manifest_hash
    state["status"] = "in_progress"
    state["stop_reason"] = None
    state["execution_error"] = None
    state["amendment_history"] = [
        {
            "id": AMENDMENT_ID,
            "source_manifest_sha256": source_manifest_hash,
            "amended_manifest_sha256": amended_manifest_hash,
            "accepted_outcomes_before_amendment": 0,
            "pre_amendment_state_path": str(backup_path.relative_to(ROOT)),
            "pre_amendment_state_sha256": backup_hash,
            "recovered_schedule_items": reopened_items,
            "recorded_at": datetime.now(UTC).isoformat(),
        }
    ]
    _write_json(STATE_PATH, state)
    return state


def amend() -> dict[str, Any]:
    if AMENDED_MANIFEST.exists():
        raise FileExistsError(AMENDED_MANIFEST)
    source, source_hash = _source_manifest()
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    if state.get("outcomes") or OUTCOMES_PATH.read_text(encoding="utf-8").strip():
        raise RuntimeError("cannot amend after a matched outcome has been accepted")

    amended = copy.deepcopy(source)
    amended["schema_version"] = "stuck-confirmatory-manifest.v1"
    amended["amended_at"] = datetime.now(UTC).isoformat()
    amended["amendment"] = {
        "id": AMENDMENT_ID,
        "scope": "orchestration_only",
        "source_manifest_path": str(SOURCE_MANIFEST.relative_to(ROOT)),
        "source_manifest_sha256": source_hash,
        "accepted_outcomes_before_amendment": 0,
        "reason": (
            "The v0 runner required the post-checkpoint scouting trajectory to have "
            "a final verifier result and reused its tail as continuation. A valid "
            "checkpoint must instead be independent of the later scout result, and "
            "all four causal arms must launch from the same rehydrated snapshot."
        ),
        "unchanged": [
            "detector and thresholds",
            "task pool and order",
            "models and routes",
            "checkpoint timing",
            "per-arm limits",
            "sampling and stopping rules",
            "analysis and decision gates",
            "twenty-dollar experiment ceiling",
        ],
        "pre_amendment_exact_spend_usd": float(
            json.loads(LEDGER_PATH.read_text(encoding="utf-8"))[
                "exact_incremental_openrouter_spend_usd"
            ]
        ),
    }
    amended["execution"]["scout_checkpoint_acceptance"] = (
        "Exactly one structurally eligible checkpoint is sufficient; the later "
        "scouting trajectory need not produce a final verifier result."
    )
    amended["execution"]["continue_arm"] = (
        "Explicit fresh Daytona branch rehydrated from the same checkpoint archive "
        "and public handoff as the preserved-state switch arms."
    )
    runner_path = ROOT / "src/horizon_supervisor/training/run_stuck_confirmatory.py"
    amended["integrity"]["code_sha256"][str(runner_path.relative_to(ROOT))] = _sha256(
        runner_path
    )
    amended["integrity"]["amendment_script_sha256"] = _sha256(Path(__file__))
    _write_json(AMENDED_MANIFEST, amended)
    amended_hash = _sha256(AMENDED_MANIFEST)
    AMENDED_MANIFEST.with_suffix(".sha256").write_text(
        f"{amended_hash}  {AMENDED_MANIFEST.name}\n", encoding="utf-8"
    )
    migrated = _migrate_state(
        manifest=amended,
        amended_manifest_hash=amended_hash,
        source_manifest_hash=source_hash,
    )
    return {
        "manifest_path": str(AMENDED_MANIFEST.relative_to(ROOT)),
        "manifest_sha256": amended_hash,
        "accepted_outcomes_before_amendment": len(migrated["outcomes"]),
        "recovered_group_id": migrated["pending_group"]["group_id"],
    }


def main() -> None:
    print(json.dumps(amend(), indent=2))


if __name__ == "__main__":
    main()
