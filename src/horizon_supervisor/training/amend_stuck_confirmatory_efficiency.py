from __future__ import annotations

import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = ROOT / "artifacts/official/stuck-confirmatory-v1"
SOURCE_MANIFEST = OUTPUT_ROOT / "frozen-manifest-v1.json"
AMENDED_MANIFEST = OUTPUT_ROOT / "frozen-manifest-v2.json"
STATE_PATH = OUTPUT_ROOT / "execution-state-v0.json"
OUTCOMES_PATH = OUTPUT_ROOT / "matched-outcomes-v0.jsonl"
AMENDMENT_ID = "stop-scout-after-sealed-checkpoint-v2"
EXPECTED_ACCEPTED_OUTCOMES = 4
CODE_PATHS = (
    "src/horizon_supervisor/benchmark/pilot_harbor.py",
    "src/horizon_supervisor/training/run_stuck_pilot.py",
    "src/horizon_supervisor/training/run_stuck_confirmatory.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _outcomes() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in OUTCOMES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def amend() -> dict[str, Any]:
    if AMENDED_MANIFEST.exists():
        raise FileExistsError(AMENDED_MANIFEST)

    sidecar = SOURCE_MANIFEST.with_suffix(".sha256")
    expected_source_hash = sidecar.read_text(encoding="utf-8").split()[0]
    source_hash = _sha256(SOURCE_MANIFEST)
    if source_hash != expected_source_hash:
        raise RuntimeError("source manifest hash mismatch")
    source = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    if source.get("schema_version") != "stuck-confirmatory-manifest.v1":
        raise RuntimeError("unexpected source manifest schema")

    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    if state.get("manifest_sha256") != source_hash:
        raise RuntimeError("execution state does not belong to the source manifest")
    if state.get("pending_group"):
        raise RuntimeError("cannot amend while a matched group is pending")
    if state.get("status") != "interrupted":
        raise RuntimeError("execution must be cleanly interrupted before amendment")

    outcomes = _outcomes()
    if len(outcomes) != EXPECTED_ACCEPTED_OUTCOMES:
        raise RuntimeError("unexpected accepted-outcome boundary")
    if outcomes != state.get("outcomes"):
        raise RuntimeError("state and sealed outcomes disagree")
    group_ids = {str(row["group_id"]) for row in outcomes}
    if len(group_ids) != 1:
        raise RuntimeError("expected exactly one sealed matched group")

    backup_path = OUTPUT_ROOT / "execution-state-pre-efficiency-amendment-v0.json"
    if backup_path.exists():
        raise FileExistsError(backup_path)
    _write_json(backup_path, state)
    backup_hash = _sha256(backup_path)
    backup_path.with_suffix(".sha256").write_text(
        f"{backup_hash}  {backup_path.name}\n", encoding="utf-8"
    )

    amended = copy.deepcopy(source)
    amended["schema_version"] = "stuck-confirmatory-manifest.v2"
    amended["amended_at"] = datetime.now(UTC).isoformat()
    amended["execution_efficiency_amendment"] = {
        "id": AMENDMENT_ID,
        "scope": "orchestration_only",
        "source_manifest_path": str(SOURCE_MANIFEST.relative_to(ROOT)),
        "source_manifest_sha256": source_hash,
        "accepted_outcomes_before_amendment": EXPECTED_ACCEPTED_OUTCOMES,
        "accepted_groups_before_amendment": 1,
        "sealed_outcomes_sha256": _sha256(OUTCOMES_PATH),
        "reason": (
            "A scout continued to spend turns after its requested checkpoint had "
            "already been archived. Future scouts stop their agent session only "
            "after the checkpoint record and portable workspace archive are sealed."
        ),
        "causal_invariance": (
            "Post-checkpoint scout work is never used for checkpoint eligibility, "
            "branch initialization, branch outcomes, selection, or analysis."
        ),
        "unchanged": [
            "detector and thresholds",
            "task pool and schedule order",
            "models and routes",
            "checkpoint timing and captured state",
            "four matched branch actions",
            "per-arm limits",
            "sampling and stopping rules",
            "analysis and decision gates",
            "twenty-dollar experiment ceiling",
        ],
    }
    for relative in CODE_PATHS:
        amended["integrity"]["code_sha256"][relative] = _sha256(ROOT / relative)
    script_path = ROOT / (
        "src/horizon_supervisor/training/amend_stuck_confirmatory_efficiency.py"
    )
    amended["integrity"]["efficiency_amendment_script_sha256"] = _sha256(
        script_path
    )

    _write_json(AMENDED_MANIFEST, amended)
    amended_hash = _sha256(AMENDED_MANIFEST)
    AMENDED_MANIFEST.with_suffix(".sha256").write_text(
        f"{amended_hash}  {AMENDED_MANIFEST.name}\n", encoding="utf-8"
    )

    state["manifest_sha256"] = amended_hash
    state["status"] = "in_progress"
    state["stop_reason"] = None
    state["execution_error"] = None
    state.setdefault("amendment_history", []).append(
        {
            "id": AMENDMENT_ID,
            "source_manifest_sha256": source_hash,
            "amended_manifest_sha256": amended_hash,
            "accepted_outcomes_before_amendment": EXPECTED_ACCEPTED_OUTCOMES,
            "accepted_groups_before_amendment": 1,
            "pre_amendment_state_path": str(backup_path.relative_to(ROOT)),
            "pre_amendment_state_sha256": backup_hash,
            "recorded_at": datetime.now(UTC).isoformat(),
        }
    )
    _write_json(STATE_PATH, state)
    return {
        "manifest_path": str(AMENDED_MANIFEST.relative_to(ROOT)),
        "manifest_sha256": amended_hash,
        "state_path": str(STATE_PATH.relative_to(ROOT)),
        "accepted_outcomes_before_amendment": EXPECTED_ACCEPTED_OUTCOMES,
    }


def main() -> None:
    print(json.dumps(amend(), indent=2))


if __name__ == "__main__":
    main()
