from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from daytona import Daytona

from horizon_supervisor.benchmark.gate7 import query_openrouter_key

ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = ROOT / "artifacts/official/stuck-intervention-pilot-v0"
PARENT_MANIFEST = OUTPUT_ROOT / "frozen-pilot-manifest-v2.json"
RESUME_MANIFEST = OUTPUT_ROOT / "frozen-pilot-manifest-v3.json"
PRIOR_EXECUTION = OUTPUT_ROOT / "execution-20260903T153354806121Z"
RECOVERED_OUTCOMES = OUTPUT_ROOT / "recovered-qwen-stuck-group-v0.jsonl"
RUNNER = ROOT / "src/horizon_supervisor/training/run_stuck_pilot.py"
RECOVERY_RUNNER = ROOT / "src/horizon_supervisor/training/recover_stuck_pilot_group.py"
RECOVERED_SCHEDULE_ITEM = "1:gate7/fixed-qwen:suspected_stuck"
RECOVERED_GROUP_ID = "suspected_stuck-01-qwen-t10"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(item.read_bytes()).digest())
    return digest.hexdigest()


def _validate_prior_execution() -> dict[str, Any]:
    ledger_path = PRIOR_EXECUTION / "execution-ledger.json"
    if not ledger_path.is_file():
        raise RuntimeError("the interrupted v2 execution ledger is missing")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if ledger.get("status") != "in_progress" or ledger.get("valid_outcome_count") != 0:
        raise RuntimeError("the interrupted v2 ledger changed")
    attempts = ledger.get("attempts") or []
    if len(attempts) != 6 or sum(bool(row.get("valid")) for row in attempts) != 5:
        raise RuntimeError("the interrupted v2 attempt roster changed")
    invalid = [row for row in attempts if not row.get("valid")]
    if len(invalid) != 1 or "restart_kimi_clean" not in invalid[0]["job_name"]:
        raise RuntimeError("the expected Kimi-clean arm is no longer the sole invalid arm")
    incomplete = [
        row
        for row in ledger.get("ineligible") or []
        if row.get("group_id") == RECOVERED_GROUP_ID
    ]
    if len(incomplete) != 1 or incomplete[0].get("invalid_actions") != [
        "restart_kimi_clean"
    ]:
        raise RuntimeError("the interrupted matched-group record changed")
    return ledger


def freeze_recovery(*, completed_key_final_usage_usd: float) -> dict[str, Any]:
    if RESUME_MANIFEST.exists() or RECOVERED_OUTCOMES.exists():
        raise FileExistsError("the v3 recovery contract or its output already exists")
    parent_hash = PARENT_MANIFEST.with_suffix(".sha256").read_text().split()[0]
    if _sha256(PARENT_MANIFEST) != parent_hash:
        raise RuntimeError("parent v2 manifest hash mismatch")
    parent = json.loads(PARENT_MANIFEST.read_text(encoding="utf-8"))
    prior_ledger = _validate_prior_execution()
    if list(Daytona().list()):
        raise RuntimeError("Daytona must be empty before freezing recovery state")

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("a fresh OPENROUTER_API_KEY is required")
    current_key = query_openrouter_key(api_key)
    current_usage = float(current_key["usage"])
    current_limit = float(current_key["limit"])
    previous_key_baseline = float(parent["budget"]["usage_before_usd"])
    previous_key_spend = completed_key_final_usage_usd - previous_key_baseline
    prior_spend = float(parent["budget"]["prior_key_spend_usd"]) + previous_key_spend
    total_cap = float(parent["budget"]["additional_openrouter_cap_usd"])
    if not 0 <= prior_spend < total_cap:
        raise ValueError("reconciled prior spend is outside the original pilot cap")
    remaining_cap = total_cap - prior_spend

    manifest = json.loads(PARENT_MANIFEST.read_text(encoding="utf-8"))
    manifest["schema_version"] = "matched-stuck-intervention-pilot.v3"
    manifest["frozen_at"] = datetime.now(UTC).isoformat()
    manifest["integrity"]["code_sha256"][str(RUNNER.relative_to(ROOT))] = _sha256(
        RUNNER
    )
    manifest["integrity"]["code_sha256"][
        str(RECOVERY_RUNNER.relative_to(ROOT))
    ] = _sha256(RECOVERY_RUNNER)
    manifest["budget"].update(
        {
            "usage_before_usd": current_usage,
            "usage_ceiling_usd": current_usage + remaining_cap,
            "dedicated_key_hard_limit_usd": current_limit,
            "effective_current_key_ceiling_usd": min(
                current_usage + remaining_cap, current_limit
            ),
            "prior_key_final_usage_usd": completed_key_final_usage_usd,
            "prior_key_spend_usd": prior_spend,
            "remaining_total_pilot_cap_usd": remaining_cap,
        }
    )

    parent_resume = parent.get("resume") or {}
    prior_roots = list(parent_resume.get("prior_execution_roots") or [])
    prior_roots.append(str(PRIOR_EXECUTION.relative_to(ROOT)))
    prior_tree_hashes = dict(parent_resume.get("prior_execution_tree_sha256") or {})
    prior_tree_hashes[str(PRIOR_EXECUTION.relative_to(ROOT))] = _tree_sha256(
        PRIOR_EXECUTION
    )
    completed_items = list(parent_resume.get("completed_schedule_items") or [])
    completed_items.append(RECOVERED_SCHEDULE_ITEM)
    prior_ineligible = [
        row
        for row in parent_resume.get("prior_ineligible") or []
        if row.get("group_id") != RECOVERED_GROUP_ID
    ]
    prior_ineligible.append(
        {
            "schedule_item": "1:gate7/fixed-qwen:healthy",
            "reason": (
                "A v2 healthy base trial was interrupted before it produced an accepted "
                "checkpoint or ledger attempt; the frozen schedule item remains eligible."
            ),
            "rescheduled": True,
        }
    )
    manifest["resume"] = {
        "parent_manifest_path": str(PARENT_MANIFEST.relative_to(ROOT)),
        "parent_manifest_sha256": parent_hash,
        "prior_execution_roots": prior_roots,
        "prior_execution_tree_sha256": prior_tree_hashes,
        "completed_schedule_items": completed_items,
        "prior_group_counts": {"suspected_stuck": 1, "healthy": 0},
        "prior_accepted_outcome_count": 6,
        "prior_outcomes_path": str(RECOVERED_OUTCOMES.relative_to(ROOT)),
        "prior_ineligible": prior_ineligible,
        "resumption_changes_are_infrastructure_only": True,
        "unchanged_experiment_fields": [
            "detector",
            "task roster and order",
            "model roster",
            "agent and branch limits",
            "branch arms",
            "analysis",
            "sampling and stopping rules",
            "total additional spend cap",
        ],
    }
    manifest["recovery"] = {
        "group_id": RECOVERED_GROUP_ID,
        "schedule_item": RECOVERED_SCHEDULE_ITEM,
        "task_id": "implement-gmm-em-cli",
        "task_position": 1,
        "base_route_id": "gate7/fixed-qwen",
        "base_model_id": "qwen/qwen3.8-27b",
        "source_execution_root": str(PRIOR_EXECUTION.relative_to(ROOT)),
        "source_execution_tree_sha256": prior_tree_hashes[
            str(PRIOR_EXECUTION.relative_to(ROOT))
        ],
        "source_ledger_path": str(
            (PRIOR_EXECUTION / "execution-ledger.json").relative_to(ROOT)
        ),
        "source_attempt_count": len(prior_ledger["attempts"]),
        "sealed_valid_actions": [
            "continue_current_state",
            "restart_current_clean",
            "switch_value_state",
            "restart_value_clean",
            "switch_kimi_state",
        ],
        "recovered_action": "restart_kimi_clean",
        "output_path": str(RECOVERED_OUTCOMES.relative_to(ROOT)),
        "policy": (
            "Run only the sole funding-invalid Kimi-clean arm, reconstruct the five "
            "sealed siblings, and accept the group only if all six arms are valid."
        ),
    }
    RESUME_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    RESUME_MANIFEST.with_suffix(".sha256").write_text(
        f"{_sha256(RESUME_MANIFEST)}  {RESUME_MANIFEST.name}\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--completed-key-final-usage-usd", type=float, required=True)
    args = parser.parse_args()
    manifest = freeze_recovery(
        completed_key_final_usage_usd=args.completed_key_final_usage_usd
    )
    print(
        json.dumps(
            {
                "manifest": str(RESUME_MANIFEST),
                "prior_key_spend_usd": manifest["budget"]["prior_key_spend_usd"],
                "remaining_total_pilot_cap_usd": manifest["budget"][
                    "remaining_total_pilot_cap_usd"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
