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
from horizon_supervisor.stuck_detector import SuspectedStuckV0

ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = ROOT / "artifacts/official/stuck-intervention-pilot-v0"
PARENT_MANIFEST = OUTPUT_ROOT / "frozen-pilot-manifest-v3.json"
RESUME_MANIFEST = OUTPUT_ROOT / "frozen-pilot-manifest-v4.json"
INTERRUPTED_EXECUTION = OUTPUT_ROOT / "execution-20260903T220230658417Z"
RUNNER = ROOT / "src/horizon_supervisor/training/run_stuck_pilot.py"
DETECTOR = ROOT / "src/horizon_supervisor/stuck_detector.py"
RECOVERED_OUTCOMES = OUTPUT_ROOT / "recovered-qwen-stuck-group-v0.jsonl"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(item.read_bytes()).digest())
    return digest.hexdigest()


def _validate_interrupted_execution() -> None:
    if not INTERRUPTED_EXECUTION.is_dir():
        raise RuntimeError("the observation-gap execution root is missing")
    outcome_files = list(INTERRUPTED_EXECUTION.glob("matched-branch-outcomes*.jsonl"))
    if outcome_files:
        raise RuntimeError("the observation-gap execution unexpectedly has outcomes")
    healthy_results = list(
        (
            INTERRUPTED_EXECUTION
            / "jobs"
            / "base-healthy-01-qwen-20260903T220230724684Z"
        ).glob("*/result.json")
    )
    if len(healthy_results) != 1:
        raise RuntimeError("the expected healthy-base result is missing")
    result = json.loads(healthy_results[0].read_text(encoding="utf-8"))
    exception = result.get("exception_info") or {}
    if (
        exception.get("exception_type") != "ValueError"
        or exception.get("exception_message")
        != "turn observations must be consecutive"
        or result.get("verifier_result") is not None
    ):
        raise RuntimeError("the expected observation-gap failure changed")
    partial_task_two = (
        INTERRUPTED_EXECUTION
        / "jobs"
        / "base-suspected_stuck-02-flash-20260903T220756784656Z"
    )
    if not partial_task_two.is_dir() or list(partial_task_two.glob("*/result.json")):
        raise RuntimeError("the interrupted task-two base state changed")


def freeze_observation_gap_resume(*, current_key_usage_usd: float) -> dict[str, Any]:
    if RESUME_MANIFEST.exists():
        raise FileExistsError(f"v4 resume manifest already exists: {RESUME_MANIFEST}")
    parent_hash = PARENT_MANIFEST.with_suffix(".sha256").read_text().split()[0]
    if _sha256(PARENT_MANIFEST) != parent_hash:
        raise RuntimeError("parent v3 manifest hash mismatch")
    parent = json.loads(PARENT_MANIFEST.read_text(encoding="utf-8"))
    _validate_interrupted_execution()
    recovered_hash = RECOVERED_OUTCOMES.with_suffix(".sha256").read_text().split()[0]
    if _sha256(RECOVERED_OUTCOMES) != recovered_hash:
        raise RuntimeError("the recovered matched group hash changed")
    if list(Daytona().list()):
        raise RuntimeError("Daytona must be empty before freezing the v4 resume")

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required")
    current_key = query_openrouter_key(api_key)
    live_usage = float(current_key["usage"])
    if abs(live_usage - current_key_usage_usd) > 1e-9:
        raise RuntimeError("the supplied current-key usage does not match OpenRouter")
    current_limit = float(current_key["limit"])
    previous_baseline = float(parent["budget"]["usage_before_usd"])
    prior_spend = float(parent["budget"]["prior_key_spend_usd"]) + (
        current_key_usage_usd - previous_baseline
    )
    total_cap = float(parent["budget"]["additional_openrouter_cap_usd"])
    if not 0 <= prior_spend < total_cap:
        raise ValueError("reconciled prior spend is outside the original pilot cap")
    remaining_cap = total_cap - prior_spend

    manifest = json.loads(PARENT_MANIFEST.read_text(encoding="utf-8"))
    manifest["schema_version"] = "matched-stuck-intervention-pilot.v4"
    manifest["frozen_at"] = datetime.now(UTC).isoformat()
    manifest["detector"] = SuspectedStuckV0.frozen_spec()
    manifest["integrity"]["code_sha256"][str(DETECTOR.relative_to(ROOT))] = _sha256(
        DETECTOR
    )
    manifest["integrity"]["code_sha256"][str(RUNNER.relative_to(ROOT))] = _sha256(
        RUNNER
    )
    manifest["budget"].update(
        {
            "usage_before_usd": current_key_usage_usd,
            "usage_ceiling_usd": current_key_usage_usd + remaining_cap,
            "dedicated_key_hard_limit_usd": current_limit,
            "effective_current_key_ceiling_usd": min(
                current_key_usage_usd + remaining_cap, current_limit
            ),
            "prior_key_spend_usd": prior_spend,
            "remaining_total_pilot_cap_usd": remaining_cap,
        }
    )
    resume = dict(parent["resume"])
    prior_roots = list(resume["prior_execution_roots"])
    interrupted_relative = str(INTERRUPTED_EXECUTION.relative_to(ROOT))
    prior_roots.append(interrupted_relative)
    prior_hashes = dict(resume["prior_execution_tree_sha256"])
    prior_hashes[interrupted_relative] = _tree_sha256(INTERRUPTED_EXECUTION)
    prior_ineligible = list(resume["prior_ineligible"])
    prior_ineligible.append(
        {
            "schedule_item": "1:gate7/fixed-qwen:healthy",
            "reason": (
                "Harness observation-gap ValueError after a valid turn-4 healthy "
                "checkpoint; no verifier result or matched branch was accepted."
            ),
            "rescheduled": True,
        }
    )
    prior_ineligible.append(
        {
            "schedule_item": "2:gate7/fixed-flash:suspected_stuck",
            "reason": "Stopped during environment setup before a trial result existed.",
            "rescheduled": True,
        }
    )
    resume.update(
        {
            "parent_manifest_path": str(PARENT_MANIFEST.relative_to(ROOT)),
            "parent_manifest_sha256": parent_hash,
            "prior_execution_roots": prior_roots,
            "prior_execution_tree_sha256": prior_hashes,
            "prior_ineligible": prior_ineligible,
            "prior_outcomes_path": str(RECOVERED_OUTCOMES.relative_to(ROOT)),
            "prior_accepted_outcome_count": 6,
            "prior_group_counts": {"suspected_stuck": 1, "healthy": 0},
            "resumption_changes_are_infrastructure_only": True,
        }
    )
    manifest["resume"] = resume
    manifest["observation_gap_amendment"] = {
        "cause": (
            "Terminus counts every model episode, while the detector observes only "
            "command-execution episodes; a commandless episode can skip an index."
        ),
        "change": (
            "Accept strictly increasing observation indices. When an index gap occurs, "
            "reset consecutive persistence and immediate-loop evidence."
        ),
        "scientific_effect": (
            "Preserves the frozen requirement for consecutive evidence and prevents a "
            "harness exception; it does not add signals or inspect outcomes."
        ),
        "failed_schedule_items_rescheduled": [
            "1:gate7/fixed-qwen:healthy",
            "2:gate7/fixed-flash:suspected_stuck",
        ],
    }
    RESUME_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    RESUME_MANIFEST.with_suffix(".sha256").write_text(
        f"{_sha256(RESUME_MANIFEST)}  {RESUME_MANIFEST.name}\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-key-usage-usd", type=float, required=True)
    args = parser.parse_args()
    manifest = freeze_observation_gap_resume(
        current_key_usage_usd=args.current_key_usage_usd
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
