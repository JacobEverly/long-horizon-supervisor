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
PARENT_MANIFEST = OUTPUT_ROOT / "frozen-pilot-manifest-v4.json"
RESUME_MANIFEST = OUTPUT_ROOT / "frozen-pilot-manifest-v5.json"
INTERRUPTED_EXECUTION = OUTPUT_ROOT / "execution-20260903T221219738870Z"
QWEN_HEALTHY_ITEM = "1:gate7/fixed-qwen:healthy"


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
    if list(INTERRUPTED_EXECUTION.glob("matched-branch-outcomes*.jsonl")):
        raise RuntimeError("the conservative-stop execution unexpectedly has outcomes")
    result_paths = list(
        (
            INTERRUPTED_EXECUTION
            / "jobs"
            / "base-healthy-01-qwen-20260903T221219800660Z"
        ).glob("*/result.json")
    )
    if len(result_paths) != 1:
        raise RuntimeError("the expected Qwen healthy-base result is missing")
    result = json.loads(result_paths[0].read_text(encoding="utf-8"))
    exception = result.get("exception_info") or {}
    if (
        exception.get("exception_type") != "OutputLengthExceededError"
        or result.get("verifier_result") is not None
    ):
        raise RuntimeError("the expected Qwen output-limit result changed")
    partial_flash = (
        INTERRUPTED_EXECUTION
        / "jobs"
        / "base-suspected_stuck-02-flash-20260903T221535203571Z"
    )
    if not partial_flash.is_dir() or list(partial_flash.glob("*/result.json")):
        raise RuntimeError("the interrupted task-two Flash state changed")


def freeze_conservative_stop_resume(*, current_key_usage_usd: float) -> dict[str, Any]:
    if RESUME_MANIFEST.exists():
        raise FileExistsError(f"v5 resume manifest already exists: {RESUME_MANIFEST}")
    parent_hash = PARENT_MANIFEST.with_suffix(".sha256").read_text().split()[0]
    if _sha256(PARENT_MANIFEST) != parent_hash:
        raise RuntimeError("parent v4 manifest hash mismatch")
    parent = json.loads(PARENT_MANIFEST.read_text(encoding="utf-8"))
    _validate_interrupted_execution()
    if list(Daytona().list()):
        raise RuntimeError("Daytona must be empty before freezing the v5 resume")

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required")
    current_key = query_openrouter_key(api_key)
    live_usage = float(current_key["usage"])
    if abs(live_usage - current_key_usage_usd) > 1e-9:
        raise RuntimeError("the supplied current-key usage does not match OpenRouter")
    previous_baseline = float(parent["budget"]["usage_before_usd"])
    prior_spend = float(parent["budget"]["prior_key_spend_usd"]) + (
        current_key_usage_usd - previous_baseline
    )
    total_cap = float(parent["budget"]["additional_openrouter_cap_usd"])
    if not 0 <= prior_spend < total_cap:
        raise ValueError("reconciled prior spend is outside the original pilot cap")
    remaining_cap = total_cap - prior_spend
    current_limit = float(current_key["limit"])

    manifest = json.loads(PARENT_MANIFEST.read_text(encoding="utf-8"))
    manifest["schema_version"] = "matched-stuck-intervention-pilot.v5"
    manifest["frozen_at"] = datetime.now(UTC).isoformat()
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
    completed = list(resume["completed_schedule_items"])
    if QWEN_HEALTHY_ITEM not in completed:
        completed.append(QWEN_HEALTHY_ITEM)
    relative_root = str(INTERRUPTED_EXECUTION.relative_to(ROOT))
    prior_roots = list(resume["prior_execution_roots"])
    prior_roots.append(relative_root)
    prior_hashes = dict(resume["prior_execution_tree_sha256"])
    prior_hashes[relative_root] = _tree_sha256(INTERRUPTED_EXECUTION)
    prior_ineligible = list(resume["prior_ineligible"])
    prior_ineligible.extend(
        [
            {
                "schedule_item": QWEN_HEALTHY_ITEM,
                "reason": (
                    "Qwen exceeded the per-response output cap after its frozen "
                    "corrective retry and produced no verifier result or checkpoint."
                ),
            },
            {
                "schedule_item": "2:gate7/fixed-flash:suspected_stuck",
                "reason": (
                    "Conservatively stopped during the first Flash base trajectory "
                    "before a verifier result; the schedule item remains eligible."
                ),
                "rescheduled": True,
            },
        ]
    )
    resume.update(
        {
            "parent_manifest_path": str(PARENT_MANIFEST.relative_to(ROOT)),
            "parent_manifest_sha256": parent_hash,
            "prior_execution_roots": prior_roots,
            "prior_execution_tree_sha256": prior_hashes,
            "completed_schedule_items": completed,
            "prior_ineligible": prior_ineligible,
        }
    )
    manifest["resume"] = resume
    manifest["conservative_stop_amendment"] = {
        "cause": (
            "Telemetry was inspected after Qwen had already terminated and after the "
            "next Flash job reset the router statistics; no route mismatch occurred."
        ),
        "qwen_item_disposition": "frozen output-limit failure; do not rerun",
        "flash_item_disposition": "interrupted before a result; rerun unchanged",
        "experiment_policy_changed": False,
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
    manifest = freeze_conservative_stop_resume(
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
