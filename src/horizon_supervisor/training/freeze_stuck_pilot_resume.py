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
PARENT_MANIFEST = OUTPUT_ROOT / "frozen-pilot-manifest-v1.json"
RESUME_MANIFEST = OUTPUT_ROOT / "frozen-pilot-manifest-v2.json"
RUNNER = ROOT / "src/horizon_supervisor/training/run_stuck_pilot.py"
EXPECTED_EXECUTIONS = (
    "execution-20260903T031421290547Z",
    "execution-20260903T032001631978Z",
)
COMPLETED_SCHEDULE_ITEMS = (
    "1:gate7/fixed-flash:suspected_stuck",
    "1:gate7/fixed-flash:healthy",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(item.read_bytes()).digest())
    return digest.hexdigest()


def _matched_checkpoint_count(path: Path) -> int:
    count = 0
    for record in (path / "records").glob("*.jsonl"):
        for line in record.read_text(encoding="utf-8").splitlines():
            if json.loads(line).get("schema_version") == "matched-checkpoint.v0":
                count += 1
    return count


def freeze_resume(*, prior_key_final_usage_usd: float) -> dict[str, Any]:
    if RESUME_MANIFEST.exists():
        raise FileExistsError(f"resume manifest already exists: {RESUME_MANIFEST}")
    parent_hash = PARENT_MANIFEST.with_suffix(".sha256").read_text().split()[0]
    if _sha256(PARENT_MANIFEST) != parent_hash:
        raise RuntimeError("parent manifest hash mismatch")
    parent = json.loads(PARENT_MANIFEST.read_text(encoding="utf-8"))
    execution_roots = [OUTPUT_ROOT / name for name in EXPECTED_EXECUTIONS]
    if any(not path.is_dir() for path in execution_roots):
        raise RuntimeError("an expected interrupted execution root is missing")
    if [_matched_checkpoint_count(path) for path in execution_roots] != [0, 1]:
        raise RuntimeError("interrupted checkpoint counts changed")
    if any(
        any(path.glob("matched-branch-outcomes*.jsonl")) for path in execution_roots
    ):
        raise RuntimeError("an interrupted execution unexpectedly has outcome rows")
    if list(Daytona().list()):
        raise RuntimeError("Daytona must be empty before freezing resume state")

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("a replacement OPENROUTER_API_KEY is required")
    current_key = query_openrouter_key(api_key)
    current_usage = float(current_key["usage"])
    current_limit = float(current_key["limit"])
    original_usage = float(parent["budget"]["usage_before_usd"])
    prior_spend = prior_key_final_usage_usd - original_usage
    total_cap = float(parent["budget"]["additional_openrouter_cap_usd"])
    if not 0 <= prior_spend < total_cap:
        raise ValueError("prior key spend is outside the original total cap")
    remaining_cap = total_cap - prior_spend

    manifest = json.loads(PARENT_MANIFEST.read_text(encoding="utf-8"))
    manifest["schema_version"] = "matched-stuck-intervention-pilot.v2"
    manifest["frozen_at"] = datetime.now(UTC).isoformat()
    manifest["retry_policy"]["infrastructure_failure"] = (
        "Retry once only for explicit Daytona, sandbox, environment, container, "
        "Docker, build, verifier, missing-result, or outer-timeout failures. "
        "Infrastructure classification takes precedence over generic connection text."
    )
    manifest["retry_policy"]["terminal_invalid_group"] = (
        "After an invalid branch remains invalid, do not run later siblings because "
        "the matched group cannot enter analysis; never rerun an already-valid sibling."
    )
    manifest["integrity"]["code_sha256"][str(RUNNER.relative_to(ROOT))] = _sha256(
        RUNNER
    )
    manifest["budget"].update(
        {
            "usage_before_usd": current_usage,
            "usage_ceiling_usd": current_usage + remaining_cap,
            "dedicated_key_hard_limit_usd": current_limit,
            "effective_current_key_ceiling_usd": min(
                current_usage + remaining_cap, current_limit
            ),
            "prior_key_final_usage_usd": prior_key_final_usage_usd,
            "prior_key_spend_usd": prior_spend,
            "remaining_total_pilot_cap_usd": remaining_cap,
        }
    )
    manifest["resume"] = {
        "parent_manifest_path": str(PARENT_MANIFEST.relative_to(ROOT)),
        "parent_manifest_sha256": parent_hash,
        "prior_execution_roots": [str(path.relative_to(ROOT)) for path in execution_roots],
        "prior_execution_tree_sha256": {
            str(path.relative_to(ROOT)): _tree_sha256(path) for path in execution_roots
        },
        "completed_schedule_items": list(COMPLETED_SCHEDULE_ITEMS),
        "prior_group_counts": {"suspected_stuck": 0, "healthy": 0},
        "prior_accepted_outcome_count": 0,
        "prior_ineligible": [
            {
                "schedule_item": COMPLETED_SCHEDULE_ITEMS[0],
                "reason": "Flash output-length failure before a stuck checkpoint",
            },
            {
                "schedule_item": COMPLETED_SCHEDULE_ITEMS[1],
                "reason": (
                    "Healthy group abandoned after the Flash clean-restart arm had an "
                    "output-length failure; later diagnostic siblings are not counted"
                ),
            },
        ],
        "prior_key_usage_source": (
            "OpenRouter API-key table exact usage percentage multiplied by the exact "
            "key limit after the key expired"
        ),
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
    RESUME_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    RESUME_MANIFEST.with_suffix(".sha256").write_text(
        f"{_sha256(RESUME_MANIFEST)}  {RESUME_MANIFEST.name}\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-key-final-usage-usd", type=float, required=True)
    args = parser.parse_args()
    manifest = freeze_resume(
        prior_key_final_usage_usd=args.prior_key_final_usage_usd
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
