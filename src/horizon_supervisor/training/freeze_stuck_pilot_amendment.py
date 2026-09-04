from __future__ import annotations

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
PARENT_MANIFEST = OUTPUT_ROOT / "frozen-pilot-manifest-v0.json"
AMENDED_MANIFEST = OUTPUT_ROOT / "frozen-pilot-manifest-v1.json"
RUNNER = ROOT / "src/horizon_supervisor/training/run_stuck_pilot.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint_count(execution_root: Path) -> int:
    count = 0
    for path in (execution_root / "records").glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if json.loads(line).get("schema_version") == "matched-checkpoint.v0":
                count += 1
    return count


def freeze_amendment(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    output_root = output_root.resolve()
    parent_path = output_root / PARENT_MANIFEST.name
    amended_path = output_root / AMENDED_MANIFEST.name
    if amended_path.exists():
        raise FileExistsError(f"amended pilot is already frozen: {amended_path}")
    parent_sidecar = parent_path.with_suffix(".sha256")
    parent_hash = parent_sidecar.read_text(encoding="utf-8").split()[0]
    if _sha256(parent_path) != parent_hash:
        raise RuntimeError("parent pilot manifest hash mismatch")
    executions = sorted(output_root.glob("execution-*"))
    if len(executions) != 1:
        raise RuntimeError("amendment expects exactly one aborted pre-branch execution")
    aborted = executions[0]
    if _checkpoint_count(aborted) != 0:
        raise RuntimeError("cannot make retry amendment after a checkpoint was captured")
    if list(aborted.glob("**/matched-branch-outcomes*.jsonl")):
        raise RuntimeError("cannot make retry amendment after branch outcomes exist")
    if list(Daytona().list()):
        raise RuntimeError("Daytona must be empty before freezing the amendment")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required")
    key_info = query_openrouter_key(api_key)
    usage = float(key_info["usage"])

    manifest = json.loads(parent_path.read_text(encoding="utf-8"))
    parent_usage = float(manifest["budget"]["usage_before_usd"])
    manifest["schema_version"] = "matched-stuck-intervention-pilot.v1"
    manifest["frozen_at"] = datetime.now(UTC).isoformat()
    manifest["retry_policy"]["infrastructure_failure"] = (
        "Retry once only for explicit Daytona, sandbox, environment, container, "
        "Docker, build, verifier, missing-result, or outer-timeout failures."
    )
    manifest["retry_policy"]["model_or_provider_failure"] = (
        "Do not retry output-length, provider, or other model/protocol failures; "
        "record them structurally and continue the frozen schedule."
    )
    manifest["integrity"]["code_sha256"][str(RUNNER.relative_to(ROOT))] = _sha256(
        RUNNER
    )
    manifest["execution_amendment"] = {
        "kind": "narrow_infrastructure_retry_classification_fix",
        "parent_manifest_path": str(parent_path.relative_to(ROOT)),
        "parent_manifest_sha256": parent_hash,
        "aborted_execution_path": str(aborted.relative_to(ROOT)),
        "aborted_execution_checkpoint_count": 0,
        "aborted_execution_matched_outcome_count": 0,
        "observed_issue": (
            "The runner retried an OutputLengthExceededError as though it were an "
            "infrastructure failure."
        ),
        "changed_fields": [
            "retry classification",
            "runner code hash",
            "default amended-manifest path",
        ],
        "unchanged_fields": [
            "detector",
            "task roster and order",
            "model roster",
            "agent limits",
            "branch arms",
            "analysis",
            "stopping rules",
            "total additional spend ceiling",
        ],
        "usage_at_amendment_usd": usage,
        "aborted_prebranch_spend_usd": max(0.0, usage - parent_usage),
        "aborted_calls_count_toward_total_pilot_spend": True,
    }
    amended_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    amended_path.with_suffix(".sha256").write_text(
        f"{_sha256(amended_path)}  {amended_path.name}\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    manifest = freeze_amendment()
    print(
        json.dumps(
            {
                "schema_version": manifest["schema_version"],
                "amendment": manifest["execution_amendment"]["kind"],
                "aborted_prebranch_spend_usd": manifest["execution_amendment"][
                    "aborted_prebranch_spend_usd"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
