"""Run frozen v7 continuation calibration with a bounded child process."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from horizon_supervisor.training import run_continuation_calibration_v6 as v6
from horizon_supervisor.training.freeze_continuation_calibration import (
    EXACT_MODELS,
    ROOT,
    _sha256,
    _tree_sha256,
)
from horizon_supervisor.training.freeze_continuation_calibration_v7 import (
    OUTPUT_ROOT,
    V6_MANIFEST_SHA256,
    V6_SMOKE_SHA256,
)

_ORIGINAL_RUN_COMMAND = v6._run_command

MANIFEST = OUTPUT_ROOT / "frozen-manifest-v7.json"
STATE_PATH = OUTPUT_ROOT / "execution-state-v7.json"
OUTCOMES_PATH = OUTPUT_ROOT / "natural-continuation-outcomes-v7.jsonl"
USAGE_PATH = OUTPUT_ROOT / "trial-usage-ledger-v7.jsonl"
CHECKPOINT_INDEX = OUTPUT_ROOT / "checkpoint-bank-index-v7.jsonl"
FIDELITY_REPORT = OUTPUT_ROOT / "snapshot-fidelity-v7.json"
LEDGER_PATH = OUTPUT_ROOT / "execution-ledger-v7.json"
REPORT_PATH = OUTPUT_ROOT / "calibration-report-v7.json"
SMOKE_REPORT = OUTPUT_ROOT / "permission-transport-smoke-v7.json"


def validate_manifest(path: Path = MANIFEST) -> tuple[dict[str, Any], str]:
    manifest = json.loads(path.read_text())
    sidecar = path.with_suffix(".sha256")
    expected = sidecar.read_text().split()[0]
    if _sha256(path) != expected:
        raise RuntimeError("v7 manifest hash mismatch")
    if manifest.get("schema_version") != "two-tier-continuation-calibration-manifest.v7":
        raise RuntimeError("unexpected v7 schema")
    if manifest.get("frozen_before_model_outcomes") is not True:
        raise RuntimeError("v7 was not frozen before outcomes")
    lineage = manifest["lineage"]
    if (
        lineage.get("v6_manifest_sha256") != V6_MANIFEST_SHA256
        or lineage.get("v6_transport_smoke_sha256") != V6_SMOKE_SHA256
    ):
        raise RuntimeError("v7 lineage changed")
    if manifest["models"]["routes"] != EXACT_MODELS:
        raise RuntimeError("v7 route mapping changed")
    if manifest["detector"]["config"] != v6.SELECTED_CONFIG.model_dump():
        raise RuntimeError("v7 detector changed")
    execution = manifest["execution"]
    if (
        execution["agent"] != "HarnessFilteredContinuationTerminus2"
        or execution["max_turns"] != 12
        or execution["natural_continuation_only"] is not True
        or execution["interventions_forbidden"] is not True
        or execution["child_timeout_seconds"] != 420
    ):
        raise RuntimeError("v7 execution contract changed")
    tasks = manifest["task_selection"]["ordered_pool"]
    if len(tasks) != 16 or [task["position"] for task in tasks] != list(range(1, 17)):
        raise RuntimeError("v7 task pool changed")
    if any(
        task["prior_terminal_outcome_count"] != 0
        or task["static_checkpoint_compatible"] is not True
        for task in tasks
    ):
        raise RuntimeError("v7 task eligibility changed")
    for task in tasks:
        root = ROOT / task["task_root"]
        if not root.is_dir() or _tree_sha256(root) != task["task_tree_sha256"]:
            raise RuntimeError(f"v7 task tree changed: {task['task_id']}")
    return manifest, expected


def validate_smoke(manifest: dict[str, Any]) -> tuple[dict[str, Any], str]:
    path = ROOT / manifest["execution"]["transport_smoke_path"]
    smoke = json.loads(path.read_text())
    if (
        smoke.get("schema_version") != "permission-transport-smoke.v7"
        or smoke.get("passed") is not True
        or smoke.get("provider_model_calls") != 0
        or smoke.get("remaining_daytona_environments") != 0
    ):
        raise RuntimeError("v7 transport smoke did not pass")
    return smoke, _sha256(path)


def _bounded_run_command(
    command: list[str], *, environment: dict[str, str], timeout: float
) -> tuple[int, str, bool]:
    return _ORIGINAL_RUN_COMMAND(
        command, environment=environment, timeout=min(float(timeout), 420.0)
    )


def _new_state(
    manifest_hash: str, smoke_hash: str, usage: float, hard_limit: float
) -> dict[str, Any]:
    return {
        "schema_version": "continuation-calibration-execution-state.v7",
        "manifest_sha256": manifest_hash,
        "transport_smoke_sha256": smoke_hash,
        "status": "in_progress",
        "started_at": v6.datetime.now(v6.UTC).isoformat(),
        "provider_usage_baseline_usd": usage,
        "provider_hard_limit_usd": hard_limit,
        "completed_schedule_items": [],
        "outcomes": [],
        "attempts": [],
        "fidelity_rows": [],
        "tranche_reports": [],
        "pending_trial": None,
    }


def _report(state: dict[str, Any], *, tranche: int | None = None) -> dict[str, Any]:
    report = v6.analyze(state["outcomes"])
    report["schema_version"] = "two-tier-continuation-calibration-report.v7"
    report["cohort"] = "fresh_v7_only"
    report["new_v7_trajectory_count"] = len(state["outcomes"])
    report["prior_outcome_count_used"] = 0
    if tranche is not None:
        report["tranche"] = tranche
    return report


def run() -> dict[str, Any]:
    # Reuse the battle-tested v6 state machine only after replacing its frozen
    # paths, validator, report labels, and child command timeout.
    v6.OUTPUT_ROOT = OUTPUT_ROOT
    v6.MANIFEST = MANIFEST
    v6.STATE_PATH = STATE_PATH
    v6.OUTCOMES_PATH = OUTCOMES_PATH
    v6.USAGE_PATH = USAGE_PATH
    v6.CHECKPOINT_INDEX = CHECKPOINT_INDEX
    v6.FIDELITY_REPORT = FIDELITY_REPORT
    v6.LEDGER_PATH = LEDGER_PATH
    v6.REPORT_PATH = REPORT_PATH
    v6.SMOKE_REPORT = SMOKE_REPORT
    v6.validate_manifest = validate_manifest
    v6.validate_transport_smoke = validate_smoke
    v6._run_command = _bounded_run_command
    v6._new_state = _new_state
    v6._report = _report
    return v6.run(MANIFEST)


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
