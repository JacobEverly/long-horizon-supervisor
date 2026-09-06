from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = (
    ROOT
    / "artifacts/official/two-tier-continuation-calibration-v4/public-summary-v4.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v4_public_summary_is_credential_free_and_integrity_linked() -> None:
    summary = json.loads(PUBLIC.read_text(encoding="utf-8"))
    assert summary["gate_passed"] is False
    assert summary["gates"] == {
        "behavioral_and_coverage_gates_passed": True,
        "all_counted_snapshots_rehydrated": False,
        "structural_failures_separate": True,
        "leakage_controls": True,
    }
    assert summary["scope"]["fresh_v4_checkpoint_replays"] == 65
    assert summary["scope"]["fresh_v4_checkpoint_replays_passed"] == 63
    assert summary["scope"]["fresh_v4_protocol_valid_trajectories"] == 39
    assert (
        summary["scope"]["fresh_v4_learning_valid_nonstructural_trajectories"]
        == 37
    )
    assert summary["fidelity_failure"]["failed_checkpoint_replays"] == 2
    assert summary["spend"]["exact_incremental_openrouter_usd"] == 2.532378529
    assert summary["cleanup"]["remaining_daytona_environments"] == 0

    integrity = summary["integrity"]
    root = PUBLIC.parent
    assert _sha256(root / "frozen-manifest-v4.json") == integrity[
        "frozen_manifest_sha256"
    ]
    assert _sha256(root / "calibration-report-v4.json") == integrity[
        "calibration_report_sha256"
    ]
    assert _sha256(root / "snapshot-fidelity-v4.json") == integrity[
        "snapshot_fidelity_report_sha256"
    ]
    assert _sha256(root / "execution-ledger-v4.json") == integrity[
        "execution_ledger_sha256"
    ]

    raw = PUBLIC.read_text(encoding="utf-8")
    forbidden = (
        "OPENROUTER_API_KEY",
        "DAYTONA_API_KEY",
        "dtn_",
        "/Users/",
        "terminal_tail",
        "analysis_text",
    )
    assert not any(value in raw for value in forbidden)
