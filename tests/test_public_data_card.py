from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "docs" / "data"


def test_v7_public_summary_matches_sealed_boundary() -> None:
    path = DATA_ROOT / "continuation-calibration-v7-summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))

    assert summary["status"] == "interrupted"
    assert summary["completed_schedule_items"] == 1
    assert summary["learning_valid_trajectories"] == 0
    assert summary["structural_trajectories"] == 1
    assert summary["counted_checkpoints"] == 0
    assert summary["recorded_attempts"] == 2
    assert summary["completed_verifier_outcomes"] == 0
    assert summary["interrupted_attempts_scored"] == 0
    assert summary["incremental_provider_spend_usd"] == 0.02814098
    assert summary["remaining_daytona_environments"] == 0
    assert summary["gate_passed"] is False

    raw = path.read_text(encoding="utf-8")
    forbidden = (
        "OPENROUTER_API_KEY",
        "DAYTONA_API_KEY",
        "dtn_",
        "/Users/",
        "terminal_tail",
        "reasoning_content",
    )
    assert not any(value in raw for value in forbidden)


def test_data_card_states_release_limits() -> None:
    card = (DATA_ROOT / "README.md").read_text(encoding="utf-8")

    for heading in (
        "## Summary",
        "## Intended use",
        "## Collection and validation",
        "## Limitations",
        "## Licensing and privacy",
        "## Versioning",
    ):
        assert heading in card
    assert "The current release contains summaries, not reusable raw trajectories" in card
    assert "license: other" in card
