from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_SCORECARD = ROOT / "docs" / "data" / "heldout-scorecard-summary-v0.json"
SOURCE_SCORECARD = (
    ROOT
    / "artifacts"
    / "official"
    / "gate8-wave3-18-task-checkpoint"
    / "frozen-policy-scorecard-18-task-v0.json"
)


def test_portfolio_headline_is_traceable_to_frozen_scorecard() -> None:
    scorecard = json.loads(PUBLIC_SCORECARD.read_text(encoding="utf-8"))
    case_study = (ROOT / "CASE_STUDY.md").read_text(encoding="utf-8")
    normalized_case_study = " ".join(case_study.split())

    assert scorecard["source"]["evaluation_role"] == "final held-out Wave 3 evaluation"
    assert scorecard["evaluation"]["tasks"] == 18
    assert scorecard["evaluation"]["learning_valid_outcomes"] == 72
    assert scorecard["evaluation"]["held_out_policy_tuning"] is False

    kimi = next(
        row for row in scorecard["policies"] if row["strategy"] == "always:gate7/fixed-kimi"
    )
    cascade = next(
        row
        for row in scorecard["policies"]
        if row["strategy"] == "frozen-cascade:flash>qwen>glm>kimi"
    )
    two_model = next(
        row
        for row in scorecard["policies"]
        if row["strategy"] == "frozen-cascade:flash>qwen"
    )

    assert (kimi["successes"], cascade["successes"], two_model["successes"]) == (7, 12, 9)
    assert round(100 * (cascade["success_rate"] - kimi["success_rate"]), 1) == 27.8
    assert round(100 * (1 - two_model["replayed_cost_usd"] / kimi["replayed_cost_usd"])) == 72
    assert round(scorecard["spend"]["exact_incremental_provider_spend_usd"], 4) == 4.1006

    for claim in (
        "38.9% → 66.7% verified completion",
        "+27.8 percentage points",
        "72% lower replayed",
        "72 learning-valid outcomes",
        "$4.1006",
    ):
        assert claim in normalized_case_study


def test_public_scorecard_matches_local_frozen_artifact_when_available() -> None:
    if not SOURCE_SCORECARD.exists():
        return

    public = json.loads(PUBLIC_SCORECARD.read_text(encoding="utf-8"))
    source_bytes = SOURCE_SCORECARD.read_bytes()
    source = json.loads(source_bytes)

    assert hashlib.sha256(source_bytes).hexdigest() == public["source"]["sha256"]
    assert source["evaluation_role"] == public["source"]["evaluation_role"]
    assert source["data"]["tasks"] == public["evaluation"]["tasks"]
    assert source["data"]["records"] == public["evaluation"]["learning_valid_outcomes"]
    assert (
        source["spend_audit"]["exact_incremental_spend_usd"]
        == public["spend"]["exact_incremental_provider_spend_usd"]
    )


def test_portfolio_chart_is_accessible_and_linked() -> None:
    case_study = (ROOT / "CASE_STUDY.md").read_text(encoding="utf-8")
    chart = (ROOT / "docs/assets/heldout-completion-cost-frontier.svg").read_text(
        encoding="utf-8"
    )

    assert "![Completion-versus-cost frontier]" in case_study
    assert 'role="img"' in chart
    assert "<title" in chart
    assert "<desc" in chart


def test_case_study_relative_links_exist_in_public_checkout() -> None:
    source = ROOT / "CASE_STUDY.md"
    text = source.read_text(encoding="utf-8")
    pattern = r"\[[^]]+\]\(([^)]+)\)|!\[[^]]*\]\(([^)]+)\)"
    targets = [left or right for left, right in re.findall(pattern, text)]

    assert targets
    assert all("://" in target or (source.parent / target).is_file() for target in targets)
