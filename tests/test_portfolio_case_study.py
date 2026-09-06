from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCORECARD = (
    ROOT
    / "artifacts"
    / "official"
    / "gate8-wave3-18-task-checkpoint"
    / "frozen-policy-scorecard-18-task-v0.json"
)


def test_portfolio_headline_is_traceable_to_frozen_scorecard() -> None:
    scorecard = json.loads(SCORECARD.read_text(encoding="utf-8"))
    case_study = (ROOT / "CASE_STUDY.md").read_text(encoding="utf-8")
    normalized_case_study = " ".join(case_study.split())

    assert scorecard["evaluation_role"] == "final held-out Wave 3 evaluation"
    assert scorecard["data"]["tasks"] == 18
    assert scorecard["data"]["records"] == 72
    assert scorecard["policy_guard"]["held_out_policy_tuning"] is False

    kimi = next(
        row for row in scorecard["static_models"] if row["route_id"] == "gate7/fixed-kimi"
    )
    cascade = next(
        row
        for row in scorecard["frozen_cascades"]
        if row["strategy"] == "frozen-cascade:flash>qwen>glm>kimi"
    )
    two_model = next(
        row
        for row in scorecard["frozen_cascades"]
        if row["strategy"] == "frozen-cascade:flash>qwen"
    )

    assert (kimi["successes"], cascade["successes"], two_model["successes"]) == (7, 12, 9)
    assert round(100 * (cascade["success_rate"] - kimi["success_rate"]), 1) == 27.8
    assert round(100 * (1 - two_model["total_cost_usd"] / kimi["total_cost_usd"])) == 72
    assert round(scorecard["spend_audit"]["exact_incremental_spend_usd"], 4) == 4.1006

    for claim in (
        "38.9% → 66.7% verified completion",
        "+27.8 percentage points",
        "72% lower replayed",
        "72 learning-valid outcomes",
        "$4.1006",
    ):
        assert claim in normalized_case_study


def test_portfolio_chart_is_accessible_and_linked() -> None:
    case_study = (ROOT / "CASE_STUDY.md").read_text(encoding="utf-8")
    chart = (ROOT / "docs/assets/heldout-completion-cost-frontier.svg").read_text(
        encoding="utf-8"
    )

    assert "![Completion-versus-cost frontier]" in case_study
    assert 'role="img"' in chart
    assert "<title" in chart
    assert "<desc" in chart
