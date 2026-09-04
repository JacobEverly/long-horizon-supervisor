from horizon_supervisor.benchmark.gate6 import ARMS, analyze_gate6


def _run(arm: str, success: bool) -> dict:
    return {
        "arm": arm,
        "success": success,
        "cost_usd": 0.1,
        "workspace_change_count": int(success),
        "last_public_test_result": {"fresh_in_current_run": success},
        "guard_events": [] if success else [{"reason": "unverified_stop"}],
    }


def test_gate6_analysis_reports_arm_behavior_and_contrasts() -> None:
    runs = []
    for arm in ARMS:
        runs.extend([_run(arm, arm == "digest_aware_clean"), _run(arm, False)])

    result = analyze_gate6(runs)

    assert result["arms"]["digest_aware_clean"]["passed"] == 1
    assert result["arms"]["digest_aware_clean"]["runs_with_mutation"] == 1
    assert result["arms"]["neutral_clean"]["runs_with_unverified_stop"] == 2
    assert result["contrasts_percentage_points"]["digest_label_vs_stale"] == 50.0


def test_gate6_analysis_accepts_missing_public_test_result() -> None:
    runs = [_run(arm, False) for arm in ARMS]
    runs[0]["last_public_test_result"] = None

    result = analyze_gate6(runs)

    assert result["arms"]["neutral_clean"]["runs_with_fresh_tests"] == 0
