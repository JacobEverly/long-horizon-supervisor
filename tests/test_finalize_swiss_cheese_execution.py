from __future__ import annotations

import pytest

from horizon_supervisor.training.finalize_swiss_cheese_execution import (
    build_execution_audit,
)


def _inputs() -> tuple[dict, dict, dict]:
    ledger = {
        "key_usage_before_usd": 27.653569393,
        "runs": [{"provider_spend_usd": 11.0}, {"provider_spend_usd": 0.5}],
    }
    matrix = {
        "record_count": 150,
        "task_count": 10,
        "route_count": 5,
        "replications_per_pair": 3,
        "all_pairs_present_once_and_learning_valid": True,
    }
    scorecard = {"spend_audit": {"exact_incremental_spend_usd": 12.078237391}}
    return ledger, matrix, scorecard


def test_builds_two_key_exact_usage_audit() -> None:
    ledger, matrix, scorecard = _inputs()
    audit = build_execution_audit(
        ledger=ledger,
        matrix_summary=matrix,
        scorecard=scorecard,
        prior_key_fingerprint="old",
        prior_key_final_usage_usd=39.694240884,
        active_key_fingerprint="new",
        active_key_initial_usage_usd=0.0,
        active_key_final_usage_usd=0.0375659,
        active_key_limit_usd=7.95,
        spend_ceiling_usd=47.653569393,
    )
    assert audit["exact_incremental_provider_spend_usd"] == pytest.approx(
        12.078237391
    )
    assert audit["under_frozen_ceiling"] is True
    assert audit["matrix"]["records"] == 150


def test_rejects_incomplete_matrix() -> None:
    ledger, matrix, scorecard = _inputs()
    matrix["record_count"] = 149
    with pytest.raises(ValueError, match="150 records"):
        build_execution_audit(
            ledger=ledger,
            matrix_summary=matrix,
            scorecard=scorecard,
            prior_key_fingerprint="old",
            prior_key_final_usage_usd=39.694240884,
            active_key_fingerprint="new",
            active_key_initial_usage_usd=0.0,
            active_key_final_usage_usd=0.0375659,
            active_key_limit_usd=7.95,
            spend_ceiling_usd=47.653569393,
        )
