from horizon_supervisor.benchmark.continuation_harbor import checkpoint_kind
from horizon_supervisor.stuck_detector_v2 import TwoTierStatus


def test_checkpoint_kind_captures_each_tier_once() -> None:
    captured: set[str] = set()

    assert (
        checkpoint_kind(
            status=TwoTierStatus.HEALTHY,
            turn=4,
            healthy_turn=4,
            captured=captured,
        )
        == "healthy"
    )
    captured.add("healthy")
    assert (
        checkpoint_kind(
            status=TwoTierStatus.HEALTHY,
            turn=4,
            healthy_turn=4,
            captured=captured,
        )
        is None
    )
    assert (
        checkpoint_kind(
            status=TwoTierStatus.NEEDS_REVIEW,
            turn=5,
            healthy_turn=4,
            captured=captured,
        )
        == "needs_review"
    )
    captured.add("needs_review")
    assert (
        checkpoint_kind(
            status=TwoTierStatus.CONFIRMED_STUCK,
            turn=6,
            healthy_turn=4,
            captured=captured,
        )
        == "confirmed_stuck"
    )


def test_checkpoint_kind_never_banks_structural_failure() -> None:
    assert (
        checkpoint_kind(
            status=TwoTierStatus.STRUCTURAL_FAILURE,
            turn=6,
            healthy_turn=4,
            captured=set(),
        )
        is None
    )


def test_healthy_control_is_only_at_the_frozen_turn() -> None:
    assert (
        checkpoint_kind(
            status=TwoTierStatus.HEALTHY,
            turn=5,
            healthy_turn=4,
            captured=set(),
        )
        is None
    )
