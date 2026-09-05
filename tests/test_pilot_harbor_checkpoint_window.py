from horizon_supervisor.benchmark.pilot_harbor import (
    _healthy_checkpoint_window_exhausted,
)


def test_healthy_window_stops_at_or_after_missed_frozen_turn() -> None:
    common = {
        "enabled": True,
        "capture_healthy": True,
        "healthy_captured": False,
        "healthy_turn": 4,
    }
    assert not _healthy_checkpoint_window_exhausted(
        **common, observation_turn=3
    )
    assert _healthy_checkpoint_window_exhausted(**common, observation_turn=4)
    assert _healthy_checkpoint_window_exhausted(**common, observation_turn=5)


def test_healthy_window_never_stops_disabled_or_completed_capture() -> None:
    assert not _healthy_checkpoint_window_exhausted(
        enabled=False,
        capture_healthy=True,
        healthy_captured=False,
        observation_turn=4,
        healthy_turn=4,
    )
    assert not _healthy_checkpoint_window_exhausted(
        enabled=True,
        capture_healthy=False,
        healthy_captured=False,
        observation_turn=12,
        healthy_turn=4,
    )
    assert not _healthy_checkpoint_window_exhausted(
        enabled=True,
        capture_healthy=True,
        healthy_captured=True,
        observation_turn=4,
        healthy_turn=4,
    )
