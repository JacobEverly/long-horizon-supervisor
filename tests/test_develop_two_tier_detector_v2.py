from horizon_supervisor.stuck_detector_v2 import FROZEN_CANDIDATE_FAMILY
from horizon_supervisor.training.develop_stuck_detector_v1 import (
    FLASH_ROUTE,
    QWEN_ROUTE,
    Trajectory,
)
from horizon_supervisor.training.develop_two_tier_detector_v2 import (
    TierCheckpoint,
    _task_folds,
    clustered_interval,
    select_projected_tiers,
)


def trajectory(
    *,
    task_id: str = "task-1",
    route_id: str = FLASH_ROUTE,
    completed: int = 0,
    error_turns: tuple[int, ...] = (5, 6),
) -> Trajectory:
    rows = tuple(
        {
            "observation": {
                "turn_index": turn,
                "error_signal_count": int(turn in error_turns),
                "pass_signal_count": 0,
            }
        }
        for turn in range(1, 13)
    )
    return Trajectory(
        trajectory_id=f"{task_id}-{route_id}",
        task_id=task_id,
        route_id=route_id,
        completed=completed,
        status="verified",
        rows=rows,
    )


def test_projection_banks_healthy_then_review_then_confirmation() -> None:
    selected = select_projected_tiers(
        [trajectory()], FROZEN_CANDIDATE_FAMILY[0]
    )
    assert [item.turn for item in selected["healthy"]] == [4]
    assert [item.turn for item in selected["needs_review"]] == [5]
    assert [item.turn for item in selected["confirmed_stuck"]] == [6]
    assert selected["confirmed_stuck"][0].remaining_turns == 6


def test_projection_never_confirms_without_prior_review() -> None:
    selected = select_projected_tiers(
        [trajectory(error_turns=(10,))], FROZEN_CANDIDATE_FAMILY[0]
    )
    assert len(selected["needs_review"]) == 1
    assert selected["confirmed_stuck"] == []


def test_structural_trajectories_are_excluded_from_recovery_projection() -> None:
    value = trajectory()
    structural = Trajectory(
        trajectory_id=value.trajectory_id,
        task_id=value.task_id,
        route_id=value.route_id,
        completed=0,
        status="agent_protocol_failure",
        rows=value.rows,
    )
    selected = select_projected_tiers([structural], FROZEN_CANDIDATE_FAMILY[0])
    assert all(not items for items in selected.values())


def test_task_folds_never_split_one_task() -> None:
    task_ids = [f"task-{index}" for index in range(15)]
    folds = _task_folds([*task_ids, *task_ids])
    assert len(folds) == 15
    assert set(folds.values()) == set(range(5))


def test_task_clustered_interval_is_positive_for_separated_groups() -> None:
    healthy = [
        TierCheckpoint("healthy", f"task-{index}", FLASH_ROUTE, str(index), 4, 8, 1)
        for index in range(8)
    ]
    confirmed = [
        TierCheckpoint(
            "confirmed_stuck",
            f"task-{index}",
            QWEN_ROUTE,
            str(index),
            6,
            6,
            0,
        )
        for index in range(8)
    ]
    lower, upper = clustered_interval(healthy, confirmed, samples=500)
    assert lower == 1.0
    assert upper == 1.0
