import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from horizon_supervisor.stuck_detector import TurnObservation
from horizon_supervisor.stuck_detector_v1 import ActionMode
from horizon_supervisor.stuck_detector_v2 import (
    FROZEN_CANDIDATE_FAMILY,
    TwoTierObservation,
    TwoTierStatus,
    TwoTierStuckDetectorV2,
)

ROOT = Path(__file__).parents[1]
V0_SHA256 = "c3319c93d823455076fd294ac16e28748a2b2ebcab10e1b81760d174088f4ffe"
V1_SHA256 = "220f5faade081cd22d283ee68c23647b37e843458afb72dec3a2833ab8b0475c"


def observation(index: int, **updates: object) -> TwoTierObservation:
    values = {
        "run_id": "run-1",
        "turn": index,
        "max_turns": 12,
        "model_id": "test/model",
        "commands": (f"python build-{index}.py",),
        "terminal_tail": "work continues",
        "workspace_digest": "unchanged",
        "output_token_budget": 49_152,
        "spend_budget_usd": 1.0,
    }
    values.update(updates)
    return TwoTierObservation(**values)


def detector(candidate_index: int = 0) -> TwoTierStuckDetectorV2:
    return TwoTierStuckDetectorV2(FROZEN_CANDIDATE_FAMILY[candidate_index])


def test_v0_and_v1_sources_remain_frozen() -> None:
    assert (
        hashlib.sha256(
            (ROOT / "src/horizon_supervisor/stuck_detector.py").read_bytes()
        ).hexdigest()
        == V0_SHA256
    )
    assert (
        hashlib.sha256(
            (ROOT / "src/horizon_supervisor/stuck_detector_v1.py").read_bytes()
        ).hexdigest()
        == V1_SHA256
    )


def test_investigation_may_need_review_but_is_not_confirmed_stuck() -> None:
    instance = detector()
    statuses = []
    for index in range(1, 10):
        result = instance.observe(
            observation(index, commands=("ls -la", "find /app -type f"))
        )
        statuses.append(result.status)
        assert result.action_mode == ActionMode.INSPECTION
    assert TwoTierStatus.NEEDS_REVIEW in statuses
    assert TwoTierStatus.CONFIRMED_STUCK not in statuses


def test_review_precedes_confirmation_of_repeated_productive_failure() -> None:
    instance = detector()
    for index in range(1, 4):
        assert instance.observe(observation(index)).status == TwoTierStatus.HEALTHY
    fourth = instance.observe(
        observation(4, commands=("pytest -q",), terminal_tail="ERROR same failure")
    )
    fifth = instance.observe(
        observation(5, commands=("pytest -q",), terminal_tail="ERROR same failure")
    )
    sixth = instance.observe(
        observation(6, commands=("pytest -q",), terminal_tail="ERROR same failure")
    )
    assert fourth.status == TwoTierStatus.HEALTHY
    assert fifth.status == TwoTierStatus.NEEDS_REVIEW
    assert sixth.status == TwoTierStatus.CONFIRMED_STUCK


@pytest.mark.parametrize(
    "update",
    [
        {"protocol_failure": True},
        {"provider_failure": True},
        {"harness_failure": True},
        {"snapshot_reproducible": False},
        {"external_state_reproducible": False},
    ],
)
def test_structural_failures_never_become_stuck(update: dict[str, object]) -> None:
    result = detector().observe(observation(1, **update))
    assert result.status == TwoTierStatus.STRUCTURAL_FAILURE


def test_remaining_turn_guard_prevents_unusable_checkpoint() -> None:
    instance = detector()
    for index in range(1, 11):
        result = instance.observe(
            observation(index, commands=("pytest -q",), terminal_tail="ERROR same")
        )
    assert result.remaining_turns == 2
    eleventh = instance.observe(
        observation(11, commands=("pytest -q",), terminal_tail="ERROR same")
    )
    assert eleventh.remaining_turns == 1
    assert eleventh.status == TwoTierStatus.HEALTHY


def test_v0_observation_converts_without_future_fields() -> None:
    old = TurnObservation(
        run_id="old",
        turn=1,
        max_turns=12,
        model_id="test/model",
        workspace_digest="digest",
        output_token_budget=100,
        spend_budget_usd=1,
    )
    converted = TwoTierObservation.from_v0(old)
    assert converted.schema_version == "stuck-turn-observation.v2"
    assert converted.snapshot_reproducible is True

    payload = converted.model_dump()
    payload["final_success"] = True
    with pytest.raises(ValidationError, match="extra_forbidden"):
        TwoTierObservation.model_validate(payload)


def test_only_frozen_candidate_configs_are_accepted() -> None:
    config = FROZEN_CANDIDATE_FAMILY[0].model_copy(update={"name": "unfrozen"})
    with pytest.raises(ValueError, match="frozen v2 candidate family"):
        TwoTierStuckDetectorV2(config)
