import json
from pathlib import Path

import pytest

from horizon_supervisor.training.develop_stuck_detector_v1 import (
    DEFAULT_CHECKPOINTS,
    DEFAULT_MANIFEST,
    DEFAULT_OUTCOMES,
    DEFAULT_STATE,
    V0_EXPECTED_SHA256,
    build_report,
    load_development_trajectories,
    replay_detector_yield,
    sha256_file,
)

ROOT = Path(__file__).parents[1]


def test_development_inputs_are_task_grouped_and_flash_qwen_only() -> None:
    trajectories = load_development_trajectories(ROOT / DEFAULT_CHECKPOINTS)
    assert len(trajectories) == 70
    assert len({item.task_id for item in trajectories}) == 35
    assert {item.route_id for item in trajectories} == {
        "gate7/fixed-flash",
        "gate7/fixed-qwen",
    }


def test_loader_rejects_non_development_rows(tmp_path: Path) -> None:
    first = json.loads((ROOT / DEFAULT_CHECKPOINTS).read_text().splitlines()[0])
    first["record_split"] = "held_out"
    path = tmp_path / "held-out.jsonl"
    path.write_text(json.dumps(first) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="development rows only"):
        load_development_trajectories(path)


def test_same_scout_replay_shows_v1_does_not_improve_yield() -> None:
    replay = replay_detector_yield(ROOT / DEFAULT_STATE)
    assert replay["planned_schedule_items"] == 84
    assert replay["v0_checkpoint_hits"] == 33
    assert replay["v1_checkpoint_hits"] == 22
    assert replay["improves_over_v0"] is False


def test_offline_gate_stops_paid_collection() -> None:
    report = build_report(
        ROOT / DEFAULT_CHECKPOINTS,
        ROOT / DEFAULT_STATE,
        ROOT / DEFAULT_MANIFEST,
        ROOT / DEFAULT_OUTCOMES,
        ROOT / "src/horizon_supervisor/stuck_detector.py",
        ROOT / "src/horizon_supervisor/stuck_detector_v1.py",
    )
    assert report["decision"] == "NOT READY — detector development gate failed"
    assert report["gate_passed"] is False
    assert report["offline_gate"]["improves_checkpoint_yield_over_v0"]["passed"] is False
    assert report["next_phase"]["paid_checkpoint_collection"] == "not_run"
    assert sha256_file(ROOT / "src/horizon_supervisor/stuck_detector.py") == V0_EXPECTED_SHA256
