import hashlib
import json
from pathlib import Path

from horizon_supervisor.stuck_detector_v2 import FROZEN_CANDIDATE_FAMILY

ROOT = Path(__file__).parents[1]
CONTRACT = (
    ROOT
    / "artifacts/official/two-tier-detector-v2/development-contract-v0.json"
)


def test_frozen_contract_matches_candidate_family_and_sources() -> None:
    contract = json.loads(CONTRACT.read_text())
    assert contract["status"] == "frozen_before_real_data_scoring"
    assert contract["candidate_family"] == [
        candidate.model_dump() for candidate in FROZEN_CANDIDATE_FAMILY
    ]
    for key, relative_path in {
        "v0_source_sha256": "src/horizon_supervisor/stuck_detector.py",
        "v1_source_sha256": "src/horizon_supervisor/stuck_detector_v1.py",
        "v2_source_sha256": "src/horizon_supervisor/stuck_detector_v2.py",
        "evaluator_source_sha256": (
            "src/horizon_supervisor/training/develop_two_tier_detector_v2.py"
        ),
    }.items():
        actual = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
        assert contract["integrity"][key] == actual


def test_contract_freezes_every_required_gate_before_scoring() -> None:
    thresholds = json.loads(CONTRACT.read_text())["gate_thresholds"]
    assert thresholds["healthy_minus_confirmed_recovery_difference"] == 0.2
    assert thresholds["needs_review_replay_checkpoints"] == 12
    assert thresholds["needs_review_replay_tasks"] == 8
    assert thresholds["confirmed_replay_checkpoints"] == 6
    assert thresholds["confirmed_replay_tasks"] == 4
    assert thresholds["healthy_replay_checkpoints"] == 12
    assert thresholds["minimum_remaining_turns"] == 2
