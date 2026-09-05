import json
from pathlib import Path
from types import SimpleNamespace

from horizon_supervisor.training.freeze_continuation_calibration import EXACT_MODELS
from horizon_supervisor.training.run_continuation_calibration_v1 import (
    _continuation_command,
    _structural_runtime_failure,
    _v1_outcome_row,
)


def test_v1_command_uses_process_delta_agent_and_natural_continuation(
    tmp_path: Path,
) -> None:
    task_root = tmp_path / "tasks" / "sample-task"
    task_root.mkdir(parents=True)
    (task_root / "task.toml").write_text("version = '1.0'\n", encoding="utf-8")
    task = {
        "task_id": "sample-task",
        "task_category": "debugging",
        "task_root": str(task_root),
    }
    command = _continuation_command(
        manifest={"execution": {"max_turns": 12}},
        server=SimpleNamespace(base_url="http://127.0.0.1:9999"),
        run_root=tmp_path,
        task=task,
        route_id="gate7/fixed-flash",
        model_id=EXACT_MODELS["gate7/fixed-flash"],
        job_name="test-job",
        record_path=tmp_path / "record.jsonl",
        provider_usage_start=0.0,
        provider_usage_ceiling=2.5,
    )
    joined = "\n".join(command)

    assert "ProcessDeltaContinuationTerminus2" in command[
        command.index("--agent") + 1
    ]
    assert "continuation_detector_config=" in joined
    assert "pilot_stop_after_checkpoint=false" in joined
    assert "pilot_stop_after_healthy_window=false" in joined


def test_v1_manifest_preserves_exact_v2_detector_thresholds() -> None:
    v0 = json.loads(
        Path(
            "artifacts/official/two-tier-continuation-calibration-v0/"
            "frozen-manifest-v0.json"
        ).read_text(encoding="utf-8")
    )

    assert v0["detector"]["config"] == {
        "name": "review-t5-confirm-t6-w2-e2",
        "review_minimum_turn": 5,
        "review_signal_threshold": 2,
        "confirmation_minimum_turn": 6,
        "confirmation_window": 2,
        "confirmation_failure_turns": 2,
        "confirmation_productive_turns": 1,
        "minimum_remaining_turns": 2,
    }


def test_provider_failure_is_structural_and_discards_checkpoint(
    tmp_path: Path,
) -> None:
    record = tmp_path / "record.jsonl"
    event = {
        "schema_version": "two-tier-observation-event.v0",
        "observation": {
            "schema_version": "stuck-turn-observation.v2",
            "run_id": "run",
            "turn": 4,
            "max_turns": 12,
            "model_id": "model",
            "commands": [],
            "terminal_tail": "",
            "workspace_digest": "digest",
            "public_tests": None,
            "successful_milestones": [],
            "required_artifacts": [],
            "protocol_failure": False,
            "provider_failure": False,
            "harness_failure": False,
            "actionable_next_step": True,
            "input_tokens": 1,
            "output_tokens": 1,
            "cached_tokens": 0,
            "reasoning_tokens": 0,
            "output_token_budget": 100,
            "spent_usd": 0.01,
            "spend_budget_usd": 0.5,
            "remaining_wall_seconds": None,
            "task_category": "debugging",
            "snapshot_reproducible": True,
            "external_state_reproducible": True,
        },
        "assessment": {"status": "HEALTHY"},
    }
    checkpoint = {
        "schema_version": "matched-checkpoint.v0",
        "run_id": "run",
        "checkpoint_kind": "healthy",
        "base_model_id": "model",
        "observation": {
            "turn": 4,
            "max_turns": 12,
            "workspace_digest": "digest",
        },
        "state_transfer_eligible": True,
        "state_transfer_ineligibility_reason": None,
        "anchor_workspace_path": "/private/anchor",
    }
    record.write_text(
        json.dumps(event) + "\n" + json.dumps(checkpoint) + "\n",
        encoding="utf-8",
    )
    trial = {
        "record_path": str(record),
        "valid": True,
        "result": {
            "verifier_result": {"rewards": {"reward": 0.0}},
            "exception_info": {
                "exception_type": "ProviderError",
                "exception_message": "upstream unavailable",
            },
            "started_at": "2026-09-05T00:00:00Z",
            "finished_at": "2026-09-05T00:00:01Z",
        },
        "stats": {"models": {}},
        "model_id": "model",
        "route_id": "gate7/fixed-flash",
        "provider_spend_usd": 0.01,
        "job_name": "run",
    }
    task = {
        "task_id": "task",
        "task_category": "debugging",
        "difficulty": "medium",
        "position": 1,
        "tranche": 1,
    }

    assert _structural_runtime_failure(trial, [event]) is True
    row = _v1_outcome_row(task=task, trial=trial, fidelity_rows=[])
    assert row["structural_failure"] is True
    assert row["checkpoints"] == []
    assert row["discarded_checkpoint_count"] == 1
