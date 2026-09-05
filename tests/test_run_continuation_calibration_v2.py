import json
from pathlib import Path

from horizon_supervisor.training import run_continuation_calibration_v2 as runner


def _checkpoint() -> dict:
    return {
        "schema_version": "matched-checkpoint.v0",
        "run_id": "run",
        "checkpoint_kind": "needs_review",
        "base_model_id": "model",
        "observation": {"turn": 6, "workspace_digest": "digest"},
        "state_transfer_eligible": True,
    }


def test_checkpoint_replay_does_not_receive_provider_credentials(
    monkeypatch, tmp_path: Path
) -> None:
    captured = {}
    monkeypatch.setenv("OPENROUTER_API_KEY", "must-not-be-forwarded")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-forwarded")
    monkeypatch.setenv("OPENAI_BASE_URL", "must-not-be-forwarded")
    monkeypatch.setenv("DAYTONA_API_KEY", "required-for-environment")
    monkeypatch.setattr(
        runner,
        "_fidelity_command",
        lambda **_: (["harbor", "run"], tmp_path / "job"),
    )

    def fake_run(command, *, environment, timeout):
        captured.update(environment)
        assert command == ["harbor", "run"]
        assert timeout == 900
        return 0, "", False

    monkeypatch.setattr(runner, "_run_command", fake_run)
    monkeypatch.setattr(
        runner,
        "_trial_result",
        lambda _: {"exception_info": None},
    )

    result = runner._validate_checkpoint(
        task={"task_id": "task"},
        checkpoint=_checkpoint(),
        run_root=tmp_path,
    )

    assert result["passed"] is True
    assert result["provider_model_calls"] == 0
    assert captured["DAYTONA_API_KEY"] == "required-for-environment"
    assert "OPENROUTER_API_KEY" not in captured
    assert "OPENAI_API_KEY" not in captured
    assert "OPENAI_BASE_URL" not in captured


def test_pending_trial_is_finalized_without_another_model_call(
    monkeypatch, tmp_path: Path
) -> None:
    checkpoint = _checkpoint()
    record_path = tmp_path / "record.jsonl"
    record_path.write_text(json.dumps(checkpoint) + "\n", encoding="utf-8")
    trial = {
        "record_path": str(record_path),
        "valid": True,
        "result": {},
        "route_id": "route",
        "model_id": "model",
    }
    state = {
        "pending_trial": {
            "schedule_item": "1:route",
            "task_position": 1,
            "trial": trial,
            "fidelity_rows": [],
        },
        "fidelity_rows": [],
        "outcomes": [],
        "completed_schedule_items": [],
        "attempts": [],
    }
    manifest = {
        "task_selection": {
            "ordered_pool": [{"position": 1, "task_id": "task"}]
        }
    }
    calls = {"fidelity": 0, "persist": 0}

    def fake_fidelity(**_):
        calls["fidelity"] += 1
        return {
            "snapshot_id": "run-needs_review-t06",
            "passed": True,
        }

    monkeypatch.setattr(runner, "_validate_checkpoint", fake_fidelity)
    monkeypatch.setattr(runner, "_structural_runtime_failure", lambda *_: False)
    monkeypatch.setattr(
        runner,
        "_v1_outcome_row",
        lambda **kwargs: {
            "task_id": kwargs["task"]["task_id"],
            "checkpoints": kwargs["fidelity_rows"],
        },
    )
    monkeypatch.setattr(
        runner,
        "_persist",
        lambda _: calls.__setitem__("persist", calls["persist"] + 1),
    )

    runner._finalize_pending(state, manifest)

    assert calls == {"fidelity": 1, "persist": 2}
    assert state["pending_trial"] is None
    assert state["completed_schedule_items"] == ["1:route"]
    assert len(state["outcomes"]) == 1
