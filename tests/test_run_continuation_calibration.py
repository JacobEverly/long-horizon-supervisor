import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from horizon_supervisor.training import freeze_continuation_calibration as freezer
from horizon_supervisor.training.run_continuation_calibration import (
    OBSERVATION_KEYS,
    _continuation_command,
    _outcome_row,
    validate_key_budget,
    validate_manifest,
)


def _budget_manifest() -> dict:
    return {
        "budget": {
            "phase_a_incremental_ceiling_usd": 5.0,
            "per_trial_incremental_ceiling_usd": 0.5,
        }
    }


def _catalog() -> dict:
    return {
        "schema_version": "continuation-model-catalog.v0",
        "captured_at": "2026-09-05T00:00:00+00:00",
        "source": freezer.MODEL_CATALOG_URL,
        "models": [
            {
                "model_id": model_id,
                "canonical_slug": model_id,
                "created": 1,
                "context_length": 1_000_000,
                "max_completion_tokens": 131_072,
                "pricing": {},
                "supported_parameters": ["tools"],
            }
            for model_id in freezer.EXACT_MODELS.values()
        ],
    }


def test_key_budget_requires_a_tight_hard_limit() -> None:
    manifest = _budget_manifest()
    validate_key_budget(manifest, {"usage": 2.0, "limit": 7.0}, baseline=2.0)

    with pytest.raises(RuntimeError, match="safety envelope"):
        validate_key_budget(
            manifest, {"usage": 2.0, "limit": 10.0}, baseline=2.0
        )
    with pytest.raises(RuntimeError, match="lacks the frozen Phase A budget"):
        validate_key_budget(
            manifest, {"usage": 2.0, "limit": 6.99}, baseline=2.0
        )


def test_continuation_command_uses_v2_agent_and_never_stops(tmp_path: Path) -> None:
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
        model_id="deepseek/deepseek-v4-flash-0731",
        job_name="test-job",
        record_path=tmp_path / "record.jsonl",
        provider_usage_start=0.0,
        provider_usage_ceiling=2.5,
    )
    joined = "\n".join(command)

    assert "ContinuationTerminus2" in command[command.index("--agent") + 1]
    assert "continuation_detector_config=" in joined
    assert "pilot_stop_after_checkpoint=false" in joined
    assert "pilot_stop_after_healthy_window=false" in joined


def test_outcome_row_excludes_a_structural_trajectory_checkpoint(
    tmp_path: Path,
) -> None:
    record = tmp_path / "record.jsonl"
    event = {
        "schema_version": "two-tier-observation-event.v0",
        "observation": {
            key: None for key in OBSERVATION_KEYS
        },
        "assessment": {"status": "STRUCTURAL_FAILURE"},
    }
    checkpoint = {
        "schema_version": "matched-checkpoint.v0",
        "run_id": "run",
        "checkpoint_kind": "healthy",
        "base_model_id": "deepseek/deepseek-v4-flash-0731",
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
            "started_at": "2026-09-05T00:00:00Z",
            "finished_at": "2026-09-05T00:00:01Z",
        },
        "stats": {"models": {}},
        "model_id": "deepseek/deepseek-v4-flash-0731",
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

    row = _outcome_row(task=task, trial=trial, fidelity_rows=[])

    assert row["structural_failure"] is True
    assert row["checkpoints"] == []
    assert row["discarded_checkpoint_count"] == 1


def test_frozen_manifest_reloads_with_all_hashes(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(freezer, "fetch_model_catalog", _catalog)
    result = freezer.freeze(tmp_path)

    manifest, digest = validate_manifest(Path(result["manifest_path"]))

    assert digest == result["manifest_sha256"]
    assert manifest["detector"]["config"] == freezer.SELECTED_CONFIG.model_dump()
