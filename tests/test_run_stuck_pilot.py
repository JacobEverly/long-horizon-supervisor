import hashlib
import json
from pathlib import Path

import pytest

from horizon_supervisor.training import run_stuck_pilot
from horizon_supervisor.training.run_stuck_pilot import (
    BASE_ROUTE_TO_MODEL,
    BRANCH_ACTIONS,
    KIMI_MODEL,
    _assert_budget,
    _branch_specs,
    _harbor_command,
    _limits,
    _load_prior_outcomes,
    _retryable_infrastructure_failure,
    _task_parent,
)


def test_task_parent_is_the_exact_wave_tasks_directory() -> None:
    task = {
        "task_root": (
            "data/supervisor/terminal-bench-pro-wave-1/tasks/implement-gmm-em-cli"
        )
    }
    parent = _task_parent(task)
    assert parent.name == "tasks"
    assert (parent / "implement-gmm-em-cli" / "task.toml").is_file()


@pytest.mark.parametrize(
    ("base_route", "other_route"),
    [
        ("gate7/fixed-flash", "gate7/fixed-qwen"),
        ("gate7/fixed-qwen", "gate7/fixed-flash"),
    ],
)
def test_branch_specs_include_reverse_switch_and_kimi(
    base_route: str, other_route: str
) -> None:
    specs = _branch_specs(base_route)
    assert specs == [
        (
            "restart_current_clean",
            base_route,
            BASE_ROUTE_TO_MODEL[base_route],
            False,
        ),
        (
            "switch_value_state",
            other_route,
            BASE_ROUTE_TO_MODEL[other_route],
            True,
        ),
        (
            "restart_value_clean",
            other_route,
            BASE_ROUTE_TO_MODEL[other_route],
            False,
        ),
        ("switch_kimi_state", "gate7/fixed-kimi", KIMI_MODEL, True),
        ("restart_kimi_clean", "gate7/fixed-kimi", KIMI_MODEL, False),
    ]


def test_base_command_has_global_cap_but_no_premature_branch_cap() -> None:
    command = _harbor_command(
        task_parent=Path("/tmp/tasks"),
        task_id="task-a",
        route_id="gate7/fixed-flash",
        model_id="deepseek/deepseek-v4-flash-0731",
        job_name="job",
        jobs_dir=Path("/tmp/jobs"),
        record_path=Path("/tmp/record.jsonl"),
        max_turns=12,
        agent_timeout_seconds=3_600,
        provider_usage_start=None,
        provider_usage_ceiling=10.0,
        stats_url="http://127.0.0.1:1234/v1/stats",
        capture_healthy=False,
        capture_stuck=True,
    )
    rendered = "\n".join(command)
    assert "pilot_provider_usage_ceiling=10.0" in rendered
    assert "pilot_provider_usage_start=" not in rendered
    assert "pilot_capture_stuck=true" in rendered
    assert "pilot_capture_healthy=false" in rendered
    assert "pilot_stop_after_checkpoint=false" in rendered
    assert "--agent-timeout-multiplier\n1.0" in rendered


def test_state_branch_uses_seed_adapter_and_public_handoff() -> None:
    command = _harbor_command(
        task_parent=Path("/tmp/tasks"),
        task_id="task-a",
        route_id="gate7/fixed-qwen",
        model_id="qwen/qwen3.8-27b",
        job_name="job",
        jobs_dir=Path("/tmp/jobs"),
        record_path=Path("/tmp/record.jsonl"),
        max_turns=7,
        agent_timeout_seconds=2_000,
        provider_usage_start=1.25,
        provider_usage_ceiling=10.0,
        stats_url="http://127.0.0.1:1234/v1/stats",
        capture_healthy=False,
        capture_stuck=False,
        stop_after_checkpoint=True,
        workspace_seed=Path("/tmp/anchor"),
        expected_workspace_digest="digest",
        handoff_path=Path("/tmp/handoff.md"),
    )
    rendered = "\n".join(command)
    assert "SeededDaytonaEnvironment" in rendered
    assert "workspace_seed_path=/tmp/anchor" in rendered
    assert "expected_workspace_digest=digest" in rendered
    assert "pilot_provider_usage_start=1.25" in rendered
    assert "pilot_stop_after_checkpoint=true" in rendered
    assert "--extra-instruction-path\n/tmp/handoff.md" in rendered


def test_limits_match_remaining_source_budget() -> None:
    limits = _limits(
        {
            "observation": {"turn": 4},
            "agent_elapsed_seconds": 125.2,
        }
    )
    assert limits == {
        "remaining_turns": 8,
        "remaining_output_tokens": 32_768,
        "maximum_wall_seconds": 3_474,
        "maximum_incremental_spend_usd": 0.5,
    }


def test_total_budget_guard_requires_a_full_trial_reserve() -> None:
    manifest = {
        "budget": {
            "usage_ceiling_usd": 15.0,
            "dedicated_key_hard_limit_usd": 8.0,
        }
    }
    _assert_budget(manifest, {"usage": 7.49})
    with pytest.raises(RuntimeError, match="spend ceiling"):
        _assert_budget(manifest, {"usage": 7.51})


def test_only_infrastructure_failures_are_retried() -> None:
    base = {"valid": False, "timed_out": False}
    output_limit = {
        **base,
        "result": {
            "exception_info": {
                "exception_type": "OutputLengthExceededError",
                "exception_message": "response was truncated",
            }
        },
    }
    daytona = {
        **base,
        "result": {
            "exception_info": {
                "exception_type": "DaytonaError",
                "exception_message": "sandbox creation failed",
            }
        },
    }
    provider = {
        **base,
        "result": {
            "exception_info": {
                "exception_type": "OpenAIError",
                "exception_message": "provider rate limit",
            }
        },
    }
    assert _retryable_infrastructure_failure(output_limit) is False
    assert _retryable_infrastructure_failure(provider) is False
    assert _retryable_infrastructure_failure(daytona) is True


def test_prior_outcomes_require_one_complete_hashed_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_stuck_pilot, "ROOT", tmp_path)
    path = tmp_path / "prior.jsonl"
    rows = [
        {
            "schema_version": "matched-stuck-branch-outcome.v0",
            "group_id": "suspected-stuck-group",
            "checkpoint_kind": "suspected_stuck",
            "branch_action": action,
            "valid": True,
        }
        for action in sorted(BRANCH_ACTIONS)
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    path.with_suffix(".sha256").write_text(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n",
        encoding="utf-8",
    )
    resume = {
        "prior_outcomes_path": path.name,
        "prior_accepted_outcome_count": 6,
        "prior_group_counts": {"suspected_stuck": 1, "healthy": 0},
    }
    assert _load_prior_outcomes(resume) == rows

    rows[0]["valid"] = False
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    path.with_suffix(".sha256").write_text(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="learning-valid"):
        _load_prior_outcomes(resume)
