from __future__ import annotations

import json
from pathlib import Path

import pytest

from horizon_supervisor.benchmark.gate8 import (
    DEFAULT_ROUTES,
    Gate8PilotConfig,
    _has_budget_for_next_task,
    build_harbor_command,
    validate_pilot_inputs,
)

ROOT = Path(__file__).resolve().parents[1]


def _config(**overrides: object) -> Gate8PilotConfig:
    values = {
        "artifacts_root": ROOT / "artifacts/official",
        "panel_path": ROOT / "data/supervisor/terminal-bench-pro-panel-v0.jsonl",
        "tasks_path": ROOT / "data/supervisor/terminal-bench-pro-wave-1/tasks",
        "switchyard_config_path": ROOT / "benchmarks/switchyard-gate7.toml",
        "budget_contract_path": ROOT / "benchmarks/gate8-budget-v0.json",
    }
    values.update(overrides)
    return Gate8PilotConfig(**values)  # type: ignore[arg-type]


def test_gate8_input_lock_produces_18_by_4_matched_trials() -> None:
    frozen = validate_pilot_inputs(_config())
    assert frozen["wave_task_count"] == 18
    assert frozen["selected_task_count"] == 18
    assert frozen["trial_count"] == 72
    assert frozen["wave"] == 1


def test_gate8_rejects_mismatched_budget_wave(tmp_path: Path) -> None:
    contract = json.loads((ROOT / "benchmarks/gate8-budget-v0.json").read_text())
    contract["wave"] = 2
    path = tmp_path / "budget.json"
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="budget contract is for wave 2"):
        validate_pilot_inputs(_config(budget_contract_path=path))


def test_gate8_command_repeats_models_and_can_limit_smoke_task(tmp_path: Path) -> None:
    config = _config(include_task_names=("implement-gmm-em-cli",))
    command = build_harbor_command(
        config,
        switchyard_base_url="http://127.0.0.1:9000",
        job_name="test",
        jobs_dir=tmp_path,
    )
    models = [command[index + 1] for index, value in enumerate(command) if value == "--model"]
    assert models == [f"openai/{route}" for route in DEFAULT_ROUTES]
    assert command[:3] == [
        command[0],
        "-m",
        "horizon_supervisor.benchmark.harbor_bounded",
    ]
    assert command[command.index("--include-task-name") + 1] == "implement-gmm-em-cli"
    assert command[command.index("--n-concurrent") + 1] == "2"
    assert "--max-retries" in command
    call_kwargs = next(
        value.removeprefix("llm_call_kwargs=")
        for value in command
        if value.startswith("llm_call_kwargs=")
    )
    assert json.loads(call_kwargs) == {"max_tokens": 4096, "timeout": 1200}


def test_gate8_rejects_tasks_outside_frozen_wave() -> None:
    with pytest.raises(ValueError, match="not in frozen wave"):
        validate_pilot_inputs(_config(include_task_names=("not-a-task",)))


def test_gate8_rejects_budget_above_frozen_cap() -> None:
    with pytest.raises(ValueError, match="exceeds frozen"):
        validate_pilot_inputs(_config(authorized_model_budget_usd=50.01))


def test_gate8_rejects_too_short_request_timeout() -> None:
    with pytest.raises(ValueError, match="request timeout"):
        validate_pilot_inputs(_config(request_timeout_seconds=59))


def test_gate8_rejects_invalid_retry_attempts() -> None:
    with pytest.raises(ValueError, match="retry attempts"):
        validate_pilot_inputs(_config(request_retry_attempts=0))


def test_gate8_rejects_negative_output_length_retries() -> None:
    with pytest.raises(ValueError, match="output-length retry attempts"):
        validate_pilot_inputs(_config(output_length_retry_attempts=-1))


def test_gate8_rejects_unknown_model_roster() -> None:
    with pytest.raises(ValueError, match="unsupported model roster"):
        validate_pilot_inputs(_config(model_roster="not-a-roster"))


def test_gate8_exposes_optional_incremental_usage_guard(tmp_path: Path) -> None:
    contract = json.loads((ROOT / "benchmarks/gate8-budget-v0.json").read_text())
    contract["model_budget"]["dedicated_key_usage_ceiling_usd"] = 12.5
    contract["model_budget"]["minimum_next_task_reserve_usd"] = 2.0
    path = tmp_path / "budget.json"
    path.write_text(json.dumps(contract))

    frozen = validate_pilot_inputs(_config(budget_contract_path=path))

    assert frozen["dedicated_key_usage_ceiling_usd"] == 12.5
    assert frozen["minimum_next_task_reserve_usd"] == 2.0


def test_gate8_rejects_usage_ceiling_above_key_hard_cap(tmp_path: Path) -> None:
    contract = json.loads((ROOT / "benchmarks/gate8-budget-v0.json").read_text())
    contract["model_budget"]["dedicated_key_usage_ceiling_usd"] = 50.01
    path = tmp_path / "budget.json"
    path.write_text(json.dumps(contract))

    with pytest.raises(ValueError, match="usage ceiling"):
        validate_pilot_inputs(_config(budget_contract_path=path))


def test_gate8_incremental_guard_reserves_one_more_task_batch() -> None:
    assert _has_budget_for_next_task(
        {"usage": 31.5},
        usage_ceiling_usd=33.552935902,
        minimum_reserve_usd=2.0,
    )
    assert not _has_budget_for_next_task(
        {"usage": 31.552935902},
        usage_ceiling_usd=33.552935902,
        minimum_reserve_usd=2.0,
    )
    assert not _has_budget_for_next_task(
        {},
        usage_ceiling_usd=33.552935902,
        minimum_reserve_usd=2.0,
    )
    assert _has_budget_for_next_task(
        {}, usage_ceiling_usd=None, minimum_reserve_usd=0.0
    )


def test_gate8_enforces_optional_execution_contract(tmp_path: Path) -> None:
    contract = json.loads((ROOT / "benchmarks/gate8-budget-v0.json").read_text())
    contract["trial_count"] = 2
    contract["execution_contract"] = {
        "route_ids": ["gate7/fixed-qwen", "gate7/fixed-glm"],
        "selected_task_names": ["implement-gmm-em-cli"],
        "run_controls": {"n_concurrent": 2, "max_turns": 12},
    }
    path = tmp_path / "budget.json"
    path.write_text(json.dumps(contract))
    config = _config(
        budget_contract_path=path,
        route_ids=("gate7/fixed-qwen", "gate7/fixed-glm"),
        include_task_names=("implement-gmm-em-cli",),
    )

    frozen = validate_pilot_inputs(config)

    assert frozen["trial_count"] == 2


def test_gate8_rejects_execution_contract_route_drift(tmp_path: Path) -> None:
    contract = json.loads((ROOT / "benchmarks/gate8-budget-v0.json").read_text())
    contract["trial_count"] = 2
    contract["execution_contract"] = {
        "route_ids": ["gate7/fixed-qwen", "gate7/fixed-glm"],
        "selected_task_names": ["implement-gmm-em-cli"],
    }
    path = tmp_path / "budget.json"
    path.write_text(json.dumps(contract))

    with pytest.raises(ValueError, match="route ids do not match"):
        validate_pilot_inputs(
            _config(
                budget_contract_path=path,
                include_task_names=("implement-gmm-em-cli",),
            )
        )


def test_gate8_rejects_execution_contract_task_drift(tmp_path: Path) -> None:
    contract = json.loads((ROOT / "benchmarks/gate8-budget-v0.json").read_text())
    contract["trial_count"] = 2
    contract["execution_contract"] = {
        "route_ids": ["gate7/fixed-qwen", "gate7/fixed-glm"],
        "selected_task_names": ["implement-gmm-em-cli"],
    }
    path = tmp_path / "budget.json"
    path.write_text(json.dumps(contract))

    with pytest.raises(ValueError, match="selected tasks do not match"):
        validate_pilot_inputs(
            _config(
                budget_contract_path=path,
                route_ids=("gate7/fixed-qwen", "gate7/fixed-glm"),
                include_task_names=("compute-best-chess-move-san",),
            )
        )
