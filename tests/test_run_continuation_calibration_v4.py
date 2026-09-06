import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from horizon_supervisor.training import run_continuation_calibration_v4 as runner
from horizon_supervisor.training.freeze_continuation_calibration import EXACT_MODELS


def _kwarg(command: list[str], name: str) -> str:
    return next(item for item in command if item.startswith(f"{name}="))


def test_v4_command_only_raises_per_response_cap(tmp_path: Path) -> None:
    task_root = tmp_path / "tasks" / "sample-task"
    task_root.mkdir(parents=True)
    (task_root / "task.toml").write_text("version = '1.0'\n", encoding="utf-8")
    command = runner._continuation_command(
        manifest={
            "execution": {
                "max_turns": 12,
                "per_response_output_tokens": 8192,
                "total_output_token_budget": 49152,
            }
        },
        server=SimpleNamespace(base_url="http://127.0.0.1:9999"),
        run_root=tmp_path,
        task={
            "task_id": "sample-task",
            "task_category": "debugging",
            "task_root": str(task_root),
        },
        route_id="gate7/fixed-flash",
        model_id=EXACT_MODELS["gate7/fixed-flash"],
        job_name="test-job",
        record_path=tmp_path / "record.jsonl",
        provider_usage_start=0.0,
        provider_usage_ceiling=1.4,
    )

    assert command[command.index("--agent") + 1].endswith(
        ":HarnessFilteredContinuationTerminus2"
    )
    model_info = json.loads(_kwarg(command, "model_info").split("=", 1)[1])
    call_kwargs = json.loads(_kwarg(command, "llm_call_kwargs").split("=", 1)[1])
    assert model_info["max_output_tokens"] == 8192
    assert call_kwargs["max_tokens"] == 8192
    assert _kwarg(command, "pilot_output_token_budget") == (
        "pilot_output_token_budget=49152"
    )


def test_v4_runtime_ceiling_is_tranche_specific() -> None:
    manifest = {
        "budget": {
            "tranche_1_incremental_ceiling_usd": 1.4,
            "tranche_2_incremental_ceiling_usd": 2.5,
            "phase_a_incremental_ceiling_usd": 3.5,
        }
    }

    assert runner._runtime_ceiling(manifest, 1.48, 1) == pytest.approx(2.88)
    assert runner._runtime_ceiling(manifest, 1.48, 2) == pytest.approx(3.98)
    assert runner._runtime_ceiling(manifest, 1.48, 3) == pytest.approx(4.98)
