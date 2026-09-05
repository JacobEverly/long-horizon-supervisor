from pathlib import Path
from types import SimpleNamespace

import pytest

from horizon_supervisor.training import run_continuation_calibration_v3 as runner
from horizon_supervisor.training.freeze_continuation_calibration import EXACT_MODELS


def test_v3_command_uses_harness_filtered_agent(tmp_path: Path) -> None:
    task_root = tmp_path / "tasks" / "sample-task"
    task_root.mkdir(parents=True)
    (task_root / "task.toml").write_text("version = '1.0'\n", encoding="utf-8")
    command = runner._continuation_command(
        manifest={"execution": {"max_turns": 12}},
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
        provider_usage_ceiling=1.6,
    )

    assert command[command.index("--agent") + 1].endswith(
        ":HarnessFilteredContinuationTerminus2"
    )


def test_v3_runtime_ceiling_is_tranche_specific() -> None:
    manifest = {
        "budget": {
            "tranche_1_incremental_ceiling_usd": 1.6,
            "tranche_2_incremental_ceiling_usd": 3.25,
            "phase_a_incremental_ceiling_usd": 4.93,
        }
    }

    assert runner._runtime_ceiling(manifest, 0.05, 1) == pytest.approx(1.65)
    assert runner._runtime_ceiling(manifest, 0.05, 2) == pytest.approx(3.30)
    assert runner._runtime_ceiling(manifest, 0.05, 3) == pytest.approx(4.98)
