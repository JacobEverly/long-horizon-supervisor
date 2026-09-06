import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from horizon_supervisor.training import run_continuation_calibration_v5 as runner
from horizon_supervisor.training.freeze_continuation_calibration import EXACT_MODELS


def _kwarg(command: list[str], name: str) -> str:
    return next(item for item in command if item.startswith(f"{name}="))


def test_v5_command_uses_permission_preserving_environment(tmp_path: Path) -> None:
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

    assert command[command.index("--env") + 1] == runner.EXPECTED_ENVIRONMENT
    assert json.loads(_kwarg(command, "model_info").split("=", 1)[1])[
        "max_output_tokens"
    ] == 8192


def test_v5_report_uses_only_fresh_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [{"task_id": "fresh-only"}]
    seen = {}

    def fake_analyze(value):
        seen["rows"] = value
        return {"gate_passed": False, "decision": "not-ready"}

    monkeypatch.setattr(runner, "analyze", fake_analyze)
    report = runner._report({"outcomes": rows}, tranche=1)

    assert seen["rows"] is rows
    assert report["cohort"] == "fresh_v5_only"
    assert report["prior_outcome_count_used"] == 0
    assert report["new_v5_trajectory_count"] == 1
    assert report["tranche"] == 1


def test_v5_runtime_ceiling_is_tranche_specific() -> None:
    manifest = {
        "budget": {
            "tranche_1_incremental_ceiling_usd": 1.4,
            "tranche_2_incremental_ceiling_usd": 2.5,
            "phase_a_incremental_ceiling_usd": 3.5,
        }
    }

    assert runner._runtime_ceiling(manifest, 4.01, 1) == pytest.approx(5.41)
    assert runner._runtime_ceiling(manifest, 4.01, 2) == pytest.approx(6.51)
    assert runner._runtime_ceiling(manifest, 4.01, 3) == pytest.approx(7.51)
