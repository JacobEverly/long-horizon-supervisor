from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from horizon_supervisor.benchmark.gate7 import (
    Gate7SmokeConfig,
    _run_harbor,
    build_harbor_command,
    enforce_dedicated_key_cap,
    validate_frozen_inputs,
    validate_harbor_task_lock,
)


def test_run_harbor_wall_timeout_kills_stubborn_process_group(tmp_path: Path) -> None:
    child_pid_path = tmp_path / "child.pid"
    script = "\n".join(
        (
            "import os, signal, subprocess, sys, time",
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)",
            "child = subprocess.Popen([sys.executable, '-c', "
            "'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "time.sleep(30)'])",
            f"open({str(child_pid_path)!r}, 'w').write(str(child.pid))",
            "print('parent-ready', flush=True)",
            "time.sleep(30)",
        )
    )
    started = time.monotonic()

    return_code, stdout, stderr, timed_out = _run_harbor(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=os.environ.copy(),
        timeout_seconds=0.2,
        termination_grace_seconds=0.1,
    )

    assert time.monotonic() - started < 2
    assert return_code != 0
    assert stdout == "parent-ready\n"
    assert "wall timeout reached after 0.2 seconds" in stderr
    assert timed_out is True
    child_pid = int(child_pid_path.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


def _config(**overrides: object) -> Gate7SmokeConfig:
    values = {"artifacts_root": Path("artifacts/official")}
    values.update(overrides)
    return Gate7SmokeConfig(**values)  # type: ignore[arg-type]


def test_gate7_frozen_inputs_contain_default_task_and_route() -> None:
    frozen = validate_frozen_inputs(_config())

    assert frozen["task"]["name"] == "log-summary-date-ranges"
    assert len(frozen["manifest"]["tasks"]) == 30


def test_gate7_command_uses_cloud_terminus_and_switchyard() -> None:
    command = build_harbor_command(
        _config(),
        switchyard_base_url="http://127.0.0.1:4012",
        job_name="smoke",
        jobs_dir=Path("jobs"),
    )

    assert Path(command[0]).name == "harbor"
    assert command[1] == "run"
    assert "daytona" in command
    assert "terminus-2" in command
    assert "openai/gate7/stage-quality" in command
    assert "api_base=http://127.0.0.1:4012/v1" in command
    assert not any("OPENROUTER" in token for token in command)


def test_gate7_requires_a_finitely_capped_dedicated_key() -> None:
    assert enforce_dedicated_key_cap({"limit_remaining": 0.98}, 1.0) == 0.98

    with pytest.raises(RuntimeError, match="finite limit"):
        enforce_dedicated_key_cap({"limit": None}, 1.0)
    with pytest.raises(RuntimeError, match="more tightly capped"):
        enforce_dedicated_key_cap({"limit_remaining": 2.0}, 1.0)


def test_gate7_validates_harbor_locked_digest(tmp_path: Path) -> None:
    frozen_task = {
        "name": "log-summary-date-ranges",
        "digest": "27b074a2f10fff7606e096f3abd8dced418ad8fda0f53d88acbe477f2d9ceaf6",
    }
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(
        """{
          "trials": [{
            "task": {
              "name": "terminal-bench/log-summary-date-ranges",
              "digest": "sha256:27b074a2f10fff7606e096f3abd8dced418ad8fda0f53d88acbe477f2d9ceaf6"
            }
          }]
        }""",
        encoding="utf-8",
    )

    locked = validate_harbor_task_lock(lock_path, frozen_task)

    assert locked["digest"] == f"sha256:{frozen_task['digest']}"

    lock_path.write_text(lock_path.read_text().replace("27b074", "000000"))
    with pytest.raises(RuntimeError, match="does not match"):
        validate_harbor_task_lock(lock_path, frozen_task)
