from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import tomllib
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_MANIFEST = Path("benchmarks/terminal-bench-2.1-gate7.json")
DEFAULT_SWITCHYARD_CONFIG = Path("benchmarks/switchyard-gate7.toml")
DEFAULT_TASK = "log-summary-date-ranges"
DEFAULT_ROUTE = "gate7/stage-quality"


@dataclass(frozen=True)
class Gate7SmokeConfig:
    artifacts_root: Path
    manifest_path: Path = DEFAULT_MANIFEST
    switchyard_config_path: Path = DEFAULT_SWITCHYARD_CONFIG
    task_name: str = DEFAULT_TASK
    route_id: str = DEFAULT_ROUTE
    environment: str = "daytona"
    max_turns: int = 12
    max_output_tokens: int = 4_096
    reasoning_effort: str = "high"
    authorized_model_budget_usd: float = 1.0
    wall_timeout_seconds: int = 4_200

    def validate(self) -> None:
        if self.max_turns < 1:
            raise ValueError("max_turns must be positive")
        if not 256 <= self.max_output_tokens <= 8_192:
            raise ValueError("max_output_tokens must be between 256 and 8192")
        if self.authorized_model_budget_usd <= 0:
            raise ValueError("authorized_model_budget_usd must be positive")
        if self.wall_timeout_seconds < 60:
            raise ValueError("wall_timeout_seconds must be at least 60")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_frozen_inputs(config: Gate7SmokeConfig) -> dict[str, Any]:
    config.validate()
    manifest_path = config.manifest_path.resolve()
    switchyard_path = config.switchyard_config_path.resolve()
    manifest = _load_json(manifest_path)
    tasks = {row["name"]: row for row in manifest["tasks"]}
    if config.task_name not in tasks:
        raise ValueError(f"task is not in the frozen Gate 7 manifest: {config.task_name}")

    switchyard = tomllib.loads(switchyard_path.read_text(encoding="utf-8"))
    routes = {row["id"] for row in switchyard.get("routes", {}).values()}
    if config.route_id not in routes:
        raise ValueError(f"route is not in the frozen Switchyard config: {config.route_id}")
    return {
        "manifest": manifest,
        "task": tasks[config.task_name],
        "manifest_sha256": _sha256(manifest_path),
        "switchyard_config_sha256": _sha256(switchyard_path),
    }


def query_openrouter_key(api_key: str) -> dict[str, Any]:
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/auth/key",
        headers={
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "long-horizon-supervisor/0.1",
        },
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.load(response)
    return payload["data"]


def key_remaining_usd(key_info: dict[str, Any]) -> float | None:
    remaining = key_info.get("limit_remaining")
    if remaining is not None:
        return float(remaining)
    limit = key_info.get("limit")
    if limit is None:
        return None
    return max(0.0, float(limit) - float(key_info.get("usage", 0.0)))


def enforce_dedicated_key_cap(
    key_info: dict[str, Any], authorized_budget_usd: float, tolerance_usd: float = 0.02
) -> float:
    remaining = key_remaining_usd(key_info)
    if remaining is None:
        raise RuntimeError("Gate 7 requires a dedicated OpenRouter key with a finite limit")
    if remaining > authorized_budget_usd + tolerance_usd:
        raise RuntimeError(
            f"dedicated key can still spend ${remaining:.2f}, above the authorized "
            f"${authorized_budget_usd:.2f} smoke budget; use a more tightly capped key"
        )
    if remaining <= 0:
        raise RuntimeError("dedicated OpenRouter key has no remaining budget")
    return remaining


def build_harbor_command(
    config: Gate7SmokeConfig,
    *,
    switchyard_base_url: str,
    job_name: str,
    jobs_dir: Path,
) -> list[str]:
    model_info = json.dumps(
        {
            "max_input_tokens": 1_000_000,
            "max_output_tokens": config.max_output_tokens,
        },
        separators=(",", ":"),
    )
    call_kwargs = json.dumps(
        {"max_tokens": config.max_output_tokens}, separators=(",", ":")
    )
    return [
        str(Path(sys.executable).with_name("harbor")),
        "run",
        "--task",
        f"terminal-bench/{config.task_name}",
        "--agent",
        "terminus-2",
        "--model",
        f"openai/{config.route_id}",
        "--env",
        config.environment,
        "--n-concurrent",
        "1",
        "--n-attempts",
        "1",
        "--max-retries",
        "0",
        "--job-name",
        job_name,
        "--jobs-dir",
        str(jobs_dir),
        "--agent-kwarg",
        f"api_base={switchyard_base_url}/v1",
        "--agent-kwarg",
        f"max_turns={config.max_turns}",
        "--agent-kwarg",
        f"reasoning_effort={config.reasoning_effort}",
        "--agent-kwarg",
        "record_terminal_session=false",
        "--agent-kwarg",
        f"model_info={model_info}",
        "--agent-kwarg",
        f"llm_call_kwargs={call_kwargs}",
        "--yes",
    ]


def _read_url_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.load(response)


def validate_harbor_task_lock(
    lock_path: Path, frozen_task: dict[str, Any]
) -> dict[str, str]:
    """Prove Harbor executed the exact content frozen in the Gate 7 manifest."""
    lock = _load_json(lock_path)
    trials = lock.get("trials", [])
    if len(trials) != 1:
        raise RuntimeError(f"expected one Harbor trial in {lock_path}, found {len(trials)}")
    actual = trials[0]["task"]
    expected_name = f"terminal-bench/{frozen_task['name']}"
    expected_digest = f"sha256:{frozen_task['digest']}"
    if actual.get("name") != expected_name or actual.get("digest") != expected_digest:
        raise RuntimeError(
            "Harbor task lock does not match the frozen Gate 7 task: "
            f"expected {expected_name}@{expected_digest}, got "
            f"{actual.get('name')}@{actual.get('digest')}"
        )
    return {"name": expected_name, "digest": expected_digest}


def _run_harbor(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout_seconds: int,
    termination_grace_seconds: float = 10,
) -> tuple[int, str, str, bool]:
    # Capture to files instead of pipes. A Harbor descendant can inherit a pipe and
    # keep communicate() blocked even after Harbor itself has been killed.
    with (
        tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stdout_file,
        tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stderr_file,
    ):
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            start_new_session=True,
        )
        process_group_id = process.pid
        timed_out = False
        deadline = time.time() + timeout_seconds
        try:
            # On macOS, the monotonic clock used by subprocess.wait(timeout=...)
            # pauses while the laptop sleeps. Recheck an absolute wall-clock
            # deadline at short intervals so sleep cannot extend a paid run.
            while process.poll() is None:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout_seconds)
                try:
                    process.wait(timeout=min(remaining, 1.0))
                except subprocess.TimeoutExpired:
                    continue
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process_group_id, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=termination_grace_seconds)
            except subprocess.TimeoutExpired:
                pass
            # Kill the complete group even if the Harbor parent exited after TERM;
            # otherwise a stubborn descendant can outlive the declared wall limit.
            try:
                os.killpg(process_group_id, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=termination_grace_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=termination_grace_seconds)

        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout = stdout_file.read()
        stderr = stderr_file.read()
        if timed_out:
            stderr += (
                f"\nGate 7 wall timeout reached after {timeout_seconds} seconds.\n"
            )
        return process.returncode, stdout, stderr, timed_out


def run_gate7_smoke(config: Gate7SmokeConfig) -> dict[str, Any]:
    frozen = validate_frozen_inputs(config)
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required")
    environment_key = {
        "daytona": "DAYTONA_API_KEY",
        "e2b": "E2B_API_KEY",
        "runloop": "RUNLOOP_API_KEY",
        "modal": "MODAL_TOKEN_ID",
    }.get(config.environment)
    if environment_key and not os.getenv(environment_key):
        raise RuntimeError(f"{environment_key} is required for {config.environment}")

    key_before = query_openrouter_key(api_key)
    remaining_before = enforce_dedicated_key_cap(
        key_before, config.authorized_model_budget_usd
    )

    from switchyard.cli.launchers.native_server import NativeServer

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_root = config.artifacts_root.resolve() / f"gate7-smoke-{timestamp}"
    jobs_dir = run_root / "jobs"
    run_root.mkdir(parents=True, exist_ok=False)
    jobs_dir.mkdir()
    job_name = f"gate7-{config.task_name}-{config.route_id.rsplit('/', 1)[-1]}"

    server = NativeServer(config.switchyard_config_path.resolve())
    try:
        command = build_harbor_command(
            config,
            switchyard_base_url=server.base_url,
            job_name=job_name,
            jobs_dir=jobs_dir,
        )
        config_record = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        }
        preflight = {
            "gate": 7,
            "kind": "cloud-agent-smoke",
            "created_at": datetime.now(UTC).isoformat(),
            "config": config_record,
            "frozen_task": frozen["task"],
            "manifest_sha256": frozen["manifest_sha256"],
            "switchyard_config_sha256": frozen["switchyard_config_sha256"],
            "switchyard_version": "0.2.0",
            "harbor_version": "0.21.0",
            "dedicated_key_remaining_before_usd": remaining_before,
            "command": command,
        }
        (run_root / "run-manifest.json").write_text(
            json.dumps(preflight, indent=2, default=str), encoding="utf-8"
        )

        process_env = os.environ.copy()
        process_env["OPENAI_API_KEY"] = "switchyard-local"
        process_env["OPENAI_BASE_URL"] = f"{server.base_url}/v1"
        return_code, stdout, stderr, timed_out = _run_harbor(
            command,
            cwd=Path.cwd(),
            env=process_env,
            timeout_seconds=config.wall_timeout_seconds,
        )
        (run_root / "harbor.log").write_text(
            stdout + stderr, encoding="utf-8"
        )
        stats = _read_url_json(f"{server.base_url}/v1/stats")
        (run_root / "routing-stats.json").write_text(
            json.dumps(stats, indent=2), encoding="utf-8"
        )
    finally:
        server.close()

    key_after = query_openrouter_key(api_key)
    remaining_after = key_remaining_usd(key_after)
    provider_spend = (
        remaining_before - remaining_after if remaining_after is not None else None
    )
    result_path = jobs_dir / job_name / "result.json"
    harbor_result = _load_json(result_path) if result_path.exists() else None
    lock_path = jobs_dir / job_name / "lock.json"
    locked_task = (
        validate_harbor_task_lock(lock_path, frozen["task"])
        if lock_path.exists()
        else None
    )
    report = {
        **preflight,
        "return_code": return_code,
        "wall_timeout_reached": timed_out,
        "provider_spend_usd": provider_spend,
        "dedicated_key_remaining_after_usd": remaining_after,
        "locked_task": locked_task,
        "harbor_result": harbor_result,
        "routing_stats": stats,
        "run_root": str(run_root),
    }
    (run_root / "gate7-smoke-report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    return report
