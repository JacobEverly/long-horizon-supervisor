from __future__ import annotations

import hashlib
import json
import os
import sys
import tomllib
import urllib.request
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from horizon_supervisor.benchmark.gate7 import (
    DEFAULT_SWITCHYARD_CONFIG,
    _read_url_json,
    _run_harbor,
    enforce_dedicated_key_cap,
    key_remaining_usd,
    query_openrouter_key,
)
from horizon_supervisor.benchmark.model_catalog import load_model_catalog

DEFAULT_PANEL = Path("data/supervisor/terminal-bench-pro-panel-v0.jsonl")
DEFAULT_TASKS = Path("data/supervisor/terminal-bench-pro-wave-1/tasks")
DEFAULT_BUDGET_CONTRACT = Path("benchmarks/gate8-budget-v0.json")
DEFAULT_ROUTES = (
    "gate7/fixed-qwen",
    "gate7/fixed-flash",
    "gate7/fixed-glm",
    "gate7/fixed-kimi",
)


@dataclass(frozen=True)
class Gate8PilotConfig:
    artifacts_root: Path
    wave: int = 1
    panel_path: Path = DEFAULT_PANEL
    tasks_path: Path = DEFAULT_TASKS
    switchyard_config_path: Path = DEFAULT_SWITCHYARD_CONFIG
    budget_contract_path: Path = DEFAULT_BUDGET_CONTRACT
    route_ids: tuple[str, ...] = DEFAULT_ROUTES
    include_task_names: tuple[str, ...] = ()
    environment: str = "daytona"
    n_concurrent: int = 2
    max_turns: int = 12
    max_output_tokens: int = 4_096
    reasoning_effort: str = "high"
    request_timeout_seconds: int = 1_200
    request_retry_attempts: int = 1
    output_length_retry_attempts: int = 1
    authorized_model_budget_usd: float = 50.0
    wall_timeout_seconds: int = 21_600
    model_roster: str = "gate4"

    def validate(self) -> None:
        if self.wave < 1:
            raise ValueError("wave must be positive")
        if not self.route_ids:
            raise ValueError("at least one route is required")
        if len(set(self.route_ids)) != len(self.route_ids):
            raise ValueError("route ids must be unique")
        if self.n_concurrent < 1:
            raise ValueError("n_concurrent must be positive")
        if self.max_turns < 1:
            raise ValueError("max_turns must be positive")
        if not 256 <= self.max_output_tokens <= 8_192:
            raise ValueError("max_output_tokens must be between 256 and 8192")
        if self.request_timeout_seconds < 60:
            raise ValueError("request timeout must be at least 60 seconds")
        if self.request_retry_attempts < 1:
            raise ValueError("request retry attempts must be positive")
        if self.output_length_retry_attempts < 0:
            raise ValueError("output-length retry attempts cannot be negative")
        if self.authorized_model_budget_usd <= 0:
            raise ValueError("authorized model budget must be positive")
        if self.wall_timeout_seconds < 60:
            raise ValueError("wall timeout must be at least 60 seconds")
        if self.model_roster not in {"gate4", "swiss_cheese"}:
            raise ValueError("unsupported model roster")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _panel_wave(path: Path, wave: int) -> dict[str, dict[str, Any]]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["wave"] == wave:
                rows[row["source_task_name"]] = row
    return rows


def validate_pilot_inputs(config: Gate8PilotConfig) -> dict[str, Any]:
    config.validate()
    panel_path = config.panel_path.resolve()
    tasks_path = config.tasks_path.resolve()
    switchyard_path = config.switchyard_config_path.resolve()
    budget_path = config.budget_contract_path.resolve()
    budget = json.loads(budget_path.read_text(encoding="utf-8"))
    panel_rows = _panel_wave(panel_path, config.wave)
    materialized_names = {
        path.name for path in tasks_path.iterdir() if path.is_dir()
    }
    if materialized_names != panel_rows.keys():
        raise RuntimeError(
            f"materialized task names do not match frozen wave {config.wave}"
        )
    selected_names = (
        set(config.include_task_names) if config.include_task_names else materialized_names
    )
    missing = selected_names - materialized_names
    if missing:
        raise ValueError(
            f"requested tasks are not in frozen wave {config.wave}: {sorted(missing)}"
        )

    switchyard = tomllib.loads(switchyard_path.read_text(encoding="utf-8"))
    available_routes = {row["id"] for row in switchyard.get("routes", {}).values()}
    missing_routes = set(config.route_ids) - available_routes
    if missing_routes:
        raise ValueError(f"routes are not in the frozen config: {sorted(missing_routes)}")
    hard_cap = budget["model_budget"]["dedicated_openrouter_key_hard_cap_usd"]
    usage_ceiling = budget["model_budget"].get(
        "dedicated_key_usage_ceiling_usd"
    )
    next_task_reserve = float(
        budget["model_budget"].get("minimum_next_task_reserve_usd", 0.0)
    )
    if usage_ceiling is not None:
        usage_ceiling = float(usage_ceiling)
        if not 0 < usage_ceiling <= float(hard_cap):
            raise ValueError(
                "dedicated key usage ceiling must be positive and no greater "
                "than the frozen key hard cap"
            )
        if not 0 <= next_task_reserve < usage_ceiling:
            raise ValueError(
                "minimum next-task reserve must be non-negative and smaller "
                "than the dedicated key usage ceiling"
            )
    budget_wave = int(budget.get("wave", 1))
    if budget_wave != config.wave:
        raise ValueError(
            f"budget contract is for wave {budget_wave}, not wave {config.wave}"
        )
    if config.authorized_model_budget_usd > hard_cap:
        raise ValueError(
            f"requested model budget exceeds frozen ${hard_cap:.2f} Gate 8 cap"
        )
    execution_contract = budget.get("execution_contract")
    if execution_contract:
        contracted_routes = tuple(execution_contract["route_ids"])
        if config.route_ids != contracted_routes:
            raise ValueError(
                "route ids do not match the budget execution contract: "
                f"expected {contracted_routes}, got {config.route_ids}"
            )
        contracted_tasks = set(execution_contract["selected_task_names"])
        if selected_names != contracted_tasks:
            raise ValueError(
                "selected tasks do not match the budget execution contract"
            )
        for control_name, expected_value in execution_contract.get(
            "run_controls", {}
        ).items():
            actual_value = getattr(config, control_name)
            if actual_value != expected_value:
                raise ValueError(
                    f"{control_name} does not match the budget execution "
                    f"contract: expected {expected_value!r}, got {actual_value!r}"
                )
        contracted_trial_count = len(contracted_routes) * len(contracted_tasks)
        if budget.get("trial_count") != contracted_trial_count:
            raise ValueError(
                "budget trial_count does not match its execution contract"
            )
    actual_digests = {
        "panel_sha256": _sha256(panel_path),
        "tasks_tree_sha256": _tree_sha256(tasks_path),
        "switchyard_config_sha256": _sha256(switchyard_path),
    }
    if actual_digests != budget["frozen_inputs"]:
        raise RuntimeError("Gate 8 inputs changed after the budget contract was frozen")
    return {
        **actual_digests,
        "budget_contract_sha256": _sha256(budget_path),
        "wave": config.wave,
        "wave_task_count": len(materialized_names),
        "selected_task_count": len(selected_names),
        "selected_task_names": sorted(selected_names),
        "trial_count": len(selected_names) * len(config.route_ids),
        "dedicated_key_usage_ceiling_usd": usage_ceiling,
        "minimum_next_task_reserve_usd": next_task_reserve,
    }


def build_harbor_command(
    config: Gate8PilotConfig,
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
        {
            "max_tokens": config.max_output_tokens,
            "timeout": config.request_timeout_seconds,
        },
        separators=(",", ":"),
    )
    command = [
        sys.executable,
        "-m",
        "horizon_supervisor.benchmark.harbor_bounded",
        "run",
        "--path",
        str(config.tasks_path.resolve()),
        "--agent",
        "terminus-2",
    ]
    for route_id in config.route_ids:
        command.extend(["--model", f"openai/{route_id}"])
    command.extend(
        [
            "--env",
            config.environment,
            "--n-concurrent",
            str(config.n_concurrent),
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
        ]
    )
    for task_name in config.include_task_names:
        command.extend(["--include-task-name", task_name])
    command.append("--yes")
    return command


def _post_url_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, method="POST")
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.load(response)


def _has_budget_for_next_task(
    key_info: dict[str, Any],
    *,
    usage_ceiling_usd: float | None,
    minimum_reserve_usd: float,
) -> bool:
    if usage_ceiling_usd is None:
        return True
    usage = key_info.get("usage")
    if usage is None:
        return False
    return usage_ceiling_usd - float(usage) > minimum_reserve_usd


def _config_record(config: Gate8PilotConfig) -> dict[str, Any]:
    record = {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in asdict(config).items()
    }
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in record.items()
    }


def _task_job_name(wave: int, index: int, task_name: str) -> str:
    safe_name = "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in task_name
    )
    return f"gate8-wave{wave}-{index:02d}-{safe_name}"


def run_gate8_pilot(config: Gate8PilotConfig) -> dict[str, Any]:
    frozen = validate_pilot_inputs(config)
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required")
    if config.environment == "daytona" and not os.getenv("DAYTONA_API_KEY"):
        raise RuntimeError("DAYTONA_API_KEY is required for Daytona")
    key_before = query_openrouter_key(api_key)
    remaining_before = enforce_dedicated_key_cap(
        key_before, config.authorized_model_budget_usd
    )

    from switchyard.cli.launchers.native_server import NativeServer

    # Include microseconds so independent route-isolation runs can start in the
    # same second without racing for the same artifact directory.
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_root = config.artifacts_root.resolve() / f"gate8-pilot-{timestamp}"
    jobs_dir = run_root / "jobs"
    task_stats_dir = run_root / "task-stats"
    run_root.mkdir(parents=True, exist_ok=False)
    jobs_dir.mkdir()
    task_stats_dir.mkdir()
    pricing_path = run_root / "pricing-snapshot.json"
    load_model_catalog(pricing_path, roster=config.model_roster)
    task_names = frozen["selected_task_names"]
    preflight = {
        "gate": 8,
        "kind": "matched-fixed-model-development-pilot",
        "created_at": datetime.now(UTC).isoformat(),
        "config": _config_record(config),
        "frozen_inputs": frozen,
        "switchyard_version": "0.2.0",
        "harbor_version": "0.21.0",
        "dedicated_key_remaining_before_usd": remaining_before,
        "task_execution_mode": "one-task-four-model-batches",
        "pricing_snapshot": str(pricing_path),
    }
    (run_root / "run-manifest.json").write_text(
        json.dumps(preflight, indent=2), encoding="utf-8"
    )

    server = NativeServer(config.switchyard_config_path.resolve())
    task_runs: list[dict[str, Any]] = []
    stop_reason: str | None = None
    try:
        process_env = os.environ.copy()
        process_env["OPENAI_API_KEY"] = "switchyard-local"
        process_env["OPENAI_BASE_URL"] = f"{server.base_url}/v1"
        process_env["HORIZON_HARBOR_LLM_ATTEMPTS"] = str(
            config.request_retry_attempts
        )
        process_env["HORIZON_HARBOR_OUTPUT_LENGTH_RETRIES"] = str(
            config.output_length_retry_attempts
        )
        for index, task_name in enumerate(task_names, start=1):
            task_key_before = query_openrouter_key(api_key)
            task_remaining_before = key_remaining_usd(task_key_before)
            if task_remaining_before is None or task_remaining_before <= 0:
                stop_reason = "dedicated_key_exhausted"
                break
            usage_ceiling = frozen["dedicated_key_usage_ceiling_usd"]
            if not _has_budget_for_next_task(
                task_key_before,
                usage_ceiling_usd=usage_ceiling,
                minimum_reserve_usd=frozen["minimum_next_task_reserve_usd"],
            ):
                stop_reason = "incremental_spend_ceiling_reserve_reached"
                break
            _post_url_json(f"{server.base_url}/v1/stats/reset")
            task_config = replace(config, include_task_names=(task_name,))
            job_name = _task_job_name(config.wave, index, task_name)
            command = build_harbor_command(
                task_config,
                switchyard_base_url=server.base_url,
                job_name=job_name,
                jobs_dir=jobs_dir,
            )
            return_code, stdout, stderr, timed_out = _run_harbor(
                command,
                cwd=Path.cwd(),
                env=process_env,
                timeout_seconds=config.wall_timeout_seconds,
            )
            (run_root / f"{job_name}.log").write_text(
                stdout + stderr, encoding="utf-8"
            )
            stats = _read_url_json(f"{server.base_url}/v1/stats")
            stats_path = task_stats_dir / f"{index:02d}-{task_name}.json"
            stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
            task_key_after = query_openrouter_key(api_key)
            task_remaining_after = key_remaining_usd(task_key_after)
            task_provider_spend = (
                task_remaining_before - task_remaining_after
                if task_remaining_after is not None
                else None
            )
            result_path = jobs_dir / job_name / "result.json"
            task_run = {
                "source_task_name": task_name,
                "job_name": job_name,
                "command": command,
                "return_code": return_code,
                "wall_timeout_reached": timed_out,
                "provider_spend_usd": task_provider_spend,
                "dedicated_key_remaining_before_usd": task_remaining_before,
                "dedicated_key_remaining_after_usd": task_remaining_after,
                "routing_stats_path": str(stats_path),
                "routing_stats": stats,
                "harbor_result": (
                    json.loads(result_path.read_text(encoding="utf-8"))
                    if result_path.exists()
                    else None
                ),
            }
            task_runs.append(task_run)
            partial = {
                **preflight,
                "status": "in_progress",
                "completed_task_batches": len(task_runs),
                "task_runs": task_runs,
            }
            (run_root / "partial-report.json").write_text(
                json.dumps(partial, indent=2), encoding="utf-8"
            )
            if return_code != 0 or timed_out:
                break
    finally:
        server.close()

    key_after = query_openrouter_key(api_key)
    remaining_after = key_remaining_usd(key_after)
    provider_spend = (
        remaining_before - remaining_after if remaining_after is not None else None
    )
    report = {
        **preflight,
        "status": (
            "complete"
            if len(task_runs) == len(task_names)
            and all(run["return_code"] == 0 for run in task_runs)
            else "stopped"
        ),
        "return_code": next(
            (run["return_code"] for run in task_runs if run["return_code"] != 0),
            0,
        ),
        "wall_timeout_reached": any(
            run["wall_timeout_reached"] for run in task_runs
        ),
        "provider_spend_usd": provider_spend,
        "dedicated_key_remaining_after_usd": remaining_after,
        "task_tree_unchanged": _tree_sha256(config.tasks_path.resolve())
        == frozen["tasks_tree_sha256"],
        "completed_task_batches": len(task_runs),
        "task_runs": task_runs,
        "stop_reason": stop_reason,
    }
    (run_root / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
