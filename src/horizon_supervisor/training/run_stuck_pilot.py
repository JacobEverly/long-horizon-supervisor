from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from daytona import Daytona
from switchyard.cli.launchers.native_server import NativeServer

from horizon_supervisor.benchmark.gate7 import query_openrouter_key

ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = ROOT / "artifacts/official/stuck-intervention-pilot-v0"
MANIFEST = OUTPUT_ROOT / "frozen-pilot-manifest-v2.json"
SWITCHYARD = ROOT / "benchmarks/switchyard-gate7.toml"
BASE_ROUTE_TO_MODEL = {
    "gate7/fixed-flash": "deepseek/deepseek-v4-flash-0731",
    "gate7/fixed-qwen": "qwen/qwen3.8-27b",
}
KIMI_ROUTE = "gate7/fixed-kimi"
KIMI_MODEL = "moonshotai/kimi-k3"
BRANCH_ACTIONS = {
    "continue_current_state",
    "restart_current_clean",
    "switch_value_state",
    "restart_value_clean",
    "switch_kimi_state",
    "restart_kimi_clean",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(item.read_bytes()).digest())
    return digest.hexdigest()


def _validate_frozen_inputs(manifest: dict[str, Any]) -> None:
    if manifest["models"]["routes"] != {
        **BASE_ROUTE_TO_MODEL,
        KIMI_ROUTE: KIMI_MODEL,
    }:
        raise RuntimeError("frozen route/model roster changed")
    integrity = manifest["integrity"]
    fixed_files = {
        SWITCHYARD: integrity["switchyard_sha256"],
        ROOT / manifest["models"]["catalog_path"]: manifest["models"][
            "catalog_sha256"
        ],
        ROOT / manifest["execution"]["snapshot_fidelity_path"]: manifest[
            "execution"
        ]["snapshot_fidelity_sha256"],
    }
    fixed_files.update(
        {ROOT / relative: expected for relative, expected in integrity["code_sha256"].items()}
    )
    for path, expected in fixed_files.items():
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"frozen input hash mismatch: {path}")
    for task in manifest["task_selection"]["ordered_pool"]:
        task_root = ROOT / task["task_root"]
        if _tree_sha256(task_root) != task["task_tree_sha256"]:
            raise RuntimeError(f"frozen task tree changed: {task['task_id']}")
    if len(manifest["task_selection"]["ordered_pool"]) != 8:
        raise RuntimeError("frozen task pool must contain exactly eight tasks")


def _post_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, method="POST")
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.load(response)


def _get_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.load(response)


def _run_command(
    command: list[str], *, environment: dict[str, str], timeout: int
) -> tuple[int, str, bool]:
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return result.returncode, result.stdout + result.stderr, False
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout.decode() if isinstance(error.stdout, bytes) else error.stdout or ""
        stderr = error.stderr.decode() if isinstance(error.stderr, bytes) else error.stderr or ""
        return 124, stdout + stderr, True


def _trial_result(job_dir: Path) -> dict[str, Any] | None:
    candidates = sorted(path for path in job_dir.glob("*/result.json") if path.is_file())
    if len(candidates) != 1:
        return None
    return json.loads(candidates[0].read_text(encoding="utf-8"))


def _task_parent(task: dict[str, Any]) -> Path:
    task_root = ROOT / Path(task["task_root"])
    parent = task_root.parent
    if not (task_root / "task.toml").is_file() or parent.name != "tasks":
        raise ValueError(f"invalid frozen task root: {task_root}")
    return parent


def _duration(result: dict[str, Any] | None) -> float:
    if not result or not result.get("started_at") or not result.get("finished_at"):
        return 0.0
    start = datetime.fromisoformat(result["started_at"].replace("Z", "+00:00"))
    finish = datetime.fromisoformat(result["finished_at"].replace("Z", "+00:00"))
    return (finish - start).total_seconds()


def _reward(result: dict[str, Any] | None) -> float:
    rewards = ((result or {}).get("verifier_result") or {}).get("rewards") or {}
    return float(rewards.get("reward", 0.0))


def _model_stats(stats: dict[str, Any], model_id: str) -> dict[str, Any]:
    return (stats.get("models") or {}).get(model_id) or {}


def _provider_error(result: dict[str, Any] | None) -> bool:
    exception = (result or {}).get("exception_info") or {}
    text = f"{exception.get('exception_type', '')} {exception.get('exception_message', '')}".lower()
    return any(
        marker in text
        for marker in (
            "openai",
            "litellm",
            "provider",
            "rate limit",
            "connection",
            "timeout",
        )
    )


def _protocol_error(record_path: Path) -> bool:
    if not record_path.exists():
        return False
    for line in record_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        assessment = row.get("assessment") or {}
        if assessment.get("immediate_signal") == "protocol_failure":
            return True
    return False


def _valid_trial(
    *, return_code: int, timed_out: bool, result: dict[str, Any] | None
) -> bool:
    return (
        return_code == 0
        and not timed_out
        and result is not None
        and result.get("exception_info") is None
        and result.get("verifier_result") is not None
    )


def _retryable_infrastructure_failure(trial: dict[str, Any]) -> bool:
    if trial["valid"]:
        return False
    if trial["timed_out"] or trial["result"] is None:
        return True
    result = trial["result"]
    exception = result.get("exception_info") or {}
    exception_type = str(exception.get("exception_type", "")).lower()
    exception_message = str(exception.get("exception_message", "")).lower()
    infrastructure_markers = (
        "daytona",
        "sandbox",
        "environment",
        "container",
        "docker",
        "verifier",
        "build",
    )
    explicit_infrastructure = any(
        marker in exception_type or marker in exception_message
        for marker in infrastructure_markers
    )
    if explicit_infrastructure:
        return True
    if _provider_error(result) or "outputlengthexceeded" in exception_type:
        return False
    return False


def _harbor_command(
    *,
    task_parent: Path,
    task_id: str,
    route_id: str,
    model_id: str,
    job_name: str,
    jobs_dir: Path,
    record_path: Path,
    max_turns: int,
    agent_timeout_seconds: int,
    provider_usage_start: float | None,
    stats_url: str,
    capture_healthy: bool,
    capture_stuck: bool,
    provider_usage_ceiling: float,
    stop_after_checkpoint: bool = False,
    workspace_seed: Path | None = None,
    expected_workspace_digest: str | None = None,
    handoff_path: Path | None = None,
) -> list[str]:
    model_info = json.dumps(
        {"max_input_tokens": 1_000_000, "max_output_tokens": 4_096},
        separators=(",", ":"),
    )
    call_kwargs = json.dumps(
        {"max_tokens": 4_096, "timeout": 1_200}, separators=(",", ":")
    )
    environment = (
        "horizon_supervisor.benchmark.pilot_harbor:SeededDaytonaEnvironment"
        if workspace_seed
        else "daytona"
    )
    command = [
        sys.executable,
        "-m",
        "horizon_supervisor.benchmark.harbor_bounded",
        "run",
        "--path",
        str(task_parent),
        "--agent",
        "horizon_supervisor.benchmark.pilot_harbor:PilotTerminus2",
        "--model",
        f"openai/{route_id}",
        "--env",
        environment,
        "--n-concurrent",
        "1",
        "--n-attempts",
        "1",
        "--max-retries",
        "0",
        "--agent-timeout-multiplier",
        str(agent_timeout_seconds / 3_600),
        "--job-name",
        job_name,
        "--jobs-dir",
        str(jobs_dir),
        "--agent-kwarg",
        f"api_base={stats_url.removesuffix('/stats')}",
        "--agent-kwarg",
        f"max_turns={max_turns}",
        "--agent-kwarg",
        "reasoning_effort=high",
        "--agent-kwarg",
        "record_terminal_session=false",
        "--agent-kwarg",
        f"model_info={model_info}",
        "--agent-kwarg",
        f"llm_call_kwargs={call_kwargs}",
        "--agent-kwarg",
        f"pilot_record_path={record_path}",
        "--agent-kwarg",
        f"pilot_run_id={job_name}",
        "--agent-kwarg",
        f"pilot_base_model_id={model_id}",
        "--agent-kwarg",
        f"pilot_capture_healthy={json.dumps(capture_healthy)}",
        "--agent-kwarg",
        f"pilot_capture_stuck={json.dumps(capture_stuck)}",
        "--agent-kwarg",
        f"pilot_stop_after_checkpoint={json.dumps(stop_after_checkpoint)}",
        "--agent-kwarg",
        "pilot_healthy_turn=4",
        "--agent-kwarg",
        f"pilot_output_token_budget={max_turns * 4096}",
        "--agent-kwarg",
        "pilot_spend_budget_usd=0.5",
        "--agent-kwarg",
        f"pilot_stats_url={stats_url}",
        "--agent-kwarg",
        f"pilot_provider_usage_ceiling={provider_usage_ceiling}",
        "--include-task-name",
        task_id,
    ]
    if provider_usage_start is not None:
        command.extend(
            [
                "--agent-kwarg",
                f"pilot_provider_usage_start={provider_usage_start}",
            ]
        )
    if workspace_seed:
        command.extend(
            [
                "--environment-kwarg",
                f"workspace_seed_path={workspace_seed}",
                "--environment-kwarg",
                f"expected_workspace_digest={expected_workspace_digest}",
            ]
        )
    if handoff_path:
        command.extend(["--extra-instruction-path", str(handoff_path)])
    command.append("--yes")
    return command


def _assert_budget(
    manifest: dict[str, Any], key_info: dict[str, Any], *, minimum_reserve: float = 0.5
) -> None:
    usage = float(key_info["usage"])
    allowed = min(
        float(manifest["budget"]["usage_ceiling_usd"]),
        float(manifest["budget"]["dedicated_key_hard_limit_usd"]),
    )
    if allowed - usage < minimum_reserve:
        raise RuntimeError("frozen OpenRouter spend ceiling lacks one-trial reserve")


def _run_trial(
    *,
    server: NativeServer,
    manifest: dict[str, Any],
    run_root: Path,
    task: dict[str, Any],
    route_id: str,
    model_id: str,
    label: str,
    max_turns: int,
    capture_healthy: bool = False,
    capture_stuck: bool = False,
    stop_after_checkpoint: bool = False,
    workspace_seed: Path | None = None,
    expected_workspace_digest: str | None = None,
    handoff_path: Path | None = None,
    enforce_branch_budget: bool = True,
    agent_timeout_seconds: int = 3_600,
) -> dict[str, Any]:
    api_key = os.environ["OPENROUTER_API_KEY"]
    before_info = query_openrouter_key(api_key)
    _assert_budget(manifest, before_info)
    usage_before = float(before_info["usage"])
    _post_json(f"{server.base_url}/v1/stats/reset")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    job_name = f"{label}-{timestamp}"
    jobs_dir = run_root / "jobs"
    record_path = run_root / "records" / f"{job_name}.jsonl"
    task_parent = _task_parent(task)
    command = _harbor_command(
        task_parent=task_parent,
        task_id=task["task_id"],
        route_id=route_id,
        model_id=model_id,
        job_name=job_name,
        jobs_dir=jobs_dir,
        record_path=record_path,
        max_turns=max_turns,
        agent_timeout_seconds=agent_timeout_seconds,
        provider_usage_start=usage_before if enforce_branch_budget else None,
        stats_url=f"{server.base_url}/v1/stats",
        capture_healthy=capture_healthy,
        capture_stuck=capture_stuck,
        stop_after_checkpoint=stop_after_checkpoint,
        provider_usage_ceiling=min(
            float(manifest["budget"]["usage_ceiling_usd"]),
            float(manifest["budget"]["dedicated_key_hard_limit_usd"]),
        ),
        workspace_seed=workspace_seed,
        expected_workspace_digest=expected_workspace_digest,
        handoff_path=handoff_path,
    )
    environment = os.environ.copy()
    environment["OPENAI_API_KEY"] = "switchyard-local"
    environment["OPENAI_BASE_URL"] = f"{server.base_url}/v1"
    environment["HORIZON_HARBOR_LLM_ATTEMPTS"] = "1"
    environment["HORIZON_HARBOR_OUTPUT_LENGTH_RETRIES"] = "1"
    return_code, output, timed_out = _run_command(
        command, environment=environment, timeout=agent_timeout_seconds + 1_500
    )
    log_path = run_root / "logs" / f"{job_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(output, encoding="utf-8")
    stats = _get_json(f"{server.base_url}/v1/stats")
    key_query_error = None
    try:
        after_info = query_openrouter_key(api_key)
        usage_after = float(after_info["usage"])
    except (urllib.error.HTTPError, urllib.error.URLError) as error:
        usage_after = usage_before
        key_query_error = f"{type(error).__name__}: {error}"
    job_dir = jobs_dir / job_name
    result = _trial_result(job_dir)
    return {
        "job_name": job_name,
        "task_id": task["task_id"],
        "route_id": route_id,
        "model_id": model_id,
        "return_code": return_code,
        "timed_out": timed_out,
        "provider_usage_before_usd": usage_before,
        "provider_usage_after_usd": usage_after,
        "provider_spend_usd": max(0.0, usage_after - usage_before),
        "key_query_error": key_query_error,
        "stats": stats,
        "record_path": str(record_path),
        "job_dir": str(job_dir),
        "result": result,
        "valid": _valid_trial(
            return_code=return_code, timed_out=timed_out, result=result
        ),
    }


def _run_with_infrastructure_retry(**kwargs: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    attempts = []
    for _attempt in range(2):
        trial = _run_trial(**kwargs)
        attempts.append(trial)
        if trial["valid"]:
            return trial, attempts
        if not _retryable_infrastructure_failure(trial):
            break
    return attempts[-1], attempts


def _checkpoint_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        row
        for line in path.read_text(encoding="utf-8").splitlines()
        if (row := json.loads(line)).get("schema_version") == "matched-checkpoint.v0"
    ]


def _limits(checkpoint: dict[str, Any]) -> dict[str, Any]:
    remaining_turns = 12 - int(checkpoint["observation"]["turn"])
    remaining_wall = max(
        60, 3_600 - math.ceil(float(checkpoint["agent_elapsed_seconds"]))
    )
    return {
        "remaining_turns": remaining_turns,
        "remaining_output_tokens": remaining_turns * 4_096,
        "maximum_wall_seconds": remaining_wall,
        "maximum_incremental_spend_usd": 0.5,
    }


def _continue_outcome(
    *,
    group_id: str,
    task: dict[str, Any],
    base_trial: dict[str, Any],
    checkpoint: dict[str, Any],
    limits: dict[str, Any],
) -> dict[str, Any]:
    result = base_trial["result"]
    model_id = base_trial["model_id"]
    final_stats = _model_stats(base_trial["stats"], model_id)
    checkpoint_stats = _model_stats(checkpoint.get("routing_stats") or {}, model_id)
    start_usage = checkpoint.get("provider_usage_usd")
    cost = (
        max(0.0, base_trial["provider_usage_after_usd"] - float(start_usage))
        if start_usage is not None
        else base_trial["provider_spend_usd"]
    )
    checkpoint_time = datetime.fromisoformat(checkpoint["branch_started_at"])
    finished = datetime.fromisoformat(result["finished_at"].replace("Z", "+00:00"))
    return {
        "schema_version": "matched-stuck-branch-outcome.v0",
        "group_id": group_id,
        "task_id": task["task_id"],
        "task_category": task["category"],
        "checkpoint_kind": checkpoint["checkpoint_kind"],
        "checkpoint_turn": checkpoint["observation"]["turn"],
        "base_model_id": base_trial["model_id"],
        "destination_model_id": base_trial["model_id"],
        "branch_action": "continue_current_state",
        "preserved_state": True,
        **limits,
        "verified_completion": _reward(result) >= 1.0,
        "verifier_reward": _reward(result),
        "cost_usd": cost,
        "input_tokens": max(
            0,
            int(final_stats.get("prompt_tokens", 0))
            - int(checkpoint_stats.get("prompt_tokens", 0)),
        ),
        "output_tokens": max(
            0,
            int(final_stats.get("completion_tokens", 0))
            - int(checkpoint_stats.get("completion_tokens", 0)),
        ),
        "cached_tokens": max(
            0,
            int(final_stats.get("cached_tokens", 0))
            - int(checkpoint_stats.get("cached_tokens", 0)),
        ),
        "reasoning_tokens": max(
            0,
            int(final_stats.get("reasoning_tokens", 0))
            - int(checkpoint_stats.get("reasoning_tokens", 0)),
        ),
        "elapsed_seconds": max(0.0, (finished - checkpoint_time).total_seconds()),
        "state_transfer_failure": False,
        "protocol_error": _protocol_error(Path(base_trial["record_path"])),
        "provider_error": False,
        "valid": True,
        "source_job": base_trial["job_name"],
    }


def _branch_outcome(
    *,
    group_id: str,
    task: dict[str, Any],
    base_model_id: str,
    checkpoint: dict[str, Any],
    action: str,
    trial: dict[str, Any],
    limits: dict[str, Any],
) -> dict[str, Any]:
    result = trial["result"]
    stats = _model_stats(trial["stats"], trial["model_id"])
    state_transfer = action in {"switch_value_state", "switch_kimi_state"}
    exception_text = json.dumps((result or {}).get("exception_info") or {}).lower()
    state_failure = state_transfer and any(
        marker in exception_text
        for marker in ("rehydrated workspace", "workspace_seed", "environment")
    )
    return {
        "schema_version": "matched-stuck-branch-outcome.v0",
        "group_id": group_id,
        "task_id": task["task_id"],
        "task_category": task["category"],
        "checkpoint_kind": checkpoint["checkpoint_kind"],
        "checkpoint_turn": checkpoint["observation"]["turn"],
        "base_model_id": base_model_id,
        "destination_model_id": trial["model_id"],
        "branch_action": action,
        "preserved_state": state_transfer,
        **limits,
        "verified_completion": _reward(result) >= 1.0,
        "verifier_reward": _reward(result),
        "cost_usd": trial["provider_spend_usd"],
        "input_tokens": int(stats.get("prompt_tokens", 0)),
        "output_tokens": int(stats.get("completion_tokens", 0)),
        "cached_tokens": int(stats.get("cached_tokens", 0)),
        "reasoning_tokens": int(stats.get("reasoning_tokens", 0)),
        "elapsed_seconds": _duration(result),
        "state_transfer_failure": state_failure,
        "protocol_error": _protocol_error(Path(trial["record_path"])),
        "provider_error": _provider_error(result),
        "valid": bool(trial["valid"]),
        "source_job": trial["job_name"],
    }


def _branch_specs(base_route: str) -> list[tuple[str, str, str, bool]]:
    other = (
        "gate7/fixed-qwen" if base_route == "gate7/fixed-flash" else "gate7/fixed-flash"
    )
    return [
        ("restart_current_clean", base_route, BASE_ROUTE_TO_MODEL[base_route], False),
        ("switch_value_state", other, BASE_ROUTE_TO_MODEL[other], True),
        ("restart_value_clean", other, BASE_ROUTE_TO_MODEL[other], False),
        ("switch_kimi_state", KIMI_ROUTE, KIMI_MODEL, True),
        ("restart_kimi_clean", KIMI_ROUTE, KIMI_MODEL, False),
    ]


def _attempt_usage_record(attempt: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "stuck-pilot-trial-usage.v0",
        "job_name": attempt["job_name"],
        "task_id": attempt["task_id"],
        "route_id": attempt["route_id"],
        "model_id": attempt["model_id"],
        "provider_usage_before_usd": attempt["provider_usage_before_usd"],
        "provider_usage_after_usd": attempt["provider_usage_after_usd"],
        "provider_spend_usd": attempt["provider_spend_usd"],
        "key_query_error": attempt.get("key_query_error"),
        "return_code": attempt["return_code"],
        "timed_out": attempt["timed_out"],
        "valid": attempt["valid"],
        "verified_completion": _reward(attempt["result"]) >= 1.0,
        "provider_error": _provider_error(attempt["result"]),
        "record_path": attempt["record_path"],
        "job_dir": attempt["job_dir"],
    }


def _load_prior_outcomes(resume: dict[str, Any]) -> list[dict[str, Any]]:
    relative = resume.get("prior_outcomes_path")
    expected_count = int(resume.get("prior_accepted_outcome_count", 0))
    expected_group_counts = {
        "suspected_stuck": int(
            (resume.get("prior_group_counts") or {}).get("suspected_stuck", 0)
        ),
        "healthy": int(
            (resume.get("prior_group_counts") or {}).get("healthy", 0)
        ),
    }
    if not relative:
        if expected_count or any(expected_group_counts.values()):
            raise RuntimeError("resume declares prior outcomes without an outcome path")
        return []

    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise RuntimeError("resume prior outcome path is missing or outside the repo")
    sidecar = path.with_suffix(".sha256")
    if not sidecar.is_file() or sidecar.read_text().split()[0] != _sha256(path):
        raise RuntimeError("resume prior outcome hash mismatch")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != expected_count:
        raise RuntimeError("resume prior outcome count mismatch")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if (
            row.get("schema_version") != "matched-stuck-branch-outcome.v0"
            or row.get("valid") is not True
            or row.get("branch_action") not in BRANCH_ACTIONS
        ):
            raise RuntimeError("resume prior outcome row is not learning-valid")
        grouped.setdefault(str(row.get("group_id")), []).append(row)
    actual_group_counts = {"suspected_stuck": 0, "healthy": 0}
    for group_rows in grouped.values():
        actions = {str(row["branch_action"]) for row in group_rows}
        kinds = {str(row.get("checkpoint_kind")) for row in group_rows}
        if len(group_rows) != 6 or actions != BRANCH_ACTIONS or len(kinds) != 1:
            raise RuntimeError("resume prior outcome group is not a complete matched group")
        kind = kinds.pop()
        if kind not in actual_group_counts:
            raise RuntimeError("resume prior outcome checkpoint kind is invalid")
        actual_group_counts[kind] += 1
    if actual_group_counts != expected_group_counts:
        raise RuntimeError("resume prior outcome group counts mismatch")
    return rows


def _cleanup_new_sandboxes(initial_ids: set[str]) -> dict[str, list[str]]:
    daytona = Daytona()
    removed = []
    errors = []
    for sandbox in list(daytona.list()):
        if sandbox.id in initial_ids:
            continue
        try:
            daytona.delete(sandbox, wait=True, timeout=120)
            removed.append(sandbox.id)
        except Exception as error:  # pragma: no cover - live recovery path
            errors.append(f"{sandbox.id}: {type(error).__name__}: {error}")
    return {"removed": removed, "errors": errors}


def run(manifest_path: Path = MANIFEST) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = (manifest_path.with_suffix(".sha256")).read_text().split()[0]
    if _sha256(manifest_path) != expected_hash:
        raise RuntimeError("frozen pilot manifest hash mismatch")
    _validate_frozen_inputs(manifest)
    if not os.getenv("OPENROUTER_API_KEY") or not os.getenv("DAYTONA_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY and DAYTONA_API_KEY are required")

    initial_sandboxes = {sandbox.id for sandbox in Daytona().list()}
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_root = OUTPUT_ROOT / f"execution-{timestamp}"
    run_root.mkdir(parents=True)
    resume = manifest.get("resume") or {}
    outcomes = _load_prior_outcomes(resume)
    attempts: list[dict[str, Any]] = []
    ineligible: list[dict[str, Any]] = list(resume.get("prior_ineligible", []))
    group_counts = {
        "suspected_stuck": int(
            (resume.get("prior_group_counts") or {}).get("suspected_stuck", 0)
        ),
        "healthy": int((resume.get("prior_group_counts") or {}).get("healthy", 0)),
    }
    skipped_schedule_items = set(resume.get("completed_schedule_items", []))
    server = NativeServer(SWITCHYARD)
    stop_reason = None
    execution_error = None
    try:
        tasks = manifest["task_selection"]["ordered_pool"]
        for task in tasks:
            if all(count >= 4 for count in group_counts.values()):
                break
            for base_route, base_model in BASE_ROUTE_TO_MODEL.items():
                if all(count >= 4 for count in group_counts.values()):
                    break
                for requested_kind in ("suspected_stuck", "healthy"):
                    if group_counts[requested_kind] >= 4:
                        continue
                    schedule_item = (
                        f"{task['position']}:{base_route}:{requested_kind}"
                    )
                    if schedule_item in skipped_schedule_items:
                        continue
                    base_label = base_route.rsplit("-", 1)[-1]
                    base, base_attempts = _run_with_infrastructure_retry(
                        server=server,
                        manifest=manifest,
                        run_root=run_root,
                        task=task,
                        route_id=base_route,
                        model_id=base_model,
                        label=(
                            f"base-{requested_kind}-{task['position']:02d}-{base_label}"
                        ),
                        max_turns=12,
                        capture_healthy=requested_kind == "healthy",
                        capture_stuck=requested_kind == "suspected_stuck",
                        enforce_branch_budget=False,
                    )
                    attempts.extend(base_attempts)
                    if not base["valid"]:
                        ineligible.append(
                            {
                                "task_id": task["task_id"],
                                "base_model_id": base_model,
                                "checkpoint_kind": requested_kind,
                                "reason": "base infrastructure failed twice",
                            }
                        )
                        continue
                    checkpoints = [
                        checkpoint
                        for checkpoint in _checkpoint_records(Path(base["record_path"]))
                        if checkpoint["checkpoint_kind"] == requested_kind
                    ]
                    if not checkpoints:
                        ineligible.append(
                            {
                                "task_id": task["task_id"],
                                "base_model_id": base_model,
                                "checkpoint_kind": requested_kind,
                                "reason": "requested checkpoint did not occur",
                            }
                        )
                        continue
                    if len(checkpoints) != 1:
                        raise RuntimeError("a base trajectory captured duplicate checkpoints")
                    checkpoint = checkpoints[0]
                    kind = requested_kind
                    if not checkpoint["state_transfer_eligible"]:
                        ineligible.append(
                            {
                                "task_id": task["task_id"],
                                "base_model_id": base_model,
                                "checkpoint_kind": kind,
                                "reason": checkpoint[
                                    "state_transfer_ineligibility_reason"
                                ],
                            }
                        )
                        continue
                    group_id = (
                        f"{kind}-{task['position']:02d}-"
                        f"{base_route.rsplit('-', 1)[-1]}-t{checkpoint['observation']['turn']:02d}"
                    )
                    limits = _limits(checkpoint)
                    group_rows = [
                        _continue_outcome(
                            group_id=group_id,
                            task=task,
                            base_trial=base,
                            checkpoint=checkpoint,
                            limits=limits,
                        )
                    ]
                    handoff_path = run_root / "handoffs" / f"{group_id}.md"
                    handoff_path.parent.mkdir(parents=True, exist_ok=True)
                    handoff_path.write_text(checkpoint["handoff"], encoding="utf-8")
                    for action, route_id, model_id, state_transfer in _branch_specs(
                        base_route
                    ):
                        seed = (
                            Path(checkpoint["anchor_workspace_path"])
                            if state_transfer
                            else None
                        )
                        trial, trial_attempts = _run_with_infrastructure_retry(
                            server=server,
                            manifest=manifest,
                            run_root=run_root,
                            task=task,
                            route_id=route_id,
                            model_id=model_id,
                            label=f"{group_id}-{action}",
                            max_turns=limits["remaining_turns"],
                            capture_healthy=False,
                            capture_stuck=False,
                            workspace_seed=seed,
                            expected_workspace_digest=(
                                checkpoint["observation"]["workspace_digest"]
                                if state_transfer
                                else None
                            ),
                            handoff_path=handoff_path if state_transfer else None,
                            agent_timeout_seconds=limits["maximum_wall_seconds"],
                        )
                        attempts.extend(trial_attempts)
                        branch_row = _branch_outcome(
                            group_id=group_id,
                            task=task,
                            base_model_id=base_model,
                            checkpoint=checkpoint,
                            action=action,
                            trial=trial,
                            limits=limits,
                        )
                        group_rows.append(branch_row)
                        if not branch_row["valid"]:
                            break
                    if all(row["valid"] for row in group_rows):
                        outcomes.extend(group_rows)
                        group_counts[kind] += 1
                    else:
                        ineligible.append(
                            {
                                "group_id": group_id,
                                "reason": "one or more branches remained invalid after retry",
                                "invalid_actions": [
                                    row["branch_action"]
                                    for row in group_rows
                                    if not row["valid"]
                                ],
                                "unexecuted_actions": sorted(
                                    BRANCH_ACTIONS - {
                                        row["branch_action"] for row in group_rows
                                    }
                                ),
                            }
                        )
                    partial = {
                        "schema_version": "stuck-pilot-execution-ledger.v0",
                        "status": "in_progress",
                        "manifest_sha256": expected_hash,
                        "group_counts": group_counts,
                        "valid_outcome_count": len(outcomes),
                        "outcomes": outcomes,
                        "ineligible": ineligible,
                        "attempts": [
                            _attempt_usage_record(attempt) for attempt in attempts
                        ],
                    }
                    (run_root / "execution-ledger.json").write_text(
                        json.dumps(partial, indent=2), encoding="utf-8"
                    )
        if not all(count >= 4 for count in group_counts.values()):
            stop_reason = "frozen_pool_exhausted_before_both_group_targets"
    except RuntimeError as error:
        if "spend ceiling" in str(error):
            stop_reason = "frozen_openrouter_spend_ceiling_reached"
        else:
            stop_reason = "execution_runtime_error"
            execution_error = f"{type(error).__name__}: {error}"
    except (urllib.error.HTTPError, urllib.error.URLError) as error:
        stop_reason = "provider_key_unavailable"
        execution_error = f"{type(error).__name__}: {error}"
    except Exception as error:  # pragma: no cover - live safety net
        stop_reason = "unexpected_execution_error"
        execution_error = f"{type(error).__name__}: {error}"
    finally:
        server.close()
        cleanup = _cleanup_new_sandboxes(initial_sandboxes)

    usage_before = float(manifest["budget"]["usage_before_usd"])
    final_key_query_error = None
    try:
        key_after = query_openrouter_key(os.environ["OPENROUTER_API_KEY"])
        usage_after = float(key_after["usage"])
    except (urllib.error.HTTPError, urllib.error.URLError) as error:
        usage_after = usage_before
        final_key_query_error = f"{type(error).__name__}: {error}"
    current_key_spend = max(0.0, usage_after - usage_before)
    prior_key_spend = float(manifest["budget"].get("prior_key_spend_usd", 0.0))
    try:
        remaining_new_sandboxes = sorted(
            {sandbox.id for sandbox in Daytona().list()} - initial_sandboxes
        )
    except Exception as error:  # pragma: no cover - live safety net
        remaining_new_sandboxes = []
        cleanup["errors"].append(
            f"final list: {type(error).__name__}: {error}"
        )
    outcomes_path = run_root / "matched-branch-outcomes-v0.jsonl"
    outcomes_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in outcomes),
        encoding="utf-8",
    )
    usage_ledger_path = run_root / "trial-usage-ledger-v0.jsonl"
    usage_ledger_path.write_text(
        "".join(
            json.dumps(_attempt_usage_record(attempt), sort_keys=True) + "\n"
            for attempt in attempts
        ),
        encoding="utf-8",
    )
    report = {
        "schema_version": "stuck-pilot-execution-ledger.v0",
        "status": (
            "complete" if all(count >= 4 for count in group_counts.values()) else "stopped"
        ),
        "stop_reason": stop_reason,
        "execution_error": execution_error,
        "manifest_path": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": expected_hash,
        "group_counts": group_counts,
        "valid_outcome_count": len(outcomes),
        "outcomes_path": str(outcomes_path),
        "trial_usage_ledger_path": str(usage_ledger_path),
        "ineligible": ineligible,
        "attempt_count": len(attempts),
        "openrouter_usage_before_usd": usage_before,
        "openrouter_usage_after_usd": usage_after,
        "current_key_spend_usd": current_key_spend,
        "prior_key_spend_usd": prior_key_spend,
        "exact_incremental_openrouter_spend_usd": (
            prior_key_spend + current_key_spend
        ),
        "final_key_query_error": final_key_query_error,
        "exact_spend_reconciled": final_key_query_error is None,
        "additional_openrouter_cap_usd": manifest["budget"][
            "additional_openrouter_cap_usd"
        ],
        "daytona_charge_usd": None,
        "daytona_charge_availability": (
            "The installed Daytona SDK exposes sandbox lifecycle, not account charges."
        ),
        "cleanup": {
            "initial_sandbox_ids": sorted(initial_sandboxes),
            "removed_new_sandbox_ids": cleanup["removed"],
            "cleanup_errors": cleanup["errors"],
            "remaining_new_sandbox_ids": remaining_new_sandboxes,
        },
    }
    (run_root / "execution-ledger.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return {"run_root": str(run_root), **report}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    print(json.dumps(run(args.manifest), indent=2))


if __name__ == "__main__":
    main()
