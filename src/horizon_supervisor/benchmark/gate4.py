from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from horizon_supervisor.benchmark.model_catalog import ModelSpec, load_model_catalog
from horizon_supervisor.benchmark.runner import _slug, aggregate_results
from horizon_supervisor.benchmark.tasks import hidden_tests_dir
from horizon_supervisor.benchmark.tools import run_hidden_tests

RECOVERY_TASK_IDS = [
    "ttl-cache-semantics",
    "feature-dependency-plan",
    "retry-policy",
    "idempotency-store",
    "webhook-reducer",
    "weighted-quota",
]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def _run_key(run: dict[str, Any]) -> tuple[str, str, str, str | None]:
    return (
        run["phase"],
        run["model_id"],
        run["task_id"],
        run.get("checkpoint_id"),
    )


def _checkpoint_context(source: dict[str, Any]) -> str:
    public = source.get("last_public_test_result") or {}
    output = str(public.get("output", ""))[-1_500:]
    guard = source.get("guard_stop_reason") or "final verifier failure"
    return (
        "\n\nHandoff context: a compact model already attempted this task, and the exact "
        "resulting workspace is mounted here. The final verifier did not pass. Continue from "
        "the current files rather than restarting. The deterministic supervisor's signal was "
        f"`{guard}`. The last public-test result was passed={public.get('passed')}. "
        "Public tests are incomplete evidence, so inspect the implementation against every "
        "requirement, make any necessary repair, and rerun them after the final edit."
        + (f"\n\nLast public-test output:\n{output}" if output else "")
    )


def _model_rows(runs: list[dict[str, Any]], models: list[ModelSpec]) -> list[dict[str, Any]]:
    return aggregate_results(runs, models)


async def _evaluate_with_hard_deadline(
    *,
    config_dir: Path,
    run_root: Path,
    model: ModelSpec,
    task_id: str,
    api_key_var: str,
    per_run_cap_usd: float,
    workspace_seed: Path | None,
    handoff_context: str,
    handoff_evidence: list[dict[str, Any]] | None = None,
    checkpoint_id: str | None,
    hard_timeout_seconds: float = 330,
) -> dict[str, Any]:
    monotonic_start = time.monotonic()
    worker_id = "-".join(
        [task_id, _slug(model.model_id), _slug(checkpoint_id or "starter")]
    )
    config_path = config_dir / f"{worker_id}.json"
    result_path = config_dir / f"{worker_id}-result.json"
    result_path.unlink(missing_ok=True)
    _write_json(
        config_path,
        {
            "run_root": str(run_root),
            "model": asdict(model),
            "task_id": task_id,
            "api_key_var": api_key_var,
            "per_run_cap_usd": per_run_cap_usd,
            "workspace_seed": str(workspace_seed) if workspace_seed else None,
            "handoff_context": handoff_context,
            "handoff_evidence": handoff_evidence or [],
            "checkpoint_id": checkpoint_id,
            "max_turns": 10,
            "hard_timeout_seconds": hard_timeout_seconds,
            "result_path": str(result_path),
        },
    )
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "horizon_supervisor.benchmark.gate4_worker",
        str(config_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    def timeout_summary() -> dict[str, Any]:
        workspace_root = run_root / "runs" / _slug(model.model_id) / task_id / "workspaces"
        candidates = sorted(
            workspace_root.glob(f"{task_id}/*"),
            key=lambda path: path.stat().st_mtime,
        )
        workspace = candidates[-1] if candidates else None
        verifier = (
            run_hidden_tests(str(workspace), hidden_tests_dir(task_id))
            if workspace is not None
            else {"passed": False, "returncode": -1, "output": "workspace unavailable"}
        )
        return {
            "model_id": model.model_id,
            "model_label": model.label,
            "task_id": task_id,
            "difficulty": None,
            "success": bool(verifier["passed"]),
            "reward": 1.0 if verifier["passed"] else 0.0,
            "cost_usd": per_run_cap_usd,
            "cost_is_timeout_reservation": True,
            "input_tokens": 0,
            "output_tokens": 0,
            "turns": 0,
            "tool_calls": 0,
            "duration_seconds": hard_timeout_seconds,
            "stop_condition": "hard_process_timeout",
            "failure_type": None if verifier["passed"] else "hard_process_timeout",
            "error": None,
            "workspace": str(workspace) if workspace else None,
            "verifier_result": verifier,
            "checkpoint_id": checkpoint_id,
            "guard_events": [
                {
                    "kind": "supervisor_guard",
                    "reason": "hard_process_timeout",
                    "hard_stop": True,
                    "details": {"limit_seconds": hard_timeout_seconds},
                }
            ],
            "handoff_recommended": True,
            "guard_stop_reason": "hard_process_timeout",
            "last_public_test_result": None,
            "last_verified_workspace_digest": None,
            "workspace_digest": None,
            "evidence_ledger": [],
            "turn_checkpoints": [],
            "deadline_met": False,
        }

    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=hard_timeout_seconds + 15
        )
    except TimeoutError:
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except TimeoutError:
            process.kill()
            await process.wait()
        summary = timeout_summary()
        summary["process_elapsed_seconds"] = time.monotonic() - monotonic_start
        return summary
    if process.returncode == 124:
        summary = timeout_summary()
        summary["process_elapsed_seconds"] = time.monotonic() - monotonic_start
        return summary
    if process.returncode != 0:
        message = stderr.decode("utf-8", errors="replace")[-4_000:]
        raise RuntimeError(
            f"Gate 4 worker failed for {model.model_id}/{task_id}: {message}"
        )
    if not result_path.exists():
        output = stdout.decode("utf-8", errors="replace")[-2_000:]
        raise RuntimeError(f"Gate 4 worker produced no result: {output}")
    summary = json.loads(result_path.read_text(encoding="utf-8"))
    summary["deadline_met"] = True
    summary["process_elapsed_seconds"] = time.monotonic() - monotonic_start
    return summary


def _strategy_analysis(
    baseline_runs: list[dict[str, Any]],
    recovery_runs: list[dict[str, Any]],
    models: list[ModelSpec],
) -> dict[str, Any]:
    floor = min(models, key=lambda model: model.tier)
    tasks = RECOVERY_TASK_IDS
    baseline_by_key = {
        (run["model_id"], run["task_id"]): run for run in baseline_runs
    }
    recovery_by_task: dict[str, list[dict[str, Any]]] = {}
    for run in recovery_runs:
        recovery_by_task.setdefault(run["task_id"], []).append(run)

    floor_successes = 0
    observed_recovery_successes = 0
    switch_rescues = 0
    switch_exclusive_rescues = 0
    safe_stepdowns = 0
    retrospective_cost = 0.0
    task_rows = []
    for task_id in tasks:
        floor_run = baseline_by_key.get((floor.model_id, task_id))
        if floor_run is None:
            continue
        retrospective_cost += floor_run["cost_usd"]
        branches = recovery_by_task.get(task_id, [])
        floor_passed = bool(floor_run["success"])
        floor_successes += int(floor_passed)
        frontier_passed = any(
            baseline_by_key.get((model.model_id, task_id), {}).get("success", False)
            for model in models
            if model.tier >= 3
        )
        safe_stepdowns += int(floor_passed and frontier_passed)

        successful_branches = [run for run in branches if run["success"]]
        non_floor_successes = [
            run for run in successful_branches if run["model_id"] != floor.model_id
        ]
        floor_continuation_passed = any(
            run["success"] and run["model_id"] == floor.model_id for run in branches
        )
        recovered = floor_passed or bool(successful_branches)
        observed_recovery_successes += int(recovered)
        switched = not floor_passed and bool(non_floor_successes)
        exclusive = switched and not floor_continuation_passed
        switch_rescues += int(switched)
        switch_exclusive_rescues += int(exclusive)
        if not floor_passed and successful_branches:
            retrospective_cost += min(run["cost_usd"] for run in successful_branches)
        task_rows.append(
            {
                "task_id": task_id,
                "floor_passed": floor_passed,
                "frontier_passed_from_starter": frontier_passed,
                "recovery_checkpoint_created": bool(branches),
                "floor_continuation_passed": floor_continuation_passed,
                "rescued_by": [run["model_label"] for run in non_floor_successes],
                "switch_rescue": switched,
                "switch_exclusive_rescue": exclusive,
                "retrospective_success": recovered,
            }
        )

    fixed = []
    for model in models:
        rows = [run for run in baseline_runs if run["model_id"] == model.model_id]
        fixed.append(
            {
                "model_id": model.model_id,
                "model_label": model.label,
                "passed": sum(run["success"] for run in rows),
                "runs": len(rows),
                "estimated_cost_usd": sum(run["cost_usd"] for run in rows),
            }
        )
    return {
        "floor_model_id": floor.model_id,
        "floor_model_label": floor.label,
        "task_count": len(task_rows),
        "floor_only_passed": floor_successes,
        "safe_stepdown_tasks": safe_stepdowns,
        "observed_floor_then_recovery_passed": observed_recovery_successes,
        "switch_rescues": switch_rescues,
        "switch_exclusive_rescues": switch_exclusive_rescues,
        "retrospective_oracle_route_cost_usd": retrospective_cost,
        "fixed_model_baselines": fixed,
        "tasks": task_rows,
    }


def render_gate4_report(report: dict[str, Any]) -> str:
    strategy = report["strategy"]
    lines = [
        "# Gate 4: deterministic guards and recovery headroom",
        "",
        f"- Status: {report['status']}",
        f"- Current models: {report['model_count']}",
        f"- Medium/hard tasks: {report['task_count']}",
        f"- Baseline runs: {len(report['baseline_runs'])}",
        f"- Identical-checkpoint continuations: {len(report['recovery_runs'])}",
        f"- Estimated experiment spend: ${report['estimated_spend_usd']:.6f}",
        f"- Experiment ceiling: ${report['authorized_budget_usd']:.2f}",
        "",
        "## Current fixed-model baselines",
        "",
        "| Model | Passed | Completion | Avg cost | Pareto |",
        "|---|---:|---:|---:|:---:|",
    ]
    for row in report["baseline_aggregates"]:
        lines.append(
            f"| {row['label']} | {row['passed']}/{row['runs']} | "
            f"{row['completion_rate']:.0%} | ${row['average_cost_usd']:.6f} | "
            f"{'yes' if row['pareto_efficient'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Strategy evidence",
            "",
            f"- Compact floor alone: {strategy['floor_only_passed']}/{strategy['task_count']}.",
            "- Compact floor followed by the cheapest observed successful continuation "
            f"(retrospective oracle): {strategy['observed_floor_then_recovery_passed']}/"
            f"{strategy['task_count']} at ${strategy['retrospective_oracle_route_cost_usd']:.6f}.",
            f"- Tasks rescued by another model: {strategy['switch_rescues']}.",
            "- Switch-exclusive rescues, where a fresh compact continuation still failed: "
            f"{strategy['switch_exclusive_rescues']}.",
            f"- Observed safe compact step-downs: {strategy['safe_stepdown_tasks']}.",
            "",
            "| Task | Floor | Floor continuation | Other-model rescue | Exclusive switch value |",
            "|---|:---:|:---:|---|:---:|",
        ]
    )
    for row in strategy["tasks"]:
        lines.append(
            f"| {row['task_id']} | {'pass' if row['floor_passed'] else 'fail'} | "
            f"{'pass' if row['floor_continuation_passed'] else 'fail'} | "
            f"{', '.join(row['rescued_by']) or '—'} | "
            f"{'yes' if row['switch_exclusive_rescue'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "The recovery route above is an empirical oracle, not a deployable policy: it picks "
            "the cheapest successful branch after observing outcomes. Its purpose is to measure "
            "headroom before training a selector.",
            "",
        ]
    )
    return "\n".join(lines)


async def run_recovery_gate(
    artifacts_root: Path,
    *,
    budget_usd: float = 5.0,
    api_key_var: str = "OPENROUTER_API_KEY",
    per_run_cap_usd: float = 0.75,
    resume_root: Path | None = None,
) -> dict[str, Any]:
    if budget_usd <= 0 or budget_usd > 10:
        raise ValueError("Gate 4 requires a positive budget no greater than $10")
    if not os.getenv(api_key_var):
        raise RuntimeError(f"{api_key_var} is not set")

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if resume_root is None:
        root = artifacts_root / f"gate4-{timestamp}"
        root.mkdir(parents=True, exist_ok=False)
        baseline_runs: list[dict[str, Any]] = []
        recovery_runs: list[dict[str, Any]] = []
        infrastructure_attempts: list[dict[str, Any]] = []
        pricing_path = root / "pricing-snapshot.json"
    else:
        root = resume_root.resolve()
        partial = json.loads((root / "partial-report.json").read_text(encoding="utf-8"))
        if budget_usd > float(partial["authorized_budget_usd"]):
            raise ValueError("resume budget exceeds the original Gate 4 ceiling")
        prior_baselines = partial.get("baseline_runs", [])
        prior_recoveries = partial.get("recovery_runs", [])
        infrastructure_attempts = partial.get("infrastructure_attempts", [])
        newly_archived = [
            run
            for run in prior_baselines + prior_recoveries
            if run.get("failure_type") == "harness_or_provider_error"
        ]
        archived_keys = {
            (
                run.get("phase"),
                run.get("model_id"),
                run.get("task_id"),
                run.get("checkpoint_id"),
                run.get("cost_usd"),
            )
            for run in infrastructure_attempts
        }
        infrastructure_attempts.extend(
            run
            for run in newly_archived
            if (
                run.get("phase"),
                run.get("model_id"),
                run.get("task_id"),
                run.get("checkpoint_id"),
                run.get("cost_usd"),
            )
            not in archived_keys
        )
        baseline_runs = [
            run
            for run in prior_baselines
            if run.get("failure_type") != "harness_or_provider_error"
        ]
        recovery_runs = [
            run
            for run in prior_recoveries
            if run.get("failure_type") != "harness_or_provider_error"
        ]
        pricing_path = root / f"pricing-snapshot-resume-{timestamp}.json"

    models = load_model_catalog(pricing_path, roster="gate4")
    floor = min(models, key=lambda model: model.tier)
    spent = sum(
        run["cost_usd"]
        for run in baseline_runs + recovery_runs + infrastructure_attempts
    )
    completed = {_run_key(run) for run in baseline_runs + recovery_runs}

    def save_partial(status: str = "in_progress") -> None:
        _write_json(
            root / "partial-report.json",
            {
                "gate": 4,
                "status": status,
                "authorized_budget_usd": budget_usd,
                "estimated_spend_usd": spent,
                "baseline_runs": baseline_runs,
                "recovery_runs": recovery_runs,
                "infrastructure_attempts": infrastructure_attempts,
            },
        )

    async def run_and_record(
        *,
        phase: str,
        model: ModelSpec,
        task_id: str,
        run_root: Path,
        workspace_seed: Path | None = None,
        handoff_context: str = "",
        checkpoint_id: str | None = None,
    ) -> dict[str, Any]:
        nonlocal spent
        key = (phase, model.model_id, task_id, checkpoint_id)
        existing = next(
            (run for run in baseline_runs + recovery_runs if _run_key(run) == key), None
        )
        if existing is not None:
            return existing
        remaining = budget_usd - spent
        if remaining <= 0:
            raise RuntimeError("Gate 4 global budget exhausted")
        for attempt in range(2):
            summary = await _evaluate_with_hard_deadline(
                config_dir=root / "worker-configs",
                run_root=run_root,
                model=model,
                task_id=task_id,
                api_key_var=api_key_var,
                per_run_cap_usd=min(per_run_cap_usd, budget_usd - spent),
                workspace_seed=workspace_seed,
                handoff_context=handoff_context,
                checkpoint_id=checkpoint_id,
            )
            summary["phase"] = phase
            spent += summary["cost_usd"]
            if summary.get("failure_type") != "harness_or_provider_error":
                break
            summary["infrastructure_attempt"] = attempt + 1
            infrastructure_attempts.append(summary)
            save_partial()
        else:
            raise RuntimeError(
                f"repeated infrastructure failure for {model.model_id}/{task_id}"
            )
        if phase == "baseline":
            baseline_runs.append(summary)
        else:
            recovery_runs.append(summary)
        completed.add(key)
        save_partial()
        return summary

    try:
        for task_id in RECOVERY_TASK_IDS:
            for model in models:
                await run_and_record(
                    phase="baseline",
                    model=model,
                    task_id=task_id,
                    run_root=root / "baseline",
                )

        floor_failures = [
            run
            for run in baseline_runs
            if run["model_id"] == floor.model_id and not run["success"]
        ]
        for source in floor_failures:
            task_id = source["task_id"]
            checkpoint_id = f"{task_id}-after-{floor.model_id.replace('/', '_')}"
            checkpoint_dir = root / "checkpoints" / checkpoint_id
            checkpoint_workspace = checkpoint_dir / "workspace"
            if not checkpoint_workspace.exists():
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                shutil.copytree(Path(source["workspace"]), checkpoint_workspace)
                _write_json(
                    checkpoint_dir / "checkpoint.json",
                    {
                        "checkpoint_id": checkpoint_id,
                        "task_id": task_id,
                        "source_model_id": floor.model_id,
                        "workspace_digest": source.get("workspace_digest"),
                        "guard_stop_reason": source.get("guard_stop_reason"),
                        "last_public_test_result": source.get("last_public_test_result"),
                        "hidden_verifier_details_exposed_to_branches": False,
                    },
                )
            context = _checkpoint_context(source)
            for model in models:
                await run_and_record(
                    phase="recovery",
                    model=model,
                    task_id=task_id,
                    run_root=root / "recovery" / checkpoint_id,
                    workspace_seed=checkpoint_workspace,
                    handoff_context=context,
                    checkpoint_id=checkpoint_id,
                )
    except Exception as error:
        save_partial("stopped_on_provider_or_harness_error")
        _write_json(
            root / "fatal-error.json",
            {
                "gate": 4,
                "status": "stopped_on_provider_or_harness_error",
                "estimated_spend_usd": spent,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise

    report = {
        "gate": 4,
        "status": "complete",
        "completed_at": datetime.now(UTC).isoformat(),
        "authorized_budget_usd": budget_usd,
        "estimated_spend_usd": spent,
        "model_count": len(models),
        "task_count": len(RECOVERY_TASK_IDS),
        "models": [asdict(model) for model in models],
        "baseline_runs": baseline_runs,
        "recovery_runs": recovery_runs,
        "infrastructure_attempts": infrastructure_attempts,
        "infrastructure_attempt_count": len(infrastructure_attempts),
        "timing_audit": {
            "wall_clock_anomaly_detected": any(
                run.get("duration_seconds", 0) > 330
                for run in baseline_runs + recovery_runs
            ),
            "duration_rankings_usable": False,
            "reason": (
                "Verifiers used wall-clock timestamps while the host clock changed during runs; "
                "process deadlines used an independent monotonic/OS timer."
            ),
        },
        "baseline_aggregates": _model_rows(baseline_runs, models),
        "strategy": _strategy_analysis(baseline_runs, recovery_runs, models),
    }
    _write_json(root / "gate4-report.json", report)
    (root / "gate4-report.md").write_text(render_gate4_report(report), encoding="utf-8")
    save_partial("complete")
    return {"artifact_root": str(root), **report}
