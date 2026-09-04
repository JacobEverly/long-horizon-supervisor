from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from horizon_supervisor.benchmark.gate4 import (
    _checkpoint_context,
    _evaluate_with_hard_deadline,
    _write_json,
)
from horizon_supervisor.benchmark.model_catalog import ModelSpec, load_model_catalog
from horizon_supervisor.benchmark.tasks import starter_dir

STATE_ARMS = ("cold_restart", "dirty_continuation", "clean_rollback")


def _index_runs(runs: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(run["model_id"], run["task_id"]): run for run in runs}


def analyze_state_arms(
    *,
    task_ids: list[str],
    models: list[ModelSpec],
    cold_runs: list[dict[str, Any]],
    dirty_runs: list[dict[str, Any]],
    rollback_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    indexes = {
        "cold_restart": _index_runs(cold_runs),
        "dirty_continuation": _index_runs(dirty_runs),
        "clean_rollback": _index_runs(rollback_runs),
    }
    rows: list[dict[str, Any]] = []
    for task_id in task_ids:
        for model in models:
            key = (model.model_id, task_id)
            arm_runs = {arm: indexes[arm][key] for arm in STATE_ARMS}
            cold = bool(arm_runs["cold_restart"]["success"])
            dirty = bool(arm_runs["dirty_continuation"]["success"])
            rollback = bool(arm_runs["clean_rollback"]["success"])
            successful_arms = [arm for arm in STATE_ARMS if arm_runs[arm]["success"]]
            rows.append(
                {
                    "task_id": task_id,
                    "model_id": model.model_id,
                    "model_label": model.label,
                    "cold_restart_success": cold,
                    "dirty_continuation_success": dirty,
                    "clean_rollback_success": rollback,
                    "clean_restart_advantage": cold and not dirty,
                    # Same handoff, different workspace: this is the state-only comparison.
                    "dirty_state_penalty": rollback and not dirty,
                    "rollback_rescue": rollback and not dirty,
                    "handoff_value": rollback and not cold,
                    "handoff_harm": cold and not rollback,
                    "successful_arms": successful_arms,
                    "incremental_cost_usd": {
                        arm: float(arm_runs[arm]["cost_usd"]) for arm in STATE_ARMS
                    },
                }
            )

    arm_summary = {}
    for arm in STATE_ARMS:
        success_key = f"{arm}_success"
        arm_summary[arm] = {
            "passed": sum(bool(row[success_key]) for row in rows),
            "runs": len(rows),
            "completion_rate": (
                sum(bool(row[success_key]) for row in rows) / len(rows) if rows else 0.0
            ),
            "total_incremental_cost_usd": sum(
                row["incremental_cost_usd"][arm] for row in rows
            ),
        }
    return {
        "comparison_count": len(rows),
        "arms": arm_summary,
        "clean_restart_advantages": sum(row["clean_restart_advantage"] for row in rows),
        "dirty_state_penalties": sum(row["dirty_state_penalty"] for row in rows),
        "rollback_rescues": sum(row["rollback_rescue"] for row in rows),
        "handoff_value_cases": sum(row["handoff_value"] for row in rows),
        "handoff_harm_cases": sum(row["handoff_harm"] for row in rows),
        "rows": rows,
    }


def render_gate5_report(report: dict[str, Any]) -> str:
    analysis = report["analysis"]
    lines = [
        "# Gate 5: external-state recovery",
        "",
        f"- Status: {report['status']}",
        f"- Source: `{report['source_gate4_root']}`",
        f"- Failed tasks: {report['task_count']}",
        f"- Models: {report['model_count']}",
        f"- New paid rollback runs: {len(report['rollback_runs'])}",
        f"- New estimated spend: ${report['estimated_spend_usd']:.6f}",
        f"- Gate 5 ceiling: ${report['authorized_budget_usd']:.2f}",
        "",
        "## Matched state arms",
        "",
        "| Arm | State | Prior-attempt handoff | Passed | Incremental cost |",
        "|---|---|:---:|---:|---:|",
    ]
    descriptions = {
        "cold_restart": ("clean starter", "no"),
        "dirty_continuation": ("Qwen's edited workspace", "yes"),
        "clean_rollback": ("clean starter", "yes"),
    }
    for arm in STATE_ARMS:
        row = analysis["arms"][arm]
        state, handoff = descriptions[arm]
        lines.append(
            f"| {arm.replace('_', ' ')} | {state} | {handoff} | "
            f"{row['passed']}/{row['runs']} | ${row['total_incremental_cost_usd']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Model/task matrix",
            "",
            "| Task | Model | Cold | Dirty | Rollback | State penalty |",
            "|---|---|:---:|:---:|:---:|:---:|",
        ]
    )
    for row in analysis["rows"]:
        lines.append(
            f"| {row['task_id']} | {row['model_label']} | "
            f"{'pass' if row['cold_restart_success'] else 'fail'} | "
            f"{'pass' if row['dirty_continuation_success'] else 'fail'} | "
            f"{'pass' if row['clean_rollback_success'] else 'fail'} | "
            f"{'yes' if row['dirty_state_penalty'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Decision evidence",
            "",
            f"- Clean-restart advantages over dirty continuation: "
            f"{analysis['clean_restart_advantages']}.",
            f"- State-only rollback rescues over dirty continuation: "
            f"{analysis['rollback_rescues']}.",
            f"- Cases where retaining the handoff helped over cold restart: "
            f"{analysis['handoff_value_cases']}.",
            f"- Cases where the handoff hurt versus cold restart: "
            f"{analysis['handoff_harm_cases']}.",
            "",
            "Cold and dirty outcomes are reused from Gate 4; only the missing clean-rollback "
            "arm is newly sampled. The rollback point is the pre-attempt clean starter because "
            "Gate 4 did not capture per-turn workspaces. Future runs should preserve verified "
            "turn-level checkpoints so the same method can evaluate finer rollback points.",
            "",
        ]
    )
    return "\n".join(lines)


async def run_state_recovery_gate(
    artifacts_root: Path,
    *,
    source_gate4_root: Path,
    budget_usd: float = 2.0,
    api_key_var: str = "OPENROUTER_API_KEY",
    per_run_cap_usd: float = 0.75,
    resume_root: Path | None = None,
) -> dict[str, Any]:
    if budget_usd <= 0 or budget_usd > 5:
        raise ValueError("Gate 5 requires a positive budget no greater than $5")
    if not os.getenv(api_key_var):
        raise RuntimeError(f"{api_key_var} is not set")

    source_gate4_root = source_gate4_root.resolve()
    source_report_path = source_gate4_root / "gate4-report.json"
    if not source_report_path.exists():
        raise ValueError(f"missing completed Gate 4 report: {source_report_path}")
    source = json.loads(source_report_path.read_text(encoding="utf-8"))
    if source.get("status") != "complete":
        raise ValueError("Gate 5 requires a completed Gate 4 source")

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if resume_root is None:
        root = artifacts_root / f"gate5-{timestamp}"
        root.mkdir(parents=True, exist_ok=False)
        rollback_runs: list[dict[str, Any]] = []
        infrastructure_attempts: list[dict[str, Any]] = []
        pricing_path = root / "pricing-snapshot.json"
    else:
        root = resume_root.resolve()
        partial = json.loads((root / "partial-report.json").read_text(encoding="utf-8"))
        if source_gate4_root != Path(partial["source_gate4_root"]).resolve():
            raise ValueError("resume source does not match the original Gate 4 source")
        if budget_usd > float(partial["authorized_budget_usd"]):
            raise ValueError("resume budget exceeds the original Gate 5 ceiling")
        prior = partial.get("rollback_runs", [])
        infrastructure_attempts = partial.get("infrastructure_attempts", [])
        rollback_runs = [
            run for run in prior if run.get("failure_type") != "harness_or_provider_error"
        ]
        pricing_path = root / f"pricing-snapshot-resume-{timestamp}.json"

    models = load_model_catalog(pricing_path, roster="gate4")
    source_model_ids = [model["model_id"] for model in source["models"]]
    if [model.model_id for model in models] != source_model_ids:
        raise ValueError("current Gate 5 roster does not exactly match the Gate 4 source")

    floor = min(models, key=lambda model: model.tier)
    source_baselines = source["baseline_runs"]
    source_recoveries = source["recovery_runs"]
    floor_failures = [
        run
        for run in source_baselines
        if run["model_id"] == floor.model_id and not run["success"]
    ]
    task_ids = [run["task_id"] for run in floor_failures]
    expected_keys = {(model.model_id, task_id) for task_id in task_ids for model in models}
    if {_key for _key in _index_runs(source_recoveries)} != expected_keys:
        raise ValueError("Gate 4 does not contain a complete dirty-continuation matrix")

    spent = sum(float(run["cost_usd"]) for run in rollback_runs + infrastructure_attempts)
    completed = {(run["model_id"], run["task_id"]) for run in rollback_runs}

    def save_partial(status: str = "in_progress") -> None:
        _write_json(
            root / "partial-report.json",
            {
                "gate": 5,
                "status": status,
                "source_gate4_root": str(source_gate4_root),
                "authorized_budget_usd": budget_usd,
                "estimated_spend_usd": spent,
                "rollback_runs": rollback_runs,
                "infrastructure_attempts": infrastructure_attempts,
            },
        )

    try:
        for source_failure in floor_failures:
            task_id = source_failure["task_id"]
            handoff = _checkpoint_context(source_failure).replace(
                "the exact resulting workspace is mounted here",
                "the edited workspace was rolled back to the clean pre-attempt state",
            ).replace(
                "Continue from the current files rather than restarting.",
                "Use the prior attempt's evidence, but solve against the clean files now mounted.",
            )
            for model in models:
                key = (model.model_id, task_id)
                if key in completed:
                    continue
                if spent >= budget_usd:
                    raise RuntimeError("Gate 5 global budget exhausted")
                checkpoint_id = f"{task_id}-clean-rollback-with-handoff"
                for attempt in range(2):
                    summary = await _evaluate_with_hard_deadline(
                        config_dir=root / "worker-configs",
                        run_root=root / "clean-rollback",
                        model=model,
                        task_id=task_id,
                        api_key_var=api_key_var,
                        per_run_cap_usd=min(per_run_cap_usd, budget_usd - spent),
                        workspace_seed=starter_dir(task_id),
                        handoff_context=handoff,
                        checkpoint_id=checkpoint_id,
                    )
                    summary["phase"] = "clean_rollback"
                    summary["state_arm"] = "clean_rollback"
                    spent += float(summary["cost_usd"])
                    if summary.get("failure_type") != "harness_or_provider_error":
                        break
                    summary["infrastructure_attempt"] = attempt + 1
                    infrastructure_attempts.append(summary)
                    save_partial()
                else:
                    raise RuntimeError(
                        f"repeated infrastructure failure for {model.model_id}/{task_id}"
                    )
                rollback_runs.append(summary)
                completed.add(key)
                save_partial()
    except Exception as error:
        save_partial("stopped_on_provider_or_harness_error")
        _write_json(
            root / "fatal-error.json",
            {
                "gate": 5,
                "status": "stopped_on_provider_or_harness_error",
                "estimated_spend_usd": spent,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise

    report = {
        "gate": 5,
        "status": "complete",
        "completed_at": datetime.now(UTC).isoformat(),
        "source_gate4_root": str(source_gate4_root),
        "authorized_budget_usd": budget_usd,
        "estimated_spend_usd": spent,
        "model_count": len(models),
        "task_count": len(task_ids),
        "models": [asdict(model) for model in models],
        "task_ids": task_ids,
        "rollback_runs": rollback_runs,
        "infrastructure_attempts": infrastructure_attempts,
        "infrastructure_attempt_count": len(infrastructure_attempts),
        "analysis": analyze_state_arms(
            task_ids=task_ids,
            models=models,
            cold_runs=[run for run in source_baselines if run["task_id"] in task_ids],
            dirty_runs=source_recoveries,
            rollback_runs=rollback_runs,
        ),
    }
    _write_json(root / "gate5-report.json", report)
    (root / "gate5-report.md").write_text(render_gate5_report(report), encoding="utf-8")
    save_partial("complete")
    return {"artifact_root": str(root), **report}
