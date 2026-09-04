from __future__ import annotations

import json
import os
import re
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import verifiers as vf

from horizon_supervisor.benchmark.atif import export_atif
from horizon_supervisor.benchmark.environment import LocalCodingEnv
from horizon_supervisor.benchmark.mock_client import DeterministicRepairClient
from horizon_supervisor.benchmark.model_catalog import ModelSpec, load_model_catalog
from horizon_supervisor.benchmark.tasks import BENCHMARK_TASKS

STATE_COLUMNS = [
    "trajectory",
    "workspace_dir",
    "task_id",
    "verifier_result",
    "normalized_events",
    "estimated_spend_usd",
    "budget_halted",
    "checkpoint_id",
    "guard_events",
    "handoff_recommended",
    "guard_stop_reason",
    "last_public_test_result",
    "last_verified_workspace_digest",
    "workspace_digest",
    "evidence_ledger",
    "turn_checkpoints",
]


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def summarize_output(output: dict[str, Any], model: ModelSpec) -> dict[str, Any]:
    token_usage = output.get("token_usage") or {}
    input_tokens = int(token_usage.get("input_tokens", 0))
    output_tokens = int(token_usage.get("output_tokens", 0))
    cost = input_tokens * model.input_usd_per_token + output_tokens * model.output_usd_per_token
    metrics = output.get("metrics") or {}
    error = output.get("error")
    success = float(output.get("reward", 0)) >= 0.5
    if success:
        failure_type = None
    elif error:
        failure_type = "harness_or_provider_error"
    elif output.get("budget_halted"):
        failure_type = "budget_halt"
    elif output.get("stop_condition") == "timeout_reached":
        failure_type = "rollout_timeout"
    elif output.get("stop_condition") == "max_turns_reached":
        failure_type = "turn_limit"
    elif output.get("stop_condition") == "max_total_completion_tokens_reached":
        failure_type = "completion_token_limit"
    elif metrics.get("total_tool_calls", output.get("total_tool_calls", 0)) == 0:
        failure_type = "no_tool_use"
    else:
        failure_type = "verifier_failure"
    timing = output.get("timing") or {}
    events = output.get("normalized_events") or []
    return {
        "model_id": model.model_id,
        "model_label": model.label,
        "task_id": output.get("task_id"),
        "difficulty": (output.get("info") or {}).get("difficulty"),
        "success": success,
        "reward": output.get("reward"),
        "cost_usd": cost,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "turns": int(output.get("num_turns", 0)),
        "tool_calls": int(output.get("total_tool_calls", 0)),
        "duration_seconds": float(timing.get("total", 0.0)),
        "stop_condition": output.get("stop_condition"),
        "failure_type": failure_type,
        "error": _jsonable(error) if error else None,
        "workspace": output.get("workspace_dir"),
        "verifier_result": output.get("verifier_result"),
        "checkpoint_id": output.get("checkpoint_id"),
        "guard_events": output.get("guard_events") or [],
        "handoff_recommended": bool(output.get("handoff_recommended")),
        "guard_stop_reason": output.get("guard_stop_reason"),
        "last_public_test_result": output.get("last_public_test_result"),
        "last_verified_workspace_digest": output.get("last_verified_workspace_digest"),
        "workspace_digest": output.get("workspace_digest"),
        "evidence_ledger": output.get("evidence_ledger") or [],
        "turn_checkpoints": output.get("turn_checkpoints") or [],
        "tool_sequence": [
            event.get("tool") for event in events if event.get("kind") == "tool_requested"
        ],
        "workspace_change_count": sum(
            bool(event.get("workspace_changed"))
            for event in events
            if event.get("kind") == "tool_result" and event.get("success")
        ),
        "evidence_invalidation_count": sum(
            event.get("kind") == "evidence_invalidated" for event in events
        ),
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Gate 3: pure-model Pareto seed",
        "",
        f"- Runs: {report['run_count']} ({report['model_count']} models × "
        f"{report['task_count']} tasks)",
        f"- Estimated API spend: ${report['estimated_spend_usd']:.6f}",
        f"- Authorized ceiling: ${report['authorized_budget_usd']:.2f}",
        "- Policy: fixed model for each complete run; no dynamic routing or automatic retries",
        "",
        "## Model aggregates",
        "",
        "| Model | Passed | Completion | Avg cost | Avg turns | Avg seconds | Pareto |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in report["aggregates"]:
        lines.append(
            f"| {row['label']} | {row['passed']}/{row['runs']} | "
            f"{row['completion_rate']:.0%} | ${row['average_cost_usd']:.6f} | "
            f"{row['average_turns']:.1f} | {row['average_duration_seconds']:.1f} | "
            f"{'yes' if row['pareto_efficient'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Individual runs",
            "",
            "| Model | Task | Difficulty | Result | Cost | Turns | Tools | Failure class |",
            "|---|---|---|:---:|---:|---:|---:|---|",
        ]
    )
    for run in report["runs"]:
        lines.append(
            f"| {run['model_label']} | {run['task_id']} | {run['difficulty']} | "
            f"{'pass' if run['success'] else 'fail'} | ${run['cost_usd']:.6f} | "
            f"{run['turns']} | {run['tool_calls']} | {run['failure_type'] or '—'} |"
        )
    efficient = [row["label"] for row in report["aggregates"] if row["pareto_efficient"]]
    dominated = [row["label"] for row in report["aggregates"] if not row["pareto_efficient"]]
    lines.extend(
        [
            "",
            "## Initial findings",
            "",
            f"- Observed Pareto set: {', '.join(efficient) or 'none'}.",
            f"- Observed dominated models: {', '.join(dominated) or 'none'}.",
            "- This gate validates the harness and reveals obvious capability/cost separation. "
            "Three tasks per model are not enough for a production routing threshold.",
            "- Dynamic switching should be evaluated only if fixed models show complementary "
            "success/failure patterns or meaningful cost separation.",
            "",
        ]
    )
    return "\n".join(lines)


def aggregate_results(runs: list[dict[str, Any]], models: list[ModelSpec]) -> list[dict[str, Any]]:
    aggregates = []
    for model in models:
        rows = [run for run in runs if run["model_id"] == model.model_id]
        if not rows:
            continue
        aggregates.append(
            {
                "model_id": model.model_id,
                "label": model.label,
                "role": model.role,
                "passed": sum(run["success"] for run in rows),
                "runs": len(rows),
                "completion_rate": sum(run["success"] for run in rows) / len(rows),
                "total_cost_usd": sum(run["cost_usd"] for run in rows),
                "average_cost_usd": sum(run["cost_usd"] for run in rows) / len(rows),
                "average_duration_seconds": (
                    sum(run["duration_seconds"] for run in rows) / len(rows)
                ),
                "average_turns": sum(run["turns"] for run in rows) / len(rows),
            }
        )
    for row in aggregates:
        row["pareto_efficient"] = not any(
            other["completion_rate"] >= row["completion_rate"]
            and other["average_cost_usd"] <= row["average_cost_usd"]
            and (
                other["completion_rate"] > row["completion_rate"]
                or other["average_cost_usd"] < row["average_cost_usd"]
            )
            for other in aggregates
            if other is not row
        )
    return aggregates


async def evaluate_one(
    root: Path,
    model: ModelSpec,
    task_id: str,
    api_key_var: str,
    per_run_cap_usd: float,
    *,
    workspace_seed: Path | None = None,
    handoff_context: str = "",
    handoff_evidence: list[dict[str, Any]] | None = None,
    checkpoint_id: str | None = None,
    max_turns: int = 10,
) -> tuple[dict[str, Any], dict[str, Any]]:
    run_dir = root / "runs" / _slug(model.model_id) / task_id
    env = LocalCodingEnv(
        artifacts_dir=run_dir,
        max_turns=max_turns,
        task_ids=[task_id],
        input_usd_per_token=model.input_usd_per_token,
        output_usd_per_token=model.output_usd_per_token,
        per_run_cap_usd=per_run_cap_usd,
        max_completion_tokens_per_call=2_048,
        workspace_seeds={task_id: workspace_seed} if workspace_seed else None,
        handoff_contexts={task_id: handoff_context} if handoff_context else None,
        handoff_evidence={task_id: handoff_evidence} if handoff_evidence else None,
        checkpoint_ids={task_id: checkpoint_id} if checkpoint_id else None,
    )
    client_config = vf.ClientConfig(
        client_type="openai_chat_completions",
        api_key_var=api_key_var,
        api_base_url="https://openrouter.ai/api/v1",
        timeout=300,
        connect_timeout=10,
        max_retries=0,
        extra_headers={
            "HTTP-Referer": "https://github.com/JacobEverly",
            "X-Title": "Long Horizon Supervisor",
        },
    )
    sampling_args: dict[str, Any] = {"max_tokens": 2_048, "temperature": 0}
    if model.reasoning_effort:
        sampling_args["reasoning_effort"] = model.reasoning_effort
    result = await env.evaluate(
        client=client_config,
        model=model.model_id,
        sampling_args=sampling_args,
        max_concurrent=1,
        max_retries=0,
        state_columns=STATE_COLUMNS,
    )
    output = result["outputs"][0]
    summary = summarize_output(output, model)
    atif = export_atif(output, model)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "trajectory.atif.json").write_text(json.dumps(atif, indent=2), encoding="utf-8")
    (run_dir / "verifiers-output.json").write_text(
        json.dumps(_jsonable(output), indent=2, default=str), encoding="utf-8"
    )
    return summary, output


async def run_mock(root: Path) -> dict[str, Any]:
    mock_model = ModelSpec("mock/deterministic-repair", "Mock repair", "gate-1", 0, 0, 1_000_000)
    env = LocalCodingEnv(root / "runs", max_turns=8)
    result = await env.evaluate(
        client=DeterministicRepairClient(),
        model=mock_model.model_id,
        sampling_args={"max_tokens": 512},
        max_concurrent=1,
        max_retries=0,
        state_columns=STATE_COLUMNS,
    )
    summaries = []
    for output in result["outputs"]:
        summary = summarize_output(output, mock_model)
        summaries.append(summary)
        run_dir = root / "exported" / output["task_id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (run_dir / "trajectory.atif.json").write_text(
            json.dumps(export_atif(output, mock_model), indent=2), encoding="utf-8"
        )
    report = {"gate": 1, "runs": summaries, "all_passed": all(row["success"] for row in summaries)}
    (root / "gate1-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


async def run_tournament(
    artifacts_root: Path,
    budget_usd: float = 50.0,
    api_key_var: str = "OPENROUTER_API_KEY",
    per_run_cap_usd: float = 3.5,
    resume_root: Path | None = None,
) -> dict[str, Any]:
    if budget_usd > 50:
        raise ValueError("this experiment is not authorized above $50")
    if not os.getenv(api_key_var):
        raise RuntimeError(f"{api_key_var} is not set")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if resume_root is None:
        root = artifacts_root / f"gate3-{timestamp}"
        root.mkdir(parents=True, exist_ok=False)
        pricing_path = root / "pricing-snapshot.json"
        runs: list[dict[str, Any]] = []
    else:
        root = resume_root.resolve()
        partial_path = root / "partial-report.json"
        if not partial_path.exists():
            raise ValueError(f"cannot resume without {partial_path}")
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        previous_budget = float(partial["authorized_budget_usd"])
        if budget_usd > previous_budget:
            raise ValueError(
                f"resume budget ${budget_usd:.2f} exceeds original ${previous_budget:.2f}"
            )
        # Infrastructure failures are retryable; model/verifier outcomes are not.
        runs = [
            run
            for run in partial.get("runs", [])
            if run.get("failure_type") != "harness_or_provider_error"
        ]
        fatal_path = root / "fatal-error.json"
        if fatal_path.exists():
            fatal_path.rename(root / f"fatal-error-{timestamp}.json")
        pricing_path = root / f"pricing-snapshot-resume-{timestamp}.json"
    models = load_model_catalog(pricing_path, roster="gate3")
    spent = sum(float(run["cost_usd"]) for run in runs)
    completed = {(run["model_id"], run["task_id"]) for run in runs}

    # The first (easy) task through each endpoint is Gate 2 and also the easy row of Gate 3.
    ordered_tasks = [task.task_id for task in BENCHMARK_TASKS]
    for task_id in ordered_tasks:
        for model in models:
            if (model.model_id, task_id) in completed:
                continue
            remaining = budget_usd - spent
            if remaining <= 0:
                raise RuntimeError("global experiment budget exhausted")
            run_cap = min(per_run_cap_usd, remaining)
            try:
                summary, _ = await evaluate_one(
                    root, model, task_id, api_key_var=api_key_var, per_run_cap_usd=run_cap
                )
            except Exception as error:
                failure = {
                    "gate": 3,
                    "status": "stopped_on_provider_or_harness_error",
                    "authorized_budget_usd": budget_usd,
                    "estimated_spend_usd": spent,
                    "failed_model": model.model_id,
                    "failed_task": task_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "completed_runs": runs,
                }
                (root / "fatal-error.json").write_text(
                    json.dumps(failure, indent=2), encoding="utf-8"
                )
                raise
            runs.append(summary)
            completed.add((model.model_id, task_id))
            spent += summary["cost_usd"]
            partial = {
                "gate": 3,
                "status": "in_progress",
                "authorized_budget_usd": budget_usd,
                "estimated_spend_usd": spent,
                "runs": runs,
            }
            (root / "partial-report.json").write_text(
                json.dumps(partial, indent=2), encoding="utf-8"
            )
            if summary["failure_type"] == "harness_or_provider_error":
                raise RuntimeError(
                    f"provider/harness failure for {model.model_id}; stopped before further spend"
                )

    report = {
        "gate": 3,
        "status": "complete",
        "completed_at": datetime.now(UTC).isoformat(),
        "authorized_budget_usd": budget_usd,
        "estimated_spend_usd": spent,
        "task_count": len(ordered_tasks),
        "model_count": len(models),
        "run_count": len(runs),
        "models": [asdict(model) for model in models],
        "runs": runs,
        "aggregates": aggregate_results(runs, models),
    }
    (root / "gate3-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (root / "gate3-report.md").write_text(render_markdown_report(report), encoding="utf-8")
    return {"artifact_root": str(root), **report}
