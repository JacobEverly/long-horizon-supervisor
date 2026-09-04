from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from horizon_supervisor.benchmark.environment import (
    render_state_bound_handoff,
    workspace_digest,
)
from horizon_supervisor.benchmark.gate4 import (
    _checkpoint_context,
    _evaluate_with_hard_deadline,
    _write_json,
)
from horizon_supervisor.benchmark.model_catalog import load_model_catalog
from horizon_supervisor.benchmark.tasks import starter_dir

TASK_ID = "idempotency-store"
MODEL_ID = "moonshotai/kimi-k3"
ARMS = (
    "neutral_clean",
    "stale_clean",
    "digest_aware_clean",
    "digest_aware_dirty",
)


def _stale_clean_handoff(source: dict[str, Any]) -> str:
    return (
        _checkpoint_context(source)
        .replace(
            "the exact resulting workspace is mounted here",
            "the edited workspace was rolled back to the clean pre-attempt state",
        )
        .replace(
            "Continue from the current files rather than restarting.",
            "Use the prior attempt's evidence, but solve against the clean files now mounted.",
        )
    )


def _public_evidence(source: dict[str, Any]) -> dict[str, Any]:
    public = source.get("last_public_test_result") or {}
    return {
        "kind": "public_tests",
        "source": "prior_qwen_attempt",
        "source_model_id": source["model_id"],
        "workspace_digest": source["workspace_digest"],
        "passed": bool(public.get("passed")),
        "returncode": public.get("returncode"),
        "observed_at": public.get("observed_at"),
    }


def analyze_gate6(runs: list[dict[str, Any]]) -> dict[str, Any]:
    arms: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        rows = [run for run in runs if run["arm"] == arm]
        passed = sum(bool(run["success"]) for run in rows)
        arms[arm] = {
            "passed": passed,
            "runs": len(rows),
            "completion_rate": passed / len(rows) if rows else 0.0,
            "total_cost_usd": sum(float(run["cost_usd"]) for run in rows),
            "runs_with_mutation": sum(run.get("workspace_change_count", 0) > 0 for run in rows),
            "runs_with_fresh_tests": sum(
                bool(
                    (run.get("last_public_test_result") or {}).get(
                        "fresh_in_current_run"
                    )
                )
                for run in rows
            ),
            "runs_with_unverified_stop": sum(
                any(event.get("reason") == "unverified_stop" for event in run["guard_events"])
                for run in rows
            ),
        }

    def rate(arm: str) -> float:
        return float(arms[arm]["completion_rate"])

    return {
        "arms": arms,
        "contrasts_percentage_points": {
            "stale_vs_neutral": 100 * (rate("stale_clean") - rate("neutral_clean")),
            "digest_clean_vs_neutral": 100
            * (rate("digest_aware_clean") - rate("neutral_clean")),
            "digest_label_vs_stale": 100
            * (rate("digest_aware_clean") - rate("stale_clean")),
            "dirty_vs_clean_digest_aware": 100
            * (rate("digest_aware_dirty") - rate("digest_aware_clean")),
        },
    }


def render_gate6_report(report: dict[str, Any]) -> str:
    analysis = report["analysis"]
    descriptions = {
        "neutral_clean": ("clean", "none"),
        "stale_clean": ("clean", "legacy, unqualified pass output"),
        "digest_aware_clean": ("clean", "prior evidence labeled stale"),
        "digest_aware_dirty": ("dirty", "prior evidence labeled matched, not fresh"),
    }
    lines = [
        "# Gate 6: state-bound handoff replication",
        "",
        f"- Status: {report['status']}",
        f"- Model: {report['model']['label']} (`{report['model']['model_id']}`)",
        f"- Task: `{TASK_ID}`",
        f"- Replications per arm: {report['replications_per_arm']}",
        f"- Paid runs: {len(report['runs'])}",
        f"- Estimated spend: ${report['estimated_spend_usd']:.6f}",
        f"- Ceiling: ${report['authorized_budget_usd']:.2f}",
        "",
        "## Arms",
        "",
        "| Arm | Workspace | Handoff | Passed | Mutated | Fresh tests | Unverified stop | Cost |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = analysis["arms"][arm]
        state, handoff = descriptions[arm]
        lines.append(
            f"| {arm.replace('_', ' ')} | {state} | {handoff} | "
            f"{row['passed']}/{row['runs']} | {row['runs_with_mutation']}/{row['runs']} | "
            f"{row['runs_with_fresh_tests']}/{row['runs']} | "
            f"{row['runs_with_unverified_stop']}/{row['runs']} | "
            f"${row['total_cost_usd']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Replications",
            "",
            "| Arm | Rep | Result | Cost | Tools | Checkpoints |",
            "|---|---:|:---:|---:|---:|---:|",
        ]
    )
    for run in sorted(report["runs"], key=lambda row: (row["arm"], row["replication"])):
        lines.append(
            f"| {run['arm']} | {run['replication']} | "
            f"{'pass' if run['success'] else 'fail'} | ${run['cost_usd']:.6f} | "
            f"{run['tool_calls']} | {len(run.get('turn_checkpoints', []))} |"
        )
    contrasts = analysis["contrasts_percentage_points"]
    lines.extend(
        [
            "",
            "## Directional contrasts",
            "",
            f"- Legacy stale handoff versus neutral: {contrasts['stale_vs_neutral']:+.1f} pp.",
            f"- Digest-aware clean handoff versus neutral: "
            f"{contrasts['digest_clean_vs_neutral']:+.1f} pp.",
            f"- Digest-aware labeling versus legacy stale handoff: "
            f"{contrasts['digest_label_vs_stale']:+.1f} pp.",
            f"- Dirty versus clean under digest-aware handoff: "
            f"{contrasts['dirty_vs_clean_digest_aware']:+.1f} pp.",
            "",
            "With only a few replications per arm, these are mechanism checks rather than "
            "calibrated production estimates or significance claims.",
            "",
        ]
    )
    return "\n".join(lines)


async def run_gate6(
    artifacts_root: Path,
    *,
    source_gate4_root: Path,
    replications_per_arm: int = 3,
    budget_usd: float = 3.0,
    api_key_var: str = "OPENROUTER_API_KEY",
    per_run_cap_usd: float = 0.5,
    resume_root: Path | None = None,
) -> dict[str, Any]:
    if replications_per_arm <= 0 or replications_per_arm > 5:
        raise ValueError("Gate 6 supports one to five replications per arm")
    if budget_usd <= 0 or budget_usd > 5:
        raise ValueError("Gate 6 requires a positive budget no greater than $5")
    if not os.getenv(api_key_var):
        raise RuntimeError(f"{api_key_var} is not set")

    source_gate4_root = source_gate4_root.resolve()
    source_path = source_gate4_root / "gate4-report.json"
    if not source_path.exists():
        raise ValueError(f"missing completed Gate 4 report: {source_path}")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("status") != "complete":
        raise ValueError("Gate 6 requires a completed Gate 4 source")
    source_failure = next(
        run
        for run in source["baseline_runs"]
        if run["model_id"] == "qwen/qwen3.8-27b"
        and run["task_id"] == TASK_ID
        and not run["success"]
    )
    checkpoint_id = f"{TASK_ID}-after-qwen_qwen3.8-27b"
    dirty_workspace = source_gate4_root / "checkpoints" / checkpoint_id / "workspace"
    if not dirty_workspace.is_dir():
        raise ValueError(f"missing dirty checkpoint workspace: {dirty_workspace}")
    if workspace_digest(dirty_workspace) != source_failure["workspace_digest"]:
        raise ValueError("dirty checkpoint digest no longer matches the Gate 4 source")

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if resume_root is None:
        root = artifacts_root / f"gate6-{timestamp}"
        root.mkdir(parents=True, exist_ok=False)
        runs: list[dict[str, Any]] = []
        infrastructure_attempts: list[dict[str, Any]] = []
        pricing_path = root / "pricing-snapshot.json"
    else:
        root = resume_root.resolve()
        partial = json.loads((root / "partial-report.json").read_text(encoding="utf-8"))
        if source_gate4_root != Path(partial["source_gate4_root"]).resolve():
            raise ValueError("resume source does not match the original Gate 4 source")
        if replications_per_arm != int(partial["replications_per_arm"]):
            raise ValueError("resume replication count does not match")
        if budget_usd > float(partial["authorized_budget_usd"]):
            raise ValueError("resume budget exceeds the original Gate 6 ceiling")
        runs = [
            run
            for run in partial.get("runs", [])
            if run.get("failure_type") != "harness_or_provider_error"
        ]
        infrastructure_attempts = partial.get("infrastructure_attempts", [])
        pricing_path = root / f"pricing-snapshot-resume-{timestamp}.json"

    models = load_model_catalog(pricing_path, roster="gate4")
    model = next(model for model in models if model.model_id == MODEL_ID)
    source_model = next(row for row in source["models"] if row["model_id"] == MODEL_ID)
    if asdict(model)["reasoning_effort"] != source_model["reasoning_effort"]:
        raise ValueError("Kimi reasoning configuration no longer matches Gate 4")

    clean_digest = workspace_digest(starter_dir(TASK_ID))
    dirty_digest = workspace_digest(dirty_workspace)
    evidence = _public_evidence(source_failure)
    prior_summary = (
        "Qwen attempted this task and the hidden verifier did not pass. Hidden-test details "
        "are not exposed to the next model."
    )
    arm_config = {
        "neutral_clean": {
            "seed": starter_dir(TASK_ID),
            "context": "",
            "evidence": [],
            "target_digest": clean_digest,
        },
        "stale_clean": {
            "seed": starter_dir(TASK_ID),
            "context": _stale_clean_handoff(source_failure),
            "evidence": [evidence],
            "target_digest": clean_digest,
        },
        "digest_aware_clean": {
            "seed": starter_dir(TASK_ID),
            "context": render_state_bound_handoff(
                current_workspace_digest=clean_digest,
                evidence=[evidence],
                prior_attempt_summary=prior_summary,
            ),
            "evidence": [evidence],
            "target_digest": clean_digest,
        },
        "digest_aware_dirty": {
            "seed": dirty_workspace,
            "context": render_state_bound_handoff(
                current_workspace_digest=dirty_digest,
                evidence=[evidence],
                prior_attempt_summary=prior_summary,
            ),
            "evidence": [evidence],
            "target_digest": dirty_digest,
        },
    }

    spent = sum(float(run["cost_usd"]) for run in runs + infrastructure_attempts)
    completed = {(run["arm"], int(run["replication"])) for run in runs}

    def save_partial(status: str = "in_progress") -> None:
        _write_json(
            root / "partial-report.json",
            {
                "gate": 6,
                "status": status,
                "source_gate4_root": str(source_gate4_root),
                "replications_per_arm": replications_per_arm,
                "authorized_budget_usd": budget_usd,
                "estimated_spend_usd": spent,
                "runs": runs,
                "infrastructure_attempts": infrastructure_attempts,
            },
        )

    try:
        for replication in range(1, replications_per_arm + 1):
            offset = (replication - 1) % len(ARMS)
            ordered_arms = ARMS[offset:] + ARMS[:offset]
            for arm in ordered_arms:
                if (arm, replication) in completed:
                    continue
                if spent >= budget_usd:
                    raise RuntimeError("Gate 6 global budget exhausted")
                config = arm_config[arm]
                run_checkpoint_id = f"gate6-{arm}-rep-{replication}"
                for attempt in range(2):
                    summary = await _evaluate_with_hard_deadline(
                        config_dir=root / "worker-configs",
                        run_root=root / "runs" / arm / f"rep-{replication}",
                        model=model,
                        task_id=TASK_ID,
                        api_key_var=api_key_var,
                        per_run_cap_usd=min(per_run_cap_usd, budget_usd - spent),
                        workspace_seed=config["seed"],
                        handoff_context=config["context"],
                        handoff_evidence=config["evidence"],
                        checkpoint_id=run_checkpoint_id,
                    )
                    summary.update(
                        {
                            "phase": "gate6_replication",
                            "arm": arm,
                            "replication": replication,
                            "initial_workspace_digest": config["target_digest"],
                            "source_evidence_digest": evidence["workspace_digest"],
                        }
                    )
                    spent += float(summary["cost_usd"])
                    if summary.get("failure_type") != "harness_or_provider_error":
                        break
                    summary["infrastructure_attempt"] = attempt + 1
                    infrastructure_attempts.append(summary)
                    save_partial()
                else:
                    raise RuntimeError(f"repeated infrastructure failure for {arm}/{replication}")
                runs.append(summary)
                completed.add((arm, replication))
                save_partial()
    except Exception as error:
        save_partial("stopped_on_provider_or_harness_error")
        _write_json(
            root / "fatal-error.json",
            {
                "gate": 6,
                "status": "stopped_on_provider_or_harness_error",
                "estimated_spend_usd": spent,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise

    report = {
        "gate": 6,
        "status": "complete",
        "completed_at": datetime.now(UTC).isoformat(),
        "source_gate4_root": str(source_gate4_root),
        "authorized_budget_usd": budget_usd,
        "estimated_spend_usd": spent,
        "replications_per_arm": replications_per_arm,
        "model": asdict(model),
        "task_id": TASK_ID,
        "arms": {
            arm: {
                "initial_workspace_digest": config["target_digest"],
                "source_evidence_digest": (
                    evidence["workspace_digest"] if config["evidence"] else None
                ),
                "evidence_state_matches": bool(
                    config["evidence"]
                    and evidence["workspace_digest"] == config["target_digest"]
                ),
            }
            for arm, config in arm_config.items()
        },
        "runs": runs,
        "infrastructure_attempts": infrastructure_attempts,
        "infrastructure_attempt_count": len(infrastructure_attempts),
        "analysis": analyze_gate6(runs),
    }
    _write_json(root / "gate6-report.json", report)
    (root / "gate6-report.md").write_text(render_gate6_report(report), encoding="utf-8")
    save_partial("complete")
    return {"artifact_root": str(root), **report}
