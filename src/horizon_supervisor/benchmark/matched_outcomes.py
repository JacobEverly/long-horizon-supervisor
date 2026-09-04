from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from horizon_supervisor.models import CandidateModelFeatures

DEFAULT_PANEL = Path("data/supervisor/terminal-bench-pro-panel-v0.jsonl")
DEFAULT_SWITCHYARD = Path("benchmarks/switchyard-gate7.toml")
LEARNING_VALID_STATUSES = frozenset({"verified", "agent_protocol_failure"})
MODEL_ATTRIBUTABLE_EXCEPTION_TYPES = frozenset({"OutputLengthExceededError"})


def _record_split(wave: int) -> str:
    """Keep the sealed Wave 3 panel out of every development-data path."""
    return "held_out" if wave == 3 else "development"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _panel_wave(path: Path, wave: int) -> dict[str, dict[str, Any]]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["wave"] == wave:
                rows[row["source_task_name"]] = row
    return rows


def _route_endpoints(path: Path) -> dict[str, str]:
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    targets = {
        target_name: target["id"]
        for target_name, target in config.get("targets", {}).items()
    }
    endpoints = {}
    for route in config.get("routes", {}).values():
        if route.get("type") == "passthrough":
            endpoints[route["id"]] = targets[route["target"]]
    return endpoints


def _duration_seconds(started_at: str | None, finished_at: str | None) -> float | None:
    if not started_at or not finished_at:
        return None
    return (
        datetime.fromisoformat(finished_at) - datetime.fromisoformat(started_at)
    ).total_seconds()


def _trial_result_paths(job_root: Path) -> list[Path]:
    paths = []
    for path in job_root.rglob("result.json"):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("trial_name"):
            paths.append(path)
    return sorted(paths)


def _run_root(job_root: Path) -> Path | None:
    for candidate in (job_root, *job_root.parents):
        if (candidate / "run-manifest.json").exists():
            return candidate
    return None


def _task_runs(
    run_root: Path | None, run_manifest: dict[str, Any] | None
) -> dict[str, dict[str, Any]]:
    if run_root is None:
        return {}
    report_path = run_root / "report.json"
    partial_path = run_root / "partial-report.json"
    payload = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.exists()
        else json.loads(partial_path.read_text(encoding="utf-8"))
        if partial_path.exists()
        else {}
    )
    runs = {
        row["source_task_name"]: row
        for row in payload.get("task_runs", [])
    }
    if runs or run_manifest is None:
        return runs
    selected = run_manifest.get("frozen_inputs", {}).get("selected_task_names", [])
    stats_path = run_root / "routing-stats.json"
    if len(selected) == 1 and stats_path.exists():
        return {
            selected[0]: {
                "source_task_name": selected[0],
                "provider_spend_usd": payload.get("provider_spend_usd"),
                "routing_stats": json.loads(stats_path.read_text(encoding="utf-8")),
            }
        }
    return {}


def _pricing_snapshot(run_root: Path | None) -> dict[str, Any]:
    if run_root is None:
        return {}
    path = run_root / "pricing-snapshot.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_features(
    endpoint: str,
    pricing_snapshot: dict[str, Any],
    runtime_config: dict[str, Any],
) -> dict[str, Any]:
    pricing = {
        row["model_id"]: row for row in pricing_snapshot.get("models", [])
    }
    if endpoint not in pricing:
        raise ValueError(f"pricing snapshot is missing candidate endpoint {endpoint!r}")
    row = pricing[endpoint]
    base_input_price = float(row["input_usd_per_token"])
    profile = CandidateModelFeatures(
        context_window_tokens=int(row["context_length"]),
        input_usd_per_million_tokens=base_input_price * 1_000_000,
        output_usd_per_million_tokens=(
            float(row["output_usd_per_token"]) * 1_000_000
        ),
        cached_input_usd_per_million_tokens=(
            float(row.get("cached_input_usd_per_token") or base_input_price)
            * 1_000_000
        ),
        cache_write_input_usd_per_million_tokens=(
            float(row.get("cache_write_input_usd_per_token") or base_input_price)
            * 1_000_000
        ),
        supports_tool_use=True,
        reasoning_effort=str(runtime_config.get("reasoning_effort", "high")),
        max_output_tokens=int(runtime_config.get("max_output_tokens", 4_096)),
        max_turns=int(runtime_config.get("max_turns", 12)),
        request_timeout_seconds=(
            int(runtime_config["request_timeout_seconds"])
            if runtime_config.get("request_timeout_seconds") is not None
            else None
        ),
        request_retry_attempts=(
            int(runtime_config["request_retry_attempts"])
            if runtime_config.get("request_retry_attempts") is not None
            else None
        ),
        output_length_retry_attempts=(
            int(runtime_config["output_length_retry_attempts"])
            if runtime_config.get("output_length_retry_attempts") is not None
            else None
        ),
        catalog_source=str(pricing_snapshot.get("source", "unknown")),
        catalog_captured_at=pricing_snapshot["captured_at"],
    )
    return profile.model_dump(mode="json")


def _list_cost(model_stats: dict[str, Any], pricing: dict[str, Any]) -> float:
    prompt_tokens = int(model_stats.get("prompt_tokens", 0))
    cached_tokens = int(model_stats.get("cached_tokens", 0))
    cache_write_tokens = int(model_stats.get("cache_creation_tokens", 0))
    base_tokens = max(prompt_tokens - cached_tokens - cache_write_tokens, 0)
    base_rate = float(pricing["input_usd_per_token"])
    cache_rate = float(pricing.get("cached_input_usd_per_token") or base_rate)
    cache_write_rate = float(
        pricing.get("cache_write_input_usd_per_token") or base_rate
    )
    output_rate = float(pricing["output_usd_per_token"])
    return (
        base_tokens * base_rate
        + cached_tokens * cache_rate
        + cache_write_tokens * cache_write_rate
        + int(model_stats.get("completion_tokens", 0)) * output_rate
    )


def _task_model_usage(
    task_run: dict[str, Any],
    pricing: dict[str, dict[str, Any]],
    *,
    trust_provider_spend: bool = True,
) -> dict[str, dict[str, Any]]:
    stats = task_run.get("routing_stats") or {}
    models = stats.get("models") or {}
    raw_costs = {
        endpoint: _list_cost(model_stats, pricing[endpoint])
        for endpoint, model_stats in models.items()
        if endpoint in pricing
    }
    raw_total = sum(raw_costs.values())
    provider_spend = (
        task_run.get("provider_spend_usd") if trust_provider_spend else None
    )
    rows = {}
    for endpoint, model_stats in models.items():
        estimated = raw_costs.get(endpoint)
        allocated = (
            float(provider_spend) * estimated / raw_total
            if provider_spend is not None
            and estimated is not None
            and raw_total > 0
            else None
        )
        rows[endpoint] = {
            "model_calls": int(model_stats.get("calls", 0)),
            "router_errors": int(model_stats.get("errors", 0)),
            "router_prompt_tokens": int(model_stats.get("prompt_tokens", 0)),
            "router_cached_tokens": int(model_stats.get("cached_tokens", 0)),
            "router_completion_tokens": int(
                model_stats.get("completion_tokens", 0)
            ),
            "router_reasoning_tokens": int(model_stats.get("reasoning_tokens", 0)),
            "estimated_list_cost_usd": estimated,
            "allocated_provider_cost_usd": allocated,
            "provider_cost_allocation_method": (
                "task-total-proportional-to-cache-aware-list-cost"
                if allocated is not None
                else None
            ),
        }
    return rows


def _result_model_usage(
    agent_result: dict[str, Any], pricing: dict[str, Any]
) -> dict[str, Any]:
    """Recover portable usage when a parent Gate 8 report was interrupted."""
    prompt_tokens = int(agent_result.get("n_input_tokens") or 0)
    cached_tokens = int(agent_result.get("n_cache_tokens") or 0)
    completion_tokens = int(agent_result.get("n_output_tokens") or 0)
    metadata = agent_result.get("metadata") or {}
    request_times = metadata.get("api_request_times_msec") or []
    calls = len(request_times) or int(metadata.get("n_episodes") or 0)
    model_stats = {
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "completion_tokens": completion_tokens,
    }
    return {
        "model_calls": calls,
        "router_errors": 0,
        "router_prompt_tokens": prompt_tokens,
        "router_cached_tokens": cached_tokens,
        "router_completion_tokens": completion_tokens,
        "router_reasoning_tokens": 0,
        "estimated_list_cost_usd": _list_cost(model_stats, pricing),
        "allocated_provider_cost_usd": None,
        "provider_cost_allocation_method": None,
        "usage_source": "harbor-agent-result-fallback",
    }


def _source_task_name(result: dict[str, Any]) -> str | None:
    task_id = result.get("task_id") or {}
    if task_id.get("name"):
        return str(task_id["name"])
    if task_id.get("path"):
        return Path(task_id["path"]).name
    task_name = result.get("task_name")
    return str(task_name).rsplit("/", maxsplit=1)[-1] if task_name else None


def _outcome_status(
    exception_type: str | None,
    *,
    router_errors: int,
    reward: float | None,
    recovered_provider_error: bool = False,
) -> str:
    """Separate attributable agent behavior from provider and environment errors."""
    # A clean verifier pass is authoritative: the task was completed even if a
    # later, non-fatal model call failed. Keep router_errors as telemetry so the
    # route's reliability can still be evaluated separately.
    if reward is not None and reward >= 1.0:
        return "verified"
    # A final attributable protocol exception is also authoritative. Earlier
    # provider errors may have been recovered by the bounded request retry;
    # retain them as router telemetry without discarding the final capability
    # observation.
    if exception_type in MODEL_ATTRIBUTABLE_EXCEPTION_TYPES:
        return "agent_protocol_failure"
    if router_errors > 0 and not recovered_provider_error:
        return "provider_error"
    if exception_type is not None:
        return "infrastructure_error"
    return "verified"


def build_matched_outcomes(
    job_root: Path,
    output_path: Path,
    summary_path: Path,
    *,
    panel_path: Path = DEFAULT_PANEL,
    switchyard_path: Path = DEFAULT_SWITCHYARD,
    expected_routes: tuple[str, ...] | None = None,
    trust_provider_spend: bool = True,
    record_split_override: str | None = None,
    evaluation_role: str | None = None,
    replication_index: int | None = None,
) -> dict[str, Any]:
    if record_split_override not in {None, "development", "held_out"}:
        raise ValueError("record split override must be development or held_out")
    if replication_index is not None and replication_index < 1:
        raise ValueError("replication index must be positive")
    run_root = _run_root(job_root)
    run_manifest_path = (
        run_root / "run-manifest.json" if run_root is not None else None
    )
    run_manifest = (
        json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if run_manifest_path is not None and run_manifest_path.exists()
        else None
    )
    wave = int((run_manifest or {}).get("config", {}).get("wave", 1))
    panel = _panel_wave(panel_path, wave)
    endpoints = _route_endpoints(switchyard_path)
    task_runs = _task_runs(run_root, run_manifest)
    pricing_snapshot = _pricing_snapshot(run_root)
    pricing = {
        row["model_id"]: row for row in pricing_snapshot.get("models", [])
    }
    usage_by_task = {
        task_name: _task_model_usage(
            task_run,
            pricing,
            trust_provider_spend=trust_provider_spend,
        )
        for task_name, task_run in task_runs.items()
    }
    if expected_routes is None:
        expected_routes = tuple(
            run_manifest["config"]["route_ids"]
            if run_manifest is not None
            else sorted(endpoints)
        )
    expected_tasks = set(
        run_manifest["frozen_inputs"]["selected_task_names"]
        if run_manifest is not None
        else panel
    )

    rows = []
    pair_counts: Counter[tuple[str, str]] = Counter()
    for result_path in _trial_result_paths(job_root):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        task_name = _source_task_name(result)
        if task_name not in panel:
            raise ValueError(
                f"trial task {task_name!r} is outside frozen wave {wave}"
            )
        model_name = result["config"]["agent"]["model_name"]
        route_id = model_name.removeprefix("openai/")
        if route_id not in expected_routes:
            raise ValueError(f"unexpected route {route_id!r}")
        pair = (task_name, route_id)
        pair_counts[pair] += 1
        if pair_counts[pair] > 1:
            raise ValueError(f"duplicate matched outcome for {task_name}/{route_id}")

        exception = result.get("exception_info")
        verifier = result.get("verifier_result")
        reward = (
            verifier.get("rewards", {}).get("reward") if verifier is not None else None
        )
        agent_result = result.get("agent_result") or {}
        panel_row = panel[task_name]
        runtime_config = (run_manifest or {}).get("config", {})
        max_turns = int(runtime_config.get("max_turns", 12))
        max_output_tokens = int(runtime_config.get("max_output_tokens", 4_096))
        request_timeout_seconds = runtime_config.get("request_timeout_seconds")
        request_retry_attempts = runtime_config.get("request_retry_attempts")
        output_length_retry_attempts = runtime_config.get(
            "output_length_retry_attempts"
        )
        matched_group = _sha256_text(
            f"gate8-wave{wave}|{panel_row['task_id']}|terminus-2|"
            f"{max_turns}|{max_output_tokens}|{request_timeout_seconds}|"
            f"{request_retry_attempts}|{output_length_retry_attempts}"
        )
        endpoint = endpoints.get(route_id)
        if endpoint is None:
            raise ValueError(f"route {route_id!r} has no passthrough endpoint")
        router_usage = usage_by_task.get(task_name, {}).get(endpoint, {})
        if not router_usage:
            router_usage = _result_model_usage(agent_result, pricing[endpoint])
        exception_type = (
            exception.get("exception_type") if exception is not None else None
        )
        status = _outcome_status(
            exception_type,
            router_errors=int(router_usage.get("router_errors", 0)),
            reward=float(reward) if reward is not None else None,
            recovered_provider_error=bool(
                int(router_usage.get("router_errors", 0)) > 0
                and reward is not None
                and exception_type is None
                and int(router_usage.get("model_calls", 0)) >= max_turns
            ),
        )
        provenance = {
            "harbor_trial_id": result["id"],
            "harbor_trial_name": result["trial_name"],
            "result_path": str(result_path),
            "task_source_revision": panel_row["source"]["revision"],
        }
        if record_split_override is not None:
            provenance["source_wave"] = wave
            provenance["record_split_override"] = record_split_override
        if evaluation_role is not None:
            provenance["evaluation_role"] = evaluation_role
        if replication_index is not None:
            provenance["replication_index"] = replication_index
        rows.append(
            {
                "schema_version": "matched-model-outcome.v1",
                "outcome_id": _sha256_text(
                    f"{result['id']}|{panel_row['task_id']}|{route_id}"
                ),
                "matched_group_id": matched_group,
                "task": {
                    "task_id": panel_row["task_id"],
                    "source_task_name": task_name,
                    "difficulty": panel_row["difficulty"],
                    "category": panel_row["category"],
                    "record_split": record_split_override or _record_split(wave),
                },
                "model": {
                    "route_id": route_id,
                    "endpoint": endpoint,
                    "agent": result.get("agent_info", {}).get("name", "terminus-2"),
                    "candidate_features": _candidate_features(
                        endpoint, pricing_snapshot, runtime_config
                    ),
                },
                "initial_state": {
                    "kind": "clean_task_start",
                    "digest": panel_row["execution_lock"]["archive_sha256"],
                },
                "outcome": {
                    "status": status,
                    "reward": float(reward) if reward is not None else None,
                    "completed": bool(
                        status == "verified"
                        and reward is not None
                        and float(reward) >= 1.0
                    ),
                    "input_tokens": agent_result.get("n_input_tokens"),
                    "cache_tokens": agent_result.get("n_cache_tokens"),
                    "output_tokens": agent_result.get("n_output_tokens"),
                    "reported_cost_usd": agent_result.get("cost_usd"),
                    **router_usage,
                    "duration_seconds": _duration_seconds(
                        result.get("started_at"), result.get("finished_at")
                    ),
                    "exception_type": exception_type,
                    "provider_error_recovered": bool(
                        int(router_usage.get("router_errors", 0)) > 0
                        and reward is not None
                        and exception_type is None
                        and int(router_usage.get("model_calls", 0)) >= max_turns
                    ),
                },
                "provenance": provenance,
            }
        )

    rows.sort(
        key=lambda row: (
            row["task"]["source_task_name"],
            row["model"]["route_id"],
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    expected_pairs = {
        (task_name, route_id)
        for task_name in expected_tasks
        for route_id in expected_routes
    }
    observed_pairs = set(pair_counts)
    outcome_status_counts = Counter(row["outcome"]["status"] for row in rows)
    route_counts = Counter(row["model"]["route_id"] for row in rows)
    task_counts = Counter(row["task"]["source_task_name"] for row in rows)
    verified_rows = [row for row in rows if row["outcome"]["status"] == "verified"]
    allocated_costs = [
        row["outcome"].get("allocated_provider_cost_usd")
        for row in rows
        if row["outcome"].get("allocated_provider_cost_usd") is not None
    ]
    summary = {
        "schema_version": "matched-model-outcome-summary.v1",
        "wave": wave,
        "learning_contract": {
            "scoring_unit": "task-model-pair",
            "prediction_targets": [
                "outcome.completed",
                "outcome.estimated_list_cost_usd",
                "outcome.duration_seconds",
            ],
            "provider_spend_is_audit_only": True,
            "portable_feature_path": "model.candidate_features",
            "identity_fields_are_cold_start_features": False,
            "calibration_must_precede_predicted_trial": True,
        },
        "record_count": len(rows),
        "expected_record_count": len(expected_pairs),
        "task_count": len(task_counts),
        "route_counts": dict(sorted(route_counts.items())),
        "record_split_counts": dict(
            sorted(Counter(row["task"]["record_split"] for row in rows).items())
        ),
        "status_counts": dict(sorted(outcome_status_counts.items())),
        "verified_completion_count": sum(
            row["outcome"]["completed"] for row in verified_rows
        ),
        "cost_attributed_record_count": len(allocated_costs),
        "allocated_provider_cost_total_usd": sum(allocated_costs),
        "provider_spend_trusted": trust_provider_spend,
        "record_split_override": record_split_override,
        "evaluation_role": evaluation_role,
        "replication_index": replication_index,
        "missing_pair_count": len(expected_pairs - observed_pairs),
        "missing_pairs": sorted(
            f"{task_name}|{route_id}"
            for task_name, route_id in expected_pairs - observed_pairs
        ),
        "all_pairs_present_once": observed_pairs == expected_pairs,
        "infrastructure_errors_are_not_failures": True,
        "outcome_path": str(output_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build matched outcomes from Harbor")
    parser.add_argument("job_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--switchyard", type=Path, default=DEFAULT_SWITCHYARD)
    parser.add_argument(
        "--ignore-provider-spend",
        action="store_true",
        help=(
            "retain token/list-price estimates but suppress provider-spend allocation "
            "when another run overlapped the key-usage window"
        ),
    )
    parser.add_argument(
        "--record-split",
        choices=("development", "held_out"),
        help="explicit split for post-hoc replication runs",
    )
    parser.add_argument(
        "--evaluation-role",
        help="auditable role attached to every extracted record",
    )
    parser.add_argument(
        "--replication-index",
        type=int,
        help="predeclared clean-start replication index",
    )
    args = parser.parse_args()
    summary = build_matched_outcomes(
        args.job_root,
        args.output,
        args.summary,
        panel_path=args.panel,
        switchyard_path=args.switchyard,
        trust_provider_spend=not args.ignore_provider_spend,
        record_split_override=args.record_split,
        evaluation_role=args.evaluation_role,
        replication_index=args.replication_index,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
