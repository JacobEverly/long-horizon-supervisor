from __future__ import annotations

import hashlib
import json
from pathlib import Path

from horizon_supervisor.training.build_swiss_cheese_matrix import (
    build_swiss_cheese_matrix,
)
from horizon_supervisor.training.swiss_cheese_scorecard import (
    build_swiss_cheese_scorecard,
)


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _row(
    task: str,
    route: str,
    endpoint: str,
    replication: int,
    completed: bool,
    *,
    split: str,
    role: str | None,
) -> dict:
    provenance = {
        "harbor_trial_id": f"{task}-{route}-{replication}",
        "replication_index": replication,
        "superseded_outcome_ids": [],
    }
    if role is not None:
        provenance["evaluation_role"] = role
    return {
        "schema_version": "matched-model-outcome.v1",
        "outcome_id": f"{task}-{route}-{replication}",
        "matched_group_id": f"group-{task}",
        "task": {
            "task_id": task,
            "source_task_name": task,
            "record_split": split,
        },
        "model": {"route_id": route, "endpoint": endpoint},
        "initial_state": {"kind": "clean_task_start", "digest": f"state-{task}"},
        "outcome": {
            "status": "verified",
            "completed": completed,
            "estimated_list_cost_usd": 0.01 if route != "small" else 0.001,
            "input_tokens": 10,
            "output_tokens": 5,
            "duration_seconds": 2.0,
        },
        "provenance": provenance,
    }


def test_replication_builder_and_scorecard_exclude_discovery_from_claims(
    tmp_path: Path,
) -> None:
    tasks = ["task-a", "task-b"]
    existing = ["flash", "qwen"]
    routes = [*existing, "small"]
    endpoints = {route: f"endpoint-{route}" for route in routes}
    baseline = tmp_path / "baseline.jsonl"
    baseline_rows = [
        _row(
            task,
            route,
            endpoints[route],
            1,
            completed=(task == "task-a" and route == "flash"),
            split="held_out",
            role=None,
        )
        for task in tasks
        for route in existing
    ]
    _write_rows(baseline, baseline_rows)
    new = tmp_path / "new.jsonl"
    new_rows = []
    for task in tasks:
        for route in routes:
            indices = (1, 2, 3) if route == "small" else (2, 3)
            for replication in indices:
                completed = (
                    (route == "qwen" and task == "task-b")
                    or (route == "small" and task == "task-a")
                )
                new_rows.append(
                    _row(
                        task,
                        route,
                        endpoints[route],
                        replication,
                        completed,
                        split="development",
                        role="posthoc_clean_start_replication",
                    )
                )
    _write_rows(new, new_rows)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "design": {
                    "tasks": tasks,
                    "routes": routes,
                    "existing_routes": existing,
                    "small_route": "small",
                    "route_endpoints": endpoints,
                    "replication_indices": [1, 2, 3],
                    "confirmatory_replication_indices": [2, 3],
                },
                "analysis": {
                    "bootstrap_seed": 7,
                    "bootstrap_samples": 100,
                    "predeclared_heterogeneous_contrasts": [
                        ["flash", "qwen"],
                        ["flash", "small"],
                    ],
                    "named_cascades": {
                        "existing": existing,
                        "small-overlay": ["flash", "small", "qwen"],
                    },
                },
                "frozen_inputs": {
                    "baseline_outcomes_sha256": hashlib.sha256(
                        baseline.read_bytes()
                    ).hexdigest()
                },
            }
        )
    )
    matrix = tmp_path / "matrix.jsonl"
    summary = build_swiss_cheese_matrix(
        manifest,
        baseline,
        [new],
        matrix,
        tmp_path / "matrix-summary.json",
    )
    assert summary["record_count"] == 18
    assert summary["record_split_counts"] == {"development": 18}
    assert summary["discovery_replication_excluded_from_confirmatory_claims"]

    report = build_swiss_cheese_scorecard(
        manifest,
        matrix,
        tmp_path / "scorecard.json",
        key_usage_before_usd=10.0,
        key_usage_after_usd=11.0,
        completed_run_report_spend_usd=0.9,
    )
    assert report["experiment"]["records"] == 18
    assert report["experiment"][
        "discovery_replication_excluded_from_confirmatory_claims"
    ]
    assert len(report["ordered_heterogeneous_pass_at_2"]) == 6
    assert report["spend_audit"]["exact_incremental_spend_usd"] == 1.0
