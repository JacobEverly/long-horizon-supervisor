from __future__ import annotations

import json
from pathlib import Path

import pytest

from horizon_supervisor.benchmark.matched_outcomes import (
    _outcome_status,
    _record_split,
    _source_task_name,
    build_matched_outcomes,
)

ROOT = Path(__file__).resolve().parents[1]


def test_wave3_is_held_out_and_earlier_waves_are_development() -> None:
    assert _record_split(1) == "development"
    assert _record_split(2) == "development"
    assert _record_split(3) == "held_out"


def test_source_task_name_supports_local_harbor_paths() -> None:
    result = {"task_id": {"path": "/tmp/tasks/example-task"}}
    assert _source_task_name(result) == "example-task"


def test_outcome_status_separates_protocol_from_external_failures() -> None:
    assert (
        _outcome_status("OutputLengthExceededError", router_errors=0, reward=None)
        == "agent_protocol_failure"
    )
    assert (
        _outcome_status("DaytonaError", router_errors=0, reward=None)
        == "infrastructure_error"
    )
    assert _outcome_status(None, router_errors=1, reward=0.0) == "provider_error"
    assert (
        _outcome_status(
            None,
            router_errors=1,
            reward=0.0,
            recovered_provider_error=True,
        )
        == "verified"
    )
    assert (
        _outcome_status("OutputLengthExceededError", router_errors=1, reward=0.0)
        == "agent_protocol_failure"
    )
    assert _outcome_status(None, router_errors=0, reward=0.0) == "verified"
    assert _outcome_status(None, router_errors=1, reward=1.0) == "verified"
    assert (
        _outcome_status("OutputLengthExceededError", router_errors=1, reward=1.0)
        == "verified"
    )


def _trial(task_name: str, route_id: str, trial_id: str, reward: float) -> dict:
    return {
        "id": trial_id,
        "task_id": {"name": task_name},
        "trial_name": f"{task_name}-{route_id}",
        "config": {"agent": {"model_name": f"openai/{route_id}"}},
        "agent_info": {"name": "terminus-2"},
        "agent_result": {
            "n_input_tokens": 10,
            "n_cache_tokens": 2,
            "n_output_tokens": 5,
            "cost_usd": 0.01,
        },
        "verifier_result": {"rewards": {"reward": reward}},
        "exception_info": None,
        "started_at": "2026-08-25T00:00:00+00:00",
        "finished_at": "2026-08-25T00:01:00+00:00",
    }


def test_matched_outcome_builder_requires_one_result_per_task_route(tmp_path: Path) -> None:
    panel_lines = (
        ROOT / "data/supervisor/terminal-bench-pro-panel-v0.jsonl"
    ).read_text().splitlines()
    source_panel = [
        json.loads(line)
        for line in panel_lines
        if json.loads(line)["wave"] == 1
    ][0]
    panel_path = tmp_path / "panel.jsonl"
    panel_path.write_text(json.dumps(source_panel) + "\n")
    job_root = tmp_path / "jobs" / "job"
    for index, (route_id, reward) in enumerate(
        (("gate7/fixed-qwen", 0.0), ("gate7/fixed-kimi", 1.0))
    ):
        trial_root = job_root / f"trial-{index}"
        trial_root.mkdir(parents=True)
        (trial_root / "result.json").write_text(
            json.dumps(
                _trial(source_panel["source_task_name"], route_id, str(index), reward)
                )
            )
    (tmp_path / "run-manifest.json").write_text(
        json.dumps(
            {
                "config": {"max_turns": 12, "max_output_tokens": 8192},
                "frozen_inputs": {
                    "selected_task_names": [source_panel["source_task_name"]]
                },
            }
        )
    )
    (tmp_path / "pricing-snapshot.json").write_text(
        json.dumps(
            {
                "captured_at": "2026-08-25T00:00:00+00:00",
                "source": "test-catalog",
                "models": [
                    {
                        "model_id": "qwen/qwen3.8-27b",
                        "input_usd_per_token": 1.0,
                        "output_usd_per_token": 1.0,
                        "context_length": 1_000_000,
                    },
                    {
                        "model_id": "moonshotai/kimi-k3",
                        "input_usd_per_token": 2.0,
                        "output_usd_per_token": 2.0,
                        "context_length": 1_048_576,
                    },
                ],
            }
        )
    )
    output = tmp_path / "outcomes.jsonl"
    summary_path = tmp_path / "summary.json"
    summary = build_matched_outcomes(
        job_root,
        output,
        summary_path,
        panel_path=panel_path,
        switchyard_path=ROOT / "benchmarks/switchyard-gate7.toml",
        expected_routes=("gate7/fixed-qwen", "gate7/fixed-kimi"),
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert summary["record_count"] == 2
    assert summary["expected_record_count"] == 2
    assert summary["all_pairs_present_once"] is True
    assert summary["verified_completion_count"] == 1
    assert {row["initial_state"]["digest"] for row in rows} == {
        source_panel["execution_lock"]["archive_sha256"]
    }
    assert {row["outcome"]["usage_source"] for row in rows} == {
        "harbor-agent-result-fallback"
    }
    assert {
        row["outcome"]["estimated_list_cost_usd"] for row in rows
    } == {15.0, 30.0}


def test_matched_outcome_builder_can_label_posthoc_replication(tmp_path: Path) -> None:
    source_panel = next(
        json.loads(line)
        for line in (
            ROOT / "data/supervisor/terminal-bench-pro-panel-v0.jsonl"
        ).read_text().splitlines()
        if json.loads(line)["wave"] == 1
    )
    panel_path = tmp_path / "panel.jsonl"
    panel_path.write_text(json.dumps(source_panel) + "\n")
    task_name = source_panel["source_task_name"]
    route_id = "gate7/fixed-qwen"
    trial_root = tmp_path / "jobs" / "job" / "trial"
    trial_root.mkdir(parents=True)
    (trial_root / "result.json").write_text(
        json.dumps(_trial(task_name, route_id, "replication", 1.0))
    )
    (tmp_path / "run-manifest.json").write_text(
        json.dumps(
            {
                "config": {"route_ids": [route_id]},
                "frozen_inputs": {"selected_task_names": [task_name]},
            }
        )
    )
    (tmp_path / "pricing-snapshot.json").write_text(
        json.dumps(
            {
                "captured_at": "2026-08-31T00:00:00+00:00",
                "source": "test-catalog",
                "models": [
                    {
                        "model_id": "qwen/qwen3.8-27b",
                        "input_usd_per_token": 1.0,
                        "output_usd_per_token": 1.0,
                        "context_length": 1_000_000,
                    }
                ],
            }
        )
    )
    output = tmp_path / "outcomes.jsonl"
    summary = build_matched_outcomes(
        trial_root.parents[1],
        output,
        tmp_path / "summary.json",
        panel_path=panel_path,
        switchyard_path=ROOT / "benchmarks/switchyard-gate7.toml",
        record_split_override="development",
        evaluation_role="posthoc_clean_start_replication",
        replication_index=2,
    )
    row = json.loads(output.read_text())
    assert row["task"]["record_split"] == "development"
    assert row["provenance"]["source_wave"] == 1
    assert row["provenance"]["replication_index"] == 2
    assert row["provenance"]["evaluation_role"] == (
        "posthoc_clean_start_replication"
    )
    assert summary["record_split_override"] == "development"
    assert summary["replication_index"] == 2


def test_matched_outcomes_attribute_task_batch_costs(tmp_path: Path) -> None:
    source_panel = next(
        json.loads(line)
        for line in (
            ROOT / "data/supervisor/terminal-bench-pro-panel-v0.jsonl"
        ).read_text().splitlines()
        if json.loads(line)["wave"] == 1
    )
    panel_path = tmp_path / "panel.jsonl"
    panel_path.write_text(json.dumps(source_panel) + "\n")
    task_name = source_panel["source_task_name"]
    jobs_root = tmp_path / "jobs"
    for index, route_id in enumerate(("gate7/fixed-qwen", "gate7/fixed-kimi")):
        trial_root = jobs_root / "task-job" / f"trial-{index}"
        trial_root.mkdir(parents=True)
        (trial_root / "result.json").write_text(
            json.dumps(_trial(task_name, route_id, str(index), 1.0))
        )
    (tmp_path / "run-manifest.json").write_text(
        json.dumps(
            {
                "config": {
                    "route_ids": ["gate7/fixed-qwen", "gate7/fixed-kimi"],
                    "max_turns": 12,
                    "max_output_tokens": 8192,
                },
                "frozen_inputs": {"selected_task_names": [task_name]},
            }
        )
    )
    (tmp_path / "pricing-snapshot.json").write_text(
        json.dumps(
            {
                "captured_at": "2026-08-25T00:00:00+00:00",
                "source": "test-catalog",
                "models": [
                    {
                        "model_id": "qwen/qwen3.8-27b",
                        "input_usd_per_token": 1.0,
                        "output_usd_per_token": 1.0,
                        "cached_input_usd_per_token": 0.5,
                        "context_length": 1_000_000,
                    },
                    {
                        "model_id": "moonshotai/kimi-k3",
                        "input_usd_per_token": 2.0,
                        "output_usd_per_token": 2.0,
                        "cached_input_usd_per_token": 1.0,
                        "context_length": 1_048_576,
                    },
                ]
            }
        )
    )
    stats = {
        "models": {
            "qwen/qwen3.8-27b": {
                "calls": 2,
                "prompt_tokens": 10,
                "cached_tokens": 2,
                "completion_tokens": 5,
                "reasoning_tokens": 3,
            },
            "moonshotai/kimi-k3": {
                "calls": 1,
                "errors": 1,
                "prompt_tokens": 10,
                "cached_tokens": 2,
                "completion_tokens": 5,
                "reasoning_tokens": 1,
            },
        }
    }
    (tmp_path / "report.json").write_text(
        json.dumps(
            {
                "task_runs": [
                    {
                        "source_task_name": task_name,
                        "provider_spend_usd": 0.3,
                        "routing_stats": stats,
                    }
                ]
            }
        )
    )
    output = tmp_path / "outcomes.jsonl"
    summary_path = tmp_path / "summary.json"
    summary = build_matched_outcomes(
        jobs_root,
        output,
        summary_path,
        panel_path=panel_path,
        switchyard_path=ROOT / "benchmarks/switchyard-gate7.toml",
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert summary["cost_attributed_record_count"] == 2
    assert summary["allocated_provider_cost_total_usd"] == pytest.approx(0.3)
    assert {row["outcome"]["router_reasoning_tokens"] for row in rows} == {1, 3}
    status_by_route = {
        row["model"]["route_id"]: row["outcome"]["status"] for row in rows
    }
    assert status_by_route["gate7/fixed-qwen"] == "verified"
    assert status_by_route["gate7/fixed-kimi"] == "verified"
    assert sum(
        row["outcome"]["allocated_provider_cost_usd"] for row in rows
    ) == pytest.approx(0.3)
    assert summary["learning_contract"]["scoring_unit"] == "task-model-pair"
    assert all(row["schema_version"] == "matched-model-outcome.v1" for row in rows)
    assert {
        row["model"]["candidate_features"]["catalog_source"] for row in rows
    } == {"test-catalog"}
    assert all(
        row["model"]["candidate_features"]["calibration"]["status"]
        == "catalog_only"
        for row in rows
    )
    assert all(
        "model_id" not in row["model"]["candidate_features"] for row in rows
    )
    assert all(
        row["model"]["candidate_features"]["request_timeout_seconds"] is None
        for row in rows
    )
    assert all(
        row["model"]["candidate_features"]["request_retry_attempts"] is None
        for row in rows
    )
    assert all(
        row["model"]["candidate_features"]["output_length_retry_attempts"]
        is None
        for row in rows
    )


def test_matched_outcomes_can_suppress_overlapping_provider_spend(
    tmp_path: Path,
) -> None:
    source_panel = next(
        json.loads(line)
        for line in (
            ROOT / "data/supervisor/terminal-bench-pro-panel-v0.jsonl"
        ).read_text().splitlines()
        if json.loads(line)["wave"] == 1
    )
    panel_path = tmp_path / "panel.jsonl"
    panel_path.write_text(json.dumps(source_panel) + "\n")
    task_name = source_panel["source_task_name"]
    route_id = "gate7/fixed-qwen"
    trial_root = tmp_path / "jobs" / "job" / "trial"
    trial_root.mkdir(parents=True)
    (trial_root / "result.json").write_text(
        json.dumps(_trial(task_name, route_id, "trial", 1.0))
    )
    (tmp_path / "run-manifest.json").write_text(
        json.dumps(
            {
                "config": {"route_ids": [route_id]},
                "frozen_inputs": {"selected_task_names": [task_name]},
            }
        )
    )
    (tmp_path / "pricing-snapshot.json").write_text(
        json.dumps(
            {
                "captured_at": "2026-08-25T00:00:00+00:00",
                "source": "test-catalog",
                "models": [
                    {
                        "model_id": "qwen/qwen3.8-27b",
                        "input_usd_per_token": 1.0,
                        "output_usd_per_token": 1.0,
                        "context_length": 1_000_000,
                    }
                ],
            }
        )
    )
    (tmp_path / "report.json").write_text(
        json.dumps(
            {
                "task_runs": [
                    {
                        "source_task_name": task_name,
                        "provider_spend_usd": 99.0,
                        "routing_stats": {
                            "models": {
                                "qwen/qwen3.8-27b": {
                                    "calls": 1,
                                    "prompt_tokens": 10,
                                    "completion_tokens": 5,
                                }
                            }
                        },
                    }
                ]
            }
        )
    )
    output = tmp_path / "outcomes.jsonl"
    summary = build_matched_outcomes(
        tmp_path / "jobs",
        output,
        tmp_path / "summary.json",
        panel_path=panel_path,
        switchyard_path=ROOT / "benchmarks/switchyard-gate7.toml",
        trust_provider_spend=False,
    )
    row = json.loads(output.read_text())
    assert row["outcome"]["estimated_list_cost_usd"] == pytest.approx(15.0)
    assert row["outcome"]["allocated_provider_cost_usd"] is None
    assert row["outcome"]["provider_cost_allocation_method"] is None
    assert summary["cost_attributed_record_count"] == 0
    assert summary["provider_spend_trusted"] is False
    assert summary["learning_contract"]["prediction_targets"] == [
        "outcome.completed",
        "outcome.estimated_list_cost_usd",
        "outcome.duration_seconds",
    ]
    assert summary["learning_contract"]["provider_spend_is_audit_only"] is True
    assert summary["record_split_counts"] == {"development": 1}
