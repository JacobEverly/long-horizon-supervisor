from horizon_supervisor.benchmark.model_catalog import ModelSpec
from horizon_supervisor.benchmark.runner import (
    aggregate_results,
    render_markdown_report,
    summarize_output,
)


def test_pareto_marks_dominated_model() -> None:
    models = [
        ModelSpec("cheap", "Cheap", "floor", 0, 0, 1),
        ModelSpec("costly", "Costly", "ceiling", 0, 0, 1),
    ]
    runs = [
        {
            "model_id": "cheap",
            "success": True,
            "cost_usd": 1.0,
            "duration_seconds": 1.0,
            "turns": 1,
        },
        {
            "model_id": "costly",
            "success": True,
            "cost_usd": 2.0,
            "duration_seconds": 1.0,
            "turns": 1,
        },
    ]
    aggregate = aggregate_results(runs, models)
    assert aggregate[0]["pareto_efficient"] is True
    assert aggregate[1]["pareto_efficient"] is False


def test_markdown_report_includes_budget_and_models() -> None:
    report = {
        "run_count": 1,
        "model_count": 1,
        "task_count": 1,
        "estimated_spend_usd": 0.01,
        "authorized_budget_usd": 50.0,
        "aggregates": [
            {
                "label": "Test",
                "passed": 1,
                "runs": 1,
                "completion_rate": 1.0,
                "average_cost_usd": 0.01,
                "average_turns": 2.0,
                "average_duration_seconds": 3.0,
                "pareto_efficient": True,
            }
        ],
        "runs": [
            {
                "model_label": "Test",
                "task_id": "task",
                "difficulty": "easy",
                "success": True,
                "cost_usd": 0.01,
                "turns": 2,
                "tool_calls": 1,
                "failure_type": None,
            }
        ],
    }
    rendered = render_markdown_report(report)
    assert "$50.00" in rendered
    assert "Test" in rendered


def test_summary_distinguishes_resource_limits_from_verifier_failures() -> None:
    model = ModelSpec("test", "Test", "test", 0.0, 0.0, 1)
    base = {
        "reward": 0.0,
        "task_id": "task",
        "metrics": {"total_tool_calls": 1},
        "total_tool_calls": 1,
    }

    timeout = summarize_output({**base, "stop_condition": "timeout_reached"}, model)
    turns = summarize_output({**base, "stop_condition": "max_turns_reached"}, model)
    tokens = summarize_output(
        {**base, "stop_condition": "max_total_completion_tokens_reached"}, model
    )

    assert timeout["failure_type"] == "rollout_timeout"
    assert turns["failure_type"] == "turn_limit"
    assert tokens["failure_type"] == "completion_token_limit"
