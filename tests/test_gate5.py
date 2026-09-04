from horizon_supervisor.benchmark.gate5 import analyze_state_arms
from horizon_supervisor.benchmark.model_catalog import ModelSpec


def _run(model_id: str, task_id: str, success: bool, cost: float = 0.1) -> dict:
    return {
        "model_id": model_id,
        "task_id": task_id,
        "success": success,
        "cost_usd": cost,
    }


def test_state_analysis_separates_workspace_and_handoff_value() -> None:
    model = ModelSpec("model", "Model", "test", 0, 0, 1)
    task_ids = ["task-a", "task-b"]
    result = analyze_state_arms(
        task_ids=task_ids,
        models=[model],
        cold_runs=[_run("model", "task-a", True), _run("model", "task-b", False)],
        dirty_runs=[_run("model", "task-a", False), _run("model", "task-b", False)],
        rollback_runs=[_run("model", "task-a", True), _run("model", "task-b", True)],
    )

    assert result["arms"]["cold_restart"]["passed"] == 1
    assert result["arms"]["dirty_continuation"]["passed"] == 0
    assert result["arms"]["clean_rollback"]["passed"] == 2
    assert result["clean_restart_advantages"] == 1
    assert result["dirty_state_penalties"] == 2
    assert result["rollback_rescues"] == 2
    assert result["handoff_value_cases"] == 1
    assert result["handoff_harm_cases"] == 0
