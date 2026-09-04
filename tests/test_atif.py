from horizon_supervisor.benchmark.atif import export_atif
from horizon_supervisor.benchmark.model_catalog import ModelSpec


def test_atif_export_has_required_shape() -> None:
    model = ModelSpec("test/model", "Test", "test", 0.001, 0.002, 1000)
    output = {
        "prompt": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
        ],
        "workspace_dir": "/tmp/run-1",
        "task_id": "task",
        "info": {"difficulty": "easy"},
        "reward": 1.0,
        "tool_defs": [],
        "trajectory": [],
    }
    trajectory = export_atif(output, model)
    assert trajectory["schema_version"] == "ATIF-v1.7"
    assert [step["source"] for step in trajectory["steps"]] == ["system", "user"]
    assert trajectory["final_metrics"]["total_cost_usd"] == 0
