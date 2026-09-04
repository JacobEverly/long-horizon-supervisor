from horizon_supervisor.benchmark.gate4 import _strategy_analysis
from horizon_supervisor.benchmark.model_catalog import GATE4_MODEL_ROLES, ModelSpec


def test_gate4_roster_pins_current_dated_endpoints() -> None:
    assert list(GATE4_MODEL_ROLES) == [
        "qwen/qwen3.8-27b",
        "deepseek/deepseek-v4-flash-0731",
        "deepseek/deepseek-v4-pro-0813",
        "z-ai/glm-5.3",
        "moonshotai/kimi-k3",
    ]


def test_strategy_separates_more_time_from_switch_value() -> None:
    floor = ModelSpec("floor", "Floor", "floor", 0, 0, 1, tier=0)
    frontier = ModelSpec("frontier", "Frontier", "frontier", 0, 0, 1, tier=3)
    baseline = [
        {
            "model_id": "floor",
            "task_id": "ttl-cache-semantics",
            "success": False,
            "cost_usd": 1.0,
        },
        {
            "model_id": "frontier",
            "task_id": "ttl-cache-semantics",
            "success": True,
            "cost_usd": 3.0,
        },
        {
            "model_id": "floor",
            "task_id": "feature-dependency-plan",
            "success": True,
            "cost_usd": 1.0,
        },
        {
            "model_id": "frontier",
            "task_id": "feature-dependency-plan",
            "success": True,
            "cost_usd": 3.0,
        },
    ]
    recovery = [
        {
            "model_id": "floor",
            "model_label": "Floor",
            "task_id": "ttl-cache-semantics",
            "success": False,
            "cost_usd": 0.5,
        },
        {
            "model_id": "frontier",
            "model_label": "Frontier",
            "task_id": "ttl-cache-semantics",
            "success": True,
            "cost_usd": 0.75,
        },
    ]

    result = _strategy_analysis(baseline, recovery, [floor, frontier])

    assert result["floor_only_passed"] == 1
    assert result["safe_stepdown_tasks"] == 1
    assert result["switch_rescues"] == 1
    assert result["switch_exclusive_rescues"] == 1
    assert result["observed_floor_then_recovery_passed"] == 2
