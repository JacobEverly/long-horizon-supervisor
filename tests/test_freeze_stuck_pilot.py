from horizon_supervisor.training.freeze_stuck_pilot import (
    EXACT_MODELS,
    select_ordered_task_pool,
)


def test_task_selection_is_deterministic_outcome_blind_and_development_only() -> None:
    first = select_ordered_task_pool()
    second = select_ordered_task_pool()
    assert first == second
    assert len(first) == 8
    assert all(row["wave"] in {1, 2} for row in first)
    assert all(row["difficulty"] == "hard" for row in first)
    assert [row["source_task_name"] for row in first[:4]] == [
        "implement-gmm-em-cli",
        "implement-chemical-equilibrium-solver",
        "create-valid-message-enc-file",
        "build-grpc-user-profile-service",
    ]


def test_exact_model_roster_excludes_glm_and_nine_billion() -> None:
    assert set(EXACT_MODELS.values()) == {
        "deepseek/deepseek-v4-flash-0731",
        "qwen/qwen3.8-27b",
        "moonshotai/kimi-k3",
    }
