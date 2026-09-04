from horizon_supervisor.training.freeze_stuck_confirmatory import (
    EXACT_MODELS,
    EXPECTED_DETECTOR_SHA256,
    ROOT,
    STATIC_SNAPSHOT_EXCLUSIONS,
    _route_endpoints,
    _sha256,
    select_ordered_task_pool,
)


def test_detector_hash_is_the_original_frozen_v0() -> None:
    assert _sha256(ROOT / "src/horizon_supervisor/stuck_detector.py") == (
        EXPECTED_DETECTOR_SHA256
    )


def test_route_endpoints_are_exactly_the_three_predeclared_models() -> None:
    assert _route_endpoints() == EXACT_MODELS


def test_pool_is_deterministic_untouched_and_stratified() -> None:
    first = select_ordered_task_pool()
    second = select_ordered_task_pool()
    assert first == second
    names = [row["source_task_name"] for row in first]
    assert len(names) == len(set(names)) == 21
    assert not set(names) & set(STATIC_SNAPSHOT_EXCLUSIONS)
    assert {row["wave"] for row in first} <= {1, 2}
    assert {row["difficulty"] for row in first} == {"hard", "medium"}
    assert len({row["category"] for row in first}) >= 7
    assert names[:8] == [
        "bash-ddos-traffic-analyzer",
        "decode-go-ctf-credentials",
        "compute-best-chess-move-san",
        "linear-sem-causal-discovery-intervention",
        "optimize-urdf-robot-for-pybullet",
        "analyze-and-run-encoded-payload",
        "mcts-solver-for-15-puzzle",
        "reproducible-latex-pdf-build-script",
    ]
