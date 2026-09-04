from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from horizon_supervisor.benchmark.gate8 import Gate8PilotConfig, validate_pilot_inputs

ROOT = Path(__file__).resolve().parents[1]


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_wave_two_sentinel_screen_partitions_the_frozen_wave() -> None:
    contract = json.loads(
        (ROOT / "benchmarks/gate8-wave2-sentinel-screen-v0.json").read_text()
    )
    panel_tasks = {
        row["source_task_name"]
        for row in _jsonl(
            ROOT / "data/supervisor/terminal-bench-pro-panel-v0.jsonl"
        )
        if row["wave"] == 2
    }
    full = set(contract["already_full_matrix_tasks"])
    in_flight = {contract["in_flight_full_matrix_task"]}
    screen = {
        task
        for tranche in contract["tranches"]
        for task in tranche["task_names"]
    }

    assert len(full) == 2
    assert len(screen) == contract["screen_task_count"] == 15
    assert not full & in_flight
    assert not full & screen
    assert not in_flight & screen
    assert full | in_flight | screen == panel_tasks


def test_sentinel_tranches_keep_the_frozen_difficulty_mix() -> None:
    contract = json.loads(
        (ROOT / "benchmarks/gate8-wave2-sentinel-screen-v0.json").read_text()
    )
    panel = {
        row["source_task_name"]: row
        for row in _jsonl(
            ROOT / "data/supervisor/terminal-bench-pro-panel-v0.jsonl"
        )
        if row["wave"] == 2
    }
    observed = defaultdict(int)
    for tranche in contract["tranches"]:
        counts = defaultdict(int)
        for task_name in tranche["task_names"]:
            difficulty = panel[task_name]["difficulty"]
            counts[difficulty] += 1
            observed[difficulty] += 1
        assert dict(counts) == tranche["difficulty_counts"]

    assert dict(observed) == {"hard": 4, "medium": 11}


def test_sentinel_prior_evidence_is_reproducible() -> None:
    contract = json.loads(
        (ROOT / "benchmarks/gate8-wave2-sentinel-screen-v0.json").read_text()
    )
    rows = _jsonl(ROOT / contract["prior_evidence"]["dataset"])
    outcomes: dict[str, dict[str, bool]] = defaultdict(dict)
    for row in rows:
        outcomes[row["task"]["source_task_name"]][row["model"]["route_id"]] = row[
            "outcome"
        ]["completed"]

    discriminating = 0
    sentinel_disagreements = 0
    missed = 0
    for routes in outcomes.values():
        is_discriminating = len(set(routes.values())) > 1
        sentinel_disagrees = (
            routes["gate7/fixed-qwen"] != routes["gate7/fixed-glm"]
        )
        discriminating += is_discriminating
        sentinel_disagreements += sentinel_disagrees
        missed += is_discriminating and not sentinel_disagrees

    evidence = contract["prior_evidence"]
    assert len(outcomes) == evidence["full_matrix_task_count"] == 17
    assert discriminating == evidence["discriminating_task_count"] == 5
    assert sentinel_disagreements == evidence["sentinel_disagreement_count"] == 5
    assert missed == evidence["sentinel_missed_discriminating_task_count"] == 0


def test_sentinel_tranche_one_budget_locks_routes_tasks_and_controls() -> None:
    budget_path = ROOT / "benchmarks/gate8-wave2-sentinel-tranche1-budget-v0.json"
    budget = json.loads(budget_path.read_text())
    manifest_path = ROOT / budget["screen_manifest"]
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == budget[
        "screen_manifest_sha256"
    ]

    execution = budget["execution_contract"]
    frozen = validate_pilot_inputs(
        Gate8PilotConfig(
            artifacts_root=ROOT / "artifacts/official",
            wave=2,
            tasks_path=ROOT / "data/supervisor/terminal-bench-pro-wave-2/tasks",
            panel_path=ROOT / "data/supervisor/terminal-bench-pro-panel-v0.jsonl",
            switchyard_config_path=ROOT / "benchmarks/switchyard-gate7.toml",
            budget_contract_path=budget_path,
            route_ids=tuple(execution["route_ids"]),
            include_task_names=tuple(execution["selected_task_names"]),
            **execution["run_controls"],
        )
    )

    assert frozen["selected_task_count"] == 5
    assert frozen["trial_count"] == budget["trial_count"] == 10
