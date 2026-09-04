from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pytest

from horizon_supervisor.training.checkpoint_continuation_risk import (
    _predict_final_artifact,
    _prepare,
    leave_one_task_out_splits,
    load_continuation_examples,
    train_checkpoint_continuation_risk,
)

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "data/supervisor/gate8-development-checkpoints-v0.jsonl"


def _rows(path: Path = CHECKPOINTS) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_checkpoint_examples_are_weighted_and_grouped_by_task() -> None:
    examples = load_continuation_examples(CHECKPOINTS)
    assert len(examples) == 1_154
    assert len({example.task_id for example in examples}) == 35
    assert len({example.trajectory_id for example in examples}) == 139
    weight_by_trajectory: defaultdict[str, float] = defaultdict(float)
    for example in examples:
        weight_by_trajectory[example.trajectory_id] += example.weight
        route_token = example.route_id.replace("/", "_").replace("-", "_")
        assert f"route_{route_token}" in example.text
    assert all(weight == pytest.approx(1.0) for weight in weight_by_trajectory.values())

    data = _prepare(examples)
    splits = leave_one_task_out_splits(data)
    assert len(splits) == 35
    for train_indices, test_indices in splits:
        train_tasks = set(data.task_ids[train_indices].tolist())
        test_tasks = set(data.task_ids[test_indices].tolist())
        assert len(test_tasks) == 1
        assert train_tasks.isdisjoint(test_tasks)


def test_continuation_loader_rejects_non_development_rows(tmp_path: Path) -> None:
    row = _rows()[0]
    row["record_split"] = "held_out"
    path = tmp_path / "heldout.jsonl"
    _write_rows(path, [row])
    with pytest.raises(ValueError, match="development rows only"):
        load_continuation_examples(path)


def test_continuation_loader_rejects_future_or_target_fields_in_observation(
    tmp_path: Path,
) -> None:
    row = _rows()[0]
    row["observation"]["remaining_output_tokens"] = 999
    path = tmp_path / "leaking.jsonl"
    _write_rows(path, [row])
    with pytest.raises(ValueError, match="pre-turn contract"):
        load_continuation_examples(path)


def test_continuation_loader_does_not_accept_switch_restart_or_stop_labels(
    tmp_path: Path,
) -> None:
    for action_name in ("switch_model", "restart_clean", "stop"):
        row = _rows()[0]
        action = next(
            item
            for item in row["available_actions"]
            if item["action"] == action_name
        )
        action["outcome"] = {"completed": True}
        path = tmp_path / f"labeled-{action_name}.jsonl"
        _write_rows(path, [row])
        with pytest.raises(ValueError, match="unobserved and unlabeled"):
            load_continuation_examples(path)


def test_small_grouped_model_serializes_and_reloads_deterministically(
    tmp_path: Path,
) -> None:
    rows = _rows()
    rows_by_task: defaultdict[str, list[dict]] = defaultdict(list)
    label_by_task: dict[str, set[bool]] = defaultdict(set)
    for row in rows:
        rows_by_task[row["task_id"]].append(row)
        continuation = next(
            action
            for action in row["available_actions"]
            if action["action"] == "continue_same"
        )
        label_by_task[row["task_id"]].add(continuation["outcome"]["completed"])

    positive_tasks = sorted(
        task_id for task_id, labels in label_by_task.items() if True in labels
    )[:3]
    negative_tasks = sorted(
        task_id for task_id, labels in label_by_task.items() if labels == {False}
    )[:3]
    selected_tasks = set(positive_tasks + negative_tasks)
    subset = [row for row in rows if row["task_id"] in selected_tasks]
    checkpoints_path = tmp_path / "development-checkpoints.jsonl"
    _write_rows(checkpoints_path, subset)
    model_path = tmp_path / "continuation.joblib"
    report_path = tmp_path / "continuation.json"

    report = train_checkpoint_continuation_risk(
        checkpoints_path=checkpoints_path,
        model_path=model_path,
        report_path=report_path,
        candidates=(0.25,),
        inner_splits=2,
    )

    assert report["data"]["tasks"] == 6
    assert report["evaluation_contract"]["outer_folds"] == 6
    assert report["evaluation_contract"][
        "all_outer_train_test_groups_disjoint"
    ] is True
    assert report["artifact"]["reload_predictions_match"] is True
    assert set(report["baselines"]) == {
        "constant_training_prevalence",
        "turn_index_only",
        "hard_coded_late_or_error_rule",
    }
    assert report["action_boundary"] == {
        "trained_action": "continue_same",
        "switch_model_outcomes_trained": False,
        "restart_clean_outcomes_trained": False,
        "stop_outcomes_trained": False,
    }
    examples = load_continuation_examples(checkpoints_path)
    first_load = joblib.load(model_path)
    second_load = joblib.load(model_path)
    assert np.array_equal(
        _predict_final_artifact(first_load, examples),
        _predict_final_artifact(second_load, examples),
    )
