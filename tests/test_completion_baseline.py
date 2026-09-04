from __future__ import annotations

import numpy as np

from horizon_supervisor.training.completion_baseline import (
    _best_f1_threshold,
    _metrics,
    _source_balanced_weights,
    _source_weight_shares,
    _target_source_weights,
    _task_subset,
)


def test_threshold_selection_uses_validation_f1() -> None:
    labels = np.asarray([0, 0, 1, 1])
    probabilities = np.asarray([0.1, 0.4, 0.6, 0.9])
    assert _best_f1_threshold(labels, probabilities) == 0.6


def test_metrics_report_imbalance_aware_scores() -> None:
    labels = np.asarray([0, 0, 1, 1])
    probabilities = np.asarray([0.1, 0.2, 0.8, 0.9])
    metrics = _metrics(labels, probabilities, 0.5)
    assert metrics["average_precision"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["f1"] == 1.0


def test_learning_curve_subsets_whole_tasks() -> None:
    examples = [
        {"task_id": "a", "split": "train"},
        {"task_id": "a", "split": "train"},
        {"task_id": "b", "split": "train"},
        {"task_id": "c", "split": "validation"},
    ]
    subset = _task_subset(examples, 0.5)
    retained_train_tasks = {row["task_id"] for row in subset if row["split"] == "train"}
    assert len(retained_train_tasks) == 1
    assert [row for row in subset if row["split"] == "validation"]


def test_learning_curve_stratifies_tasks_by_source() -> None:
    examples = [
        {"task_id": f"a-{index}", "source_id": "a", "split": "train"}
        for index in range(10)
    ] + [
        {"task_id": f"b-{index}", "source_id": "b", "split": "train"}
        for index in range(10)
    ]
    subset = _task_subset(examples, 0.5)
    retained = {
        source: {row["task_id"] for row in subset if row["source_id"] == source}
        for source in ("a", "b")
    }
    assert {source: len(tasks) for source, tasks in retained.items()} == {"a": 5, "b": 5}


def test_source_balanced_weights_give_each_source_equal_total_weight() -> None:
    source_ids = ["large", "large", "large", "small"]
    weights = _source_balanced_weights(source_ids)
    assert np.isclose(weights[:3].sum(), weights[3:].sum())
    assert np.isclose(weights.mean(), 1.0)


def test_weighted_threshold_can_balance_sources() -> None:
    labels = np.asarray([1, 0, 1, 0])
    probabilities = np.asarray([0.9, 0.8, 0.6, 0.1])
    weights = np.asarray([1.0, 1.0, 5.0, 5.0])
    assert _best_f1_threshold(labels, probabilities, weights) == 0.6


def test_target_source_weights_match_requested_mixture() -> None:
    source_ids = ["target", "target", "other", "other", "other"]
    weights = _target_source_weights(source_ids, "target", 0.75)
    shares = _source_weight_shares(source_ids, weights)
    assert np.isclose(weights.mean(), 1.0)
    assert np.isclose(shares["target"], 0.75)
    assert np.isclose(shares["other"], 0.25)
