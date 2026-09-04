from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

DEFAULT_CHECKPOINTS = Path("data/supervisor/terminal-pivot-checkpoints-v0.jsonl")
DEFAULT_TASKS = Path("data/supervisor/terminal-pivot-tasks-v0.jsonl")
DEFAULT_REPORT = Path("artifacts/training/completion-baseline-v0.json")
DEFAULT_MODEL = Path("artifacts/training/completion-baseline-v0.joblib")
SEED = 17

NUMERIC_FEATURES = [
    "turn_index",
    "history_message_count",
    "prior_assistant_turn_count",
    "prior_user_update_count",
    "observed_history_chars",
    "terminal_chars",
    "terminal_lines",
    "error_signal_count",
    "pass_signal_count",
    "test_signal_count",
    "shell_prompt_count",
    "terminal_tail_truncated",
]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_task_ids(path: Path) -> set[str]:
    task_ids = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            task_ids.add(row["task_id"])
    return task_ids


def _load_examples(checkpoints_path: Path, tasks_path: Path) -> list[dict[str, Any]]:
    task_ids = _load_task_ids(tasks_path)
    examples = []
    with checkpoints_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            task_id = row["task_id"]
            if task_id not in task_ids:
                raise ValueError(f"checkpoint references missing task {task_id}")
            online = row["input"]
            examples.append(
                {
                    "checkpoint_id": row["checkpoint_id"],
                    "task_id": task_id,
                    "split": row["record_split"],
                    "label": int(row["target"]["task_complete"]),
                    "source_id": row["provenance"]["source_id"],
                    "text": online["terminal_tail"],
                    "numeric": [float(online[name]) for name in NUMERIC_FEATURES],
                }
            )
    return examples


def _arrays(
    examples: list[dict[str, Any]], split: str
) -> tuple[list[str], np.ndarray, np.ndarray, list[str], list[str]]:
    rows = [example for example in examples if example["split"] == split]
    texts = [row["text"] for row in rows]
    numeric = np.asarray([row["numeric"] for row in rows], dtype=np.float64)
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    task_ids = [row["task_id"] for row in rows]
    source_ids = [row["source_id"] for row in rows]
    return texts, numeric, labels, task_ids, source_ids


def _task_rank(task_id: str) -> str:
    return hashlib.sha256(f"completion-baseline-v0|{task_id}".encode()).hexdigest()


def _task_subset(examples: list[dict[str, Any]], fraction: float) -> list[dict[str, Any]]:
    tasks_by_source: dict[str, set[str]] = defaultdict(set)
    for row in examples:
        if row["split"] == "train":
            tasks_by_source[row.get("source_id", "all-sources")].add(row["task_id"])
    selected: set[str] = set()
    for source_tasks in tasks_by_source.values():
        ranked = sorted(source_tasks, key=_task_rank)
        count = max(1, round(len(ranked) * fraction))
        selected.update(ranked[:count])
    return [
        row
        for row in examples
        if row["split"] != "train" or row["task_id"] in selected
    ]


def _best_f1_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> float:
    weights = (
        np.ones(labels.shape, dtype=np.float64)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float64)
    )
    order = np.argsort(-probabilities, kind="stable")
    sorted_labels = labels[order]
    sorted_probabilities = probabilities[order]
    sorted_weights = weights[order]
    true_positives = np.cumsum(sorted_weights * sorted_labels)
    false_positives = np.cumsum(sorted_weights * (1 - sorted_labels))
    total_positives = float((weights * labels).sum())
    false_negatives = total_positives - true_positives
    denominator = 2 * true_positives + false_positives + false_negatives
    f1_values = np.divide(
        2 * true_positives,
        denominator,
        out=np.zeros_like(true_positives),
        where=denominator > 0,
    )
    precision_values = np.divide(
        true_positives,
        true_positives + false_positives,
        out=np.zeros_like(true_positives),
        where=(true_positives + false_positives) > 0,
    )
    recall_values = (
        true_positives / total_positives
        if total_positives > 0
        else np.zeros_like(true_positives)
    )
    boundary = np.r_[
        sorted_probabilities[:-1] != sorted_probabilities[1:],
        True,
    ]
    candidate_indices = np.flatnonzero(boundary)
    best_index = max(
        candidate_indices,
        key=lambda index: (
            f1_values[index],
            precision_values[index],
            recall_values[index],
        ),
    )
    return float(sorted_probabilities[best_index])


def _metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float | int]:
    predictions = probabilities >= threshold
    return {
        "examples": int(labels.size),
        "positives": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "threshold": threshold,
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
    }


def _metrics_by_source(
    labels: np.ndarray,
    probabilities: np.ndarray,
    source_ids: list[str],
    threshold: float,
) -> dict[str, dict[str, float | int]]:
    result = {}
    source_array = np.asarray(source_ids)
    for source_id in sorted(set(source_ids)):
        mask = source_array == source_id
        result[source_id] = _metrics(labels[mask], probabilities[mask], threshold)
    return result


def _fit_numeric_model(
    train_numeric: np.ndarray,
    train_labels: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler()
    transformed = scaler.fit_transform(np.log1p(np.maximum(train_numeric, 0)))
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1_000,
        random_state=SEED,
        solver="liblinear",
    )
    model.fit(transformed, train_labels, sample_weight=sample_weight)
    return scaler, model


def _fit_full_model(
    train_texts: list[str],
    train_numeric: np.ndarray,
    train_labels: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> tuple[TfidfVectorizer, StandardScaler, LogisticRegression]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=30_000,
        min_df=3,
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents="unicode",
    )
    text_matrix = vectorizer.fit_transform(train_texts)
    scaler = StandardScaler()
    numeric_matrix = scaler.fit_transform(np.log1p(np.maximum(train_numeric, 0)))
    matrix = hstack([text_matrix, csr_matrix(numeric_matrix)], format="csr")
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1_000,
        random_state=SEED,
        solver="liblinear",
    )
    model.fit(matrix, train_labels, sample_weight=sample_weight)
    return vectorizer, scaler, model


def _predict_numeric(
    scaler: StandardScaler, model: LogisticRegression, numeric: np.ndarray
) -> np.ndarray:
    transformed = scaler.transform(np.log1p(np.maximum(numeric, 0)))
    return model.predict_proba(transformed)[:, 1]


def _predict_full(
    vectorizer: TfidfVectorizer,
    scaler: StandardScaler,
    model: LogisticRegression,
    texts: list[str],
    numeric: np.ndarray,
) -> np.ndarray:
    text_matrix = vectorizer.transform(texts)
    numeric_matrix = scaler.transform(np.log1p(np.maximum(numeric, 0)))
    matrix = hstack([text_matrix, csr_matrix(numeric_matrix)], format="csr")
    return model.predict_proba(matrix)[:, 1]


def _source_balanced_weights(source_ids: list[str]) -> np.ndarray:
    counts = Counter(source_ids)
    source_count = len(counts)
    total = len(source_ids)
    return np.asarray(
        [total / (source_count * counts[source_id]) for source_id in source_ids],
        dtype=np.float64,
    )


def _target_source_weights(
    source_ids: list[str], target_source_id: str, target_share: float
) -> np.ndarray:
    counts = Counter(source_ids)
    if target_source_id not in counts:
        raise ValueError(f"target source {target_source_id!r} is not present")
    if len(counts) < 2:
        raise ValueError("targeted source weighting requires at least two sources")
    if not 0 < target_share < 1:
        raise ValueError("target_share must be strictly between zero and one")
    other_share = (1 - target_share) / (len(counts) - 1)
    total = len(source_ids)
    weights = []
    for source_id in source_ids:
        share = target_share if source_id == target_source_id else other_share
        weights.append(total * share / counts[source_id])
    return np.asarray(weights, dtype=np.float64)


def _source_weight_shares(
    source_ids: list[str], sample_weight: np.ndarray | None
) -> dict[str, float]:
    weights = (
        np.ones(len(source_ids), dtype=np.float64)
        if sample_weight is None
        else sample_weight
    )
    total = float(weights.sum())
    shares: Counter[str] = Counter()
    for source_id, weight in zip(source_ids, weights, strict=True):
        shares[source_id] += float(weight)
    return {source_id: weight / total for source_id, weight in sorted(shares.items())}


def train_completion_baseline(
    checkpoints_path: Path = DEFAULT_CHECKPOINTS,
    tasks_path: Path = DEFAULT_TASKS,
    report_path: Path = DEFAULT_REPORT,
    model_path: Path = DEFAULT_MODEL,
    *,
    source_weighting: str = "none",
    include_learning_curve: bool = True,
    target_source_id: str | None = None,
    target_source_share: float | None = None,
) -> dict[str, Any]:
    examples = _load_examples(checkpoints_path, tasks_path)
    val_texts, val_numeric, val_labels, val_task_ids, val_source_ids = _arrays(
        examples, "validation"
    )
    test_texts, test_numeric, test_labels, test_task_ids, test_source_ids = _arrays(
        examples, "internal_test"
    )
    sealed_texts, sealed_numeric, sealed_labels, sealed_task_ids, sealed_source_ids = (
        _arrays(examples, "sealed_test")
    )

    train_texts, train_numeric, train_labels, train_task_ids, train_source_ids = _arrays(
        examples, "train"
    )
    if source_weighting not in {"none", "balanced", "targeted"}:
        raise ValueError("source_weighting must be 'none', 'balanced', or 'targeted'")
    if source_weighting == "targeted":
        if target_source_id is None or target_source_share is None:
            raise ValueError("targeted weighting requires a source id and share")
        train_weights = _target_source_weights(
            train_source_ids, target_source_id, target_source_share
        )
        validation_weights = _target_source_weights(
            val_source_ids, target_source_id, target_source_share
        )
    elif source_weighting == "balanced":
        train_weights = _source_balanced_weights(train_source_ids)
        validation_weights = _source_balanced_weights(val_source_ids)
    else:
        train_weights = None
        validation_weights = None
    prevalence = float(train_labels.mean())
    constant_probabilities = np.full(test_labels.shape, prevalence)
    constant_metrics = _metrics(test_labels, constant_probabilities, 0.5)
    sealed_constant_probabilities = np.full(sealed_labels.shape, prevalence)

    turn_scaler, turn_model = _fit_numeric_model(
        train_numeric[:, :1], train_labels, train_weights
    )
    turn_val = _predict_numeric(turn_scaler, turn_model, val_numeric[:, :1])
    turn_threshold = _best_f1_threshold(val_labels, turn_val, validation_weights)
    turn_test = _predict_numeric(turn_scaler, turn_model, test_numeric[:, :1])
    turn_sealed = _predict_numeric(turn_scaler, turn_model, sealed_numeric[:, :1])

    numeric_scaler, numeric_model = _fit_numeric_model(
        train_numeric, train_labels, train_weights
    )
    numeric_val = _predict_numeric(numeric_scaler, numeric_model, val_numeric)
    numeric_threshold = _best_f1_threshold(
        val_labels, numeric_val, validation_weights
    )
    numeric_test = _predict_numeric(numeric_scaler, numeric_model, test_numeric)
    numeric_sealed = _predict_numeric(numeric_scaler, numeric_model, sealed_numeric)

    vectorizer, full_scaler, full_model = _fit_full_model(
        train_texts, train_numeric, train_labels, train_weights
    )
    full_val = _predict_full(
        vectorizer, full_scaler, full_model, val_texts, val_numeric
    )
    full_threshold = _best_f1_threshold(val_labels, full_val, validation_weights)
    full_test = _predict_full(
        vectorizer, full_scaler, full_model, test_texts, test_numeric
    )
    full_sealed = _predict_full(
        vectorizer, full_scaler, full_model, sealed_texts, sealed_numeric
    )

    learning_curve = []
    fractions = (0.1, 0.25, 0.5) if include_learning_curve else ()
    for fraction in fractions:
        subset = _task_subset(examples, fraction)
        sub_texts, sub_numeric, sub_labels, sub_task_ids, sub_source_ids = _arrays(
            subset, "train"
        )
        if source_weighting == "targeted":
            sub_weights = _target_source_weights(
                sub_source_ids, str(target_source_id), float(target_source_share)
            )
        elif source_weighting == "balanced":
            sub_weights = _source_balanced_weights(sub_source_ids)
        else:
            sub_weights = None
        sub_vectorizer, sub_scaler, sub_model = _fit_full_model(
            sub_texts, sub_numeric, sub_labels, sub_weights
        )
        sub_val = _predict_full(
            sub_vectorizer, sub_scaler, sub_model, val_texts, val_numeric
        )
        threshold = _best_f1_threshold(val_labels, sub_val, validation_weights)
        sub_test = _predict_full(
            sub_vectorizer, sub_scaler, sub_model, test_texts, test_numeric
        )
        learning_curve.append(
            {
                "train_task_fraction": fraction,
                "train_tasks": len(set(sub_task_ids)),
                "train_examples": len(sub_labels),
                "train_positives": int(sub_labels.sum()),
                "internal_test": _metrics(test_labels, sub_test, threshold),
                "internal_test_by_source": _metrics_by_source(
                    test_labels, sub_test, test_source_ids, threshold
                ),
            }
        )
    learning_curve.append(
        {
            "train_task_fraction": 1.0,
            "train_tasks": len(set(train_task_ids)),
            "train_examples": len(train_labels),
            "train_positives": int(train_labels.sum()),
            "internal_test": _metrics(test_labels, full_test, full_threshold),
            "internal_test_by_source": _metrics_by_source(
                test_labels, full_test, test_source_ids, full_threshold
            ),
        }
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "schema_version": "completion-baseline-model.v0",
            "numeric_features": NUMERIC_FEATURES,
            "terminal_tail_chars": 4_096,
            "threshold": full_threshold,
            "source_weighting": source_weighting,
            "target_source_id": target_source_id,
            "target_source_share": target_source_share,
            "vectorizer": vectorizer,
            "numeric_scaler": full_scaler,
            "model": full_model,
        },
        model_path,
    )
    reloaded = joblib.load(model_path)
    reloaded_probabilities = _predict_full(
        reloaded["vectorizer"],
        reloaded["numeric_scaler"],
        reloaded["model"],
        test_texts,
        test_numeric,
    )
    if not np.allclose(full_test, reloaded_probabilities):
        raise RuntimeError("serialized model predictions changed after reload")

    report = {
        "schema_version": "completion-baseline-report.v0",
        "created_at": "2026-08-25",
        "objective": "Predict whether the next teacher action marks the task complete.",
        "data": {
            "checkpoint_sha256": _sha256_file(checkpoints_path),
            "task_table_sha256": _sha256_file(tasks_path),
            "split_group": "normalized task-description SHA-256 across sources",
            "train_examples": len(train_labels),
            "train_tasks": len(set(train_task_ids)),
            "validation_examples": len(val_labels),
            "validation_tasks": len(set(val_task_ids)),
            "internal_test_examples": len(test_labels),
            "internal_test_tasks": len(set(test_task_ids)),
            "sealed_test_examples": len(sealed_labels),
            "sealed_test_tasks": len(set(sealed_task_ids)),
        },
        "features": {
            "text": "already-observed 4,096-character terminal tail",
            "numeric": NUMERIC_FEATURES,
            "future_or_reference_fields_used": False,
        },
        "training_policy": {
            "source_weighting": source_weighting,
            "validation_threshold_weighting": source_weighting,
            "learning_curve_included": include_learning_curve,
            "target_source_id": target_source_id,
            "target_source_share": target_source_share,
            "effective_train_source_shares": _source_weight_shares(
                train_source_ids, train_weights
            ),
        },
        "baselines": {
            "constant_train_prevalence": constant_metrics,
            "turn_index_only": _metrics(test_labels, turn_test, turn_threshold),
            "structured_online_features": _metrics(
                test_labels, numeric_test, numeric_threshold
            ),
            "text_plus_structured": _metrics(test_labels, full_test, full_threshold),
        },
        "learning_curve": learning_curve,
        "text_plus_structured_by_source": _metrics_by_source(
            test_labels, full_test, test_source_ids, full_threshold
        ),
        "sealed_evaluation": {
            "constant_train_prevalence": _metrics(
                sealed_labels, sealed_constant_probabilities, 0.5
            ),
            "turn_index_only": _metrics(
                sealed_labels, turn_sealed, turn_threshold
            ),
            "structured_online_features": _metrics(
                sealed_labels, numeric_sealed, numeric_threshold
            ),
            "text_plus_structured": _metrics(
                sealed_labels, full_sealed, full_threshold
            ),
            "text_plus_structured_by_source": _metrics_by_source(
                sealed_labels, full_sealed, sealed_source_ids, full_threshold
            ),
        },
        "source_counts": {
            "train": dict(sorted(Counter(train_source_ids).items())),
            "validation": dict(sorted(Counter(val_source_ids).items())),
            "internal_test": dict(sorted(Counter(test_source_ids).items())),
            "sealed_test": dict(sorted(Counter(sealed_source_ids).items())),
        },
        "artifact": {
            "path": str(model_path),
            "sha256": _sha256_file(model_path),
            "reload_predictions_match": True,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "seed": SEED,
        },
        "interpretation_guard": (
            "This validates stage/completion signal on unseen source tasks. It does not "
            "validate model switching, cost optimization, failure recovery, or "
            "Terminal-Bench completion."
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the completion-recognition baseline")
    parser.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--source-weighting",
        choices=("none", "balanced", "targeted"),
        default="none",
    )
    parser.add_argument("--target-source-id")
    parser.add_argument("--target-source-share", type=float)
    parser.add_argument("--skip-learning-curve", action="store_true")
    args = parser.parse_args()
    report = train_completion_baseline(
        checkpoints_path=args.checkpoints,
        tasks_path=args.tasks,
        report_path=args.report,
        model_path=args.model,
        source_weighting=args.source_weighting,
        include_learning_curve=not args.skip_learning_curve,
        target_source_id=args.target_source_id,
        target_source_share=args.target_source_share,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
