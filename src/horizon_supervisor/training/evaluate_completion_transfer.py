from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from horizon_supervisor.training.completion_baseline import (
    NUMERIC_FEATURES,
    _metrics,
    _predict_full,
)

DEFAULT_MODEL = Path("artifacts/training/completion-baseline-v2-sealed.joblib")
DEFAULT_CHECKPOINTS = Path("data/supervisor/swe-pivot-transfer-checkpoints-v0.jsonl")
DEFAULT_CONTRACT = Path("benchmarks/swe-transfer-acceptance-v0.json")
DEFAULT_REPORT = Path("artifacts/training/swe-transfer-evaluation-v0.json")
BATCH_SIZE = 5_000


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_examples(
    checkpoints_path: Path,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, list[str]]:
    texts = []
    numeric = []
    labels = []
    pass_rates = []
    task_ids = []
    with checkpoints_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            online = row["input"]
            texts.append(online["terminal_tail"])
            numeric.append([float(online[name]) for name in NUMERIC_FEATURES])
            labels.append(int(row["target"]["task_complete"]))
            pass_rates.append(float(row["audit_only"]["pass_rate"]))
            task_ids.append(row["task_id"])
    return (
        texts,
        np.asarray(numeric, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
        np.asarray(pass_rates, dtype=np.float64),
        task_ids,
    )


def _predict_batches(
    artifact: dict[str, Any], texts: list[str], numeric: np.ndarray
) -> np.ndarray:
    probabilities = []
    for start in range(0, len(texts), BATCH_SIZE):
        end = start + BATCH_SIZE
        probabilities.append(
            _predict_full(
                artifact["vectorizer"],
                artifact["numeric_scaler"],
                artifact["model"],
                texts[start:end],
                numeric[start:end],
            )
        )
    return np.concatenate(probabilities)


def _bootstrap_by_task(
    labels: np.ndarray,
    probabilities: np.ndarray,
    task_ids: list[str],
    threshold: float,
    *,
    replications: int = 200,
) -> dict[str, dict[str, float]]:
    indices_by_task: dict[str, list[int]] = defaultdict(list)
    for index, task_id in enumerate(task_ids):
        indices_by_task[task_id].append(index)
    unique_tasks = sorted(indices_by_task)
    rng = np.random.default_rng(17)
    values: dict[str, list[float]] = defaultdict(list)
    for _ in range(replications):
        sampled_tasks = rng.choice(unique_tasks, size=len(unique_tasks), replace=True)
        sampled_indices = np.concatenate(
            [np.asarray(indices_by_task[task_id]) for task_id in sampled_tasks]
        )
        metrics = _metrics(
            labels[sampled_indices], probabilities[sampled_indices], threshold
        )
        for name in ("average_precision", "roc_auc", "f1"):
            values[name].append(float(metrics[name]))
    return {
        name: {
            "low_95": float(np.quantile(metric_values, 0.025)),
            "median": float(np.quantile(metric_values, 0.5)),
            "high_95": float(np.quantile(metric_values, 0.975)),
        }
        for name, metric_values in sorted(values.items())
    }


def evaluate_completion_transfer(
    model_path: Path = DEFAULT_MODEL,
    checkpoints_path: Path = DEFAULT_CHECKPOINTS,
    contract_path: Path = DEFAULT_CONTRACT,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    model_sha256 = _sha256_file(model_path)
    if model_sha256 != contract["model"]["sha256"]:
        raise RuntimeError(
            f"model digest mismatch: expected {contract['model']['sha256']}, "
            f"got {model_sha256}"
        )
    artifact = joblib.load(model_path)
    texts, numeric, labels, pass_rates, task_ids = _load_examples(checkpoints_path)
    probabilities = _predict_batches(artifact, texts, numeric)
    threshold = float(artifact["threshold"])
    all_metrics = _metrics(labels, probabilities, threshold)

    confidence_min = float(contract["evaluation"]["high_confidence_pass_rate_min"])
    high_confidence = pass_rates >= confidence_min
    high_metrics = _metrics(
        labels[high_confidence], probabilities[high_confidence], threshold
    )
    acceptance = contract["acceptance"]
    checks = {
        "all_rows_average_precision": (
            all_metrics["average_precision"]
            >= acceptance["all_rows_average_precision_min"]
        ),
        "all_rows_roc_auc": (
            all_metrics["roc_auc"] >= acceptance["all_rows_roc_auc_min"]
        ),
        "all_rows_average_precision_multiple_over_prevalence": (
            all_metrics["average_precision"] / all_metrics["prevalence"]
            >= acceptance[
                "all_rows_average_precision_multiple_over_prevalence_min"
            ]
        ),
        "high_confidence_average_precision": (
            high_metrics["average_precision"]
            >= acceptance["high_confidence_average_precision_min"]
        ),
        "high_confidence_roc_auc": (
            high_metrics["roc_auc"]
            >= acceptance["high_confidence_roc_auc_min"]
        ),
    }
    report = {
        "schema_version": "completion-transfer-evaluation.v0",
        "created_at": "2026-08-25",
        "model": {
            "path": str(model_path),
            "sha256": model_sha256,
            "training_on_transfer_source": False,
            "stored_threshold": threshold,
        },
        "data": {
            "path": str(checkpoints_path),
            "sha256": _sha256_file(checkpoints_path),
            "examples": len(labels),
            "tasks": len(set(task_ids)),
            "positive_label": contract["evaluation"]["positive_label"],
        },
        "all_rows": all_metrics,
        "high_confidence": {
            "pass_rate_min": confidence_min,
            "metrics": high_metrics,
        },
        "task_cluster_bootstrap": _bootstrap_by_task(
            labels, probabilities, task_ids, threshold
        ),
        "acceptance_contract": str(contract_path),
        "acceptance_checks": checks,
        "all_acceptance_checks_passed": all(checks.values()),
        "interpretation_guard": (
            "This is zero-shot harness/source transfer for finish-action ranking. It is "
            "not SWE-bench task completion or model-switch evidence."
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate completion-stage transfer")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = evaluate_completion_transfer(
        model_path=args.model,
        checkpoints_path=args.checkpoints,
        contract_path=args.contract,
        report_path=args.report,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
