from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from horizon_supervisor.benchmark.matched_outcomes import LEARNING_VALID_STATUSES

DEFAULT_CHECKPOINTS = Path(
    "data/supervisor/gate8-development-checkpoints-v0.jsonl"
)
DEFAULT_REPORT = Path(
    "artifacts/training/checkpoint-continuation-risk-v0.json"
)
DEFAULT_MODEL = Path(
    "artifacts/training/checkpoint-continuation-risk-v0.joblib"
)
SEED = 17
DEFAULT_CANDIDATES = (0.05, 0.25, 1.0)
HASH_FEATURES = 2**10

NUMERIC_FEATURES = (
    "turn_index",
    "prior_agent_turn_count",
    "prior_tool_call_count",
    "prior_observation_chars",
    "cumulative_input_tokens",
    "cumulative_cache_tokens",
    "cumulative_output_tokens",
    "terminal_chars",
    "terminal_lines",
    "error_signal_count",
    "pass_signal_count",
    "test_signal_count",
    "shell_prompt_count",
    "terminal_tail_truncated",
)
OBSERVATION_FIELDS = frozenset(NUMERIC_FEATURES) | {
    "current_route_id",
    "terminal_tail",
}


@dataclass(frozen=True)
class ContinuationExample:
    checkpoint_id: str
    task_id: str
    leakage_group: str
    trajectory_id: str
    route_id: str
    text: str
    numeric: tuple[float, ...]
    label: int
    weight: float


@dataclass(frozen=True)
class PreparedData:
    examples: tuple[ContinuationExample, ...]
    texts: tuple[str, ...]
    numeric: np.ndarray
    labels: np.ndarray
    weights: np.ndarray
    task_ids: np.ndarray
    text_matrix: Any


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _observed_continuation(row: dict[str, Any]) -> dict[str, Any]:
    actions = row.get("available_actions")
    if not isinstance(actions, list) or not actions:
        raise ValueError("checkpoint has no available actions")
    observed = [action for action in actions if action.get("observed") is True]
    if len(observed) != 1 or observed[0].get("action") != "continue_same":
        raise ValueError("continue_same must be the only observed action")
    continuation = observed[0]
    if continuation.get("outcome") is None:
        raise ValueError("observed continuation is missing its outcome")
    for action in actions:
        if action is continuation:
            continue
        if action.get("observed") is not False or action.get("outcome") is not None:
            raise ValueError(
                "switch, restart, and stop actions must remain unobserved and unlabeled"
            )
        if action.get("action") not in {"switch_model", "restart_clean", "stop"}:
            raise ValueError(f"unsupported unobserved action: {action.get('action')}")
    return continuation


def load_continuation_examples(
    checkpoints_path: Path = DEFAULT_CHECKPOINTS,
) -> list[ContinuationExample]:
    examples = []
    checkpoint_ids: set[str] = set()
    trajectory_identity: dict[str, tuple[str, str, str, int]] = {}
    task_leakage_groups: dict[str, str] = {}
    trajectory_weights: defaultdict[str, float] = defaultdict(float)

    with checkpoints_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON on {checkpoints_path}:{line_number}"
                ) from exc
            if row.get("schema_version") != "supervisor-continuation-checkpoint.v0":
                raise ValueError("unsupported continuation-checkpoint schema")
            if row.get("record_split") != "development":
                raise ValueError("continuation training accepts development rows only")
            checkpoint_id = str(row["checkpoint_id"])
            if checkpoint_id in checkpoint_ids:
                raise ValueError(f"duplicate checkpoint id: {checkpoint_id}")
            checkpoint_ids.add(checkpoint_id)

            observation = row.get("observation")
            if not isinstance(observation, dict):
                raise ValueError(f"checkpoint {checkpoint_id} has no observation")
            unexpected = set(observation) - OBSERVATION_FIELDS
            missing = OBSERVATION_FIELDS - set(observation)
            if unexpected or missing:
                raise ValueError(
                    f"checkpoint observation fields differ from the pre-turn contract; "
                    f"unexpected={sorted(unexpected)}, missing={sorted(missing)}"
                )
            route_id = str(observation["current_route_id"])
            logged_action = row.get("logged_action", {})
            if logged_action != {
                "action": "continue_same",
                "target_route_id": route_id,
            }:
                raise ValueError("logged action must continue the observed current route")
            continuation = _observed_continuation(row)
            if continuation.get("target_route_id") != route_id:
                raise ValueError("observed continuation targets a different route")
            outcome = continuation["outcome"]
            if outcome.get("status") not in LEARNING_VALID_STATUSES:
                raise ValueError("continuation outcome is not learning-valid")
            completed = bool(outcome.get("completed"))
            if completed and outcome.get("status") != "verified":
                raise ValueError("a positive continuation label must be verifier-confirmed")
            label = int(completed and outcome.get("status") == "verified")

            task_id = str(row["task_id"])
            leakage_group = str(row["leakage_group"])
            previous_group = task_leakage_groups.setdefault(task_id, leakage_group)
            if previous_group != leakage_group:
                raise ValueError("one task appears in multiple leakage groups")
            trajectory_id = str(row["trajectory_id"])
            identity = (task_id, leakage_group, route_id, label)
            previous_identity = trajectory_identity.setdefault(trajectory_id, identity)
            if previous_identity != identity:
                raise ValueError("trajectory identity or terminal label changes across turns")

            numeric_values = []
            for name in NUMERIC_FEATURES:
                value = observation[name]
                if isinstance(value, bool):
                    numeric_values.append(float(value))
                elif isinstance(value, (int, float)) and float(value) >= 0:
                    numeric_values.append(float(value))
                else:
                    raise ValueError(
                        f"checkpoint {checkpoint_id} has invalid numeric feature {name}"
                    )
            weight = float(row.get("training", {}).get("trajectory_weight", 0))
            if not np.isfinite(weight) or weight <= 0:
                raise ValueError("trajectory weight must be finite and positive")
            trajectory_weights[trajectory_id] += weight
            terminal_tail = observation["terminal_tail"]
            if not isinstance(terminal_tail, str):
                raise ValueError("terminal_tail must be text")
            route_token = "route_" + "_".join(
                part for part in route_id.replace("/", "_").replace("-", "_").split("_") if part
            )
            examples.append(
                ContinuationExample(
                    checkpoint_id=checkpoint_id,
                    task_id=task_id,
                    leakage_group=leakage_group,
                    trajectory_id=trajectory_id,
                    route_id=route_id,
                    text=f"{route_token}\n{terminal_tail}",
                    numeric=tuple(numeric_values),
                    label=label,
                    weight=weight,
                )
            )

    if not examples:
        raise ValueError("continuation checkpoint dataset is empty")
    invalid_weights = {
        trajectory_id: weight
        for trajectory_id, weight in trajectory_weights.items()
        if abs(weight - 1.0) > 1e-9
    }
    if invalid_weights:
        raise ValueError(
            f"checkpoint weights must sum to one per trajectory: {invalid_weights}"
        )
    task_by_group: dict[str, str] = {}
    for task_id, leakage_group in task_leakage_groups.items():
        other = task_by_group.setdefault(leakage_group, task_id)
        if other != task_id:
            raise ValueError("one leakage group maps to multiple task ids")
    return examples


def _vectorizer() -> HashingVectorizer:
    return HashingVectorizer(
        n_features=HASH_FEATURES,
        alternate_sign=False,
        lowercase=True,
        ngram_range=(1, 2),
        norm="l2",
        strip_accents="unicode",
    )


def _prepare(examples: Sequence[ContinuationExample]) -> PreparedData:
    texts = tuple(example.text for example in examples)
    numeric = np.asarray([example.numeric for example in examples], dtype=np.float64)
    labels = np.asarray([example.label for example in examples], dtype=np.int64)
    weights = np.asarray([example.weight for example in examples], dtype=np.float64)
    task_ids = np.asarray([example.task_id for example in examples], dtype=object)
    text_matrix = _vectorizer().transform(texts)
    return PreparedData(
        examples=tuple(examples),
        texts=texts,
        numeric=numeric,
        labels=labels,
        weights=weights,
        task_ids=task_ids,
        text_matrix=text_matrix,
    )


def _fit_model(
    data: PreparedData,
    indices: np.ndarray,
    *,
    c_value: float,
) -> tuple[StandardScaler, LogisticRegression | float]:
    labels = data.labels[indices]
    weights = data.weights[indices]
    if len(np.unique(labels)) == 1:
        return StandardScaler(), float(labels[0])
    scaler = StandardScaler()
    numeric = scaler.fit_transform(
        np.log1p(np.maximum(data.numeric[indices], 0)),
        sample_weight=weights,
    )
    matrix = hstack(
        [data.text_matrix[indices], csr_matrix(numeric)], format="csr"
    )
    model = LogisticRegression(
        C=c_value,
        max_iter=1_000,
        random_state=SEED,
        solver="liblinear",
    )
    model.fit(matrix, labels, sample_weight=weights)
    return scaler, model


def _predict_model(
    data: PreparedData,
    indices: np.ndarray,
    scaler: StandardScaler,
    model: LogisticRegression | float,
) -> np.ndarray:
    if isinstance(model, float):
        return np.full(len(indices), model, dtype=np.float64)
    numeric = scaler.transform(np.log1p(np.maximum(data.numeric[indices], 0)))
    matrix = hstack(
        [data.text_matrix[indices], csr_matrix(numeric)], format="csr"
    )
    return model.predict_proba(matrix)[:, 1]


def _fit_turn_only(
    data: PreparedData, indices: np.ndarray
) -> tuple[StandardScaler, LogisticRegression | float]:
    labels = data.labels[indices]
    weights = data.weights[indices]
    if len(np.unique(labels)) == 1:
        return StandardScaler(), float(labels[0])
    scaler = StandardScaler()
    turns = scaler.fit_transform(
        np.log1p(data.numeric[indices, :1]), sample_weight=weights
    )
    model = LogisticRegression(
        C=1.0,
        max_iter=1_000,
        random_state=SEED,
        solver="liblinear",
    )
    model.fit(turns, labels, sample_weight=weights)
    return scaler, model


def _predict_turn_only(
    data: PreparedData,
    indices: np.ndarray,
    scaler: StandardScaler,
    model: LogisticRegression | float,
) -> np.ndarray:
    if isinstance(model, float):
        return np.full(len(indices), model, dtype=np.float64)
    turns = scaler.transform(np.log1p(data.numeric[indices, :1]))
    return model.predict_proba(turns)[:, 1]


def _weighted_prevalence(
    labels: np.ndarray, weights: np.ndarray
) -> float:
    return float(np.average(labels, weights=weights))


def _metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    predictions = probabilities >= threshold
    result: dict[str, float | int] = {
        "checkpoints": int(labels.size),
        "effective_trajectory_weight": float(weights.sum()),
        "weighted_prevalence": _weighted_prevalence(labels, weights),
        "weighted_brier": float(
            brier_score_loss(labels, probabilities, sample_weight=weights)
        ),
        "weighted_log_loss": float(log_loss(labels, clipped, sample_weight=weights)),
        "weighted_average_precision": float(
            average_precision_score(labels, probabilities, sample_weight=weights)
        ),
        "threshold": threshold,
        "weighted_f1": float(
            f1_score(labels, predictions, sample_weight=weights, zero_division=0)
        ),
        "weighted_precision": float(
            precision_score(
                labels, predictions, sample_weight=weights, zero_division=0
            )
        ),
        "weighted_recall": float(
            recall_score(labels, predictions, sample_weight=weights, zero_division=0)
        ),
    }
    result["weighted_roc_auc"] = (
        float(roc_auc_score(labels, probabilities, sample_weight=weights))
        if len(np.unique(labels)) > 1
        else 0.5
    )
    return result


def _hard_coded_rule(data: PreparedData, indices: np.ndarray) -> np.ndarray:
    feature_index = {name: index for index, name in enumerate(NUMERIC_FEATURES)}
    rows = data.numeric[indices]
    late_without_pass = (
        (rows[:, feature_index["turn_index"]] >= 8)
        & (rows[:, feature_index["pass_signal_count"]] == 0)
    )
    repeated_error_signal = rows[:, feature_index["error_signal_count"]] >= 2
    predicted_failure = late_without_pass | repeated_error_signal
    return np.where(predicted_failure, 0.20, 0.60)


def leave_one_task_out_splits(data: PreparedData) -> list[tuple[np.ndarray, np.ndarray]]:
    splits = []
    for held_out_task in sorted(set(data.task_ids.tolist())):
        test_indices = np.flatnonzero(data.task_ids == held_out_task)
        train_indices = np.flatnonzero(data.task_ids != held_out_task)
        if not len(train_indices) or not len(test_indices):
            raise ValueError("leave-one-task-out requires at least two task groups")
        splits.append((train_indices, test_indices))
    return splits


def _select_c(
    data: PreparedData,
    outer_train_indices: np.ndarray,
    *,
    candidates: tuple[float, ...],
    inner_splits: int,
) -> float:
    groups = data.task_ids[outer_train_indices]
    unique_groups = len(set(groups.tolist()))
    split_count = min(inner_splits, unique_groups)
    if split_count < 2:
        return min(candidates)
    group_kfold = GroupKFold(n_splits=split_count)
    scores = []
    for c_value in candidates:
        labels = []
        probabilities = []
        weights = []
        for inner_train_local, inner_validation_local in group_kfold.split(
            outer_train_indices,
            data.labels[outer_train_indices],
            groups,
        ):
            inner_train = outer_train_indices[inner_train_local]
            inner_validation = outer_train_indices[inner_validation_local]
            scaler, model = _fit_model(data, inner_train, c_value=c_value)
            labels.append(data.labels[inner_validation])
            probabilities.append(
                _predict_model(data, inner_validation, scaler, model)
            )
            weights.append(data.weights[inner_validation])
        all_labels = np.concatenate(labels)
        all_probabilities = np.concatenate(probabilities)
        all_weights = np.concatenate(weights)
        brier = float(
            brier_score_loss(
                all_labels, all_probabilities, sample_weight=all_weights
            )
        )
        scores.append((brier, c_value))
    return min(scores, key=lambda item: (item[0], item[1]))[1]


def _predict_final_artifact(
    artifact: dict[str, Any], examples: Sequence[ContinuationExample]
) -> np.ndarray:
    texts = [example.text for example in examples]
    numeric = np.asarray([example.numeric for example in examples], dtype=np.float64)
    text_matrix = artifact["vectorizer"].transform(texts)
    numeric_matrix = artifact["numeric_scaler"].transform(
        np.log1p(np.maximum(numeric, 0))
    )
    matrix = hstack([text_matrix, csr_matrix(numeric_matrix)], format="csr")
    return artifact["model"].predict_proba(matrix)[:, 1]


def train_checkpoint_continuation_risk(
    checkpoints_path: Path = DEFAULT_CHECKPOINTS,
    report_path: Path = DEFAULT_REPORT,
    model_path: Path = DEFAULT_MODEL,
    *,
    candidates: tuple[float, ...] = DEFAULT_CANDIDATES,
    inner_splits: int = 5,
) -> dict[str, Any]:
    if not candidates or any(value <= 0 for value in candidates):
        raise ValueError("regularization candidates must be positive")
    if inner_splits < 2:
        raise ValueError("inner_splits must be at least two")
    examples = load_continuation_examples(checkpoints_path)
    data = _prepare(examples)
    outer_splits = leave_one_task_out_splits(data)

    model_probabilities = np.empty(len(examples), dtype=np.float64)
    constant_probabilities = np.empty(len(examples), dtype=np.float64)
    turn_probabilities = np.empty(len(examples), dtype=np.float64)
    hard_rule_probabilities = np.empty(len(examples), dtype=np.float64)
    selected_candidates: Counter[float] = Counter()
    group_audit = []

    for train_indices, test_indices in outer_splits:
        train_tasks = set(data.task_ids[train_indices].tolist())
        test_tasks = set(data.task_ids[test_indices].tolist())
        if train_tasks & test_tasks or len(test_tasks) != 1:
            raise RuntimeError("outer task groups overlap")
        c_value = _select_c(
            data,
            train_indices,
            candidates=candidates,
            inner_splits=inner_splits,
        )
        selected_candidates[c_value] += 1
        scaler, model = _fit_model(data, train_indices, c_value=c_value)
        model_probabilities[test_indices] = _predict_model(
            data, test_indices, scaler, model
        )
        prevalence = _weighted_prevalence(
            data.labels[train_indices], data.weights[train_indices]
        )
        constant_probabilities[test_indices] = prevalence
        turn_scaler, turn_model = _fit_turn_only(data, train_indices)
        turn_probabilities[test_indices] = _predict_turn_only(
            data, test_indices, turn_scaler, turn_model
        )
        hard_rule_probabilities[test_indices] = _hard_coded_rule(data, test_indices)
        group_audit.append(
            {
                "test_task_id": next(iter(test_tasks)),
                "train_task_count": len(train_tasks),
                "test_checkpoint_count": len(test_indices),
                "train_test_task_overlap_count": 0,
                "selected_c": c_value,
            }
        )

    all_indices = np.arange(len(examples))
    final_c = _select_c(
        data,
        all_indices,
        candidates=candidates,
        inner_splits=inner_splits,
    )
    final_scaler, final_model = _fit_model(data, all_indices, c_value=final_c)
    if isinstance(final_model, float):
        raise RuntimeError("final continuation dataset contains only one label")
    artifact = {
        "schema_version": "checkpoint-continuation-risk-model.v0",
        "objective": (
            "Predict eventual verifier-confirmed completion if the current route "
            "continues from the observed pre-turn state."
        ),
        "action_scope": "continue_same_only",
        "record_split": "development",
        "numeric_features": list(NUMERIC_FEATURES),
        "observation_fields": sorted(OBSERVATION_FIELDS),
        "terminal_text": "pre-turn terminal_tail plus current_route_id token",
        "hash_features": HASH_FEATURES,
        "selected_c": final_c,
        "vectorizer": _vectorizer(),
        "numeric_scaler": final_scaler,
        "model": final_model,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    reloaded = joblib.load(model_path)
    before_reload = _predict_final_artifact(artifact, examples)
    after_reload = _predict_final_artifact(reloaded, examples)
    reload_match = bool(np.allclose(before_reload, after_reload, atol=0, rtol=0))
    if not reload_match:
        raise RuntimeError("serialized continuation model changed predictions")

    task_count = len(set(data.task_ids.tolist()))
    trajectory_count = len({example.trajectory_id for example in examples})
    report = {
        "schema_version": "checkpoint-continuation-risk-report.v0",
        "objective": artifact["objective"],
        "data": {
            "path": str(checkpoints_path),
            "sha256": _sha256_file(checkpoints_path),
            "record_split": "development",
            "checkpoints": len(examples),
            "tasks": task_count,
            "trajectories": trajectory_count,
            "positive_trajectories_weight": float(
                data.weights[data.labels == 1].sum()
            ),
            "negative_trajectories_weight": float(
                data.weights[data.labels == 0].sum()
            ),
        },
        "features": {
            "numeric": list(NUMERIC_FEATURES),
            "text": "pre-turn terminal_tail plus current_route_id token",
            "future_fields_used": False,
            "task_identity_used": False,
            "agent_messages_or_reasoning_used": False,
            "verifier_artifacts_used_as_input": False,
        },
        "evaluation_contract": {
            "outer_evaluation": "leave-one-task-out",
            "outer_folds": len(outer_splits),
            "inner_selection": f"task-grouped {min(inner_splits, task_count - 1)}-fold",
            "inner_objective": "minimum trajectory-weighted Brier score",
            "split_unit": "task_id",
            "all_outer_train_test_groups_disjoint": True,
            "trajectory_weighting": "each nonempty trajectory contributes total weight one",
            "candidate_c_values": list(candidates),
            "selected_c_counts": {
                str(key): value for key, value in sorted(selected_candidates.items())
            },
            "folds": group_audit,
        },
        "baselines": {
            "constant_training_prevalence": _metrics(
                data.labels, constant_probabilities, data.weights
            ),
            "turn_index_only": _metrics(
                data.labels, turn_probabilities, data.weights
            ),
            "hard_coded_late_or_error_rule": {
                "rule": (
                    "P(success)=0.20 when error_signal_count>=2 or turn_index>=8 "
                    "without a pass signal; otherwise 0.60."
                ),
                "metrics": _metrics(
                    data.labels, hard_rule_probabilities, data.weights
                ),
            },
        },
        "task_held_out_model": _metrics(
            data.labels, model_probabilities, data.weights
        ),
        "final_training": {
            "fit_tasks": task_count,
            "fit_checkpoints": len(examples),
            "selected_c": final_c,
            "heldout_or_wave3_rows_used": 0,
        },
        "artifact": {
            "path": str(model_path),
            "sha256": _sha256_file(model_path),
            "reload_predictions_match": reload_match,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "seed": SEED,
        },
        "action_boundary": {
            "trained_action": "continue_same",
            "switch_model_outcomes_trained": False,
            "restart_clean_outcomes_trained": False,
            "stop_outcomes_trained": False,
        },
        "interpretation_guard": (
            "This estimates fixed-route continuation risk from logged development "
            "trajectories. It does not identify the outcome of switching, restarting, "
            "stopping, or preserving a live workspace."
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the development-only checkpoint continuation-risk model"
    )
    parser.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--candidate-c", type=float, action="append")
    parser.add_argument("--inner-splits", type=int, default=5)
    args = parser.parse_args()
    report = train_checkpoint_continuation_risk(
        checkpoints_path=args.checkpoints,
        report_path=args.report,
        model_path=args.model,
        candidates=tuple(args.candidate_c or DEFAULT_CANDIDATES),
        inner_splits=args.inner_splits,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
