from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from horizon_supervisor.benchmark.matched_outcomes import LEARNING_VALID_STATUSES


def _iter_rows(paths: Iterable[Path]) -> Iterable[tuple[Path, dict[str, Any]]]:
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("schema_version") != "matched-model-outcome.v1":
                    raise ValueError(
                        f"{path} contains unsupported schema "
                        f"{row.get('schema_version')!r}"
                    )
                yield path, row


def merge_matched_outcomes(
    input_paths: list[Path],
    output_path: Path,
    summary_path: Path,
    *,
    expected_routes: tuple[str, ...],
    accepted_statuses: frozenset[str] = LEARNING_VALID_STATUSES,
    exclude_tasks: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Merge recovery runs without treating execution errors as model failures.

    One learning-valid outcome wins over any number of infrastructure/provider
    errors. Verified verifier outcomes and attributable agent protocol failures
    are learning-valid. Multiple learning-valid trials for the same task-model
    pair are rejected rather than cherry-picked; repeated attempts need a
    separately declared sampling design before they can be aggregated.
    """

    if not input_paths:
        raise ValueError("at least one matched-outcome input is required")
    grouped: dict[tuple[str, str], list[tuple[Path, dict[str, Any]]]] = defaultdict(
        list
    )
    input_statuses: Counter[str] = Counter()
    excluded_record_count = 0
    for source_path, row in _iter_rows(input_paths):
        if row["task"]["source_task_name"] in exclude_tasks:
            excluded_record_count += 1
            continue
        key = (row["task"]["task_id"], row["model"]["route_id"])
        grouped[key].append((source_path, row))
        input_statuses[row["outcome"]["status"]] += 1

    retained: list[dict[str, Any]] = []
    superseded_statuses: Counter[str] = Counter()
    for (task_id, route_id), candidates in grouped.items():
        accepted = [
            pair
            for pair in candidates
            if pair[1]["outcome"]["status"] in accepted_statuses
        ]
        if len(accepted) > 1:
            label = (
                "learning-valid"
                if accepted_statuses == LEARNING_VALID_STATUSES
                else "accepted"
            )
            raise ValueError(
                f"multiple {label} trials for task {task_id} and route {route_id}"
            )
        if accepted:
            selected_path, selected = accepted[0]
        elif len(candidates) == 1:
            selected_path, selected = candidates[0]
        else:
            statuses = sorted(pair[1]["outcome"]["status"] for pair in candidates)
            raise ValueError(
                f"task {task_id} and route {route_id} has only repeated invalid "
                f"trials: {statuses}"
            )
        discarded = [row for _, row in candidates if row is not selected]
        for row in discarded:
            superseded_statuses[row["outcome"]["status"]] += 1
        selected = json.loads(json.dumps(selected))
        selected["provenance"]["source_outcome_path"] = str(selected_path)
        selected["provenance"]["superseded_outcome_ids"] = [
            row["outcome_id"] for row in discarded
        ]
        retained.append(selected)

    retained.sort(
        key=lambda row: (
            row["task"]["source_task_name"],
            row["model"]["route_id"],
        )
    )
    task_names = sorted({row["task"]["source_task_name"] for row in retained})
    expected_pairs = {
        (task_name, route_id)
        for task_name in task_names
        for route_id in expected_routes
    }
    observed_pairs = {
        (row["task"]["source_task_name"], row["model"]["route_id"])
        for row in retained
    }
    if observed_pairs != expected_pairs:
        missing = sorted(expected_pairs - observed_pairs)
        extra = sorted(observed_pairs - expected_pairs)
        raise ValueError(f"merged panel is not rectangular; missing={missing}, extra={extra}")
    invalid = [
        row
        for row in retained
        if row["outcome"]["status"] not in accepted_statuses
    ]
    if invalid:
        pairs = [
            f"{row['task']['source_task_name']}|{row['model']['route_id']}"
            for row in invalid
        ]
        raise ValueError(f"merged panel still contains unaccepted trials: {pairs}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in retained:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "schema_version": "matched-model-outcome-merge-summary.v0",
        "input_paths": [str(path) for path in input_paths],
        "input_record_count": sum(input_statuses.values()),
        "input_status_counts": dict(sorted(input_statuses.items())),
        "excluded_task_names": sorted(exclude_tasks),
        "excluded_record_count": excluded_record_count,
        "record_count": len(retained),
        "task_count": len(task_names),
        "route_count": len(expected_routes),
        "record_split_counts": dict(
            sorted(
                Counter(row["task"].get("record_split", "missing") for row in retained).items()
            )
        ),
        "verified_completion_count": sum(
            row["outcome"]["completed"] for row in retained
        ),
        "learning_status_counts": dict(
            sorted(Counter(row["outcome"]["status"] for row in retained).items())
        ),
        "superseded_status_counts": dict(sorted(superseded_statuses.items())),
        "accepted_statuses": sorted(accepted_statuses),
        "all_pairs_present_once_and_accepted": True,
        "all_pairs_present_once_and_learning_valid": all(
            row["outcome"]["status"] in LEARNING_VALID_STATUSES
            for row in retained
        ),
        "all_pairs_present_once_and_verified": all(
            row["outcome"]["status"] == "verified" for row in retained
        ),
        "learning_contract": {
            "scoring_unit": "task-model-pair",
            "portable_feature_path": "model.candidate_features",
            "identity_fields_are_cold_start_features": False,
        },
        "output_path": str(output_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge matched model outcomes")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--expected-route", action="append", required=True)
    parser.add_argument(
        "--exclude-task",
        action="append",
        default=[],
        help="source task name to omit before validating the rectangular panel",
    )
    parser.add_argument(
        "--allow-provider-errors",
        action="store_true",
        help=(
            "accept provider_error as an observed deployment outcome for a "
            "completion-first screen; capability training still excludes it"
        ),
    )
    args = parser.parse_args()
    summary = merge_matched_outcomes(
        args.inputs,
        args.output,
        args.summary,
        expected_routes=tuple(args.expected_route),
        exclude_tasks=frozenset(args.exclude_task),
        accepted_statuses=(
            LEARNING_VALID_STATUSES | {"provider_error"}
            if args.allow_provider_errors
            else LEARNING_VALID_STATUSES
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
