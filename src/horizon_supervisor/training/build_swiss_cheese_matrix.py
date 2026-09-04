from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from horizon_supervisor.benchmark.matched_outcomes import LEARNING_VALID_STATUSES


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _development_copy(
    row: dict[str, Any],
    *,
    replication_index: int,
    evaluation_role: str,
    source_path: Path,
) -> dict[str, Any]:
    copied = json.loads(json.dumps(row))
    original_split = copied["task"].get("record_split")
    copied["task"]["record_split"] = "development"
    provenance = copied.setdefault("provenance", {})
    provenance["original_record_split"] = original_split
    provenance["record_split_override"] = "development"
    provenance["evaluation_role"] = evaluation_role
    provenance["replication_index"] = replication_index
    provenance["source_outcome_path"] = str(source_path)
    provenance.setdefault("superseded_outcome_ids", [])
    return copied


def build_swiss_cheese_matrix(
    manifest_path: Path,
    baseline_path: Path,
    replication_paths: list[Path],
    output_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = tuple(manifest["design"]["tasks"])
    routes = tuple(manifest["design"]["routes"])
    existing_routes = tuple(manifest["design"]["existing_routes"])
    replication_indices = tuple(manifest["design"]["replication_indices"])
    expected_pairs = {
        (task, route, replication)
        for task in tasks
        for route in routes
        for replication in replication_indices
    }

    if _sha256(baseline_path) != manifest["frozen_inputs"]["baseline_outcomes_sha256"]:
        raise RuntimeError("baseline outcomes changed after the experiment was frozen")

    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in _rows(baseline_path):
        task = row["task"]["source_task_name"]
        route = row["model"]["route_id"]
        if task in tasks and route in existing_routes:
            copied = _development_copy(
                row,
                replication_index=1,
                evaluation_role="posthoc_outcome_selected_discovery_replication",
                source_path=baseline_path,
            )
            grouped[(task, route, 1)].append(copied)

    input_statuses: Counter[str] = Counter()
    for path in replication_paths:
        for row in _rows(path):
            task = row["task"]["source_task_name"]
            route = row["model"]["route_id"]
            replication = row.get("provenance", {}).get("replication_index")
            if task not in tasks or route not in routes:
                raise ValueError(f"unexpected replication pair: {task}|{route}")
            if replication not in replication_indices:
                raise ValueError(f"missing or unexpected replication index in {path}")
            if row["task"].get("record_split") != "development":
                raise ValueError("new replication rows must be development-only")
            if row.get("provenance", {}).get("evaluation_role") != (
                "posthoc_clean_start_replication"
            ):
                raise ValueError("new rows lack the post-hoc replication role")
            row = json.loads(json.dumps(row))
            row["provenance"]["source_outcome_path"] = str(path)
            grouped[(task, route, int(replication))].append(row)
            input_statuses[row["outcome"]["status"]] += 1

    retained: list[dict[str, Any]] = []
    superseded: Counter[str] = Counter()
    for key, candidates in grouped.items():
        valid = [
            row
            for row in candidates
            if row["outcome"]["status"] in LEARNING_VALID_STATUSES
        ]
        if len(valid) > 1:
            raise ValueError(f"multiple learning-valid outcomes for {key}")
        if not valid:
            continue
        selected = valid[0]
        discarded = [row for row in candidates if row is not selected]
        for row in discarded:
            superseded[row["outcome"]["status"]] += 1
        selected["provenance"]["superseded_outcome_ids"] = [
            row["outcome_id"] for row in discarded
        ]
        retained.append(selected)

    observed_pairs = {
        (
            row["task"]["source_task_name"],
            row["model"]["route_id"],
            int(row["provenance"]["replication_index"]),
        )
        for row in retained
    }
    if observed_pairs != expected_pairs:
        missing = sorted(expected_pairs - observed_pairs)
        extra = sorted(observed_pairs - expected_pairs)
        raise ValueError(
            f"replication matrix is not rectangular; missing={missing}, extra={extra}"
        )

    for task in tasks:
        digests = {
            row["initial_state"]["digest"]
            for row in retained
            if row["task"]["source_task_name"] == task
        }
        if len(digests) != 1:
            raise ValueError(f"task {task} does not share one clean initial state")
    for route in routes:
        endpoints = {
            row["model"]["endpoint"]
            for row in retained
            if row["model"]["route_id"] == route
        }
        if endpoints != {manifest["design"]["route_endpoints"][route]}:
            raise ValueError(f"route endpoint drift for {route}: {sorted(endpoints)}")

    retained.sort(
        key=lambda row: (
            row["task"]["source_task_name"],
            row["model"]["route_id"],
            row["provenance"]["replication_index"],
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in retained:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary = {
        "schema_version": "swiss-cheese-replication-matrix-summary.v0",
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "baseline_path": str(baseline_path),
        "baseline_sha256": _sha256(baseline_path),
        "replication_paths": [str(path) for path in replication_paths],
        "record_count": len(retained),
        "expected_record_count": len(expected_pairs),
        "task_count": len(tasks),
        "route_count": len(routes),
        "replications_per_pair": len(replication_indices),
        "record_split_counts": dict(
            sorted(Counter(row["task"]["record_split"] for row in retained).items())
        ),
        "status_counts": dict(
            sorted(Counter(row["outcome"]["status"] for row in retained).items())
        ),
        "verified_completion_count": sum(
            bool(row["outcome"]["completed"]) for row in retained
        ),
        "new_input_status_counts": dict(sorted(input_statuses.items())),
        "superseded_status_counts": dict(sorted(superseded.items())),
        "all_pairs_present_once_and_learning_valid": True,
        "all_initial_states_matched_by_task": True,
        "all_routes_endpoint_locked": True,
        "confirmatory_replication_indices": manifest["design"][
            "confirmatory_replication_indices"
        ],
        "discovery_replication_excluded_from_confirmatory_claims": True,
        "output_path": str(output_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Swiss-cheese replication matrix")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--replication", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args()
    result = build_swiss_cheese_matrix(
        args.manifest,
        args.baseline,
        args.replication,
        args.output,
        args.summary,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
