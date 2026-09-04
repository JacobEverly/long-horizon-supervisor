from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download

from horizon_supervisor.chooser.build_sample import (
    ADAPTERS,
    _adapt_apps,
)
from horizon_supervisor.chooser.schema import (
    ChooserDatasetRecord,
    ChooserTargets,
    DatasetSplit,
    ModelOutcomeAggregate,
    OutcomeProvenance,
    chooser_record_json_schema,
)

DEFAULT_REGISTRY = Path("data/chooser/source-registry-v0.json")
DEFAULT_OUTPUT = Path("data/chooser/corpus-v0.jsonl")
DEFAULT_SUMMARY = Path("data/chooser/corpus-v0-summary.json")
DEFAULT_SCHEMA = Path("data/chooser/chooser-record-schema-v0.json")
SPLIT_SEED = "chooser-corpus-v0|2026-08-25"

PARQUET_COLUMNS = {
    "code_contests": [
        "description",
        "source",
        "cf_rating",
        "is_description_translated",
        "cf_contest_id",
        "cf_index",
        "cf_tags",
        "time_limit",
    ],
    "swe_bench_extra": [
        "instance_id",
        "problem_statement",
        "repo",
        "created_at",
        "license",
    ],
    "trajectory_outcomes": ["instance_id", "model_name", "target", "exit_status"],
}


def _verify_source_revisions(
    sources: list[dict[str, Any]], api: HfApi
) -> dict[str, dict[str, str]]:
    verified: dict[str, dict[str, str]] = {}
    for source in sources:
        if source["kind"] != "huggingface" or "collection" not in source:
            continue
        source_sha = api.dataset_info(
            source["dataset_id"], revision=source["revision"]
        ).sha
        if source_sha != source["revision"]:
            raise RuntimeError(
                f"source revision mismatch for {source['dataset_id']}: "
                f"expected {source['revision']}, found {source_sha}"
            )
        values = {"source_revision": source_sha}
        if source["collection"]["reader"] == "parquet":
            parquet_sha = api.dataset_info(
                source["dataset_id"], revision="refs/convert/parquet"
            ).sha
            if parquet_sha != source["parquet_revision"]:
                raise RuntimeError(
                    f"converted parquet moved for {source['dataset_id']}: "
                    f"expected {source['parquet_revision']}, found {parquet_sha}; "
                    "audit before updating the registry"
                )
            values["parquet_revision"] = parquet_sha
        verified[source["source_id"]] = values
    return verified


def _iter_parquet_rows(
    source: dict[str, Any], api: HfApi, fs: HfFileSystem
) -> Iterator[dict[str, Any]]:
    collection = source["collection"]
    adapter = collection["adapter"]
    info = api.dataset_info(
        source["dataset_id"], revision=source["parquet_revision"]
    )
    filenames = sorted(
        sibling.rfilename
        for sibling in info.siblings
        if sibling.rfilename.startswith(collection["path_prefix"])
        and sibling.rfilename.endswith(".parquet")
    )
    if not filenames:
        raise RuntimeError(f"no parquet files found for {source['source_id']}")
    for filename in filenames:
        path = (
            f"datasets/{source['dataset_id']}@{source['parquet_revision']}/{filename}"
        )
        with fs.open(path, "rb") as handle:
            parquet = pq.ParquetFile(handle)
            for row_group in range(parquet.num_row_groups):
                table = parquet.read_row_group(
                    row_group, columns=PARQUET_COLUMNS[adapter]
                )
                yield from table.to_pylist()


def _iter_apps_rows(source: dict[str, Any]) -> Iterator[dict[str, Any]]:
    filename = source["collection"]["filename"]
    path = hf_hub_download(
        repo_id=source["dataset_id"],
        repo_type="dataset",
        filename=filename,
        revision=source["revision"],
    )
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            row["problem_id"] = row.pop("id")
            yield row


def _collect_task_records(
    sources: list[dict[str, Any]], api: HfApi, fs: HfFileSystem
) -> tuple[list[ChooserDatasetRecord], dict[str, Any]]:
    records: list[ChooserDatasetRecord] = []
    raw_counts: Counter[str] = Counter()
    eligible_counts: Counter[str] = Counter()
    for source in sources:
        collection = source.get("collection")
        if not collection or collection["adapter"] == "trajectory_outcomes":
            continue
        adapter_name = collection["adapter"]
        adapter = ADAPTERS[adapter_name]
        rows = (
            _iter_apps_rows(source)
            if collection["reader"] == "jsonl"
            else _iter_parquet_rows(source, api, fs)
        )
        for row_index, row in enumerate(rows):
            raw_counts[source["source_id"]] += 1
            item = {"row_idx": row_index, "row": row}
            record = _adapt_apps(source, item) if adapter_name == "apps" else adapter(source, item)
            if record is not None:
                eligible_counts[source["source_id"]] += 1
                records.append(record)
    return records, {
        "raw_source_rows": dict(sorted(raw_counts.items())),
        "eligible_before_deduplication": dict(sorted(eligible_counts.items())),
    }


def _collect_outcomes(
    source: dict[str, Any], api: HfApi, fs: HfFileSystem
) -> tuple[dict[str, list[ModelOutcomeAggregate]], dict[str, Any]]:
    counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    model_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    exit_statuses: Counter[str] = Counter()
    total = 0
    for row in _iter_parquet_rows(source, api, fs):
        instance_id = row["instance_id"]
        model_name = row["model_name"]
        success = int(bool(row["target"]))
        counts[(instance_id, model_name)][0] += 1
        counts[(instance_id, model_name)][1] += success
        model_totals[model_name][0] += 1
        model_totals[model_name][1] += success
        exit_statuses[row["exit_status"] or "unknown"] += 1
        total += 1

    by_instance: dict[str, list[ModelOutcomeAggregate]] = defaultdict(list)
    for (instance_id, model_name), (attempts, successes) in sorted(counts.items()):
        by_instance[instance_id].append(
            ModelOutcomeAggregate(
                model_id=model_name,
                deployment_id=f"{model_name}@dataset-declared-unversioned",
                agent_id="nebius-swe-agent-framework@dataset-declared-unversioned",
                verifier_id="linked-pull-request-tests@dataset-declared-unversioned",
                attempts=attempts,
                successes=successes,
                source_id=source["source_id"],
                provenance=OutcomeProvenance.DATASET_DECLARED_UNVERSIONED,
                notes=[
                    "The public dataset identifies the model family but not an immutable "
                    "weights or agent-configuration revision."
                ],
            )
        )
    model_summary = {
        model: {
            "attempts": values[0],
            "successes": values[1],
            "observed_success_rate": values[1] / values[0],
        }
        for model, values in sorted(model_totals.items())
    }
    return dict(by_instance), {
        "raw_trajectory_rows": total,
        "unique_trajectory_tasks": len(by_instance),
        "task_model_aggregates": len(counts),
        "model_totals": model_summary,
        "exit_status_counts": dict(sorted(exit_statuses.items())),
        "excluded_trajectory_fields": ["trajectory", "generated_patch", "eval_logs"],
    }


def _deduplicate(
    records: list[ChooserDatasetRecord],
) -> tuple[list[ChooserDatasetRecord], dict[str, Any]]:
    source_priority = {
        "deepmind-code-contests": 0,
        "apps": 1,
        "nebius-swe-bench-extra": 2,
    }
    ordered = sorted(
        records,
        key=lambda record: (
            source_priority.get(record.source.source_id, 99),
            record.example_id,
        ),
    )
    seen_hashes: set[str] = set()
    seen_codeforces: set[str] = set()
    retained: list[ChooserDatasetRecord] = []
    reasons: Counter[str] = Counter()
    dropped_sources: Counter[str] = Counter()
    for record in ordered:
        normalized_hash = record.leakage.normalized_task_sha256
        group = record.leakage.leakage_group
        reason = None
        if normalized_hash in seen_hashes:
            reason = "normalized_prompt_duplicate"
        elif group.startswith("codeforces:") and group in seen_codeforces:
            reason = "canonical_codeforces_duplicate"
        if reason:
            reasons[reason] += 1
            dropped_sources[record.source.source_id] += 1
            continue
        seen_hashes.add(normalized_hash)
        if group.startswith("codeforces:"):
            seen_codeforces.add(group)
        retained.append(record)
    return retained, {
        "dropped_total": len(records) - len(retained),
        "dropped_by_reason": dict(sorted(reasons.items())),
        "dropped_by_source": dict(sorted(dropped_sources.items())),
    }


def _split_for_group(leakage_group: str) -> DatasetSplit:
    digest = hashlib.sha256(f"{SPLIT_SEED}|{leakage_group}".encode()).digest()
    bucket = int.from_bytes(digest[:4], "big") % 100
    if bucket < 80:
        return DatasetSplit.TRAIN
    if bucket < 90:
        return DatasetSplit.VALIDATION
    return DatasetSplit.TEST


def build_corpus(
    registry_path: Path,
) -> tuple[list[ChooserDatasetRecord], dict[str, Any]]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    sources = registry["sources"]
    api = HfApi()
    fs = HfFileSystem()
    verified_revisions = _verify_source_revisions(sources, api)
    records, collection_summary = _collect_task_records(sources, api, fs)
    trajectory_source = next(
        source
        for source in sources
        if source.get("collection", {}).get("adapter") == "trajectory_outcomes"
    )
    outcomes, outcome_summary = _collect_outcomes(trajectory_source, api, fs)
    records, deduplication_summary = _deduplicate(records)

    joined_attempts = 0
    records_with_outcomes = 0
    finalized: list[ChooserDatasetRecord] = []
    for record in records:
        model_outcomes = outcomes.get(record.source.record_id, [])
        if model_outcomes:
            records_with_outcomes += 1
            joined_attempts += sum(outcome.attempts for outcome in model_outcomes)
        targets = ChooserTargets(
            difficulty=record.targets.difficulty,
            ambiguity=record.targets.ambiguity,
            domain=record.targets.domain,
            model_outcomes=model_outcomes,
        )
        finalized.append(
            record.model_copy(
                update={
                    "record_split": _split_for_group(record.leakage.leakage_group),
                    "targets": targets,
                }
            )
        )
    finalized.sort(key=lambda record: record.example_id)

    source_counts = Counter(record.source.source_id for record in finalized)
    split_counts = Counter(record.record_split.value for record in finalized)
    difficulty_counts = Counter(
        record.targets.difficulty.value.value
        if record.targets.difficulty is not None
        else "unlabeled"
        for record in finalized
    )
    domain_counts = Counter(record.targets.domain.value.value for record in finalized)
    split_source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    joined_model_sets: Counter[str] = Counter()
    for record in finalized:
        split_source_counts[record.record_split.value][record.source.source_id] += 1
        model_ids = sorted(outcome.model_id for outcome in record.targets.model_outcomes)
        if model_ids:
            joined_model_sets[" + ".join(model_ids)] += 1

    summary = {
        "schema_version": "chooser-corpus-summary.v0",
        "split_seed": SPLIT_SEED,
        "source_registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
        "verified_revisions": verified_revisions,
        **collection_summary,
        "deduplication": deduplication_summary,
        "record_count": len(finalized),
        "source_counts": dict(sorted(source_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "split_source_counts": {
            split: dict(sorted(counts.items()))
            for split, counts in sorted(split_source_counts.items())
        },
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "domain_counts": dict(sorted(domain_counts.items())),
        "outcomes": {
            **outcome_summary,
            "records_with_joined_outcomes": records_with_outcomes,
            "joined_trajectory_attempts": joined_attempts,
            "unjoined_trajectory_attempts": (
                outcome_summary["raw_trajectory_rows"] - joined_attempts
            ),
            "joined_task_model_sets": dict(sorted(joined_model_sets.items())),
        },
        "gold_fields_used_as_model_input": False,
        "purpose": (
            "Public metadata corpus for task classification and preliminary outcome "
            "modeling; not the sealed final evaluation."
        ),
    }
    return finalized, summary


def write_corpus(
    records: list[ChooserDatasetRecord],
    summary: dict[str, Any],
    *,
    output_path: Path,
    summary_path: Path,
    schema_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = record.model_dump(mode="json")
            handle.write(json.dumps(payload, ensure_ascii=True, separators=(",", ":")))
            handle.write("\n")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    schema_path.write_text(
        json.dumps(chooser_record_json_schema(), indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the pinned chooser v0 corpus")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    args = parser.parse_args()
    records, summary = build_corpus(args.registry)
    write_corpus(
        records,
        summary,
        output_path=args.output,
        summary_path=args.summary,
        schema_path=args.schema,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
