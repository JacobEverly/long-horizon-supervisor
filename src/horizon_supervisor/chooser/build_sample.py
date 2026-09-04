from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from horizon_supervisor.chooser.schema import (
    ChooserDatasetRecord,
    ChooserInput,
    ChooserTargets,
    DatasetSplit,
    Difficulty,
    DifficultyTarget,
    DomainTarget,
    LabelEvidence,
    LabelMethod,
    LeakageAudit,
    SourceReference,
    TaskDomain,
    chooser_record_json_schema,
)

DEFAULT_REGISTRY = Path("data/chooser/source-registry-v0.json")
DEFAULT_OUTPUT = Path("data/chooser/sample-v0.jsonl")
DEFAULT_SUMMARY = Path("data/chooser/sample-v0-summary.json")
DEFAULT_SCHEMA = Path("data/chooser/chooser-record-schema-v0.json")
SAMPLE_SEED = "chooser-sample-v0|2026-08-25"
CODEFORCES_URL = re.compile(
    r"codeforces\.com/(?:problemset/problem/(\d+)/|contest/(\d+)/problem/)([A-Za-z0-9]+)"
)


def _fetch_json(url: str, timeout: int = 60) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "long-horizon-supervisor/0.1 chooser-data"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _huggingface_dataset_info(dataset_id: str) -> dict[str, Any]:
    encoded = urllib.parse.quote(dataset_id, safe="/")
    return _fetch_json(f"https://huggingface.co/api/datasets/{encoded}")


def _dataset_rows(source: dict[str, Any], offset: int, length: int) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode(
        {
            "dataset": source["dataset_id"],
            "config": source["config"],
            "split": source["source_split"],
            "offset": offset,
            "length": length,
        }
    )
    payload = _fetch_json(f"https://datasets-server.huggingface.co/rows?{query}")
    return payload["rows"]


def normalize_apps_difficulty(native: str) -> Difficulty:
    mapping = {
        "introductory": Difficulty.EASY,
        "interview": Difficulty.MEDIUM,
        "competition": Difficulty.HARD,
    }
    try:
        return mapping[native.lower()]
    except KeyError as error:
        raise ValueError(f"unknown APPS difficulty: {native}") from error


def normalize_codeforces_rating(rating: int) -> Difficulty:
    if rating <= 0:
        raise ValueError("Codeforces rating must be positive")
    if rating <= 1_200:
        return Difficulty.EASY
    if rating <= 1_900:
        return Difficulty.MEDIUM
    return Difficulty.HARD


def _task_hash(task_text: str) -> str:
    return hashlib.sha256(task_text.encode("utf-8")).hexdigest()


def _normalized_task_hash(task_text: str) -> str:
    normalized = unicodedata.normalize("NFKC", task_text).lower()
    normalized = re.sub(r"\W+", " ", normalized).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _rank(example_id: str) -> str:
    return hashlib.sha256(f"{SAMPLE_SEED}|{example_id}".encode()).hexdigest()


def canonical_apps_leakage_group(url: str | None, record_id: str) -> str:
    if url:
        match = CODEFORCES_URL.search(url)
        if match:
            contest_id = match.group(1) or match.group(2)
            return f"codeforces:{contest_id}-{match.group(3).upper()}"
        return f"apps-url:{url}"
    return f"apps-id:{record_id}"


def _source_ref(
    source: dict[str, Any],
    *,
    record_id: str,
    row_index: int,
    license_name: str | None = None,
) -> SourceReference:
    return SourceReference(
        source_id=source["source_id"],
        dataset_id=source["dataset_id"],
        revision=source["revision"],
        config=source["config"],
        source_split=source["source_split"],
        record_id=record_id,
        row_index=row_index,
        card_url=source["card_url"],
        license=license_name or source["license"],
    )


def _record(
    *,
    source: dict[str, Any],
    record_id: str,
    row_index: int,
    task_text: str,
    task_family: str,
    domain: TaskDomain,
    domain_note: str,
    leakage_group: str,
    public_metadata: dict[str, str | int | float | bool | None],
    repository: str | None = None,
    programming_language: str | None = None,
    difficulty: Difficulty | None = None,
    difficulty_native: str | int | float | None = None,
    difficulty_note: str | None = None,
    license_name: str | None = None,
) -> ChooserDatasetRecord:
    text = task_text.strip()
    difficulty_target = None
    if difficulty is not None:
        difficulty_target = DifficultyTarget(
            value=difficulty,
            evidence=LabelEvidence(
                source_id=source["source_id"],
                method=LabelMethod.NORMALIZED_RULE,
                method_version="difficulty-normalization.v0",
                native_value=difficulty_native,
                confidence=0.9,
                note=difficulty_note,
            ),
        )
    domain_target = DomainTarget(
        value=domain,
        evidence=LabelEvidence(
            source_id=source["source_id"],
            method=LabelMethod.NORMALIZED_RULE,
            method_version="source-family-domain.v0",
            confidence=0.95,
            note=domain_note,
        ),
    )
    example_id = f"{source['source_id']}:{record_id}"
    return ChooserDatasetRecord(
        example_id=example_id,
        record_split=DatasetSplit.TRAIN,
        source=_source_ref(
            source,
            record_id=record_id,
            row_index=row_index,
            license_name=license_name,
        ),
        input=ChooserInput(
            task_text=text,
            repository=repository,
            programming_language=programming_language,
            task_family=task_family,
            public_metadata=public_metadata,
        ),
        targets=ChooserTargets(
            difficulty=difficulty_target,
            domain=domain_target,
        ),
        leakage=LeakageAudit(
            task_text_sha256=_task_hash(text),
            normalized_task_sha256=_normalized_task_hash(text),
            leakage_group=leakage_group,
            excluded_source_fields=source["excluded_source_fields"],
            notes=[
                "Only information available before agent execution is included.",
                "Gold solutions, tests, patches, and verifier outputs are excluded.",
            ],
        ),
    )


def _adapt_apps(source: dict[str, Any], item: dict[str, Any]) -> ChooserDatasetRecord:
    row = item["row"]
    native = row["difficulty"]
    difficulty = normalize_apps_difficulty(native)
    record_id = str(row["problem_id"])
    return _record(
        source=source,
        record_id=record_id,
        row_index=item["row_idx"],
        task_text=row["question"],
        task_family="standalone_programming_problem",
        domain=TaskDomain.ALGORITHMIC_PROBLEM_SOLVING,
        domain_note="APPS consists of standalone programming problems.",
        leakage_group=canonical_apps_leakage_group(row["url"], record_id),
        public_metadata={"source_url": row["url"], "native_difficulty": native},
        programming_language="python",
        difficulty=difficulty,
        difficulty_native=native,
        difficulty_note="introductory→easy, interview→medium, competition→hard",
    )


def _adapt_code_contests(
    source: dict[str, Any], item: dict[str, Any]
) -> ChooserDatasetRecord | None:
    row = item["row"]
    if row["source"] != 2 or row["cf_rating"] <= 0 or row["is_description_translated"]:
        return None
    rating = int(row["cf_rating"])
    difficulty = normalize_codeforces_rating(rating)
    record_id = f"{row['cf_contest_id']}-{row['cf_index']}"
    tags = ",".join(row["cf_tags"])
    return _record(
        source=source,
        record_id=record_id,
        row_index=item["row_idx"],
        task_text=row["description"],
        task_family="competitive_programming_problem",
        domain=TaskDomain.ALGORITHMIC_PROBLEM_SOLVING,
        domain_note="The selected CodeContests rows are rated Codeforces problems.",
        leakage_group=f"codeforces:{record_id}",
        public_metadata={
            "cf_rating": rating,
            "cf_tags": tags,
            "time_limit_seconds": (row.get("time_limit") or {}).get("seconds"),
        },
        difficulty=difficulty,
        difficulty_native=rating,
        difficulty_note="rating≤1200→easy, 1300–1900→medium, ≥2000→hard",
    )


def _adapt_swe_bench_extra(
    source: dict[str, Any], item: dict[str, Any]
) -> ChooserDatasetRecord | None:
    row = item["row"]
    license_name = (row.get("license") or "").lower()
    allowed = set(
        source.get("collection", source["sample"])["allowed_repository_licenses"]
    )
    if license_name not in allowed:
        return None
    record_id = row["instance_id"]
    return _record(
        source=source,
        record_id=record_id,
        row_index=item["row_idx"],
        task_text=row["problem_statement"],
        task_family="repository_issue_resolution",
        domain=TaskDomain.SOFTWARE_MAINTENANCE,
        domain_note="SWE-bench-extra instances are executable GitHub issue-resolution tasks.",
        leakage_group=f"repository:{row['repo']}",
        public_metadata={
            "created_at": row.get("created_at"),
            "repository_license": license_name,
        },
        repository=row["repo"],
        license_name=f"cc-by-4.0 dataset; {license_name} repository",
    )


ADAPTERS = {
    "apps": _adapt_apps,
    "code_contests": _adapt_code_contests,
    "swe_bench_extra": _adapt_swe_bench_extra,
}


def _select_sample(
    records: list[ChooserDatasetRecord], sample_config: dict[str, Any]
) -> list[ChooserDatasetRecord]:
    unique = {record.example_id: record for record in records}
    candidates = sorted(unique.values(), key=lambda record: _rank(record.example_id))
    per_difficulty = sample_config.get("per_difficulty")
    if per_difficulty is not None:
        grouped: dict[str, list[ChooserDatasetRecord]] = defaultdict(list)
        for record in candidates:
            if record.targets.difficulty is not None:
                grouped[record.targets.difficulty.value.value].append(record)
        expected = {difficulty.value for difficulty in Difficulty}
        if set(grouped) != expected:
            raise RuntimeError(
                f"sample lacks difficulty strata: expected {expected}, found {set(grouped)}"
            )
        selected = []
        for difficulty in Difficulty:
            group = grouped[difficulty.value]
            if len(group) < per_difficulty:
                raise RuntimeError(
                    f"not enough {difficulty.value} rows: {len(group)} < {per_difficulty}"
                )
            selected.extend(group[:per_difficulty])
        return selected
    count = int(sample_config["count"])
    if len(candidates) < count:
        raise RuntimeError(f"not enough eligible rows: {len(candidates)} < {count}")
    return candidates[:count]


def build_sample(registry_path: Path) -> tuple[list[ChooserDatasetRecord], dict[str, Any]]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    records: list[ChooserDatasetRecord] = []
    revisions: dict[str, str] = {}
    for source in registry["sources"]:
        sample_config = source.get("sample")
        if source["kind"] != "huggingface" or sample_config is None:
            continue
        info = _huggingface_dataset_info(source["dataset_id"])
        actual_revision = info["sha"]
        if actual_revision != source["revision"]:
            raise RuntimeError(
                f"{source['dataset_id']} moved: expected {source['revision']}, "
                f"found {actual_revision}; audit before updating the registry"
            )
        revisions[source["source_id"]] = actual_revision
        adapter = ADAPTERS[sample_config["adapter"]]
        candidates = []
        for offset in sample_config["offsets"]:
            for item in _dataset_rows(source, offset, sample_config["length"]):
                adapted = adapter(source, item)
                if adapted is not None:
                    candidates.append(adapted)
        records.extend(_select_sample(candidates, sample_config))

    records.sort(key=lambda record: record.example_id)
    difficulty_counts = Counter(
        record.targets.difficulty.value.value
        if record.targets.difficulty is not None
        else "unlabeled"
        for record in records
    )
    domain_counts = Counter(record.targets.domain.value.value for record in records)
    source_counts = Counter(record.source.source_id for record in records)
    summary = {
        "schema_version": "chooser-sample-summary.v0",
        "sample_seed": SAMPLE_SEED,
        "source_registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
        "source_revisions": revisions,
        "record_count": len(records),
        "source_counts": dict(sorted(source_counts.items())),
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "domain_counts": dict(sorted(domain_counts.items())),
        "gold_fields_used_as_model_input": False,
        "purpose": "Schema and label-policy smoke sample; not a final training corpus.",
    }
    return records, summary


def write_sample(
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
    parser = argparse.ArgumentParser(description="Build the pinned chooser v0 sample")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    args = parser.parse_args()
    records, summary = build_sample(args.registry)
    write_sample(
        records,
        summary,
        output_path=args.output,
        summary_path=args.summary,
        schema_path=args.schema,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
