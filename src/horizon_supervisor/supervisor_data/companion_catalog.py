from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download

DEFAULT_REGISTRY = Path("data/supervisor/source-registry-v0.json")
DEFAULT_OUTPUT = Path("data/supervisor/companion-task-catalog-v0.jsonl")
DEFAULT_SUMMARY = Path("data/supervisor/companion-task-catalog-v0-summary.json")

HARBOR_REGISTRY_REVISION = "c459f5ac67a552cd75f745bee50182d94e7bfa14"
HARBOR_REGISTRY_SHA256 = "da1446bce05eabbd72a25eb9eef5a2f5db94645ce88c28e2497581433b3d2e60"
HARBOR_REGISTRY_URL = (
    "https://raw.githubusercontent.com/laude-institute/harbor/"
    f"{HARBOR_REGISTRY_REVISION}/registry.json"
)

HARBOR_COLLECTIONS = {
    "reasoning-gym-harbor-easy": {
        "category": "games",
        "task_filter": lambda name: "reasoning-gym-games-" in name,
        "difficulty": "easy",
    },
    "reasoning-gym-harbor-hard": {
        "category": "games",
        "task_filter": lambda name: "reasoning-gym-games-" in name,
        "difficulty": "hard",
    },
    "ml-dev-bench-harbor": {
        "category": "machine-learning",
        "task_filter": lambda name: True,
        "difficulty": "mixed",
    },
    "algotune-harbor": {
        "category": "optimization",
        "task_filter": lambda name: True,
        "difficulty": "mixed",
    },
    "gso-harbor": {
        "category": "optimization",
        "task_filter": lambda name: True,
        "difficulty": "hard",
    },
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sources_by_id(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {source["source_id"]: source for source in registry["sources"]}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _split(task_id: str) -> str:
    bucket = int(task_id[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "internal_test"


def _harbor_registry() -> list[dict[str, Any]]:
    with urllib.request.urlopen(HARBOR_REGISTRY_URL) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != HARBOR_REGISTRY_SHA256:
        raise RuntimeError(
            f"Harbor registry digest mismatch: expected {HARBOR_REGISTRY_SHA256}, got {digest}"
        )
    return json.loads(payload)


def _harbor_rows(
    source: dict[str, Any], registry: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    dataset_name, version = source["dataset_id"].split("@", maxsplit=1)
    dataset = next(
        (
            item
            for item in registry
            if item["name"] == dataset_name and str(item["version"]) == version
        ),
        None,
    )
    if dataset is None:
        raise ValueError(f"missing Harbor dataset {source['dataset_id']}")

    rows = []
    for task in dataset["tasks"]:
        if task["git_commit_id"] != source["revision"]:
            raise RuntimeError(
                f"task {task['name']} moved from pinned commit {source['revision']}"
            )
        if not config["task_filter"](task["name"]):
            continue
        task_id = _sha256(
            f"{source['source_id']}|{source['revision']}|{task['path']}|{task['name']}"
        )
        rows.append(
            {
                "schema_version": "companion-task-catalog.v0",
                "task_id": task_id,
                "source_task_name": task["name"],
                "category": config["category"],
                "difficulty": config["difficulty"],
                "recommended_split": _split(task_id),
                "source": {
                    "source_id": source["source_id"],
                    "dataset_id": source["dataset_id"],
                    "revision": source["revision"],
                    "git_url": task["git_url"],
                    "path": task["path"],
                    "license": source["license"],
                },
            }
        )
    return rows


def _workplace_rows(source: dict[str, Any]) -> list[dict[str, Any]]:
    path = Path(
        hf_hub_download(
            repo_id=source["dataset_id"],
            repo_type="dataset",
            filename="train.jsonl",
            revision=source["revision"],
        )
    )
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            source_task_name = f"train:{record['id']}"
            task_id = _sha256(
                f"{source['source_id']}|{source['revision']}|{source_task_name}"
            )
            rows.append(
                {
                    "schema_version": "companion-task-catalog.v0",
                    "task_id": task_id,
                    "source_task_name": source_task_name,
                    "category": "personal-assistant",
                    "source_category": record["category"],
                    "environment_name": record["environment_name"],
                    "difficulty": "mixed",
                    "recommended_split": _split(task_id),
                    "source": {
                        "source_id": source["source_id"],
                        "dataset_id": source["dataset_id"],
                        "revision": source["revision"],
                        "source_split": "train",
                        "license": source["license"],
                    },
                }
            )
    return rows


def _seta_machine_learning_rows(source: dict[str, Any]) -> list[dict[str, Any]]:
    info = HfApi().dataset_info(source["dataset_id"], revision=source["revision"])
    if info.sha != source["revision"]:
        raise RuntimeError(
            f"SETA revision mismatch: expected {source['revision']}, got {info.sha}"
        )
    rows = []
    for sibling in info.siblings:
        filename = sibling.rfilename
        if not filename.endswith("/instruction.md"):
            continue
        parts = filename.split("/")
        if len(parts) != 3 or not parts[1].startswith("kaggle_notebook__"):
            continue
        source_variant = parts[0]
        source_task_name = parts[1]
        task_id = _sha256(
            f"{source['source_id']}|{source['revision']}|{source_variant}|{source_task_name}"
        )
        rows.append(
            {
                "schema_version": "companion-task-catalog.v0",
                "task_id": task_id,
                "source_task_name": source_task_name,
                "category": "machine-learning",
                "source_category": "kaggle-notebook",
                "difficulty": "evolved" if source_variant == "SETA_Evolve" else "base",
                "recommended_split": _split(task_id),
                "source": {
                    "source_id": source["source_id"],
                    "dataset_id": source["dataset_id"],
                    "revision": source["revision"],
                    "path": f"{source_variant}/{source_task_name}",
                    "license": source["license"],
                },
            }
        )
    return rows


def build_companion_catalog(
    registry_path: Path = DEFAULT_REGISTRY,
    output_path: Path = DEFAULT_OUTPUT,
    summary_path: Path = DEFAULT_SUMMARY,
) -> dict[str, Any]:
    sources = _sources_by_id(_load_json(registry_path))
    harbor_registry = _harbor_registry()
    rows = []
    for source_id, config in HARBOR_COLLECTIONS.items():
        rows.extend(_harbor_rows(sources[source_id], harbor_registry, config))
    rows.extend(_seta_machine_learning_rows(sources["seta-env"]))
    rows.extend(_workplace_rows(sources["nemotron-workplace-assistant"]))
    rows.sort(key=lambda row: (row["category"], row["source"]["source_id"], row["task_id"]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    source_counts = Counter(row["source"]["source_id"] for row in rows)
    category_counts = Counter(row["category"] for row in rows)
    split_counts = Counter(row["recommended_split"] for row in rows)
    source_category_counts = Counter(
        row.get("source_category", "not-applicable") for row in rows
    )
    summary = {
        "schema_version": "companion-task-catalog-summary.v0",
        "record_count": len(rows),
        "source_counts": dict(sorted(source_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "source_category_counts": {
            key: value
            for key, value in sorted(source_category_counts.items())
            if key != "not-applicable"
        },
        "harbor_registry": {
            "revision": HARBOR_REGISTRY_REVISION,
            "sha256": HARBOR_REGISTRY_SHA256,
            "url": HARBOR_REGISTRY_URL,
        },
        "scope": (
            "Metadata only; instructions, solutions, tests, and ground truth are not "
            "retained."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Catalog benchmark-aligned companion tasks")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    summary = build_companion_catalog(args.registry, args.output, args.summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
