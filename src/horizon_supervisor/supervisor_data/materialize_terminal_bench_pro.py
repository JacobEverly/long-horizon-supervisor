from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

DEFAULT_PANEL = Path("data/supervisor/terminal-bench-pro-panel-v0.jsonl")
DEFAULT_OUTPUT_ROOT = Path("data/supervisor/terminal-bench-pro-wave-1")
SOURCE_FILE = "Terminal_Bench_Pro_Public-200.parquet"


def _iter_jsonl(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _safe_members(archive: tarfile.TarFile, expected_root: str) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe archive path {member.name!r}")
        if not path.parts or path.parts[0] != expected_root:
            raise ValueError(
                f"archive member {member.name!r} is outside {expected_root!r}"
            )
        if member.issym() or member.islnk():
            raise ValueError(f"links are not allowed in task archive: {member.name!r}")
    return members


def materialize_wave(
    panel_path: Path = DEFAULT_PANEL,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    wave: int = 1,
) -> dict[str, Any]:
    selected = [row for row in _iter_jsonl(panel_path) if row["wave"] == wave]
    if not selected:
        raise ValueError(f"panel contains no tasks for wave {wave}")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output_root}")

    source = selected[0]["source"]
    parquet_path = Path(
        hf_hub_download(
            repo_id=source["dataset_id"],
            repo_type="dataset",
            filename=SOURCE_FILE,
            revision=source["revision"],
        )
    )
    wanted = {row["source_task_name"]: row for row in selected}
    source_rows = {
        row["task_id"]: row
        for row in pq.read_table(
            parquet_path,
            columns=["task_id", "instruction", "config", "archive"],
        ).to_pylist()
        if row["task_id"] in wanted
    }
    missing = wanted.keys() - source_rows.keys()
    if missing:
        raise ValueError(f"missing {len(missing)} selected tasks in pinned Parquet")

    tasks_root = output_root / "tasks"
    tasks_root.mkdir(parents=True, exist_ok=True)
    file_counts: dict[str, int] = {}
    for source_task_name, panel_row in sorted(wanted.items()):
        row = source_rows[source_task_name]
        archive_bytes = row["archive"]
        lock = panel_row["execution_lock"]
        if _sha256_bytes(archive_bytes) != lock["archive_sha256"]:
            raise RuntimeError(f"archive digest changed for {source_task_name}")
        if _sha256_bytes(row["config"].encode()) != lock["config_sha256"]:
            raise RuntimeError(f"config digest changed for {source_task_name}")
        if _sha256_bytes(row["instruction"].encode()) != panel_row["instruction_sha256"]:
            raise RuntimeError(f"instruction digest changed for {source_task_name}")

        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
            members = _safe_members(archive, source_task_name)
            archive.extractall(tasks_root, members=members, filter="data")
            file_counts[source_task_name] = sum(member.isfile() for member in members)
        task_root = tasks_root / source_task_name
        for required in (
            "instruction.md",
            "task.toml",
            "environment/Dockerfile",
            "tests/test.sh",
        ):
            if not (task_root / required).is_file():
                raise RuntimeError(f"{source_task_name} is missing {required}")

    summary = {
        "schema_version": "terminal-bench-pro-materialization-summary.v0",
        "wave": wave,
        "task_count": len(selected),
        "tasks_root": str(tasks_root),
        "source": {
            "dataset_id": source["dataset_id"],
            "revision": source["revision"],
            "file": SOURCE_FILE,
        },
        "all_execution_locks_verified": True,
        "all_archive_paths_safe": True,
        "file_counts": file_counts,
        "harbor_command_shape": f"harbor run --path {tasks_root}",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize a pinned rollout wave")
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--wave", type=int, default=1)
    args = parser.parse_args()
    summary = materialize_wave(args.panel, args.output_root, wave=args.wave)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
