"""Build a deterministic, credential-redacted public ATIF trace archive."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

BENCHMARK_REVISION = "b79dec3e67acd466f90127cde0d24516d9132e74"


REDACTIONS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "private_key",
        re.compile(
            r"-----BEGIN [^-\n]*PRIVATE KEY-----.*?-----END [^-\n]*PRIVATE KEY-----",
            re.DOTALL,
        ),
        "<REDACTED_PRIVATE_KEY>",
    ),
    (
        "private_key_header",
        re.compile(r"-----(?:BEGIN|END) [^-\n]*PRIVATE KEY-----"),
        "<REDACTED_PRIVATE_KEY_HEADER>",
    ),
    ("daytona_key", re.compile(r"\bdtn_[A-Za-z0-9]{24,}\b"), "<REDACTED_TOKEN>"),
    ("openai_style_key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"), "<REDACTED_TOKEN>"),
    ("huggingface_token", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"), "<REDACTED_TOKEN>"),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"), "<REDACTED_TOKEN>"),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "<REDACTED_TOKEN>"),
    (
        "bearer_token",
        re.compile(r"(?i)(authorization\s*:\s*bearer\s+)[A-Za-z0-9._~+/=-]{12,}"),
        r"\1<REDACTED_TOKEN>",
    ),
    (
        "credential_assignment",
        re.compile(
            r"(?i)\b(api[_-]?key|access[_-]?token|auth[_-]?token|password|passwd|secret)"
            r"(\s*[:=]\s*)([\"']?)[^\s,;\"']{8,}"
        ),
        r"\1\2<REDACTED_CREDENTIAL>",
    ),
    (
        "credential_url",
        re.compile(r"(?i)([a-z][a-z0-9+.-]*://)[^\s:/]+:[^\s@/]+@"),
        r"\1<REDACTED_CREDENTIAL>@",
    ),
    (
        "email",
        re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"),
        "<REDACTED_EMAIL>",
    ),
    ("local_home", re.compile(r"/Users/[^/\s\"']+"), "<LOCAL_HOME>"),
)


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _duration_seconds(result: dict[str, Any] | None, trajectory: dict[str, Any]) -> float:
    execution = (result or {}).get("agent_execution") or {}
    start = _parse_time(execution.get("started_at"))
    finish = _parse_time(execution.get("finished_at"))
    if start and finish:
        return max(0.0, (finish - start).total_seconds())

    steps = trajectory.get("steps") or []
    if not steps:
        return 0.0
    start = _parse_time(steps[0].get("timestamp"))
    finish = _parse_time(steps[-1].get("timestamp"))
    return max(0.0, (finish - start).total_seconds()) if start and finish else 0.0


def redact_text(value: str, counts: Counter[str] | None = None) -> str:
    """Remove credential-shaped strings, emails, and personal local paths."""

    redaction_counts = counts if counts is not None else Counter()
    for name, pattern, replacement in REDACTIONS:
        value, replacements = pattern.subn(replacement, value)
        redaction_counts[name] += replacements
    return value


def redact_value(value: Any, counts: Counter[str]) -> Any:
    if isinstance(value, str):
        return redact_text(value, counts)
    if isinstance(value, list):
        return [redact_value(item, counts) for item in value]
    if isinstance(value, dict):
        return {key: redact_value(item, counts) for key, item in value.items()}
    return value


def _reward(result: dict[str, Any] | None) -> float | None:
    rewards = ((result or {}).get("verifier_result") or {}).get("rewards") or {}
    reward = rewards.get("reward")
    return float(reward) if isinstance(reward, (int, float)) else None


def _release_row(
    trajectory_path: Path,
    artifacts_dir: Path,
    counts: Counter[str],
) -> dict[str, Any]:
    trajectory = json.loads(trajectory_path.read_text())
    result_path = trajectory_path.parent.parent / "result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else None
    session_id = str(trajectory.get("session_id") or trajectory_path)
    trace_id = hashlib.sha256(session_id.encode()).hexdigest()
    relative = trajectory_path.relative_to(artifacts_dir)
    experiment = relative.parts[1] if relative.parts[0] == "official" else relative.parts[0]
    reward = _reward(result)
    exception = bool((result or {}).get("exception_info"))

    return {
        "schema_version": "public-agent-trace.v0",
        "trace_id": trace_id,
        "source": {
            "benchmark": "alibabagroup/terminal-bench-pro",
            "benchmark_license": "apache-2.0",
            "benchmark_revision": BENCHMARK_REVISION,
            "experiment": experiment,
        },
        "task_name": (result or {}).get("task_name"),
        "model": (trajectory.get("agent") or {}).get("model_name"),
        "duration_seconds": round(_duration_seconds(result, trajectory), 6),
        "step_count": len(trajectory.get("steps") or []),
        "status": "errored" if exception else ("verified" if reward is not None else "unverified"),
        "completed": reward == 1.0 if reward is not None else None,
        "trajectory": redact_value(trajectory, counts),
    }


def build_release(artifacts_dir: Path, output_path: Path, manifest_path: Path) -> dict[str, Any]:
    paths = sorted(artifacts_dir.rglob("trajectory.json"))
    counts: Counter[str] = Counter()
    rows = [_release_row(path, artifacts_dir, counts) for path in paths]
    if len({row["trace_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate session IDs found; refusing to publish ambiguous traces")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as archive:
            for row in rows:
                archive.write(json.dumps(row, sort_keys=True, separators=(",", ":")).encode())
                archive.write(b"\n")

    status_counts = Counter(row["status"] for row in rows)
    manifest = {
        "schema_version": "public-agent-trace-manifest.v0",
        "archive": output_path.name,
        "archive_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "trajectory_count": len(rows),
        "agent_execution_seconds": round(sum(row["duration_seconds"] for row in rows), 6),
        "agent_execution_hours": round(
            sum(row["duration_seconds"] for row in rows) / 3600,
            1,
        ),
        "status_counts": dict(sorted(status_counts.items())),
        "redaction_counts": dict(sorted(counts.items())),
        "source": {
            "benchmark": "alibabagroup/terminal-bench-pro",
            "benchmark_license": "apache-2.0",
            "benchmark_revision": BENCHMARK_REVISION,
            "dataset_card": "https://huggingface.co/datasets/alibabagroup/terminal-bench-pro",
        },
        "release_guard": (
            "Includes successful, failed, interrupted, and infrastructure-affected traces. "
            "A record is not training-valid unless a downstream audit says so."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/data/public-agent-traces-v0.jsonl.gz"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("docs/data/public-agent-traces-v0-manifest.json"),
    )
    args = parser.parse_args()
    print(json.dumps(build_release(args.artifacts, args.output, args.manifest), indent=2))


if __name__ == "__main__":
    main()
