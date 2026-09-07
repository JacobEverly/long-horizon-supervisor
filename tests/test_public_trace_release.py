from __future__ import annotations

import gzip
import json
from pathlib import Path

from horizon_supervisor.supervisor_data.public_trace_release import build_release, redact_text


def test_redact_text_removes_credentials_and_local_identity() -> None:
    source = (
        "Authorization: Bearer secret-token-value-12345 "
        "sk-exampleexampleexampleexample "
        "person@example.com /Users/jacob/project "
        "-----BEGIN OPENSSH PRIVATE KEY-----"
    )

    redacted = redact_text(source)

    assert "secret-token-value" not in redacted
    assert "sk-example" not in redacted
    assert "person@example.com" not in redacted
    assert "/Users/jacob" not in redacted
    assert "BEGIN OPENSSH PRIVATE KEY" not in redacted


def test_build_release_is_deterministic_and_labels_errors(tmp_path: Path) -> None:
    trial = tmp_path / "artifacts" / "official" / "pilot" / "trial"
    agent = trial / "agent"
    agent.mkdir(parents=True)
    trajectory = {
        "schema_version": "ATIF-v1.7",
        "session_id": "session-1",
        "agent": {"model_name": "openai/example"},
        "steps": [
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "message": "token=ghp_abcdefghijklmnopqrstuvwxyz123456",
            },
            {"timestamp": "2026-01-01T00:01:00Z", "message": "done"},
        ],
    }
    (agent / "trajectory.json").write_text(json.dumps(trajectory))
    (trial / "result.json").write_text(
        json.dumps(
            {
                "trial_name": "trial",
                "task_name": "task",
                "agent_execution": {
                    "started_at": "2026-01-01T00:00:00Z",
                    "finished_at": "2026-01-01T00:02:00Z",
                },
                "exception_info": {"type": "example"},
            }
        )
    )
    first = tmp_path / "first.jsonl.gz"
    second = tmp_path / "second.jsonl.gz"
    first_manifest = tmp_path / "first-manifest.json"
    second_manifest = tmp_path / "second-manifest.json"

    manifest = build_release(tmp_path / "artifacts", first, first_manifest)
    build_release(tmp_path / "artifacts", second, second_manifest)

    assert first.read_bytes() == second.read_bytes()
    assert manifest["trajectory_count"] == 1
    assert manifest["agent_execution_seconds"] == 120
    with gzip.open(first, "rt") as archive:
        row = json.loads(archive.readline())
    assert row["status"] == "errored"
    assert "ghp_" not in json.dumps(row)
