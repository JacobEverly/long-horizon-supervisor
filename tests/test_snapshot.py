import os
import stat
import subprocess
from pathlib import Path

import pytest

from horizon_supervisor.snapshot import (
    FilesystemSnapshotAdapter,
    PublicTestState,
    ServiceSpec,
    SnapshotCounters,
    assert_no_secret_files,
    deterministic_workspace_digest,
    safe_environment_metadata,
    verify_clone_isolation,
)


def counters() -> SnapshotCounters:
    return SnapshotCounters(
        turn=4,
        max_turns=12,
        input_tokens=100,
        output_tokens=200,
        output_token_budget=1_000,
        spent_usd=0.1,
        spend_budget_usd=1.0,
    )


def git_workspace(root: Path) -> Path:
    workspace = root / "workspace"
    workspace.mkdir()
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    subprocess.run(
        ["git", "-C", str(workspace), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(["git", "-C", str(workspace), "config", "user.name", "Test"], check=True)
    executable = workspace / "run.sh"
    executable.write_text("#!/bin/sh\necho ready\n", encoding="utf-8")
    executable.chmod(0o751)
    (workspace / "tracked.txt").write_text("original\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(workspace), "add", "."], check=True)
    subprocess.run(["git", "-C", str(workspace), "commit", "-qm", "base"], check=True)
    (workspace / "tracked.txt").write_text("dirty\n", encoding="utf-8")
    (workspace / "untracked.txt").write_text("untracked\n", encoding="utf-8")
    os.symlink("tracked.txt", workspace / "link")
    return workspace


def test_snapshot_preserves_files_permissions_git_state_and_metadata(tmp_path: Path) -> None:
    workspace = git_workspace(tmp_path)
    adapter = FilesystemSnapshotAdapter(tmp_path / "snapshots")
    manifest = adapter.snapshot(
        snapshot_id="checkpoint-1",
        workspace=workspace,
        counters=counters(),
        public_test_state=PublicTestState(
            observed=True,
            passed=3,
            failed=1,
            failure_fingerprints=("failure-a",),
            workspace_digest=deterministic_workspace_digest(workspace),
        ),
        environment={"PATH": "/usr/bin", "LANG": "C.UTF-8", "OPENROUTER_API_KEY": "never"},
        environment_allowlist=("PATH", "LANG"),
        service_specs=(
            ServiceSpec(
                name="demo",
                start_command=("python", "-m", "http.server", "8123"),
                healthcheck_command=("curl", "-fsS", "http://127.0.0.1:8123"),
            ),
        ),
        transcript_path="public-transcript.json",
    )
    clone_root = tmp_path / "clones"
    clone_root.mkdir()
    adapter.clone(snapshot_id="checkpoint-1", clone_id="a", destination_root=clone_root)
    adapter.clone(snapshot_id="checkpoint-1", clone_id="b", destination_root=clone_root)

    assert manifest.environment_metadata == {"PATH": "/usr/bin", "LANG": "C.UTF-8"}
    assert manifest.contains_private_reasoning is False
    assert manifest.process_state_fidelity == "recipe_rehydrated"
    assert stat.S_IMODE((clone_root / "a" / "run.sh").stat().st_mode) == 0o751
    assert (clone_root / "a" / "link").is_symlink()
    assert verify_clone_isolation((clone_root / "a", clone_root / "b"))["isolated"] is True


def test_snapshot_rejects_secret_files_and_sensitive_metadata(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".env").write_text("TOKEN=secret\n", encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden secret"):
        assert_no_secret_files(workspace)
    with pytest.raises(ValueError, match="sensitive environment"):
        safe_environment_metadata({"OPENROUTER_API_KEY": "secret"}, ("OPENROUTER_API_KEY",))


def test_digest_changes_with_permissions_and_symlink_target(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target"
    target.write_text("value", encoding="utf-8")
    os.symlink("target", workspace / "link")
    first = deterministic_workspace_digest(workspace)
    target.chmod(0o700)
    second = deterministic_workspace_digest(workspace)
    assert first != second
