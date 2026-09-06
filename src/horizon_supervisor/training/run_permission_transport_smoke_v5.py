from __future__ import annotations

import argparse
import asyncio
import json
import stat
import tempfile
from pathlib import Path
from typing import Any

from daytona import Daytona
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

from horizon_supervisor.benchmark.permission_preserving_daytona import (
    PermissionPreservingDaytonaEnvironment,
    deterministic_workspace_digest,
)
from horizon_supervisor.benchmark.pilot_harbor import (
    _WORKSPACE_DIGEST_COMMAND,
    SeededDaytonaEnvironment,
)
from horizon_supervisor.training.freeze_continuation_calibration import ROOT
from horizon_supervisor.training.run_continuation_calibration import _write_json
from horizon_supervisor.training.run_continuation_calibration_v5 import (
    MANIFEST,
    SMOKE_REPORT,
    validate_manifest,
)
from horizon_supervisor.training.run_stuck_pilot import _cleanup_new_sandboxes

GIT_OBJECT = Path(".git/objects/aa/permission-probe")


def _last_line(value: str | None) -> str:
    return (value or "").strip().splitlines()[-1]


async def _exercise_transport(temporary: Path) -> dict[str, Any]:
    environment_dir = temporary / "environment"
    environment_dir.mkdir(parents=True)
    (environment_dir / "Dockerfile").write_text(
        "FROM python:3.13-slim\nWORKDIR /app\n", encoding="utf-8"
    )
    config = EnvironmentConfig(cpus=1, memory_mb=2_048, workdir="/app")

    source_paths = TrialPaths(temporary / "source-trial")
    source_paths.mkdir()
    source = PermissionPreservingDaytonaEnvironment(
        environment_dir=environment_dir,
        environment_name="horizon-permission-source",
        session_id="horizon-permission-source",
        trial_paths=source_paths,
        task_env_config=config,
    )
    source_started = False
    snapshot = temporary / "downloaded-workspace"
    try:
        await source.start(force_build=False)
        source_started = True
        setup = await source.exec(
            "mkdir -p .git/objects/aa locked && "
            "printf 'immutable-object\\n' > .git/objects/aa/permission-probe && "
            "printf 'ordinary-file\\n' > payload.txt && "
            "chmod 0444 .git/objects/aa/permission-probe && chmod 0750 locked",
            cwd="/app",
            timeout_sec=120,
        )
        if setup.return_code != 0:
            raise RuntimeError(f"could not prepare transport probe: {setup.stderr}")
        source_digest_result = await source.exec(
            _WORKSPACE_DIGEST_COMMAND, cwd="/app", timeout_sec=120
        )
        if source_digest_result.return_code != 0:
            raise RuntimeError("could not digest source workspace")
        source_digest = _last_line(source_digest_result.stdout)
        await source.download_dir("/app", snapshot)
    finally:
        if source_started:
            await source.stop(delete=True)

    local_digest = deterministic_workspace_digest(snapshot)
    local_object_mode = stat.S_IMODE((snapshot / GIT_OBJECT).lstat().st_mode)

    target_paths = TrialPaths(temporary / "target-trial")
    target_paths.mkdir()
    target = SeededDaytonaEnvironment(
        environment_dir=environment_dir,
        environment_name="horizon-permission-target",
        session_id="horizon-permission-target",
        trial_paths=target_paths,
        task_env_config=config,
        workspace_seed_path=str(snapshot),
        expected_workspace_digest=source_digest,
    )
    target_started = False
    try:
        await target.start(force_build=False)
        target_started = True
        target_digest_result = await target.exec(
            _WORKSPACE_DIGEST_COMMAND, cwd="/app", timeout_sec=120
        )
        if target_digest_result.return_code != 0:
            raise RuntimeError("could not digest rehydrated workspace")
        target_digest = _last_line(target_digest_result.stdout)
        target_mode_result = await target.exec(
            "stat -c '%a' .git/objects/aa/permission-probe",
            cwd="/app",
            timeout_sec=30,
        )
        if target_mode_result.return_code != 0:
            raise RuntimeError("could not inspect rehydrated Git object mode")
        target_object_mode = int(_last_line(target_mode_result.stdout), 8)
    finally:
        if target_started:
            await target.stop(delete=True)

    return {
        "source_workspace_digest": source_digest,
        "local_workspace_digest": local_digest,
        "rehydrated_workspace_digest": target_digest,
        "local_git_object_mode": oct(local_object_mode),
        "rehydrated_git_object_mode": oct(target_object_mode),
        "remote_to_local_digest_match": source_digest == local_digest,
        "local_to_remote_digest_match": local_digest == target_digest,
        "read_only_git_object_mode_preserved": (
            local_object_mode == 0o444 and target_object_mode == 0o444
        ),
    }


def run(
    manifest_path: Path = MANIFEST, output_path: Path = SMOKE_REPORT
) -> dict[str, Any]:
    _, manifest_hash = validate_manifest(manifest_path)
    initial_ids = {sandbox.id for sandbox in Daytona().list()}
    if initial_ids:
        raise RuntimeError("permission transport smoke requires zero sandboxes")
    result: dict[str, Any] = {}
    error = None
    try:
        with tempfile.TemporaryDirectory() as temporary:
            result = asyncio.run(_exercise_transport(Path(temporary)))
    except Exception as caught:  # preserve a credential-free failure artifact
        error = f"{type(caught).__name__}: {caught}"
    cleanup = _cleanup_new_sandboxes(initial_ids)
    remaining = sorted({sandbox.id for sandbox in Daytona().list()} - initial_ids)
    passed = bool(result) and all(
        result[key]
        for key in (
            "remote_to_local_digest_match",
            "local_to_remote_digest_match",
            "read_only_git_object_mode_preserved",
        )
    )
    passed = passed and not cleanup["errors"] and not remaining and error is None
    report = {
        "schema_version": "permission-transport-smoke.v5",
        "passed": passed,
        "manifest_sha256": manifest_hash,
        "provider_model_calls": 0,
        "remote_to_local_digest_match": result.get(
            "remote_to_local_digest_match", False
        ),
        "local_to_remote_digest_match": result.get(
            "local_to_remote_digest_match", False
        ),
        "read_only_git_object_mode_preserved": result.get(
            "read_only_git_object_mode_preserved", False
        ),
        "local_git_object_mode": result.get("local_git_object_mode"),
        "rehydrated_git_object_mode": result.get("rehydrated_git_object_mode"),
        "source_workspace_digest": result.get("source_workspace_digest"),
        "local_workspace_digest": result.get("local_workspace_digest"),
        "rehydrated_workspace_digest": result.get("rehydrated_workspace_digest"),
        "cleanup_errors": cleanup["errors"],
        "remaining_daytona_environments": len(remaining),
        "error": error,
    }
    output_path = output_path.resolve()
    if output_path != ROOT / (
        "artifacts/official/two-tier-continuation-calibration-v5/"
        "permission-transport-smoke-v5.json"
    ):
        raise RuntimeError("v5 smoke output path changed")
    _write_json(output_path, report)
    if not passed:
        raise RuntimeError("permission-preserving Daytona transport smoke failed")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path, default=SMOKE_REPORT)
    args = parser.parse_args()
    print(json.dumps(run(args.manifest, args.output), indent=2))


if __name__ == "__main__":
    main()
