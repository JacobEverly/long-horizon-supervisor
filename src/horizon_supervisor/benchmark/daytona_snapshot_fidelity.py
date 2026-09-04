from __future__ import annotations

import hashlib
import json
import os
import shlex
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


def _exec(sandbox: Any, command: str, *, timeout: int = 60) -> str:
    response = sandbox.process.exec(command, timeout=timeout)
    if response.exit_code != 0:
        raise RuntimeError(
            f"sandbox command failed with exit {response.exit_code}: {response.result}"
        )
    return str(response.result)


_WORKSPACE = "/home/daytona/workspace"


_REMOTE_MANIFEST = r'''python3 - <<'PY'
import hashlib,json,os,stat
from pathlib import Path
root=Path('/home/daytona/workspace')
rows=[]
for path in sorted([root,*root.rglob('*')],key=lambda p: p.relative_to(root).as_posix()):
    rel='.' if path==root else path.relative_to(root).as_posix()
    info=path.lstat(); mode=stat.S_IMODE(info.st_mode)
    if path.is_symlink():
        rows.append({'path':rel,'kind':'symlink','mode':mode,'target':os.readlink(path)})
    elif path.is_dir(): rows.append({'path':rel,'kind':'directory','mode':mode})
    elif path.is_file():
        rows.append({'path':rel,'kind':'file','mode':mode,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
payload=json.dumps(rows,sort_keys=True,separators=(',',':'))
print(json.dumps({'digest':hashlib.sha256(payload.encode()).hexdigest(),'files':rows}))
PY'''


def _manifest(sandbox: Any) -> dict[str, Any]:
    return json.loads(_exec(sandbox, _REMOTE_MANIFEST).strip())


def _git_state(sandbox: Any) -> dict[str, str]:
    output = _exec(
        sandbox,
        f"git -C {_WORKSPACE} rev-parse HEAD && "
        "printf '\\n--STATUS--\\n' && "
        f"git -C {_WORKSPACE} status --porcelain=v2 --untracked-files=all",
    )
    head, status = output.split("\n--STATUS--\n", maxsplit=1)
    return {"head": head.strip(), "status_porcelain_v2": status}


def _health(sandbox: Any) -> bool:
    response = sandbox.process.exec(
        "python3 -c \"import urllib.request; "
        "print(urllib.request.urlopen('http://127.0.0.1:8123',timeout=2).status)\"",
        timeout=5,
    )
    return response.exit_code == 0 and "200" in str(response.result)


def run_daytona_snapshot_fidelity(output_path: Path) -> dict[str, Any]:
    """Exercise the exact filesystem-snapshot fallback used by the pilot.

    Daytona snapshots preserve filesystem state, not process memory. The pilot
    therefore records explicit public service recipes and starts them anew in
    each clone. A service that cannot be represented this way is ineligible.
    """

    if not os.getenv("DAYTONA_API_KEY"):
        raise RuntimeError("DAYTONA_API_KEY is required")

    from daytona import Daytona

    daytona = Daytona()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleanup_path = output_path.with_name("snapshot-fidelity-cleanup-errors.json")
    cleanup_path.unlink(missing_ok=True)
    suffix = uuid4().hex[:12]
    source = None
    clones: list[Any] = []
    cleanup_errors: list[str] = []
    started_at = datetime.now(UTC)
    try:
        source = daytona.create(timeout=120)
        _exec(
            source,
            f"mkdir -p {_WORKSPACE} && cd {_WORKSPACE} && git init -q && "
            "git config user.email fidelity@example.invalid && "
            "git config user.name HorizonFidelity && "
            "printf '#!/bin/sh\\necho ready\\n' > run.sh && chmod 751 run.sh && "
            "printf 'original\\n' > tracked.txt && git add run.sh tracked.txt && "
            "git commit -qm base && printf 'dirty\\n' > tracked.txt && "
            "printf 'untracked\\n' > untracked.txt && ln -s tracked.txt link && "
            "printf 'public test passed\\n' > public-test-state.txt",
        )
        _exec(
            source,
            f"cd {_WORKSPACE} && nohup python3 -m http.server 8123 "
            ">/tmp/horizon-fidelity-service.log 2>&1 </dev/null &",
        )
        source_service_healthy = _health(source)
        source_git = _git_state(source)
        source_manifest = _manifest(source)
        archive_remote = f"/tmp/horizon-fidelity-{suffix}.tar.gz"
        _exec(
            source,
            f"tar czpf {archive_remote} -C {_WORKSPACE} .",
            timeout=120,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive_local = Path(temporary_directory) / "workspace.tar.gz"
            source.fs.download_file(archive_remote, str(archive_local))
            for _index in range(3):
                clone = daytona.create(timeout=120)
                clones.append(clone)
                clone_archive = f"/tmp/horizon-fidelity-{suffix}.tar.gz"
                clone.fs.upload_file(str(archive_local), clone_archive)
                _exec(
                    clone,
                    f"mkdir -p {_WORKSPACE} && tar xzpf {clone_archive} "
                    f"-C {_WORKSPACE} && rm -f {clone_archive}",
                    timeout=120,
                )

        clone_manifests_transfer = [_manifest(clone) for clone in clones]
        clone_git = [_git_state(clone) for clone in clones]
        clone_manifests_before_isolation = [_manifest(clone) for clone in clones]
        process_memory_preserved = [_health(clone) for clone in clones]
        for clone in clones:
            _exec(
                clone,
                f"cd {_WORKSPACE} && nohup python3 -m http.server 8123 "
                ">/tmp/horizon-fidelity-service-rehydrated.log 2>&1 </dev/null &",
            )
        services_rehydrated = [_health(clone) for clone in clones]

        _exec(
            clones[0],
            f"printf 'clone-zero-only\\n' > {_WORKSPACE}/isolation-probe.txt",
        )
        clone_manifests_after = [_manifest(clone) for clone in clones]
        clone_isolation = (
            clone_manifests_after[0]["digest"]
            != clone_manifests_before_isolation[0]["digest"]
            and all(
                clone_manifests_after[index]["digest"]
                == clone_manifests_before_isolation[index]["digest"]
                for index in range(1, len(clones))
            )
        )
        filesystem_fidelity = all(
            manifest == source_manifest for manifest in clone_manifests_transfer
        )
        git_fidelity = all(state == source_git for state in clone_git)
        permissions_preserved = all(
            next(
                item["mode"]
                for item in manifest["files"]
                if item["path"] == "run.sh"
            )
            == 0o751
            for manifest in clone_manifests_transfer
        )

        report = {
            "schema_version": "daytona-snapshot-fidelity.v0",
            "created_at": datetime.now(UTC).isoformat(),
            "snapshot_mechanism": (
                "permission-preserving workspace archive downloaded from Daytona "
                "and rehydrated into fresh Daytona sandboxes"
            ),
            "clone_count": len(clones),
            "filesystem_fidelity": filesystem_fidelity,
            "permissions_preserved": permissions_preserved,
            "git_state_preserved": git_fidelity,
            "public_test_state_preserved": all(
                any(
                    item["path"] == "public-test-state.txt"
                    and item["sha256"]
                    == next(
                        source_item["sha256"]
                        for source_item in source_manifest["files"]
                        if source_item["path"] == "public-test-state.txt"
                    )
                    for item in manifest["files"]
                )
                for manifest in clone_manifests_transfer
            ),
            "deterministic_workspace_digest_preserved": all(
                manifest["digest"] == source_manifest["digest"]
                for manifest in clone_manifests_transfer
            ),
            "clone_isolation": clone_isolation,
            "source_service_healthy": source_service_healthy,
            "process_memory_preserved": process_memory_preserved,
            "service_recipe_rehydration_succeeded": services_rehydrated,
            "process_state_fidelity": "recipe_rehydrated",
            "process_limitation": (
                "The configured Daytona runtime supports neither timely reusable snapshots "
                "nor sandbox forks. The archive fallback preserves the task workspace, "
                "not live-process memory. "
                "The pilot may preserve only services with a frozen public restart recipe; "
                "all others are structurally ineligible."
            ),
            "environment_metadata_contract": (
                "Task-declared non-secret environment configuration is recreated by Harbor; "
                "provider keys and secret-valued variables are excluded."
            ),
            "counter_contract": (
                "Turn, token, wall-time, and provider-spend counters live in the normalized "
                "snapshot manifest and are applied identically to all branches."
            ),
            "source_workspace_digest": source_manifest["digest"],
            "duration_seconds": (datetime.now(UTC) - started_at).total_seconds(),
            "passed": all(
                [
                    filesystem_fidelity,
                    permissions_preserved,
                    git_fidelity,
                    clone_isolation,
                    source_service_healthy,
                    *services_rehydrated,
                ]
            ),
        }
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    finally:
        for sandbox in [*clones, source] if source is not None else clones:
            if sandbox is None:
                continue
            try:
                daytona.delete(sandbox, wait=True, timeout=120)
            except Exception as error:  # pragma: no cover - live cleanup path
                cleanup_errors.append(f"{type(error).__name__}: {error}")
        if cleanup_errors:
            cleanup_path.write_text(
                json.dumps({"errors": cleanup_errors}, indent=2), encoding="utf-8"
            )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_daytona_snapshot_fidelity(args.output)
    print(json.dumps({"passed": report["passed"], "output": shlex.quote(str(args.output))}))


if __name__ == "__main__":
    main()
