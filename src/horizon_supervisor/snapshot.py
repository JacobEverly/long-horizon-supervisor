from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BranchAction(StrEnum):
    CONTINUE_STATE = "continue_state"
    RESTART_CLEAN = "restart_clean"
    SWITCH_STATE = "switch_state"


class SnapshotCounters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turn: int = Field(ge=0)
    max_turns: int = Field(gt=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cached_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    output_token_budget: int = Field(gt=0)
    spent_usd: float = Field(default=0, ge=0)
    spend_budget_usd: float = Field(gt=0)


class PublicTestState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observed: bool = False
    passed: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    failure_fingerprints: tuple[str, ...] = ()
    workspace_digest: str | None = None


class ServiceSpec(BaseModel):
    """A public, deterministic recipe for rehydrating a relevant service."""

    model_config = ConfigDict(extra="forbid")

    name: str
    start_command: tuple[str, ...]
    cwd: str = "."
    healthcheck_command: tuple[str, ...] = ()

    @field_validator("start_command", "healthcheck_command")
    @classmethod
    def commands_cannot_embed_secret_assignments(
        cls, value: tuple[str, ...]
    ) -> tuple[str, ...]:
        joined = " ".join(value).lower()
        forbidden = ("api_key=", "token=", "password=", "secret=")
        if any(marker in joined for marker in forbidden):
            raise ValueError("service recipes cannot embed secret assignments")
        return value


class FileRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    kind: Literal["file", "directory", "symlink"]
    mode: int
    sha256: str | None = None
    symlink_target: str | None = None


class GitState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    present: bool
    head: str | None = None
    status_porcelain_v2: str = ""


class SnapshotManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["normalized-snapshot.v0"] = "normalized-snapshot.v0"
    snapshot_id: str
    source_adapter: str
    workspace_digest: str
    files: tuple[FileRecord, ...]
    git: GitState
    environment_metadata: dict[str, str]
    public_test_state: PublicTestState
    counters: SnapshotCounters
    service_specs: tuple[ServiceSpec, ...] = ()
    process_state_fidelity: Literal["none_required", "recipe_rehydrated"]
    transcript_path: str | None = None
    contains_private_reasoning: Literal[False] = False
    contains_hidden_verifier_artifacts: Literal[False] = False
    contains_provider_secrets: Literal[False] = False


class BranchDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: BranchAction
    source_snapshot_id: str | None
    destination_model_id: str
    remaining_turns: int = Field(ge=0)
    remaining_output_tokens: int = Field(ge=0)
    maximum_incremental_spend_usd: float = Field(ge=0)
    maximum_wall_seconds: int = Field(gt=0)


class SnapshotAdapter(Protocol):
    """Harness-owned boundary used by the portable supervisor."""

    def snapshot(self, *args: object, **kwargs: object) -> SnapshotManifest: ...

    def clone(self, *args: object, **kwargs: object) -> SnapshotManifest: ...


_SECRET_FILENAMES = {
    ".env",
    ".env.local",
    "id_rsa",
    "id_ed25519",
    "credentials.json",
    "service-account.json",
}
_SENSITIVE_ENV_MARKERS = ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "PRIVATE_KEY")


def _file_records(root: Path) -> tuple[FileRecord, ...]:
    records: list[FileRecord] = []
    paths = sorted(
        [root, *root.rglob("*")],
        key=lambda item: item.relative_to(root).as_posix(),
    )
    for path in paths:
        relative = "." if path == root else path.relative_to(root).as_posix()
        info = path.lstat()
        mode = stat.S_IMODE(info.st_mode)
        if path.is_symlink():
            records.append(
                FileRecord(
                    path=relative,
                    kind="symlink",
                    mode=mode,
                    symlink_target=os.readlink(path),
                )
            )
        elif path.is_dir():
            records.append(FileRecord(path=relative, kind="directory", mode=mode))
        elif path.is_file():
            records.append(
                FileRecord(
                    path=relative,
                    kind="file",
                    mode=mode,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            )
    return tuple(records)


def deterministic_workspace_digest(root: Path) -> str:
    records = [record.model_dump(mode="json") for record in _file_records(root.resolve())]
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def inspect_git_state(root: Path) -> GitState:
    if not (root / ".git").exists():
        return GitState(present=False)
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status_output = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain=v2", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return GitState(present=True, head=head, status_porcelain_v2=status_output)


def safe_environment_metadata(
    environment: dict[str, str], allowlist: tuple[str, ...]
) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for key in allowlist:
        upper = key.upper()
        if any(marker in upper for marker in _SENSITIVE_ENV_MARKERS):
            raise ValueError(f"sensitive environment name is not allowed: {key}")
        if key in environment:
            metadata[key] = environment[key]
    return metadata


def assert_no_secret_files(root: Path) -> None:
    offenders = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name.lower() in _SECRET_FILENAMES
    ]
    if offenders:
        raise ValueError(f"snapshot workspace contains forbidden secret files: {offenders}")


class FilesystemSnapshotAdapter:
    """Local reference adapter used to prove snapshot invariants.

    Process memory is deliberately not claimed as portable. Relevant services
    must either be absent or represented by public deterministic restart recipes.
    """

    adapter_name = "filesystem-copy-v0"

    def __init__(self, snapshot_root: Path) -> None:
        self.snapshot_root = snapshot_root.resolve()
        self.snapshot_root.mkdir(parents=True, exist_ok=True)

    def snapshot(
        self,
        *,
        snapshot_id: str,
        workspace: Path,
        counters: SnapshotCounters,
        public_test_state: PublicTestState,
        environment: dict[str, str],
        environment_allowlist: tuple[str, ...],
        service_specs: tuple[ServiceSpec, ...] = (),
        transcript_path: str | None = None,
    ) -> SnapshotManifest:
        source = workspace.resolve()
        assert_no_secret_files(source)
        # Git may refresh its index stat cache while computing status. Capture
        # semantic Git state before the byte-for-byte copy so the snapshot
        # payload is stable after it is written.
        git = inspect_git_state(source)
        destination = self.snapshot_root / snapshot_id
        if destination.exists():
            raise FileExistsError(f"snapshot already exists: {snapshot_id}")
        payload = destination / "workspace"
        destination.mkdir(parents=True)
        shutil.copytree(source, payload, symlinks=True, copy_function=shutil.copy2)
        files = _file_records(payload)
        workspace_digest = deterministic_workspace_digest(payload)
        process_fidelity = "recipe_rehydrated" if service_specs else "none_required"
        manifest = SnapshotManifest(
            snapshot_id=snapshot_id,
            source_adapter=self.adapter_name,
            workspace_digest=workspace_digest,
            files=files,
            git=git,
            environment_metadata=safe_environment_metadata(
                environment, environment_allowlist
            ),
            public_test_state=public_test_state,
            counters=counters,
            service_specs=service_specs,
            process_state_fidelity=process_fidelity,
            transcript_path=transcript_path,
        )
        (destination / "manifest.json").write_text(
            json.dumps(manifest.model_dump(mode="json"), indent=2), encoding="utf-8"
        )
        return manifest

    def clone(
        self, *, snapshot_id: str, clone_id: str, destination_root: Path
    ) -> SnapshotManifest:
        snapshot_dir = self.snapshot_root / snapshot_id
        manifest = SnapshotManifest.model_validate_json(
            (snapshot_dir / "manifest.json").read_text(encoding="utf-8")
        )
        destination = destination_root.resolve() / clone_id
        if destination.exists():
            raise FileExistsError(f"clone already exists: {clone_id}")
        shutil.copytree(
            snapshot_dir / "workspace", destination, symlinks=True, copy_function=shutil.copy2
        )
        if deterministic_workspace_digest(destination) != manifest.workspace_digest:
            raise RuntimeError("cloned workspace digest does not match snapshot")
        files = _file_records(destination)
        git = inspect_git_state(destination)
        if git != manifest.git:
            raise RuntimeError("cloned Git state does not match snapshot")
        if files != manifest.files:
            raise RuntimeError("cloned file contents or permissions do not match snapshot")
        return manifest


def verify_clone_isolation(clones: tuple[Path, ...]) -> dict[str, object]:
    if len(clones) < 2:
        raise ValueError("at least two clones are required")
    initial = [deterministic_workspace_digest(path) for path in clones]
    marker = clones[0] / ".horizon-isolation-probe"
    marker.write_text("branch-zero-only\n", encoding="utf-8")
    final = [deterministic_workspace_digest(path) for path in clones]
    isolated = final[0] != initial[0] and all(
        final[index] == initial[index] and not (clones[index] / marker.name).exists()
        for index in range(1, len(clones))
    )
    return {
        "clone_count": len(clones),
        "isolated": isolated,
        "initial_digests": initial,
        "final_digests": final,
    }
