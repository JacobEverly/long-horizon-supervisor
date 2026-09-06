from __future__ import annotations

import hashlib
import json
import os
import shlex
import stat
import tarfile
import tempfile
from pathlib import Path
from uuid import uuid4

from harbor.environments.daytona.environment import DaytonaEnvironment


def _safe_permission_preserving_filter(
    member: tarfile.TarInfo, destination: str
) -> tarfile.TarInfo | None:
    """Apply tar's data safety checks without normalizing ordinary rwx bits."""
    original_mode = member.mode
    filtered = tarfile.data_filter(member, destination)
    if filtered is None:
        return None
    if original_mode & 0o7000:
        raise tarfile.FilterError(
            f"special permission bits are forbidden: {member.name!r}"
        )
    filtered.mode = original_mode & 0o777
    return filtered


def extract_permission_preserving_archive(
    archive_path: Path, destination: Path
) -> None:
    """Extract a trusted-origin checkpoint safely and preserve ordinary modes."""
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="r:gz") as archive:
        archive.extractall(destination, filter=_safe_permission_preserving_filter)


def deterministic_workspace_digest(root: Path) -> str:
    """Match the frozen remote workspace digest including paths and POSIX modes."""
    root = root.resolve()
    rows: list[list[str | int]] = []
    paths = sorted(
        [root, *root.rglob("*")], key=lambda path: path.relative_to(root).as_posix()
    )
    for path in paths:
        relative = "." if path == root else path.relative_to(root).as_posix()
        mode = stat.S_IMODE(path.lstat().st_mode)
        if path.is_symlink():
            rows.append([relative, "l", mode, os.readlink(path)])
        elif path.is_dir():
            rows.append([relative, "d", mode])
        elif path.is_file():
            rows.append(
                [relative, "f", mode, hashlib.sha256(path.read_bytes()).hexdigest()]
            )
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


class PermissionPreservingDaytonaEnvironment(DaytonaEnvironment):
    """Daytona transport that keeps checkpoint file modes without unsafe extraction."""

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        if os.name == "nt":  # pragma: no cover - Phase A runs on POSIX only
            await super().download_dir(source_dir, target_dir)
            return

        destination = Path(target_dir)
        remote_archive = f"/tmp/horizon_checkpoint_{uuid4().hex}.tar.gz"
        remote_digest_command = (
            "python3 - <<'PY'\n"
            "import hashlib,json,os,stat\n"
            "from pathlib import Path\n"
            "root=Path('.').resolve(); rows=[]\n"
            "for p in sorted([root,*root.rglob('*')],"
            "key=lambda x:x.relative_to(root).as_posix()):\n"
            " r='.' if p==root else p.relative_to(root).as_posix(); "
            "s=p.lstat(); m=stat.S_IMODE(s.st_mode)\n"
            " if p.is_symlink(): rows.append([r,'l',m,os.readlink(p)])\n"
            " elif p.is_dir(): rows.append([r,'d',m])\n"
            " elif p.is_file(): rows.append([r,'f',m,"
            "hashlib.sha256(p.read_bytes()).hexdigest()])\n"
            "payload=json.dumps(rows,sort_keys=True,separators=(',',':'))\n"
            "print(hashlib.sha256(payload.encode()).hexdigest())\n"
            "PY"
        )
        digest_result = await self.exec(
            remote_digest_command, cwd=source_dir, timeout_sec=120
        )
        if digest_result.return_code != 0:
            raise RuntimeError(
                f"failed to digest checkpoint source: {digest_result.stderr}"
            )
        expected_digest = (digest_result.stdout or "").strip().splitlines()[-1]

        pack = await self.exec(
            f"tar -czf {shlex.quote(remote_archive)} -C "
            f"{shlex.quote(source_dir)} .",
            cwd=source_dir,
            timeout_sec=600,
        )
        if pack.return_code != 0:
            raise RuntimeError(f"failed to archive checkpoint: {pack.stderr}")
        try:
            with tempfile.TemporaryDirectory() as temporary:
                local_archive = Path(temporary) / "checkpoint.tar.gz"
                await super().download_file(remote_archive, local_archive)
                extract_permission_preserving_archive(local_archive, destination)
        finally:
            await self.exec(
                f"rm -f -- {shlex.quote(remote_archive)}",
                cwd=source_dir,
                timeout_sec=30,
            )

        actual_digest = deterministic_workspace_digest(destination)
        if actual_digest != expected_digest:
            raise RuntimeError(
                "permission-preserving checkpoint transfer digest mismatch: "
                f"expected {expected_digest}, got {actual_digest}"
            )
