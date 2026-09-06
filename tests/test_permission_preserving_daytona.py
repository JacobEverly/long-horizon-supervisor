from __future__ import annotations

import io
import os
import stat
import tarfile
from pathlib import Path

import pytest

from horizon_supervisor.benchmark.permission_preserving_daytona import (
    deterministic_workspace_digest,
    extract_permission_preserving_archive,
)


def _archive(source: Path, destination: Path) -> None:
    with tarfile.open(destination, mode="w:gz") as archive:
        archive.add(source, arcname=".")


def test_safe_extract_preserves_read_only_execute_symlink_and_empty_dir(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir(mode=0o750)
    readonly = source / "git-object"
    readonly.write_bytes(b"object")
    readonly.chmod(0o444)
    executable = source / "run.sh"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    (source / "empty").mkdir(mode=0o700)
    os.symlink("git-object", source / "object-link")
    archive = tmp_path / "checkpoint.tar.gz"
    _archive(source, archive)

    destination = tmp_path / "destination"
    extract_permission_preserving_archive(archive, destination)

    assert stat.S_IMODE((destination / "git-object").stat().st_mode) == 0o444
    assert stat.S_IMODE((destination / "run.sh").stat().st_mode) == 0o755
    assert stat.S_IMODE((destination / "empty").stat().st_mode) == 0o700
    assert (destination / "object-link").is_symlink()
    assert deterministic_workspace_digest(destination) == deterministic_workspace_digest(
        source
    )


def test_safe_extract_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "escape.tar.gz"
    with tarfile.open(archive, mode="w:gz") as output:
        member = tarfile.TarInfo("../../escape")
        payload = b"no"
        member.size = len(payload)
        output.addfile(member, io.BytesIO(payload))

    with pytest.raises(tarfile.FilterError):
        extract_permission_preserving_archive(archive, tmp_path / "destination")
    assert not (tmp_path / "escape").exists()


def test_safe_extract_rejects_special_permission_bits(tmp_path: Path) -> None:
    archive = tmp_path / "special.tar.gz"
    with tarfile.open(archive, mode="w:gz") as output:
        member = tarfile.TarInfo("setuid-tool")
        member.mode = 0o4755
        payload = b"tool"
        member.size = len(payload)
        output.addfile(member, io.BytesIO(payload))

    with pytest.raises(tarfile.FilterError):
        extract_permission_preserving_archive(archive, tmp_path / "destination")
