from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def _resolve(workspace_dir: str, relative_path: str) -> Path:
    root = Path(workspace_dir).resolve()
    target = (root / relative_path).resolve()
    if target != root and root not in target.parents:
        raise ValueError("path escapes the task workspace")
    return target


def list_files(path: str = ".", *, workspace_dir: str) -> str:
    """List files below a workspace-relative directory."""
    root = _resolve(workspace_dir, path)
    if not root.exists():
        raise FileNotFoundError(path)
    files = [
        str(candidate.relative_to(Path(workspace_dir)))
        for candidate in root.rglob("*")
        if candidate.is_file() and not any(part.startswith(".") for part in candidate.parts)
    ]
    return "\n".join(sorted(files)) or "(no files)"


def read_file(path: str, *, workspace_dir: str) -> str:
    """Read a UTF-8 text file using a workspace-relative path."""
    target = _resolve(workspace_dir, path)
    if target.stat().st_size > 100_000:
        raise ValueError("file is too large")
    return target.read_text(encoding="utf-8")


def write_file(path: str, content: str, *, workspace_dir: str) -> str:
    """Create or replace a UTF-8 text file inside the task workspace."""
    target = _resolve(workspace_dir, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"wrote {len(content.encode('utf-8'))} bytes to {path}"


def replace_in_file(old: str, new: str, path: str, *, workspace_dir: str) -> str:
    """Replace one exact, unique text fragment in a workspace file."""
    target = _resolve(workspace_dir, path)
    content = target.read_text(encoding="utf-8")
    count = content.count(old)
    if count != 1:
        raise ValueError(f"expected exactly one match, found {count}")
    target.write_text(content.replace(old, new), encoding="utf-8")
    return f"replaced one fragment in {path}"


def _run_suite(workspace_dir: str, tests_dir: str, timeout_seconds: int = 20) -> dict[str, object]:
    workspace = Path(workspace_dir).resolve()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(workspace / "src")
    result = subprocess.run(
        [
            os.environ.get("PYTHON", "python"),
            "-m",
            "unittest",
            "discover",
            "-s",
            tests_dir,
            "-v",
        ],
        cwd=workspace,
        env=environment,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    output = (result.stdout + result.stderr)[-12_000:]
    return {"passed": result.returncode == 0, "returncode": result.returncode, "output": output}


def run_tests(*, workspace_dir: str) -> str:
    """Run the task's public test suite and return structured results."""
    return json.dumps(_run_suite(workspace_dir, "tests"), indent=2)


def run_hidden_tests(workspace_dir: str, hidden_tests_dir: Path) -> dict[str, object]:
    workspace = Path(workspace_dir).resolve()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(workspace / "src")
    result = subprocess.run(
        [
            os.environ.get("PYTHON", "python"),
            "-m",
            "unittest",
            "discover",
            "-s",
            str(hidden_tests_dir.resolve()),
            "-v",
        ],
        cwd=workspace,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    return {
        "passed": result.returncode == 0,
        "returncode": result.returncode,
        "output": (result.stdout + result.stderr)[-12_000:],
    }
