from pathlib import Path

import pytest

from horizon_supervisor.benchmark.tools import read_file, replace_in_file, write_file


def test_workspace_tools_persist_and_reject_escape(tmp_path: Path) -> None:
    write_file("src/example.py", "value = 1\n", workspace_dir=str(tmp_path))
    replace_in_file("1", "2", "src/example.py", workspace_dir=str(tmp_path))
    assert read_file("src/example.py", workspace_dir=str(tmp_path)) == "value = 2\n"

    with pytest.raises(ValueError, match="escapes"):
        read_file("../secret", workspace_dir=str(tmp_path))
