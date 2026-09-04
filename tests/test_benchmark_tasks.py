import shutil
from pathlib import Path

from horizon_supervisor.benchmark.tasks import BENCHMARK_TASKS, hidden_tests_dir, starter_dir
from horizon_supervisor.benchmark.tools import run_hidden_tests, write_file


def test_starters_fail_and_gold_repairs_pass(tmp_path: Path) -> None:
    for task in BENCHMARK_TASKS:
        workspace = tmp_path / task.task_id
        shutil.copytree(starter_dir(task.task_id), workspace)
        assert not run_hidden_tests(str(workspace), hidden_tests_dir(task.task_id))["passed"]
        write_file(task.editable_file, task.gold_content, workspace_dir=str(workspace))
        result = run_hidden_tests(str(workspace), hidden_tests_dir(task.task_id))
        assert result["passed"], result["output"]
