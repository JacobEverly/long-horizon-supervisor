from horizon_supervisor.benchmark.continuation_harbor_v2 import (
    PROCESS_NAMES_WITHOUT_TASK_STATE_V2,
    update_unmanaged_processes_v2,
)


def _process(pid: int, name: str) -> dict:
    return {"pid": pid, "name": name, "cwd": "/app"}


def test_tail_created_between_probes_is_harness_state() -> None:
    active, identities, new = update_unmanaged_processes_v2(
        before=[],
        after=[_process(438, "tail")],
        previously_unmanaged=set(),
    )

    assert "tail" in PROCESS_NAMES_WITHOUT_TASK_STATE_V2
    assert active == []
    assert identities == set()
    assert new == []


def test_task_worker_created_between_probes_remains_unmanaged() -> None:
    worker = _process(501, "python3")
    active, identities, new = update_unmanaged_processes_v2(
        before=[],
        after=[worker],
        previously_unmanaged=set(),
    )

    assert active == [worker]
    assert identities == {(501, "python3", "/app")}
    assert new == [worker]
