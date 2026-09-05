from horizon_supervisor.benchmark.continuation_harbor_v1 import (
    process_identity,
    update_unmanaged_processes,
)


def _process(pid: int, name: str, cwd: str = "/app") -> dict:
    return {"pid": pid, "ppid": 1, "name": name, "cwd": cwd}


def test_resident_daytona_processes_are_reproducible_harness_baseline() -> None:
    baseline = [
        _process(1, "daytona"),
        _process(97, "python3"),
        _process(103, "tail"),
    ]

    rows, active, new = update_unmanaged_processes(
        before=baseline,
        after=baseline,
        previously_unmanaged=set(),
    )

    assert rows == []
    assert active == set()
    assert new == []


def test_new_task_process_is_carried_until_it_exits() -> None:
    baseline = [_process(1, "daytona"), _process(97, "python3")]
    server = _process(240, "node")

    first_rows, active, first_new = update_unmanaged_processes(
        before=baseline,
        after=[*baseline, server],
        previously_unmanaged=set(),
    )
    carried_rows, carried, carried_new = update_unmanaged_processes(
        before=[*baseline, server],
        after=[*baseline, server],
        previously_unmanaged=active,
    )
    final_rows, final_active, _ = update_unmanaged_processes(
        before=[*baseline, server],
        after=baseline,
        previously_unmanaged=carried,
    )

    assert first_rows == [server]
    assert first_new == [server]
    assert carried_rows == [server]
    assert carried_new == []
    assert final_rows == []
    assert final_active == set()


def test_terminal_shell_turnover_is_not_task_external_state() -> None:
    baseline = [_process(1, "daytona")]
    after = [*baseline, _process(148, "bash"), _process(151, "cat")]

    rows, active, new = update_unmanaged_processes(
        before=baseline,
        after=after,
        previously_unmanaged=set(),
    )

    assert rows == []
    assert active == set()
    assert new == []
    assert process_identity(after[-1]) == (151, "cat", "/app")
