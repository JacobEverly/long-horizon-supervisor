from horizon_supervisor.training import run_continuation_calibration_v7 as module


def test_bounded_run_command_caps_child_timeout(monkeypatch):
    seen = {}

    def fake_run(command, *, environment, timeout):
        seen.update(command=command, environment=environment, timeout=timeout)
        return 0, "ok", False

    monkeypatch.setattr(module, "_ORIGINAL_RUN_COMMAND", fake_run)

    result = module._bounded_run_command(
        ["harbor", "run"], environment={"MODE": "test"}, timeout=999
    )

    assert result == (0, "ok", False)
    assert seen == {
        "command": ["harbor", "run"],
        "environment": {"MODE": "test"},
        "timeout": 420.0,
    }


def test_v7_initial_state_is_empty_and_versioned():
    state = module._new_state("manifest", "smoke", 5.50, 7.55)

    assert state["schema_version"] == "continuation-calibration-execution-state.v7"
    assert state["manifest_sha256"] == "manifest"
    assert state["transport_smoke_sha256"] == "smoke"
    assert state["provider_usage_baseline_usd"] == 5.50
    assert state["provider_hard_limit_usd"] == 7.55
    assert state["status"] == "in_progress"
    assert state["pending_trial"] is None
    assert state["completed_schedule_items"] == []
    assert state["outcomes"] == []
    assert state["attempts"] == []
    assert state["fidelity_rows"] == []
    assert state["tranche_reports"] == []
