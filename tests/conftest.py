from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# These suites validate the private, immutable experiment corpus rather than the
# reusable package. The large/raw inputs are deliberately absent from the public
# repository. A local research checkout still runs every suite automatically.
EXPERIMENT_DATA_SUITES = {
    "test_checkpoint_continuation_risk.py",
    "test_chooser_data.py",
    "test_combined_supervisor_data.py",
    "test_completion_transfer.py",
    "test_freeze_stuck_confirmatory.py",
    "test_freeze_stuck_pilot.py",
    "test_freeze_continuation_calibration.py",
    "test_gate8.py",
    "test_matched_outcomes.py",
    "test_openthoughts_data.py",
    "test_sentinel_screen.py",
    "test_supervisor_data.py",
    "test_swe_transfer_data.py",
    "test_terminal_bench_pro_materialization.py",
    "test_terminal_bench_pro_panel.py",
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    has_private_corpus = all(
        path.is_file()
        for path in (
            ROOT / "data/supervisor/terminal-bench-pro-panel-v0.jsonl",
            ROOT
            / "artifacts/official/gate8-proportional-30-task-checkpoint/"
            "matched-outcomes-140-v1.jsonl",
        )
    )
    if has_private_corpus:
        return

    unavailable = pytest.mark.skip(
        reason="requires the non-public experiment corpus and paid-run artifacts"
    )
    for item in items:
        filename = Path(str(item.fspath)).name
        if filename in EXPERIMENT_DATA_SUITES:
            item.add_marker(unavailable)
        elif (
            filename == "test_run_stuck_pilot.py"
            and item.name == "test_task_parent_is_the_exact_wave_tasks_directory"
        ):
            item.add_marker(unavailable)
        elif (
            filename == "test_supervisor_policy_dataset.py"
            and item.name
            == "test_authoritative_development_build_is_rectangular_and_leakage_safe"
        ):
            item.add_marker(unavailable)
        elif (
            filename == "test_run_continuation_calibration.py"
            and item.name == "test_frozen_manifest_reloads_with_all_hashes"
        ):
            item.add_marker(unavailable)
