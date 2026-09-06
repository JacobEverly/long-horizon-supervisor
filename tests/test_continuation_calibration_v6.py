
from pathlib import Path

from horizon_supervisor.training import (
    run_permission_transport_smoke_v6 as smoke,
)
from horizon_supervisor.training.freeze_continuation_calibration_v6 import (
    EXPECTED_V5_FAILURE_SHA256,
    EXPECTED_V5_MANIFEST_SHA256,
    EXPECTED_V5_SMOKE_SHA256,
    _v5_inputs,
    freeze,
)
from horizon_supervisor.training.run_continuation_calibration_v6 import (
    validate_manifest,
)


def test_v6_accepts_only_the_sealed_zero_model_v5_failure() -> None:
    manifest = _v5_inputs()
    lineage = manifest["lineage"]

    assert manifest["schema_version"].endswith(".v5")
    assert lineage["detector_thresholds_changed"] is False
    assert EXPECTED_V5_MANIFEST_SHA256
    assert EXPECTED_V5_SMOKE_SHA256
    assert EXPECTED_V5_FAILURE_SHA256


def test_v6_freeze_is_reloadable_without_changing_experiment(tmp_path: Path) -> None:
    frozen = freeze(tmp_path)
    manifest, digest = validate_manifest(Path(frozen["manifest_path"]))

    assert digest == frozen["manifest_sha256"]
    assert manifest["lineage"]["task_selection_changed"] is False
    assert manifest["lineage"]["models_changed"] is False
    assert manifest["lineage"]["detector_thresholds_changed"] is False
    assert manifest["execution"]["transport_smoke_cleanup_wait_seconds"] == 24


def test_v6_cleanup_treats_settled_conflict_as_transient(monkeypatch) -> None:
    monkeypatch.setattr(
        smoke,
        "_cleanup_new_sandboxes",
        lambda _: {"removed": [], "errors": ["deletion still in progress"]},
    )
    monkeypatch.setattr(smoke, "_wait_for_cleanup", lambda _: ([], []))

    cleanup, remaining = smoke._settled_cleanup(set())

    assert cleanup == {
        "transient_errors": ["deletion still in progress"],
        "final_errors": [],
    }
    assert remaining == []


def test_v6_cleanup_keeps_error_when_sandbox_remains(monkeypatch) -> None:
    monkeypatch.setattr(
        smoke,
        "_cleanup_new_sandboxes",
        lambda _: {"removed": [], "errors": ["deletion still in progress"]},
    )
    monkeypatch.setattr(smoke, "_wait_for_cleanup", lambda _: (["sandbox"], []))

    cleanup, remaining = smoke._settled_cleanup(set())

    assert cleanup["final_errors"] == ["deletion still in progress"]
    assert remaining == ["sandbox"]
