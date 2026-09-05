import json
from pathlib import Path

from horizon_supervisor.training import freeze_continuation_calibration as module


def _catalog() -> dict:
    return {
        "schema_version": "continuation-model-catalog.v0",
        "captured_at": "2026-09-05T00:00:00+00:00",
        "source": module.MODEL_CATALOG_URL,
        "models": [
            {
                "model_id": model_id,
                "canonical_slug": model_id,
                "created": 1,
                "context_length": 1_000_000,
                "max_completion_tokens": 131_072,
                "pricing": {"prompt": "0.1", "completion": "1.0"},
                "supported_parameters": ["tools"],
            }
            for model_id in module.EXACT_MODELS.values()
        ],
    }


def test_wave_four_selection_is_fresh_balanced_and_statically_safe(
    tmp_path: Path,
) -> None:
    tasks, excluded = module.select_task_pool(official_root=tmp_path)

    assert len(tasks) == 16
    assert [task["position"] for task in tasks] == list(range(1, 17))
    assert {task["tranche"] for task in tasks[:8]} == {1}
    assert {task["tranche"] for task in tasks[8:]} == {2}
    assert len({task["task_category"] for task in tasks[:8]}) == 8
    assert all(task["prior_official_reference_count"] == 0 for task in tasks)
    assert {row["task_id"] for row in excluded} == set(module.STATIC_EXCLUSIONS)
    assert not set(module.STATIC_EXCLUSIONS) & {task["task_id"] for task in tasks}


def test_freeze_writes_outcome_blind_contract(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(module, "fetch_model_catalog", _catalog)
    select_task_pool = module.select_task_pool
    monkeypatch.setattr(
        module,
        "select_task_pool",
        lambda: select_task_pool(official_root=tmp_path / "empty-official"),
    )

    result = module.freeze(tmp_path)
    manifest_path = Path(result["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["frozen_before_model_outcomes"] is True
    assert manifest["execution"]["natural_continuation_only"] is True
    assert manifest["execution"]["interventions_forbidden"] is True
    assert manifest["task_selection"]["maximum_tasks"] == 16
    assert manifest["sampling"]["tranches"][0]["task_positions"] == list(
        range(1, 9)
    )
    assert manifest["budget"]["phase_a_incremental_ceiling_usd"] == 5.0
    assert manifest["analysis"]["gates"]["confirmed_checkpoints"] == 6
    sidecar = manifest_path.with_suffix(".sha256")
    assert sidecar.read_text(encoding="utf-8").split()[0] == result[
        "manifest_sha256"
    ]


def test_selection_rejects_a_prior_official_reference(tmp_path: Path) -> None:
    prior = tmp_path / "prior.json"
    prior.write_text(
        json.dumps({"task_id": "advanced-poker-hand-classifier"}),
        encoding="utf-8",
    )

    try:
        module.select_task_pool(official_root=tmp_path)
    except RuntimeError as error:
        assert "appears in prior official evidence" in str(error)
    else:  # pragma: no cover - assertion helper
        raise AssertionError("prior task reference was accepted")
