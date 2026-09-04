import hashlib
import json
from pathlib import Path

from horizon_supervisor.training import amend_stuck_confirmatory_efficiency as module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_efficiency_amendment_preserves_sealed_evidence(monkeypatch, tmp_path) -> None:
    root = tmp_path
    output = root / "artifacts/official/stuck-confirmatory-v1"
    output.mkdir(parents=True)
    source_manifest = output / "frozen-manifest-v1.json"
    amended_manifest = output / "frozen-manifest-v2.json"
    state_path = output / "execution-state-v0.json"
    outcomes_path = output / "matched-outcomes-v0.jsonl"

    amendment_script = (
        "src/horizon_supervisor/training/amend_stuck_confirmatory_efficiency.py"
    )
    for relative in (*module.CODE_PATHS, amendment_script):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")

    source = {
        "schema_version": "stuck-confirmatory-manifest.v1",
        "integrity": {"code_sha256": {}},
    }
    source_manifest.write_text(json.dumps(source), encoding="utf-8")
    source_hash = _sha256(source_manifest)
    source_manifest.with_suffix(".sha256").write_text(
        f"{source_hash}  {source_manifest.name}\n", encoding="utf-8"
    )
    outcomes = [
        {"group_id": "group-1", "branch_action": f"action-{index}"}
        for index in range(4)
    ]
    outcomes_path.write_text(
        "".join(json.dumps(row) + "\n" for row in outcomes), encoding="utf-8"
    )
    state = {
        "manifest_sha256": source_hash,
        "status": "interrupted",
        "stop_reason": "operator_interrupt",
        "execution_error": None,
        "pending_group": None,
        "outcomes": outcomes,
        "completed_schedule_items": ["sealed-item"],
        "amendment_history": [{"id": "prior-amendment"}],
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")

    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module, "OUTPUT_ROOT", output)
    monkeypatch.setattr(module, "SOURCE_MANIFEST", source_manifest)
    monkeypatch.setattr(module, "AMENDED_MANIFEST", amended_manifest)
    monkeypatch.setattr(module, "STATE_PATH", state_path)
    monkeypatch.setattr(module, "OUTCOMES_PATH", outcomes_path)

    result = module.amend()
    amended = json.loads(amended_manifest.read_text(encoding="utf-8"))
    resumed = json.loads(state_path.read_text(encoding="utf-8"))

    assert amended["schema_version"] == "stuck-confirmatory-manifest.v2"
    assert amended["execution_efficiency_amendment"]["scope"] == "orchestration_only"
    assert amended["execution_efficiency_amendment"][
        "accepted_outcomes_before_amendment"
    ] == 4
    assert resumed["outcomes"] == outcomes
    assert resumed["completed_schedule_items"] == ["sealed-item"]
    assert resumed["status"] == "in_progress"
    assert resumed["manifest_sha256"] == result["manifest_sha256"]
    assert resumed["amendment_history"][-1]["id"] == module.AMENDMENT_ID
