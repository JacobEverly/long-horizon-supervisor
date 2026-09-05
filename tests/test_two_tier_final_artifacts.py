import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
OUTPUT = ROOT / "artifacts/official/two-tier-detector-v2"
DECISION = "NOT READY — two-tier detector gate failed"


def _json(name: str) -> dict:
    return json.loads((OUTPUT / name).read_text(encoding="utf-8"))


def _sha256(relative_path: str) -> str:
    return hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()


def test_final_manifest_preserves_the_frozen_result_and_sources() -> None:
    manifest = _json("frozen-manifest-v0.json")

    assert manifest["decision"] == DECISION
    assert manifest["freeze_provenance"]["frozen_before_real_data_scoring"]

    for version, expected_path in {
        "v0": "src/horizon_supervisor/stuck_detector.py",
        "v1": "src/horizon_supervisor/stuck_detector_v1.py",
        "v2": "src/horizon_supervisor/stuck_detector_v2.py",
    }.items():
        detector = manifest["detectors"][version]
        assert detector["path"] == expected_path
        assert detector["sha256"] == _sha256(expected_path)

    contract = manifest["freeze_provenance"]
    assert contract["contract_sha256"] == _sha256(contract["contract_path"])
    report = manifest["development_report"]
    assert report["sha256"] == _sha256(report["path"])
    assert report["gate_passed"] is False


def test_failed_gate_stopped_every_paid_or_training_phase() -> None:
    report = _json("development-report-v0.json")
    eligibility = _json("task-eligibility-report-v0.json")
    fidelity = _json("snapshot-fidelity-v0.json")
    ledger = _json("execution-ledger-v0.json")

    assert report["decision"] == DECISION
    assert report["gate_passed"] is False
    assert eligibility["status"] == "not_run"
    assert eligibility["tasks_inspected"] == 0
    assert eligibility["paid_calls_made"] is False
    assert fidelity["status"] == "not_run"
    assert fidelity["accepted_snapshots"] == 0

    execution = ledger["execution"]
    assert execution["model_scouts_run"] == 0
    assert execution["intervention_branches_run"] == 0
    assert execution["models_called"] == []
    assert execution["policy_training_runs"] == 0
    assert ledger["openrouter"]["exact_incremental_project_spend_usd"] == 0.0
    assert ledger["daytona"]["environments_created"] == 0
    assert ledger["cleanup"]["new_daytona_environments_remaining"] == 0


def test_checkpoint_bank_is_empty_and_report_leads_with_decision() -> None:
    checkpoint_rows = [
        line
        for line in (OUTPUT / "checkpoint-bank-index-v0.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    report_lines = (
        ROOT / "docs/two-tier-detector-v2-final.md"
    ).read_text(encoding="utf-8").splitlines()

    assert checkpoint_rows == []
    assert report_lines[0] == DECISION
