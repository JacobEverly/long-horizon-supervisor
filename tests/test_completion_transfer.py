from __future__ import annotations

import json
from pathlib import Path

from horizon_supervisor.training.evaluate_completion_transfer import (
    _load_examples,
)

ROOT = Path(__file__).resolve().parents[1]


def test_transfer_loader_uses_no_post_generation_confidence_as_input() -> None:
    texts, numeric, labels, pass_rates, task_ids = _load_examples(
        ROOT / "data/supervisor/swe-pivot-transfer-checkpoints-v0.jsonl"
    )
    assert len(texts) == len(numeric) == len(labels) == len(pass_rates) == len(task_ids)
    assert len(labels) == 50_308
    assert labels.sum() == 1_042
    assert numeric.shape[1] == 12


def test_transfer_report_honors_frozen_contract() -> None:
    report = json.loads(
        (ROOT / "artifacts/training/swe-transfer-evaluation-v0.json").read_text()
    )
    contract = json.loads(
        (ROOT / "benchmarks/swe-transfer-acceptance-v0.json").read_text()
    )
    assert report["model"]["sha256"] == contract["model"]["sha256"]
    assert report["data"]["examples"] == 50_308
    assert report["model"]["training_on_transfer_source"] is False
