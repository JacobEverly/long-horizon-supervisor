from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from horizon_supervisor.training.freeze_continuation_calibration import (
    ROOT,
    _sha256,
)
from horizon_supervisor.training.freeze_continuation_calibration_v5 import (
    OUTPUT_ROOT as V5_ROOT,
)

V5_MANIFEST = V5_ROOT / "frozen-manifest-v5.json"
V5_SMOKE = V5_ROOT / "permission-transport-smoke-v5.json"
V5_FAILURE = V5_ROOT / "smoke-failure-summary-v5.json"
EXPECTED_V5_MANIFEST_SHA256 = (
    "3c3e5b1b2b06461a0ac9a617208e55eeaff837e1ae87cbbd9ddf7f4895c8cea8"
)
EXPECTED_V5_SMOKE_SHA256 = (
    "cb5529e5339a6c255d39abc76a6874fe1e33ecd8da56ec86f0003f02fb2185aa"
)
EXPECTED_V5_FAILURE_SHA256 = (
    "721adce5854e4334923b1209984fe7f451508e6a6321888d9c6222cbd45b8e70"
)
OUTPUT_ROOT = ROOT / "artifacts/official/two-tier-continuation-calibration-v6"


def _v5_inputs() -> dict[str, Any]:
    expected = {
        V5_MANIFEST: EXPECTED_V5_MANIFEST_SHA256,
        V5_SMOKE: EXPECTED_V5_SMOKE_SHA256,
        V5_FAILURE: EXPECTED_V5_FAILURE_SHA256,
    }
    for path, digest in expected.items():
        if not path.is_file() or _sha256(path) != digest:
            raise RuntimeError(f"immutable v5 input changed: {path}")
    manifest = json.loads(V5_MANIFEST.read_text(encoding="utf-8"))
    smoke = json.loads(V5_SMOKE.read_text(encoding="utf-8"))
    failure = json.loads(V5_FAILURE.read_text(encoding="utf-8"))
    if (
        smoke["passed"] is not False
        or smoke["provider_model_calls"] != 0
        or smoke["remote_to_local_digest_match"] is not True
        or smoke["local_to_remote_digest_match"] is not True
        or smoke["read_only_git_object_mode_preserved"] is not True
        or smoke["remaining_daytona_environments"] != 1
        or failure["model_outcomes_observed"] is not False
        or failure["v5_may_be_overwritten_or_reinterpreted"] is not False
    ):
        raise RuntimeError("v5 terminal preflight failure changed")
    return manifest


def freeze(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    output_root = output_root.resolve()
    manifest_path = output_root / "frozen-manifest-v6.json"
    if manifest_path.exists():
        raise FileExistsError(f"continuation calibration v6 is frozen: {manifest_path}")
    v5 = _v5_inputs()
    manifest = deepcopy(v5)
    manifest["schema_version"] = "two-tier-continuation-calibration-manifest.v6"
    manifest["frozen_at"] = datetime.now(UTC).isoformat()
    manifest["lineage"] |= {
        "v5_manifest_sha256": EXPECTED_V5_MANIFEST_SHA256,
        "v5_failed_smoke_sha256": EXPECTED_V5_SMOKE_SHA256,
        "v5_failure_summary_sha256": EXPECTED_V5_FAILURE_SHA256,
        "v5_provider_model_calls": 0,
        "v5_model_outcomes_observed": 0,
        "v5_failed_only_cleanup_finalization": True,
        "minimal_revision": (
            "wait up to 24 seconds for asynchronous Daytona deletion to settle "
            "before finalizing the otherwise unchanged zero-model transport smoke"
        ),
        "task_selection_changed": False,
        "detector_thresholds_changed": False,
        "analysis_gate_thresholds_changed": False,
        "models_changed": False,
        "max_turns_changed": False,
        "token_limits_changed": False,
        "natural_continuation_protocol_changed": False,
    }
    manifest["execution"] |= {
        "transport_smoke_path": (
            "artifacts/official/two-tier-continuation-calibration-v6/"
            "permission-transport-smoke-v6.json"
        ),
        "transport_smoke_schema": "permission-transport-smoke.v6",
        "transport_smoke_cleanup_wait_seconds": 24,
    }
    manifest["analysis"] |= {
        "cohort": "fresh_v6_only",
        "prior_outcomes_used_for_fit_or_tuning": False,
    }
    code_paths = list(v5["integrity"]["code_sha256"])
    code_paths.extend(
        [
            "src/horizon_supervisor/training/freeze_continuation_calibration_v6.py",
            "src/horizon_supervisor/training/run_continuation_calibration_v6.py",
            "src/horizon_supervisor/training/run_permission_transport_smoke_v6.py",
        ]
    )
    if len(code_paths) != len(set(code_paths)):
        raise RuntimeError("v6 integrity code path list contains duplicates")
    for relative in code_paths:
        if not (ROOT / relative).is_file():
            raise FileNotFoundError(ROOT / relative)
    manifest["integrity"]["code_sha256"] = {
        relative: _sha256(ROOT / relative) for relative in code_paths
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    digest = _sha256(manifest_path)
    manifest_path.with_suffix(".sha256").write_text(
        f"{digest}  {manifest_path.name}\n", encoding="utf-8"
    )
    return {"manifest_path": str(manifest_path), "manifest_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    print(json.dumps(freeze(args.output_root), indent=2))


if __name__ == "__main__":
    main()
