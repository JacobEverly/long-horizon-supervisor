from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import pytest


def test_gate7_switchyard_config_loads_and_exposes_expected_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    switchyard = pytest.importorskip("switchyard.cli.launchers.native_server")
    config = Path("benchmarks/switchyard-gate7.toml").resolve()
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-config-validation")

    server = switchyard.NativeServer(config)
    try:
        with urllib.request.urlopen(f"{server.base_url}/health", timeout=2) as response:
            assert response.status == 200
        with urllib.request.urlopen(f"{server.base_url}/v1/models", timeout=2) as response:
            payload = json.load(response)
    finally:
        server.close()

    route_ids = {row["id"] for row in payload["data"]}
    assert route_ids == {
        "gate7/fixed-flash",
        "gate7/fixed-glm",
        "gate7/fixed-kimi",
        "gate7/fixed-pro",
        "gate7/fixed-qwen",
        "gate7/stage-cost",
        "gate7/stage-quality",
    }
    assert os.environ["OPENROUTER_API_KEY"] == "offline-config-validation"
