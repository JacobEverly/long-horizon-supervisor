from __future__ import annotations

import json
from pathlib import Path

from horizon_supervisor.supervisor_data.openthoughts_checkpoints import (
    _checkpoint,
    extract_action_json,
)

ROOT = Path(__file__).resolve().parents[1]


def test_action_parser_accepts_direct_and_embedded_json() -> None:
    direct, direct_mode = extract_action_json(
        '{"commands": [{"keystrokes": "ls\\n"}], "task_complete": false}'
    )
    embedded, embedded_mode = extract_action_json(
        'I checked the state.\n{"commands": [], "task_complete": true}'
    )
    assert direct_mode == "direct"
    assert direct and direct["task_complete"] is False
    assert embedded_mode == "embedded"
    assert embedded and embedded["task_complete"] is True
    assert extract_action_json("not json") == (None, "failed")


def test_openthoughts_checkpoint_drops_reference_command_text() -> None:
    source = {
        "source_id": "openthoughts-agent-v1-sft",
        "dataset_id": "open-thoughts/OpenThoughts-Agent-v1-SFT",
        "revision": "c5dc896981f4e3b7c5382669b1d1be0bc4b6a1a6",
    }
    task = {
        "task_id": "task-id",
        "leakage_group": "leakage-group",
        "record_split": "train",
    }
    row = {
        "run_id": "run-1",
        "model": "teacher",
        "agent": "terminus-2",
    }
    history = [
        {
            "role": "user",
            "content": (
                "wrapper\nTask Description:\nFix it.\n\nCurrent terminal state:\n"
                "Current Terminal Screen:\n$ "
            ),
        }
    ]
    answer = {
        "analysis": "reference reasoning",
        "plan": "reference plan",
        "commands": [{"keystrokes": "secret-reference-command\n"}],
        "task_complete": False,
    }
    checkpoint = _checkpoint(
        row, source, task, history, answer, 0, 4, "direct", "trajectory-1"
    )
    serialized = json.dumps(checkpoint)
    assert checkpoint["target"]["command_count"] == 1
    assert "secret-reference-command" not in serialized
    assert "reference reasoning" not in serialized
    assert checkpoint["audit_only"]["total_source_agent_turns"] == 4


def test_generated_openthoughts_dataset_matches_audited_counts() -> None:
    summary = json.loads(
        (ROOT / "data/supervisor/openthoughts-checkpoints-v0-summary.json").read_text()
    )
    assert summary["raw_trajectory_count"] == 15_209
    assert summary["trajectory_count"] == 15_201
    assert summary["task_count"] == 15_201
    assert summary["raw_assistant_turn_count"] == 115_800
    assert summary["checkpoint_count"] == 114_803
    assert summary["unique_checkpoint_id_count"] == 114_803
    assert summary["dropped"] == {
        "trajectory_task_parse_failures": 8,
        "assistant_action_parse_failures": 934,
    }
