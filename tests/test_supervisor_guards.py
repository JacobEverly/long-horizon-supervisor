import asyncio
import json
from pathlib import Path

import verifiers as vf

from horizon_supervisor.benchmark.environment import (
    LocalCodingEnv,
    render_state_bound_handoff,
)
from horizon_supervisor.benchmark.tasks import TASK_BY_ID
from horizon_supervisor.benchmark.tools import write_file


def _state() -> dict:
    return {"info": {"task_id": "retry-policy"}, "trajectory_id": "guard-test"}


def test_unverified_stop_is_blocked_and_repeated_attempt_requests_handoff(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        env = LocalCodingEnv(tmp_path, task_ids=["retry-policy"])
        state = _state()
        await env.setup_state(state)  # type: ignore[arg-type]
        message = vf.AssistantMessage(content="done", tool_calls=[])

        await env.env_response([message], state)  # type: ignore[arg-type]
        assert state["guard_stop_reason"] is None
        await env.env_response([message], state)  # type: ignore[arg-type]
        assert state["guard_stop_reason"] == "unverified_stop"
        assert state["handoff_recommended"] is True

    asyncio.run(exercise())


def test_repeated_failed_edits_trip_no_progress_guard(tmp_path: Path) -> None:
    async def exercise() -> None:
        env = LocalCodingEnv(tmp_path, task_ids=["retry-policy"])
        state = _state()
        await env.setup_state(state)  # type: ignore[arg-type]
        calls = [
            vf.ToolCall(
                id=f"call-{index}",
                name="replace_in_file",
                arguments=(
                    '{"path":"src/retry_policy.py","old":"not present",'
                    '"new":"replacement"}'
                ),
            )
            for index in range(3)
        ]
        await env.env_response(  # type: ignore[arg-type]
            [vf.AssistantMessage(content=None, tool_calls=calls)], state
        )
        assert state["guard_stop_reason"] == "repeated_tool_errors"

    asyncio.run(exercise())


def test_oversized_tool_batch_is_blocked_before_execution(tmp_path: Path) -> None:
    async def exercise() -> None:
        env = LocalCodingEnv(
            tmp_path,
            task_ids=["retry-policy"],
            max_tool_calls_per_turn=2,
        )
        state = _state()
        await env.setup_state(state)  # type: ignore[arg-type]
        calls = [
            vf.ToolCall(id=f"call-{index}", name="list_files", arguments="{}")
            for index in range(3)
        ]
        result = await env.env_response(  # type: ignore[arg-type]
            [vf.AssistantMessage(content=None, tool_calls=calls)], state
        )
        assert result == []
        assert state["guard_stop_reason"] == "tool_batch_limit"
        assert state["guard_total_tool_calls"] == 0

    asyncio.run(exercise())


def test_verified_current_workspace_may_stop(tmp_path: Path) -> None:
    async def exercise() -> None:
        env = LocalCodingEnv(tmp_path, task_ids=["retry-policy"])
        state = _state()
        await env.setup_state(state)  # type: ignore[arg-type]
        task = TASK_BY_ID["retry-policy"]
        write_file(task.editable_file, task.gold_content, workspace_dir=state["workspace_dir"])
        run_call = vf.ToolCall(id="tests", name="run_tests", arguments="{}")
        await env.env_response(  # type: ignore[arg-type]
            [vf.AssistantMessage(content=None, tool_calls=[run_call])], state
        )
        final_message = vf.AssistantMessage(content="done", tool_calls=[])
        state["trajectory"] = [{"completion": [final_message]}]
        assert await env.no_tools_called(state) is True  # type: ignore[arg-type]

    asyncio.run(exercise())


def test_public_test_evidence_is_bound_to_digest_and_invalidated(tmp_path: Path) -> None:
    async def exercise() -> None:
        env = LocalCodingEnv(tmp_path, task_ids=["retry-policy"])
        state = _state()
        await env.setup_state(state)  # type: ignore[arg-type]
        task = TASK_BY_ID["retry-policy"]
        write_file(task.editable_file, task.gold_content, workspace_dir=state["workspace_dir"])
        run_call = vf.ToolCall(id="tests", name="run_tests", arguments="{}")
        responses = await env.env_response(  # type: ignore[arg-type]
            [vf.AssistantMessage(content=None, tool_calls=[run_call])], state
        )
        verified_digest = state["workspace_digest"]
        assert state["last_public_test_result"]["workspace_digest"] == verified_digest
        assert state["last_public_test_result"]["valid_for_current_workspace"] is True
        assert state["last_public_test_result"]["fresh_in_current_run"] is True
        assert verified_digest in str(responses[0].content)

        edit_call = vf.ToolCall(
            id="edit",
            name="write_file",
            arguments=json.dumps(
                {
                    "path": "src/retry_policy.py",
                    "content": task.gold_content + "\n# changed\n",
                }
            ),
        )
        await env.env_response(  # type: ignore[arg-type]
            [vf.AssistantMessage(content=None, tool_calls=[edit_call])], state
        )
        assert state["workspace_digest"] != verified_digest
        assert state["last_public_test_result"]["valid_for_current_workspace"] is False
        assert state["last_verified_workspace_digest"] is None
        assert any(
            event["kind"] == "evidence_invalidated"
            for event in state["normalized_events"]
        )

    asyncio.run(exercise())


def test_turn_checkpoints_preserve_workspace_and_evidence_metadata(tmp_path: Path) -> None:
    async def exercise() -> None:
        env = LocalCodingEnv(tmp_path, task_ids=["retry-policy"])
        state = _state()
        await env.setup_state(state)  # type: ignore[arg-type]
        assert len(state["turn_checkpoints"]) == 1
        first = state["turn_checkpoints"][0]
        assert Path(first["workspace"]).is_dir()

        call = vf.ToolCall(id="files", name="list_files", arguments="{}")
        await env.env_response(  # type: ignore[arg-type]
            [vf.AssistantMessage(content=None, tool_calls=[call])], state
        )
        assert len(state["turn_checkpoints"]) == 2
        latest = state["turn_checkpoints"][-1]
        metadata = Path(latest["workspace"]).parent / "checkpoint.json"
        assert metadata.is_file()
        assert latest["workspace_digest"] == state["workspace_digest"]

    asyncio.run(exercise())


def test_state_bound_handoff_labels_matching_and_stale_evidence() -> None:
    rendered = render_state_bound_handoff(
        current_workspace_digest="current",
        evidence=[
            {"kind": "public_tests", "workspace_digest": "current", "passed": True},
            {"kind": "public_tests", "workspace_digest": "old", "passed": True},
        ],
        prior_attempt_summary="The hidden verifier did not pass.",
    )

    assert "STATE-MATCHED BUT NOT FRESH" in rendered
    assert "STALE FOR THIS STATE" in rendered
    assert "run public tests in this rollout" in rendered
