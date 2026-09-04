from __future__ import annotations

import json
import time
import uuid
from typing import Any

import verifiers as vf
from verifiers.clients.client import Client

from horizon_supervisor.benchmark.tasks import TASK_BY_ID


class DeterministicRepairClient(Client[None, Any, Any, Any]):
    """Zero-cost client that exercises the real tool loop with known repairs."""

    def __init__(self) -> None:
        super().__init__(None)

    def setup_client(self, config: vf.ClientConfig) -> None:
        del config
        return None

    async def to_native_tool(self, tool: vf.Tool) -> Any:
        return tool

    async def to_native_prompt(self, messages: vf.Messages) -> tuple[Any, dict]:
        return messages, {}

    async def get_native_response(
        self, prompt: Any, model: str, sampling_args: dict, tools=None, **kwargs
    ):
        del prompt, model, sampling_args, tools, kwargs
        raise NotImplementedError

    async def raise_from_native_response(self, response: Any) -> None:
        del response

    async def from_native_response(self, response: Any) -> vf.Response:
        return response

    async def close(self) -> None:
        return None

    async def get_response(
        self,
        prompt: vf.Messages,
        model: str,
        sampling_args: dict,
        tools: list[vf.Tool] | None = None,
        **kwargs,
    ) -> vf.Response:
        del prompt, sampling_args, tools
        state = kwargs["state"]
        turn = len(state["trajectory"])
        task = TASK_BY_ID[state["info"]["task_id"]]
        if turn == 0:
            tool_calls = [vf.ToolCall(id=str(uuid.uuid4()), name="list_files", arguments="{}")]
            content = "I will inspect the repository first."
            reason = "tool_calls"
        elif turn == 1:
            tool_calls = [
                vf.ToolCall(
                    id=str(uuid.uuid4()),
                    name="read_file",
                    arguments=json.dumps({"path": task.editable_file}),
                )
            ]
            content = "I found the relevant source and will inspect it."
            reason = "tool_calls"
        elif turn == 2:
            tool_calls = [
                vf.ToolCall(
                    id=str(uuid.uuid4()),
                    name="write_file",
                    arguments=json.dumps(
                        {"path": task.editable_file, "content": task.gold_content}
                    ),
                )
            ]
            content = "I will apply the repair."
            reason = "tool_calls"
        elif turn == 3:
            tool_calls = [vf.ToolCall(id=str(uuid.uuid4()), name="run_tests", arguments="{}")]
            content = "I will validate the implementation."
            reason = "tool_calls"
        else:
            tool_calls = None
            content = "The repair is implemented and the public tests pass."
            reason = "stop"
        completion_tokens = max(1, len(content) // 4)
        return vf.Response(
            id=str(uuid.uuid4()),
            created=int(time.time()),
            model=model,
            usage=vf.Usage(
                prompt_tokens=100,
                reasoning_tokens=0,
                completion_tokens=completion_tokens,
                total_tokens=100 + completion_tokens,
            ),
            message=vf.ResponseMessage(
                role="assistant",
                content=content,
                tool_calls=tool_calls,
                finish_reason=reason,
                is_truncated=False,
            ),
        )
