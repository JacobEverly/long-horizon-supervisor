from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from horizon_supervisor.benchmark.model_catalog import ModelSpec


def _value(obj: Any, field: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(field, default)
    return getattr(obj, field, default)


def _dump(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, Mapping):
        return {key: _dump(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_dump(value) for value in obj]
    return obj


def _text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(_dump(content), ensure_ascii=False)


def _cost(usage: Any, model: ModelSpec) -> float:
    if usage is None:
        return 0.0
    return (
        int(_value(usage, "prompt_tokens", 0)) * model.input_usd_per_token
        + int(_value(usage, "completion_tokens", 0)) * model.output_usd_per_token
    )


def export_atif(output: Mapping[str, Any], model: ModelSpec) -> dict[str, Any]:
    trajectory = output.get("trajectory") or []
    initial_prompt = output.get("prompt") or []
    steps: list[dict[str, Any]] = []

    for message in initial_prompt:
        role = _value(message, "role")
        if role not in {"system", "user"}:
            continue
        steps.append(
            {
                "step_id": len(steps) + 1,
                "source": role,
                "message": _text(_value(message, "content")),
                "extra": {},
            }
        )

    total_prompt = 0
    total_completion = 0
    total_cost = 0.0
    for index, turn in enumerate(trajectory):
        response = _value(turn, "response")
        assistant = _value(response, "message")
        usage = _value(response, "usage")
        prompt_tokens = int(_value(usage, "prompt_tokens", 0)) if usage else 0
        completion_tokens = int(_value(usage, "completion_tokens", 0)) if usage else 0
        reasoning_tokens = int(_value(usage, "reasoning_tokens", 0)) if usage else 0
        call_cost = _cost(usage, model)
        total_prompt += prompt_tokens
        total_completion += completion_tokens
        total_cost += call_cost

        tool_calls = []
        for call in _value(assistant, "tool_calls", []) or []:
            raw_arguments = _value(call, "arguments", "{}")
            try:
                arguments = json.loads(raw_arguments)
            except (TypeError, json.JSONDecodeError):
                arguments = {"_raw": str(raw_arguments)}
            tool_calls.append(
                {
                    "tool_call_id": _value(call, "id"),
                    "function_name": _value(call, "name"),
                    "arguments": arguments,
                    "extra": {},
                }
            )

        observations = []
        if index + 1 < len(trajectory):
            next_prompt = _value(trajectory[index + 1], "prompt", []) or []
            current_prompt = _value(turn, "prompt", []) or []
            completion = _value(turn, "completion", []) or []
            appended = next_prompt[len(current_prompt) + len(completion) :]
            for message in appended:
                if _value(message, "role") == "tool":
                    observations.append(
                        {
                            "source_call_id": _value(message, "tool_call_id"),
                            "content": _text(_value(message, "content")),
                            "extra": {},
                        }
                    )

        created = int(_value(response, "created", 0) or 0)
        step: dict[str, Any] = {
            "step_id": len(steps) + 1,
            "timestamp": datetime.fromtimestamp(created, UTC).isoformat() if created else None,
            "source": "agent",
            "model_name": model.model_id,
            "message": _text(_value(assistant, "content")),
            "llm_call_count": 1,
            "metrics": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost_usd": call_cost,
                "extra": {"reasoning_tokens": reasoning_tokens},
            },
            "extra": {"finish_reason": _value(assistant, "finish_reason")},
        }
        reasoning = _value(assistant, "reasoning_content")
        if reasoning:
            step["reasoning_content"] = reasoning
        if tool_calls:
            step["tool_calls"] = tool_calls
        if observations:
            step["observation"] = {"results": observations}
        steps.append({key: value for key, value in step.items() if value is not None})

    tool_definitions = [
        {
            "type": "function",
            "function": {
                "name": _value(tool, "name"),
                "description": _value(tool, "description"),
                "parameters": _dump(_value(tool, "parameters", {})),
            },
        }
        for tool in output.get("tool_defs") or []
    ]
    return {
        "schema_version": "ATIF-v1.7",
        "session_id": str(output.get("workspace_dir", "")).rstrip("/").split("/")[-1],
        "trajectory_id": str(output.get("workspace_dir", "")).rstrip("/").split("/")[-1],
        "agent": {
            "name": "long-horizon-supervisor-verifiers-adapter",
            "version": "0.1.0",
            "model_name": model.model_id,
            "tool_definitions": tool_definitions,
            "extra": {"harness": "verifiers-0.3.0"},
        },
        "steps": steps,
        "notes": "One ATIF agent step is emitted per Verifiers model response.",
        "final_metrics": {
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_cached_tokens": 0,
            "total_cost_usd": total_cost,
            "total_steps": len(steps),
            "extra": {},
        },
        "extra": {
            "task_id": output.get("task_id"),
            "difficulty": (output.get("info") or {}).get("difficulty"),
            "verifier_reward": output.get("reward"),
        },
    }
