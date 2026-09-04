from __future__ import annotations

import hashlib
import json
import math
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import verifiers as vf
from datasets import Dataset

from horizon_supervisor.benchmark.tasks import (
    BENCHMARK_TASKS,
    TASK_BY_ID,
    hidden_tests_dir,
    starter_dir,
)
from horizon_supervisor.benchmark.tools import (
    list_files,
    read_file,
    replace_in_file,
    run_hidden_tests,
    run_tests,
    write_file,
)

SYSTEM_PROMPT = """You are repairing a small Python repository in a persistent workspace.
Use the provided tools to inspect and edit files. Run the public tests as often as useful.
The final verifier includes additional tests. Do not claim completion until the code is credible.
Keep changes scoped to the requested behavior."""


def workspace_digest(workspace: str | Path) -> str:
    root = Path(workspace).resolve()
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def render_state_bound_handoff(
    *,
    current_workspace_digest: str,
    evidence: list[dict[str, Any]],
    prior_attempt_summary: str,
) -> str:
    lines = [
        "",
        "",
        "State-bound handoff:",
        f"- Mounted workspace digest: `{current_workspace_digest}`.",
        f"- Prior attempt summary: {prior_attempt_summary}",
        "- Prior evidence:",
    ]
    for item in evidence:
        source_digest = str(item.get("workspace_digest") or "unknown")
        state_matches = source_digest == current_workspace_digest
        status = "STATE-MATCHED BUT NOT FRESH" if state_matches else "STALE FOR THIS STATE"
        lines.append(
            f"  - {item.get('kind', 'evidence')} from `{source_digest}`: {status}; "
            f"passed={bool(item.get('passed'))}."
        )
    if not evidence:
        lines.append("  - none")
    lines.extend(
        [
            "- Evidence from another digest cannot validate this workspace.",
            "- Even state-matched prior evidence is not fresh in this rollout.",
            "- Inspect the mounted files, make any required repair, and run public tests in "
            "this rollout after the final edit before finishing.",
        ]
    )
    return "\n".join(lines)


class LocalCodingEnv(vf.StatefulToolEnv):
    """Verifiers environment backed by a persistent, locally scoped workspace."""

    def __init__(
        self,
        artifacts_dir: Path,
        max_turns: int = 10,
        task_ids: list[str] | None = None,
        input_usd_per_token: float = 0.0,
        output_usd_per_token: float = 0.0,
        per_run_cap_usd: float = 5.0,
        max_completion_tokens_per_call: int = 2_048,
        workspace_seeds: dict[str, Path] | None = None,
        handoff_contexts: dict[str, str] | None = None,
        handoff_evidence: dict[str, list[dict[str, Any]]] | None = None,
        checkpoint_ids: dict[str, str] | None = None,
        max_tool_calls_per_turn: int = 12,
        max_total_tool_calls: int = 48,
        max_consecutive_tool_errors: int = 3,
        max_unverified_stops: int = 2,
    ) -> None:
        selected = set(task_ids or TASK_BY_ID)
        handoff_contexts = handoff_contexts or {}
        handoff_evidence = handoff_evidence or {}
        rows = [
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": task.prompt + handoff_contexts.get(task.task_id, ""),
                    }
                ],
                "answer": "",
                "info": {
                    "task_id": task.task_id,
                    "difficulty": task.difficulty,
                    "title": task.title,
                },
            }
            for task in BENCHMARK_TASKS
            if task.task_id in selected
        ]
        missing = selected - TASK_BY_ID.keys()
        if missing:
            raise ValueError(f"unknown tasks: {', '.join(sorted(missing))}")
        rubric = vf.Rubric(funcs=[self.final_verifier], weights=[1.0])
        super().__init__(
            eval_dataset=Dataset.from_list(rows),
            system_prompt=SYSTEM_PROMPT,
            rubric=rubric,
            max_turns=max_turns,
            timeout_seconds=300,
            max_workers=1,
        )
        self.artifacts_dir = Path(artifacts_dir)
        self.input_usd_per_token = input_usd_per_token
        self.output_usd_per_token = output_usd_per_token
        self.per_run_cap_usd = per_run_cap_usd
        self.max_completion_tokens_per_call = max_completion_tokens_per_call
        self.workspace_seeds = workspace_seeds or {}
        self.handoff_evidence = handoff_evidence
        self.checkpoint_ids = checkpoint_ids or {}
        self.max_tool_calls_per_turn = max_tool_calls_per_turn
        self.max_total_tool_calls = max_total_tool_calls
        self.max_consecutive_tool_errors = max_consecutive_tool_errors
        self.max_unverified_stops = max_unverified_stops
        self.set_max_total_completion_tokens(12_000)
        self.add_tool(list_files, args_to_skip=["workspace_dir"])
        self.add_tool(read_file, args_to_skip=["workspace_dir"])
        self.add_tool(write_file, args_to_skip=["workspace_dir"])
        self.add_tool(replace_in_file, args_to_skip=["workspace_dir"])
        self.add_tool(run_tests, args_to_skip=["workspace_dir"])

    async def setup_state(self, state: vf.State) -> vf.State:
        task_id = state["info"]["task_id"]
        run_id = str(state["trajectory_id"])
        workspace = self.artifacts_dir / "workspaces" / task_id / run_id
        workspace.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.workspace_seeds.get(task_id, starter_dir(task_id)), workspace)
        state["workspace_dir"] = str(workspace.resolve())
        state["task_id"] = task_id
        state["checkpoint_id"] = self.checkpoint_ids.get(task_id)
        state["verifier_result"] = None
        state["estimated_spend_usd"] = 0.0
        state["budget_halted"] = False
        state["guard_events"] = []
        state["handoff_recommended"] = False
        state["guard_stop_reason"] = None
        state["consecutive_tool_errors"] = 0
        state["consecutive_no_change_edits"] = 0
        state["unverified_stop_attempts"] = 0
        state["guard_total_tool_calls"] = 0
        state["last_public_test_result"] = None
        state["last_verified_workspace_digest"] = None
        state["workspace_digest"] = self._workspace_digest(workspace)
        state["evidence_ledger"] = [
            {
                **item,
                "valid_for_current_workspace": (
                    item.get("workspace_digest") == state["workspace_digest"]
                ),
                "fresh_in_current_run": False,
            }
            for item in self.handoff_evidence.get(task_id, [])
        ]
        state["turn_checkpoints"] = []
        state["normalized_events"] = [
            {
                "kind": "task_started",
                "task_id": task_id,
                "workspace": state["workspace_dir"],
                "workspace_digest": state["workspace_digest"],
            }
        ]
        for evidence in state["evidence_ledger"]:
            state["normalized_events"].append(
                {
                    "kind": "evidence_loaded",
                    "evidence_kind": evidence.get("kind"),
                    "workspace_digest": evidence.get("workspace_digest"),
                    "valid_for_current_workspace": evidence["valid_for_current_workspace"],
                    "fresh_in_current_run": False,
                }
            )
        self._save_turn_checkpoint(state, reason="workspace_initialized")
        return state

    @staticmethod
    def _workspace_digest(workspace: str | Path) -> str:
        return workspace_digest(workspace)

    def _save_turn_checkpoint(self, state: vf.State, *, reason: str) -> None:
        workspace = Path(state["workspace_dir"])
        digest = self._workspace_digest(workspace)
        history = state["turn_checkpoints"]
        sequence = len(history)
        turn = len(state.get("trajectory") or [])
        checkpoint_id = f"turn-{turn:03d}-{sequence:03d}"
        destination = (
            self.artifacts_dir
            / "turn-checkpoints"
            / state["task_id"]
            / str(state["trajectory_id"])
            / checkpoint_id
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            workspace,
            destination / "workspace",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        public = state.get("last_public_test_result")
        record = {
            "checkpoint_id": checkpoint_id,
            "turn": turn,
            "reason": reason,
            "workspace_digest": digest,
            "workspace": str((destination / "workspace").resolve()),
            "public_test_passed": bool(public and public.get("passed")),
            "public_test_valid_for_checkpoint": bool(
                public
                and public.get("passed")
                and public.get("workspace_digest") == digest
                and public.get("fresh_in_current_run")
            ),
            "created_at": datetime.now(UTC).isoformat(),
        }
        (destination / "checkpoint.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
        history.append(record)
        state["workspace_digest"] = digest
        state["normalized_events"].append(
            {
                "kind": "workspace_checkpointed",
                "checkpoint_id": checkpoint_id,
                "turn": turn,
                "reason": reason,
                "workspace_digest": digest,
                "public_test_valid_for_checkpoint": record[
                    "public_test_valid_for_checkpoint"
                ],
            }
        )

    def _record_guard(
        self,
        state: vf.State,
        reason: str,
        *,
        hard_stop: bool,
        details: dict[str, object] | None = None,
    ) -> None:
        event = {
            "kind": "supervisor_guard",
            "reason": reason,
            "hard_stop": hard_stop,
            "details": details or {},
        }
        state["guard_events"].append(event)
        state["normalized_events"].append(event)
        state["handoff_recommended"] = True
        if hard_stop:
            state["guard_stop_reason"] = reason
            state["final_env_response"] = [
                vf.UserMessage(
                    content=(
                        "The deterministic supervisor stopped this attempt after detecting "
                        f"{reason}. The workspace has been preserved for a controlled handoff."
                    )
                )
            ]

    async def get_prompt_messages(self, state: vf.State) -> vf.Messages:
        messages = await super().get_prompt_messages(state)
        serialized = json.dumps(
            [
                message.model_dump() if hasattr(message, "model_dump") else message
                for message in messages
            ],
            default=str,
        )
        # Dividing characters by three deliberately overestimates most English/code tokenization.
        forecast_input_tokens = math.ceil(len(serialized) / 3)
        forecast_call_cost = (
            forecast_input_tokens * self.input_usd_per_token
            + self.max_completion_tokens_per_call * self.output_usd_per_token
        )
        if state["estimated_spend_usd"] + forecast_call_cost > self.per_run_cap_usd:
            state["budget_halted"] = True
            state["normalized_events"].append(
                {
                    "kind": "budget_halted",
                    "spent_usd": state["estimated_spend_usd"],
                    "forecast_next_call_usd": forecast_call_cost,
                    "cap_usd": self.per_run_cap_usd,
                }
            )
            state["final_env_response"] = [
                vf.UserMessage(content="The per-run budget cap was reached; execution stopped.")
            ]
        return messages

    async def add_model_response(
        self,
        state: vf.State,
        prompt_messages: vf.Messages,
        response: vf.Response,
    ) -> None:
        await super().add_model_response(state, prompt_messages, response)
        usage = response.usage
        if usage is None:
            return
        call_cost = (
            usage.prompt_tokens * self.input_usd_per_token
            + usage.completion_tokens * self.output_usd_per_token
        )
        state["estimated_spend_usd"] += call_cost
        state["normalized_events"].append(
            {
                "kind": "model_response",
                "model": response.model,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "reasoning_tokens": usage.reasoning_tokens,
                "estimated_cost_usd": call_cost,
            }
        )

    @vf.stop
    async def no_tools_called(self, state: vf.State) -> bool:
        if len(state["trajectory"]) == 0:
            return False
        last_message = state["trajectory"][-1]["completion"][-1]
        if last_message.role != "assistant" or getattr(last_message, "tool_calls", None):
            return False
        current = self._workspace_digest(state["workspace_dir"])
        state["workspace_digest"] = current
        return state.get("last_verified_workspace_digest") == current

    @vf.stop
    async def max_turns_reached(self, state: vf.State) -> bool:
        reached = len(state["trajectory"]) >= self.max_turns and self.max_turns > 0
        if reached and state.get("guard_stop_reason") is None:
            self._record_guard(
                state,
                "turn_limit",
                hard_stop=False,
                details={"turns": len(state["trajectory"]), "limit": self.max_turns},
            )
        return reached

    @vf.stop
    async def max_total_completion_tokens_reached(self, state: vf.State) -> bool:
        if self.max_total_completion_tokens <= 0:
            return False
        usage = self.get_state_usage(state)
        reached = usage is not None and usage["output_tokens"] >= self.max_total_completion_tokens
        if reached and state.get("guard_stop_reason") is None:
            self._record_guard(
                state,
                "completion_token_limit",
                hard_stop=False,
                details={
                    "output_tokens": usage["output_tokens"] if usage else 0,
                    "limit": self.max_total_completion_tokens,
                },
            )
        return reached

    def mark_timed_out(self, state: vf.State) -> None:
        self._record_guard(state, "rollout_timeout", hard_stop=False)
        super().mark_timed_out(state)

    async def env_response(
        self, messages: vf.Messages, state: vf.State, **kwargs
    ) -> vf.Messages:
        del kwargs
        last_message = messages[-1]
        tool_calls = list(getattr(last_message, "tool_calls", None) or [])
        if not tool_calls:
            state["unverified_stop_attempts"] += 1
            attempts = state["unverified_stop_attempts"]
            self._record_guard(
                state,
                "unverified_stop",
                hard_stop=attempts >= self.max_unverified_stops,
                details={"attempt": attempts, "limit": self.max_unverified_stops},
            )
            self._save_turn_checkpoint(state, reason="unverified_completion_attempt")
            current = state["workspace_digest"]
            prior = state.get("last_public_test_result") or {}
            prior_digest = prior.get("workspace_digest") or "none"
            return [
                vf.UserMessage(
                    content=(
                        "Supervisor check: completion is not yet verified against the current "
                        f"workspace digest `{current}`. The last public-test evidence is bound to "
                        f"`{prior_digest}` and is not valid fresh evidence for this completion. "
                        "Make a concrete repair if needed, run the public tests in this rollout "
                        "after the final edit, and only then finish."
                    )
                )
            ]

        if len(tool_calls) > self.max_tool_calls_per_turn:
            self._record_guard(
                state,
                "tool_batch_limit",
                hard_stop=True,
                details={"requested": len(tool_calls), "limit": self.max_tool_calls_per_turn},
            )
            self._save_turn_checkpoint(state, reason="tool_batch_limit")
            return []
        if state["guard_total_tool_calls"] + len(tool_calls) > self.max_total_tool_calls:
            self._record_guard(
                state,
                "total_tool_call_limit",
                hard_stop=True,
                details={
                    "requested_total": state["guard_total_tool_calls"] + len(tool_calls),
                    "limit": self.max_total_tool_calls,
                },
            )
            self._save_turn_checkpoint(state, reason="total_tool_call_limit")
            return []

        tool_messages: vf.Messages = []
        mutation_tools = {"write_file", "replace_in_file"}
        for tool_call in tool_calls:
            tool_call_id = tool_call.id
            tool_name = tool_call.name
            state["guard_total_tool_calls"] += 1
            try:
                if tool_name not in self.tool_map:
                    raise ValueError(f"unknown or malformed tool name: {tool_name}")
                parsed_args = json.loads(tool_call.arguments)
                if not isinstance(parsed_args, dict):
                    raise ValueError("tool arguments must be a JSON object")
                tool_args = self.update_tool_args(tool_name, parsed_args, messages, state)
                before = self._workspace_digest(state["workspace_dir"])
                message = await super().call_tool(tool_name, tool_args, tool_call_id)
                after = self._workspace_digest(state["workspace_dir"])
                state["workspace_digest"] = after
                state["consecutive_tool_errors"] = 0

                if tool_name in mutation_tools:
                    if before == after:
                        state["consecutive_no_change_edits"] += 1
                    else:
                        state["consecutive_no_change_edits"] = 0
                        prior = state.get("last_public_test_result")
                        if prior and prior.get("workspace_digest") != after:
                            prior = {**prior, "valid_for_current_workspace": False}
                            state["last_public_test_result"] = prior
                            if state.get("last_verified_workspace_digest") != after:
                                state["last_verified_workspace_digest"] = None
                            state["evidence_ledger"] = [
                                {
                                    **evidence,
                                    "valid_for_current_workspace": (
                                        evidence.get("workspace_digest") == after
                                    ),
                                }
                                for evidence in state["evidence_ledger"]
                            ]
                            state["normalized_events"].append(
                                {
                                    "kind": "evidence_invalidated",
                                    "evidence_kind": "public_tests",
                                    "evidence_workspace_digest": prior.get(
                                        "workspace_digest"
                                    ),
                                    "current_workspace_digest": after,
                                    "reason": "workspace_changed",
                                }
                            )
                if tool_name == "run_tests":
                    result = json.loads(str(message.content))
                    result.update(
                        {
                            "kind": "public_tests",
                            "workspace_digest": after,
                            "observed_at": datetime.now(UTC).isoformat(),
                            "valid_for_current_workspace": True,
                            "fresh_in_current_run": True,
                        }
                    )
                    state["last_public_test_result"] = result
                    state["evidence_ledger"].append(dict(result))
                    if result.get("passed"):
                        state["last_verified_workspace_digest"] = after
                    else:
                        state["last_verified_workspace_digest"] = None
                    message = vf.ToolMessage(
                        role="tool",
                        content=json.dumps(result, indent=2),
                        tool_call_id=tool_call_id,
                    )
                tool_messages.append(message)
                state["normalized_events"].append(
                    {
                        "kind": "tool_result",
                        "tool": tool_name,
                        "success": True,
                        "workspace_changed": before != after,
                    }
                )
            except Exception as error:
                state["consecutive_tool_errors"] += 1
                if tool_name in mutation_tools:
                    state["consecutive_no_change_edits"] += 1
                tool_messages.append(
                    vf.ToolMessage(
                        role="tool",
                        content=str(error),
                        tool_call_id=tool_call_id,
                    )
                )
                state["normalized_events"].append(
                    {
                        "kind": "tool_result",
                        "tool": tool_name,
                        "success": False,
                        "error_type": type(error).__name__,
                    }
                )

            if state["consecutive_tool_errors"] >= self.max_consecutive_tool_errors:
                self._record_guard(
                    state,
                    "repeated_tool_errors",
                    hard_stop=True,
                    details={"count": state["consecutive_tool_errors"]},
                )
                break
            if state["consecutive_no_change_edits"] >= self.max_consecutive_tool_errors:
                self._record_guard(
                    state,
                    "repeated_no_change_edits",
                    hard_stop=True,
                    details={"count": state["consecutive_no_change_edits"]},
                )
                break
        self._save_turn_checkpoint(state, reason="tool_batch_completed")
        return tool_messages

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict,
        messages: vf.Messages,
        state: vf.State,
        **kwargs,
    ) -> dict:
        del messages, kwargs
        tool_args["workspace_dir"] = state["workspace_dir"]
        logged_args = tool_args | {"workspace_dir": "<scoped>"}
        if "content" in logged_args:
            logged_args["content"] = f"<{len(str(logged_args['content']))} characters>"
        state["normalized_events"].append(
            {"kind": "tool_requested", "tool": tool_name, "arguments": logged_args}
        )
        return tool_args

    async def call_tool(
        self, tool_name: str, tool_args: dict, tool_call_id: str, **kwargs
    ) -> vf.ToolMessage:
        message = await super().call_tool(tool_name, tool_args, tool_call_id, **kwargs)
        return message

    async def final_verifier(self, state: vf.State) -> float:
        result = run_hidden_tests(
            state["workspace_dir"], hidden_tests_dir(state["info"]["task_id"])
        )
        state["verifier_result"] = result
        state["normalized_events"].append(
            {"kind": "validation_result", "passed": bool(result["passed"]), "suite": "hidden"}
        )
        return 1.0 if result["passed"] else 0.0
