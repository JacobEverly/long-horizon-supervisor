from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from harbor.agents.terminus_2.terminus_2 import Command, Terminus2
from harbor.environments.base import BaseEnvironment
from harbor.environments.daytona.environment import DaytonaEnvironment
from harbor.llms.base import LLMResponse

from horizon_supervisor.benchmark.gate7 import query_openrouter_key
from horizon_supervisor.stuck_detector import (
    PublicTestObservation,
    StuckStatus,
    SuspectedStuckV0,
    TurnObservation,
)

_TEST_COMMAND = re.compile(
    r"(?:^|\s)(?:pytest|python\s+-m\s+pytest|npm\s+test|cargo\s+test|"
    r"go\s+test|ctest|(?:\./|/)[\w./-]*test(?:s)?\.sh)(?:\s|$)",
    re.IGNORECASE,
)
_PASSED = re.compile(r"(?i)\b(\d+)\s+passed\b")
_FAILED = re.compile(r"(?i)\b(\d+)\s+failed\b")
_SAFE_PROCESS_NAMES = {
    "bash",
    "sh",
    "dash",
    "tmux: server",
    "ps",
    "sleep",
    "timeout",
    "tini",
}
_WORKSPACE_DIGEST_COMMAND = (
    "python3 - <<'PY'\n"
    "import hashlib,json,os,stat\n"
    "from pathlib import Path\n"
    "root=Path('.').resolve(); rows=[]\n"
    "for p in sorted([root,*root.rglob('*')],"
    "key=lambda x:x.relative_to(root).as_posix()):\n"
    " r='.' if p==root else p.relative_to(root).as_posix(); s=p.lstat(); "
    "m=stat.S_IMODE(s.st_mode)\n"
    " if p.is_symlink(): rows.append([r,'l',m,os.readlink(p)])\n"
    " elif p.is_dir(): rows.append([r,'d',m])\n"
    " elif p.is_file(): rows.append([r,'f',m,"
    "hashlib.sha256(p.read_bytes()).hexdigest()])\n"
    "payload=json.dumps(rows,sort_keys=True,separators=(',',':'))\n"
    "print(hashlib.sha256(payload.encode()).hexdigest())\n"
    "PY"
)


def _last_int(pattern: re.Pattern[str], text: str) -> int | None:
    matches = pattern.findall(text)
    return int(matches[-1]) if matches else None


def _public_test_observation(
    commands: tuple[str, ...], terminal_output: str
) -> PublicTestObservation | None:
    joined = "\n".join(commands)
    if not _TEST_COMMAND.search(joined):
        return None
    passed = _last_int(_PASSED, terminal_output)
    failed = _last_int(_FAILED, terminal_output)
    if passed is None and failed is None:
        return None
    return PublicTestObservation(
        command_fingerprint=SuspectedStuckV0.command_fingerprint(commands),
        passed=passed or 0,
        failed=failed or 0,
        failure_fingerprints=SuspectedStuckV0.error_fingerprints(terminal_output),
    )


def _safe_handoff(record: dict[str, Any]) -> str:
    assessment = record["assessment"]
    observation = record["observation"]
    tests = observation.get("public_tests")
    if tests:
        test_line = f"{tests['passed']} passed and {tests['failed']} failed"
    else:
        test_line = "no standardized public-test count was observed"
    return (
        "\n\nMatched-state handoff (public evidence only):\n"
        f"- Checkpoint kind: {record['checkpoint_kind']}.\n"
        f"- Prior agent completed {observation['turn']} turns.\n"
        f"- Mounted workspace digest: {observation['workspace_digest']}.\n"
        f"- Public-test state: {test_line}.\n"
        f"- Detector signals: {', '.join(assessment['active_signals']) or 'none'}.\n"
        "- No hidden verifier result, sibling outcome, provider secret, or private "
        "reasoning is included. Inspect the mounted state and continue independently."
    )


class SeededDaytonaEnvironment(DaytonaEnvironment):
    """Harbor adapter that restores a frozen workspace into a fresh sandbox."""

    def __init__(
        self,
        *args: Any,
        workspace_seed_path: str,
        expected_workspace_digest: str,
        **kwargs: Any,
    ) -> None:
        self._workspace_seed_path = Path(workspace_seed_path).resolve()
        self._expected_workspace_digest = expected_workspace_digest
        if not self._workspace_seed_path.is_dir():
            raise FileNotFoundError(self._workspace_seed_path)
        super().__init__(*args, **kwargs)
        if self._compose_mode:
            raise ValueError("matched-state restoration supports direct Daytona tasks only")

    async def start(self, force_build: bool) -> None:
        await super().start(force_build)
        workdir_result = await self.exec(command="pwd", timeout_sec=30)
        workdir = (workdir_result.stdout or "").strip().splitlines()[-1]
        workdir_path = Path(workdir)
        if (
            workdir_result.return_code != 0
            or not workdir_path.is_absolute()
            or workdir_path == Path("/")
            or len(workdir_path.parts) < 2
        ):
            raise ValueError("seeded branch requires a narrow absolute task workdir")
        clear = await self.exec(
            command="find . -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +",
            cwd=workdir,
            user="root",
            timeout_sec=120,
        )
        if clear.return_code != 0:
            raise RuntimeError(f"failed to clear fresh branch workspace: {clear.stderr}")
        await self.upload_dir(self._workspace_seed_path, workdir)
        digest = await self.exec(
            command=_WORKSPACE_DIGEST_COMMAND,
            cwd=workdir,
            timeout_sec=120,
        )
        actual = (digest.stdout or "").strip().splitlines()[-1]
        if digest.return_code != 0 or actual != self._expected_workspace_digest:
            raise RuntimeError(
                "rehydrated workspace digest mismatch: "
                f"expected {self._expected_workspace_digest}, got {actual}"
            )


class PilotTerminus2(Terminus2):
    """Terminus 2 with an outcome-blind detector and portable checkpoints."""

    def __init__(
        self,
        *args: Any,
        pilot_record_path: str,
        pilot_run_id: str,
        pilot_base_model_id: str,
        pilot_capture_healthy: bool = True,
        pilot_capture_stuck: bool = True,
        pilot_stop_after_checkpoint: bool = False,
        pilot_healthy_turn: int = 4,
        pilot_output_token_budget: int = 49_152,
        pilot_spend_budget_usd: float = 0.5,
        pilot_stats_url: str | None = None,
        pilot_provider_usage_start: float | None = None,
        pilot_provider_usage_ceiling: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._pilot_record_path = Path(pilot_record_path)
        self._pilot_run_id = pilot_run_id
        self._pilot_base_model_id = pilot_base_model_id
        self._pilot_capture_healthy = pilot_capture_healthy
        self._pilot_capture_stuck = pilot_capture_stuck
        self._pilot_stop_after_checkpoint = pilot_stop_after_checkpoint
        self._pilot_healthy_turn = pilot_healthy_turn
        self._pilot_output_token_budget = pilot_output_token_budget
        self._pilot_spend_budget_usd = pilot_spend_budget_usd
        self._pilot_stats_url = pilot_stats_url
        self._pilot_provider_usage_start = pilot_provider_usage_start
        self._pilot_provider_usage_ceiling = pilot_provider_usage_ceiling
        self._pilot_detector = SuspectedStuckV0()
        self._pilot_environment: BaseEnvironment | None = None
        self._pilot_last_public_tests: PublicTestObservation | None = None
        self._pilot_protocol_failure = False
        self._pilot_actionable_next_step = True
        self._pilot_healthy_captured = False
        self._pilot_stuck_captured = False
        self._pilot_successful_milestones: set[str] = set()
        self._pilot_budget_halted = False
        self._pilot_agent_started_at: datetime | None = None

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: Any,
    ) -> None:
        self._pilot_agent_started_at = datetime.now(UTC)
        await super().run(instruction, environment, context)

    async def _query_llm(
        self,
        chat: Any,
        prompt: str,
        original_instruction: str = "",
        session: Any = None,
    ) -> LLMResponse:
        usage = self._provider_usage()
        reserve = 0.05
        branch_cap_reached = (
            self._pilot_provider_usage_start is not None
            and usage is not None
            and usage - self._pilot_provider_usage_start
            >= self._pilot_spend_budget_usd - reserve
        )
        global_cap_reached = (
            self._pilot_provider_usage_ceiling is not None
            and usage is not None
            and usage >= self._pilot_provider_usage_ceiling - reserve
        )
        if branch_cap_reached or global_cap_reached:
            self._pilot_budget_halted = True
            event = {
                "schema_version": "pilot-budget-halt.v0",
                "created_at": datetime.now(UTC).isoformat(),
                "kind": "branch_budget_halt",
                "provider_usage_start_usd": self._pilot_provider_usage_start,
                "provider_usage_current_usd": usage,
                "maximum_incremental_spend_usd": self._pilot_spend_budget_usd,
                "provider_usage_ceiling_usd": self._pilot_provider_usage_ceiling,
                "request_reserve_usd": reserve,
            }
            self._pilot_record_path.parent.mkdir(parents=True, exist_ok=True)
            with self._pilot_record_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
            if session is not None:
                await session.stop()
            return LLMResponse(
                content=json.dumps(
                    {
                        "analysis": "Execution stopped by the external budget guard.",
                        "plan": "",
                        "commands": [],
                        "task_complete": False,
                    }
                )
            )
        return await super()._query_llm(chat, prompt, original_instruction, session)

    def _routing_stats(self) -> dict[str, Any] | None:
        if not self._pilot_stats_url:
            return None
        try:
            with urllib.request.urlopen(self._pilot_stats_url, timeout=5) as response:
                return json.load(response)
        except Exception:
            return None

    @staticmethod
    def _provider_usage() -> float | None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return None
        try:
            return float(query_openrouter_key(api_key)["usage"])
        except Exception:
            return None

    async def setup(self, environment: BaseEnvironment) -> None:
        self._pilot_environment = environment
        await super().setup(environment)

    async def _handle_llm_interaction(
        self,
        chat: Any,
        prompt: str,
        original_instruction: str = "",
        session: Any = None,
    ) -> tuple[list[Command], bool, str, str, str, Any]:
        result = await super()._handle_llm_interaction(
            chat, prompt, original_instruction, session
        )
        commands, is_complete, feedback, _analysis, plan, _response = result
        self._pilot_protocol_failure = bool(feedback and "ERROR:" in feedback)
        self._pilot_actionable_next_step = bool(commands or is_complete or plan.strip())
        return result

    async def _workspace_probe(self) -> dict[str, Any]:
        if self._pilot_environment is None:
            raise RuntimeError("pilot environment is not initialized")
        workdir_result = await self._pilot_environment.exec(
            command="pwd", timeout_sec=30
        )
        workdir = (workdir_result.stdout or "").strip().splitlines()[-1]
        workdir_path = Path(workdir)
        if (
            workdir_result.return_code != 0
            or not workdir_path.is_absolute()
            or workdir_path == Path("/")
            or len(workdir_path.parts) < 2
        ):
            raise RuntimeError("could not resolve a narrow absolute task workdir")
        digest_result = await self._pilot_environment.exec(
            command=_WORKSPACE_DIGEST_COMMAND,
            cwd=workdir,
            timeout_sec=120,
        )
        if digest_result.return_code != 0:
            raise RuntimeError(f"workspace digest probe failed: {digest_result.stderr}")
        git_result = await self._pilot_environment.exec(
            command=(
                "if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then "
                "git rev-parse HEAD; git status --porcelain=v2 --untracked-files=all; "
                "else echo NO_GIT; fi"
            ),
            cwd=workdir,
            timeout_sec=60,
        )
        process_result = await self._pilot_environment.exec(
            command=(
                "for process in /proc/[0-9]*; do "
                "pid=${process##*/}; "
                "row=$(ps -p \"$pid\" -o pid=,ppid=,comm= 2>/dev/null) || continue; "
                "cwd=$(readlink \"$process/cwd\" 2>/dev/null || true); "
                "printf '%s\\t%s\\n' \"$row\" \"$cwd\"; "
                "done"
            ),
            timeout_sec=60,
        )
        process_rows = []
        for line in (process_result.stdout or "").splitlines():
            process, _, cwd = line.partition("\t")
            parts = process.strip().split(maxsplit=2)
            if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
                process_rows.append(
                    {
                        "pid": int(parts[0]),
                        "ppid": int(parts[1]),
                        "name": parts[2],
                        "cwd": cwd.strip(),
                    }
                )
        relevant = [
            row
            for row in process_rows
            if (
                row["cwd"] == workdir
                or row["cwd"].startswith(f"{workdir.rstrip('/')}/")
            )
            and row["name"].lower() not in _SAFE_PROCESS_NAMES
        ]
        return {
            "workspace_digest": (digest_result.stdout or "").strip().splitlines()[-1],
            "git_state": (git_result.stdout or "").strip(),
            "process_inventory": process_rows,
            "unmanaged_relevant_processes": relevant,
            "environment_metadata": {
                "workdir": workdir,
                "network_mode": str(
                    self._pilot_environment.task_env_config.network_mode
                ),
            },
        }

    async def _capture_checkpoint(
        self,
        *,
        kind: str,
        observation: TurnObservation,
        assessment: Any,
        probe: dict[str, Any],
    ) -> None:
        if self._pilot_environment is None:
            raise RuntimeError("pilot environment is not initialized")
        provider_usage = self._provider_usage()
        has_remaining_turns = observation.turn < observation.max_turns
        state_transfer_eligible = (
            not probe["unmanaged_relevant_processes"] and has_remaining_turns
        )
        record = {
            "schema_version": "matched-checkpoint.v0",
            "created_at": datetime.now(UTC).isoformat(),
            "checkpoint_kind": kind,
            "run_id": self._pilot_run_id,
            "base_model_id": self._pilot_base_model_id,
            "observation": observation.model_dump(mode="json"),
            "assessment": assessment.model_dump(mode="json"),
            "git_state": probe["git_state"],
            "process_inventory": probe["process_inventory"],
            "unmanaged_relevant_processes": probe["unmanaged_relevant_processes"],
            "environment_metadata": probe["environment_metadata"],
            "state_transfer_eligible": state_transfer_eligible,
            "state_transfer_ineligibility_reason": (
                None
                if state_transfer_eligible
                else (
                    "unmanaged process state has no frozen rehydration recipe"
                    if probe["unmanaged_relevant_processes"]
                    else "checkpoint has no remaining agent turns"
                )
            ),
            "contains_hidden_verifier_artifacts": False,
            "contains_private_reasoning": False,
            "contains_provider_secrets": False,
            "routing_stats": self._routing_stats(),
            "provider_usage_usd": provider_usage,
        }
        record["anchor_workspace_path"] = None
        if record["state_transfer_eligible"]:
            anchor_path = (
                self._pilot_record_path.parent
                / "anchors"
                / self._pilot_run_id
                / f"{kind}-turn-{observation.turn:02d}"
                / "workspace"
            )
            workdir = probe["environment_metadata"]["workdir"]
            await self._pilot_environment.download_dir(workdir, anchor_path)
            provider_secret_values = [
                value.encode()
                for name, value in os.environ.items()
                if value
                and any(
                    marker in name.upper()
                    for marker in ("API_KEY", "TOKEN", "PASSWORD", "PRIVATE_KEY")
                )
            ]
            secret_found = any(
                secret in path.read_bytes()
                for path in anchor_path.rglob("*")
                if path.is_file() and path.stat().st_size <= 10_000_000
                for secret in provider_secret_values
            )
            if secret_found:
                shutil.rmtree(anchor_path)
                record["state_transfer_eligible"] = False
                record["state_transfer_ineligibility_reason"] = (
                    "provider secret value appeared in workspace archive"
                )
            else:
                record["anchor_workspace_path"] = str(anchor_path)
                record["archive_transfer"] = {
                    "source_workdir": workdir,
                    "preserves_permissions": True,
                    "preserves_git_directory": True,
                    "process_memory_preserved": False,
                }
        record["handoff"] = _safe_handoff(record)
        branch_started_at = datetime.now(UTC)
        record["branch_started_at"] = branch_started_at.isoformat()
        record["agent_elapsed_seconds"] = (
            (branch_started_at - self._pilot_agent_started_at).total_seconds()
            if self._pilot_agent_started_at is not None
            else 0.0
        )
        if self._pilot_provider_usage_start is None and provider_usage is not None:
            self._pilot_provider_usage_start = provider_usage
        self._pilot_record_path.parent.mkdir(parents=True, exist_ok=True)
        with self._pilot_record_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    async def _execute_commands(
        self, commands: list[Command], session: Any
    ) -> tuple[bool, str]:
        timeout_occurred, terminal_output = await super()._execute_commands(
            commands, session
        )
        command_text = tuple(command.keystrokes for command in commands)
        observed_tests = _public_test_observation(command_text, terminal_output)
        if observed_tests is not None:
            self._pilot_last_public_tests = observed_tests
        if SuspectedStuckV0.looks_like_successful_milestone(terminal_output):
            fingerprint = hashlib.sha256(
                terminal_output[-2_000:].encode(errors="replace")
            ).hexdigest()[:16]
            self._pilot_successful_milestones.add(fingerprint)
        probe = await self._workspace_probe()
        chat = getattr(self, "_chat", None)
        routing_stats = self._routing_stats() or {}
        model_stats = (routing_stats.get("models") or {}).get(
            self._pilot_base_model_id, {}
        )
        observation = TurnObservation(
            run_id=self._pilot_run_id,
            turn=self._n_episodes,
            max_turns=self._max_episodes,
            model_id=self._pilot_base_model_id,
            commands=command_text,
            terminal_tail=terminal_output[-12_000:],
            workspace_digest=probe["workspace_digest"],
            public_tests=self._pilot_last_public_tests,
            successful_milestones=tuple(sorted(self._pilot_successful_milestones)),
            protocol_failure=self._pilot_protocol_failure,
            actionable_next_step=self._pilot_actionable_next_step,
            input_tokens=int(
                model_stats.get("prompt_tokens", getattr(chat, "total_input_tokens", 0))
            ),
            output_tokens=int(
                model_stats.get(
                    "completion_tokens", getattr(chat, "total_output_tokens", 0)
                )
            ),
            cached_tokens=int(
                model_stats.get("cached_tokens", getattr(chat, "total_cache_tokens", 0))
            ),
            reasoning_tokens=int(model_stats.get("reasoning_tokens", 0)),
            output_token_budget=self._pilot_output_token_budget,
            spent_usd=max(0.0, float(getattr(chat, "total_cost", 0.0))),
            spend_budget_usd=self._pilot_spend_budget_usd,
        )
        assessment = self._pilot_detector.observe(observation)
        event = {
            "schema_version": "stuck-observation-event.v0",
            "created_at": datetime.now(UTC).isoformat(),
            "kind": "detector_observation",
            "observation": observation.model_dump(mode="json"),
            "assessment": assessment.model_dump(mode="json"),
        }
        self._pilot_record_path.parent.mkdir(parents=True, exist_ok=True)
        with self._pilot_record_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")

        should_capture_healthy = (
            self._pilot_capture_healthy
            and not self._pilot_healthy_captured
            and observation.turn == self._pilot_healthy_turn
            and assessment.status == StuckStatus.HEALTHY
        )
        should_capture_stuck = (
            self._pilot_capture_stuck
            and not self._pilot_stuck_captured
            and assessment.status == StuckStatus.SUSPECTED_STUCK
        )
        checkpoint_captured = False
        if should_capture_healthy:
            await self._capture_checkpoint(
                kind="healthy",
                observation=observation,
                assessment=assessment,
                probe=probe,
            )
            self._pilot_healthy_captured = True
            checkpoint_captured = True
        if should_capture_stuck:
            await self._capture_checkpoint(
                kind="suspected_stuck",
                observation=observation,
                assessment=assessment,
                probe=probe,
            )
            self._pilot_stuck_captured = True
            checkpoint_captured = True
        if (
            checkpoint_captured
            and self._pilot_stop_after_checkpoint
            and self._pilot_environment is not None
        ):
            # The checkpoint archive and public handoff are already sealed.  Ending
            # the scout's tmux session prevents unused post-checkpoint model calls;
            # the verifier may still run, but its result is not used for checkpoint
            # eligibility or any matched branch outcome.
            await self._pilot_environment.exec(
                command=f"tmux kill-session -t {shlex.quote(self.name())}",
                timeout_sec=10,
            )
        return timeout_occurred, terminal_output
