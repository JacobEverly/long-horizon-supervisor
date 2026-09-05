from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from harbor.agents.terminus_2.terminus_2 import Command, Terminus2

from horizon_supervisor.benchmark.pilot_harbor import (
    PilotTerminus2,
    _public_test_observation,
)
from horizon_supervisor.stuck_detector import SuspectedStuckV0
from horizon_supervisor.stuck_detector_v2 import (
    TwoTierDetectorConfig,
    TwoTierObservation,
    TwoTierStatus,
    TwoTierStuckDetectorV2,
)


def checkpoint_kind(
    *,
    status: TwoTierStatus,
    turn: int,
    healthy_turn: int,
    captured: set[str],
) -> str | None:
    """Return the first eligible checkpoint kind without using future outcomes."""
    if status == TwoTierStatus.STRUCTURAL_FAILURE:
        return None
    if (
        status == TwoTierStatus.HEALTHY
        and turn == healthy_turn
        and "healthy" not in captured
    ):
        return "healthy"
    if status == TwoTierStatus.NEEDS_REVIEW and "needs_review" not in captured:
        return "needs_review"
    if status == TwoTierStatus.CONFIRMED_STUCK and "confirmed_stuck" not in captured:
        return "confirmed_stuck"
    return None


class ContinuationTerminus2(PilotTerminus2):
    """Terminus 2 that records both detector tiers but never intervenes or stops."""

    def __init__(
        self,
        *args: Any,
        continuation_detector_config: str | dict[str, Any],
        continuation_task_category: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        config = (
            TwoTierDetectorConfig.model_validate_json(continuation_detector_config)
            if isinstance(continuation_detector_config, str)
            else TwoTierDetectorConfig.model_validate(continuation_detector_config)
        )
        self._pilot_detector = TwoTierStuckDetectorV2(config)
        self._continuation_task_category = continuation_task_category
        self._continuation_captured: set[str] = set()

    async def _execute_commands(
        self, commands: list[Command], session: Any
    ) -> tuple[bool, str]:
        # Bypass PilotTerminus2's v0 detector. The underlying command execution is
        # identical, while this class owns the separately versioned v2 observations.
        timeout_occurred, terminal_output = await Terminus2._execute_commands(
            self, commands, session
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
        routing_stats = self._routing_stats() or {}
        model_stats = (routing_stats.get("models") or {}).get(
            self._pilot_base_model_id, {}
        )
        chat = getattr(self, "_chat", None)
        provider_usage = self._provider_usage()
        provider_spend = 0.0
        if self._pilot_provider_usage_start is not None and provider_usage is not None:
            provider_spend = max(
                0.0, provider_usage - self._pilot_provider_usage_start
            )
        chat_spend = max(0.0, float(getattr(chat, "total_cost", 0.0)))
        state_reproducible = not bool(probe["unmanaged_relevant_processes"])
        observation = TwoTierObservation(
            run_id=self._pilot_run_id,
            turn=self._n_episodes,
            max_turns=self._max_episodes,
            model_id=self._pilot_base_model_id,
            commands=command_text,
            terminal_tail=terminal_output[-12_000:],
            workspace_digest=probe["workspace_digest"],
            public_tests=self._pilot_last_public_tests,
            successful_milestones=tuple(sorted(self._pilot_successful_milestones)),
            required_artifacts=(),
            protocol_failure=self._pilot_protocol_failure,
            provider_failure=False,
            harness_failure=False,
            actionable_next_step=self._pilot_actionable_next_step,
            input_tokens=int(
                model_stats.get(
                    "prompt_tokens", getattr(chat, "total_input_tokens", 0)
                )
            ),
            output_tokens=int(
                model_stats.get(
                    "completion_tokens", getattr(chat, "total_output_tokens", 0)
                )
            ),
            cached_tokens=int(
                model_stats.get(
                    "cached_tokens", getattr(chat, "total_cache_tokens", 0)
                )
            ),
            reasoning_tokens=int(model_stats.get("reasoning_tokens", 0)),
            output_token_budget=self._pilot_output_token_budget,
            spent_usd=max(provider_spend, chat_spend),
            spend_budget_usd=self._pilot_spend_budget_usd,
            task_category=self._continuation_task_category,
            snapshot_reproducible=state_reproducible,
            external_state_reproducible=state_reproducible,
        )
        assessment = self._pilot_detector.observe(observation)
        event = {
            "schema_version": "two-tier-observation-event.v0",
            "created_at": datetime.now(UTC).isoformat(),
            "kind": "detector_observation",
            "observation": observation.model_dump(mode="json"),
            "assessment": assessment.model_dump(mode="json"),
        }
        self._pilot_record_path.parent.mkdir(parents=True, exist_ok=True)
        with self._pilot_record_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")

        kind = checkpoint_kind(
            status=assessment.status,
            turn=observation.turn,
            healthy_turn=self._pilot_healthy_turn,
            captured=self._continuation_captured,
        )
        if kind is not None:
            await self._capture_checkpoint(
                kind=kind,
                observation=observation,
                assessment=assessment,
                probe=probe,
            )
            self._continuation_captured.add(kind)

        # Phase A is natural continuation. A review or confirmation is recorded,
        # never acted on, and never used to terminate the scout.
        return timeout_occurred, terminal_output


def record_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
