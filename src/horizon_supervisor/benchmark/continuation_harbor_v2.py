from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from harbor.agents.terminus_2.terminus_2 import Command, Terminus2

from horizon_supervisor.benchmark.continuation_harbor import checkpoint_kind
from horizon_supervisor.benchmark.continuation_harbor_v1 import (
    PROCESS_NAMES_WITHOUT_TASK_STATE,
    ProcessDeltaContinuationTerminus2,
    ProcessIdentity,
    process_identity,
)
from horizon_supervisor.benchmark.pilot_harbor import _public_test_observation
from horizon_supervisor.stuck_detector import SuspectedStuckV0
from horizon_supervisor.stuck_detector_v2 import TwoTierObservation

PROCESS_NAMES_WITHOUT_TASK_STATE_V2 = PROCESS_NAMES_WITHOUT_TASK_STATE | {"tail"}


def update_unmanaged_processes_v2(
    *,
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    previously_unmanaged: set[ProcessIdentity],
) -> tuple[list[dict[str, Any]], set[ProcessIdentity], list[dict[str, Any]]]:
    before_ids = {process_identity(row) for row in before}
    current = {process_identity(row): row for row in after}
    carried = previously_unmanaged.intersection(current)
    newly_created = {
        identity
        for identity, row in current.items()
        if identity not in before_ids
        and str(row["name"]).lower() not in PROCESS_NAMES_WITHOUT_TASK_STATE_V2
    }
    active = carried | newly_created
    active_rows = [current[identity] for identity in sorted(active)]
    new_rows = [current[identity] for identity in sorted(newly_created)]
    return active_rows, active, new_rows


class HarnessFilteredContinuationTerminus2(ProcessDeltaContinuationTerminus2):
    """Process-delta continuation with frozen terminal-service exclusions."""

    async def _execute_commands(
        self, commands: list[Command], session: Any
    ) -> tuple[bool, str]:
        before_probe = await self._workspace_probe()
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
        unmanaged, active_identities, newly_unmanaged = update_unmanaged_processes_v2(
            before=before_probe["process_inventory"],
            after=probe["process_inventory"],
            previously_unmanaged=self._continuation_unmanaged_identities,
        )
        self._continuation_unmanaged_identities = active_identities
        probe["unmanaged_relevant_processes"] = unmanaged

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
        state_reproducible = not unmanaged
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
            "process_reproducibility": {
                "schema_version": "action-process-delta.v2",
                "before_process_count": len(before_probe["process_inventory"]),
                "after_process_count": len(probe["process_inventory"]),
                "new_unmanaged_processes": newly_unmanaged,
                "active_unmanaged_processes": unmanaged,
                "baseline_processes_are_harness_state": True,
                "ignored_harness_process_names": sorted(
                    PROCESS_NAMES_WITHOUT_TASK_STATE_V2
                ),
            },
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

        return timeout_occurred, terminal_output
