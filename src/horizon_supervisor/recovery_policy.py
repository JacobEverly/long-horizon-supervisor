from __future__ import annotations

from horizon_supervisor.models import (
    CheckpointKind,
    RecoveryAction,
    RecoveryDecision,
    RecoveryObservation,
)


class EvidenceAwareRecoveryPolicy:
    """Choose a state action before model routing.

    This is a deterministic safety baseline. Gate 6 supports its ordering, but
    does not establish calibrated probabilities or optimal thresholds.
    """

    def decide(self, observation: RecoveryObservation) -> RecoveryDecision:
        verified = any(
            item.kind == "public_tests"
            and item.passed
            and item.fresh_in_current_run
            and item.workspace_digest == observation.current_workspace_digest
            for item in observation.evidence
        )

        if observation.completion_requested and verified and not observation.final_verifier_failed:
            return RecoveryDecision(
                action=RecoveryAction.STOP_SUCCESS,
                current_completion_verified=True,
                reason="fresh passing evidence is bound to the current workspace digest",
            )

        if observation.final_verifier_failed and observation.current_workspace_dirty:
            verified_checkpoints = [
                checkpoint
                for checkpoint in observation.checkpoints
                if checkpoint.kind == CheckpointKind.VERIFIED
                and checkpoint.public_test_verified
                and checkpoint.workspace_digest != observation.current_workspace_digest
            ]
            if verified_checkpoints:
                target = max(verified_checkpoints, key=lambda checkpoint: checkpoint.sequence)
                return RecoveryDecision(
                    action=RecoveryAction.ROLLBACK,
                    target_checkpoint_id=target.checkpoint_id,
                    current_completion_verified=False,
                    reason=(
                        "final verification failed in a dirty workspace; prefer the latest "
                        "distinct verified checkpoint before changing models"
                    ),
                )

            clean_checkpoints = [
                checkpoint
                for checkpoint in observation.checkpoints
                if checkpoint.kind == CheckpointKind.CLEAN_BASE
            ]
            if clean_checkpoints:
                target = max(clean_checkpoints, key=lambda checkpoint: checkpoint.sequence)
                return RecoveryDecision(
                    action=RecoveryAction.RESTART_CLEAN,
                    target_checkpoint_id=target.checkpoint_id,
                    current_completion_verified=False,
                    reason=(
                        "final verification failed in inherited dirty state and no verified "
                        "rollback point exists; restart from the clean base"
                    ),
                )

        if observation.completion_requested and not verified:
            reason = (
                "completion evidence is missing, stale, or not fresh for the current digest; "
                "continue and verify before stopping"
            )
        elif observation.final_verifier_failed:
            reason = (
                "final verification failed but no safer state transition is available; "
                "continue to model routing from the current clean state"
            )
        else:
            reason = "no state intervention is required before model routing"
        return RecoveryDecision(
            action=RecoveryAction.CONTINUE,
            current_completion_verified=verified,
            reason=reason,
        )
