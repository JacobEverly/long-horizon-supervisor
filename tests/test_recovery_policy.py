from horizon_supervisor import EvidenceAwareRecoveryPolicy
from horizon_supervisor.models import (
    CheckpointKind,
    EvidenceRecord,
    RecoveryAction,
    RecoveryCheckpoint,
    RecoveryObservation,
)


def observation(**updates: object) -> RecoveryObservation:
    values = {
        "current_workspace_digest": "current",
        "current_workspace_dirty": False,
    }
    values.update(updates)
    return RecoveryObservation(**values)


def test_fresh_current_evidence_allows_successful_stop() -> None:
    result = EvidenceAwareRecoveryPolicy().decide(
        observation(
            completion_requested=True,
            evidence=[
                EvidenceRecord(
                    kind="public_tests",
                    workspace_digest="current",
                    passed=True,
                    fresh_in_current_run=True,
                )
            ],
        )
    )

    assert result.action == RecoveryAction.STOP_SUCCESS
    assert result.current_completion_verified is True


def test_stale_or_prior_run_evidence_cannot_stop_completion() -> None:
    result = EvidenceAwareRecoveryPolicy().decide(
        observation(
            completion_requested=True,
            evidence=[
                EvidenceRecord(
                    kind="public_tests",
                    workspace_digest="old",
                    passed=True,
                    fresh_in_current_run=False,
                )
            ],
        )
    )

    assert result.action == RecoveryAction.CONTINUE
    assert result.current_completion_verified is False
    assert "stale" in result.reason


def test_failed_dirty_workspace_prefers_latest_verified_checkpoint() -> None:
    result = EvidenceAwareRecoveryPolicy().decide(
        observation(
            current_workspace_dirty=True,
            final_verifier_failed=True,
            checkpoints=[
                RecoveryCheckpoint(
                    checkpoint_id="clean",
                    workspace_digest="clean",
                    kind=CheckpointKind.CLEAN_BASE,
                ),
                RecoveryCheckpoint(
                    checkpoint_id="verified-2",
                    workspace_digest="verified",
                    kind=CheckpointKind.VERIFIED,
                    public_test_verified=True,
                    sequence=2,
                ),
            ],
        )
    )

    assert result.action == RecoveryAction.ROLLBACK
    assert result.target_checkpoint_id == "verified-2"


def test_failed_dirty_workspace_restarts_clean_without_verified_checkpoint() -> None:
    result = EvidenceAwareRecoveryPolicy().decide(
        observation(
            current_workspace_dirty=True,
            final_verifier_failed=True,
            checkpoints=[
                RecoveryCheckpoint(
                    checkpoint_id="clean",
                    workspace_digest="clean",
                    kind=CheckpointKind.CLEAN_BASE,
                )
            ],
        )
    )

    assert result.action == RecoveryAction.RESTART_CLEAN
    assert result.target_checkpoint_id == "clean"


def test_clean_failure_falls_through_to_model_routing() -> None:
    result = EvidenceAwareRecoveryPolicy().decide(
        observation(final_verifier_failed=True)
    )

    assert result.action == RecoveryAction.CONTINUE
    assert "model routing" in result.reason
