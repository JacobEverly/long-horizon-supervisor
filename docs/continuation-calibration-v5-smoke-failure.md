# Continuation calibration v5: preflight failure

## Decision

**V5 was stopped before model execution.** No model outcome was observed and no
OpenRouter call was made.

The frozen permission-transport smoke successfully preserved the workspace
digest in both directions and retained a read-only Git object at mode `0444`.
Its final cleanup assertion nevertheless failed because Daytona reported that
one sandbox deletion was still in progress. A later read-only check found zero
remaining sandboxes.

This is an orchestration race, not evidence about the detector, either model,
or checkpoint fidelity. The failed smoke report remains immutable and V5 will
not be executed or reinterpreted as a pass.

## Smallest justified next revision

V6 may change only the smoke-test cleanup finalization: after requesting
deletion, wait for Daytona's asynchronous state transition to settle before
recording the final sandbox count. The task cohort, routes, detector thresholds,
turn and token limits, analysis gates, and natural-continuation protocol remain
unchanged.

## Evidence

- V5 frozen manifest SHA-256:
  `3c3e5b1b2b06461a0ac9a617208e55eeaff837e1ae87cbbd9ddf7f4895c8cea8`
- Failed smoke report SHA-256:
  `cb5529e5339a6c255d39abc76a6874fe1e33ecd8da56ec86f0003f02fb2185aa`
- Provider model calls: **0**
- Model outcomes observed: **0**
- Sandboxes at the later cleanup check: **0**
