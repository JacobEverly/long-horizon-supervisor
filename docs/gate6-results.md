# Gate 6 results: state-bound handoff replication

## Outcome

Gate 6 converts Gate 5's one-off observation into a four-arm replicated
mechanism test. It uses the same Kimi K3 endpoint, idempotency task, tool set,
reasoning setting, verifier, and limits for every run. Three fresh replications
were executed per arm, with arm order rotated across replications.

| Arm | Workspace | Handoff | Passed | Cost |
|---|---|---|---:|---:|
| neutral clean | clean starter | none | 3/3 | $0.177387 |
| stale clean | clean starter | legacy unqualified pass output | 0/3 | $0.270432 |
| digest-aware clean | clean starter | prior evidence explicitly stale | 2/3 | $0.207720 |
| digest-aware dirty | Qwen checkpoint | prior evidence matched but not fresh | 0/3 | $0.216816 |

The catalog/token estimate was **$0.872355**. Provider-reported usage was
**$0.58846548**, below the **$3** hard key and experiment ceiling. No provider
or harness attempt failed.

## What changed in the product

Before the paid experiment, the supervisor gained two deterministic features:

1. Public-test results now carry their workspace digest, observation time,
   source, and current-run freshness. Editing files invalidates prior
   verification.
2. Every tool turn creates a recoverable workspace checkpoint with metadata and
   a revalidated digest.

The receiving model sees prior evidence labeled as one of:

- `STALE FOR THIS STATE` when the mounted workspace digest differs; or
- `STATE-MATCHED BUT NOT FRESH` when files match but the evidence came from an
  earlier rollout.

Neither label satisfies completion. Fresh public tests on the current digest
remain mandatory.

## Behavioral mechanism

The neutral arm edited and freshly tested in all three runs, passing all three.
The digest-aware clean arm also edited and freshly tested in all three, passing
two. Its failure was an incorrect repair that public tests did not catch—not a
premature no-op.

All legacy stale-handoff runs failed. One stopped after inspection without an
edit; two edited and tested but still missed hidden behavior. Every stale run
encountered an unverified-completion guard.

The digest-aware dirty arm freshly tested all three inherited workspaces but
never changed their code. All three failed the hidden verifier and encountered
an unverified-completion guard. Correct evidence labeling did not make locally
plausible inherited code easier to repair.

Directional contrasts:

- digest-aware clean versus legacy stale: **+66.7 percentage points**;
- digest-aware dirty versus digest-aware clean: **−66.7 points**;
- neutral clean versus digest-aware clean: **+33.3 points**.

With three samples per arm and one model/task cell, these are mechanism checks,
not calibrated population effects or statistical-significance claims.

## Product decision

The evidence supports a conservative recovery default:

```text
failed final verification
        ↓
invalidate unbound pass claims
        ↓
restart from a clean or known-good checkpoint
        ↓
provide only state-bound, minimal prior context
        ↓
require fresh verification on the current digest
```

Do not automatically preserve a dirty workspace merely because public tests
pass. Do not automatically inject a prior attempt summary merely because it is
available. On this cell, the neutral clean restart was both more reliable and
less expensive than every carryover condition.

This does **not** yet justify a learned recovery selector. The next sizable
evidence gate must test the same actions on a held-out multi-task, multi-model
suite. The production target remains a higher completion rate than practical
static/rules baselines at an acceptable cost premium, not success on one
carefully selected task.

## Integrity audit

- Paid runs: **12**.
- Infrastructure failures: **0**.
- Harbor-valid ATIF trajectories: **12/12**.
- Turn checkpoints with revalidated workspace digests: **64/64**.
- Repository tests: **28 passed**.
- Lint: **passed**.

The machine-readable report and all workspaces are under
`artifacts/official/gate6-20260825T022556Z`.
