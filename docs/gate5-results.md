# Gate 5 results: rollback, restart, and state-bound evidence

## Question

Gate 4 showed that Kimi K3 could solve idempotency from a clean starter but not
from Qwen's failed workspace. Gate 5 asks a narrower causal question:

> Did the edited external state block recovery, or did carrying information
> from the prior attempt fail for another reason?

The experiment completes a matched three-arm comparison for the two tasks Qwen
failed in Gate 4 and the same five current model endpoints.

| Arm | Workspace | Prior-attempt handoff | Source |
|---|---|:---:|---|
| cold restart | clean starter | no | Gate 4 baseline |
| dirty continuation | Qwen's edited workspace | yes | Gate 4 recovery |
| clean rollback | clean starter | yes | 10 new Gate 5 runs |

The task contract, model endpoint, reasoning effort, tools, turn limit, token
limit, verifier, and per-run ceiling remain fixed. Only the missing rollback arm
was newly sampled, avoiding 20 redundant paid runs.

The rollback point is the clean pre-attempt state, not a verified intermediate
checkpoint. Gate 4 did not preserve per-turn workspaces, which is an
instrumentation gap to fix before evaluating finer rollback points.

## Result

| Arm | Passed | Total per-run cost represented |
|---|---:|---:|
| cold restart | 1/10 | $0.307164 |
| dirty continuation | 0/10 | $0.297033 |
| clean rollback plus handoff | 0/10 | $0.288659 |

All five models failed all three arms on webhook. On idempotency, Kimi K3 alone
passed the cold arm; it failed both dirty continuation and clean rollback.

Observed comparisons:

- clean restart beat dirty continuation in one model/task cell;
- clean rollback beat dirty continuation in zero cells;
- retaining the handoff beat cold restart in zero cells;
- cold restart beat the same clean state with the handoff in one cell.

There is therefore **no observed state-only rollback rescue**. Gate 4's Kimi
result cannot be attributed specifically to poisoned files. The carryover
condition underperformed, but a single sample per cell cannot separate a
harmful handoff from normal rollout variance.

## What the Kimi trajectory teaches

The Kimi cold run inspected the implementation, rewrote it, ran tests, and
passed the hidden verifier. In the clean-rollback run, Kimi listed and read the
same clean files but made no edit and ran no test. It attempted to finish twice;
the unverified-completion guard rejected the first attempt and stopped the
second.

The rollback handoff included a statement that the prior workspace's public
tests had passed, plus their output. Those tests were true for a different
workspace digest. Although the prompt described them as prior-attempt evidence,
the model behaved as if they established the current clean workspace was done.

This exposes a supervisor contract error: evidence is not portable merely
because text describing it is portable.

## Product decision

Do not learn a rollback or switching policy from these 10 comparisons. There
are no positive recovery labels, and there is only one sample in each cell.

Implement these deterministic rules first:

1. Attach a workspace digest and source to every test, progress, and completion
   claim.
2. Invalidate test evidence whenever the workspace changes, rolls back, or
   restarts.
3. Carry hypotheses and attempted approaches across a handoff, but do not carry
   an unqualified "tests passed" claim to a different digest.
4. Require fresh verification on the current digest before accepting
   completion.
5. Save turn-level checkpoints, including the last digest with a fresh public
   test pass, so future rollback experiments can use more than the clean base.

After that, replicate the decisive Kimi/idempotency cells across multiple runs:
clean/no handoff, clean/stale handoff, clean/digest-bound handoff, and dirty/
digest-bound handoff. That experiment can determine whether better evidence
packaging changes behavior before we spend money training a selector.

## Audit

- New paid runs: **10**.
- Catalog/token estimate: **$0.288659242**.
- Provider-reported incremental usage: **$0.246934297**.
- Gate 5 ceiling: **$2.00**.
- Infrastructure failures: **0**.
- Harbor-valid ATIF trajectories: **10/10**.
- Repository tests: **23 passed**.

The machine-readable report and every final workspace are under
`artifacts/official/gate5-20260821T050227Z`.
