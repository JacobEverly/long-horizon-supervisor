# Experiment v0: completion first, efficiency second

## Product question

Can a lightweight supervisor dynamically choose among model tiers during a
long-running coding task while preserving task completion and reducing the
frontier compute used per successful task?

The key word is **during**. Prompt difficulty alone is insufficient because a
task changes shape: architecture may require frontier reasoning, implementation
may become mechanical, and an unexpected failure may require escalation again.

## Decision rule

At checkpoint state `s`, for every available model `m`, estimate:

- `reliability(s, m)`: chance that the model can successfully advance the
  current phase and ultimately preserve task completion;
- `remaining_cost(s, m)`: forecast cost if that model handles the current work.

Select:

```text
the cheapest model whose reliability clears the completion threshold
```

If no affordable model clears the threshold, select the most reliable affordable
model and mark the constraint as unmet. Recovery may use the explicitly reserved
budget. If no model is affordable, halt instead of silently exceeding the user's
budget. Cost never makes a known-unreliable model look good.

In the first implementation, `reliability` is an explicitly uncalibrated
heuristic score. It exists to validate the product and software contract. Later,
the trained supervisor will replace the heuristic with calibrated estimates.

## Reward versus monitoring

The task verifier determines the primary outcome:

```text
task_completed = 1 if the benchmark verifier passes, otherwise 0
```

Cost, elapsed time, model switches, tokens, and frontier share begin as
monitoring metrics. We do not immediately blend them into the reward because a
cheap failed task is not useful.

We compare policies at fixed budgets and report:

1. completion rate;
2. cost per completed task;
3. frontier-token share;
4. time per completed task;
5. missed escalations and unsafe downgrades;
6. switching and handoff overhead.

## Baselines

1. Best fixed single model.
2. All-frontier quality reference.
3. Static frontier-plan -> cheap-execute workflow.
4. NVIDIA Switchyard's rule-based capable-first stage router.
5. Our transparent rule-based completion-constrained scheduler.
6. Our learned completion-constrained scheduler.

An empirical oracle is reported only as a retrospective headroom estimate: the
best observed route among branches that we actually ran. It is not deployable
and should not be presented as a fair competitor.

## Data and evaluation splits

The first paid experiment will use a small, stratified sample of executable
long-horizon coding tasks. Tasks, repositories, and model endpoints will be
versioned. We will keep repositories disjoint where practical:

- development: policy and feature iteration;
- validation: threshold selection and calibration;
- test: one final comparison after decisions are frozen.

We will retain complete trajectories, task-verifier outcomes, model/tier used at
each step, tokens, cost, elapsed time, and normalized supervisor state.

## Training ladder

We only climb this ladder when the previous step shows measurable headroom:

1. Public benchmark priors; no paid runs.
2. Rule-based policy on a small paired-model calibration set.
3. Supervised LoRA on checkpoint outcome or routing labels.
4. Calibrate predicted completion probabilities on held-out data.
5. Only then consider preference learning or RL.

This ordering is deliberate. RL cannot rescue an ill-defined environment,
uninformative reward, or task distribution on which all models always pass or
always fail.

## Current Gate 3 budget ceiling: $50

The ceiling is not a spending target. The runner refuses a larger value. Gate 3
uses four fixed model endpoints across three executable tasks, with no dynamic
routing and no automatic model retries.

| Use | Ceiling |
|---|---:|
| Gate 1 local environment and integration | $0 |
| Gate 2 easy-task endpoint checks (also Gate 3 easy rows) | $14 |
| Gate 3 medium and hard fixed-model runs | $28 |
| Provider/error reserve | $8 |

Every paid phase has a stop condition. We do not spend the next tranche until
the preceding phase produces usable trajectories and outcome diversity.

At the catalog prices captured on 2026-08-20, expected spend is far below the
ceiling. The larger ceilings are safety limits for context growth and provider
variance, not expected invoices. Any later routing, branching, or training
experiment requires a fresh budget decision.

## Gate definitions

1. **Gate 1 — instrumentation:** all three starters fail, known repairs pass,
   the real Verifiers tool loop passes every task, and ATIF exports validate.
2. **Gate 2 — endpoint viability:** the easy task runs once through every
   candidate endpoint. This checks authentication, tool calling, telemetry,
   and final verification before harder runs.
3. **Gate 3 — pure-model Pareto seed:** each candidate runs the easy, medium,
   and hard tasks once. We report completion, estimated token cost, time,
   turns, tool calls, and failure class. With only three observations per
   model, this is directional evidence rather than a leaderboard claim.

## Gate 3 result

Gate 3 completed on 2026-08-20 with all 12 planned runs and 12 valid ATIF
trajectories. The task gradient behaved as intended: 4/4 models passed easy,
3/4 passed medium, and 0/4 passed hard. DeepSeek V4 Flash, Kimi K3, and GLM 5.2
each passed 2/3 tasks; DeepSeek was the only observed cost/completion Pareto
point. Qwen3 8B passed 1/3.

The fixed-model estimate was $0.435273. The provider reported $0.357900 on the
dedicated experiment key, including a small connectivity probe, against the
$50 user ceiling. See [`gate3-results.md`](gate3-results.md) for the full audit
and interpretation.

## Stop conditions

Pause or redesign if:

- every model scores near 0% or near 100% on the selected tasks;
- the simple static handoff matches the empirical oracle closely;
- routing overhead consumes the apparent savings;
- verifier failures are dominated by broken environments;
- a single middle-tier model dominates both completion and cost;
- the available checkpoint state does not predict future success.
