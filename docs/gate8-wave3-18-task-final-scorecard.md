# Gate 8 Wave 3 final scorecard

## Executive result

The sealed evaluation supports the product thesis that model complementarity can
raise completion rates on long-horizon agent tasks. It does not yet support the
claim that the first learned supervisor chooses or switches models better than a
transparent fixed cascade.

Across 18 previously held-out tasks and four frozen model routes, the best
single model completed 7 tasks (38.9%). The fixed four-model cascade completed
12 tasks (66.7%), stopping as soon as the external verifier confirmed success.
That is a gain of 5 completed tasks, or 27.8 percentage points. The two-model
Flash-to-Qwen cascade completed 9 tasks (50.0%) at substantially lower replayed
cost than always using Kimi.

The learned task-start router was frozen before final evaluation and evaluated
without fitting or tuning. On this held-out set it produced the same ordering
and therefore the same completion and cost results as the fixed cascade at all
four route depths. The checkpoint continuation model showed only a tiny
calibration improvement on development data and weak discrimination. Neither
learned component should control live mid-run switching yet.

## Sealed evaluation integrity

- Evaluation panel: 18 tasks x 4 routes = 72 learning-valid outcomes.
- Split: all 72 final rows are `held_out`.
- Source records: 85; 8 infrastructure errors and 5 provider errors were
  superseded route-for-route by valid recovery runs.
- Final learning statuses: 67 verifier-scored outcomes and 5 valid agent
  protocol failures.
- Held-out tuning: none. Route identities, model versions, prompts, limits,
  cascade order, stopping rule, and verifier behavior remained frozen.
- Task patterns: 2 all-model successes, 6 all-model failures, and 10
  discriminating tasks where route choice mattered.

## Static models and frozen cascades

Costs below are comparable, cache-aware catalog replay costs for the exact
observed attempts. They are not the experiment's provider bill.

| Policy | Completed | Success rate | Replayed cost |
|---|---:|---:|---:|
| Flash only | 6/18 | 33.3% | $0.1208 |
| GLM only | 5/18 | 27.8% | $2.8431 |
| Kimi only | 7/18 | 38.9% | $4.0559 |
| Qwen only | 5/18 | 27.8% | $1.4019 |
| Flash -> Qwen | 9/18 | 50.0% | $1.1187 |
| Flash -> Qwen -> GLM | 10/18 | 55.6% | $2.3749 |
| Flash -> Qwen -> GLM -> Kimi | 12/18 | 66.7% | $3.8475 |

Every point below is Pareto-optimal in the observed completion-versus-cost
tradeoff:

| Operating point | Completed | Replayed cost |
|---|---:|---:|
| Cheapest useful baseline: Flash | 6 | $0.1208 |
| Low-cost cascade: Flash -> Qwen | 9 | $1.1187 |
| Midpoint: add GLM | 10 | $2.3749 |
| Completion-first: add Kimi | 12 | $3.8475 |

The completion-first choice is the four-route cascade. The two-route cascade is
the strongest efficiency result: it completed two more tasks than always using
Kimi while its replayed cost was 72% lower. Six tasks defeated every tested
route, so no chooser over only these four clean-start outcomes could exceed
12/18 completion.

The final resource-heavy task illustrates the complementarity. Flash and Kimi
passed the Prime HTTP-server verifier, while Qwen and GLM failed it. Flash took
about 28 minutes and generated roughly 181k tokens but cost only about one cent;
Kimi finished in about 9 minutes with roughly 121k tokens and cost about $0.28.
Latency, tokens, dollars, and success are distinct product constraints.

## Exact experiment spend

The dedicated OpenRouter key moved from $23.552935902 to $27.653569393 of usage.
Exact incremental provider spend was therefore **$4.100633491**, below the
authorized $10 ceiling. Completed and stopped run reports account for
$4.066255204; the $0.034378287 difference is delayed provider accounting between
run-level snapshots. No paid held-out call was omitted from the key delta.

## What the initial supervisor learned

The development-only training data contains:

- 140 matched clean-start outcomes: 35 tasks x 4 routes; and
- 1,154 leakage-controlled pre-turn checkpoints from 139 non-empty
  trajectories, with each trajectory receiving total weight one.

The task-start router trains one text-and-public-metadata success model per
route and ranks routes with a success-first cost margin. Its frozen held-out
result exactly matched the fixed Flash -> Qwen -> GLM -> Kimi order:

| Maximum routes | Fixed cascade | Learned router | Replayed cost, both |
|---|---:|---:|---:|
| 1 | 6/18 | 6/18 | $0.1208 |
| 2 | 9/18 | 9/18 | $1.1187 |
| 3 | 10/18 | 10/18 | $2.3749 |
| 4 | 12/18 | 12/18 | $3.8475 |

This is a valid trained baseline and a reusable inference interface, but not an
adaptive improvement. The 35 development tasks were too few, or their public
text features too weak, to justify task-specific ordering.

The continuation-risk model estimates eventual success if the current route
continues. On task-held-out development folds its weighted Brier score was
0.24460 versus 0.24561 for constant training prevalence, but AP was 0.3608 and
ROC-AUC was 0.4306. That is too weak to govern escalation or de-escalation.

## Decision

**Go** on the supervisor product and harness layer, using the fixed
completion-first cascade plus deterministic budget, progress, protocol, and
verifier guards. The held-out result demonstrates real route complementarity
and a useful success-versus-cost curve.

**No-go** on claiming a learned live switching policy. The current logs observe
only `continue_same` from intermediate states. They do not show what would have
happened if the same saved workspace had switched models, restarted cleanly, or
stopped. Creating those labels from unrelated trajectories would be fabricated
counterfactual supervision.

The honest product positioning is therefore: **increase the empirical chance
that long-running agents finish, while exposing a selectable cost frontier**.
It is not a guarantee of success; the observed completion-first ceiling was
66.7% on this panel.

## Next harness experiment

Connect the existing normalized `observe -> snapshot -> decide -> act` contract
to one persistent-workspace harness such as Hermes Agent or OpenCode. The
adapter, not the policy, should own harness-specific event translation and model
configuration.

At selected intermediate checkpoints, clone the exact same workspace and run
matched branches for:

1. continue the current model;
2. switch models in the same workspace;
3. restart cleanly on the next route; and
4. stop when success is externally verified or the budget is exhausted.

For each branch, record new verifier-confirmed progress, final completion,
incremental tokens, incremental dollars, elapsed time, and any state-transfer
failure. Group every split by original task. That experiment will create the
first defensible labels for escalation and de-escalation and allow the learned
controller to compete against no supervisor, fixed escalation, and the current
hard-coded guards.

New models remain modular at the harness layer. A new route can enter through a
small calibration panel and use the fixed fallback immediately; the learned
ranker should be retrained only after enough development outcomes exist. The
full agent or task harness does not need to be rebuilt for every model release.

## Reproducible artifacts

- Final outcomes: `artifacts/official/gate8-wave3-18-task-checkpoint/matched-outcomes-72-v1.jsonl`
- Merge audit: `artifacts/official/gate8-wave3-18-task-checkpoint/matched-outcomes-72-summary-v1.json`
- Fixed scorecard: `artifacts/official/gate8-wave3-18-task-checkpoint/frozen-policy-scorecard-18-task-v0.json`
- Fit-free learned scorecard: `artifacts/official/gate8-wave3-18-task-checkpoint/task-start-router-heldout-scorecard-v0.json`
- Development task-route data: `data/supervisor/gate8-development-task-route-v0.jsonl`
- Development checkpoint data: `data/supervisor/gate8-development-checkpoints-v0.jsonl`
- Frozen task-start artifact: `artifacts/official/task-start-router-development-v0/task-start-router-v0.joblib`
- Continuation-risk artifact: `artifacts/training/checkpoint-continuation-risk-v0.joblib`
- Initial policy contract: `docs/initial-supervisor-policy-v0.md`
