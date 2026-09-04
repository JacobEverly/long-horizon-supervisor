# Initial supervisor policy v0

## Outcome

The development evidence supports two small, reloadable components:

1. a task-start router that ranks the four candidate models from public task
   information; and
2. a continuation-risk estimator that scores whether the current fixed-model
   trajectory will eventually reach verifier-confirmed completion.

Neither component currently beats the strongest transparent baseline by enough
to justify replacing it. More importantly, the logs do not contain observed
outcomes for switching, restarting, or stopping at the same intermediate state.
The initial runtime policy must therefore keep those actions conservative and
rule-based. It must not present the continuation score as a learned switching
policy.

## Product contract

The supervisor is a harness-portable decision layer. The harness owns the live
workspace and tool execution; the supervisor consumes normalized observations
and recommends an action.

At a clean task start, the observable input is:

- public instruction;
- declared difficulty, category, and tags; and
- the set of currently available model routes.

Before a later agent turn, the observable input is:

- current route and turn index;
- prior agent-turn and tool-call counts;
- cumulative input, cache, and output tokens;
- terminal-output length and a bounded tail;
- counts of error, test, pass, and shell-prompt signals; and
- whether the terminal tail was truncated.

The policy action space is:

| Action | Meaning | Training status in v0 |
|---|---|---|
| `continue_same` | Give the current model another turn in the same workspace | Observed |
| `switch_model` | Hand the same persistent workspace to another model | Unobserved |
| `restart_clean` | Retry from the original clean task state on another model | Observed only at task start through matched clean-start rollouts |
| `stop` | Terminate because success is verified or further work is not worth its budget | Unobserved as a mid-run counterfactual |

The end product should choose among these actions using a completion-first
utility, with cost as a secondary objective:

```text
utility = completion_value * P(verifier success) - expected incremental cost
```

For evaluation, the result is reported as a success-versus-cost Pareto frontier
rather than hiding the product tradeoff in one arbitrary scalar. A production
deployment can then choose a point on that frontier with a strong bias toward
completion.

## Development data

The matched development panel contains 35 tasks run from the same clean state on
each of four routes: 140 task-route outcomes. It also contains 1,154 pre-turn
checkpoints from 139 non-empty trajectories, plus one zero-turn protocol failure.

The task-start table records verifier completion, learning-valid status, tokens,
duration, and exact modeled cost for every route. The checkpoint table includes
all four possible actions in its schema, but only `continue_same` has an observed
outcome. Outcomes for unobserved actions remain null.

Each non-empty trajectory contributes total weight one, regardless of how many
turns it contains. This prevents longer trajectories from dominating the
continuation model merely because they yield more checkpoints.

The split and leakage rules are:

- all training rows have `record_split=development`;
- all cross-validation folds hold out entire tasks, never individual turns;
- the 18 Wave 3 held-out prompts have zero normalized-text overlap with the 35
  development tasks;
- observations contain no agent analysis, hidden reasoning, current/future tool
  output, verifier stdout, final run totals, or fabricated counterfactual labels;
- both trainers reject held-out rows and Wave 3 paths; and
- the final task-start artifact was fit on all 35 development tasks and frozen
  before the full 18-task evaluation was complete.

## Task-start router result

The router fits one TF-IDF logistic success head per model route using only
public task metadata. Nested leave-one-task-out evaluation chooses the
success-first cost margin without seeing the outer task. Costs are estimated
from training-fold medians.

| Policy | Development successes | Replayed cost |
|---|---:|---:|
| Best static model, Kimi | 15/35 | $6.6410 |
| Learned one-route choice | 13/35 | $0.4328 |
| Frozen Flash -> Qwen | 17/35 | $1.6879 |
| Learned two-route cascade | 17/35 | $1.6879 |
| Frozen Flash -> Qwen -> GLM | 18/35 | $3.8079 |
| Learned three-route cascade | 17/35 | $4.4739 |
| Frozen four-route cascade | 19/35 | $6.2841 |
| Learned four-route cascade | 19/35 | $6.2841 |

The learned router does not improve on fixed ordering. Its value today is a
rigorous, replaceable baseline and inference interface—not evidence that task
text alone can choose the best model from 35 examples.

## Continuation-risk result

The continuation model uses a hashed representation of the bounded terminal
tail, a route token, and 14 pre-turn counters. Hyperparameters are selected
inside each outer task-held-out fold. The target is eventual verifier-confirmed
completion if the logged model continues.

| Estimator | Weighted Brier (lower is better) | Weighted AP | Weighted ROC-AUC |
|---|---:|---:|---:|
| Constant training prevalence | 0.24561 | 0.2833 | 0.0389 |
| Turn index only | 0.24699 | 0.2315 | 0.0935 |
| Hard late/error rule | 0.28231 | 0.3776 | 0.4921 |
| Learned continuation risk | 0.24460 | 0.3608 | 0.4306 |

The learned model gives only a tiny calibration improvement over constant
prevalence and weak discrimination. It is not reliable enough to govern live
escalation. The result suggests that coarse terminal tails and counters do not
capture enough semantic progress, the development set is too small, or both.

## Runtime policy justified today

The evidence supports this conservative composite:

1. rank clean-start routes with the frozen task-start artifact, while retaining
   the fixed Flash-first cascade as the benchmark default;
2. continue the current model while deterministic budget, output, progress, and
   verification guards permit;
3. after a verifier-confirmed success, stop immediately;
4. after a model-attributable terminal failure or a deterministic no-progress
   threshold, perform a clean restart on the next route; and
5. log the learned continuation score for analysis, but do not let it make an
   unsupported persistent-workspace handoff decision.

This is a real supervisor baseline: it is modular, budgeted, auditable, and can
be connected to different harnesses. It is not yet the final learned supervisor.

## Relationship to existing harnesses

As of August 30, 2026, Hermes Agent and OpenCode contain useful integration
plumbing but not this complete control loop.

Hermes supports manual mid-session model switching, fixed model overrides for
delegates and auxiliary tasks, mixtures of agents, and automatic provider
fallback on capacity errors. Its open context-aware-routing proposal describes
automatic task-sensitive routing as a missing feature. Capacity fallback does
not evaluate whether a healthy model is making enough task progress to justify
its next turn.

OpenCode supports primary agents, subagents, and model-bound agent definitions.
That enables static role routing. Its current subagent interface binds the model
to the agent definition rather than exposing a free model choice at every
invocation, and its documented agent contract does not include a learned live
progress-and-cost supervisor.

Therefore the category claim is deliberately narrow: model routing itself is
not new. The contribution is a harness-portable, empirically evaluated,
closed-loop supervisor for long-running work—especially escalation and
de-escalation based on task progress, verifier evidence, and incremental cost.

References:

- [Hermes model configuration and switching](https://github.com/NousResearch/hermes-agent/blob/main/website/docs/user-guide/configuring-models.md)
- [Hermes provider fallback](https://github.com/NousResearch/hermes-agent/blob/main/website/docs/user-guide/features/fallback-providers.md)
- [Hermes context-aware routing proposal](https://github.com/NousResearch/hermes-agent/issues/66020)
- [OpenCode agent configuration](https://opencode.ai/docs/agents/)
- [OpenCode runtime subagent model-selection discussion](https://github.com/anomalyco/opencode/issues/41233)

## Next evidence required

The next training experiment must branch identical intermediate checkpoints.
For each selected state, continue the current model and also try at least one
alternative model in an equivalent persistent workspace. Record verifier
completion, new validated progress, incremental tokens, incremental cost,
elapsed time, and state-preservation failures for every branch.

Those matched branches create the missing labels for `continue_same`,
`switch_model`, `restart_clean`, and `stop`. Only then can a learned controller
be compared honestly with no supervisor, fixed escalation, and the task-start
cascade on completion and cost.

The first practical harness integration should wrap the existing normalized
`observe -> snapshot -> decide` interface around OpenCode or Hermes Agent. Both
already provide model/provider configuration, subagents, and persistent agent
loops. The adapter should translate their native events and apply a supervisor
decision without making the policy depend on either harness's internal classes.

## Artifacts

- `data/supervisor/gate8-development-task-route-v0.jsonl`
- `data/supervisor/gate8-development-checkpoints-v0.jsonl`
- `data/supervisor/gate8-development-policy-dataset-v0-summary.json`
- `artifacts/official/task-start-router-development-v0/task-start-router-v0.joblib`
- `artifacts/official/task-start-router-development-v0/nested-loocv-report-v0.json`
- `artifacts/training/checkpoint-continuation-risk-v0.joblib`
- `artifacts/training/checkpoint-continuation-risk-v0.json`
- `src/horizon_supervisor/training/evaluate_task_start_router.py` (fit-free held-out replay)
