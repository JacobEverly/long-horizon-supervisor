# Gate 8 twelve-task routing checkpoint

## Outcome

The first task-held-out learned selector does **not** beat the cheapest static
route. Both solve 5 of 12 tasks. This is a data result, not a reason to make the
model more complicated: only 2 of the 12 tasks produce different completion
outcomes across the four routes.

The empirical oracle solves 6 of 12 tasks by using DeepSeek V4 Flash on eleven
tasks and Qwen 3.8 27B on one hard security task. That proves there is routing
headroom, but the current sample does not contain enough contrast to learn it
reliably.

## Audited data

The canonical dataset is
`artifacts/official/gate8-twelve-task-development/matched-outcomes-48-v1.jsonl`.
It contains:

- 12 task groups;
- 4 routes per task;
- 48 verified task-model outcomes;
- 20 successes and 28 failures;
- no missing costs or durations;
- no infrastructure failures represented as model failures.

Four earlier infrastructure errors for the causal-discovery task were
superseded by clean recovery trials. Every retained task-model pair starts from
the same task archive and uses the same agent settings as the other routes in
its matched group.

The task patterns are:

| Pattern | Tasks | Share |
|---|---:|---:|
| Every model succeeds | 4 | 33.3% |
| Every model fails | 6 | 50.0% |
| Models disagree | 2 | 16.7% |

With only 12 tasks, the 95% Wilson interval for the true disagreement rate is
wide: 4.7% to 44.8%.

## Policy comparison

| Strategy | Successful tasks | Total attributed cost |
|---|---:|---:|
| Always DeepSeek V4 Flash | 5/12 | $0.0642 |
| Always GLM 5.3 | 5/12 | $1.1955 |
| Always Kimi K3 | 5/12 | $1.8259 |
| Always Qwen 3.8 27B | 5/12 | $0.9791 |
| Best task-held-out learned policy | 5/12 | $0.0642 |
| Hindsight cheapest-success oracle | 6/12 | $0.2005 |

The aggressive learned policy performs worse: 4/12 at $0.5911. Once the
success-probability margin reaches 0.05, uncertainty correctly makes it fall
back to Flash, reproducing the best static result. This is the behavior a
completion-first policy should have when the evidence is weak.

## Why the evaluation is honest

The learned policies use leave-one-task-out evaluation. For each prediction:

1. all four outcomes for the held-out task are removed;
2. a scorer is fit on the other eleven task groups;
3. the scorer sees only the public instruction and `task.toml` metadata;
4. route cost is forecast from the training-fold median, not the held-out run;
5. the selected route is scored using its previously hidden verifier outcome.

The oracle is explicitly non-deployable because it uses the current task's
outcomes after the fact.

## Data decision

The project now needs two named datasets rather than one ambiguous pool:

1. **Representative panel.** Finish the frozen 18-task wave, then expand with
   outcome-blind proportional sampling toward 50 matched tasks. This estimates
   real benchmark performance without selection bias.
2. **Contrast set.** Screen candidate tasks with the stable GLM/Qwen pair and
   run all four routes when the sentinels disagree or are uncertain. This gives
   the scorer more examples where a routing decision can matter. It must be
   reported separately from the representative benchmark.

The immediate representative gap is six tasks. At the observed 16.7%
disagreement rate, random collection would need about 120 total tasks to yield
20 disagreement cases. Sentinel screening is therefore a data-efficiency tool,
not a replacement for the representative panel.

## Reproduce

```bash
python -m horizon_supervisor.training.route_baseline
```

The machine-readable report is
`artifacts/official/gate8-twelve-task-development/route-baseline-v0.json`.
