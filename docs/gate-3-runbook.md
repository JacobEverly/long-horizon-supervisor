# Gate 3 runbook: pure-model Pareto seed

## What this gate answers

Before training a supervisor, determine whether the candidate worker models
actually occupy different completion/cost regions on the same executable tasks.
If one fixed model is both cheaper and at least as successful as the others,
that is a useful result: routing has not yet earned its complexity.

## Candidate roles

| Model | Experimental role |
|---|---|
| Qwen3 8B | Small, cheap capability floor |
| DeepSeek V4 Flash | Fast reasoning/value endpoint |
| GLM 5.2 | Middle-tier generalist |
| Kimi K3 | High-reasoning ceiling |

These labels are hypotheses, not rankings. The price snapshot and advertised
tool support are fetched immediately before the experiment.

## Task ladder

1. `ledger-accumulation` — one localized arithmetic bug.
2. `ttl-cache-semantics` — interacting TTL and falsy-value bugs across modules.
3. `feature-dependency-plan` — a new deterministic graph feature with error
   handling, cycles, transitive dependencies, and stable ordering.

Every starter is proven to fail hidden tests. The reference repair for every
task is proven to pass. Agents can run public tests, but cannot see hidden tests.

## Controls

- Same system prompt, tools, task workspace, and hidden verifier for every model.
- One rollout per model/task pair; no dynamic routing.
- Sequential execution, no automatic model retry.
- 2,048 maximum completion tokens per call, 12,000 per rollout, and 10 turns.
- Conservative prompt-size forecast before every call.
- $3.50 per-run halt and $50 hard experiment ceiling.
- Actual estimate uses the timestamped provider catalog price and returned
  prompt/completion token counts.

If a provider or harness connection fails, the run stops immediately. An
operator can resume the same artifact directory with `--resume`; successful
model/task pairs are not repeated, while infrastructure failures are retried
and preserved as separate failure-attempt artifacts.

## Reading the result

`gate3-report.json` contains each run and per-model aggregates. A model is
marked Pareto-efficient when no observed model has both at least its completion
rate and at most its average cost, with one strict improvement.

Three tasks are enough to validate infrastructure and expose obvious dominance.
They are not enough for narrow statistical claims. Gate 4 should add more
long-horizon tasks only if Gate 3 yields capability or failure diversity.
