# Gate 3 results: fixed-model Pareto seed

## Outcome

Gate 3 is complete. Four models each attempted the same easy, medium, and hard
coding task through the Prime Intellect Verifiers harness. Every task used a
persistent isolated workspace, public tools/tests during the run, a hidden
final verifier, normalized supervisor events, and an ATIF v1.7 trajectory.

- 12/12 planned model-task runs completed.
- 12/12 final trajectories pass Harbor's official ATIF validator.
- 7/12 runs passed their hidden task verifier.
- Total wall-clock model time was 1,469.7 seconds (24.5 minutes).
- No dynamic routing was allowed; this is the uncontaminated fixed-model
  baseline required before testing a supervisor.

## Model results

| Model | Passed | Completion | Estimated total | Est. cost/success | Avg. seconds | Pareto |
|---|---:|---:|---:|---:|---:|:---:|
| Kimi K3 | 2/3 | 66.7% | $0.344703 | $0.172352 | 153.0 | no |
| GLM 5.2 | 2/3 | 66.7% | $0.065421 | $0.032710 | 121.8 | no |
| DeepSeek V4 Flash | 2/3 | 66.7% | $0.004582 | $0.002291 | 45.5 | yes |
| Qwen3 8B | 1/3 | 33.3% | $0.020568 | $0.020568 | 169.5 | no |

The Pareto rule uses completion rate and average estimated cost: a model is
dominated when another model is at least as successful and no more expensive,
with one strict improvement. On these 12 runs, DeepSeek V4 Flash is the only
observed Pareto point. It matched the two larger models' 2/3 completion while
being approximately 14× cheaper than GLM and 75× cheaper than Kimi on average.

This does **not** establish a universal ranking. Three tasks per model produce
directional product evidence, not a statistically stable capability estimate.

## Task gradient

| Task | Difficulty | Passed | Estimated task spend | What it tested |
|---|---|---:|---:|---|
| ledger accumulation | easy | 4/4 | $0.063217 | basic inspection, edit, and validation |
| TTL cache semantics | medium | 3/4 | $0.119543 | multiple interacting bugs and edge cases |
| feature dependency plan | hard | 0/4 | $0.252513 | graph reasoning, ordering, cycles, and errors |

The 4/4, 3/4, 0/4 staircase is useful. It avoids a benchmark where every model
always passes or always fails, while showing that the current hard task sits at
or just beyond the harness limits for these endpoints.

## Failure analysis

| Run | Observable failure | Supervisor implication |
|---|---|---|
| Qwen medium | Repeated six edits that all failed to match the file, reran the same failing test, and reached 10 turns without changing the source. | Detect repeated tool errors/no state change and escalate or change strategy before consuming the turn budget. |
| Kimi hard | Implemented most of the graph algorithm, passed 3/4 hidden tests, missed validation of an unknown enabled dependency, and hit the 5-minute limit. | Track verified progress separately from elapsed time; near-complete states may need a short reviewer/continuation, not a full restart. |
| GLM hard | Emitted 188 tool calls, dominated by malformed or repeated read requests, and never edited the starter. | A deterministic protocol/circuit-breaker guard should stop tool-call explosions before any learned routing decision. |
| DeepSeek hard | Read five files, made no edit, and stopped after three turns; the starter still raised `NotImplementedError`. | Require evidence of task progress or a verifier pass before accepting an agent's voluntary stop. |
| Qwen hard | Generated 12,296 completion tokens in one turn, produced a malformed tool interaction, made no edit, and hit the completion-token ceiling. | Enforce per-turn output/protocol limits and escalate on malformed tool use. |

These are not one generic “model failure.” They separate deterministic safety
rules (malformed calls, tool explosions, no-change loops, premature stopping)
from the later learned question: which compatible model is most likely to
finish from a particular valid checkpoint?

## Budget audit

- User-authorized maximum: **$50.00**.
- Dedicated OpenRouter key limit: **$50.00**.
- Runner's stricter experiment cap: **$42.00**, based on account credit
  remaining before the tournament.
- Catalog-price estimate from recorded tokens: **$0.4352731882**.
- Provider-reported usage on the dedicated key: **$0.357899674**, including a
  $0.000049179 post-connection diagnostic probe.
- Remaining limit on that key at audit: **$49.642100326**.
- The dedicated key expires automatically on **2026-08-21 at 18:59:11 UTC**.

The provider total is the authoritative billed amount for this dedicated key.
The local estimate is deliberately conservative and reproducible from the
captured token counts and price snapshot. Gate 3 used about 0.72% of the user
ceiling.

## What Gate 3 says about the product

1. **Start with deterministic supervision.** We already have concrete rules
   worth implementing: stop malformed/tool-call explosions, detect repeated
   no-change loops, reject unverified voluntary completion, and distinguish
   near-complete work from zero-progress work.
2. **Do not train a router yet.** The sample is too small, and protocol failures
   would create misleading model-capability labels.
3. **DeepSeek is the fixed-model baseline to beat.** A dynamic policy must
   improve completion above this fixed baseline or preserve completion while
   improving another user-important constraint. Cost savings alone are not a
   compelling claim here because DeepSeek is already extremely cheap.
4. **The routing hypothesis is still alive, but not proved.** Kimi's near-pass
   and the other models' qualitatively different failures suggest checkpoint
   state matters. We have not yet run the counterfactual branches needed to
   show that switching models would rescue those states.

## Next evidence gate

The next teachable checkpoint should be a small, capped **recovery-headroom
experiment**, not training:

1. Add deterministic guards for malformed calls, repeated tool errors,
   unchanged workspaces, and unverified stopping.
2. Add enough medium-to-hard tasks to avoid conclusions from three examples.
3. Save comparable checkpoints, then branch the same state to two or more
   models. This controls for context and directly tests whether a switch
   improves completion.
4. Plot the success/cost Pareto curve at several completion-first thresholds.
5. Train a small supervisor only if those branches reveal repeatable routing
   headroom over DeepSeek and the deterministic guard baseline.

This sequence preserves the product positioning: increase the chance that a
long-horizon agent finishes, while preventing failed or unnecessarily expensive
runs from consuming the budget.
