# Swiss-cheese clean-start replication: final decision

## Executive result

The experiment supports a **go** decision on a transparent, verifier-gated
portfolio of models. It supports only a **suggestive, not confirmed** claim that
cross-model rescue is repeatably better than giving the same model another
attempt.

The strongest predeclared comparison was Flash followed by Qwen. It completed
13 of 20 confirmatory task-replication units (65%), while Flash followed by an
independent Flash retry completed 6 of 20 (30%). That is a 35-point observed
gain. However, the task-clustered bootstrap 95% interval for the gain was
0 to 70 points, so the ten-task panel is too small to call the causal effect
statistically established.

The practical result is still useful. Model failures were jagged rather than a
strict weak-to-strong ladder, and the full clean-start cascade completed 16 of
20 units (80%). The smaller 9B model repeated its XRD success in both
confirmatory replications and slightly improved the dollar-cost frontier when
inserted before Qwen. It did not add unique task coverage, and it increased
tokens and elapsed time.

## What was actually tested

- Panel: the ten Wave 3 tasks on which the original four routes disagreed.
- Role: post-hoc replication/development, not a second held-out evaluation.
- Routes: DeepSeek V4 Flash 0731, Qwen3.8 27B, GLM 5.3, Kimi K3, and
  Qwen3.5 9B.
- Matrix: 10 tasks x 5 routes x 3 clean starts = 150 valid outcomes.
- Confirmatory evidence: replications 2 and 3 only, or 100 outcomes.
- Discovery evidence: replication 1 is descriptive only.
- Completion: external task verifier reward of 1.
- Cascades: clean restart between models and stop after first verified success.
- Mid-run persistent-workspace handoff: not tested here.

The final matrix contains 136 verifier-scored outcomes and 14 attributable
agent-protocol failures. Infrastructure-invalid attempts were excluded and
recovered route-for-route without rerunning any valid result. Every retained
task-model-replication key appears exactly once and every task began from the
same locked initial-state digest.

## Static and same-model results

The table uses the 20 confirmatory task-replication units. Costs are comparable,
cache-aware list-price replays for the observed attempts, not the provider bill.

| Model | One attempt | Success | Replayed cost | Two same-model attempts | Success |
|---|---:|---:|---:|---:|---:|
| Flash | 4/20 | 20% | $0.1687 | 6/20 | 30% |
| Qwen 27B | 10/20 | 50% | $2.1426 | 10/20 | 50% |
| GLM | 7/20 | 35% | $2.8326 | 10/20 | 50% |
| Kimi | 10/20 | 50% | $4.5920 | 12/20 | 60% |
| Qwen 9B | 2/20 | 10% | $0.2986 | 2/20 | 10% |

Retrying was not uniformly useful. A second Flash attempt rescued 2 of 16
first-attempt failures; Qwen and the 9B model gained nothing from retrying;
GLM gained three units; and Kimi gained two.

## Same-model versus different-model controls

Each comparison shares the identical first attempt. Rescue rates are calculated
only where that first attempt failed.

| First -> second | Completed | Cross-model rescues | Same-model control | Completion delta | Cost delta | Task-bootstrap 95% interval |
|---|---:|---:|---:|---:|---:|---:|
| Flash -> Qwen | 13/20 | 9/16 | Flash -> Flash: 6/20 | +35 points | +$1.3187 | 0 to +70 points |
| Kimi -> Flash | 13/20 | 3/10 | Kimi -> Kimi: 12/20 | +5 points | -$2.2349 | -15 to +30 points |
| Kimi -> Qwen | 12/20 | 2/10 | Kimi -> Kimi: 12/20 | 0 points | -$1.5797 | 0 to 0 points |
| Kimi -> GLM | 14/20 | 4/10 | Kimi -> Kimi: 12/20 | +10 points | -$1.0796 | -10 to +30 points |
| Flash -> 9B | 6/20 | 2/16 | Flash -> Flash: 6/20 | 0 points | +$0.1096 | -20 to +25 points |

Flash -> Qwen is the strongest evidence for diversity rather than mere retry:
Qwen rescued five distinct task types after Flash failed. Kimi -> GLM is also a
useful product result because its point estimate is higher and its replayed cost
is lower than Kimi -> Kimi, but its uncertainty interval is wide.

No predeclared comparison has a bootstrap lower bound above zero. The correct
causal conclusion is therefore **suggestive cross-model complementarity, not a
confirmed effect**.

## Coverage was genuinely jagged

Cells show passes across the two confirmatory replications.

| Task | Flash | Qwen 27B | GLM | Kimi | Qwen 9B |
|---|---:|---:|---:|---:|---:|
| CRC32 logic gates | 2/2 | 0/2 | 1/2 | 0/2 | 0/2 |
| Code similarity | 1/2 | 2/2 | 2/2 | 1/2 | 0/2 |
| Portfolio allocation | 0/2 | 2/2 | 0/2 | 1/2 | 0/2 |
| Polyglot text stats | 0/2 | 2/2 | 2/2 | 2/2 | 0/2 |
| YAML validator | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| Prime HTTP server | 0/2 | 2/2 | 0/2 | 2/2 | 0/2 |
| Sanitize PostgreSQL WAL | 1/2 | 0/2 | 0/2 | 2/2 | 0/2 |
| Recover password from Git | 0/2 | 0/2 | 1/2 | 0/2 | 0/2 |
| Repair shell pipeline | 0/2 | 0/2 | 1/2 | 0/2 | 0/2 |
| XRD peak fitting | 0/2 | 2/2 | 0/2 | 2/2 | 2/2 |

GLM had the only strict unique task contributions: Git-history recovery and
shell-pipeline repair. The 9B model was not unique because Qwen and Kimi also
solved XRD. It was still useful as a cheap early attempt on that task.

Pairwise overlap reinforces the jaggedness. Flash and Qwen shared only one
success and had negatively correlated failures (phi -0.25). Flash and Kimi had
the same pattern. Qwen and Kimi were much more redundant: eight shared
successes and failure phi 0.60. GLM and Kimi were mildly complementary, with
failure phi -0.10. The complete ten-pair table is in the machine-readable
scorecard.

## Cascades and Pareto frontier

| Policy | Completed | Replayed cost | Tokens | Sequential elapsed time |
|---|---:|---:|---:|---:|
| Flash only | 4/20 | $0.1687 | 2.12M | 7.3h |
| Flash -> Flash | 6/20 | $0.2878 | 3.60M | 11.9h |
| Flash -> Qwen | 13/20 | $1.6066 | 3.83M | 11.6h |
| Flash -> Qwen -> GLM -> Kimi | 16/20 | $3.3202 | 4.84M | 13.7h |
| Flash -> 9B -> Qwen -> GLM -> Kimi | 16/20 | $3.2055 | 6.50M | 15.2h |

These are the observed success-cost Pareto points: Flash; Flash -> Flash;
Flash -> Qwen; and Flash -> 9B -> Qwen -> GLM -> Kimi.

The 9B overlay preserved 16/20 completion and reduced replayed dollar cost by
$0.1147, or 3.5%, because its two XRD wins avoided later expensive calls. The
tradeoff was 34.5% more tokens and 11.1% more sequential elapsed time. It earns
a **conditional** portfolio slot when dollar cost matters more than latency or
token volume; it is not a universally better route.

In the five-model cascade, wins were attributed to Flash 4 times, the 9B model
2 times, Qwen 7 times, GLM 2 times, and Kimi once. Four units still failed. The
non-deployable empirical oracle also reached 16/20 within the same replication,
and at least one model succeeded on 9 of 10 tasks across either confirmatory
replication. The YAML validator remained unsolved.

## Stability and uncertainty

Model rank was unstable across repetitions, which argues against a single
global quality ladder.

| Replication | Flash | Qwen | GLM | Kimi | 9B | Leader |
|---|---:|---:|---:|---:|---:|---|
| 1, outcome-selected discovery | 4 | 3 | 3 | 5 | 1 | Kimi |
| 2, confirmatory | 1 | 5 | 3 | 6 | 1 | Kimi |
| 3, confirmatory | 3 | 5 | 4 | 4 | 1 | Qwen |

The full cascade's task-bootstrap 95% interval is 55% to 100%. The wide range
is a direct consequence of having only ten independent task clusters. Repeated
starts increase measurement precision within a task, but they do not create
ten additional task types.

## Exact provider spend

The experiment used one dedicated OpenRouter key until it expired and a capped
one-day continuation key for the last three valid outcomes and one invalid
infrastructure attempt.

- Frozen experiment baseline: $27.653569393.
- First key final usage: $39.694240884.
- Continuation-key final usage: $0.037565900.
- Conceptual combined final usage: $39.731806784.
- **Exact incremental provider spend: $12.078237391.**
- Authorized ceiling: $20.00.
- Run reports account for $11.874926981.
- Key delta not allocated to reports: $0.203310410, including provider
  accounting lag and orchestration calls; it remains included in the exact
  total.

No Daytona environments or benchmark processes remained after collection.

Final verification passed: all 162 repository tests succeeded with the project
environment active, Ruff reported no lint errors, and the 150-row matrix passed
its rectangularity, learning-validity, initial-state, and endpoint-lock checks.

## Decision and next experiment

### Product decision

Use a verifier-gated clean-start portfolio now. Offer two transparent operating
points:

1. **Default efficient policy:** Flash -> Qwen. It is the clearest low-cost
   diversity gain, at 13/20 observed completion.
2. **Completion-first policy:** Flash -> 9B -> Qwen -> GLM -> Kimi. It reaches
   the observed ceiling of 16/20 and is slightly cheaper in dollars than the
   existing four-model cascade, while explicitly accepting higher latency and
   token use.

Do not market either as a guarantee. Do not claim that the learned supervisor
has solved live switching.

### Exact subset for the matched mid-run branching experiment

Start with **Flash and Qwen only**. This pair had the largest predeclared
same-versus-different rescue gap, the lowest-cost strong heterogeneous result,
and low success overlap. At each frozen checkpoint, clone the exact workspace
into four matched branches:

1. continue Flash in the existing workspace;
2. restart Flash from the clean task start;
3. switch to Qwen in the existing workspace; and
4. restart Qwen from the clean task start.

Those four arms distinguish extra compute, model diversity, and preservation of
external state. Record verifier-confirmed incremental progress, final success,
additional tokens, dollars, time, and state-transfer failures. Group all train,
validation, and evaluation splits by original task.

Keep the 9B model as an optional clean-start pre-pass, not as a first live
handoff target: it showed repeatable cheap task competence but no advantage
over the matched Flash retry control. Hold GLM and Kimi for a later
generalization round after the Flash/Qwen branch protocol works; otherwise the
first matched-state experiment multiplies cost before the core handoff question
is answered.

This is the data the future supervisor actually needs. The present 150 rows can
train or calibrate task-start ordering, but they cannot provide valid labels for
`continue`, `switch`, `restart`, or `stop` from an identical mid-run state.

## Reproducible artifacts

- Frozen design: `artifacts/official/swiss-cheese-replication-v0/frozen-experiment-manifest-v0.json`
- Final 150-row matrix: `artifacts/official/swiss-cheese-replication-v0/swiss-cheese-matrix-150-v0.jsonl`
- Matrix integrity summary: `artifacts/official/swiss-cheese-replication-v0/swiss-cheese-matrix-150-v0-summary.json`
- Machine-readable scorecard: `artifacts/official/swiss-cheese-replication-v0/swiss-cheese-scorecard-v0.json`
- Exact execution ledger: `artifacts/official/swiss-cheese-replication-v0/execution-ledger-v0.json`
