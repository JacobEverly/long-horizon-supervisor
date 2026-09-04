# Stage-data checkpoint: is the dataset ready?

## Decision

Yes for the first completion/stage recognizer. No for the final model-switching
claim.

The current dataset is large and diverse enough to stop collecting generic
successful trajectories. It supports learning whether an observed agent state
looks ready to finish across terminal and software-engineering harnesses. It
does not yet say which candidate model should act next, whether a switch will
recover a failure, or whether the supervisor improves Terminal-Bench task
completion.

## Training recipe

`data/supervisor/supervisor-checkpoints-v2.jsonl` contains 160,279 online-safe
checkpoints from three pinned, permissively licensed sources:

| Source | Retained checkpoints | Raw share | Effective training weight |
|---|---:|---:|---:|
| NVIDIA Terminal Pivot | 31,111 | 19.4% | 60% |
| OpenThoughts Agent | 114,803 | 71.6% | 20% |
| NVIDIA SWE Pivot | 14,365 | 9.0% | 20% |

The weights, rather than the raw row counts, define the product priority. The
terminal source receives most of the influence because Terminal-Bench is the
primary benchmark. The broader source and OpenHands-style SWE source each
receive enough weight to test cross-harness generality.

Only SWE trajectories with generation pass rate at least 0.625 are retained.
Pass rate is used as an offline row-quality filter and is never a model input.
This removes 35,943 checkpoints and 63 task records.

The resulting splits contain:

| Split | Checkpoints | Task groups represented by checkpoints |
|---|---:|---:|
| Train | 119,925 | 14,141 |
| Validation | 15,908 | 1,905 |
| Internal development test | 16,732 | 1,949 |
| Sealed test | 7,714 | 990 |

Every split is assigned by normalized task-description hash. No normalized task
appears in multiple sources or splits. The SWE source has zero exact instance-ID
overlap and zero normalized problem-text overlap with the 500-task pinned
SWE-bench Verified holdout.

## Result

The baseline uses only the already-observed 4,096-character terminal tail and
numeric progress features. It deliberately excludes the task description,
future turns, reference answers, expected commands, and verifier scores.

On the sealed test:

| Slice | Positive rate | Average precision | ROC-AUC |
|---|---:|---:|---:|
| All sources | 14.4% | 79.0% | 94.2% |
| NVIDIA Terminal Pivot | 9.4% | 35.8% | 83.3% |
| OpenThoughts Agent | 16.9% | 84.2% | 95.6% |
| NVIDIA SWE Pivot | 3.9% | 26.3% | 88.4% |

The untouched SWE transfer diagnostic had previously produced 5.2% average
precision on high-confidence rows. After adding only the high-confidence SWE
training split, the held-out SWE slice reaches 26.3%. This is evidence that the
missing ingredient was harness-specific supervision, not merely a larger
classifier.

The original seven stage-data reference thresholds are all met by this
three-source model: overall average precision and ROC-AUC, terminal average
precision and ROC-AUC, OpenThoughts average precision and ROC-AUC, and the
full-model gain over structured features. This comparison is diagnostic rather
than a new precommitted sealed test because the SWE source was added after the
first sealed evaluation.

## Size decision

The source-stratified learning curve is:

| Training tasks | Training checkpoints | Internal-test average precision |
|---:|---:|---:|
| 1,414 | 12,125 | 66.5% |
| 3,535 | 29,997 | 76.6% |
| 7,070 | 59,980 | 78.9% |
| 14,141 | 119,925 | 80.4% |

Doubling the final tranche adds 1.5 points. The curve is still rising, but the
gain is small enough that another generic public trajectory dump has lower
expected value than collecting the missing labels.

The SWE slice contains only 28 positives in the internal test and 29 in the
sealed test, so its per-fraction learning curve is noisy and non-monotonic. That
is a reason to collect targeted SWE failure/completion states later, not a
reason to let the largest public source dominate the mixture.

## Relationship to the product benchmarks

| Benchmark | Dataset relationship | What is proved now | Still required |
|---|---|---|---|
| Terminal-Bench 2.1 | A 1,950-task source panel exactly matches the frozen 30-task category proportions; terminal trajectories receive 60% training weight | Held-out completion-stage signal | Run fixed models and the supervisor on the frozen executable tasks |
| SWE-bench Verified | SWE pivot is OpenHands-style and has zero task/text overlap with Verified | Held-out cross-harness completion-stage signal | Repository-level completion evaluation |
| BigCodeBench | Kept out of training | Clean component holdout | Evaluate static routing and code-generation components |
| LiveCodeBench | Kept out of training and license-quarantined | No leakage into training | Resolve license and use a time-based holdout |

“Good for the benchmark” therefore means benchmark-aligned pretraining, not
benchmark success. Actual success remains the official executable verifier
result.

## Next collection

Stop adding generic successful trajectories. The next candidate set is now
frozen in `data/supervisor/terminal-bench-pro-panel-v0.jsonl`: 72 public
Terminal-Bench Pro tasks arranged as four identical 18-task waves. Every wave
preserves the difficulty/category proportions of the 18 frozen benchmark seats
that Terminal-Bench Pro covers, and no selected task name overlaps the final
30-task manifest. This covers 60% of the final benchmark's seats; the nine
missing difficulty/category strata remain an explicit limitation.

Run wave one first, with every candidate model receiving the same task or
recoverable checkpoint. Preserve completion, verifier evidence, tokens, cost,
duration, and external-state digest for every arm. Include failed and stalled
states, because successful teacher traces cannot teach recovery or escalation.

Wave one has been materialized into 18 Harbor-compatible task directories with
all source hashes and archive paths verified. A local oracle smoke did not reach
the task: Docker timed out contacting Terminal-Bench Pro's Alibaba-hosted base
image registry. That run is an infrastructure error, not a zero reward. Retry
the unchanged task in the x86 cloud sandbox before starting paid model arms.

The matched runner is also frozen. It expands the 18 tasks across Qwen3.8 27B,
DeepSeek V4 Flash, GLM 5.3, and Kimi K3 for 72 same-task trials. Every arm uses
Terminus 2, 12 turns, a 4,096-token response ceiling, the same clean task
archive, and the same Daytona environment. The OpenRouter provider key is a
hard global experiment cap; Harbor runs no retries and at most two trials
concurrently.

The outcome builder refuses duplicate task/model pairs, records missing pairs,
keeps infrastructure errors separate from model failures, and attaches the
initial task-archive digest to every outcome. The full wave is accepted as
matched data only when all 72 task/model pairs appear exactly once.

That matched panel supplies the scarce targets:

1. whether to continue, restart, roll back, or switch;
2. which model maximizes completion probability from the same state;
3. which successful choice is cheapest;
4. where the completion-versus-cost Pareto frontier actually lies.

Continue through waves two to four only if routing regret, uncertainty, and the
budget justify expansion from 18 to 36, 54, and 72 matched tasks. Expand toward
400 shared tasks only if the pilot's learning curves show that additional
matched outcomes improve the policy.
