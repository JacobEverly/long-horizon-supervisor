NOT READY — two-tier detector gate failed

# Two-tier detector development result

The two-tier design did not pass its frozen offline gate, so no fresh tasks, paid model calls, Daytona environments, checkpoint snapshots, interventions, or training runs were launched.

## Why two tiers

Detector v1 made “stuck” more meaningful by requiring repeated productive failures, but exact replay reduced stuck hits from 22 to 3. V2 separates two jobs that one threshold could not perform simultaneously:

- `NEEDS_REVIEW` is a broad signal that a run deserves supervision.
- `CONFIRMED_STUCK` is a later, stricter signal requiring repeated observable failures and productive work.

`HEALTHY` remains distinct. Protocol, provider, harness, snapshot, and external-state failures become `STRUCTURAL_FAILURE`; they cannot become model-stuck labels.

The v0 and v1 sources remain byte-for-byte unchanged. V2, its four candidate configurations, the selection procedure, and every gate threshold were committed in `a291cf6` before scoring the real development evidence.

## Task-grouped development result

The evaluation used 58 natural verifier-valid Flash and Qwen trajectories across 35 development tasks. Twelve structural or protocol trajectories were excluded from recovery labels. Candidate selection and evaluation used five task-grouped folds.

| Tier | Checkpoints | Tasks | Natural continuation completion |
|---|---:|---:|---:|
| Healthy | 58 | 32 | 26/58 (44.8%) |
| Needs review | 39 | 23 | 17/39 (43.6%) |
| Confirmed stuck | 16 | 13 | 4/16 (25.0%) |

Healthy recovery exceeded confirmed-stuck recovery by 19.8 percentage points, narrowly below the frozen 20-point requirement. The task-clustered 95% interval was −0.8 to 37.3 points, so it included zero. The direction remained positive for both models and after excluding any one task, but those robustness checks do not override the failed primary gates.

The broad review tier itself barely separated from healthy continuation. It should be interpreted as a coverage mechanism, not a reliable intervention decision.

## Exact prior-scout replay

The selected configuration was replayed over the 84 frozen prior scout slots using their complete pre-outcome observations. Twenty-two externally unreproducible slots were correctly classified as structural.

| Tier | Checkpoint hits | Tasks | Flash / Qwen |
|---|---:|---:|---:|
| Healthy | 18 | 12 | 7 / 11 |
| Needs review | 7 | 6 | 4 / 3 |
| Confirmed stuck | 0 | 0 | 0 / 0 |

The frozen thresholds required at least 12 review hits across eight tasks and six confirmations across four tasks and both models. Both gates failed.

The absence of confirmations is informative about the evidence, not only the rule. Many prior scouts were configured to stop at the first v0 checkpoint. A two-tier detector needs at least one later observation to determine whether an early review signal persists, but those stopped trajectories do not contain it. Existing full-observation records therefore cannot establish confirmation yield.

## Gate result

The following gates passed:

- direction was positive for Flash and Qwen;
- no single task drove the difference;
- healthy checkpoint coverage exceeded 12;
- structural failures remained separate; and
- leakage controls passed.

The following gates failed:

- 19.8-point recovery separation was below 20 points;
- the clustered interval included zero;
- review replay coverage was 7 checkpoints across 6 tasks, below 12 across 8;
- confirmation replay coverage was zero; and
- no replayed confirmation could demonstrate two remaining turns.

Thresholds were not changed after scoring.

## Spend and execution

- Incremental OpenRouter spend: **$0.00**
- Incremental Daytona spend: **$0.00**
- Model calls: **0**
- New tasks inspected: **0**
- Intervention outcomes: **0**
- Training runs: **0**

Tracked project OpenRouter spend remains **$49.257980597**.

## What should happen next

Do not train an intervention policy yet. We still lack matched intervention labels, and the full-observation evidence is truncated before the second tier can be evaluated.

The next experiment should be a small continuation-only calibration run whose scouts do **not** stop at `NEEDS_REVIEW`. They should save the review snapshot, continue the same model, save any later confirmation snapshot, and then continue naturally to a verifier outcome. That would measure review-to-confirmation transition rates and continuation recovery on one complete live schema.

This is a material design change and should be frozen as a new goal before any paid calls. It should not retroactively change this milestone’s failed decision.

Machine-readable results are in `artifacts/official/two-tier-detector-v2/development-report-v0.json`.
