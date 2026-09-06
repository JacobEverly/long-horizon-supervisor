# Continuation calibration v3: final result

## Decision

**NOT READY — continuation calibration gate failed.**

The detector produced a large and statistically positive recovery separation,
but only two independent `CONFIRMED_STUCK` tasks. That is not enough support for
an intervention experiment or a learned mid-run policy.

This is a data-sufficiency failure, not evidence that the product hypothesis is
false.

## Product question

The supervisor should increase verified completion without paying for a
frontier model on every turn. Before testing any switch or escalation, it must
answer a more basic question:

> Can information visible during a run distinguish an agent that should keep
> working from one that is unlikely to recover naturally?

That is the important capability demonstrated here. Daytona, Harbor, and the
provider APIs are replaceable execution tools; the durable work is specifying
the decision, collecting causal evidence, separating infrastructure from model
behavior, and enforcing a stopping rule.

## Frozen design

The v3 manifest was frozen before model outcomes with SHA-256:

`ed056a2a9fbaad3354a43ee2113a49c877b0d5c9bbbe35069426787dc996964e`

The run used:

- 24 preselected Terminal-Bench Pro tasks in three eight-task tranches;
- two exact routes: DeepSeek V4 Flash 0731 and Qwen 3.8 27B;
- the unchanged two-tier detector `review-t5-confirm-t6-w2-e2`;
- natural continuation only—no switching, escalation, or early stop based on a
  detector label;
- state capture at `HEALTHY`, `NEEDS_REVIEW`, and `CONFIRMED_STUCK`;
- provider-free rehydration of every counted snapshot in a fresh environment;
- task-clustered uncertainty and coverage gates frozen before outcomes; and
- per-trial, tranche, key, and project spend ceilings.

Earlier v0–v2 artifacts and thresholds were not changed.

## Results

| State at checkpoint | Checkpoints | Tasks | Flash / Qwen | Natural completions | Completion rate |
|---|---:|---:|---:|---:|---:|
| `HEALTHY` | 16 | 11 | 6 / 10 | 7 | 43.75% |
| `NEEDS_REVIEW` | 10 | 8 | 3 / 7 | 3 | 30.00% |
| `CONFIRMED_STUCK` | 2 | 2 | 1 / 1 | 0 | 0.00% |

Additional results:

- 48 terminal trajectories were collected;
- 18 were learning-valid and 30 were structural or otherwise invalid;
- all 28 counted snapshot replays passed in fresh environments;
- healthy-minus-confirmed recovery was **43.75 percentage points**;
- the task-clustered 95% interval was **[14.29, 70.59] points**;
- the recovery direction was positive for both Flash and Qwen;
- removing any one checkpoint-bearing task left a positive minimum difference;
  and
- exact incremental OpenRouter spend was **$1.428196097**, bringing tracked
  project spend to **$50.736977389**.

No Daytona environments remained after cleanup. The cleanup ledger records one
benign not-found response for a sandbox that had already disappeared, with the
final remaining-environment set empty.

## Why the gate failed

The effect-size gates improved and passed, but evidence coverage did not:

- `HEALTHY` coverage passed;
- `NEEDS_REVIEW` had 10 checkpoints, below the frozen minimum of 12;
- `CONFIRMED_STUCK` had 2 checkpoints from 2 tasks, below the frozen minimum of
  6 checkpoints from 4 tasks; and
- each stuck task contributed half of the confirmed class, above the frozen
  25% maximum single-task share.

The correct conclusion is therefore not “train a classifier on the observed
gap.” With only two stuck tasks, a classifier could learn task identity or
incidental phrasing rather than a reusable intervention signal.

## What we learned

Several `NEEDS_REVIEW` trajectories later returned to `HEALTHY`. That supports
the two-tier product design: a warning should not automatically trigger an
expensive switch. The supervisor needs temporal confirmation.

The dominant bottleneck is now data collection. Structural or protocol-invalid
execution removed 30 of 48 trajectories. More broad random collection would be
inefficient; the next design should use only public task-package information to
preselect fresh hard tasks whose state can be reproduced without unmanaged
services or process memory.

The task selector must remain outcome-blind. It may use public difficulty,
category, and static checkpoint-compatibility checks, but not v3 success,
failure, or detector labels to select new task IDs. Detector thresholds and
analysis gates remain frozen.

## Next decision

Run a fresh targeted calibration successor whose only purpose is to fill the
confirmed-stuck and review coverage deficits. Stop if it cannot do so within
its frozen task and spend ceilings.

If the aggregate continuation gate passes, freeze the checkpoint bank and run
matched interventions from identical confirmed-stuck states:

1. continue the same model;
2. switch to the complementary model with preserved state;
3. escalate to a stronger reasoning model with preserved state; and
4. clean-restart the stronger model.

Only those matched counterfactual outcomes can justify training the first
mid-run intervention policy.

## Public artifacts

- Frozen manifest:
  `artifacts/official/two-tier-continuation-calibration-v3/frozen-manifest-v3.json`
- Final calibration report:
  `artifacts/official/two-tier-continuation-calibration-v3/calibration-report-v3.json`
- Exact execution and cleanup ledger:
  `artifacts/official/two-tier-continuation-calibration-v3/execution-ledger-v3.json`
- Aggregate snapshot fidelity report:
  `artifacts/official/two-tier-continuation-calibration-v3/snapshot-fidelity-v3.json`

Raw trajectories, terminal content, credentials, private benchmark material,
and live provider logs remain excluded from the public repository.
