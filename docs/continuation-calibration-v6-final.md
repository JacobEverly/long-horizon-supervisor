# Continuation calibration v6

## Problem

Before training a live intervention policy, we need evidence that the
supervisor's `CONFIRMED_STUCK` state identifies runs that recover less often
when the current model simply continues.

## Frozen test

Flash and Qwen ran naturally on a fresh, non-overlapping task pool. The run
saved decision-time checkpoints, rehydrated them in a corrected Daytona
transport, and let each trajectory continue to the same external verifier.
The detector, task cohort, routes, thresholds, budget, and analysis were frozen
before collection. Infrastructure and protocol failures were excluded from
model-stuck labels.

## Result

The run produced **26 trajectories**: **17 learning-valid natural
continuations** and **9 structural failures**. All **28 counted checkpoints**
passed the transport-fidelity check. Incremental provider spend was
**$0.9829**; no Daytona environments remained.

| Detector tier | Checkpoints | Tasks | Completed | Natural recovery |
|---|---:|---:|---:|---:|
| Healthy | 17 | 9 | 10 | 58.8% |
| Needs review | 10 | 8 | 6 | 60.0% |
| Confirmed stuck | 1 | 1 | 0 | 0.0% |

The healthy-versus-confirmed gap is directionally large, but the gate failed:
there was only one confirmed-stuck checkpoint, no adequate review/confirmed
coverage, no two-model directional estimate, and the confirmed tier depended
on one task. The correct conclusion is **not ready to train**.

## What this proves

- The corrected checkpoint transport and provenance controls work.
- The detector can produce a rare confirmed-stuck state on a natural run.
- The current evidence is insufficient to estimate intervention value.

It does **not** prove that switching or escalation beats continuation, or that
the detector is ready for deployment. The next experiment is one minimal,
independently frozen coverage revision; training remains gated on matched
counterfactual outcomes.
