# Research program

This document records the evidence behind the Long-Horizon Supervisor. The
[README](../README.md) is the executive summary; the
[roadmap](roadmap.md) contains only unfinished work.

## Research question

Can a harness-neutral supervisor increase verified completion on long-horizon
coding tasks, then minimize cost and tokens among similarly successful policies?

The program separates two decisions:

1. **Portfolio routing:** which model should try next after an external verifier
   rejects a completed run?
2. **Live intervention:** when should the system change course before the current
   run exhausts its budget?

The first is supported by held-out evidence. The second is not yet established.

## 1. Held-out model portfolio

Eighteen sealed Terminal-Bench Pro tasks were run independently through four
frozen model routes, producing 72 learning-valid outcomes. Tasks, model versions,
tool limits, route order, stopping logic, and verifiers were fixed before the
evaluation.

| Policy | Verified tasks | Replayed model cost |
|---|---:|---:|
| Flash | 6/18 | $0.1208 |
| Best single model: Kimi | 7/18 | $4.0559 |
| Flash → Qwen | 9/18 | $1.1187 |
| Flash → Qwen → GLM | 10/18 | $2.3749 |
| Flash → Qwen → GLM → Kimi | **12/18** | **$3.8475** |

The completion-first portfolio gained five tasks over the best single model.
The two-model route gained two tasks at 72% lower replayed cost than Kimi alone.
Ten tasks discriminated between routes; six defeated every tested model, which
bounds any chooser over these observed routes at 12/18.

The exact provider spend for collecting the panel was $4.1006. Replayed costs
use comparable catalog prices and are not the provider invoice.

Evidence: [held-out methodology and scorecard](gate8-wave3-18-task-final-scorecard.md).

## 2. Portfolio replication

Across 20 confirmatory task-replication units, Flash → Qwen completed 13/20;
two independent Flash attempts completed 6/20. The observed gain was 35
percentage points, although the task-clustered uncertainty interval touched
zero. A five-model cascade completed 16/20 units. A small 9B model added no
unique coverage but sometimes improved the cost frontier by solving work before
expensive routes ran.

Interpretation: model complementarity repeated, but the causal effect of this
specific order remains uncertain.

Evidence: [replication study](swiss-cheese-replication-v0-final.md).

## 3. Learned task-start baselines

A development-only task router used public task text and metadata to order the
four routes. It matched the fixed cascade on held-out tasks but did not improve
the order. A continuation-risk model produced only a small calibration gain
over constant prevalence and weak ranking performance.

Interpretation: the available clean-start data justifies transparent static
rules, not a claim that learning improves task-start selection.

Evidence: [initial learned baselines](initial-supervisor-policy-v0.md).

## 4. First matched-state intervention pilot

Four suspected-stuck states were saved and replayed from identical workspaces.

| Action | Verified tasks |
|---|---:|
| Continue current model | 1/4 |
| Switch Flash ↔ Qwen with state | 0/4 |
| Switch to Kimi with state | **2/4** |
| Restart Kimi clean | 1/4 |

Kimi produced two unique rescues, suggesting that some states need additional
reasoning rather than a merely different model. The sample was too small for
training or deployment.

Evidence: [first intervention pilot](stuck-intervention-pilot-v0-final.md).

## 5. Confirmatory intervention result

A frozen successor precommitted the detector, task order, actions, budget, and
analysis. Its 21-task pool produced only three stuck and three healthy groups
across four tasks. Continue and Kimi each completed one of three stuck groups,
and they solved the same group. Kimi therefore had no unique rescue over
continuing.

Interpretation: the result was inconclusive; the bottleneck was detector
coverage, not policy fitting.

Evidence: [confirmatory result](stuck-confirmatory-v1-final.md).

## 6. Detector development

### Detector v1

Detector v1 stopped treating unchanged workspaces during normal inspection as
stuck. In task-grouped development folds, projected stuck checkpoints recovered
21.3 percentage points less often than healthy checkpoints, with an uncertainty
interval above zero. Exact replay, however, found only three stuck hits and
reduced total checkpoint yield. The precommitted gate failed and paid collection
was skipped.

Evidence: [checkpoint coverage v1](checkpoint-coverage-v1-final.md).

### Two-tier detector

The next detector separated broad `NEEDS_REVIEW` states from stricter
`CONFIRMED_STUCK` states. Offline, confirmed states recovered 25.0% versus 44.8%
for healthy controls, missing the required 20-point gap by 0.2 points; the
uncertainty interval included zero. Historical scouts often stopped before a
second-tier decision could occur, so exact replay produced no confirmations.

Evidence: [two-tier detector](two-tier-detector-v2-final.md).

### Natural-continuation calibration v3

Flash and Qwen ran naturally on 24 preselected tasks. Among 48 trajectories, 18
were learning-valid and all 28 counted snapshots rehydrated exactly. Healthy
checkpoints completed 43.8% of the time, review checkpoints 30.0%, and
confirmed-stuck checkpoints 0.0%. The healthy-minus-confirmed gap was 43.8
points with a positive task-clustered interval and positive directions for both
models.

The gate still failed: only two confirmed checkpoints from two independent
tasks were observed.

Evidence: [continuation calibration v3](continuation-calibration-v3-final.md).

### Independent calibration v4

A fresh 48-trajectory successor filled the aggregate count deficit. Across v3
and v4, confirmed checkpoints recovered 14.3% versus 47.9% for healthy controls,
a 33.6-point gap with a positive task-clustered interval. Both model-specific
directions were positive.

V4 remained invalid because two of 65 fresh checkpoint replays changed Git
object permission bits during safe archive extraction. The transport defect was
fixed, but the frozen result was preserved as failed rather than reinterpreted.

Evidence: [continuation calibration v4](continuation-calibration-v4-final.md).

## 7. Current independent calibration

The active calibration changes only the corrected checkpoint transport and
evaluation provenance. It uses a fresh, non-overlapping task cohort and the
already-frozen Flash/Qwen routes, detector, sampling rule, thresholds, budget,
and analysis. Candidate checkpoints must rehydrate exactly, and structural or
infrastructure failures cannot become model-stuck labels.

Its result will be published whether the gate passes or fails. No intervention
experiment begins unless the continuation-only gate passes unchanged.

## Evidence boundaries

| Claim | Status |
|---|---|
| Different models have complementary coverage | Supported |
| A verifier-gated clean-start portfolio improves completion | Supported |
| Model quality forms a universal ladder | Rejected by observed jaggedness |
| A learned task-start router beats the fixed order | Not supported |
| Repeated failure signals identify lower-recovery states | Promising |
| Kimi should always replace a stuck model | Not supported |
| A learned live supervisor is ready to deploy | Not yet |

## Experimental principles

- Freeze task selection, actions, budgets, analysis, and gates before outcomes.
- Group uncertainty and data splits by task, not checkpoint.
- Use only information observable at decision time as model input.
- Preserve external state and verify exact rehydration before comparing actions.
- Keep infrastructure, protocol, and model failures distinct.
- Retain negative results and do not tune on held-out labels.
- Optimize completion first; compare cost and tokens among comparable policies.
