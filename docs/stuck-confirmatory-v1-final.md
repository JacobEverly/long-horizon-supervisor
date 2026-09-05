# INCONCLUSIVE — improve coverage and repeat

## Executive result

The confirmatory matched-state experiment did **not** establish that
`suspected_stuck_v0` identifies states with lower natural recovery, and it did
**not** establish that escalating the preserved state to Kimi improves
completion over continuing.

The frozen 21-task development pool was exhausted after producing 6 complete
matched groups and 24 valid branch outcomes, far below the preregistered target
of 24 groups and 96 outcomes across at least 8 tasks. Only 4 tasks contributed
accepted groups, and only 2 contributed suspected-stuck groups. The proper
decision is therefore inconclusive rather than confirmed or rejected.

Within that limited sample:

- continuing completed 1/3 suspected-stuck groups and 1/3 healthy groups;
- preserved-state Kimi completed 1/3 stuck groups and 0/3 healthy groups;
- the Kimi success was the same group that continuing completed, so Kimi
  produced **zero unique rescues over continuing**;
- preserved-state Flash/Qwen switching completed 0/3 stuck groups and 0/3
  healthy groups; and
- clean-start Kimi completed 0/3 groups of either kind.

The product implication is straightforward: **do not train or deploy a learned
live intervention policy from this evidence.** Keep the verifier-gated
clean-start cascade as the supported behavior. Before repeating live
intervention, first prove that a fresh, outcome-blind task source can reliably
yield enough reproducible checkpoints.

## What was tested

- Detector: `suspected_stuck_v0`, unchanged and hash-verified at
  `c3319c93d823455076fd294ac16e28748a2b2ebcab10e1b81760d174088f4ffe`.
- Base models: `deepseek/deepseek-v4-flash-0731` and `qwen/qwen3.8-27b`.
- Reasoning escalation: `moonshotai/kimi-k3`.
- Checkpoints: the first valid `SUSPECTED_STUCK` trigger or turn 4 while still
  `HEALTHY`.
- Matched actions from one normalized snapshot:
  continue the current model, switch Flash/Qwen with state, switch to Kimi
  with state, or restart Kimi clean.
- Completion: the unchanged external Harbor verifier.
- Accepted evidence: exactly four valid arms per group with equal remaining
  turn, token, wall-time, and spend limits.
- Stopping rule: run the frozen pool in order until the target was met or the
  pool was exhausted.

The final state was `frozen_pool_exhausted_before_dataset_target`: 84 frozen
scouting slots completed, representing two base models and two checkpoint kinds
for each of 21 tasks. The runner recorded 114 attempts. Invalid infrastructure
and protocol attempts remained visible and were excluded rather than converted
to model failures.

## Coverage achieved

| Requirement | Target | Observed | Met |
|---|---:|---:|:---:|
| Suspected-stuck groups | 12 | 3 | No |
| Healthy groups | 12 | 3 | No |
| Valid branch outcomes | 96 | 24 | No |
| Unique tasks overall | at least 8 | 4 | No |
| Unique tasks in stuck groups | at least 4 | 2 | No |
| Unique tasks in healthy groups | at least 4 | 3 | No |
| Flash groups per kind | at least 4 | 1 stuck / 1 healthy | No |
| Qwen groups per kind | at least 4 | 2 stuck / 2 healthy | No |
| Groups from one task and kind | at most 2 | 2 | Yes |

The four contributing tasks were:

- `bash-ddos-traffic-analyzer`;
- `tabular-q-learning-mountaincar-agent`;
- `secure-model-pipeline-deployment`; and
- `cube-grid-cross-expansion`.

`cube-grid-cross-expansion` supplied two of the three stuck groups. This is why
the apparent sample size of three cannot be treated as three fully independent
task replications.

## Why the pool was sparse

The execution state records 79 exclusion or ineligibility entries:

| Recorded reason | Count |
|---|---:|
| Requested checkpoint did not occur | 49 |
| Unmanaged process lacked a frozen rehydration recipe | 22 |
| One or more branches remained invalid after the frozen retry | 5 |
| Checkpoint had no remaining agent turns | 1 |
| Predeclared per-task group cap reached | 1 |
| Checkpoint recovered by the pre-outcome branching amendment | 1 |

This is the main experimental result. The bottleneck was not budget or model
availability; it was the ability to obtain intervention-ready, reproducible
states under the frozen detector and snapshot contract.

## Detector result

| Continuation condition | Completed | Rate | Task-bootstrap 95% interval | Cost |
|---|---:|---:|---:|---:|
| Suspected stuck | 1/3 | 33.3% | 0% to 100% | $0.084386 |
| Healthy at turn 4 | 1/3 | 33.3% | 0% to 100% | $0.132110 |

Healthy-minus-stuck recovery was **0 percentage points**, with a task-clustered
95% interval from **-100 to +50 points**. The preregistered detector gate
required at least +20 points, an interval excluding zero, and robustness to
leaving out a task and separating the two base models. It passed none of those
requirements.

One of three stuck checkpoints recovered when the current model continued, so
the directly observed false-positive stuck-trigger rate was 33.3%. With only
three groups, that is a warning rather than a stable rate estimate.

## Intervention results

### Suspected-stuck checkpoints

| Action | Completed | Cost | Input | Output | Cached | Reasoning | Sequential time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Continue current state | 1/3 | $0.084386 | 137,046 | 33,175 | 92,912 | 27,057 | 7.2 min |
| Switch Flash/Qwen with state | 0/3 | $0.030415 | 111,705 | 21,352 | 88,256 | 11,193 | 6.1 min |
| Switch to Kimi with state | 1/3 | $0.211709 | 63,132 | 11,775 | 30,464 | 7,148 | 10.7 min |
| Restart Kimi clean | 0/3 | $0.169620 | 53,310 | 5,923 | 36,544 | 775 | 6.7 min |

### Healthy turn-4 checkpoints

| Action | Completed | Cost | Input | Output | Cached | Reasoning | Sequential time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Continue current state | 1/3 | $0.132110 | 146,549 | 44,646 | 93,248 | 34,266 | 14.6 min |
| Switch Flash/Qwen with state | 0/3 | $0.029431 | 85,627 | 22,269 | 62,016 | 14,129 | 7.1 min |
| Switch to Kimi with state | 0/3 | $0.293988 | 90,713 | 15,015 | 57,408 | 7,802 | 16.5 min |
| Restart Kimi clean | 0/3 | $0.380330 | 113,517 | 16,849 | 79,040 | 7,482 | 16.4 min |

All 24 accepted outcomes had zero provider errors, protocol errors, and
state-transfer failures. Verifier-confirmed progress equals verified completion
because these task verifiers returned binary reward.

## Predeclared comparisons

### Kimi preserved state versus continuing

At suspected-stuck checkpoints, both policies completed 1/3: a 0-point
difference with a task-bootstrap interval of 0 to 0 points. They completed the
same group, so Kimi had zero rescues and zero harms relative to continuing.
Kimi cost $0.211709 versus $0.084386 for continuation.

At healthy checkpoints, Kimi completed 0/3 versus 1/3 for continuing: -33.3
points, with an interval from -100 to 0 points, zero rescues, and one harm.

The difference-in-differences was +33.3 points, with an interval from 0 to 100
points. That positive number is **not evidence of a stuck-state rescue**: it is
created by Kimi tying continuation at stuck states while harming a healthy
state. The stuck-state gain itself was zero.

### Kimi preserved state versus clean Kimi

At stuck checkpoints, preserved-state Kimi completed 1/3 while clean Kimi
completed 0/3: +33.3 points, interval 0 to 100 points, one rescue, and no harms.
That supports preserving useful accumulated work over discarding it, but it is
only one task and does not establish that Kimi beats continuation.

At healthy checkpoints, neither Kimi arm completed a group, so preserved state
had zero rescues and zero harms relative to clean Kimi.

### Kimi versus the cheaper Flash/Qwen switch

At stuck checkpoints, Kimi completed 1/3 versus 0/3 for the cheaper switch:
+33.3 points, interval 0 to 100 points, one rescue, and no harms. The
preregistered Kimi gate required at least two unique rescues over continuing,
not merely over another failing intervention, so it did not pass.

At healthy checkpoints, both Kimi and the cheaper switch completed 0/3, with no
rescue or harm between them.

### Unique rescues and harms versus continuing

| Intervention | Stuck rescues | Stuck harms | Healthy rescues | Healthy harms |
|---|---:|---:|---:|---:|
| Switch Flash/Qwen with state | 0 | 1 | 0 | 1 |
| Switch to Kimi with state | 0 | 0 | 0 | 1 |
| Restart Kimi clean | 0 | 1 | 0 | 1 |

A rescue means the intervention completed while the identical continuation did
not; a harm means continuation completed while the intervention did not. This
matched definition is stricter and more useful than comparing aggregate rates
from unrelated trajectories.

### Cheap model switching and directionality

The Flash/Qwen switch had zero rescues over continuing at stuck checkpoints
and harmed the one stuck group that continuation solved. It repeated that
pattern at healthy checkpoints: zero rescues and one harm.

- Flash to Qwen was observed once at a stuck checkpoint: neither the switch nor
  continuation completed.
- Qwen to Flash was observed twice: the switch completed 0/2 while continuation
  completed 1/2, producing one harm and no rescue.

Neither direction met the preregistered requirement of two unique rescues, so
there is no confirmatory evidence for a live Flash/Qwen jaggedness route.
Including healthy checkpoints gives the same qualitative result: Flash to Qwen
completed 0/2 versus 0/2 for continuation, while Qwen to Flash completed 0/4
versus 2/4 for continuation.

### Success-versus-cost frontier

At stuck checkpoints, continuation and the cheaper Flash/Qwen switch are the
two mathematically non-dominated observed points: switching cost less but
completed nothing; continuation cost more and completed one group. Because the
product objective prioritizes verified completion, continuation is the useful
operating point in this sample. Kimi is dominated by continuation because it
completed the same group at higher cost, and clean Kimi completed none at still
higher cost than the value switch.

## Group-level outcomes

| Task and checkpoint | Continue | Value state | Kimi state | Kimi clean |
|---|:---:|:---:|:---:|:---:|
| Bash DDoS, Qwen stuck t3 | Pass | Fail | Pass | Fail |
| MountainCar, Qwen healthy t4 | Pass | Fail | Fail | Fail |
| Secure pipeline, Qwen healthy t4 | Fail | Fail | Fail | Fail |
| Cube grid, Flash stuck t8 | Fail | Fail | Fail | Fail |
| Cube grid, Flash healthy t4 | Fail | Fail | Fail | Fail |
| Cube grid, Qwen stuck t6 | Fail | Fail | Fail | Fail |

This table exposes the key fact hidden by aggregate Kimi rates: its only stuck
completion was not a unique rescue. The current model had already recovered the
same state.

## Frozen integrity and amendments

The original manifest was frozen before paid collection. Three amendments were
made only to correct or shorten orchestration behavior:

1. Explicit four-way snapshot branching replaced reuse of a post-checkpoint
   scouting tail. It was frozen after $0.009085 and before any accepted group.
2. Stuck scouts stopped after sealing a checkpoint rather than spending on an
   outcome that was not part of the four-arm comparison.
3. Healthy-only scouts stopped once the frozen turn-4 checkpoint could no
   longer be collected.

The amendments did not change the detector, model set, task order, checkpoint
timing, branch budgets, target, analysis, or decision gates. Each amendment has
a frozen manifest hash and a pre-amendment execution-state backup. The final
manifest is `frozen-manifest-v3.json`, SHA-256
`1617ad8e827caa33348749d16cbf7c1eedd6895c91cb14cb59aa9368c5c1700c`.
The frozen analysis code also retained its preregistered hash.

Snapshot fidelity passed for files, permissions, Git state, public-test state,
deterministic workspace digest, counter metadata, service-recipe rehydration,
and isolation across three clones. The archive mechanism preserves workspace
state, not live process memory; tasks with unmanaged required processes were
correctly ineligible.

## Spend and cleanup

- Exact dedicated-key OpenRouter spend: **$3.370920394**.
- Authorized confirmatory ceiling: **$20.00**.
- Dedicated-key remaining limit at final reconciliation: **$16.629079606**.
- Accepted matched-branch cost: **$1.331989007**.
- Scouting, invalid attempts, and recovery overhead: **$2.038931387**.
- Prior supervisor-project OpenRouter spend: **$45.887060203**.
- Supervisor-project OpenRouter spend after this experiment:
  **$49.257980597**.
- Account-wide final usage was separately reconciled in the private execution
  ledger. It includes activity outside this project; because an exact
  account-wide pre-run snapshot was not preserved, no account-wide experiment
  delta is claimed or published.
- Daytona charges: unavailable through the installed SDK and excluded from the
  OpenRouter totals.

At final reconciliation, Daytona listed zero sandboxes and no Harbor,
Switchyard, or confirmatory benchmark process remained. The runner recorded one
delete-while-state-changing conflict during immediate shutdown; the sandbox
finished disappearing before the authoritative recheck. The temporary
OpenRouter key file was deleted after the final usage query.

## Decision gates

| Gate | Result | Why |
|---|:---:|---|
| Dataset target | Fail | 6/24 groups, 24/96 outcomes, 4/8 tasks |
| Detector confirmation | Fail | Healthy-minus-stuck recovery was 0 points |
| Kimi intervention | Fail | 0-point stuck gain and zero unique rescues over continuation |
| Flash/Qwen jaggedness route | Fail | Neither direction produced a rescue |
| Proceed to training-sized collection | **No** | Requires detector and intervention gates |

The formal decision remains **INCONCLUSIVE — improve coverage and repeat**
because the frozen pool exhausted before the target. The observed point
estimates themselves are unfavorable, but the task-independent sample is too
small to convert that into a broad rejection of all live intervention.

## Logical next milestone

Do not launch training or another full four-arm replication yet. Run a
**checkpoint-coverage feasibility gate** on fresh public development tasks:

1. Statically preflight every task for workspace-only state or a complete,
   public service-rehydration recipe. Exclude unmanaged-process tasks before
   model spend.
2. Freeze a new `suspected_stuck_v1` only after evaluating detector mechanics
   on a separate development corpus. Do not tune it on the next confirmatory
   outcomes.
3. Run base-model scouts first and bank sealed, reproducible snapshots without
   inspecting intervention outcomes.
4. Require at least 12 stuck and 12 healthy checkpoints across at least 8 tasks,
   with balanced Flash/Qwen representation, **before** launching the four arms.
5. If that coverage gate passes, execute the same matched interventions and
   preserve the same external verifier. If it fails, deprioritize live
   switching and rely on verifier-gated clean-start cascades.

This sequencing makes the next dollar answer the causal question. It avoids
spending most of the budget discovering that a task cannot produce or preserve
the required checkpoint.

## Authoritative artifacts

- Frozen original manifest:
  `artifacts/official/stuck-confirmatory-v1/frozen-manifest-v0.json`
- Final frozen amendment manifest:
  `artifacts/official/stuck-confirmatory-v1/frozen-manifest-v3.json`
- Snapshot fidelity:
  `artifacts/official/stuck-confirmatory-v1/snapshot-fidelity-v0.json`
- Valid matched outcomes:
  `artifacts/official/stuck-confirmatory-v1/matched-outcomes-v0.jsonl`
- Exact execution ledger:
  `artifacts/official/stuck-confirmatory-v1/execution-ledger-v0.json`
- Frozen fit-free analysis:
  `artifacts/official/stuck-confirmatory-v1/confirmatory-analysis-v0.json`

Raw paid-run artifacts remain private and are excluded from the public
repository. This aggregate report contains the decision and enough metrics to
audit the claims without publishing model reasoning, provider data, or hidden
benchmark material.
