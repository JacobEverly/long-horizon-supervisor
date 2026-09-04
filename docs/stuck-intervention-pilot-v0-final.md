# Matched-state stuck-detection and intervention pilot: final decision

## Executive result

This pilot is a **go for a small confirmatory matched-state replication, but a
no-go for training or deploying a learned supervisor yet**.

The strongest intervention signal was reasoning escalation with preserved
state. At four `SUSPECTED_STUCK` checkpoints, switching the exact saved state
to Kimi completed 2 of 4 tasks (50%), while continuing the current model
completed 1 of 4 (25%) and switching between Flash and Qwen completed 0 of 4
(0%). Kimi produced two unique rescues relative to each alternative. It cost
$0.2279 across the four branches, compared with $0.1195 for continuing and
$0.0874 for the cheaper cross-model switch.

That is useful evidence for the product hypothesis—some stuck states benefit
from more reasoning, not merely another model—but it is not a validated policy.
Kimi also harmed one state that the original Qwen run recovered, the
task-clustered intervals are extremely wide, and only three independent tasks
contributed stuck groups.

The detector result is weaker. Continuing recovered from 1 of 4 stuck
checkpoints (25%) versus 1 of 2 healthy checkpoints (50%), an observed
25-point separation with a task-bootstrap 95% interval of 0 to 50 points. One
of the four stuck triggers was therefore a directly observed false positive.
More importantly, both healthy groups came from the same task. The frozen
eight-task pool was exhausted before four healthy controls could be collected,
so the detector's specificity and the trigger-versus-fixed-turn interaction
are not established.

## What was actually tested

- Detector: frozen, outcome-blind `suspected_stuck_v0`.
- Models: DeepSeek V4 Flash 0731 and Qwen3.8 27B as base/value models; Kimi K3
  as the high-reasoning escalation.
- Panel: a predeclared ordered pool of eight hard development tasks selected
  from public metadata only.
- Accepted evidence: 6 complete matched groups x 6 arms = 36 valid outcomes.
- Checkpoints: 4 suspected-stuck groups and 2 healthy turn-4 groups.
- Base representation: one Flash and three Qwen stuck groups; one Flash and
  one Qwen healthy group.
- Independent task coverage: three tasks for stuck groups and one task for
  healthy groups.
- Completion: the same external Harbor verifier with reward at least 1.
- Valid matched outcomes: 8 of 36 completed.
- Invalid infrastructure or protocol attempts were recorded separately and
  were not counted as model failures.

The experiment stopped with the predeclared reason
`frozen_pool_exhausted_before_both_group_targets`. It met the minimum of four
complete stuck groups and collected valid healthy negative controls, but it
did not reach the target of four healthy groups. No task, detector threshold,
or branch setting was changed to fill the missing cells after outcomes were
visible.

## The frozen detector

`suspected_stuck_v0` receives only normalized information observable during
the run. It triggers immediately on a protocol failure or a clear repeated
action/error loop, or after at least two independent stuck signals persist
across two consecutive observed turns.

Its signals include repeated errors, unchanged failing public tests, unchanged
execution state, repeated equivalent commands, workspace-state cycles,
consecutive turns without meaningful progress, substantial budget use without
new evidence, and inability to state an actionable next step. A skipped
command-bearing observation breaks the consecutive-evidence chain.

Meaningful progress is explicitly limited to public evidence: more tests
passing, fewer distinct test failures, a new successful execution milestone,
resolution of an observed error, or creation of a required artifact. A file
change or additional token use alone is not progress. The schema excludes
hidden verifier output, future outcomes, final success, private reasoning, and
sibling-branch outcomes.

## Snapshot fidelity and harness boundary

The normalized contract remains:

`observe -> snapshot -> decide -> act`

The supervisor sees normalized observations and emits normalized actions. The
harness adapter owns Harbor and Daytona behavior.

Local and Daytona fidelity checks passed for workspace files, permissions, Git
state, public-test state, deterministic workspace digests, counters, and clone
isolation across three independently modified clones. No accepted preserved-
state branch had a state-transfer failure.

Daytona did not provide timely reusable VM snapshots or live sandbox forks in
this runtime. The narrow fallback is a permission-preserving workspace archive
rehydrated into a fresh sandbox plus a public, reasoning-free handoff. This
does **not** preserve live process memory. A task with a declared public service
can use a frozen restart recipe; a checkpoint with an unmanaged relevant
process is structurally ineligible. Clean restarts are labeled separately and
are never presented as preserved-state switches.

## Matched intervention results at stuck states

| Action from the identical stuck point | Completed | Cost | Input tokens | Output tokens | Cached tokens | Reasoning tokens | Sequential time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Continue current state | 1/4 | $0.1195 | 243,763 | 27,358 | 143,552 | 20,772 | 18.3 min |
| Restart current model clean | 0/4 | $0.1917 | 109,397 | 55,410 | 41,088 | 48,974 | 24.0 min |
| Switch Flash/Qwen with state | 0/4 | $0.0874 | 113,801 | 28,460 | 29,824 | 19,245 | 11.2 min |
| Restart destination value model clean | 1/4 | $0.0515 | 110,883 | 22,048 | 51,264 | 12,788 | 10.5 min |
| Switch to Kimi with state | 2/4 | $0.2279 | 92,822 | 17,377 | 46,912 | 8,424 | 14.3 min |
| Restart Kimi clean | 1/4 | $0.6107 | 123,541 | 39,347 | 81,664 | 24,089 | 18.0 min |

All 24 accepted stuck-state branches had zero state-transfer failures, protocol
errors, and provider errors. Their verifier-confirmed-progress count equals
their verified-completion count because the external verifier was binary on
these tasks.

## The eight predeclared comparisons

### 1. Continue at stuck versus healthy

The current model recovered from 1/4 stuck states (25%) and 1/2 healthy states
(50%). The healthy-minus-stuck difference was +25 points; the task-bootstrap
95% interval was 0 to +50 points. This is directionally sensible but too little
independent healthy evidence to validate the detector.

### 2. Cross-model preserved state versus continuing

Flash/Qwen switching completed 0/4; continuing completed 1/4. The matched
difference was -25 points (task-bootstrap interval -100 to 0), with zero
rescues and one harm. Cheap model diversity alone was not a useful stuck
intervention in this sample.

### 3. Kimi preserved state versus continuing

Kimi completed 2/4 versus 1/4 for continuing: +25 points, two rescues, and one
harm. The task-bootstrap interval was -100 to +100 points. The point estimate
supports escalation, but the uncertainty prohibits a general claim.

### 4. Kimi escalation versus cheaper cross-model switching

Kimi completed 2/4 versus 0/4 for Flash/Qwen switching: +50 points, two
rescues, and no harms relative to that arm. The task-bootstrap interval was 0
to +100 points. Kimi cost $0.1404 more across the four matched branches. This
is the pilot's strongest result and earns Kimi a **provisional escalation
candidate** for confirmation, not an automatic production role.

### 5. Preserved-state switch versus the destination model clean

The Flash/Qwen preserved-state switch completed 0/4, while the same destination
models started clean completed 1/4: -25 points (interval -50 to 0). Preserving
work did not help the cheaper switch in this panel. By contrast, Kimi with
preserved state completed 2/4 for $0.2279 versus 1/4 for $0.6107 with clean
Kimi, a suggestive state-and-reasoning interaction.

### 6. Continue versus restart the current model clean

Continuing completed 1/4; clean same-model restart completed 0/4. The matched
difference was +25 points (interval 0 to +100), with one rescue and no harm for
continuation. A generic restart-on-stuck rule is not supported.

### 7. Flash->Qwen versus Qwen->Flash

Neither direction produced a stuck-state rescue. Flash->Qwen was observed once
and failed; Qwen->Flash was observed three times and failed all three. For the
Qwen-base groups, continuing recovered one state that switching to Flash did
not. The directionality sample is too small and unbalanced to infer that one
direction is generally superior.

### 8. Trigger-based versus fixed-turn intervention

For Flash/Qwen switching, the effect relative to continuing was -25 points at
stuck triggers and 0 at healthy checkpoints, for a -25-point
difference-in-differences. For Kimi, it was +25 points at stuck triggers and 0
at healthy checkpoints, for a +25-point difference-in-differences. This is the
desired qualitative interaction for Kimi, but both healthy controls are from
one task, so it is not confirmatory evidence.

## Group-level outcomes

| Checkpoint | Continue | Current clean | Value state | Value clean | Kimi state | Kimi clean |
|---|---:|---:|---:|---:|---:|---:|
| GMM, Qwen stuck t10 | pass | fail | fail | fail | fail | pass |
| Predictive maintenance, Flash stuck t3 | fail | fail | fail | pass | pass | fail |
| Predictive maintenance, Flash healthy t4 | fail | fail | pass | fail | pass | fail |
| Predictive maintenance, Qwen stuck t5 | fail | fail | fail | fail | pass | fail |
| Predictive maintenance, Qwen healthy t4 | pass | fail | fail | fail | fail | fail |
| Django validation, Qwen stuck t7 | fail | fail | fail | fail | fail | fail |

The GMM state is the measured false-positive trigger: Qwen eventually completed
when allowed to continue. It also demonstrates why state and model cannot be
collapsed into one variable: clean Kimi passed while Kimi inheriting the state
failed. Predictive maintenance showed the opposite pattern for Kimi: preserved
state passed while clean Kimi failed.

## Cost and execution integrity

- Exact incremental OpenRouter spend for this pilot: **$3.751379760**.
- Authorized additional ceiling: $15.00.
- Final usage on the newest recovery key: $3.500046166.
- Accepted matched-branch cost represented in the outcome table: $1.977335415.
- Remaining $1.774044345 covered base trajectories, invalid attempts,
  infrastructure recovery, and other provider-accounted pilot calls; it is
  retained in the exact billed total.
- Total OpenRouter spend across the supervisor project to date:
  **$45.887060203**.
- Daytona charges: unavailable through the installed SDK and therefore not
  included in either OpenRouter figure.

The final accepted table contains 36 rows, six complete groups, exactly one of
each predeclared arm per group, and equal limits within every group. Accepted
rows contain no provider, protocol, or state-transfer failures. The runner
recorded 59 execution attempts and kept invalid groups and structurally
ineligible checkpoints out of the analysis.

At closeout, no Harbor/benchmark process was running and Daytona reported zero
sandboxes. The runner's cleanup ledger contains one harmless remove-after-gone
race for a sandbox that Daytona had already deleted; the authoritative final
Daytona listing was empty.

## Decision and logical next step

Do **not** fit a learned stuck detector or intervention policy from these 36
rows. The labels are too sparse, the healthy controls cover only one task, and
the same task contributes four of the six groups. A learned model would mostly
memorize task-specific behavior and turn pilot noise into false confidence.

The experiment did validate the causal harness: exact state can be branched,
preserved state can be separated from clean restart, model diversity can be
separated from reasoning escalation, and invalid infrastructure outcomes can
be excluded without silently filling cells.

The next experiment should therefore be a **narrow confirmatory replication**,
not a broad model sweep:

1. Freeze a new outcome-blind development pool selected for checkpoint
   eligibility, especially tasks without unmanaged process state.
2. Collect at least 12 stuck and 12 healthy groups across at least eight unique
   tasks, with no task supplying more than two groups of either kind.
3. Keep three primary arms: continue current state, switch to Kimi with state,
   and restart Kimi clean. Retain a smaller randomized Flash/Qwen-switch arm as
   a model-diversity control.
4. Predeclare the primary interaction: Kimi-minus-continue at stuck checkpoints
   minus Kimi-minus-continue at healthy checkpoints.
5. Require the detector gap and Kimi interaction to remain positive under
   task-clustered uncertainty before collecting a training-sized dataset.

This preserves the product insight—escalate reasoning only when the run appears
stuck—while directly repairing the pilot's limiting weakness: insufficient
independent negative controls.

## Reproducible artifacts

- Frozen manifest: `artifacts/official/stuck-intervention-pilot-v0/frozen-pilot-manifest-v5.json`
- Snapshot fidelity report: `artifacts/official/stuck-intervention-pilot-v0/daytona-fork-fidelity-v0.json`
- Final matched outcomes: `artifacts/official/stuck-intervention-pilot-v0/execution-20260903T221959235932Z/matched-branch-outcomes-v0.jsonl`
- Frozen fit-free analysis: `artifacts/official/stuck-intervention-pilot-v0/stuck-intervention-analysis-v0.json`
- Exact execution ledger: `artifacts/official/stuck-intervention-pilot-v0/execution-20260903T221959235932Z/execution-ledger.json`
