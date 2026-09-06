# Continuation calibration v4: final result

## Decision

**NOT READY — continuation calibration gate failed.**

The behavioral result is promising: checkpoints labeled `CONFIRMED_STUCK`
recovered naturally much less often than healthy controls, the task-clustered
interval excluded zero, and the direction was positive for both models.

The experiment still fails its predeclared integrity gate. Two of 65 fresh
checkpoint replays changed the workspace digest, so v4 cannot authorize a
matched intervention experiment.

## Product question

A useful long-horizon supervisor must answer two questions separately:

1. Is the current agent unlikely to finish if left alone?
2. Would a particular intervention improve its chance of completion enough to
   justify the added cost?

V4 tests only the first question. Every checkpoint continued with the same
model, without a prompt, restart, switch, or early stop. A verifier—not the
detector—assigned the terminal task outcome.

## Frozen design

The v4 manifest was frozen before model outcomes with SHA-256:

`7a3b661a296f2b8ceaa3a9234b6b76d8fc0e0d161a73163fe21575c25903b260`

The run used:

- 24 fresh, preselected hard Terminal-Bench Pro tasks;
- DeepSeek V4 Flash 0731 and Qwen 3.8 27B on every task;
- three outcome-blind tranches of eight tasks;
- the unchanged two-tier detector `review-t5-confirm-t6-w2-e2`;
- a 12-turn and 49,152-token total run budget;
- natural continuation only;
- healthy, review, and confirmed-stuck checkpoint capture;
- provider-free rehydration of every candidate checkpoint; and
- task-clustered uncertainty, coverage, fidelity, leakage, and spend gates
  frozen before outcomes.

Earlier v0–v3 evidence and detector thresholds were not changed.

## Results

The frozen aggregate analysis combines the 48 v3 trajectories with 48 new v4
trajectories, as declared before collection:

| State at checkpoint | Checkpoints | Tasks | Natural completions | Completion rate |
|---|---:|---:|---:|---:|
| `HEALTHY` | 48 | 30 | 23 | 47.92% |
| `NEEDS_REVIEW` | 36 | 26 | 16 | 44.44% |
| `CONFIRMED_STUCK` | 7 | 7 | 1 | 14.29% |

The healthy-minus-confirmed recovery gap was **33.63 percentage points**. Its
task-clustered 95% interval was **[2.84, 57.78] points**. The route-specific
gaps were positive for both Flash (**23.48 points**) and Qwen (**52.00
points**), and removing any one checkpoint-bearing task left a positive gap.

Every behavioral, coverage, remaining-turn, structural-separation, and leakage
gate passed. The only failed gate was exact checkpoint fidelity.

The 48 fresh v4 trajectories contained 39 protocol-valid outcomes, of which 37
were nonstructural and learning-valid. Eleven outcomes were classified as
structural; protocol validity and structural classification are deliberately
nonexclusive fields. As a post-hoc diagnostic—not a tuning or pass/fail
result—the fresh cohort alone showed a 30-point gap, five confirmed-stuck
checkpoints from five tasks, and a task-clustered interval of **[-10.00, 61.29]
points**. The fresh cohort therefore points in the same direction but is not
independently conclusive.

## Why the fidelity gate failed

Sixty-three of 65 fresh checkpoint replays reproduced their source workspace
exactly. Both failures came from one Git-repository task:

- one `HEALTHY` checkpoint; and
- one `NEEDS_REVIEW` checkpoint.

The source archive contained Git object files with mode `0444`. Python's safe
`data` extraction filter normalized those regular files to `0644`; file bytes
were unchanged, but the permission change correctly altered the workspace
digest. The checkpoints are therefore excluded from admissible intervention
evidence.

The transport implementation now preserves ordinary POSIX permission bits
while retaining safe path, link, device, and special-bit checks. That fix does
not retroactively change v4: this result remains failed and immutable.

## Spend and cleanup

- Exact incremental OpenRouter spend: **$2.532378529**
- Tracked project spend after v4: **$53.269355918**
- Authorized project ceiling: **$200.00**
- Completed scheduled outcomes: **48/48**
- Remaining Daytona environments: **0**
- Execution errors: **0**

## What this proves—and does not prove

V4 provides evidence that the strict detector state contains useful
information about natural completion. It also supplies concrete examples of
why `NEEDS_REVIEW` must not automatically trigger a switch: several warned
trajectories recovered and passed.

It does **not** prove that switching improves a stuck run. No alternative action
was taken from an identical checkpoint. It also does not provide admissible
training data yet because the full fidelity contract did not pass.

## Next decision

Freeze an independent v5 cohort before seeing new outcomes. The justified
revision is deliberately narrow:

1. use the permission-preserving checkpoint transport;
2. exclude statically incompatible workspaces that contain harness-owned
   special files;
3. evaluate v5 on fresh outcomes only, so previously inspected labels are not
   repeatedly reused; and
4. leave models, detector thresholds, state definitions, turn/token limits,
   success-first gates, and natural-continuation behavior unchanged.

Only if all v5 calibration gates pass may the project freeze a matched-state
pilot comparing `CONTINUE_SAME`, complementary/cross-model handoff,
reasoning escalation, and a declared restart action from identical confirmed
snapshots.

## Public artifacts

- Frozen manifest:
  `artifacts/official/two-tier-continuation-calibration-v4/frozen-manifest-v4.json`
- Aggregate public summary:
  `artifacts/official/two-tier-continuation-calibration-v4/public-summary-v4.json`
- Final calibration report SHA-256:
  `56505afe6dbb22789855f8f153ef388957e484f38e2c8dbb394bfe184f4cd7b9`
- Snapshot fidelity report SHA-256:
  `128526e01fc2f690e16e8d6f9150a6fe88bd7d8b3a774a363348e6a6c1972534`
- Execution ledger SHA-256:
  `d020de59ae3318dc0997e982ca25709d15e84f82f6f3d55c5819282112314ac5`

Raw trajectories, terminal content, credentials, private benchmark material,
absolute local paths, and live provider logs remain excluded from the public
repository.
