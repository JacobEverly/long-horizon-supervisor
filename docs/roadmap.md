# Roadmap to a trained intervention policy

The clean-start model portfolio is supported by held-out evidence. The remaining
research question is whether a supervisor can recognize a low-recovery state and
choose a better action from the same saved workspace.

Training is deliberately gated on evidence rather than scheduled by date.

## A. Validate the intervention trigger

Run Flash and Qwen naturally on a fresh, frozen task cohort. Save one healthy
control, the first `NEEDS_REVIEW` state, and any later `CONFIRMED_STUCK` state;
then let the same model continue to the external verifier.

The gate requires:

- at least a 20-point healthy-versus-confirmed recovery gap;
- a task-clustered 95% interval excluding zero;
- positive separation for both models and no one-task dependence;
- adequate healthy, review, and confirmed coverage across independent tasks;
- enough remaining turns to make intervention meaningful; and
- terminal outcomes, exact checkpoint fidelity, and clean provenance for every
  counted example.

If the gate fails, preserve the result and make one minimal, independently
frozen revision. Do not tune and score a detector on the same outcomes.

## B. Measure intervention value

Only after Gate A passes, branch identical confirmed-stuck snapshots into:

1. continue the current model;
2. switch to a complementary model with state preserved;
3. escalate to a stronger reasoning model; and
4. clean restart where applicable.

Randomize branch order before execution. Every branch reaches the same external
verifier and records success, tokens, duration, cost, and transfer failures.
Proceed only if at least one intervention is not dominated by continuing and
different actions win on different groups.

## C. Build training-ready data

Expand outcome-blind to at least 40 valid matched groups across at least 20
tasks, subject to a precommitted power analysis. Each group includes continuing
plus at least two applicable interventions, and failed actions remain in the
dataset.

Before fitting:

- build a rectangular or explicitly masked checkpoint-by-action table;
- split train/development and untouched validation data by task;
- freeze the input schema, actions, objective, weights, missing-outcome handling,
  rule baselines, and success-first Pareto evaluation;
- exclude task IDs, sibling outcomes, private reasoning, future state, and
  hidden verifier information; and
- pass formal leakage, integrity, support, and one-task/model-dependence audits.

The first model will be trained only after these gates pass. Until then, the
transparent verifier-gated portfolio remains the supported policy.
