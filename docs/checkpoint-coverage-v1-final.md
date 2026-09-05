NOT READY — detector development gate failed

# Checkpoint coverage feasibility

We should not launch another paid matched-intervention experiment yet. Detector v1 found a more meaningful definition of “stuck,” but it did not produce enough stuck checkpoints to satisfy the frozen collection gate.

## What the previous experiment taught us

The previous confirmatory run planned 84 base-scout slots across 21 tasks, two models, and two checkpoint kinds. Its final disposition was:

- 49 requested checkpoints never occurred;
- 22 slots depended on unmanaged external process state;
- 5 checkpoint groups failed matched-branch validity;
- 1 checkpoint had no remaining agent turns;
- 1 slot hit the predeclared per-task selection cap; and
- 6 groups were accepted, producing 24 valid branch outcomes.

The detector and infrastructure problems were distinct. The recorded v0 assessments emitted 33 raw checkpoint hits, but external-state eligibility, remaining budget, and branch validity reduced those to six complete matched groups. V0 also treated unchanged workspaces during normal read-only investigation as evidence of no progress, producing false alarms on tasks where inspection was the correct next action.

## What changed in detector v1

V1 preserves v0 byte-for-byte and is implemented separately. It makes four deliberate changes:

1. An unchanged workspace during inspection is not stuck evidence.
2. A stuck checkpoint requires productive work with observable errors or failing public tests on two consecutive turns.
3. The detector cannot declare stuck before turn 6.
4. Harness or protocol failures are classified as structural failures, not model failures.

The rule remains outcome-blind. It cannot read hidden-verifier results, future observations, private reasoning, sibling outcomes, or task identity.

## Offline development result

Evaluation used 58 natural, verifier-valid Flash and Qwen trajectories from 35 development tasks. Twelve protocol or structural failures were excluded from recovery labels. Candidate selection happened inside five task-grouped folds, so one task never appeared in both the development and evaluation sides of a fold.

| Checkpoint | Count | Tasks | Continuation completion |
|---|---:|---:|---:|
| Healthy | 58 | 32 | 26/58 (44.8%) |
| Projected stuck | 17 | 14 | 4/17 (23.5%) |

The healthy-minus-stuck difference was 21.3 percentage points. Its task-clustered 95% interval was 3.0 to 38.0 points. The difference remained positive for both base models and after leaving out any single task.

That is evidence that repeated errors are more useful than workspace stability for identifying low-recovery states. It is not enough to pass the full gate.

## Why the gate failed

We replayed v1 on the exact same full pre-outcome observations used by the prior v0 scouts:

| Requested checkpoint | V0 hits | V1 hits |
|---|---:|---:|
| Healthy at turn 4 | 11 | 19 |
| Suspected stuck | 22 | 3 |
| Total | 33 | 22 |

V1 preserved more useful healthy checkpoints, but its strict failure requirement eliminated most stuck checkpoints. The predeclared gate required improved checkpoint yield over v0 as well as a recovery difference. V1 therefore failed even though its recovery separation exceeded 20 points.

We did not relax the threshold or redefine yield after seeing the result.

## Spend and execution

- Incremental OpenRouter spend: **$0.00**
- Incremental Daytona environments: **0**
- Model calls: **0**
- Intervention outcomes generated: **0**
- Training runs: **0**

Task-source search, pool freezing, paid scouts, snapshot creation, and rehydration were not run because the detector-development gate failed first.

## Limitations

The 35-task development corpus contains online-safe pre-turn summaries, but not complete per-turn workspace digests, commands, public-test snapshots, or reproducibility state. The recovery analysis therefore evaluates a projection of v1’s repeated-error rule. Exact detector-yield replay uses the richer confirmatory observations, but those tasks are development evidence and cannot become fresh confirmation tasks.

We still have no counterfactual intervention label at these development checkpoints. This milestone evaluates checkpoint quality and availability, not whether switching models helps.

## Decision and logical next step

Do not train an intervention policy yet. Training becomes justified only after we can reproducibly bank both checkpoint kinds and collect matched continuation-versus-intervention outcomes.

The next design iteration should preserve v1’s repeated-failure precision while recovering stuck-checkpoint recall. The clearest candidate is a two-tier detector: a broad, high-recall “needs review” state followed by a stricter, evidence-based stuck confirmation. That detector must be evaluated on the same full observation schema and pass the unchanged coverage gate before any new paid task collection.

Machine-readable details are in `artifacts/official/checkpoint-coverage-v1/detector-development-report-v1.json`.
