# Gate 8 Wave 3: 11-task provisional scorecard

## Outcome

The frozen four-route cascade completed **6 of 11 held-out tasks (54.5%)**, versus
**4 of 11 (36.4%)** for the best static model, Kimi. That is two additional
completed tasks, an 18.2 percentage-point lift, and a 50% relative lift in
completion.

This is an encouraging provisional result, not a general production claim. The
held-out panel is still only 11 tasks.

## Frozen-policy comparison

The cascade order was selected on the 35-task development set and was not changed
after observing Wave 3 outcomes.

| Strategy | Successes | Success rate | Replayed model cost |
|---|---:|---:|---:|
| Static Flash | 3/11 | 27.3% | $0.0457 |
| Static Qwen | 2/11 | 18.2% | $0.8139 |
| Static GLM | 3/11 | 27.3% | $1.3761 |
| Static Kimi | 4/11 | 36.4% | $2.2687 |
| Flash → Qwen | 4/11 | 36.4% | $0.6419 |
| Flash → Qwen → GLM | 4/11 | 36.4% | $1.5444 |
| Flash → Qwen → GLM → Kimi | 6/11 | 54.5% | $2.8720 |

“Replayed model cost” uses cache-aware catalog pricing for the clean-start
attempts the strategy would make, stopping after verifier-confirmed success. It
is distinct from actual experiment spend.

## Preliminary cost-completion Pareto frontier

Three frozen strategies are non-dominated:

| Strategy | Successes | Replayed model cost | Interpretation |
|---|---:|---:|---|
| Static Flash | 3/11 | $0.0457 | Cheapest low-success option |
| Flash → Qwen | 4/11 | $0.6419 | Matches Kimi's completion at 71.7% lower replay cost |
| Flash → Qwen → GLM → Kimi | 6/11 | $2.8720 | Highest completion; two successes above any static model |

The three-route cascade is dominated by the two-route cascade at this checkpoint:
GLM adds cost but no additional successful task after Flash and Qwen. That is an
evaluation result, not a reason to alter the frozen policy before Wave 3 is
complete.

## Task-level patterns

The panel contains five all-fail tasks, one all-success task, and five
discriminating tasks.

| Task | Flash | Qwen | GLM | Kimi |
|---|:---:|:---:|:---:|:---:|
| apache-log-security-analyzer | Fail | Fail | Fail | Fail |
| boot-debian-qemu-with-ssh-check | Fail | Fail | Fail | Fail |
| decrypt-and-restore-backup-fragments | Pass | Pass | Pass | Pass |
| implement-crc32-with-logic-gates | Pass | Fail | Fail | Fail |
| implement-mitm-attack-for-24bit-double-cipher | Fail | Fail | Fail | Fail |
| improve-code-similarity-feature-extraction | Pass | Fail | Pass | Fail |
| optimize-portfolio-allocation | Fail | Fail | Fail | Pass |
| optimize-triton-rope-kernel | Fail | Fail | Fail | Fail |
| polyglot-text-stats-script | Fail | Pass | Pass | Pass |
| polyglot-yaml-config-validator | Fail | Fail | Fail | Pass |
| python-sokoban-bfs-solver | Fail | Fail | Fail | Fail |

The discriminating cases support both parts of the product thesis:

- Cheap-first routing matters because Flash alone solves CRC32 and shares a
  success on code-similarity extraction.
- Escalation matters because Kimi alone solves portfolio allocation and the YAML
  validator.
- A fixed GLM step is not yet justified by incremental held-out success, although
  it does solve tasks that other individual routes miss.

## Spend and integrity audit

- Dedicated OpenRouter key usage before this checkpoint: **$22.370639814**
- Dedicated OpenRouter key usage after this checkpoint: **$23.552935902**
- Exact incremental OpenRouter spend: **$1.182296088**
- Sum attributed by the four completed run reports: **$1.160615933**
- Key-delta/report difference: **$0.021680155**
- Authorized checkpoint expectation: under $5
- Daytona sandboxes remaining: **0**
- Held-out records: **44**, covering exactly 11 tasks × 4 routes
- Learning-valid records: **44**
- Superseded infrastructure/provider-error records: **12**
- Frozen policy code and development-report hashes: unchanged

The dedicated-key delta is the authoritative experiment-spend number. Per-run
provider reports are retained as a secondary allocation audit.

## Claim boundary and next decision

This scorecard estimates clean-start restart-and-escalate behavior. It does not
yet prove that a supervisor can detect failure mid-run, preserve state, and hand
the same workspace to another model with identical results.

The provisional evidence favors finishing the remaining seven Wave 3 tasks with
the frozen policy unchanged. The decision should be revisited only after the full
18-task held-out wave is complete.
