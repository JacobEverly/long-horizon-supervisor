# Gate 8 Wave 1 freeze

## Outcome

Wave 1 produced a strict 15-task by 4-route dataset: 60 learning-valid
task–route rows. There are 26 completions, 34 failures, and four tasks on which
the routes disagree. Ten additional learning-valid cells exist in three
non-rectangular task groups, giving 70/72 usable individual outcomes, but those
groups are excluded from strict training rather than silently mixing contracts
or external failures.

The frozen dataset is
`artifacts/official/gate8-fifteen-task-development/matched-outcomes-60-v1.jsonl`.
Its task-held-out evaluation is
`artifacts/official/gate8-fifteen-task-development/route-baseline-v0.json`.

## What the benchmark says

- Always using Flash completes 7/15 tasks.
- The current held-out learned policy also completes 7/15 and falls back to
  Flash on every task.
- A hindsight oracle completes 8/15, proving one task of real routing headroom.
- Four of 15 tasks are discriminating (26.7%). At that observed rate, about 75
  representative tasks are needed to observe 20 contrasts.

This is a successful data and evaluation checkpoint, not yet evidence that the
learned router generalizes. The router cannot learn a stable task boundary from
four contrasts, even though the oracle proves that a boundary is worth finding.

## Why three tasks are quarantined

- `build-grpc-user-profile-service`: all four observed routes fail, but Flash
  used the bounded retry contract while the older three cells predate it.
- `build-nginx-1-24-production-server`: Flash, Qwen, and Kimi fail; GLM repeatedly
  fails during sandbox startup, and Flash is from the older contract.
- `polyglot-bash-python-config-parser`: Qwen, GLM, and Kimi pass; Flash repeatedly
  stalls before a first response. That is external reliability evidence, not a
  model-failure label.

Re-running these low-information cells would improve rectangularity but is a
worse use of budget than collecting new representative tasks.

## Fastest defensible expansion

1. Run the full four-route matrix on proportional Waves 2 and 3. This takes the
   strict representative set from 15 to 51 tasks and clears the 50-task floor.
2. On Wave 4, screen every task with Qwen and GLM. That pair disagrees on all
   four current contrastive tasks.
3. Run all four routes on every sentinel disagreement, plus a random audit of
   sentinel agreements to measure missed contrasts.

This preserves the proportional benchmark distribution while concentrating the
remaining four-route spend where it can add routing signal.

## Cost integrity

Exact OpenRouter spend at freeze is $7.5978 from the dedicated $50 capped key.
Policy comparisons use cache-aware catalog list prices, which remain portable
to new model deployments. Provider-key spend windows that overlapped during
parallel collection are explicitly suppressed rather than attributed to the
wrong task. No Daytona sandboxes were live at freeze.
