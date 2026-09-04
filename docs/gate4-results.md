# Gate 4 results: guards, current models, and recovery headroom

## Outcome

Gate 4 completed the four requested steps:

1. refreshed every model role to a current, exact endpoint;
2. implemented deterministic supervisor guards;
3. expanded the executable task set and saved immutable failed-run checkpoints;
4. branched identical checkpoints across models and measured step-down and
   switching value.

The final dataset contains 30 fixed-model baselines (five models × six
medium/hard tasks) and 10 controlled continuations (five models × two failed
Qwen checkpoints). Seven transient connection-error attempts are retained in a
separate infrastructure audit and were retried instead of being counted as
model failures.

## Current model roster

| Tier | Exact endpoint | Role | Reasoning effort | Catalog input/output per 1M |
|---:|---|---|---|---:|
| 0 | `qwen/qwen3.8-27b` | compact floor | high | $0.45 / $3.20 |
| 1 | `deepseek/deepseek-v4-flash-0731` | value | high | $0.14 / $0.28 |
| 2 | `deepseek/deepseek-v4-pro-0813` | reasoning step-up | high | $1.188 / $3.564 |
| 3 | `z-ai/glm-5.3` | long-horizon frontier | max | $1.40 / $4.40 |
| 4 | `moonshotai/kimi-k3` | ceiling | high | $3.00 / $15.00 |

The exact IDs, context limits, prices, and advertised tool support were captured
from OpenRouter's public catalog on 2026-08-20. GLM 5.3 replaces 5.2, Qwen3.8
27B replaces the old Qwen3 8B floor, and the DeepSeek entries are pinned dated
releases rather than moving aliases. Kimi K3 remains the current Moonshot slot.

Release references:

- GLM 5.3: <https://z.ai/blog/glm-5.3>
- Qwen3.8: <https://qwen.ai/blog?id=qwen3.8>
- DeepSeek V4: <https://api-docs.deepseek.com/news/news260424/>
- OpenRouter deployment catalog: <https://openrouter.ai/api/v1/models>

## Deterministic guard layer

The Verifiers environment now records and acts on:

- completion attempts without a test pass on the current workspace digest;
- three consecutive tool/edit errors or no-change mutations;
- more than 12 tool calls in one model response;
- more than 48 tool calls in one run;
- turn, completion-token, environment-time, process-time, per-run cost, and
  experiment-cost ceilings;
- malformed/unknown tools and provider or harness errors.

An unverified completion receives one corrective prompt. A second unverified
completion stops the attempt and preserves the workspace for handoff. Hard
protocol/resource failures stop without executing the dangerous batch.

One paid baseline supplies direct guard value: DeepSeek V4 Pro attempted to stop
the retry-policy task without current verification, received the deterministic
correction, continued, and passed. That is one observed rescue, not proof that
the thresholds are optimal. Nineteen paid runs encountered an unverified-stop
guard; only one ultimately passed, so the trigger and the recovery action must
be evaluated separately.

## Expanded tasks

The executable suite grew from three to seven tasks. Gate 4 used the six
medium/hard tasks:

- TTL cache semantics;
- feature dependency planning;
- bounded retry policy;
- idempotent request lifecycle;
- versioned webhook replay;
- weighted quota allocation.

Every starter fails hidden tests and every checked-in gold repair passes. The
tasks cover temporal semantics, graph reasoning, state machines, duplicate
handling, numeric edge cases, and fair allocation rather than one repeated bug
shape.

## Fixed-model results

| Model | Passed | Completion | Estimated task cost | Avg. estimated cost | Pareto |
|---|---:|---:|---:|---:|:---:|
| Qwen3.8 27B | 4/6 | 66.7% | $0.094160 | $0.015693 | yes |
| DeepSeek V4 Flash 0731 | 2/6 | 33.3% | $0.014890 | $0.002482 | yes |
| DeepSeek V4 Pro 0813 | 2/6 | 33.3% | $0.235949 | $0.039325 | no |
| GLM 5.3 | 1/6 | 16.7% | $0.171823 | $0.028637 | no |
| Kimi K3 | 3/6 | 50.0% | $0.364320 | $0.060720 | no |

Task success matrix:

| Task | Models that passed from the clean starter |
|---|---|
| TTL cache | all five |
| feature dependency | Qwen3.8 27B only |
| retry policy | Qwen3.8 27B, DeepSeek V4 Pro, Kimi K3 |
| idempotency store | Kimi K3 only |
| webhook reducer | none |
| weighted quota | Qwen3.8 27B, DeepSeek V4 Flash |

Model size and price were not monotonic with task completion. The compact Qwen
model passed two tasks where every frontier slot failed, while Kimi uniquely
passed idempotency. The product should therefore predict model-task/checkpoint
compatibility rather than encode “harder means larger.”

GLM 5.3's 1/6 result is a local harness result, not a general capability claim.
The official release reports large gains over 5.2 on long-horizon coding; this
small tool loop, its prompting, max effort, and strict limits may be a poor fit.

## Identical-checkpoint experiment

Qwen failed idempotency and webhook from their clean starters. Each resulting
workspace was copied once into an immutable checkpoint. Its digest was recorded
and later revalidated. Every continuation received:

- an independent copy of the exact same checkpoint;
- the same task contract and handoff summary;
- the same limits and tools;
- no hidden-test details.

A fresh Qwen continuation was included alongside the four other models. This
control separates “more time/context” from “switching models.”

| Checkpoint | Qwen control | Flash | Pro | GLM 5.3 | Kimi K3 |
|---|:---:|:---:|:---:|:---:|:---:|
| idempotency after Qwen | fail | fail | fail | fail | fail |
| webhook after Qwen | fail | fail | fail | fail | fail |

Observed switch rescues: **0/2**. Switch-exclusive rescues: **0/2**.

This is not a null result. Kimi passed idempotency from the clean starter but
failed from Qwen's partial state. Preserving external state made the task harder
than restarting. The supervisor's action space must therefore be:

```text
continue same model | switch model | roll back checkpoint | restart cleanly | stop
```

“Always preserve the workspace and escalate” is not supported by this evidence.
The next recovery experiment should branch both the dirty checkpoint and its
last known-good predecessor so switching and rollback can be compared directly.

## Step-down evidence

Qwen passed four of six tasks. On two of those tasks a frontier model also
passed, producing two observed safe step-downs. On feature dependency and
weighted quota, Qwen passed while the frontier slots failed, so those are better
described as compatibility wins than step-downs.

The result supports starting with a capable compact model for some states, but
not a universal compact-first production rule. Kimi's unique idempotency pass
shows that escalation headroom exists from a clean state even though dirty-state
switching did not recover it.

## Budget and artifact audit

- Gate 4 hard ceiling: **$5.00**.
- Token/catalog estimate including infrastructure attempts: **$1.271219448**.
- Provider-reported incremental Gate 4 usage: **$1.130996808**.
- Provider-reported total dedicated-key usage across Gates 3 and 4:
  **$1.488896482**.
- Remaining dedicated-key limit at audit: **$48.511103518**.
- Final valid trajectories: **40/40 Harbor-valid ATIF v1.7 files**.
- Immutable checkpoints: **2/2 digests revalidated**.

Elapsed-time rankings are intentionally omitted. The host wall clock changed
during several rollouts, corrupting Verifiers' wall-clock duration metric. The
independent process watchdog remained bounded, while token, cost, workspace,
guard, and verifier evidence remained usable. Future runs now also record a
monotonic process duration.

## Product decision

Do not train a model-switch selector from this dataset. It contains no positive
dirty-checkpoint switch label. The evidence instead supports three immediate
product rules:

1. verify before accepting completion;
2. choose models by task/checkpoint compatibility, not parameter tier;
3. preserve a last-known-good checkpoint and evaluate rollback/restart alongside
   continuation and switching.

The project is now a credible supervisor/evaluation artifact: it has portable
state, executable verifiers, deterministic safety baselines, current model
endpoints, immutable branch controls, spend accounting, and a falsifiable next
experiment rather than an unsupported routing demo.
