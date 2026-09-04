# Chooser dataset v0: task understanding before model routing

## What this checkpoint does

This checkpoint defines the exam the model chooser must pass before we train it.
It freezes:

1. information the chooser may see before an agent run;
2. labels it may learn;
3. evidence required for each label;
4. outcome data needed to recommend a model;
5. train, validation, test, and quarantine boundaries;
6. fields that must never leak from solutions or verifiers.

The schema is executable in `horizon_supervisor.chooser.schema`. The pinned
source registry is `data/chooser/source-registry-v0.json`. The reproducible
36-row smoke sample is `data/chooser/sample-v0.jsonl`.

The smoke sample validates the contract and source adapters. It is not large
enough to train a useful model.

## Product contract

The learned component estimates task properties and model outcomes. A separate
product policy selects the model.

```text
task text
    ↓
task understanding
    difficulty + ambiguity + domain
    ↓
outcome estimator
    P(success), tokens, cost, latency for every candidate deployment
    ↓
product policy
    capability-first | balanced | cost-efficient
    ↓
selected model or abstention
```

`recommended_model` is deliberately not a ground-truth field in the dataset.
It is derived from empirical outcome estimates and a product policy. This lets
us change the cost-versus-success preference without relabeling or retraining
the task classifier.

## Allowed input

The v0 chooser may receive only information available before agent execution:

- the user-visible task text;
- optional repository identity;
- programming language when supplied publicly;
- task family;
- public metadata such as contest rating, creation date, or tags.

It may not receive:

- canonical or reference solutions;
- gold patches or generated patches;
- public, private, or hidden tests;
- hints that were not shown to the evaluated agent;
- verifier output or reward;
- post-run logs, token usage, or cost.

The outcome estimator may use those post-run values as **targets**, never as
task input. Every record carries exact and normalized prompt hashes so duplicate
tasks can be grouped across datasets.

## Targets and evidence

### Difficulty

`easy`, `medium`, or `hard`. The normalized value is stored alongside its
native label and mapping version.

Current high-confidence mappings:

| Source | Native value | Normalized value |
|---|---|---|
| APPS | introductory | easy |
| APPS | interview | medium |
| APPS | competition | hard |
| Codeforces | rating ≤ 1200 | easy |
| Codeforces | rating 1300–1900 | medium |
| Codeforces | rating ≥ 2000 | hard |
| SWE-bench Verified | `<15 min fix` | easy |
| SWE-bench Verified | `15 min - 1 hour` | medium |
| SWE-bench Verified | `1-4 hours` or `>4 hours` | hard |

The bands are source-specific ordinal supervision, not a claim that a 1900-rated
algorithm problem equals a one-hour repository fix. We retain the native value
and evaluate domain transfer separately.

### Ambiguity

`low`, `medium`, or `high`. None of the selected training sources provides a
trusted ambiguity label, so the smoke records leave this target null. The first
full dataset will add a small double-annotated human set with adjudication. We
will not use a teacher model's labels as if they were human truth.

### Domain

The v0 taxonomy is:

- `software_maintenance`
- `algorithmic_problem_solving`
- `data_processing`
- `testing`
- `systems`
- `architecture_design`
- `research`
- `general`

The smoke sample uses only the first two because those are directly supported
by its source families. Finer domains require manual validation.

### Model outcomes

For an immutable model deployment and agent configuration, store:

- attempts and verifier-confirmed successes;
- mean input and output tokens;
- mean provider cost;
- mean latency;
- exact model deployment, agent, and verifier identifiers.

One attempt is an observation, not a stable success probability. We will fit a
partial-pooling model or calibrated classifier across tasks and require repeated
attempts where a routing decision is sensitive.

## Source selection

Hugging Face is a distribution mechanism, not the quality test. The source
registry uses three quality states:

- **A:** original benchmark maintainer with objective or human-validated labels;
- **B:** established research release or mirror with documented provenance;
- **Q:** quarantined pending license, overlap, or label review.

### Training and calibration sources

- [DeepMind CodeContests](https://huggingface.co/datasets/deepmind/code_contests)
  is an original DeepMind release under CC BY 4.0. We use English Codeforces
  tasks with an objective rating and exclude all solutions and tests.
- [APPS](https://huggingface.co/datasets/codeparrot/apps) is a widely used MIT
  mirror with authored difficulty bands. It is useful, but receives tier B
  because the Hugging Face publisher is not the original paper author.
- [Nebius SWE-bench-extra](https://huggingface.co/datasets/nebius/SWE-bench-extra)
  supplies real repository issue prompts. We retain per-repository license
  metadata and use only permissively licensed rows in the smoke sample.
- [Nebius SWE-agent trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories)
  contains 80,036 agent trajectories with model identity and pass/fail targets.
  It is a promising outcome-pretraining source, but its generated outputs remain
  excluded until the generating-model and repository licenses are audited.

### Evaluation-only sources

- [SWE-bench Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified)
  contains 500 human-validated repository tasks with expert-time bands. It is
  evaluation-only and grouped by repository to prevent leakage.
- [BigCodeBench](https://huggingface.co/datasets/bigcode/bigcodebench) is an
  Apache-2.0, expert-reviewed functional-coding benchmark. It remains held out.
- [LiveCodeBench](https://huggingface.co/datasets/livecodebench/code_generation_lite)
  offers time-based, contamination-aware evaluation. Its card currently gives
  an underspecified `cc` license, so it is quarantined until clarified.
- [Terminal-Bench 2.1](https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2-1/latest)
  is the final agentic evaluation. The existing 30-task sample remains sealed
  and may never be used to train or tune the initial chooser.

## Split and leakage policy

The full build must split by leakage group, never by independent rows:

- repository for GitHub issue tasks;
- contest/problem identity and normalized text hash for algorithmic tasks;
- benchmark task digest for Terminal-Bench;
- model deployment plus agent configuration for outcome observations.

Before freezing a split:

1. exact-deduplicate by input hash;
2. near-deduplicate normalized prompts across sources;
3. group all repeated attempts for one task together;
4. keep repositories disjoint between training and SWE-bench Verified;
5. use release dates for LiveCodeBench cutoffs;
6. quarantine any task found in Terminal-Bench 2.1.

Gold fields may be used offline to compute complexity features or verifier
outcomes only if those features cannot reveal the answer and are explicitly
marked target-side. They are never serialized in `ChooserInput`.

## Smoke-sample result

The pinned sample contains 36 records:

| Source | Rows | Difficulty labels | Domain |
|---|---:|---|---|
| APPS | 12 | 4 easy / 4 medium / 4 hard | algorithmic |
| DeepMind CodeContests | 12 | 4 easy / 4 medium / 4 hard | algorithmic |
| Nebius SWE-bench-extra | 12 | intentionally unlabeled | software maintenance |

The 12 repository issues come from 12 repositories and carry permissive
repository licenses. The sample has no exact normalized-prompt duplicates. Task
text ranges from 282 to 5,912 characters. No answer fields enter model input.

Rebuild it with:

```bash
python -m horizon_supervisor.chooser.build_sample
```

The builder checks every current Hugging Face repository revision against the
pinned SHA and stops if a source moved.

## Next evidence gate

The full metadata build is now complete. See `docs/data-sufficiency-v0.md` for
the resulting 13,207 tasks, 76,480 joined outcomes, and the matched-model
coverage limitation. The next gate is still not fine-tuning. It is a label and
matched-outcome audit:

1. double-annotate 200 examples to stabilize the ambiguity rubric;
2. expand to at least 1,000 human ambiguity labels;
3. run a 50–100-task matched panel across the current candidate models;
4. train rules and a lightweight statistical baseline;
5. measure calibration, routing regret, and domain transfer;
6. fine-tune a small language model only if the statistical baseline leaves
   measurable headroom.
