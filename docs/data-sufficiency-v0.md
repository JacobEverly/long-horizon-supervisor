# Data sufficiency v0: how much router data is enough?

## The short answer

For this project, a credible first target is not “one giant dataset.” It is three
linked datasets with different evidence requirements:

| Evidence | Credible v0 | Stronger follow-up | Current |
|---|---:|---:|---:|
| Leakage-safe coding prompts | 10,000–25,000 | 50,000+ from real sessions | 13,207 |
| Human ambiguity labels | 1,000 | 2,000–5,000 | 0 |
| Shared tasks run on every candidate model | 400 | 1,000+ | 48 for all three public models |
| Verifier-confirmed rollout attempts | 10,000+ | 50,000+ balanced attempts | 76,480 joined, but imbalanced |

The current prompt corpus clears the v0 task-classification bar. It does not
clear the multi-model outcome bar. Collecting more generic prompts has lower
value now than collecting matched outcomes for the same tasks and models.

## Why these numbers

[RouteLLM](https://arxiv.org/abs/2406.18665) trained on roughly 80,000 Chatbot
Arena preference battles. Its paper also reports poor out-of-distribution
routing on MMLU and GSM8K until relevant synthetic data was added. The lesson is
that dataset similarity and paired outcome quality matter more than merely
reaching 80,000 rows.

[Morph](https://www.morphllm.com/llm-router) says its production coding router
uses millions of real coding prompts. That is useful evidence about the scale of
a mature commercial product, but it is a company-reported number rather than a
reproducible public recipe. It should not be our prototype acceptance criterion.

For a simple aggregate success-rate estimate, about 385 independent tasks per
model gives a 95% confidence interval of approximately ±5 percentage points in
the worst case. About 1,067 gives approximately ±3 points. Routing needs coverage
within task types as well, so 400 matched tasks is a minimum useful bar and 1,000
is a better target.

## What was collected

The pinned full build starts with 13,434 eligible public task rows and removes
227 exact-normalized or canonical Codeforces duplicates. The result is 13,207
records:

| Split | Records |
|---|---:|
| Train | 10,800 |
| Validation | 1,138 |
| Internal test | 1,269 |

The split is performed by leakage group. All tasks from one repository stay in
one split, and the same Codeforces problem cannot appear through both APPS and
CodeContests.

The task mix is:

- 4,870 APPS prompts;
- 2,107 rated DeepMind CodeContests prompts;
- 6,230 permissively licensed SWE-bench-extra repository issues;
- 6,977 prompts with defensible difficulty labels;
- 6,230 repository issues whose difficulty remains intentionally unlabeled.

The outcome join reads only `instance_id`, `model_name`, `target`, and
`exit_status` from Nebius. It excludes trajectories, generated patches, and
evaluation logs. Of 80,036 source attempts, 76,480 join to 3,303 retained tasks.

## The important limitation

The 76,480 attempts are not a balanced 80,000-row routing dataset:

- 2,734 retained tasks have only 70B outcomes;
- 471 compare 70B with 8B;
- 50 compare 70B with 405B;
- only 48 contain all three models.

The published dataset also names model families without immutable weights and
agent-configuration revisions. These outcomes are useful for pretraining and
for studying task solvability, but not as final proof that one current model
should be selected over another.

Raw model-wide success rates must not be compared as a leaderboard because the
models were evaluated on different task subsets. The defensible comparison is
within matched tasks.

## Collection decision

The next data priorities are:

1. label 200 prompts twice to stabilize the ambiguity rubric, then expand to at
   least 1,000 human-labeled examples;
2. create a matched panel of 50–100 tasks across the current model roster as a
   budgeted pilot;
3. expand toward 400 shared tasks only if the learning curve and routing regret
   show that additional outcomes are valuable;
4. add real agent-session turns later, because benchmark problem statements do
   not represent routine actions such as search, test repair, and documentation;
5. stop adding generic public prompts unless held-out error analysis reveals a
   specific missing domain.

This is a success-first collection strategy: first obtain enough matched data
to distinguish model capability, then optimize the Pareto policy for cost.
