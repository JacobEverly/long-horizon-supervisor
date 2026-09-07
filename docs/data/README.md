---
pretty_name: Long-Horizon Agent Reliability Evidence
language:
  - en
license: other
size_categories:
  - n<1K
tags:
  - agentic-coding
  - long-horizon-agents
  - model-routing
  - evaluation
  - aggregate-results
---

# Long-Horizon Agent Reliability Evidence

## Summary

This directory contains public, aggregate evidence from experiments on verified
completion, model complementarity, and stuck-state detection in long-running
coding agents. It is the release layer for the Long-Horizon Agent Reliability
project.

The current release contains aggregate summaries and 715 credential-redacted
ATIF trajectories totaling 172.1 hours of agent execution. The archive preserves
successful, failed, interrupted, and infrastructure-affected runs so downstream
work can distinguish capability from operational failure.

## Files

- `heldout-scorecard-summary-v0.json`: 72 valid outcomes from 18 sealed tasks and
  four fixed model routes, reported as static and sequential portfolio policies.
- `continuation-calibration-v6-summary.json`: natural-continuation recovery by
  detector state, including structural failures and the failed readiness gate.
- `continuation-calibration-v6-resume-summary.json`: provider-failure addendum;
  it contributes no learning-valid examples.
- `continuation-calibration-v7-summary.json`: the interrupted fresh-cohort run
  and its exact operational boundary.
- [`public-agent-traces-v0.jsonl.gz`](public-agent-traces-v0.jsonl.gz): 715
  redacted ATIF trajectories with task, model, duration, verification status,
  and completion metadata.
- [`public-agent-traces-v0-manifest.json`](public-agent-traces-v0-manifest.json):
  archive checksum, exact counts, duration, source attribution, and redaction
  totals.
- [`NOTICE.md`](NOTICE.md): source attribution, transformations, and combined
  licensing caveats.

Load the compressed JSONL directly with Hugging Face Datasets:

```python
from datasets import load_dataset

traces = load_dataset(
    "json",
    data_files="public-agent-traces-v0.jsonl.gz",
    split="train",
)
```

## Intended use

The aggregates support reproduction of the published claims and inspection of
the project's evidence gates. The trajectories support research on agent
failure, model complementarity, and trace representation. They do not by
themselves supply valid counterfactual labels for intervention timing.

## Collection and validation

Tasks, routes, budgets, stopping rules, and analysis gates are frozen before
outcomes are observed. Success comes from executable external verifiers, not
model self-report. Infrastructure failures are recorded separately from model
failure. Task-level grouping prevents sibling trajectories from crossing data
splits.

## Limitations

The archive is intentionally broader than the headline evaluation. It contains
retries, pilots, structural failures, and provider or sandbox errors; a record
must not be treated as training-valid solely because it is present. The headline
evaluation contains 18 coding tasks and four model deployments and does not
establish universal model rankings or the value of mid-run switching. Costs are
provider- and date-specific.

## Licensing and privacy

Project-authored code is MIT licensed. The
[Terminal-Bench Pro dataset card](https://huggingface.co/datasets/alibabagroup/terminal-bench-pro)
identifies its public dataset as Apache-2.0; the release retains that provenance
and does not include protected verifier files. Trajectory text is scrubbed for
credential-shaped strings, emails, private-key material, and personal local
paths. Because model-output terms and source-specific terms may differ, the
combined dataset remains `license: other`.

## Versioning

Public summaries are immutable and schema-versioned. Corrections or resumed runs
are published as new files rather than overwriting prior evidence.
