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

The current release contains summaries, not reusable raw trajectories. A trace
release will follow only after source licenses, redaction, and row-level schema
checks pass.

## Files

- `heldout-scorecard-summary-v0.json`: 72 valid outcomes from 18 sealed tasks and
  four fixed model routes, reported as static and sequential portfolio policies.
- `continuation-calibration-v6-summary.json`: natural-continuation recovery by
  detector state, including structural failures and the failed readiness gate.
- `continuation-calibration-v6-resume-summary.json`: provider-failure addendum;
  it contributes no learning-valid examples.
- `continuation-calibration-v7-summary.json`: the interrupted fresh-cohort run
  and its exact operational boundary.

## Intended use

The aggregates support reproduction of the published claims, comparison of
success-versus-cost policies, and inspection of the project's evidence gates.
They are not sufficient to train an intervention model.

The planned row-level release is intended for research on continuation risk,
model choice, and intervention timing. Each example will describe only state
available at decision time and will link sibling actions through a matched-group
identifier.

## Collection and validation

Tasks, routes, budgets, stopping rules, and analysis gates are frozen before
outcomes are observed. Success comes from executable external verifiers, not
model self-report. Infrastructure failures are recorded separately from model
failure. Task-level grouping prevents sibling trajectories from crossing data
splits.

## Limitations

The headline evaluation contains 18 coding tasks and four model deployments. It
does not establish universal model rankings or the value of mid-run switching.
Current stuck-state evidence is sparse, and some later runs were interrupted by
provider or sandbox failures. Costs are provider- and date-specific.

## Licensing and privacy

Project-authored code is MIT licensed. Benchmark prompts, tests, and raw model
outputs are not redistributed here. Any future row-level release must retain
source provenance, document source-specific terms, remove credentials and local
paths, and exclude protected verifier content. The dataset therefore uses
`license: other` until that audit is complete.

## Versioning

Public summaries are immutable and schema-versioned. Corrections or resumed runs
are published as new files rather than overwriting prior evidence.
