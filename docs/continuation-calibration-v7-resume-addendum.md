# V7 resumed-run addendum

After transport recovered, the frozen v7 calibration was resumed and then
stopped at the operator's request. The run reached one terminal schedule item:

- Flash on `maze-shortest-path-solver` timed out twice at the frozen 420-second
  child bound;
- both attempts were classified as `STRUCTURAL_FAILURE` with no verifier
  outcome and no checkpoints;
- Qwen on the same task started afterward, but was interrupted before a
  terminal outcome and is not included in the learning-valid evidence; and
- Daytona cleanup completed with zero remaining sandboxes.

The run therefore ends with **one structural trajectory, zero valid natural
trajectories, and zero counted checkpoints**. It does not change any detector
gate, produce intervention evidence, or authorize training.

The exact provider usage moved from **$5.505483417** to **$5.533624397**, an
increment of **$0.02814098**. The v7 manifest and task cohort remain frozen;
the partial Qwen attempt is retained only as an operational record and must
not be scored or reused as a counterfactual.

Evidence:

- frozen manifest: `69a9c88a11e5e21bb710ea2b2c93c0ea38ed2f9943f560e5f62347edbf31bf30`;
- execution ledger: `artifacts/official/two-tier-continuation-calibration-v7/execution-ledger-v7.json`;
- execution state: `artifacts/official/two-tier-continuation-calibration-v7/execution-state-v7.json`;
- outcome stream: `artifacts/official/two-tier-continuation-calibration-v7/natural-continuation-outcomes-v7.jsonl`.
