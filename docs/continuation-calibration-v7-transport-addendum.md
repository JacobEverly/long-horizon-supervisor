# V7 transport addendum

The first execution attempt of the frozen v7 continuation calibration did not
produce a learning-valid trajectory. Harbor reached the task's Dockerfile
build, but the host then could not resolve either `app.daytona.io` or
`openrouter.ai`. The runner is no longer active.

The attempt therefore records:

- one pending Flash trial;
- zero completed outcomes, checkpoints, or model calls;
- no token count or model cost in the Harbor result; and
- no model-stuck label.

This is a **STRUCTURAL_FAILURE/provider transport event**, not evidence about
Flash, Qwen, the detector, or continuation recovery. The v7 manifest,
thresholds, task cohort, and natural-continuation protocol remain frozen and
unchanged. The next permitted action is to retry the same v7 state only after
both provider endpoints resolve, then verify the exact Daytona cleanup before
continuing to the next tranche.

Evidence:

- frozen manifest: `69a9c88a11e5e21bb710ea2b2c93c0ea38ed2f9943f560e5f62347edbf31bf30`;
- Harbor run id: `14d734d2-efac-44ed-bd6e-3f794d78d6bb`;
- job log: `artifacts/official/two-tier-continuation-calibration-v7/jobs/continuation-v6-01-flash-20260907T134958488293Z/job.log`;
- result: `artifacts/official/two-tier-continuation-calibration-v7/jobs/continuation-v6-01-flash-20260907T134958488293Z/result.json`.
