# V6 resume addendum

The original v6 report is preserved at its first completed analysis boundary:
26 trajectories, 17 learning-valid and 9 structural. A later resume attempted
seven additional schedule items from the same frozen pool. All seven were
structural/provider failures, so the resume added **zero learning-valid
trajectories and zero checkpoints**.

Qwen requests repeatedly returned upstream HTTP 502 transport errors and some
Harbor children remained alive for hours without producing a usable trajectory.
The children were terminated at the process boundary; no model-stuck labels
were created. The resumed state therefore ends at 33 trajectory rows, 17
learning-valid rows, 16 structural rows, and 28 faithful checkpoints.

The resume consumed approximately **$0.4855** beyond the original v6 report
boundary. All Daytona environments were removed. Because the resume produced no
new valid evidence, it does not change the v6 Gate A conclusion or authorize an
intervention experiment.

This is an operational finding: the next calibration needs a bounded provider
transport contract and a genuinely fresh task cohort before it can add evidence
to the stuck-state question.
