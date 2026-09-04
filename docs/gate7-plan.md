# Gate 7: held-out long-horizon evaluation

## Sizable outcome

Produce a state-aware supervisor that improves the probability of completing
long-horizon terminal tasks without being dominated on cost. The evidence must
come from a recognized external benchmark, not only the repository's authored
repair tasks.

The primary result will compare three strategies on the same frozen tasks:

1. a fixed economical model;
2. a fixed high-reasoning model;
3. the supervisor, which may continue, roll back, restart clean, or change models.

The report will plot completion rate against total model and sandbox cost. A
strategy is useful only if it is on the observed Pareto frontier. Completion is
the primary objective; cost breaks ties and determines whether a higher success
rate is practically sustainable.

## Benchmark and frozen sample

Gate 7 uses Terminal-Bench 2.1 through Harbor 0.21.0. The benchmark contains 89
containerized terminal tasks. The frozen manifest is
`benchmarks/terminal-bench-2.1-gate7.json`.

The sample was selected from metadata only, before inspecting task instructions:

- medium and hard tasks only;
- 20 medium and 10 hard tasks;
- every eligible category receives at least one seat within each difficulty;
- remaining seats are allocated proportionally;
- a fixed SHA-256 rank chooses tasks within each category;
- vision/video/OCR, nested virtualization, explicitly unverifiable tasks, and
  resource profiles above 2 CPUs or 4 GB RAM are excluded as harness confounds.

This left 64 eligible tasks and selected 30. Every selected task is pinned by its
Harbor content digest so a later dataset update cannot silently alter the test.

## Harness boundary

Harbor owns the task container, persistent filesystem, agent, verifier, timeouts,
and trial result. NVIDIA NeMo Switchyard 0.2 is the provider-neutral model proxy.
The agent keeps one native conversation while Switchyard changes the backend for
individual model turns. The supervisor consumes normalized progress and evidence
signals and supplies routing decisions without importing task-specific verifier
code.

Switchyard's built-in `stage_router` is a required baseline. It already routes
coding-agent turns from tool-use and progress signals, so the project must beat
or complement it rather than presenting generic stage escalation as novel.

`benchmarks/switchyard-gate7.toml` pins five current endpoints and exposes a
passthrough route for each. It also defines two matched Kimi K3/Qwen3.8 27B
stage routes: `stage-quality` falls open to the capable model, while
`stage-cost` falls open to the efficient model. Both use Switchyard's recommended
0.5 signal threshold and three-turn window. The five-task fixed-model pilot may
change the capable/efficient pair before the 30-task evaluation is frozen; it
may not change that pair after seeing full-evaluation outcomes.

The first official oracle smoke test used `log-summary-date-ranges` and passed
with reward 1.0. It spent no model tokens. It took 7 minutes 20 seconds locally
because the ARM laptop emulated an x86 image. Gate 7 paid trials must therefore
run in an x86 cloud sandbox; local Docker remains a correctness smoke path only.

## Evaluation sequence

1. **Oracle smoke:** one selected task, one reference-solution attempt. Complete.
2. **Agent smoke:** one selected task with the adapter and a strict spend cap.
3. **Five-task pilot:** three strategies, one attempt each. Use this to measure
   empirical tokens, cost, runtime, and failure modes.
4. **Power and budget checkpoint:** choose attempt count and full matrix using
   the pilot variance. Do not infer it from catalog prices alone.
5. **Frozen evaluation:** run all 30 tasks with interleaved strategy order.
6. **Replication:** add attempts where the confidence interval or policy ranking
   is decision-sensitive. Five attempts per task is the leaderboard-grade target;
   three is the minimum planned internal comparison.

## Metrics and anti-leakage rules

Primary metric: task completion rate from the official verifier.

Secondary metrics:

- total API cost and sandbox cost per task;
- cost per successful task;
- turns, tokens, tool calls, and elapsed time;
- rollback/restart/model-switch frequency;
- success after each recovery action;
- policy position on the completion-versus-cost Pareto frontier.

The task list, model endpoints, policy code, thresholds, and per-run limits are
frozen before the full run. Oracle output is used only for harness validation,
never as supervisor input. A task with an infrastructure error is rerun under
the same assigned strategy and is not counted as a model failure.

## Next teachable checkpoint

Implement the Harbor agent adapter and pass a one-task paid smoke test on x86
cloud infrastructure. Before that paid call, record the exact model endpoint,
maximum tokens, maximum turns, per-call cap, per-trial cap, and sandbox ceiling.
