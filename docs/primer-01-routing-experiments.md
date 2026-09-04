# Primer 01: Building a model router for long-horizon agents

This primer explains the system we are about to build and the experiment that
will tell us whether it is worth training. It should take about 20 minutes.

## 1. The product problem

A long-running coding agent does not perform one uniform kind of work. During a
single task, it may need to:

1. understand an unfamiliar repository;
2. form a plan under uncertainty;
3. make mechanical edits;
4. run and interpret tests;
5. recover from an unexpected failure;
6. verify and summarize the result.

The best model for step 2 may be unnecessarily expensive for steps 3 and 6.
The cheapest model for step 3 may be unable to recover during step 5.

Our product question is therefore not:

> Which model is best at coding?

It is:

> Given the current state of this task, which model should handle the next
> portion of work so that we preserve completion while controlling cost?

This is a **sequential decision problem**. Each decision affects the future
state. A poor model choice may waste money, damage the workspace, or create a
harder recovery problem for the next model.

## 2. The five pieces of the system

### Environment

The environment is the world the agent acts upon:

- repository and filesystem;
- shell and tools;
- running processes;
- tests and task verifier;
- time and resource limits.

The environment persists when models change. Switching from an 8B worker to a
frontier model should not reset the repository.

### Agent harness

The harness runs the interaction loop:

```text
send state to model -> receive action -> execute tool -> return observation
```

Mini-SWE-Agent, OpenHands, Codex, and OpenCode are examples of harnesses. The
harness decides how tools are exposed, how context is constructed, and when the
run ends. This matters because a reported benchmark score belongs to a
**model-plus-harness combination**, not to the model alone.

### Worker model

The worker model performs the coding work. We expect a pool containing a cheap
small model, a capable middle tier, and a frontier reasoner. These are the
models being routed; they are not necessarily the model we train.

### Supervisor

The supervisor observes normalized state and estimates the consequences of
using each worker model. It does not edit the repository itself.

For each candidate model `m` and task state `s`, it should eventually estimate:

```text
P(complete | state=s, next_model=m)
E(remaining_cost | state=s, next_model=m)
```

The first implementation uses transparent rules. A later implementation will
replace the estimator with a trained small model.

### Verifier

The verifier determines whether the task is actually complete. For coding tasks,
this should usually be executable: tests, builds, linters, or task-specific
checks. A model saying "done" is not a verifier.

## 3. How the frameworks divide the work

```text
Harbor task / benchmark
          ↓
Prime Intellect Verifiers environment
  - taskset
  - harness
  - sandbox and tools
  - reward and metrics
          ↓
Our supervisor
  - normalized state
  - completion/cost estimates
  - product policy
          ↓
NVIDIA Switchyard
  - compatible model API
  - backend selection
  - provider translation
  - usage telemetry
          ↓
Worker models
```

The boundaries are intentional:

- **Verifiers** tells us what task is being attempted, how the agent acts, and
  whether the result passes.
- **Switchyard** executes the selected model route.
- **Our supervisor** decides which route is appropriate from task and budget
  state.
- **Harbor/ATIF** gives us benchmark tasks and a portable trajectory format.

If our core policy imports internal classes from one agent harness, it is not
portable. Each harness should instead translate its events into our small shared
event vocabulary.

## 4. Prediction versus policy

This distinction is central.

The estimator predicts what is likely to happen:

| Model | Estimated completion probability | Forecast remaining cost |
|---|---:|---:|
| 8B worker | 0.62 | $0.20 |
| Middle tier | 0.84 | $1.10 |
| Frontier | 0.92 | $5.00 |

The product policy expresses what risk the user accepts. Suppose the required
completion probability is `0.80`.

The 8B model is rejected because `0.62 < 0.80`. Both remaining models qualify,
so the policy chooses the middle tier because it is cheaper.

Formally:

```text
choose the model with minimum forecast cost
subject to estimated completion probability >= reliability threshold
```

If no affordable model clears the threshold, choose the most reliable
affordable option and flag that the guarantee is not met. If nothing is
affordable, stop or ask for more budget rather than silently overspending.

Why separate these?

- We can change the user's risk preference without retraining.
- We can recalibrate predictions without rewriting product logic.
- We can explain a decision by showing estimates, threshold, budget, and route.

## 5. What counts as task state?

The supervisor should not depend on hidden chain-of-thought. It should use
observable evidence:

- original objective and constraints;
- current phase and milestone;
- whether a concrete plan exists;
- recent tool calls and results;
- files changed and Git diff metadata;
- test counts and whether they are improving;
- repeated failures or repeated actions;
- time, tokens, and money spent;
- current model and recent model switches;
- remaining context and budget.

Some of this can be computed deterministically. Test trends, repeated commands,
and cost do not require a language model. A small trained supervisor should focus
on residual semantic questions such as whether the plan still explains the
failure or whether new evidence materially reduces uncertainty.

## 6. Why we need baselines

A trained router is valuable only if it beats simpler alternatives.

The important baselines are:

1. **Fixed model:** use one model for the whole run.
2. **All-frontier:** expensive quality reference.
3. **Static handoff:** frontier plans, cheap model executes, frontier reviews.
4. **Switchyard stage router:** existing rule-based dynamic routing.
5. **Our explicit rules:** milestones, validation trends, recovery, and budget.
6. **Learned supervisor:** added only after the earlier results show headroom.

If one middle-tier model is both cheaper and more successful than the dynamic
systems, we should use that model instead of inventing routing complexity.

## 7. The Pareto frontier

For each strategy, record:

- completion rate;
- mean cost per task;
- cost per completed task;
- time per completed task;
- frontier-token share;
- unsafe downgrades and missed escalations.

Imagine these results:

| Strategy | Completion | Mean cost |
|---|---:|---:|
| Always 8B | 45% | $0.20 |
| Always middle | 70% | $1.20 |
| Static handoff | 68% | $1.60 |
| Dynamic router | 80% | $2.00 |
| Always frontier | 81% | $6.00 |

Static handoff is **dominated** by always-middle: it costs more and completes
fewer tasks. Always-frontier is not strictly dominated by the dynamic router
because its completion is one point higher, but the extra point costs $4 per
task. Whether that is worthwhile is a product decision and requires uncertainty
estimates around the measured completion rates.

The non-dominated strategies form the Pareto frontier. Our completion-first
default should be the cheapest point whose completion is acceptably close to
the quality reference—not the strategy with the largest success-per-dollar
ratio.

## 8. Why routing experiments are difficult

At a checkpoint, suppose we continue with the 8B model and it fails. We do not
automatically know that switching to the frontier model would have succeeded.
That unobserved outcome is a **counterfactual**.

The strongest evidence comes from branching the same checkpoint:

```text
saved task state
   ├── continue with 8B     -> fails, costs $0.20
   ├── switch to middle    -> passes, costs $0.90
   └── switch to frontier  -> passes, costs $4.00
```

For that checkpoint, the middle model is the cheapest observed successful
choice. But branching every checkpoint through every model would be expensive.
We therefore:

1. run pure-model baselines;
2. identify informative success/failure regions;
3. branch only a small number of high-value checkpoints;
4. call the best observed branch an **empirical oracle**, not a true oracle.

Offline replay is useful for testing policy logic, but it cannot perfectly
predict the future trajectory created by a different model.

## 9. How we decide whether to train

Training is justified only if all of the following are true:

1. Different worker models are best in different states.
2. One fixed middle-tier model does not dominate.
3. Existing rules leave a meaningful gap to the empirical oracle.
4. Observable checkpoint state predicts future outcomes.
5. Routing savings exceed inference and context-handoff overhead.

If those conditions hold, we train the smallest useful estimator:

```text
rules -> boosted trees -> 0.5B LoRA -> 1.5B LoRA -> larger only if justified
```

The trained model should first predict outcomes, not memorize route commands.
The deterministic product policy will convert predictions into routing actions.

## 10. What we will build next

The next checkpoint is one real, inspectable task passing through the system:

1. load an executable coding task;
2. create a persistent sandbox;
3. run a simple agent harness;
4. translate live events into `SupervisorState`;
5. make a routing decision at each model turn;
6. run the verifier;
7. save the trajectory, decisions, costs, and outcome.

We will first use a mock or local endpoint. This validates the integration
without confusing infrastructure bugs with model capability or spending money.

## Official reading path

Read these in order. The question beside each link is more important than
memorizing the API.

1. [Prime Intellect Verifiers overview](https://docs.primeintellect.ai/verifiers/overview)
   - Question: What are the three things an environment must contain?

2. [Prime Intellect environments](https://docs.primeintellect.ai/verifiers/environments)
   - Question: Which state belongs to the taskset, harness, rollout, and rubric?

3. [NVIDIA Switchyard stage router](https://nvidia-nemo.github.io/Switchyard/routing_algorithms/stage_router_routing/)
   - Question: What observable signals push a turn toward the capable or
     efficient model, and where would those signals fail?

4. [Harbor Agent Trajectory Interchange Format](https://www.harborframework.com/docs/agents/trajectory-format)
   - Question: Which fields let one trajectory remain useful for evaluation,
     training, and cost analysis across different harnesses?

5. [Prime Intellect training guidance](https://docs.primeintellect.ai/verifiers/training)
   - Question: Why are near-zero baseline reward, near-perfect baseline reward,
     and no reward diversity all warning signs before RL?

6. [RouteLLM paper](https://arxiv.org/abs/2406.18665)
   - Question: What training signal makes a router different from a generic
     prompt-difficulty classifier?

## Check your understanding

Before the next build, try answering these without looking back:

1. Why does a model switch not require copying the repository?
2. Why do we estimate completion probability instead of directly training
   `STEP_UP` and `STEP_DOWN` labels?
3. Why can an offline policy replay not prove that a different model would have
   completed the task?
4. When is a strategy dominated on the Pareto curve?
5. What experimental result would convince us not to train a supervisor?
