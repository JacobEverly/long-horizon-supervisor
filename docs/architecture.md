# Architecture: portable supervision over persistent external state

## Ownership boundaries

| Layer | Owns | Does not own |
|---|---|---|
| Environment/harness | Filesystem, Git checkout, processes, tools, tests | Product routing policy |
| Harness adapter | Translation of native events into normalized events | Model selection |
| Supervisor | Task progress, validation trend, budget, model history | Filesystem contents |
| Switchyard adapter | Applying a selected route and collecting usage | Task semantics |
| Verifier | Whether the completed task is correct | Which model should act |

The external workspace persists when the model changes. The supervisor records
references and summaries—workspace ID, working directory, Git revision, diff
hash, changed files, test results—but it does not duplicate the repository.

## Live event flow

```text
1. Agent requests an LLM turn.
2. Harness adapter emits normalized events accumulated since the last turn.
3. Reducer updates SupervisorState.
4. Recovery policy decides whether to continue, roll back, restart clean, or
   accept verified completion.
5. If work continues, the estimator scores every available model for the
   resulting state.
6. Routing policy chooses a model.
7. Switchyard sends the request to that backend.
8. Harness executes tool calls in the same persistent workspace.
9. Results become the next normalized events.
```

The first implementation uses an in-process Python library. A later sidecar can
expose the same `observe`, `snapshot`, and `decide` operations over HTTP without
changing the state or policy contract.

The held-out evaluation uses Harbor's standard Terminus-2 agent with Switchyard
as its model gateway. Harbor remains the owner of sandboxes and verification;
Switchyard supplies per-turn routing without modifying the agent or task. This
keeps the integration portable across Harbor environment providers. Paid Gate 7
trials run in x86 cloud sandboxes; local Docker is only a zero-cost harness smoke
path.

## Portable event contract

The normalized vocabulary is intentionally small:

- `task_started`
- `phase_changed`
- `plan_committed`
- `tool_result`
- `files_changed`
- `validation_result`
- `milestone_completed`
- `context_compacted`
- `task_finished`

Framework-specific detail remains in optional metadata. An adapter may have
richer internal events, but the policy must not import harness-native classes.

Harbor's Agent Trajectory Interchange Format (ATIF) is the durable log and
training interchange format. The normalized events are the live control-plane
format. A completed event stream can be exported to ATIF, while ATIF trajectories
can be replayed through the reducer for offline training and evaluation.

The first adapter uses Verifiers 0.3.0 `StatefulToolEnv`. Every rollout receives
its own copied workspace that persists for all model turns. The model can list,
read, replace, and write workspace-relative files and run only the fixed public
test command. Path validation prevents `..` or absolute-path escape. Hidden
tests remain outside the agent workspace and determine the final reward.

Each model response becomes one ATIF v1.7 agent step. Tool observations are
correlated using tool-call IDs, and per-step token cost is calculated against a
timestamped OpenRouter catalog snapshot. Gate 1 exports pass Harbor's official
trajectory validator.

## Context handoff

The receiving model gets:

1. original objective and constraints;
2. current phase and committed plan;
3. completed and active milestones;
4. verified evidence and recent validation trend;
5. recent failures and rejected approaches;
6. next recommended action;
7. access to the persistent workspace.

We pass conclusions and evidence, not hidden chain-of-thought. Raw trajectory
history remains retrievable, but a multi-hour transcript is not injected into
every request.

### Evidence is bound to state

Every public-test result carries the digest of the workspace that produced it,
its observation time, and whether it was produced freshly in the current run.
Any file mutation invalidates prior verification for completion purposes. A
result from another digest may remain useful historical context, but it cannot
be presented as evidence that the mounted files are correct.

The handoff renderer labels prior evidence as either `STALE FOR THIS STATE` or
`STATE-MATCHED BUT NOT FRESH`. Neither satisfies the deterministic completion
guard. The receiving model must run verification in its own rollout after its
final edit.

### Turn checkpoints

The Verifiers adapter snapshots the workspace at initialization and after each
tool turn or guarded completion attempt. Each checkpoint records its workspace
digest, turn, reason, public-test status, and whether the public evidence is
valid for that exact checkpoint. This makes rollback an explicit, auditable
action rather than an attempt to reconstruct files from a transcript.

## Why prediction and policy are separate

The estimator answers an empirical question:

```text
How reliable is model M from state S, and what will it cost?
```

The policy answers a product question:

```text
How much completion risk will we accept for this user and budget?
```

Separating them lets us recalibrate predictions or change product thresholds
without retraining both at once. It also makes a decision explainable: we can
show the model estimates, threshold, forecast cost, and chosen route.
