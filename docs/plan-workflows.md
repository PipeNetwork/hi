# Plan workflows: `hi workflow run plan.md`

The local plan runner drives a markdown plan of objectives through the
workflow engine (`hi-agent-runtime::WorkflowExecutor`): each objective runs as
an isolated, verified delegate child; a final trusted verification gate runs
the whole-workspace pipeline; every wave seals a resumable checkpoint.

This is the ADR-001 self-hosted carve-out: verification reports are signed
`local-signed:` (ed25519, local key) and are never managed RSI evidence.

## Usage

```bash
hi workflow run plan.md --dry-run          # parse check: list extracted objectives
hi workflow run plan.md                    # build it
hi workflow run plan.md --verify "cargo test --quiet" --parallel 4 --retries 1
hi workflow run plan.md --check-off       # mark succeeded objectives [x] in the plan
hi workflow resume plan.md                 # continue the latest sealed checkpoint
```

From the TUI: `/workflow plan <plan.md> [flags]` runs the same engine as a
detached child (`/workflow plan status`, `/workflow plan stop`).
`/goal --workflow <plan.md>` is the interactive alias for that command.

## `/goal` and plan.md

`/goal` ingests a checklist `.md` (unchecked boxes, numbered items, or at
least two bullets) as structured sub-goals before calling the planner.
`- [x]` rows start `Done` so a rerun does not redo them.

- Interactive `/goal implement plan.md` drives in-session (skip stalled
  steps; one second pass if anything completed).
- `/goal unattended on` (or `/goal --unattended <objective>`) elevates
  Goal-drive turns to Always (auto-approve mutations) for that goal’s
  life, then restores the previous permission mode. Esc still pauses.
- `/goal --workflow plan.md` detaches `hi workflow run` instead of
  installing an in-session goal. `--unattended` is irrelevant there
  (the child is already headless Always). Prefer this when the repo is
  git-backed and a verify command exists; otherwise `/goal` drives
  in-session.
- Plain one-shot `hi --goal "implement plan.md"` with a solid checklist
  also hands off to `hi workflow run`.
- Fleet `--session-file` / `--goal` children already have a worktree:
  they ingest and drive in-process. `--workflow` is refused.

## Plan format

Objectives are unchecked checkboxes first, else numbered items, else bullets;
`- [x]` items are respected as done:

```markdown
- [x] set up the crate
- [ ] add a `--seed` flag to the trainer, with a unit test
- [ ] tokenize the corpus into shards, with a round-trip test
```

Objectives should be concrete and test-gated — each becomes a standalone
delegate prompt. Prose documents (PRDs) parse badly; always `--dry-run` first.
`hi workflow run` **fails closed** on meta or vague process-only rows
("investigate the parser", "Final workspace validation"). Interactive `/goal`
warns and still drives those rows. Cap: 512 objectives per plan.

## Execution model

- Graph: `intake → ingest_plan → scatter ⇉ objective_NNNN ⇉ join →
  objectives_gate → verify → complete/failed`.
- Objectives run in waves of `--parallel` (default 4); the cross-process
  resource governor additionally caps live children machine-wide. Dependent
  objectives need `--parallel 1` (waves apply sequentially — each objective
  sees its predecessors' merged changes).
- An objective "passes" only when its delegate's diff was independently
  verified and merged: applied AND verified, never narrated. `--retries N`
  (max 3) re-runs a failed objective with the previous failure summary in the
  prompt. `--bestof N` (2–4) escalates an objective that exhausted its
  retries: N diverse candidates run in parallel worktrees and the gate merges
  at most one verified winner — serial retries share the failed attempt's
  framing; diverse candidates don't.
- Failed objectives don't abort the run; they flow to the objectives gate,
  which fails the workflow at the end and names them. Completed objectives'
  changes stay applied — with `--check-off` they're marked `- [x]` in the plan
  automatically, so a rerun only retries what failed. A plan whose objectives
  are all checked succeeds immediately with nothing to do.
- Checkpoints are keyed by plan content hash under the project state root;
  editing the plan starts a fresh run rather than resuming a mismatched graph.

## Model routing

`HI_IMPLEMENTER_MODEL=<model>` routes objective delegates to a different
(usually faster) model than the session default. The verification gate is
model-agnostic — a cheaper implementer cannot lower the bar for what merges;
it can only fail more often. Best-of escalation (`--bestof`) still uses the
session model.

## Requirements

- A git repository (objectives execute in isolated worktrees).
- A verification pipeline: auto-detected for Rust workspaces, otherwise pass
  `--verify "<command>"`. Both the per-objective gate and the final workspace
  gate run it.

See also: `docs/orchestration-operations.md` (scheduler and governor),
`docs/adr/001-rsi-runtime-boundary.md` (why local runs are labeled
unattested), `docs/workflow-engine-proposal.md` (the scripted `/workflow
<name>` engine — a separate system).
