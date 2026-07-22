# Workflow engine for `/dashboard` — design proposal

## Why now

`/dashboard` already runs a fleet: N worktree-isolated child `hi` sessions, each
with its own row, merge gate, goal drive, and peek panel. What it lacks is
**orchestration structure** — the fleet is a flat list of independent rows. You
dispatch five prompts and watch five isolated agents; there's no notion of a
multi-phase plan that coordinates them, no shared scratch state, no budget
across the fleet, no replayable journal, and no phase trail in the UI.

grok-build's `xai-workflow` crate solves exactly this. It's a deterministic,
replayable, Rhai-scripted workflow engine where a script declares phases, spawns
agents (serially or in parallel), pauses for user input, tracks a token/agent
budget, and journals every host call so a run can be resumed after a crash or
cancel. The pager side renders it as a rich workflows overlay: a run list with
phase trails (✓ done · ● active · ○ pending), per-agent rows grouped by phase,
budget meters, and pause/resume/stop management actions.

This doc maps what grok-build has, what hi has today, and what we'd add to hi to
close the gap — phased so each step is independently shippable.

---

## What grok-build has (the `xai-workflow` crate)

### Core engine (`engine.rs`, `run.rs`, `host.rs`)

A workflow is a **Rhai script** with a required `let meta = #{ ... };` header.
The engine compiles it, injects `args` as a JSON value, and executes. Host calls
(agent spawns, phase markers, logs, budget queries, template renders, scratch
file I/O, git diffs) are the only way the script touches the outside world — and
every one is **journaled** for deterministic replay.

**Scripting API (Rhai functions registered by the engine):**

| Function | Purpose |
|---|---|
| `agent(prompt, opts?)` | Spawn one agent, block until it returns `AgentResult` |
| `parallel(jobs)` | Spawn N agents concurrently, block until all finish, return array |
| `phase("Title")` | Emit a phase marker (host gets `WorkflowHostRequest::Phase`) |
| `log(message)` / `print` / `debug` | Emit a log line (host gets `WorkflowHostRequest::Log`) |
| `telemetry_event(name, fields)` | Structured telemetry event |
| `budget()` | Query `BudgetState { total, spent, reserved, remaining }` |
| `complete(value)` | End the workflow with a result |
| `pause(kind, message)` | Pause: `user`, `back_off`, `no_progress`, `verification`, `infra` |
| `await_user(kind, message)` | Pause that's journaled as a host call (replayable) |
| `render_template(name, vars)` | Host renders a prompt template |
| `write_scratch_file(name, content)` / `read_scratch_file(name)` | Shared scratch space |
| `git_diff_since(commit)` | Get the diff since a commit |
| `fingerprint(text)` | Stable hash (pure, deterministic) |
| `json_encode(value)` | Safe JSON encoding (quotes untrusted strings) |

**Determinism guards:** `timestamp()`, `sleep()`, and `exit()` are all
**disabled** — they raise runtime errors. The script must be pure; wall-clock
time and nondeterminism break resume. Timestamps come in via `args`.

**Outcome model (`WorkflowOutcome`):**
```
Completed { result } | Paused { kind, message } | BudgetExceeded { message } | Cancelled | Failed { error }
```

**Agent options (`AgentOpts`):** prompt, label, model, max_output_tokens,
agent_type, capability_mode, isolation_worktree, fork_context, resume_from,
output_schema, **phase**.

**Agent result (`AgentResult`):** agent_id, success, output (JSON), cancelled,
tokens_used, duration_ms.

### Journal (`journal.rs`)

Append-only JSONL, one `JournalEntry` per host call:
```
{ seq, kind, req_hash, result, at_ms }
```
- `seq` is dense (0, 1, 2, …) — gaps are a `Sequence` error.
- `req_hash` is a stable hash of `(kind, request_payload)` with sorted map keys,
  so replay can detect **divergence** — if the script issues a different call
  than the journal recorded, the run fails with a `Divergence` error (the script
  is nondeterministic or was edited mid-run).
- Replay: on resume, the engine replays journaled results without calling the
  host; live calls resume from where the journal ends.
- 64 MiB / 10k-entry cap — a journal that would exceed the cap fails with `Full`
  so a run is never stranded unresumable.

### Meta + validation (`meta.rs`, `validate.rs`)

`extract_meta` parses the `let meta = #{ ... };` header (must be the first
statement), validates it, and returns:
```
WorkflowMeta { name, description, when_to_use?, phases: Vec<PhaseMeta { title, detail? }> }
```
- `name`: lowercase ASCII, hyphen-separated (slug).
- Limits: 64 phases max, 128-char phase titles, 1024-char descriptions.
- `validate_script` does a **dry-run** with a stub host (all agents return
  canned success) to verify the script compiles, the meta is well-formed, the
  outcome is `Completed`, and the agent budget isn't exceeded.

### Pager rendering (`scrollback/blocks/workflow.rs`, `views/workflows.rs`)

**Scrollback block** (inline in the conversation):
```
Workflow deep-research: ○ Scan · ● Analyze · ○ Synthesize (3 agents)
```
- Phase trail: `✓` done, `●` active, `○` pending, joined with ` · `.
- Status verbs: running (name + agent count), done (duration), failed, cancelled
  (dim `◌`, not failure-colored), paused.

**Full-screen workflows view** (`WorkflowRunSnapshot`):
- Run list: name, objective, status, phase trail, active agent count, elapsed.
- Detail panel: per-phase agent rows (`WorkflowAgentRowView`: agent_id, label,
  phase, model, state, tokens_used), budget meter (`agents_used / reserved /
  remaining`), pause message, result summary.
- Management actions: pause / resume / stop / save — gated by `can_pause`,
  `can_resume`, `can_stop`, `can_save` based on status + `management_available`.
- Mouse + keyboard: click a phase to pin it, click a run row to open detail,
  `Enter` drills in, `q` closes the overlay.

---

## What hi has today (`/dashboard`)

### Fleet model (`dashboard.rs`, `dashboard_goal.rs`)

**`FleetRow`** — one row = one worktree-isolated child `hi` session:
- `id`, `title`, `worktree`, `base` (snapshot commit), `session` file.
- `state: RowState` — `Working | Idle | Failed | Closed`.
- `merge: MergeState` — `None | Merged(n) | Held(reason) | Failed(reason)`.
- `changed: Vec<String>` — files changed vs base.
- `activity` (last output line), `tail: Vec<String>` (capped at 200 lines).
- `pending: VecDeque<String>` — follow-ups queued while a turn runs.
- `reply: InputLine` — per-row reply input (peek panel).
- `kill: Option<oneshot::Sender<()>>` — kills in-flight child turn.
- `started: Option<Instant>`, `turns: u32`, `usage: u64` (session-cumulative
  tokens from child's `--report` JSON).
- `goal: Option<RowGoal>` — `done / total / active / paused` from report.
- `goal_objective`, `last_goal_json`, `driving`, `drive_stall` — autonomous
  `/goal` drive state.
- `stale` (another row merged), `attention` (waiting on user).

**Rendering:** header → fleet table (up to 8 rows) → peek/attach panel →
dispatch input → footer hints. Each table row shows: spinner glyph, attention
badge, `#id`, elapsed, queued marker, `↓tokens`, `◎done/total` goal progress,
`⟳` stale, merge badge, truncated activity lead.

**Isolation & merge:** every row gets its own git worktree at dispatch. Each
turn is a child `hi` run in that worktree. On success: diff vs base → verify
gate → overlap hold → auto-merge into real tree → re-verify combined tree →
refresh base. Failed/killed rows never touch the real tree.

**Goal drive:** `/goal <objective>` → child plans via `--goal` → dashboard
auto-continues turns while goal is active and not paused → stalls after
`GOAL_DRIVE_STALL_LIMIT` unchanged drive turns → user reply resets stall.

### What's missing vs grok-build

| Capability | grok-build | hi `/dashboard` |
|---|---|---|
| **Phased plans** | `phase("Title")` markers, `WorkflowMeta.phases` | ❌ flat rows, no phases |
| **Multi-agent coordination** | `parallel(jobs)`, serial `agent()` chains | ❌ rows are independent |
| **Shared scratch state** | `write_scratch_file` / `read_scratch_file` | ❌ each worktree is isolated |
| **Fleet-wide budget** | `BudgetState`, `reserve_agent_calls` | ❌ per-row token count only |
| **Replayable journal** | JSONL journal, divergence detection, resume | ❌ sessions persist but no workflow-level replay |
| **Deterministic scripting** | Rhai script with disabled `timestamp`/`sleep`/`exit` | ❌ no scripting layer |
| **Phase trail UI** | `✓ ● ○` trail in scrollback + full view | ❌ just `◎done/total` |
| **Per-agent rows in a phase** | `WorkflowAgentRowView` grouped by phase | ❌ one row = one agent, no grouping |
| **Pause/resume/stop management** | `can_pause`, `can_resume`, `can_stop` + `/workflow` commands | ⚠️ kill + reply only |
| **Workflow validation/dry-run** | `validate_script` with stub host | ❌ none |
| **Template rendering** | `render_template(name, vars)` | ❌ none |

---

## What to add to hi — phased proposal

### Phase 1: Phase trail + structured goal display (UI only, no engine)

**Goal:** Make the existing fleet rows show phase-level progress instead of just
`◎done/total`.

**Approach:** The child `hi` already writes a `--report` JSON with a `goal`
block. Extend that report to include an optional `phases: [{ title, state }]`
array (the agent's planner already produces sub-goals — surface them as phases).
The dashboard's `RowGoal` gains a `phases: Vec<(String, String)>` field. The
table row renders a compact phase trail (`✓ ● ○`) when present, falling back to
`◎done/total` when not.

**Files touched:**
- `hi/crates/hi-tui/src/dashboard_goal.rs` — add `phases` to `RowGoal` + parse
  from report.
- `hi/crates/hi-tui/src/dashboard.rs` — render phase trail in `render_table`.
- `hi/crates/hi-agent/src/` — emit phases in the report's goal block.

**Effort:** Small. Pure UI + report schema extension. No new crate.

### Phase 2: A `hi-workflow` crate (the engine, adapted)

**Goal:** Port the core of `xai-workflow` — the Rhai engine, journal, meta, and
host protocol — as a new `hi-workflow` crate that hi's dashboard can drive.

**What to port directly:**
- `journal.rs` — the JSONL journal, `JournalEntry`, `request_hash`, divergence
  detection, dense-sequence enforcement, 64 MiB cap. This is ~400 lines and
  almost entirely portable (only depends on `serde`, `sha2`, `tempfile`).
- `meta.rs` — `WorkflowMeta`, `PhaseMeta`, `extract_meta`, validation. ~300
  lines, depends on `rhai` + `serde`.
- `host.rs` — `AgentOpts`, `AgentResult`, `BudgetState`, `HostError`,
  `WorkflowHostRequest`. ~110 lines, pure types.
- `run.rs` — `PauseKind`, `WorkflowOutcome`. ~50 lines.
- `engine.rs` — `run_workflow`, `WorkflowRunParams`, the Rhai function
  registrations, replay logic. ~850 lines. This is the core; it depends on
  `rhai`, `tokio` (channels + `CancellationToken`), `tokio-util`.
- `validate.rs` — `validate_script` dry-run with stub host. ~300 lines.

**What to adapt for hi:**
- `AgentOpts.isolation_worktree` maps directly to hi's worktree-per-row model —
  the workflow host spawns child `hi` runs in worktrees, exactly as the
  dashboard does today.
- `AgentOpts.resume_from` maps to hi's `--session-file` / `--resume` — a
  workflow agent can resume an existing session.
- `AgentOpts.phase` lets the script tag which phase an agent belongs to — the
  dashboard can group agent rows by phase.
- `WorkflowHostRequest::GitDiffSince` maps to hi's existing merge-diff logic.
- `WorkflowHostRequest::RenderTemplate` — hi doesn't have a template system yet;
  either skip it initially or wire it to hi's profile/prompt system.
- `WorkflowHostRequest::WriteScratchFile` / `ReadScratchFile` — shared scratch
  space outside any worktree; useful for cross-agent handoff (e.g. agent A
  writes a plan, agent B reads it).

**New `Cargo.toml` deps:** `rhai` (with `serde` feature), `sha2`, `tokio-util`.
All are standard and don't conflict with hi's existing dependency tree.

**Workspace registration:** add `hi-workflow = { path = "crates/hi-workflow" }`
to `[workspace.dependencies]` in `hi/Cargo.toml`.

**Effort:** Medium-large. The code is proven and well-tested (the grok-build
crate has extensive unit tests for journal, replay, divergence, budget,
parallel, pause/resume). Porting is mostly mechanical (adjust imports, remove
grok-specific hints). The tests port too.

### Phase 3: Dashboard integration — workflow runs as a fleet mode

**Goal:** Let `/dashboard` launch and render a **workflow run** (a scripted
multi-phase, multi-agent plan) alongside its current flat-row fleet mode.

**Approach:**

1. **New dispatch prefix:** `/workflow <name> [args...]` in the dispatch box
   loads a workflow script (from `~/.hi/workflows/<name>.rhai` or a built-in),
   validates it, and starts a `run_workflow` call in a background thread.

2. **Host bridge:** The dashboard becomes the workflow host. It receives
   `WorkflowHostRequest`s and maps them to its existing primitives:
   - `SpawnAgent` → create a `FleetRow` with a worktree, run the child `hi` turn,
     return `AgentResult` when the turn completes. The row's `phase` field (from
     `AgentOpts.phase`) drives grouping in the UI.
   - `Phase` → update the run's current phase; the dashboard header shows the
     phase trail.
   - `BudgetQuery` → aggregate `usage` across all rows in the run.
   - `Log` / `Telemetry` → append to the run's output tail.
   - `GitDiffSince` → use the existing merge-diff logic.
   - `WriteScratchFile` / `ReadScratchFile` → read/write in the run's scratch
     dir (outside any worktree).

3. **Rendering:** A workflow run gets a **grouped table** — one header row per
   phase (with `✓ ● ○` trail), indented agent rows beneath. The peek panel
   shows the run's phase trail + the selected agent's output. This reuses the
   `WorkflowRunSnapshot` / `WorkflowAgentRowView` rendering pattern from
   grok-build's `views/workflows.rs`, adapted to ratatui (hi already uses
   ratatui 0.29).

4. **Management actions:** Add `p` (pause), `P` (resume), `X` (stop) keybindings
   for workflow runs, gated by `can_pause` / `can_resume` / `can_stop` — same
   logic as grok-build's `workflows_overlay.rs`.

5. **Journal + resume:** The workflow journal is written to
   `~/.hi/workflows/<run-id>/journal.jsonl`. On TUI restart, `/fleet status`
   lists resumable workflow runs (alongside resumable sessions); selecting one
   reloads the journal and resumes from the last recorded seq.

**Files touched:**
- `hi/crates/hi-workflow/` — new crate (Phase 2).
- `hi/crates/hi-tui/src/dashboard.rs` — workflow run state, host bridge,
  grouped rendering, management keybindings.
- `hi/crates/hi-tui/src/dashboard_goal.rs` — extend or generalize to workflow
  phase state.
- `hi/crates/hi-cli/src/main.rs` — `--workflow <name>` CLI flag for headless
  workflow execution (no TUI needed).
- `hi/crates/hi-cli/src/commands.rs` — `/workflow` slash command.

**Effort:** Large. This is the integration layer — the engine is ready, but
wiring it to the dashboard's event loop, worktree management, and merge gate is
substantial. The payoff is that `/dashboard` goes from "N independent agents" to
"orchestrated multi-phase plans with shared state, budgets, and replay."

### Phase 4: Built-in workflow scripts + authoring UX

**Goal:** Ship ready-to-use workflows and make authoring safe.

**Built-in workflows** (Rhai scripts in `hi/crates/hi-workflow/scripts/`):
- `deep-research.rhai` — scan → analyze → synthesize (parallel analyze agents,
  one synthesizer).
- `review-and-fix.rhai` — review (parallel skeptics) → triage → fix (serial) →
  verify.
- `port-feature.rhai` — plan → implement (worktree-isolated) → verify → merge.

**Authoring UX:**
- `hi workflow validate <file>` — runs `validate_script` (dry-run with stub
  host) and reports errors with the Rhai hints (the `with_rhai_hint` function
  that explains `Expression exceeds maximum complexity` → split `+` into `+=`,
  reserved keywords → rename, `char` indexing → check `type_of`).
- `hi workflow list` — lists available workflows (built-in + `~/.hi/workflows/`).
- `hi workflow show <name>` — prints the meta (name, description, phases).

**Effort:** Small-medium. Scripts are short (50-100 lines of Rhai). The
validation infrastructure ports directly from `validate.rs`.

---

## Recommended sequencing

1. **Phase 1** (phase trail UI) — ship first, standalone, no new deps. Gives
   immediate visual improvement to the existing fleet and validates the report
   schema extension.
2. **Phase 2** (`hi-workflow` crate) — port the engine. This is the foundation;
   it's independently testable (the unit tests port) and doesn't touch the TUI.
3. **Phase 3** (dashboard integration) — the big integration. Depends on Phase 2.
   Start with the host bridge (SpawnAgent → FleetRow) and phase trail rendering;
   add journal/resume and management actions incrementally.
4. **Phase 4** (built-in scripts + authoring) — polish. Depends on Phase 2 but
   not Phase 3 (scripts can run headless via `--workflow`).

## Key design decisions to make

- **Rhai vs. a simpler DSL:** grok-build chose Rhai for Turing-complete
  orchestration (loops, conditionals, parallel fan-out). Hi could start with a
  declarative YAML/TOML phase spec (no scripting) and add Rhai later — but
  that means two orchestration models. Recommendation: port Rhai directly; the
  determinism guards and journal make it safe, and the scripting power is the
  whole point (parallel agents, conditional phases, retry loops).
- **Workflow runs vs. fleet rows:** Should a workflow run *replace* the flat
  fleet, or coexist? Recommendation: coexist. `/dashboard` stays as-is for ad-hoc
  dispatch; `/workflow <name>` launches a structured run that renders as grouped
  rows. The two modes share the same worktree/merge infrastructure.
- **Scratch file scope:** Per-run (isolated) or per-project (shared across
  runs)? Recommendation: per-run initially (matches grok-build), with a
  project-level shared dir as a future extension.
