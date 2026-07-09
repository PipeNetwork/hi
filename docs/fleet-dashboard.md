# `/dashboard` — control a fleet, not an agent

A full-screen TUI mode for dispatching, monitoring, and steering multiple concurrent agent
sessions from one screen. Type a prompt in the dispatch box, hit Enter, and you've launched
another agent without leaving the screen — do it five times in a row. Prefix a dispatch with
`/goal ` and the row becomes an autonomous multi-turn worker on a week-sized objective.

```
┌ hi fleet · 5 agent(s) · 3 working ─────────────────────────────────┐
│ ⠹  #1  2m14s ↓23.1k  ◎2/4              write cli.py                │
│ · ●#2  3 turn(s) ↓8.4k          ⇡held   fix the flaky retry test   │
│ ⠹  #3  0m41s ↓31.9k  ◎3/7   ⟳          port the parser to Rust     │
│ ·  #4  1 turn(s) ↓2.0k          ✓3      update the docs            │
├ #3 · port the parser to Rust ──────────────────────────────────────┤
│ ✓ merged 2 file(s) into your tree: src/lexer.rs, src/token.rs      │
│ ⟳ goal drive                                                       │
├ dispatch — Enter spawns · /goal <obj> = driven · Ctrl+S attaches ──┤
│ › ▌                                                                │
└ ↑↓ · Tab reply · 1-9 answer · m merge · r rebase · x close · Esc ──┘
```

## The isolation & merge model

Every row gets its **own git worktree**, checked out to a snapshot of your tree at dispatch
(uncommitted work included). Each turn runs as a child `hi` in that worktree, resuming the
row's own session file — so anything you dispatch is individually resumable later with
`--resume`. Your real tree only ever receives **verified, non-overlapping diffs**:

1. **Turn ends** → the row's diff vs its base is computed (Python bytecode stripped; binary
   assets handled).
2. **Verify gate** — when the session has a verify command, it must pass *in the worktree*
   before anything merges.
3. **Overlap hold** — if the diff touches files another open row changed, the merge is held
   visibly (`⇡held`); `m` forces it.
4. **Auto-merge** — otherwise the diff applies to your tree, the *combined* tree is
   re-verified (a diff can pass in isolation yet break the combine), and the row's base
   auto-refreshes so future diffs stay minimal.
5. Other rows get a `⟳ stale` badge after any merge; `r` rebases an idle row onto the
   current tree (refused while it has unmerged changes).

Failed, killed, or abandoned rows never touch your tree. Worktrees are cleaned on row close
and TUI exit; sessions always persist.

## Goal-driven rows

Dispatch `​/goal <objective>` and the child plans the objective (the profile's planner model,
glm-5.2 on pipenetwork) before its first turn. After each successful turn the dashboard reads
the child's report: while the goal is **Active**, it auto-continues the row with the standard
goal-drive prompt — the same engine as the chat `/goal` auto-drive, with the same stall guard
(two no-progress drive turns park the row with a `●` attention badge; any reply resumes).
The `◎ done/total` column tracks progress live.

## Keys

| key | does |
|---|---|
| `Enter` (dispatch box) | spawn a new agent for the typed prompt (`/goal ` prefix = goal-driven row) |
| `Ctrl+S` | dispatch **and** attach (full-screen view of that row); with an empty box, attach the selected row |
| `↑`/`↓` | select a row (peek panel follows) |
| `Tab` | focus the selected row's reply input ↔ back to dispatch |
| `Enter` (reply) | send to the row — runs now if idle, queues FIFO if working |
| `1`–`9` | instant reply on an idle row with an empty reply box (answer "1) … or 2) …?" in one keystroke) |
| `m` | force-merge the selected row's held/unverified diff |
| `r` | rebase an idle row's worktree onto a fresh snapshot (clears `⟳ stale`) |
| `x` | close a row: worktree removed, session kept resumable |
| `Ctrl+K` | kill the selected row's in-flight turn |
| `PgUp`/`PgDn` | scroll the peek panel's output tail |
| `Esc` | leave attach → table → exit; double-Esc kills in-flight turns (sessions survive) |

A `●` badge (plus a terminal ping when the window is unfocused) marks rows waiting on you:
an agent that asked a question, a held merge, a failure, or a parked drive.

## Ground truth per turn

Every child turn writes a `--report` JSON the dashboard consumes: session-cumulative token
totals (the `↓` column), `verify_passed`, `changed_files`, and a `goal` block
(`objective/done/total/status/paused`) that drives auto-continue.

## Validated behavior

The engine is exercised by two live harnesses (both need `PIPENETWORK_API_KEY`):

- `scripts/fleet_live_test.sh` — five concurrent rows on a real mini-project with a
  deliberate file collision: disjoint work auto-merges, the collision holds and
  force-merges, the combined suite goes failing → green, all sessions stay resumable.
- `scripts/goal_challenge_live.sh` — one `/goal`-driven row builds a complete kanban CLI
  (13 planned sub-goals) across multiple capped turns: verify-gated merge lands mid-drive,
  the drive stops itself at Done, and the finished tool passes 79 tests.

These harnesses found (and their fixes closed) a real bug: Python bytecode left by a child's
test runs used to break merge-back entirely.

## Known limits

- Full-screen TUI only (`hi` without `--plain`); the plain REPL prints a notice.
- A completed row's later turns can leave runtime artifacts (e.g. an app's own data file
  written by its tests) that fail to re-apply after the substance already merged — the row
  shows the `git apply` reason; `x` closes it. Artifact-aware filtering beyond bytecode is
  future work.
- Rows live for the TUI session (worktrees are per-process); their *sessions* are permanent
  and resumable.
