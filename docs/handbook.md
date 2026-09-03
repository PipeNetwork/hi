# hi handbook

Full reference for hi 0.3.1. Everyday getting started lives in the [README](../README.md).

`hi` is an agentic coding tool written in Rust. Point it at any model — local or remote — and it reads, writes, and edits files and runs shell commands in your project to do what you ask.

Its distinguishing feature is **verification-in-the-loop**: give it a test command and it runs the model, checks the result, feeds failures back, and iterates until the tests pass — something a single-shot completion endpoint structurally can't do.

Workspace version **0.3.1** continues the post-0.2 core API. The intentional
0.2 break (CLI, report, and benchmark schema) is documented in the
[0.2 migration guide](0.2-migration.md). For crate layout — interactive
agent vs RSI control-plane — see [architecture.md](architecture.md).
GPU and local-inference crates live in the separately released
[`hi-local-runtime`](https://github.com/PipeNetwork/hi-local-runtime) repository;
the core workspace only discovers its optional `hi-local` sidecar through the
executable and HTTP contract described in [`local-runtime.md`](local-runtime.md).

```bash
# Fix failing tests with a local model, iterating until green:
hi "the tests in test_parser.py are failing — fix the parser"
```

## Quick start

```bash
cargo build --release           # fast core binary at target/release/hi
cargo build --release --features voice  # include microphone + local Whisper
cargo install --path crates/hi-cli --locked

# OpenRouter (default endpoint)
HI_API_KEY=sk-or-... hi -m anthropic/claude-sonnet-4 "add a --json flag to the CLI"

# pipenetwork.ai (OpenAI-compatible coding endpoint; defaults to ipop/coder-balanced)
PIPENETWORK_API_KEY=... hi --provider pipenetwork "add a --json flag to the CLI"

# A local Ollama model (no API key needed)
hi --provider ollama -m qwen2.5-coder "..."

# Native Anthropic
HI_API_KEY=sk-ant-... hi --provider anthropic -m claude-sonnet-4-20250514 "..."

# xAI (Grok); defaults to grok-4.6
XAI_API_KEY=xai-... hi --provider xai "add a --json flag to the CLI"
```

`--provider` accepts `openai` (any OpenAI-compatible URL), `anthropic`, `pipenetwork`, `ollama`, and `xai`. All but the first two are presets that set the right base URL, key env var, and — for pipenetwork and xai — a default model, so they work with no extra flags.

Run with no prompt for an interactive session; pass a prompt for one-shot. Piped stdin is folded into a one-shot prompt as context, so `hi` composes with other tools:

```bash
cargo test 2>&1 | hi "fix the failing tests"
cat error.log | hi "what's going wrong here?"
cat data.json | hi -q "extract every email address" | sort -u   # -q: text only, no chatter
```

## Models & providers

One OpenAI-compatible client covers **OpenRouter, pipenetwork.ai, Ollama, llama.cpp, LM Studio, and vLLM** — they differ only by `--base-url` and `--api-key`. A native **Anthropic** adapter (`--provider anthropic`) adds extended thinking and tool-use blocks. A native **xAI** adapter (`--provider xai`) uses the Responses API for Grok tool calling and encrypted reasoning.

Settings resolve in this order: **CLI flags → profile → environment → defaults.**

| What | Flag | Env | Default |
|---|---|---|---|
| Model | `-m, --model` | `HI_MODEL` | — (required) |
| Base URL | `--base-url` | `HI_BASE_URL` | OpenRouter / `api.anthropic.com` |
| API key | `--api-key` | `HI_API_KEY`, then provider-specific (`OPENROUTER_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `PIPENETWORK_API_KEY` / `OLLAMA_API_KEY` / `XAI_API_KEY`) | — (required; Ollama ignores it) |
| Tool mode | `--tool-mode` | — | `auto` |
| Compatibility | `--compat` | — | `auto` |
| Nucleus sampling | `--top-p` | — | unset |
| Output-token field | `--output-token-parameter` | — | `auto` |
| Trace capture | `--trace-capture metadata\|full` / `--trace-full` | `HI_TRACE_CAPTURE` | metadata |
| Execution | `--durable` | `HI_EXECUTION_MODE=durable\|ephemeral` | durable for saved ordinary sessions; ephemeral for no-save/measured runs |

### Config profiles

Keep several models on hand in `./hi.toml` or `~/.config/hi/config.toml` and use one with `-p` at startup or `/provider` mid-session:

```toml
default_profile = "sonnet"

[profiles.sonnet]
provider = "anthropic"
model = "claude-sonnet-4-20250514"
api_key_env = "ANTHROPIC_API_KEY"

[profiles.local]
provider = "ollama"
# no model field — set one later with /model
# execution = "durable"  # checkpoint prompts and completed tool batches
```

`/provider <name>` changes the active profile (base URL, API key, wire format) mid-session, then opens the model picker over the live model list. The `model` field is optional and can be set later with `/model`. `/provider add` creates a new profile interactively (in the TUI, a form with provider picker, API key, model, and base URL fields); `/provider edit [name]` modifies an existing one. Both write to your config file.

### Fallback chain

Give a profile a `fallback` list (or pass `--fallback <profile>`, repeatable); if a turn needs another configured profile, `hi` announces the handoff and retries there:

```toml
default_profile = "cloud"

[profiles.cloud]
provider = "pipenetwork"
api_key = "..."
fallback = ["local"]      # → falls back to the `local` profile

[profiles.local]
provider = "ollama"
model = "qwen2.5-coder"
```

### Compatibility

OpenAI-compatible endpoints vary in how much of Chat Completions they implement. The default `--compat auto` retries common simpler shapes, such as retrying without streamed usage metadata when a provider rejects `stream_options`. Tool calling is not silently downgraded: if a request advertises tools and the provider rejects them, the turn fails fast instead of continuing chat-only. Use `--compat strict` to send only the initial request shape. Tool availability is controlled separately with `--tool-mode auto|required|chat-only|read-only`.

| Env | Controls | Default |
|---|---|---|
| `HI_TUI_WATCHDOG_SECS` | Soft TUI "still waiting" notice (does not mark the model degraded) | 180s |
| `HI_DEBUG_STREAM` | `1` dumps raw provider bytes for diagnosing one that returns nothing | off |
| `HI_GLOBAL_PROCESS_CONCURRENCY` | Shared cross-process cap per setup/model/verifier resource class | adaptive, 2–4 |
| `HI_GLOBAL_DELEGATE_CONCURRENCY` | Delegate-specific global cap | 4 |
| `HI_PARALLEL_DELEGATES` | Maximum delegates admitted in one agent tool wave | 4, max 16 |
| `HI_DELEGATE_SESSION_LIMIT` | Optional positive per-turn delegate count cap | off (unlimited) |
| `HI_BESTOF_VERIFY_CONCURRENCY` | Parallel best-of parent verification jobs | adaptive |
| `HI_DELEGATE_QUEUE_TIMEOUT_SECS` | Optional delegate capacity wait timeout; unset or `0` waits until capacity or cancellation | off |
| `HI_DELEGATE_TIMEOUT_SECS` | Optional delegate child execution timeout; unset or `0` allows continual execution | off |
| `HI_BEST_OF_TIMEOUT_SECS` | Optional best-of candidate execution timeout; unset or `0` allows continual execution | off |
| `HI_BEST_OF_QUEUE_TIMEOUT_SECS` | Optional best-of setup/model capacity wait timeout | off |
| `HI_VERIFIER_QUEUE_TIMEOUT_SECS` | Optional shared verifier capacity wait timeout | off |
| `HI_MERGE_QUEUE_TIMEOUT_SECS` | Optional exclusive destination-merge capacity wait timeout | off |
| `HI_VERIFY_TIMEOUT_SECS` | Optional positive verification process timeout; unset or `0` allows continual execution | off |
| `HI_BASH_TIMEOUT_SECS` | Optional positive foreground shell-command timeout; unset or `0` allows continual execution (slow commands may still be handed to the background registry) | off |
| `HI_MCP_CONNECT_TIMEOUT_SECS` | Optional positive lazy MCP handshake timeout; unset or `0` waits until connection, failure, or turn cancellation | off |
| `HI_MCP_TOOL_TIMEOUT_SECS` | Optional positive MCP `tools/call` timeout; unset or `0` allows continual execution until completion or turn cancellation | off |
| `HI_MODEL_REQUEST_TIMEOUT_SECS` | Optional positive absolute deadline for one model HTTP request (including bounded retries/backoffs); unset or `0` allows continual execution | off |
| `HI_RSI_WAIT_TIMEOUT_SECS` | Optional remote RSI run wait timeout; per-request HTTP transport timeouts remain active | off |
| `HI_LOOP_TURN_TIMEOUT_SECS` | Optional `/loop` firing and auto-fix child timeout; also derives a child turn deadline | off |
| `HI_LOOP_TRIGGER_TIMEOUT_SECS` | Optional positive on-change shell trigger timeout; unset or `0` allows continual execution | off |
| `HI_HOOK_TIMEOUT_SECS` | Optional trusted lifecycle-hook process timeout; unset or `0` allows continual execution until completion or turn cancellation | off |
| `HI_SCHEDULER_PRESET` | `conservative`, `balanced`, or `throughput` orchestration policy | balanced |
| `HI_ADAPTIVE_SCHEDULER` | Set `0` to disable adaptive admission | on |
| `HI_WARM_WORKERS` | Set `0` to disable warm worker reuse | on |

## Local model sidecars

`hi-local` is an optional executable from the
[`hi-local-runtime`](https://github.com/PipeNetwork/hi-local-runtime) repository.
It serves GGUF and MLX models through the OpenAI-compatible
`/v1/chat/completions`, `/v1/models`, and `/health` API. Core resolves it from
`HI_LOCAL_BIN`, beside `hi`, or `PATH`, in that order; it never builds the GPU
runtime as a side effect of a normal harness command.

Install a matching runtime bundle or build it in the external repository, then
use the documented `hi-local serve …` command. The sidecar advertises protocol
`1.x`, backend, runtime version, and readiness from `/health`; core rejects
incompatible versions, wrong backends, crashes, and startup timeouts promptly.
See [`local-runtime.md`](local-runtime.md) for the contract, installation, and
native acceptance links.

## Verification-in-the-loop

The headline feature. After the model stops, `hi` automatically detects and runs a staged check pipeline. If a check fails, the output is fed back and the model repairs the work. Productive repair/check cycles continue until verification passes or a no-progress/fault circuit fires. Use `--max-verify-repairs N` only when an explicit finite cap is required.

```bash
hi --verify "cargo test" "make the failing test pass"
hi "..."                   # auto-detects cargo check+test, go build+test,
                           # tsc+npm test, ruff+pytest, or make test
```

Automatic verification builds a **multi-stage pipeline** per project: `cargo check` then `cargo test`, `go build` then `go test`, `tsc` then `npm test` (when a tsconfig is present), `ruff check` then `pytest` (when ruff is configured), or `make test`. Repeat `--verify CMD` to replace detection with exact ordered stages. `--no-verify` produces an explicitly unverified outcome; a mutating one-shot still exits nonzero unless `--allow-unverified` is also given.

Model rounds are unlimited by default. `--max-steps N` (or `/config steps N`)
installs an explicit per-turn cap; `/config steps auto` returns to the unlimited
default and `/config steps off` is an equivalent explicit opt-out. A capped
turn gets one final tool-free round to report where it left the work, then still
runs normal workspace verification and settlement. Incomplete productive work
settles as a typed failure at an explicit cap; a read-only turn with a usable
wrap-up remains completed. State-aware repetition and no-progress guards remain
active. Whole turns have no default deadline;
`--turn-deadline SECS` installs an explicit soft settlement deadline. Each turn
prints `[N in · N out · N total · k/k ctx]`.

Tool executions are also unlimited by default. `--max-tool-calls N` installs an
explicit independent cap. Parallel batches reserve the remaining budget before
dispatch and return typed denials for the model-ordered suffix, so concurrency
cannot overspend it.

## Best-of-N

Run several attempts and keep the one that actually passes — the **test suite is the judge**. `hi --best-of N` is the headless form of `/race`.

```bash
hi --best-of 3 "implement the spec in README"
```

It runs N candidates (varied temperature) in isolated **git worktrees**, gives each its own verify-loop, independently verifies eligible diffs, and applies the deterministically ranked winner. It requires a resolved automatic or explicit verifier and a git repo; tracked edits and untracked files are snapshotted into every candidate.

The legacy command now uses the same deterministic quality gates and ranking as the
interactive race. For a day-to-day coding task, configure two or more saved profiles
in `.hi/config.toml` and use the TUI race instead:

```toml
[race]
max_candidates = 2
max_concurrency = 2
fuzz_command = "cargo fuzz run my_target -- -runs=1000"
fuzz_timeout_secs = 120

[[race.targets]]
name = "fast"
profile = "local-fast"
model = "model-a"
priority = 0

[[race.targets]]
name = "strong"
profile = "cloud"
model = "model-b"
priority = 1
```

In the full-screen TUI, `/race setup` creates a roster from saved profiles,
`/race <task>` runs the candidates against the same workspace snapshot, and the
scoreboard preselects the highest-ranked candidate that passes independent
verification (and fuzzing, when configured). Review the diff, use `↑/↓` to inspect
another eligible candidate, then press `a` or run `/race apply`. Applying is rejected
if the workspace changed since the race began and requires a fresh exact patch check.
`/race status` and `/race cancel` manage the active run. Credentials stay in the
existing profiles; the project roster stores only profile/model names.

Selection is not a model vote: a failed test or fuzz stage is a hard exclusion. Among
passing candidates, the runner prefers fewer changed files and lines, then lower
runtime/cost, configured priority, and a stable candidate id. The separate Diff Lab
compares existing deterministic implementations against identical seeded inputs; it
is the right mode for parser, runtime, refactor, and optimization differential tests.

## Long-horizon goals

`/goal <objective>` is for the tasks you'd normally break into a week of tickets — "port this
service from Python to Rust," "get coverage above 80% in this crate." A goal isn't a prompt,
it's a contract: every top-level provider gets a durable structured goal; when a planner model is
configured (glm-5.2 by default on Pipenetwork), it decomposes the objective into sub-goals,
otherwise the executor grows an initial single milestone as it discovers work. Explicitly
referenced workspace documents are read before decomposition, so this is supported directly:

```text
/goal review the plan.md document and fully build this
```

The agent keeps pulling toward the goal **turn after turn on its own** — through compactions, test
failures, session resume, and refactors-within-refactors — while you monitor and steer. Checklist
progress is provisional until the settled workspace revision passes deterministic verification and
review. Type at any time to redirect; Esc pauses; `/goal resume` continues; the plan grows as work
is discovered, with no default cap (`/goal limit N` sets one). A pinned checklist + `goal d/t`
badge track progress in the TUI.

**Skeptic gate (`/goal team`, experimental).** By default a single agent plans, implements, and
verifies each turn. Point a reviewer model at it (`HI_SKEPTIC_MODEL=<model>`, or a profile
`skeptic_model`) and turn on `/goal team`, and before a turn may mark a sub-goal *done* a second
model reviews the turn's diff — plus the sub-goal and verify result — and can send it back to retry
with concrete objections, which become notes the next turn must address. It's off by default (an
extra model call per advance), fail-open (a reviewer error or timeout never blocks progress), and
scoped to where orchestration has a real shot: a long-horizon goal, not a single bounded turn.
`/goal team` alone reports the state and how many advances the skeptic has blocked. Headless runs
(one-shot `--goal`, the daemon, fleet rows) enable it with `HI_GOAL_TEAM=1`. The review covers both
ways a turn claims a sub-goal done — the heuristic advance and an explicit `update_plan` — and an
objection reverts the turn's goal progress (the edits stay on disk for the next turn to fix).

## Fleet dashboard

`/fleet` scales that to a fleet: the dispatch box at the bottom always spawns a *new*
session — type a prompt, hit Enter, and you've launched another agent without leaving the
screen. Each row works in its **own git worktree**; verified, non-overlapping diffs
**auto-merge back** (collisions hold visibly, `m` forces). Select a row for a peek panel with
a live reply input — answer an idle agent with a single keystroke (`1`–`9`) or queue a
follow-up; `Ctrl+S` dispatches *and* attaches. Prefix a dispatch with `/goal ` and the row
drives a whole objective autonomously. Every row is its own resumable session. Details:
[fleet-dashboard.md](fleet-dashboard.md).

## Loops

`/loop 30m check whether CI on main is green` — the same prompt, on a cadence. Intervals run
from 60 seconds to days (`90s`, `30m`, `2h`, `1d`); loops run until explicitly cancelled by id
(`/loop list`, `/loop cancel 3`). The shape is built for **watching things**:
CI logs, a canary deploy, a live service, a flaky test you're trying to catch in the act.

Each firing is a full agent turn, not a dumb cron job: it resumes the loop's own session, so it
*remembers* previous checks, compares instead of re-describing, and replies `NOTHING NEW` when
nothing changed — quiet firings land as a dim one-liner, real changes land loud (with a terminal
ping when you're unfocused). Loops persist per project and re-arm when `hi` restarts (they fire
while `hi` is running).

`/loop trio <prompt>` is the transient plan→execute→review workflow: a planner drafts the approach,
the session agent implements it, and a reviewer sends concrete objections back for another round.
It has no round limit by default and continues until approval, cancellation, or a typed execution
failure. Use `/loop trio --rounds N <prompt>` only when you want an explicit finite round cap.

`/watch` opens a **full-screen dashboard of every active loop**: a live table with per-loop
countdowns to the next firing, a spinner while one is checking, each loop's last result
(dim `· nothing new` or a loud one-line change), and its **running token spend**. Select a loop
to peek its recent firing history; `f` fires the selected loop immediately, `p` pauses/resumes it,
`c` cancels it, and `n` arms a new one from the same `<interval> <prompt>` box — all without
leaving the screen. The loops keep firing in the background; Esc returns to the chat.

**Cost guard.** Each firing is a full agent turn, so a fast long-running loop adds up. Every loop
tracks its cumulative token spend, and you can cap it:
`/loop budget 3 500k` auto-**pauses** loop #3 once it has spent 500k tokens (it stays resumable —
raise the budget or `/loop resume 3` to continue). Pause and resume any loop by hand with
`/loop pause <id>` / `/loop resume <id>` (or `p` in `/watch`); a paused loop holds its place and
its cost without firing.

**PR review.** The mirror image of auto-fix (which *opens* PRs): `/loop review` arms a watcher that,
on each firing, lists your repo's open pull requests, reviews any it hasn't seen yet (`gh pr diff` →
assess correctness, tests, risks), and **posts a review comment** with `gh pr review <n> --comment`.
Its session remembers what it's reviewed, so a firing with nothing new is a silent `NOTHING NEW`.
`/loop review 1h` sets the cadence (default 30m). Needs `gh` authenticated; it posts real
review comments (a comment — never approve/request-changes), so it's opt-in by arming it.

**Windows & cost.** Loops fire 24/7 by default; give one a local-time window so it only fires when
it matters: `/loop window 3 9-17` (or `9-17 weekdays`, or `off` to clear) — outside it, the loop
quietly defers to the next interval. And `/loop cost` shows a token-spend breakdown across loops
(each loop's spend, its budget, and the total) — cheap control for running many watchers.

**Triggers — a watcher that acts.** Attach a shell command that runs whenever a firing reports a
real change: `/loop on 3 notify-send "CI is red"`. It runs via `sh -c` only on a *loud* firing
(never on `NOTHING NEW` or an error), with the change summary in `$HI_LOOP_SUMMARY` (plus
`$HI_LOOP_ID` / `$HI_LOOP_NAME`), and its outcome surfaced in the transcript and the
`/watch` peek. Triggers have no execution deadline by default; set a positive
`HI_LOOP_TRIGGER_TIMEOUT_SECS` when you explicitly want one. Compose anything — desktop
notifications, a webhook `curl`, a file touch, even
another `hi -p "…"` to kick off a fix. `/loop on 3 off` clears it. (The command is yours and runs
with your shell's privileges — treat it like a git hook.)

**Auto-fix — a watcher that repairs.** Take the trigger idea to its conclusion: `/loop fix 3 on`
makes loop #3, on a loud change, dispatch a **worktree-isolated agent to fix the problem** — and
land the fix **only if it passes your verify command** (`/verify`). It's the fleet's
detect→fix→verify→merge cycle, driven by a watcher: *"watch CI; when it goes red, an agent fixes it
and the fix lands only if it's green."* Guardrails are the point — an unverified change is never
landed (no verify command → the fix is reported but not applied), one fix runs per loop at a time,
and every attempt lands in the transcript and the digest.

Two landing modes: `/loop fix 3 on` **merges** the verified fix into your working tree (great for a
scratch repo); `/loop fix 3 pr` instead commits it to a branch, pushes, and **opens a PR** (`gh`)
for review (great for a real one — nothing touches your tree until you merge). No remote or `gh`?
It degrades gracefully to a local/pushed branch and tells you. `/loop fix 3 off` disables it.

**Digest — what changed while you were away.** Loops write every loud event (a change they found, a
budget pause, an expiry) to a per-project activity feed that survives restarts — and so do **fleet
rows** (verified merges, combined-tree verify failures, goal completions). `/digest` shows the feed
grouped by source (each loop, each fleet row) — how many changes each produced and the most recent,
with a `•` on everything new since you last looked. Start `hi` after leaving work running and you'll
see a one-line `⟳ N loop change(s) since you last looked — /digest to review` nudge. It's one pane
for everything autonomous that happened. `/inbox` is the other pane: parked confirms the unattended
goal or loop could not answer live (`hi inbox allow|deny <id>`). Digest is what changed; inbox is
blocked on you.

**Daemon — keep firing when the terminal's closed.** Loops only fire while a `hi` is running. Run
`hi --loops-daemon` to keep this project's loops firing (and auto-fixing) headless in the background,
logging each change, until you `Ctrl-C` (or `kill`) it. A per-project lock guarantees exactly one
firer — the daemon and a TUI never both fire the same loops; whichever starts second reads the shared
feed instead (`/digest`) and says so. Set your loops up in the TUI, close it, `hi --loops-daemon &`,
and come back later to `/digest` what it caught.

**Project tickets — dashboard Board executed by hi.** Org Projects hold `ticket_*` work items (not one `task_*` per card). Pair with `/login pipenetwork` and pick the project, `cd` into the repo, then `hi tickets`. The daemon heartbeats, claims a queued **local** ticket, and spawns `hi --goal "<goal>" --verify "<cmd>"` until the report passes or the ticket ceiling is hit. A child crash with remaining budget completes as `repairing` so the next claim retries. Do not send sandbox tickets to the laptop; those dispatch to `POST /v1/tasks`. Session `--daemon` free-text input is not a ticket.

**Notifications — reach you when you're away.** A background daemon logs to a transcript you're not
watching, so loud events (a change a firing found, a landed fix, a budget pause) can also be pushed
to you, opt-in via the environment:

```bash
HI_NOTIFY_DESKTOP=1 hi --loops-daemon                 # macOS terminal-notifier / Linux notify-send
HI_NOTIFY_WEBHOOK=https://hooks.slack.com/… hi --loops-daemon   # JSON {"text":…} POST (Slack-compatible)
```

Both sinks are best-effort — a missing tool or a failed POST never blocks a firing — and work in the
TUI too. The daemon prints which sinks are active on startup.

## Sessions

Every session is saved as JSONL under `~/.local/share/hi/sessions/`.

```bash
hi -c "and now add tests"          # --continue the latest session
hi resume                           # TUI resume of the latest session here
hi --resume <id> "..."             # resume a specific one
hi --list-sessions                 # list saved sessions
hi --no-save "..."                 # don't persist
hi --durable "..."                 # explicitly require boundary checkpointing
```

Saved ordinary sessions use durable execution by default, checkpointing the
prompt and each completed tool batch. Set `execution = "ephemeral"` globally
or per profile to opt out. In the full-screen TUI, it is a live, visible session control:
run `/durable on` before a long task, watch the `durable` badge in the title bar,
and use `/durable status` to confirm the mode. It requires a persisted session
and checkpoints after the user prompt and each completed tool batch; use
`--continue` or `/sessions` after a restart to pick up the recorded state.

## In-session commands & context

Slash commands (TUI or plain REPL):

| command | does |
|---|---|
| `/help` | core commands; `/help project`, `/help modes`, `/help platform`, or `/help all` for the rest |
| `/model [id]` | set by id, or — with no id — open an interactive picker over the live model list (type to filter, ↑/↓, Enter). |
| `/provider [name\|add\|edit]` | use a configured profile (no name lists them), `add` to create a new profile interactively, `edit [name]` to modify one. |
| `/durable [on\|off\|status]` | TUI-first live execution control; checkpoint the current saved session at prompt and completed tool boundaries. |
| `/verify [cmd\|off]` | show, set, or clear the test command turns iterate against — turn the verify-loop on without restarting |
| `/race <task>` | run two to four configured model/profile candidates in isolated worktrees, independently verify them, and open the review scoreboard |
| `/race setup\|status\|cancel\|apply` | configure saved-profile targets or manage, review, and explicitly apply a completed race |
| `/diff` | show what files have changed this session (`git diff` + new files) |
| `/copy [all]` | copy the last assistant response to the terminal clipboard; `all` copies the transcript |
| `/goal [obj\|pause\|resume\|limit N\|team on\|off\|clear]` | set a long-horizon goal: a planner model decomposes it into sub-goals the agent then **drives autonomously turn after turn** (your input always takes priority; Esc pauses). `pause`/`resume` hold and continue; `limit N` caps plan growth (unbounded by default); `team on` adds a skeptic reviewer that must approve each advance (needs `HI_SKEPTIC_MODEL`) |
| `/loop trio [--rounds N] <prompt>` | transient plan→execute→review workflow. It continues until approval by default; `--rounds N` opts into a finite revision cap. |
| `/loop <interval> <prompt>` | the same prompt, on a cadence (60s–7d: `90s`, `30m`, `2h`, `1d`): each firing is a **full agent turn** that remembers previous checks and reports only what changed, and the loop runs until cancelled. `/loop list`, `/loop cancel <id>`, `/loop pause\|resume <id>`, `/loop budget <id> <count\|off>` (token cap → auto-pause), `/loop on <id> <cmd\|off>` (run a shell command on each change, `$HI_LOOP_SUMMARY` in env), `/loop fix <id> <on\|pr\|off>` (verify-gated auto-fix on a loud change — `on` merges, `pr` opens a PR), `/loop window <id> <9-17 [weekdays]\|off>` (local-time fire window), `/loop cost` (token-spend breakdown), `/loop review [interval]` (a PR-review watcher — reviews open PRs via `gh`) |
| `/watch` | full-screen live dashboard of all active loops: per-loop countdowns, firing spinners, last result, token spend, and recent history — with `f` fire-now, `p` pause, `c` cancel, `n` arm a new loop |
| `/digest` (`/activity`) | what your loops have noticed, grouped by loop, with what's new since you last looked (a persisted, cross-restart feed of every loud change) |
| `/inbox [allow\|deny <id>]` | parked confirms blocked on you (unattended/daemon). `/digest` is what changed; inbox is what needs a decision. `hi inbox` works from the shell |
| `/fleet` (`/dashboard`) | control a fleet, not an agent: dispatch, monitor, and steer multiple concurrent sessions — each in its own git worktree with verified diffs auto-merging back; `/fleet status` lists this project's resumable fleet sessions ([docs](fleet-dashboard.md)) |
| `/delegate [on\|off\|risk]` | write-capable delegate subagent: worktree-isolated child; changes land only if they verify. Default **risk** (multi-file / isolation-shaped tasks); `on` = every mutation; `off` = never. Read-only `explore` is on by default for repo tasks |
| `/init` | scan the repo and write an `HI.md` project guide (loaded as context in future sessions) |
| `/compact [kind]` | reclaim context — `hybrid` (summarize old turns, keep recent), `full` (summarize everything), or `elide` (drop old tool output, no model call) |
| `/context` (`/context-doctor`) | occupancy breakdown plus a fresh-session **injection census** (system, guides, skills, tool schemas, volatile memory) |
| `/retry` | re-run your last message (drops the previous attempt — pairs with `/model`) |
| `/undo` | revert the file changes the last turn made (restores its git checkpoint) |
| `/commit` | commit files this session touched (never `git add -A`; refuses a secret-looking staged diff) |
| `/status` | show provider, model, queue, context, last turn state, and session `$` when the model publishes a price |
| `/login <provider>` | subscription pairing only: `xai`, `pipenetwork`, or `x402` |
| `/auth <openai\|anthropic\|xai> [key]` | paste an API key, probe `/models`, then write a profile. HTTP 401/403 is never saved. The key is masked and omitted from ↑ history. `hi auth <provider>` does the same outside a session |
| `/mcp [pipe\|name reconnect\|allow\|deny]\|add` | workspace MCP status table (includes auto-attached `pipe`); `pipe` inspects the unfiltered provider `mcp_url`; `allow`/`deny` persist per-server tool lists; `add` writes `.hi/mcp/<name>.json` |
| `/log` | write a local debug log for this session (`.hi-debug.log`) |
| `/export [path]` | export the conversation to a file (default: `transcript.md`) |
| `/version` | show version |
| `/clear` | start a fresh conversation |
| `/exit` | quit |

Drop an `HI.md` or `AGENTS.md` in your project and its contents are appended to the system prompt — per-project conventions, for free. `/init` scans the repo and writes an `HI.md` for you. Put standing user rules in `~/.config/hi/me.md` (stable prefix, not volatile `.hi/memory.md`). Scan also picks up workspace `.agents/skills/*/SKILL.md` (Agent Skills spec; `.hi/skills` wins on name). Built-in packs include stack loops (`rust-workspace`, `pytest-package`, `ts-monorepo`), `code-review`, and optional `/skill` recipes `secret-scan` and `dep-audit` (not auto-injected; they hint to stay on `/permissions`). `/agents` remains user markdown.

**MCP.** Workspace servers come from `.hi/mcp/*.json` (wins on name), Claude `.mcp.json`, and optionally Codex `~/.codex/config.toml` when `[mcp_import.codex] enabled = true` in `hi.toml`. Gate with `[mcp_import.claude] only = [...]` / `exclude = [...]` (`exclude` wins). Per-server tool lists live in that JSON (`only` / `exclude`) or `[mcp.servers.<name>]` in `hi.toml`; `/mcp <name> allow|deny <tool>` persists them. `/mcp add <name> --stdio <cmd> [args…]` or `--http <url>` writes `.hi/mcp/<name>.json` and registers without restart. Imported servers default to all tools visible (writes still hit egress confirms). When the active provider has `mcp_url` and an API key, hi also auto-attaches first-party Pipe MCP as server `pipe` (HTTP). The agent may call `pipe.models.list` and `pipe.models.health` only; nested `pipe.chat.completions.create` / `pipe.responses.create` stay off the coding loop even if listed in `[mcp.pipe] allow`. Opt out with `[mcp.pipe] enabled = false`. If `.hi/mcp/pipe.json` (or another import) already defines `pipe`, the workspace file wins and auto-attach is skipped. Folder trust still gates repo-local stdio servers; remote Pipe does not require trust. Startup registers servers without waiting; the first `use_tool` connects with a short grace, then fail-fast (`/mcp <name> reconnect`). `/mcp` is the workspace table (including `pipe`); `/mcp pipe` is the **full** provider `mcp_url` inspector (all six Pipe tools, unfiltered). `hi mcp test <name>` is for CI. `hi mcp serve` exposes `read`/`bash`/`edit`/`write` over MCP stdio for other harnesses (sandbox + denylist still apply). The model still sees only `search_tool` / `use_tool` — MCP tool JSON is never dumped into the request.

**Browser.** `browser_exec` is on by default and injected on page/login/UI-shaped tasks (not the global catalog) in an interactive TUI/REPL. Headless one-shot, `--loops-daemon`, and `hi mcp serve` do not advertise it. Set `[browser] enabled = false` in `hi.toml` to hide it everywhere. `allow_private_urls` opts into RFC1918/loopback; cloud metadata hosts stay blocked (including after DNS and redirects). `hi browser install` writes an unpacked Chrome extension under `~/.config/hi/browser-extension/`. Pairing (`/login`) stays separate from pasted keys (`/auth`).

**Auto-memory.** At the end of an interactive session, `hi` distills durable lessons into `.hi/memory.md` (and user-level `~/.config/hi/memory.md`) with stable `[#n]` bullet ids. `/remember` appends a numbered note; `memory_update` / `memory_forget` correct it; `/undo-memory` restores the previous file. Disable with `--no-memory`.

**Auto-compact.** During long tool loops, `hi` elides older bulky tool results once the local context estimate passes ~45% full, keeping the newest verbatim. Before a new turn, if the previous request used ~80% of the context window, it summarizes the conversation and resets to that summary. Disable with `--no-auto-compact`; trigger manually any time with `/compact`. Tool payloads are also bounded: `read` returns 240 lines unless paged with `offset`/`limit`, and `HI_TOOL_RESULT_CHARS` controls the per-result character cap.

**Undo.** Before mutation, `hi` creates a recoverable checkpoint: a dangling
commit with a throwaway index when Git is usable, otherwise a content-addressed
internal snapshot. `/undo` restores created, modified, and deleted files plus
modes and symlink targets. It refuses to overwrite a file changed externally
since the turn. If no checkpoint backend is available, normal YOLO mode pins a
warning and continues without prompting. `--confirm-edits` makes that case
strict; combine it with `--allow-no-checkpoint` to retain the YOLO fallback.
Checkpoints cannot undo non-file side effects.

**No nag-prompts — but a guard for the irreversible.** Rather than asking permission for every command (the thing everyone turns off), `hi` lets the model run freely and relies on `/undo` for recovery. The one exception is a small denylist of operations a checkpoint *can't* undo — `sudo`, `rm -rf` of home/root/system paths, `git push --force`, `curl … | sh`, `dd` to a disk, `mkfs`, fork bombs, shutdown — which are refused with a reason the model can act on. It's a seatbelt against accidents, not a security boundary; set `HI_ALLOW_DANGEROUS=1` to disable it. Tool results, web/research pages, browser AX/eval output, MCP payloads, and inbound `hi mcp serve` calls are untrusted data, not instructions.

**Egress confirms.** Attended default remains YOLO (`/permissions always`). In Ask and Auto, `browser_exec` and MCP `use_tool` pause on a confirm overlay; web fetch/research pause only in Ask. Session standing grants apply to an MCP `server`+`tool` pair, never to bash or the browser. Unattended goals and loop children keep Ask/Auto: confirms that cannot be answered live are parked in `/inbox` (not auto-approved). `hi mcp serve` has no human on stdio — it stays denylist + sandbox + folder trust, with no inbox.

**Dry run.** Pass `--dry-run` to preview what the model *would* do without
executing anything. Each tool call that survives policy, budget, and protocol
checks is printed as a planned action (`[dry-run] would run …`) and a synthetic
result is returned to the model — mutating calls are flagged as such, but
nothing touches the workspace and no process is spawned. Useful for inspecting
an agent's plan before letting it act.

**OS sandbox (default on).** Shell *writes* are confined to the project (plus temp) by default (`HI_SANDBOX=workspace`). Set `HI_SANDBOX=off` when normal tool caches under `$HOME` must stay writable. macOS uses Seatbelt; Linux confines writes when `pipe-wrap` is available (otherwise hi warns and continues). See [sandbox.md](sandbox.md).

**TUI.** Interactive sessions open a full-screen TUI by default (ratatui): a bordered, scrollable transcript with a title bar showing live token usage, and an input box that turns into a working spinner (with elapsed seconds) while a turn runs. **Keep typing while it works to queue the next command(s)** — they're listed under the prompt and run in order as each turn finishes. Ctrl-C interrupts the current turn (and drops the queue), PgUp/PgDn scrolls, Up/Down recalls history, `/exit` quits. Pass `--plain` (or pipe input) for the line-based REPL.

**Reports.** One-shot automation can write schema-v2 JSON with
`--report path.json`. Reports contain the typed turn outcome, verification
stages, review status, typed tool results, actual provider/model route,
turn/session usage, and exact file changes. Reports are written for failed and
blocked turns as well as successful ones; legacy report fields are no longer
emitted. Historical `incomplete`/`stalled` outcome values remain readable as
`failed`/`no_progress`, but new reports never emit the historical names.
Explicit model-step or tool-call caps on unfinished productive work emit failed
status with a typed limit reason; usable read-only wrap-ups may still complete
at a cap. In particular, session token totals now live at
`usage.session.total_tokens`, not the legacy top-level `total_tokens` field.

**RSI candidate channel.** In the TUI, `/config rsi` shows readiness, candidate
attribution, rollout phase, learning-loop health, evidence policy, and training
state. Stable is the default; `/config rsi channel beta` explicitly joins
deterministic canaries and `/config rsi channel stable` leaves them.
`/config rsi spend-limit 5` sets the per-run ceiling to $5, and `/config rsi on`
or `off` enables or disables it. These changes apply
immediately and are saved; the public gateway remains
`https://api.pipenetwork.ai`. RSI can also be enabled with `--rsi`,
`HI_RSI_ENABLED=true`, or `[rsi] enabled = true` in `hi.toml`; `--no-rsi`
overrides configuration. Enabling validates the authenticated Pipe RSI service
and confirms repository plus bounded conversation-context upload, 30-day
operational evidence retention, and training off without separate consent.
Each subsequent turn runs on the managed `rsi-hi-worker`, reports reconnectable
status, validates the exact result against baseline BLAKE3 hashes, and applies
all changes atomically. It never falls back to local execution. Use `/rsi list`,
`/rsi status RUN`, `/rsi cancel RUN`, `/rsi apply RUN`, or
`/rsi artifacts RUN` to recover after a disconnect. Use
`/rsi feedback [RUN] good|bad [reason]` to add supporting outcome evidence;
feedback alone never authorizes promotion. Internal/test deployments
may still select a test gateway in `hi.toml`.

**Outcome tasks (`POST /v1/tasks`).** Ordinary `hi` sessions stay on the direct provider route so they have no paid-task deadline or attempt ceiling. `--tasks` opts every turn into the bounded Outcome task contract; `[outcome] mode = "auto"` opts Cargo mutations and test-gated prompts into `code.change` with `cargo_test` and `cargo_clippy` (plus `review` when `--review always` or risk-on-mutation). `--no-tasks` forces direct chat. `--rsi-managed` never calls `/v1/tasks`. If the Outcome API returns `tasks_unavailable`, 401/404, a missing key, or no RSI worker heartbeat, hi prints one line and continues on local chat (or `/v1/rsi/runs` when `--rsi` is on). `/rsi repair` posts remaining budget to `/v1/repairs`. `/rsi status` shows `contract_hash` after `POST /v1/receipts/verify`.

Laptop loopback (not evaluation-grade on macOS):

```bash
ipop/scripts/rsi-dev-up.sh
# equivalent: IPOP_ROOT=/path/to/ipop hi rsi up
```

Point hi at the printed public API:

```toml
[outcome]
mode = "auto"
base_url = "http://127.0.0.1:13000/v1"
```

`json_schema` can complete without a sandbox. Cargo-backed `code.change` needs the unsandboxed `rsi-hi-worker` plus local `hi`, and is not GA until that path has a verified worker run. `hi rsi down` stops the stack.

The trusted worker still uses the hidden managed evidence contract
(`--rsi-managed`, an expiring runtime descriptor, a fixed trace directory and
byte limit, and `--api-unix-socket`). The descriptor binds effective budgets,
tools, isolation, run, candidate, signed manifest, binary, and repository
snapshot to every hash-chained trace. The worker independently verifies that
provenance before upload. Managed evidence remains mandatory and is stored
server-side; the normal client retains only pending IDs, baseline hashes, and
result summaries unless artifacts are explicitly downloaded.

## Local traces & workflow evidence (`hi trace`, `hi workflow verify`)

Self-hosted runs record a local trace and (for `hi workflow run`) a signed
verification report. Both carry a `local-signed:` ed25519 attestation — real
tamper-evidence, but **not** worker attestation, since the key lives on the
same machine.

- `hi trace list [n]` — recent traces with an `INTEGRITY` column
  (`ok`/`TAMPERED`) and the attestation scheme, so tampered or unsigned runs
  are visible at a glance.
- `hi trace show [id]` — one trace's detail with the integrity status inline.
- `hi trace verify [id]` — recompute the hash chain and check it against the
  manifest root; for `local-signed:` traces, also validate the signature.
  Fails on a broken chain.
- `hi workflow verify [report.json]` — validate a workflow report's
  `local-signed:` attestation against the local key. With no argument it
  resolves the latest persisted report (under the state root's
  `workflow/<plan>/report.json`); a forged or tampered signature fails.

The local signing key lives at `$XDG_STATE_HOME/hi/trace-signing-key` (falling
back to `$HOME/.local/state/hi/trace-signing-key`), created owner-only (`0600`)
on first signing run. Trace integrity is local consistency, not authenticity —
external anchoring is the managed worker's job (see
[ADR 001](adr/001-rsi-runtime-boundary.md) and `architecture.md`).

## Architecture

A cargo workspace:

| crate | role |
|---|---|
| `hi-ai` | provider-neutral types, the `Provider` trait, OpenAI + Anthropic adapters, retry |
| `hi-tools` | the tools: `read` / `write` / `edit` / `multi_edit` / `apply_patch` / `bash` / `bash_output` / `bash_kill` / `list` / `grep` / `glob` / `diff` / `commit` / `update_plan` / `record_decision` |
| `hi-agent` | the agent loop, verify-loop, sessions, the `Ui` trait |
| `hi-rsi-runtime` | managed candidate descriptor, workflow, budget, checkpoint, verification, failure, and exact-replay contracts |
| `hi-trace` | bounded content-addressed RSI artifacts and crash-safe hash-chained event journals |
| `hi-tui` | full-screen terminal UI (transcript, spinner, queue, slash commands) |
| `hi-race` | local-first race contracts, workspace snapshots, stage execution, and deterministic ranking |
| `hi-cli` | the `hi` binary: config, sessions, best-of-N, slash commands |
| `hi-local-runtime` | optional external sidecar repository containing `hi-local`, `hi-local-core`, `hi-gguf`, `hi-cuda`, and `hi-mlx` |
| `hi-eval` | the benchmark runner (see below) |

Richer capabilities come from **subprocess CLI tools** the model invokes via `bash` rather than a plugin runtime. New built-in tools must clear the [tool admission bar](adr/002-tool-admission.md) (prefer bash/skills unless structure, safety, or reliability requires a first-class tool).

## Benchmarks (`hi-eval`)

`bench/` measures whether orchestration changes beat a baseline. Task schema v2
declares the prompt, allowed-change globs, optional visible feedback, and an
immutable final-oracle command and optional bundle kept outside the candidate.
`hi-eval` captures the oracle before launch, runs the candidate using only
`fixture/`, then injects the captured bytes into a fresh verification copy.
Candidate-side test edits therefore cannot change the final score. Candidate
runs default to 900 seconds and final-oracle checks to 120 seconds; the suite
defaults to three trials.

The default matrix includes `baseline`, `verify`, heterogeneous `best-of-3`,
and `goal-team`. Artifacts preserve every candidate's temperature/seed, actual
route and outcome, patch, checks, turn/session usage, known cost, and wall time.
`summary.json` reports candidate pass rate and solve@N separately; standard
pass@k is emitted only for exchangeable samples.

```bash
cargo run -p hi-eval -- bench --validate          # validate every task/oracle (no model)

# Compare configs against any model (env flows through to hi):
HI_MODEL=anthropic/claude-sonnet-4 HI_API_KEY=$OPENROUTER_API_KEY \
  cargo run -p hi-eval -- bench/spec

# The raw-Fusion line to beat (Fusion is selected via env, not a flag):
HI_MODEL=openrouter/fusion HI_API_KEY=$OPENROUTER_API_KEY \
  cargo run -p hi-eval -- bench/spec
```

### 0.2 baseline (coding north star)

Locked metrics live in `eval-baseline/core-0.2.json`. The first provider-backed
capture has not landed yet — `solve_rate` / `cost_per_solved` / failure buckets
remain null until then. After a full matrix run:

```bash
# North-star ladder (regression floor → multi-file):
HI_MODEL=… HI_API_KEY=… cargo run -p hi-eval -- \
  --configs=baseline,verify --trials=3 \
  bench/tasks   # then bench/spec, bench/vloop-dense, bench/hidden

# Capture from the run's summary.json:
cargo run -p hi-eval -- --write-baseline=path/to/summary.json

# Compare a later run (exit 2 on regression when flagged):
cargo run -p hi-eval -- --compare-baseline=path/to/summary.json
cargo run -p hi-eval -- --compare-baseline=path/to/summary.json --fail-on-baseline-regression
```

Tracked metrics: **solve_rate**, **false_verified_rate**, **cost_per_solved**,
**tokens_per_solved**, **infrastructure_error_rate**, and failure buckets
(`no-edits` / `compile` / `logic` / `error`). Every full `hi-eval` run prints a
baseline compare block when the file is present.

## Fast local feedback

Use focused commands while editing. Voice support (`cpal` and Whisper) is
opt-in so ordinary coding builds avoid the native audio stack:

```bash
cargo check -p hi-agent --lib
cargo test -p hi-agent --lib
cargo test -p hi-shell --lib

# One-time setup for parallel test execution:
cargo install cargo-nextest --locked
cargo nextest run -p hi

# Confirm the release feature set before shipping:
cargo check -p hi --features voice
```

For repeated clean or dependency-heavy builds, install `sccache` and let Cargo
reuse compiler output across branches and worktrees:

```bash
brew install sccache
export RUSTC_WRAPPER="$(command -v sccache)"
sccache --show-stats
```

`cargo nextest run` parallelizes independent test binaries and is preferred for
local suites. The release checklist below intentionally remains broad and uses
all targets so CI still catches feature, example, and benchmark regressions.

## Core 0.2 release checklist

- `cargo fmt --all`
- `cargo clippy -p hi-ai -p hi-tools -p hi-lsp -p hi-agent -p hi-tui -p hi -p hi-eval --all-targets -- -D warnings`
- `cargo test -p hi-ai -p hi-tools -p hi-lsp -p hi-agent -p hi-tui -p hi -p hi-eval`
- `cargo install --path crates/hi-cli --locked`
- Smoke an OpenAI-compatible endpoint with `--compat auto` and `--tool-mode auto`
- Validate eval tasks and immutable oracles with `cargo run -p hi-eval -- bench --validate`

The 0.1 GPU/local-inference crates have their own hardware-specific release
checks and are not gates for the core 0.2 release.

## Status

Early but functional. The multi-provider core, full-screen TUI, sessions, verify-loop, best-of-N, compatibility fallbacks, changed-file reporting, and eval harness are built and tested. Optional local CUDA/MLX inference is released and tested through the external `hi-local-runtime` sidecar contract. The TUI's rendering is verified via ratatui's TestBackend; its live key/scroll behavior is best confirmed in a real terminal. Cargo install is the first release target; binary archives and Homebrew can follow later.
