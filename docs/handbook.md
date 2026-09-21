# hi handbook

Full reference for hi 0.3.1. Everyday getting started lives in the [README](../README.md).

`hi` is an agentic coding tool written in Rust. Point it at any model — local or remote — and it reads, writes, and edits files and runs shell commands in your project to do what you ask.

Default inference is Pipe Network (`api.pipenetwork.ai`, model
`pipe/deepseek-v4-flash-0731`). `/verify <cmd>` is a **post-turn check**; it
does not auto-repair. `/undo` restores the last turn's git checkpoint.

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
cargo build --release                    # hi + hi-sentinel (workspace default members)
cargo build --release --features voice  # include microphone + local Whisper
cargo install --path crates/hi-cli --locked
cargo install --path crates/hi-sentinel --locked

hi login pipenetwork            # browser pairing; writes the API key into config.toml
PIPENETWORK_API_KEY=pk_live_... hi "add a --json flag"
hi auth pipenetwork             # paste a key, probe /models, store it
hi -m pipe/deepseek-v4-flash-0731 "…"
```

Interactive `hi` talks to Pipe Network. An optional `openai` profile in config
is only used for dashboard rows whose model id is prefixed `openai/`.

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
| `HI_BASH_TIMEOUT_SECS` | Optional hard shell-command timeout; setting it keeps the command foreground until completion or timeout | off |
| `HI_BASH_AUTO_BACKGROUND` | Hand a shell command that outlives its foreground attachment budget to the managed background registry; set `0` to opt out | on |
| `HI_BASH_FOREGROUND_BUDGET_SECS` | Time a shell command may remain attached to the active turn before managed handoff | 15s |
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
| `TYPESAFE_API_KEY` | Optional Jev next-action / auto / effort hints, and `/jev-compact` tool prune | off |
| `TYPESAFE_BASE_URL` | TypeSafe API origin | `https://api.typesafe.ai` |
| `TYPESAFE_DEFAULT_MODEL` | Jev model id | `jev-latest` |

Set `TYPESAFE_API_KEY` (or `[typesafe]` in `~/.config/hi/config.toml`) to let Jev flavor inspect-only review-and-fix rounds (prefer tests over more grep), score `/permissions auto` (withhold risky-looking safe files; auto-approve high-confidence reversible shell), and hint a turn-scoped reasoning effort. The gate is fail-open and machine-scoped; a timeout or error leaves the existing harness policy in place. `/jev-compact on` (alias `/compact-jev`) is a separate, session-only switch: with a TypeSafe key it scores tool calls and results, then drops or truncates stale ones while keeping everything else verbatim. It does not persist to config; `--jev-compact` or `--compaction jev` turns it on at launch. Failures fall back to cheap shrink and the model summary.

```toml
[typesafe]
# enabled = true   # default: on when a key is present
api_key_env = "TYPESAFE_API_KEY"
# model = "jev-latest"
# min_confidence = 0.55
# auto = true      # /permissions auto hints
# effort = true    # turn-scoped reasoning-effort hints
```

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

`/goal` was removed with the old harness. For concurrent work, use `/dashboard`
and dispatch several sub-agents yourself.

## Agent dashboard

`/dashboard` (aliases `/fleet`, `/agents-dashboard`, `Ctrl+\`) is a roster of
**independent** coding sessions. You are the manager (this hi session). Each
dispatch creates a sub-agent in-process. There is no auto-merge and no
parent-child tool. `Ctrl+M` sets the next sub-agent to `pipe/gpt-6` on Pipe;
`Ctrl+W` optionally isolates it in a git worktree. Details:
[fleet-dashboard.md](fleet-dashboard.md).

## Loops, watch, digest, inbox

`/loop`, `/watch`, `/digest`, `/inbox`, and `hi --loops-daemon` were removed with
the old harness. Use `/dashboard` for concurrent sessions.

## Sessions

Every session is saved as JSONL under `~/.local/share/hi/projects/<digest>/sessions/`. After a reboot nothing is restarted automatically — unfinished work stays on disk until you resume it.

```bash
hi sessions                         # needs-attention roster (pending turn, open plan, live/stopped lock)
hi sessions --all                   # every session, including completed plans
hi resume                           # unfinished work in this directory (picker if several)
hi --resume <id>                    # open that file; binds the workspace from the project sidecar
hi -c "and now add tests"           # same as resume, then send a prompt
hi --no-save "..."                  # don't persist
```

`hi sessions` groups by workspace and prints a copy-paste command:

```
cd /Users/david/chat && hi --resume 1789672473100-9612198e-102fa-0
  PLAN 1/8 · active: Forward HISTORY pagination
```

Dashboard child JSONL under `projects/<digest>/dashboard/sessions/` is tagged `dashboard`. `/sessions` in the TUI shows the same flags.

If a process still holds the session (including a SIGSTOP'd sentinel), resume prints `held by pid N (stopped); kill or fg` instead of opening a second writer. A leftover pid file after reboot is ignored once the lock is free.

Resuming an unmatched mid-turn shows a Continue / Dismiss card in the TUI (Sentinel still auto-continues with `HI_SENTINEL_RESUME_INCOMPLETE=1`). An open, unpaused plan sets an Enter-to-continue suggestion; it does not enqueue a turn on startup.

## In-session commands & context

Slash commands (TUI or plain REPL):

| command | does |
|---|---|
| `/help` | list slash commands |
| `/model [id]` | set by id, or — with no id — open an interactive picker over the live model list (type to filter, ↑/↓, Enter). |
| `/verify [cmd\|off]` | post-turn check command (does not auto-repair) |
| `/files` | files changed this session, with the Changes pane's `N files changed +A -D` summary and per-file counts |
| `/config [reasoning <level>]` | show or set request settings |
| `/mouse [on\|off]` | click-to-expand vs terminal highlight-to-copy |
| `/diff` (`/changes`) | open the Changes pane (same as `Ctrl-G`): the session's running diff |
| `/copy` | copy the last assistant reply |
| `/compact [context]` | summarize the conversation to reclaim context |
| `/context` | context-window breakdown (`/usage` Context tab) |
| `/retry` | re-run the last prompt |
| `/review [audit\|status\|stop] [all\|path…]` | spec review: audit the code against `plan.md`/`spec.md` (or, on a large repo without one, the uncommitted changes or the last commit; `all` chunks the whole repo), report coverage and defects, then fix P0/P1s in a bounded loop; headless `hi --spec-review` ([docs](review-command.md)) |
| `/undo` | restore files from the last turn checkpoint |
| `/status` | session status |
| `/usage` (`/cost`) | credit/token usage modal; `/usage manage` opens billing |
| `/login [pipenetwork]` | sign in to pipenetwork.ai |
| `/auth pipenetwork [key]` | store a Pipe API key |
| `/logout` | forget the stored Pipe credential |
| `/sessions` | list saved sessions |
| `/rewind <n>` | drop back to user turn n |
| `/dashboard` (`/fleet`, `/agents-dashboard`) | concurrent agent roster — you are the manager; each dispatch is a sub-agent ([docs](fleet-dashboard.md)) |
| `/permissions [ask\|auto\|always]` | confirm ladder (`/auto`, `/yolo`) |
| `/effort [low\|medium\|high\|xhigh\|off]` | reasoning effort |
| `/doctor` | check key, Pipe `/models`, git, and sandbox |
| `/tutorial` | interactive tour |
| `/clear` | reset the conversation |
| `/version` | show hi version |
| `/exit` | quit |

Removed with the old harness (the command prints a short notice): `/goal`, `/loop`, `/watch`, `/digest`, `/inbox`, `/delegate`, `/race`, `/mcp`, `/workflow`, `/btw`.

Drop an `HI.md` or `AGENTS.md` in your project and its contents are appended to the system prompt — per-project conventions, for free. `/init` scans the repo and writes an `HI.md` for you. Put standing user rules in `~/.config/hi/me.md` (stable prefix, not volatile `.hi/memory.md`). Scan also picks up workspace `.agents/skills/*/SKILL.md` (Agent Skills spec; `.hi/skills` wins on name). Built-in packs include stack loops (`rust-workspace`, `pytest-package`, `ts-monorepo`), `code-review`, and optional `/skill` recipes `secret-scan` and `dep-audit` (not auto-injected; they hint to stay on `/permissions`). `/agents` remains user markdown.

**MCP.** Workspace servers come from `.hi/mcp/*.json` (wins on name), Claude `.mcp.json`, and optionally Codex `~/.codex/config.toml` when `[mcp_import.codex] enabled = true` in `hi.toml`. Gate with `[mcp_import.claude] only = [...]` / `exclude = [...]` (`exclude` wins). Per-server tool lists live in that JSON (`only` / `exclude`) or `[mcp.servers.<name>]` in `hi.toml`; `/mcp <name> allow|deny <tool>` persists them. `/mcp add <name> --stdio <cmd> [args…]` or `--http <url>` writes `.hi/mcp/<name>.json` and registers without restart. Imported servers default to all tools visible (writes still hit egress confirms). When the active provider has `mcp_url` and an API key, hi also auto-attaches first-party Pipe MCP as server `pipe` (HTTP). The agent may call `pipe.models.list` and `pipe.models.health` only; nested `pipe.chat.completions.create` / `pipe.responses.create` stay off the coding loop even if listed in `[mcp.pipe] allow`. Opt out with `[mcp.pipe] enabled = false`. If `.hi/mcp/pipe.json` (or another import) already defines `pipe`, the workspace file wins and auto-attach is skipped. Folder trust still gates repo-local stdio servers; remote Pipe does not require trust. Startup registers servers without waiting; the first `use_tool` connects with a short grace, then fail-fast (`/mcp <name> reconnect`). `/mcp` is the workspace table (including `pipe`); `/mcp pipe` is the **full** provider `mcp_url` inspector (all six Pipe tools, unfiltered). `hi mcp test <name>` is for CI. `hi mcp serve` exposes `read`/`bash`/`edit`/`write` over MCP stdio for other harnesses (sandbox + denylist still apply). The model still sees only `search_tool` / `use_tool` — MCP tool JSON is never dumped into the request.

**Browser.** `browser_exec` is on by default and injected on page/login/UI-shaped tasks (not the global catalog) in an interactive TUI/REPL. Headless one-shot, `--loops-daemon`, and `hi mcp serve` do not advertise it. Set `[browser] enabled = false` in `hi.toml` to hide it everywhere. `allow_private_urls` opts into RFC1918/loopback; cloud metadata hosts stay blocked (including after DNS and redirects). `hi browser install` writes an unpacked Chrome extension under `~/.config/hi/browser-extension/`. Pairing (`/login`) stays separate from pasted keys (`/auth`).

**Auto-memory.** At the end of an interactive session, `hi` distills durable lessons into `.hi/memory.md` (and user-level `~/.config/hi/memory.md`) with stable `[#n]` bullet ids. `/remember` appends a numbered note; `memory_update` / `memory_forget` correct it; `/undo-memory` restores the previous file. Disable with `--no-memory`.

**Auto-compact.** Occupancy is `max(last Pipe request tokens, local estimate)` against the active model's `/models` context window (128k if metadata is missing). Once occupancy passes ~45% of that window, `hi` stubs tool-result bodies from *earlier turns* (keeps the newest six of those verbatim) and drops stale thinking — the in-progress turn's reads stay intact so the model can still see the files. If occupancy is still ~85%, older current-turn results may be stubbed too, then the model is asked for a structured summary and the live session is rewritten to the original user query plus that summary. If `/jev-compact` is on (session-only; needs `TYPESAFE_API_KEY`), that 85% reclaim scores tool pairs with Jev first and prunes in place instead of summarizing when occupancy then drops below 85%. Jev failures fall back to cheap shrink + summary. If the summary call fails while still over 85%, a local emergency compact runs and Sentinel may auto-repair. `--no-auto-compact` disables the automatic shrink/summary; `/compact [context]` still shrinks and requests a summary unless Jev prune already dropped occupancy under 85%. Tool payloads are also bounded: `read` returns 240 lines unless paged with `offset`/`limit`, and `HI_TOOL_RESULT_CHARS` controls the per-result character cap.

**Changes pane.** Every edit paints as a Claude Code-style row in the
transcript — `Updated src/ws.rs (+4 -1)` (or `Created` / `Deleted`) with the
numbered green/red hunks inline underneath; diffs longer than a dozen rows fold
to `… N more lines` and open on click or `Ctrl-O`. `Ctrl-G` (or `/diff`) opens
the running diff of the whole session: a `8 files changed +394 -46` summary,
one row per file with its own counts (`new` / `deleted` flagged), then each
file's hunks under a bold path header. The pane follows the session — it
re-diffs as each edit lands, at turn end, and after `/undo` — and files the
model creates appear even before git tracks them. Until the session has edited
anything it shows the whole working tree. Wide terminals dock it beside the
transcript (`Tab` focuses it, drag the `│` to resize); narrow ones use a
full-screen overlay. Click a file row to jump to its section; click a hunk or
press `n`/`p` to select one, which writes an `@path:N-M` chip into the prompt
so the next message quotes that hunk for the model. Clicking the `changed: …`
line above the prompt opens the pane pinned to the last turn's files.

**Undo.** Before mutation, `hi` creates a recoverable checkpoint: a dangling
commit with a throwaway index when Git is usable, otherwise a content-addressed
internal snapshot. `/undo` restores created, modified, and deleted files plus
modes and symlink targets. It refuses to overwrite a file changed externally
since the turn. If no checkpoint backend is available, normal YOLO mode pins a
warning and continues without prompting. `--confirm-edits` makes that case
strict; combine it with `--allow-no-checkpoint` to retain the YOLO fallback.
Checkpoints cannot undo non-file side effects.

**No nag-prompts — but a guard for the irreversible.** Rather than asking permission for every command (the thing everyone turns off), `hi` lets the model run freely and relies on `/undo` for recovery. The one exception is a small denylist of operations a checkpoint *can't* undo — `sudo`, `rm -rf` of home/root/system paths, `git push --force`, `curl … | sh`, `dd` to a disk, `mkfs`, fork bombs, shutdown — which are refused with a reason the model can act on. It's a seatbelt against accidents, not a security boundary; set `HI_ALLOW_DANGEROUS=1` to disable it. Tool results, web/research pages, browser AX/eval output, MCP payloads, and inbound `hi mcp serve` calls are untrusted data, not instructions.

**Egress confirms.** Attended default remains YOLO (`/permissions always`). In Ask and Auto, `browser_exec` and MCP `use_tool` pause on a confirm overlay; web fetch/research pause only in Ask. Session standing grants apply to an MCP `server`+`tool` pair, never to bash or the browser. Unattended confirms that cannot be answered live return unavailable (denied), not auto-approved. With a TypeSafe key, `/permissions auto` may skip the overlay for high-confidence reversible shell; denylisted commands and secret-adjacent files still confirm. `hi mcp serve` has no human on stdio — it stays denylist + sandbox + folder trust.

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
| `hi-ai` | provider types, HTTP, Pipe / OpenAI / Anthropic adapters |
| `hi-tools` | local tools: `read` / `write` / `edit` / `bash` / `list` / `grep` / `glob` plus repo/LSP helpers |
| `hi-harness` | Pipe coding loop, JSONL sessions, `/undo`, `/dashboard` runtime |
| `hi-dashboard-store` | SQLite membership for `/dashboard` (ported from Grok dashboard-store) |
| `hi-rsi-runtime` | managed candidate descriptor, workflow, budget, checkpoint, verification, failure, and exact-replay contracts |
| `hi-trace` | bounded content-addressed RSI artifacts and crash-safe hash-chained event journals |
| `hi-tui` | full-screen terminal UI (transcript, slash commands, agent dashboard) |
| `hi-cli` | the `hi` binary: config, Pipe session, login |
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
cargo check -p hi-harness --lib
cargo test -p hi-harness --lib
cargo test -p hi-dashboard-store --lib

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
- `cargo clippy -p hi-ai -p hi-tools -p hi-lsp -p hi-harness -p hi-tui -p hi -p hi-eval --all-targets -- -D warnings`
- `cargo test -p hi-ai -p hi-tools -p hi-lsp -p hi-harness -p hi-tui -p hi -p hi-eval`
- `cargo install --path crates/hi-cli --locked`
- Smoke an OpenAI-compatible endpoint with `--compat auto` and `--tool-mode auto`
- Validate eval tasks and immutable oracles with `cargo run -p hi-eval -- bench --validate`

The 0.1 GPU/local-inference crates have their own hardware-specific release
checks and are not gates for the core 0.2 release.

## Status

Early but functional. The multi-provider core, full-screen TUI, sessions, verify-loop, best-of-N, compatibility fallbacks, changed-file reporting, and eval harness are built and tested. Optional local CUDA/MLX inference is released and tested through the external `hi-local-runtime` sidecar contract. The TUI's rendering is verified via ratatui's TestBackend; its live key/scroll behavior is best confirmed in a real terminal. Cargo install is the first release target; binary archives and Homebrew can follow later.
