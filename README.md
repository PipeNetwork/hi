# hi

A minimal agentic coding tool in Rust — a port of the [pi](https://github.com/badlogic/pi-mono) coding agent, built to support **many models** (local and remote) and to **beat single-shot quality with ground-truth iteration**.

`hi` reads, writes, and edits files and runs shell commands in your project, driven by whatever model you point it at. Its distinguishing feature is **verification-in-the-loop**: give it a test command and it iterates until the tests pass — something a single-shot completion endpoint structurally can't do.

```bash
# Fix failing tests with a local model, iterating until green:
hi --auto-verify "the tests in test_parser.py are failing — fix the parser"
```

## Quick start

```bash
cargo build --release           # binary at target/release/hi
cargo install --path crates/hi-cli --locked

# OpenRouter (default endpoint)
HI_API_KEY=sk-or-... hi -m anthropic/claude-sonnet-4 "add a --json flag to the CLI"

# terminaili.com (OpenAI-compatible coding endpoint; defaults to ipop/coder-balanced)
TERMINAILI_API_KEY=... hi --provider terminaili "add a --json flag to the CLI"

# A local Ollama model (no API key needed)
hi --provider ollama -m qwen2.5-coder "..."

# Native Anthropic
HI_API_KEY=sk-ant-... hi --provider anthropic -m claude-sonnet-4-20250514 "..."
```

`--provider` accepts `openai` (any OpenAI-compatible URL), `anthropic`, `terminaili`, and `ollama`. The latter two are presets: they set the right base URL, key env var (`TERMINAILI_API_KEY`), and — for terminaili — a default model, so they work with no extra flags. `hi --version` prints the Cargo package version.

### Fallback chain

A single dead or overloaded provider shouldn't kill your session. Give a profile a `fallback` list (or pass `--fallback <profile>`, repeatable); if the primary errors or returns nothing, `hi` announces the switch and retries the next one:

```toml
default_profile = "cloud"

[profiles.cloud]
provider = "terminaili"
api_key = "..."
fallback = ["local"]      # → falls back to the `local` profile

[profiles.local]
provider = "ollama"
model = "qwen2.5-coder"
```

A model that streams only keep-alive heartbeats with no output is treated as failed after `HI_STREAM_TIMEOUT` seconds (default 120; set lower to fail over faster). `HI_DEBUG_STREAM=1` dumps raw provider bytes for diagnosing one that returns nothing.

OpenAI-compatible endpoints vary in how much of Chat Completions they implement. The default `--compat auto` retries common simpler shapes: first without streamed usage metadata, then without tool calling when the provider rejects tools. Use `--compat strict` to send only the initial request shape. Tool availability is controlled separately with `--tool-mode auto|required|chat-only|read-only`; `required` fails instead of degrading when tools are rejected, `chat-only` advertises no tools, and `read-only` only advertises safe inspection tools.

Run with no prompt for an interactive session; pass a prompt for one-shot. Piped stdin is folded into a one-shot prompt as context, so `hi` composes with other tools:

```bash
cargo test 2>&1 | hi --auto-verify "fix the failing tests"
cat error.log | hi "what's going wrong here?"
cat data.json | hi -q "extract every email address" | sort -u   # -q: text only, no chatter
```

## Models & providers

One OpenAI-compatible client covers **OpenRouter, terminaili.com, Ollama, llama.cpp, LM Studio, and vLLM** — they differ only by `--base-url` and `--api-key`. A native **Anthropic** adapter (`--provider anthropic`) adds extended thinking and tool-use blocks.

Settings resolve in this order: **CLI flags → profile → environment → defaults.**

| What | Flag | Env | Default |
|---|---|---|---|
| Model | `-m, --model` | `HI_MODEL` | — (required) |
| Base URL | `--base-url` | `HI_BASE_URL` | OpenRouter / `api.anthropic.com` |
| API key | `--api-key` | `HI_API_KEY`, then `OPENROUTER_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | — (required) |
| Tool mode | `--tool-mode` | — | `auto` |
| Compatibility | `--compat` | — | `auto` |

### Config profiles

Keep several models on hand in `./hi.toml` or `~/.config/hi/config.toml` and switch with `-p`:

```toml
default_profile = "sonnet"

[profiles.sonnet]
provider = "anthropic"
model = "claude-sonnet-4-20250514"
api_key_env = "ANTHROPIC_API_KEY"

[profiles.local]
provider = "openai"
base_url = "http://localhost:11434/v1"
model = "qwen2.5-coder"
tool_mode = "read-only"   # auto|required|chat-only|read-only
compat = "auto"           # auto|strict
```

### Model registry

`hi --refresh-models` pulls the [models.dev](https://models.dev) catalog into a local cache (~2700 models). It powers the per-turn cost/context display, caps `--max-tokens` to a model's limit, and warns when a model isn't known to support tool calling.

One-shot automation can write a JSON report with `--report path.json`. Reports include token/cost totals, `verify_passed`, `provider_error_kind`, `compat_fallbacks_used`, `tool_mode_effective`, and `changed_files`.

## Verification-in-the-loop

The headline feature. After the model stops, `hi` runs a check; if it fails, the output is fed back and the model iterates (up to `--max-verify` rounds, default 3).

```bash
hi --verify "cargo test" "make the failing test pass"
hi --auto-verify "..."     # detects cargo test / pytest / npm test / go test / make test
```

A `--max-steps` cap (default 50) stops runaway tool loops. Each turn prints `[N in · N out · N total · $cost · k/k ctx]`.

## Best-of-N

Run several attempts and keep the one that actually passes — the **test suite is the judge**, not another model.

```bash
hi --best-of 3 --auto-verify "implement the spec in README"
```

It runs N candidates (varied temperature) in isolated **git worktrees**, each with its own verify-loop, stops at the first that passes verification, and applies that candidate's diff back to your working tree. Requires a git repo and `--verify`/`--auto-verify`; run from a clean tree (candidates branch from HEAD).

## Sessions

Every session is saved as JSONL under `~/.local/share/hi/sessions/`.

```bash
hi -c "and now add tests"          # --continue the latest session
hi --resume <id> "..."             # resume a specific one
hi --list-sessions                 # list saved sessions
hi --no-save "..."                 # don't persist
```

## In-session commands & context

Slash commands (TUI or plain REPL):

| command | does |
|---|---|
| `/help` | list commands |
| `/model [id]` | switch by id, or — with no id — open an interactive picker over the models **your provider actually serves** (live `/v1/models`; type to filter, ↑/↓, Enter). Falls back to the models.dev catalog if the endpoint can't list them. |
| `/verify [cmd\|off]` | show, set, or clear the test command turns iterate against — turn the verify-loop on without restarting |
| `/diff` | show what files have changed this session (`git diff` + new files) |
| `/copy [all]` | copy the last assistant response to the terminal clipboard; `all` copies the transcript |
| `/goal [text\|clear]` | show, set, or clear the current session goal injected into future turns |
| `/compact` | summarize the conversation and reset context to the summary (reclaims room on long sessions) |
| `/retry` | re-run your last message (drops the previous attempt — pairs with `/model`) |
| `/undo` | revert the file changes the last turn made (restores its git checkpoint) |
| `/tokens` | cumulative token usage |
| `/clear` | start a fresh conversation |
| `/exit` | quit |

Drop an `HI.md` or `AGENTS.md` in your project and its contents are appended to the system prompt — per-project conventions, for free.

Long sessions **auto-compact**: during long tool loops, `hi` elides older bulky tool results once the local context estimate passes ~45% full, keeping the newest tool results verbatim. Before a new turn, if the previous request used ~80% of the context window, it summarizes the conversation and resets to that summary. Disable with `--no-auto-compact`; trigger manually any time with `/compact`. Tool payloads are also bounded by default: `read` returns 240 lines unless paged with `offset`/`limit`, and `HI_TOOL_RESULT_CHARS` controls the per-result character cap.

**Undo.** In a git repo, `hi` snapshots the working tree before every turn into a *dangling* commit — built in a throwaway index, so it never touches your branch, staging area, or history. `/undo` restores the latest snapshot, reverting every file the turn created, modified, or deleted in one step. That's what makes running without confirmation prompts safe: anything the agent does to your files is one command away from being undone. (Covers non-ignored files; it can't undo non-file side effects.)

**No nag-prompts — but a guard for the irreversible.** Rather than asking permission for every command (the thing everyone turns off), `hi` lets the model run freely and relies on `/undo` for recovery. The one exception is a small denylist of operations a checkpoint *can't* undo — `sudo`, `rm -rf` of home/root/system paths, `git push --force`, `curl … | sh`, `dd` to a disk, `mkfs`, fork bombs, shutdown — which are refused with a reason the model can act on. It's a seatbelt against accidents, not a security boundary; set `HI_ALLOW_DANGEROUS=1` to disable it.

Interactive sessions open a **full-screen TUI** by default (ratatui): a bordered, scrollable transcript with a title bar showing live token/cost, and an input box that turns into a working spinner (with elapsed seconds) while a turn runs. **Keep typing while it works to queue the next command(s)** — they're listed under the prompt and run in order as each turn finishes. Ctrl-C interrupts the current turn (and drops the queue), PgUp/PgDn scrolls, Up/Down recalls history, `/exit` quits. Pass `--plain` (or pipe input) for the line-based REPL.

## Architecture

A cargo workspace:

| crate | role |
|---|---|
| `hi-ai` | provider-neutral types, the `Provider` trait, OpenAI + Anthropic adapters, retry, models.dev registry |
| `hi-tools` | the `read` / `write` / `edit` / `bash` / `list` / `grep` tools |
| `hi-agent` | the agent loop, verify-loop, sessions, the `Ui` trait |
| `hi-tui` | full-screen terminal UI (transcript, spinner, queue, slash commands) |
| `hi-cli` | the `hi` binary: config, sessions, best-of-N, slash commands |
| `hi-eval` | the benchmark runner (see below) |

Richer capabilities come from **subprocess CLI tools** the model invokes via `bash` (pi's philosophy) rather than a plugin runtime.

## Benchmarks (`hi-eval`)

`bench/` measures whether the levers actually beat a baseline — including a real backend like [OpenRouter Fusion](https://openrouter.ai/blog/announcements/fusion-beats-frontier/). Each task ships a buggy `fixture/` and a `verify` command (the spec; the agent can't game it). `hi-eval` runs each task under three configs — `baseline`, `verify`, `best-of-3` — in isolated copies and scores pass/fail by ground truth. It separates provider failures from model behavior, records changed files, preserves verify output for compile-vs-logic bucketing, reports request-shape fallbacks, and writes per-run JSON plus `runs.jsonl` artifacts under `target/hi-eval/runs/...` by default.

```bash
cargo run -p hi-eval -- --validate bench/spec     # check tasks are well-formed (no model)

# Compare configs against any model (env flows through to hi):
HI_MODEL=anthropic/claude-sonnet-4 HI_API_KEY=$OPENROUTER_API_KEY \
  cargo run -p hi-eval -- bench/spec

# The raw-Fusion line to beat:
HI_MODEL=openrouter/fusion HI_API_KEY=$OPENROUTER_API_KEY \
  cargo run -p hi-eval -- bench/spec

# Manual terminaili smoke (never part of default CI):
TERMINAILI_API_KEY=... \
  cargo run -p hi-eval -- --profile=terminaili --configs=baseline,verify bench/tasks

# After the smoke is stable, compare in this order:
cargo run -p hi-eval -- --profile=terminaili bench/tasks
cargo run -p hi-eval -- --profile=terminaili bench/spec
cargo run -p hi-eval -- --profile=terminaili bench/hard
```

Three task tiers: `bench/tasks` (easy bugs), `bench/hard` (edge-case algorithms), `bench/spec` (behavior pinned by the test, not the prompt). See `bench/README.md` to add tasks.

### What we've measured

`--trials=3` on the 20-task `bench/spec` tier with a local **`qwen2.5:7b`** (Ollama), scored by ground truth — `pass@1` = mean ± std tasks passed per trial; `pass@k` = tasks solved by at least one of the three trials:

| config | pass@1 | pass@k |
|---|---|---|
| baseline | 1.3 ± 0.5 / 20 | 2 / 20 |
| **verify** | **3.0 ± 0.8 / 20** | **5 / 20** |

Verification-in-the-loop more than doubles the local pass rate (the ±std bands don't overlap) and lifts the ceiling from 2 → 5 distinct tasks — ground-truth iteration a single-shot endpoint structurally can't do. That gap is the mechanism by which `hi` aims to overperform an ensemble like Fusion. Two caveats from the same harness, kept honest:

- **Capability gates the payoff.** A coder-tuned `qwen2.5-coder:7b` scored **0/20** — coder-tuning broke its tool-use (it stops emitting edits), a *worse* agent than the general 7b. A larger general `qwen2.5:32b` *is* a competent agent and the lift replicates (it converts `leap-year` baseline-fail → verify-pass), but it's slow (~2–6 min per verify task) and still can't crack the hardest hidden edges (`gcd`, `roman-to-int`).
- **A tweak that failed its own measurement.** A "reflect-then-fix + don't-repeat" feedback rewrite *lowered* the local verify score (3.0 → 2.0) and was reverted; the simple, direct "compare expected vs actual and fix" feedback wins on a weak model.

The headline Fusion comparison needs an OpenRouter key and is not yet run.

## Release checklist

- `cargo fmt --all`
- `cargo test --workspace`
- `cargo install --path crates/hi-cli --locked`
- Smoke an OpenAI-compatible endpoint with `--compat auto` and `--tool-mode auto`
- Validate eval tasks with `cargo run -p hi-eval -- --validate bench/spec`

## Status

Early but functional. The multi-provider core, full-screen TUI, sessions, verify-loop, best-of-N, compatibility fallbacks, changed-file reporting, and eval harness are built and tested (`cargo fmt --all` and `cargo test --workspace`). The TUI's rendering is verified via ratatui's TestBackend; its live key/scroll behavior is best confirmed in a real terminal. Cargo install is the first release target; binary archives and Homebrew can follow later.
