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

# OpenRouter (default endpoint)
HI_API_KEY=sk-or-... hi -m anthropic/claude-sonnet-4 "add a --json flag to the CLI"

# A local model (Ollama / llama.cpp / LM Studio / vLLM — all OpenAI-compatible)
hi --base-url http://localhost:11434/v1 --api-key local -m qwen2.5-coder "..."

# Native Anthropic
HI_API_KEY=sk-ant-... hi --provider anthropic -m claude-sonnet-4-20250514 "..."
```

Run with no prompt for an interactive session; pass a prompt for one-shot.

## Models & providers

One OpenAI-compatible client covers **OpenRouter, terminaili.com, Ollama, llama.cpp, LM Studio, and vLLM** — they differ only by `--base-url` and `--api-key`. A native **Anthropic** adapter (`--provider anthropic`) adds extended thinking and tool-use blocks.

Settings resolve in this order: **CLI flags → profile → environment → defaults.**

| What | Flag | Env | Default |
|---|---|---|---|
| Model | `-m, --model` | `HI_MODEL` | — (required) |
| Base URL | `--base-url` | `HI_BASE_URL` | OpenRouter / `api.anthropic.com` |
| API key | `--api-key` | `HI_API_KEY`, then `OPENROUTER_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | — (required) |

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
```

### Model registry

`hi --refresh-models` pulls the [models.dev](https://models.dev) catalog into a local cache (~2700 models). It powers the per-turn cost/context display, caps `--max-tokens` to a model's limit, and warns when a model isn't known to support tool calling.

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

Slash commands (interactive or TUI): `/help`, `/model [id]`, `/tokens`, `/clear`, `/exit`.

Drop an `HI.md` or `AGENTS.md` in your project and its contents are appended to the system prompt — per-project conventions, for free.

A `--tui` flag enables an experimental ratatui interface (transcript in scrollback + a live input/status region).

## Architecture

A cargo workspace:

| crate | role |
|---|---|
| `pi-ai` | provider-neutral types, the `Provider` trait, OpenAI + Anthropic adapters, retry, models.dev registry |
| `pi-tools` | the `read` / `write` / `edit` / `bash` tools |
| `pi-agent` | the agent loop, verify-loop, sessions, the `Ui` trait |
| `pi-tui` | inline terminal UI |
| `pi-cli` | the `hi` binary: config, sessions, best-of-N, slash commands |
| `pi-eval` | the benchmark runner (see below) |

Richer capabilities come from **subprocess CLI tools** the model invokes via `bash` (pi's philosophy) rather than a plugin runtime.

## Benchmarks (`pi-eval`)

`bench/` measures whether the levers actually beat a baseline — including a real backend like [OpenRouter Fusion](https://openrouter.ai/blog/announcements/fusion-beats-frontier/). Each task ships a buggy `fixture/` and a `verify` command (the spec; the agent can't game it). `pi-eval` runs each task under three configs — `baseline`, `verify`, `best-of-3` — in isolated copies and scores pass/fail by ground truth.

```bash
cargo run -p pi-eval -- --validate bench/spec     # check tasks are well-formed (no model)

# Compare configs against any model (env flows through to hi):
HI_MODEL=anthropic/claude-sonnet-4 HI_API_KEY=$OPENROUTER_API_KEY \
  cargo run -p pi-eval -- bench/spec

# The raw-Fusion line to beat:
HI_MODEL=openrouter/fusion HI_API_KEY=$OPENROUTER_API_KEY \
  cargo run -p pi-eval -- bench/spec
```

Three task tiers: `bench/tasks` (easy bugs), `bench/hard` (edge-case algorithms), `bench/spec` (behavior pinned by the test, not the prompt). See `bench/README.md` to add tasks.

### What we've measured

On a local 30B coder, easy and hard tiers saturate at 6/6 baseline — a capable model self-verifies via `bash` and aces well-specified tasks. The `bench/spec` tier (unstated conventions the model can't self-test for) is where the loop earns its keep: **baseline 0/3 → verify 2/3.** That gap is the mechanism by which `hi` overperforms a single-shot ensemble like Fusion: ground-truth iteration it can't do. The headline Fusion comparison needs an OpenRouter key and is not yet run.

## Status

Early but functional. The multi-provider core, sessions, verify-loop, best-of-N, and eval harness are built and tested; `cargo test` / `cargo clippy --workspace` are clean. The TUI is experimental (its interactive rendering needs a real terminal). No published releases yet.
