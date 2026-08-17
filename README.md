# hi

`hi` is a verification-first coding agent. Point it at a model — local or remote — and it reads, writes, and edits files and runs shell commands until your tests pass.

```bash
# Install (needs a Rust toolchain):
./scripts/install.sh

# First run opens a provider wizard. Then:
hi "the tests in test_parser.py are failing — fix the parser"
```

Interactive sessions open a full-screen TUI. Pass a prompt for one-shot. Piped stdin is folded in as context:

```bash
cargo test 2>&1 | hi "fix the failing tests"
```

## Everyday path

1. Ask for an outcome, not a patch recipe.
2. After edits, hi runs an auto-detected check pipeline (`cargo test`, `pytest`, `go test`, …) or `/verify <cmd>`.
3. Failures go back to the model. `/undo` restores the last turn.

In the TUI: **Ctrl-K** is the command palette (core commands first; type to search). `/help` is the same grouping. `/tutorial` is an eight-lesson tour, offered once on a fresh session.

| Job | Command |
|---|---|
| Finish line | `/verify [cmd\|off]` |
| Take it back | `/undo` |
| See the diff | `/diff` or Ctrl-G |
| Resume work | `hi resume` / `/sessions` |
| Settings | `/config` |

## Providers

`--provider` accepts `openai` (OpenRouter and any OpenAI-compatible URL), `anthropic`, `pipenetwork`, `ollama`, and `xai`. First-run `hi` or `hi setup` walks through all of them.

```bash
HI_API_KEY=sk-or-... hi -m anthropic/claude-sonnet-4 "add a --json flag"
PIPENETWORK_API_KEY=... hi --provider pipenetwork "…"
hi --provider ollama -m qwen2.5-coder "…"
XAI_API_KEY=xai-... hi --provider xai "…"
```

Profiles live in `./hi.toml` or `~/.config/hi/config.toml`. `/provider` switches mid-session.

## Named modes (when you need them)

| Intent | Name | How |
|---|---|---|
| Several attempts at one task | Race | `/race <task>` — headless: `hi --best-of N "…"` |
| Several tasks at once | Fleet | `/fleet` (`/dashboard` is an alias) |
| Helpers inside a turn | Delegates | `/delegate` · explore is on by default |
| Work for a week | Goal | `/goal <objective>` |
| Keep watching | Watch | `/loop`, `/watch`, `hi --loops-daemon` |
| This machine | Local | `/local` (MLX) or `--provider ollama` |

Power commands (RSI, Diff Lab, traces, eval) are in `/help platform` and the [handbook](docs/handbook.md).

## Trust

Default is YOLO with a seatbelt: no nag prompts, a denylist for irreversible commands, checkpoints for `/undo`. Shell writes stay in the project on macOS (Seatbelt) and on Linux when `pipe-wrap` is available. `HI_SANDBOX=off` disables that. The status bar shows **sandbox** and **undo**.

## Docs

- [Handbook](docs/handbook.md) — full CLI, TUI, loops, RSI, local GPU, eval
- [Architecture](docs/architecture.md) — interactive agent vs RSI control plane
- [Fleet](docs/fleet-dashboard.md)
- [Sandbox](docs/sandbox.md)
- [0.2 migration](docs/0.2-migration.md)

Homebrew formula (tap yourself or `brew install --build-from-source`): [packaging/homebrew/hi.rb](packaging/homebrew/hi.rb). Binary archives can follow; `cargo install --path crates/hi-cli --locked` is still the supported build.
