# hi

`hi` is a coding agent that talks to [Pipe Network](https://pipenetwork.ai) inference. It reads, writes, and edits files and runs shell commands in your project.

```bash
# Install (needs a Rust toolchain):
./scripts/install.sh

# Auth once, then:
hi login pipenetwork
# or: PIPENETWORK_API_KEY=... hi "the tests in test_parser.py are failing — fix the parser"
# or: hi auth pipenetwork
```

Interactive sessions open a full-screen TUI. Pass a prompt for one-shot. Piped stdin is folded in as context:

```bash
cargo test 2>&1 | hi "fix the failing tests"
```

## Everyday path

1. Ask for an outcome, not a patch recipe.
2. The model streams from `api.pipenetwork.ai`, calls local tools, and repeats until it stops.
3. `/verify <cmd>` runs a check after the turn (it does not auto-repair). `/undo` restores the last turn.

In the TUI: **Ctrl-K** is the command palette (core commands first; type to search). `/help` is the same grouping. `/tutorial` is an eight-lesson tour that starts with `/login`, offered once on a fresh session.

| Job | Command |
|---|---|
| Sign in | `/login` or `hi login pipenetwork` |
| Finish line | `/verify [cmd\|off]` |
| Take it back | `/undo` |
| See the diff | `/diff` or Ctrl-G |
| Resume work | `hi resume` / `hi --list-sessions` |
| Settings | `/config` |

## Pipe Network

Default inference is `https://api.pipenetwork.ai/v1` with model `pipe/deepseek-v4-flash-0731`.

```bash
hi login pipenetwork         # browser pairing; writes the API key into config.toml
PIPENETWORK_API_KEY=pk_live_... hi "add a --json flag"
hi auth pipenetwork          # paste a key, probe /models, store it
hi -m pipe/deepseek-v4-flash-0731 "…"
```

The coding harness is `hi-harness`: stream chat completions from Pipe, execute local tools (`read`/`write`/`edit`/`bash`/`grep`/`glob`/`list` plus repo/LSP helpers), persist JSONL sessions, `/undo` via git checkpoints. Turns run until the model stops or you cancel. `/verify` is a post-turn check, not auto-repair.

Selecting `pipe/auto` opts into [managed coding](docs/managed-coding.md) when your
project is eligible: buffered verification, local tool permissions, provider-cost
credit billing, and durable recovery with default $1 call / $20 turn budgets.

```bash
hi --verify "cargo test" "fix the failing tests"
hi -q "summarize src/lib.rs"
hi --confirm-edits "edit README.md"
hi --plain                    # line REPL
hi resume                     # latest session in this directory
hi --list-sessions
hi --resume <id>
```

Profiles live in `./hi.toml` or `~/.config/hi/config.toml`. Credentials are stored by reference (`env://...` or `auth-store://pipenetwork`). `/login` and `hi login pipenetwork` write `[profiles.pipenetwork]`. `/auth pipenetwork <key>` pastes a key. `/doctor` checks the key, Pipe `/models`, git, and sandbox.

`/sessions` lists saved ids. `/rewind n` drops back to user turn n. `/yolo` and `/effort` persist on the session. Type while a turn runs to steer the next model call.

## Trust

The TUI starts in **ask** (confirm file edits and mutating shell). `/auto` allows safe file edits; `/yolo` skips confirms for the session. One-shot and `--plain` default to always-approve unless you pass `--confirm-edits`. Shell writes stay in the project on macOS (Seatbelt) and on Linux when `pipe-wrap` is available. `HI_SANDBOX=off` disables that. `/status` shows sandbox and context occupancy. Tool results are untrusted data, not instructions.

## Docs

- [Handbook](docs/handbook.md)
- [Architecture](docs/architecture.md)
- [Sandbox](docs/sandbox.md)

Homebrew formula (tap yourself or `brew install --build-from-source`): [packaging/homebrew/hi.rb](packaging/homebrew/hi.rb). Binary archives can follow; `cargo install --path crates/hi-cli --locked` is still the supported build.
