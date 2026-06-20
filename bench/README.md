# bench — coding-task eval for `hi`

Measures whether a lever (e.g. verification-in-the-loop) actually beats a
baseline, including a real backend like `openrouter/fusion`. Without numbers,
"overperform" is just vibes — this is how we get the numbers.

## How it works

`hi-eval` runs every task under every config in an isolated copy of the task's
`fixture/`, then scores pass/fail with the task's own `verify` command (ground
truth — the compiler/tests, not a judge model). It reports pass-rate, cost, and
tokens per config.

Configs (in `crates/hi-eval/src/main.rs`):
- `baseline` — `hi` runs the prompt once, no verification.
- `verify` — `hi --verify <task.verify>`, so the agent iterates until green.

## Tasks

Each task is a directory under `bench/tasks/<name>/`:

```
bench/tasks/<name>/
  task.toml        # name, prompt, verify (shell command; exit 0 = solved)
  fixture/         # buggy code copied into the agent's work dir
  fixed/           # reference fix, overlaid only by `--validate` (never seen by the agent)
```

The check lives in `verify` (a command), not a test file in the fixture, so the
agent can't game it by editing the test. The current suite is small,
dependency-light coding bugs (Python + shell): `factorial`, `fizzbuzz`,
`flatten`, `binary-search`, `count-vowels`, `greet-sh`, plus the `answer-42`
smoke task.

### Validate tasks (no model needed)

```bash
cargo run -p hi-eval -- --validate bench/tasks
```

Confirms every task is well-formed: `verify` fails on the raw `fixture/` and
passes once `fixed/` is overlaid. Run this when adding tasks.

Add a task by writing `task.toml` + `fixture/` + `fixed/`, then validating.

## Running

Model selection flows to `hi` via env vars, so you compare backends by swapping
env — not code:

```bash
cargo build -p hi

# A single frontier model through hi
HI_MODEL=anthropic/claude-sonnet-4 HI_API_KEY=$OPENROUTER_API_KEY \
  cargo run -p hi-eval -- bench/tasks

# OpenRouter Fusion as the backend — the thing to beat
HI_MODEL=openrouter/fusion HI_API_KEY=$OPENROUTER_API_KEY \
  cargo run -p hi-eval -- bench/tasks
```

Both default to OpenRouter's base URL. The win condition: the `verify` config
beats raw Fusion on pass-rate, and beats Fusion-as-a-model on $/solved-task.
