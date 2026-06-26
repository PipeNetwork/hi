# bench — coding-task eval for `hi`

Measures whether a lever (e.g. verification-in-the-loop) actually beats a
baseline, including a real backend like `openrouter/fusion`. Without numbers,
"overperform" is just vibes — this is how we get the numbers.

## How it works

`hi-eval` runs every task under every config in an isolated copy of the task's
`fixture/`, then scores pass/fail with the task's own `verify` command (ground
truth — the compiler/tests, not a judge model). It reports pass-rate, cost, and
tokens per config, and writes machine-readable artifacts for each task/config
cell. By default artifacts go under `target/hi-eval/runs/<timestamp>-<pid>/`
as one JSON file per run plus an append-only `runs.jsonl`.

Configs (in `crates/hi-eval/src/main.rs`):
- `baseline` — `hi` runs the prompt once, no verification.
- `verify` — `hi --verify <task.verify>`, so the agent iterates until green.
- `best-of-3` — three verified candidates at different temperatures; the config
  passes if any candidate passes.

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

### A/B the tool-output condenser

`hi` condenses test/compiler output (drops passing-test noise, keeps failures)
before it re-enters context. To measure what that's worth, run the same matrix
twice — toggling it with `HI_CONDENSE` — and compare the `tok/task` column (mean
tokens to attempt one task):

```bash
HI_MODEL=… HI_API_KEY=… \
  cargo run -p hi-eval -- bench/tasks                 # condense on (default)
HI_CONDENSE=0 HI_MODEL=… HI_API_KEY=… \
  cargo run -p hi-eval -- bench/tasks                 # condense off
```

The toggle is inherited by the `hi` subprocess; the run header and every
artifact record `condense=on|off`, so `runs.jsonl` rows are self-labeled for
offline analysis. The bet: `tok/task` drops with condensing on while `pass@1` /
`pass@k` hold — fewer tokens for the same solve rate.

### A/B recovery sampling

When a round comes back empty/garbled, `hi` resamples hotter (temperature +
nucleus + frequency penalty) on the retry to escape the stuck state.
`HI_RECOVERY_SAMPLING=0` disables it (the retry just re-runs at the configured
sampling); the header/artifacts record `recovery=on|off`.

Unlike the condenser, this **won't move `tok/task`** — it only fires on a stall,
which healthy models rarely produce. Its value shows up in **`pass@1` and the
`error` failure bucket**: on a flaky local model, recovery on should convert some
`error` cells (model returned nothing → gave up) into solves. On a model that
never stalls, both sides are identical — which is the honest result, not a bug.

```bash
HI_MODEL=… cargo run -p hi-eval -- bench/tasks                    # recovery on (default)
HI_RECOVERY_SAMPLING=0 HI_MODEL=… cargo run -p hi-eval -- bench/tasks  # recovery off
```

### Manual pipenetwork runs

Live provider evals are explicit and should not run in default CI. The
pipenetwork profile passes `--provider pipenetwork --compat auto --tool-mode auto`
to `hi` and requires `PIPENETWORK_API_KEY` (or `HI_API_KEY`):

```bash
PIPENETWORK_API_KEY=... \
  cargo run -p hi-eval -- --profile=pipenetwork --configs=baseline,verify bench/tasks
```

Use the tiers in order: `bench/tasks` smoke first, then `bench/spec`, then
selected `bench/hard`. Compare `baseline`, `verify`, and `best-of-3`; treat
artifact rows with `provider_error_kind` separately from model behavior buckets
like `no-edits`, `compile`, and `logic`.
