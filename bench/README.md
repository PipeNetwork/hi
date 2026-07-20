# bench — coding-task eval for `hi`

Measures whether a lever (e.g. verification-in-the-loop) actually beats a
baseline, including a real backend like `openrouter/fusion`. Without numbers,
"overperform" is just vibes — this is how we get the numbers.

## How it works

`hi-eval` runs every task under every config in an isolated copy of `fixture/`.
Before candidate launch it captures the external oracle bundle. After the
candidate exits, it makes a new verification copy and injects those captured
bytes as `.hi-eval-oracle/`; neither the bundle nor its command is exposed
during the attempt. Candidate-side tests therefore cannot change the score.
Copies never follow symlinks: contained relative links are recreated with their
targets and file modes intact, while escaping links and special filesystem
nodes are rejected. Integrity snapshots include VCS metadata and pre-existing
dependency/build-tree entries; only explicitly recognized new runtime artifacts
(such as Cargo outputs and evaluator reports) are excluded from change scoring,
and those excluded artifacts are removed from the fresh oracle copy so they
cannot become hidden scorer inputs.

Artifacts retain every candidate (temperature, actual route, typed process
outcome, patch, checks, usage/cost when known, and duration). The run directory
also contains `summary.json` with candidate pass rate, solve@N, standard pass@k
only for exchangeable samples, false-verified count, infrastructure error rate,
solve rate, and cost per solved task.

Configs (in `crates/hi-eval/src/main.rs`):
- `baseline` — `hi --no-verify --allow-unverified` runs the prompt once.
- `verify` — uses task-visible feedback when provided, otherwise `hi`'s Auto
  verification pipeline.
- `best-of-3` — three verified candidates at different temperatures; the config
  passes if any candidate passes.

## Tasks

Each task is a directory under `bench/tasks/<name>/`:

```
bench/tasks/<name>/
  task.toml        # schema v2 contract, allowed changes, timeouts, oracle
  fixture/         # buggy code copied into the agent's work dir
  fixed/           # reference fix, overlaid only by `--validate` (never seen by the agent)
  oracle/          # optional final scorer bundle (captured, never exposed)
```

Task schema v2 requires `schema_version`, `prompt`, nonempty
`allowed_changes`, `[final_oracle].command`, and optional
`[final_oracle].bundle` / `[visible_feedback]`. `[timeouts]` independently
controls candidate, visible-feedback, and final-oracle deadlines; defaults are
900, 120, and 120 seconds. Optional `[workspace]` setup supports hermetic
`non_git`, `clean_git`, and `dirty_git` scenarios; dirty fixtures declare the
tracked path and exact pre-attempt user contents.

### Verify-loop suites: `bench/vloop` and `bench/vloop-dense`

These are the only categories whose tasks define `[visible_feedback]`, so they
are where the `verify` config actually differs from `baseline`: the feedback
command is handed to the agent as `--verify`, the repair loop runs on failure,
and the hidden oracle stays a strict superset of the visible checks (a run can
pass verify yet fail the oracle — the `false_verified` signal). `bench/vloop`
holds single-requirement tasks; `bench/vloop-dense` holds constraint-dense
prompts (5–7 explicit rules each, the specification-neglect scenario) where a
first attempt plausibly misses a stated rule and must recover through verify
feedback. Without these suites, a `baseline`-vs-`verify` comparison measures
nothing — no other bench task engages verification.

### Validate tasks (no model needed)

```bash
cargo run -p hi-eval -- --validate bench/tasks
```

Confirms every task fails before, passes after `fixed/`, rejects forbidden and
no-op changes, and cannot be passed with a candidate-created oracle file.

Add a task by writing `task.toml` + `fixture/` + `fixed/` and, where useful, an
`oracle/` bundle, then validate it. `cargo run -p hi-eval -- bench --validate`
recursively validates the complete suite.

## North-star ladder & baseline

Ordered suites for coding quality (cheap → multi-file):

| Tier | Path | Signal |
|---|---|---|
| Floor | `bench/tasks`, `bench/spec` | smoke / edge-case solve rate |
| Spec neglect | `bench/vloop-dense` | `false_verified` (visible green, oracle red) |
| Multi-file | `bench/hidden` | realistic package fixes |
| Long / plan | `bench/long`, `bench/plan` | horizon (optional) |

Locked metrics: `eval-baseline/core-0.2.json`. Capture a provider-backed matrix:

```bash
# Full north-star ladder (tasks + spec + vloop-dense + hidden), capture baseline
cargo run -p hi-eval -- --north-star --configs=baseline,verify --trials=1 \
  --write-baseline --artifacts=eval-artifacts/north-star-$(date +%Y%m%d)

# Or capture from an existing summary.json
hi-eval --write-baseline=eval-artifacts/.../summary.json
```

Compare later runs with auto-compare (always on when the baseline file exists)
or `hi-eval --compare-baseline=…` / `--fail-on-baseline-regression`.

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

Runs default to three trials, a 900-second candidate deadline, a 120-second
oracle deadline, and global candidate concurrency 4. Override with
`--trials`, `--candidate-timeout`, `--oracle-timeout`, and `--concurrency`;
zero concurrency is rejected.

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
offline analysis. The bet: `tok/task` drops while candidate pass rate and
solve@N hold — fewer tokens for the same solve rate.

### A/B recovery sampling

When a round comes back empty/garbled, `hi` resamples hotter (temperature +
nucleus + frequency penalty) on the retry to escape the stuck state.
`HI_RECOVERY_SAMPLING=0` disables it (the retry just re-runs at the configured
sampling); the header/artifacts record `recovery=on|off`.

Unlike the condenser, this **won't move `tok/task`** — it only fires on a stall,
which healthy models rarely produce. Its value shows up in **candidate pass rate and the
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
