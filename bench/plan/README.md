# Plan-completion evals (`/goal`)

These fixtures measure the thing `/goal` is *for*: given a multi-section `plan.md`
and "build all of it", **what fraction of the plan does the agent actually
deliver?** Each fixture's hidden oracle scores per section and prints a fraction,
so a run that finishes 4 of 6 sections scores **0.67**, not a flat fail — which
is what you need to see an improvement like *20% → 75%*, rather than pass/fail.

Unlike the single-turn `bench/tasks`, these run in **goal-drive mode**: the
candidate sets a long-horizon goal from the prompt and the harness drives it
across `HI_EVAL_TURNS` session-continuing turns (the real `/goal` cadence).

## Fixtures

- **`textkit/`** — a small Python text-utilities library: six independent
  modules (`slugify`, `wordcount`, `roman`, `caesar`, `rpn`, `csvmini`), each an
  exact contract in `fixture/plan.md`. Score = modules delivered / 6.

## Running it

Build the `hi` binary you want to measure, then point the harness at it:

```sh
# From the repo root. Uses the binary at $HI_BIN (or target/{debug,release}/hi).
HI_BIN=./target/release/hi \
HI_MODEL=<provider/model> HI_API_KEY=<key> \
HI_EVAL_GOAL=1 HI_EVAL_TURNS=12 \
cargo run -p hi-eval -- bench/plan
```

- `HI_EVAL_GOAL=1` turns on goal-drive mode (`--goal <prompt>` on turn 1).
- `HI_EVAL_TURNS=12` gives the goal up to 12 drive turns to finish the plan.
  Raise it for larger plans.

## Reading the score

The hidden scorer (`oracle/score.py`) prints, in the run's `final_oracle`
artifact:

```
=== textkit plan-completion score ===
[PASS] slugify
[FAIL] roman: ...
...
sections delivered: 4/6 (67%)
HI_EVAL_SCORE=0.6667
```

`HI_EVAL_SCORE` (0..1) is the "% of the plan delivered." A fully-delivered plan
also exits 0, so it registers as a normal harness **pass**.

### Baseline vs. candidate

Run the same fixture against two binaries and compare the fractions — e.g. the
old goal engine vs. the completion fixes:

```sh
HI_BIN=./hi-baseline    ... cargo run -p hi-eval -- bench/plan   # e.g. 0.17
HI_BIN=./target/release/hi ... cargo run -p hi-eval -- bench/plan   # e.g. 0.83
```

## Validating a fixture (no model, deterministic)

The harness self-checks each fixture — the empty `fixture/` must score below
100% and the `fixed/` reference must score 100%:

```sh
cargo run -p hi-eval -- --validate bench/plan
```

You can also score any built workspace directly:

```sh
mkdir -p /path/to/built/.hi-eval-oracle
cp bench/plan/textkit/oracle/score.py /path/to/built/.hi-eval-oracle/
( cd /path/to/built && python3 .hi-eval-oracle/score.py )
```

## Adding a fixture

Mirror `textkit/`:

- `fixture/plan.md` — the spec the candidate reads (sections with exact
  contracts + examples). Do **not** put the oracle here.
- `oracle/score.py` — the hidden scorer: check each section independently,
  print `sections delivered: X/N` and `HI_EVAL_SCORE=<0..1>`, exit 0 iff all
  delivered. It runs from the candidate's workspace (`sys.path` includes cwd).
- `fixed/` — a reference implementation that scores 100% (proves the plan is
  buildable; used by `--validate`).
- `task.toml` — `prompt` = the objective, `allowed_changes` globs, and
  `[final_oracle]` with `bundle = "oracle"` + the scorer command.

Keep sections small, independent, and standard-library-only so scoring is fast
and deterministic.
