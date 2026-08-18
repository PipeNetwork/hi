# Harness-of-the-harness

Process + budget tests for the hi agent loop. Outcome still uses `task.toml`
oracles; `judge.toml` scores the `--report` tape.

```text
# no provider — replay committed tapes
cargo run -p hi-eval -- judge --suite bench/harness
scripts/check_harness_regression.sh

# validate fail-before / pass-after fixtures
cargo run -p hi-eval -- --validate bench/harness

# live (optional; one trial, isolated artifacts)
HI_STATE_DIR=/tmp/hi-state HI_MODEL=… cargo run -p hi-eval -- \
  bench/harness --configs=verify --trials=1 --artifacts=artifacts/harness
scripts/check_harness_regression.sh artifacts/harness
```

A candidate can pass the hidden oracle and still fail process or budget.
`tape/fail.json` must violate the sibling `judge.toml`; `tape/pass.json` must not.
Live-only tasks may ship only a fail tape plus a README in `tape/`.

Optional `[run]` in `judge.toml` is live-only and does not change `bench/tasks`:

```toml
[run]
steps = [3, 8]                 # two hi invocations, same --session-file
seed_image_chars = 2000000     # token-bomb-image
seed_tool_result_chars = 80000 # resume-elide
ignore_change_prefixes = ["bug/"]  # inner-task tree; restored before oracle
```

`HI_EVAL_RESUME=1` is the env fallback (`steps = [3, 8]`) when a task has no `[run]` block.
