# Quality routing suite

First-tool + forbidden-tool rows for the hi agent loop. Outcome still uses a
cheap `answer.txt` oracle; `judge.toml` scores the `--report` tape. Live cells
seed a throwaway git checkout (`[run] init_git`) so git identity is visible
without committing a nested `.git`.

```text
# no provider — replay committed tapes
cargo run -p hi-eval -- judge --suite bench/quality
scripts/check_harness_regression.sh

# validate fail-before / pass-after fixtures
cargo run -p hi-eval -- --validate bench/quality

# live (optional; one trial, isolated artifacts)
cargo build -p hi
HI_STATE_DIR=/tmp/hi-quality-state HI_BIN=target/debug/hi cargo run -p hi-eval -- \
  bench/quality --configs=baseline --trials=1 --artifacts=artifacts/quality
```

Prefer `--configs=baseline` so verify-loop noise does not drown routing.
A candidate can pass the hidden oracle and still fail process.

Live process/budget is gated against `eval-baseline/quality.json` (5pp drop,
same as harness). Core CI only replays tapes; the Monday
`evaluate-quality` job runs the live suite. Do not average quality
`process_pass_rate` with harness or SWE.

```text
scripts/check_quality_regression.sh artifacts/quality eval-baseline/quality.json
```

`--report` / `--eval-input` omit `ask_user` from the catalog and fail the
tool if the model emits it anyway. `broad-web-search` needs a live
`web_search` backend (`HI_WEB_SEARCH_API_KEY`); without a key the cell
should fail process if the model falls back to `web_fetch`.

Two-binary A/B (swapped trial order, SHA-256 of each `hi`, process-pass deltas):

```text
hi-eval ab --baseline-bin /abs/hi-old --candidate-bin /abs/hi-new \
  bench/quality --configs=baseline --trials=3 --artifacts=artifacts/quality-ab
```

Also accepts `HI_BIN_BASELINE` / `HI_BIN_CANDIDATE`. Writes `ab_meta.json` and
`ab_report.json` under the artifacts root.

Live routing status (passing / partial / known_gap) lives in
`bench/quality/matrix.toml`. Keep it in sync when adding a row.

## Matrix

| id | category | first tools | forbidden | coverage | notes |
|----|----------|-------------|-----------|----------|-------|
| local-discovery | local search | grep/glob/read/find_symbol/repo_map | web_search, ask_user | tape + live | unique `quality_marker_fn` |
| inspect-before-ask | ask-user misuse | local inspect or bash | web_search, ask_user | tape + live | must read source, not ask |
| github-handle-not-needed | ask-user misuse | none (context-only is a win) | web_search, ask_user | tape + live | needs git identity origin |
| changelog-local-first | GitHub routing | local inspect or bash | web_search, ask_user | tape + live | unique changelog token |
| git-history-local | local search | bash matching `^git` | web_search, ask_user | tape + live | commit subject from `init_git` |
| known-url-web-fetch | web routing | web_fetch | web_search, ask_user | tape + live | exact URL; heading from example.com |
| broad-web-search | web routing | web_search | web_fetch, web_download, ask_user | tape + live | no guessed fetch; cited https URL |
