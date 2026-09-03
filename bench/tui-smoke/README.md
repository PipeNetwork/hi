# Interactive TUI smoke suite

`hi-smoke` launches the real `hi` binary in a pseudoterminal, drives user input
and keys, and checks typed lifecycle events together with terminal, session,
provider, process, and workspace evidence. It complements the model-facing
coding evaluations and the headless process/budget harness.

```bash
cargo build -p hi -p hi-smoke --no-default-features
target/debug/hi-smoke validate bench/tui-smoke
target/debug/hi-smoke run bench/tui-smoke \
  --mode scripted --tag pr --artifacts artifacts/tui-smoke
target/debug/hi-smoke fuzz bench/tui-smoke \
  --seed-start 0 --seeds 12 --jobs 4 \
  --artifacts artifacts/tui-smoke-fuzz
```

The runner resolves the candidate binary from `--hi-bin`, then `HI_BIN`, then
`target/debug/hi`. Failed runs write a self-contained `replay.toml` bundle;
reproduce one with `target/debug/hi-smoke replay <bundle>/replay.toml`. A live
replay restores the exact scenario, provider, model, and base URL while reading
the API key from the replaying process's environment. Remote model responses
remain nondeterministic; exact byte-for-byte response replay requires promoting
the failure to scripted provider steps.

Scripted and live cases normally give the child an explicit 30-second and
240-second soft turn deadline, respectively. A continual-operation regression
can set `hi.turn_deadline_secs = 0` and widen the independent harness watchdog
with `hi.outer_turn_kill_secs`; its scenario timeout remains the final external
process-cleanup boundary without changing Hi's product behavior.

## Tags and automation

- `pr` is the fast, provider-free curated subset required by Core CI.
- `curated` contains every deterministic regression and runs weekly on macOS.
- `live` contains exactly three small provider-backed canaries and supplies the
  reviewed nightly baseline. `live_extended` covers queued input/resize,
  plan-drive approval/editing, and workspace mutation in a separate nightly
  run. Live metrics remain separate from scripted and coding-eval scores.

The nightly Ubuntu campaign also executes 250 deterministic chaos seeds across
transport faults and stateful approval, pause/resume, restart, resize, queued
input, and tool-cancellation templates. The weekly macOS campaign executes 50.
Live runs use `HI_MODEL`, `HI_API_KEY`, and `HI_BASE_URL`, following the
evaluation workflow conventions. `HI_PROVIDER` is optional and defaults to
`openai`; set it to `pipenetwork` to exercise Pipe's production provider path.
Credentials are forwarded to the child through the provider-specific
environment variable and never placed in its argument list. Scenario TOML may
not override any supported provider credential alias or HTTP proxy variable;
the harness also inserts its selected credential after scenario variables as
defense in depth. Session sync is
forced off inside the isolated smoke config so a canary never publishes its
synthetic transcript to a remote dashboard, and the optional Pipe exit-rating
prompt is disabled so normal shutdown remains non-interactive. The child also
runs with `HI_TRACE_CAPTURE=off`: the harness-owned semantic JSONL is the
diagnostic contract, and an ordinary local RSI trace would duplicate evidence
outside the case bundle.

On failure, live-provider evidence contains only the typed scalar wire audit
(route/model, request-shape flags, attempt, acceptance, and HTTP status), never
the request body or authorization data. Screen evidence replaces the random
workspace and isolation paths with stable placeholders and trims unused
terminal cells. `assertions.json` evaluates the remaining assertions even when
an action fails early, while `process.json` records the observed or cleanup
exit code and descendant-leak evidence. The harness also content-hashes the
entire writable isolation tree before and after each case (excluding the
workspace, which already has patch/listing evidence). Only structurally known
Hi lifecycle files—such as the selected session/event streams, crash marker,
model cache, portal database, and digest-namespaced project runtime—may change.
Any other HOME/XDG/config/state/cache/tmp or fixture-snapshot mutation is a hard
infrastructure failure recorded in `isolation-evidence.json` with relative
paths and content digests, never file bodies. These guarantees make bundles
useful for diagnosis without depending on raw terminal text or exposing
prompts and credentials.

Passing case summaries and the suite summary retain the same non-secret live
route plus aggregate request, acceptance, and HTTP-status counts, so discarded
isolation directories do not erase proof that the configured provider route
handled wire traffic. Nightly baseline evaluation rejects any passing live case
without accepted HTTP evidence. Case summaries also retain a zero-valued
unexpected-isolation-mutation count.

Live provider request counts and HTTP statuses are recorded as non-gating health
and cost evidence. Provider compatibility fallback, empty-stream recovery, and
capacity retries may compose as needed; a successfully recovered `hi` turn does
not fail solely because it crossed an arbitrary request-count ceiling.

## Generic-completion negative control

The three plan-drive generic-completion regressions carry the
`generic-completion-guard-negative-control` tag. Run their executable
negative-control proof with:

```bash
scripts/check_tui_smoke_generic_completion_negative_control.sh
```

The script first builds `hi` with the deliberately unsafe, non-default
`smoke-negative-control-disable-generic-completion-guards` feature and requires
each regression to fail. It then rebuilds the ordinary feature-free binary and
requires all three to pass. That feature exists only to prove the smoke tests
can detect removal of the production guards; never use it for a user, packaged,
or release build.

Do not create `eval-baseline/tui-live.json` from an ad hoc run. Observe seven
successful nightly runs, explicitly review the resulting metrics, and only then
capture that baseline. A nightly advances this window only when the deterministic
campaign, the three baseline canaries, the extended live regressions, and evidence
upload all succeed. Once captured, live gating requires zero crashes or
infrastructure loops and no more than a five percentage-point scenario-pass
regression.
