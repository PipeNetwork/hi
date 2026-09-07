# Harness simplification implementation

Baseline: `4d2867f`. Implementation uses the existing native harness, journal,
workspace controllers and session formats. The pre-existing untracked
`crates/hi-tui/.hi/` is unrelated to this change.

## Policy and ownership

| Concern | Implemented default / owner |
| --- | --- |
| Provider silence | 300 seconds from physical dispatch; advancing model content only |
| Inference recovery | Four physical sends per operation; 60 seconds total backoff |
| Semantic recovery | Three shared interventions; only a new best objective result replenishes |
| Productive work | No new overall step or time cap |
| Settlement | Agent runner; one 60-second caller deadline |
| Missing final answer | Deterministic closeout without another model request |
| Journal writes | One retained writer per canonical database path; bounded queue of 128 |
| Session writes | One retained owner per installed sink; bounded queue of 128 |
| Restore | Authoritative SessionReducer for local and remote records |

Provider settings are `provider.max_attempts` and
`provider.no_progress_timeout` (milliseconds; zero explicitly disables silence
recovery). Semantic recovery uses `recovery.max_interventions`. Existing explicit
operator and tool limits remain additional ceilings.

## Delivered changes

1. Removed the WASM host, guest and API crates, shadow NativeDirector, generation
   leases, module watchers and engine build script. Removed their resolved
   Wasmtime/Cranelift dependencies. Historical saved settings remain readable;
   active use receives a migration error. Evaluation identity now records native
   turn-policy version 1 rather than director version.
2. Added one shared execution context to ChatRequest. Physical dispatch, replay
   identity, usage, route events, retry decisions and backoff survive transport,
   authentication, compatibility and fallback changes. MoA reserves aggregation
   before children and shares three recovery sends. Explicit nonretryable errors
   terminate recovery; provisional calls cannot execute after abandoned output.
3. Centralized liveness from dispatch through headers and streaming. Queue and
   explicit approval waits are excluded; heartbeat/status output does not reset
   silence. TUI activity is a rendering of typed provider events.
4. Added persisted task recovery, canonical validation observations and explicit
   answer state. Failure cycles and exhausted recovery return typed non-success
   outcomes without automatic drive continuation. Exact scope and revision bind
   validation evidence; unrelated passing checks cannot resolve broader failures.
   One typed execution decision selects verification or absorbing settlement.
   Model answers and deterministic closeouts have distinct typed provenance;
   missing answers and private repair markers cannot trigger suggestion inference.
   Settlement never rebinds a past pass to new prose or source bytes. Optional
   skill curation and coding memory publish once before settlement, followed by
   deterministic revalidation when they change canonical inputs. Exhausting the
   existing verification ceiling retains typed invalidation and cannot trigger
   another main-model request.
   Task review survives separate owned metadata writes only with exact proof that
   reviewed content remains unchanged; full-workspace verification still reruns.
   Source/external changes and explicitly requested metadata invalidate that review.
5. Unified native and nested-program tool observation. The runner owns cleanup,
   rollback, reconciliation, durable terminal receipt and terminal publication.
   Frontend failures cancel and drain that owner. TurnFailure carries the terminal
   outcome, original error, cleanup diagnostics and pending settlement disposition.
   Failed turns retain queued user instructions for an explicit subsequent turn.
   A final receipt check includes suggestion, extension and trusted repository
   hook effects. Intermediate goal records withhold new automatic completion;
   the final outcome atomically carries the settled goal and recovery credit.
6. Isolated SQLite and session persistence from async runtime threads. Accepted
   operations outlive cancelled waiters; revisions, receipts, FULL synchronization
   and atomic transition/binding/event updates remain intact. Canonical job state
   replaces BridgeState and duplicate terminal storage. Cancellation signals jobs
   before publication locks, then drains concurrently against one deadline.
   Entry signals turn-owned jobs before cooperative cancellation grace or any
   plan/session write. The compatibility cleanup API shares the remaining deadline.
   Ambiguous publication retains recovery fences; local audit-degraded and PipeFS
   fail-closed policies remain distinct.
   Memory and skill transactions also reuse the retained metadata owner,
   including host-local memory, so a blocked file lock cannot occupy the runtime
   thread. Their accepted writes remain covered by the cancellation barrier.
   File transactions no longer prune the shared journal namespace beneath another
   workspace preparing a write; per-workspace transaction cleanup remains intact.
7. Replaced separate local/remote restore loops with translators into the reducer.
   Interrupted workspace execution repairs once. Snapshot version 3 migrates
   version 2 with integrity verification and rejects unsupported future versions.
   Historical replay streams events without per-event full-state cloning/hashing.
   Live snapshot/patch transport remains for client synchronization.
   Synthetic drive detection governs recovery resets independently of prompt
   whitespace or available checklist context. Validation command scopes preserve
   quoted and interior whitespace so distinct checks cannot clear one another.

The resolved lockfile drops from 1,053 to 983 package/version entries: 70 removed,
zero added. The removed set includes all three engine crates and all Wasmtime and
Cranelift entries. This measures dependency removal, not a build-time speedup.

## Baseline evidence

The baseline archive at `/tmp/hi-harness-baseline-4d2867f` contains exactly the
starting commit. Baseline test binaries were built from that unchanged archive.
Matched build measurements use separate initially empty target directories;
production validation binaries are explicitly rebuilt from the current tree.
`cargo-nextest` is unavailable here; validation uses Cargo and compiled test
binaries.

Captured baseline gates: provider idle guard 4, verification convergence 4,
journal 17, background lifecycle 27, workspace 46, hooks 18, session reducer 12
and session projection six tests passed. Final baseline capture also passed all
642 TUI tests and 86 CLI session tests from the unchanged archive.
The baseline compilation used cached dependencies; its elapsed build time is not
a clean-build comparison.

## Validation evidence

All 18 test executables for the 12 Core CI packages passed: **4,355 tests**, zero
failures, six intentional Agent ignores. The latest full Agent run passed 1,681
tests in 45.05 seconds. Executables ran sequentially, with four test threads and
isolated XDG state, without competing builds. Native process, Git and loopback
fixtures ran with the required local access. Commands, summaries and logs are in
`/tmp/hi-final-core-tests/results.json`.

| Suite | Passed |
| --- | ---: |
| Agent | 1,681 |
| Provider / hi-ai | 408 |
| Tools | 586 |
| TUI | 640 |
| CLI unit / fake-provider end-to-end | 577 / 4 |
| Control / Workspace | 47 / 49 |
| PipeFS / native Git roundtrip | 97 / 1 |
| Evaluation, including hermetic judge | 112 |
| LSP / Shell / smoke runner | 30 / 16 / 107 |

Build gates passed: `cargo check --workspace --all-targets --locked`, Core CI
Clippy (`--all-targets -- -A clippy::large-enum-variant -D warnings`), production
`hi` and `hi-smoke`, and combined optional voice/codebase-graph features. The
Clippy exception matches the existing CI workflow. Exact commands and logs are
recorded in `/tmp/hi-final-build-results.json`. Format, diff whitespace and the
source-size ratchet pass without raising existing source ceilings.

The combined blocked-SQLite regression acknowledged cancellation of three real
jobs in **1.344 ms**, under its 200 ms assertion and the one-second release gate.
It retained exactly three pending identities until releasing the write lock
produced durable Cancelled states. A separate current-thread runner regression
signals three jobs within one second while a synthetic plan's session write is
blocked. Compatibility cleanup proves it uses the remaining deadline, retaining
accepted work after its waiter times out. These are local fixtures, not production
latency percentiles.

The remaining provider-free release gates pass: 94 benchmark schemas, 21 harness
tapes, 16 quality tapes, valid trace acceptance and tampered trace rejection.
Core doctests completed with zero runnable examples and one existing ignore.
Exact commands and logs are in `/tmp/hi-final-script-results.json`.

Review tests retain their original success assertions and additionally reject
source/external changes, requested metadata changes, and mutation during a fresh
verification pass. The journal-directory regression coordinates two workspaces
to prove one transaction cannot remove a shared parent beneath another writer.
All four fake-provider end-to-end cases pass, including typed non-success after
unproductive recovery within the physical-request allowance.

All 38 TUI scenarios parse. All 23 curated scenarios passed: 22 in the full run,
then the persistent-empty case after correcting its expected drive-stop metadata
to `no_progress`. Its four-request, failure-outcome and idle assertions remained
unchanged. All 12 fixed chaos seeds passed. Commands and artifacts are recorded
in `/tmp/hi-final-ui-results.json`. Four older smoke scenarios now exercise finite
recovery, explicit resume and genuinely productive continuation.
The default graph contains no removed engines, Wasmtime or Cranelift. The CI
orchestration microbenchmark passed at 12 / 11 / 14 / 20 ms for 1 / 2 / 4 / 8
candidates with its configured 100 ms baseline thresholds
(`/tmp/hi-orchestration-final.log`).

Validation ran on this macOS host. Linux-only sandbox-pivot behavior remains a
platform-specific CI gate. Uninterruptible native/file-lock callbacks can outlive
the caller's 60-second deadline; their accepted work retains its owner and the
caller receives a pending failure, with session reuse fenced until authoritative
recovery. No live paid-provider evaluation was run.

## Matched local measurements

Both source trees used this host (Darwin 25.6.0, arm64), Rust/Cargo 1.96.0, the
same dev profile and four build jobs. Each clean build had its own initially
empty target directory; compiler wrappers were disabled and dependency sources
were already cached. The command was identical:

```sh
cargo build -p hi --no-default-features --locked --offline
```

The no-op measurement repeated that command. The incremental measurement touched
only `crates/hi-cli/src/main.rs`, then restored its timestamp without changing
source bytes. There was no competing build/test workload. Each build mode was
measured once; OS cache and thermal state were not controlled.

| Build | Baseline `4d2867f` | Current |
| --- | ---: | ---: |
| Clean | 111.961 s | 97.343 s |
| No-op | 1.070 s | 0.748 s |
| Touch-only incremental | 3.108 s | 3.102 s |

This local clean build was about 13% faster; incremental rebuild time was
essentially unchanged. Raw commands/timings and environment records are in
`/tmp/hi-harness-build-measurement/`.

The separately built binaries also ran identical loopback HTTP fixtures, three
times each, with fresh workspace/HOME/XDG directories. Both used
`--no-save --no-memory --no-auto-compact --no-finalize --report <path>` and
`HI_SUGGEST_NEXT_PROMPT=0`. The prompt was `say hello`. The server either returned
`Hello!`, returned empty SSE completions,
or held response headers until SIGINT was sent after observing the first request.
Every run produced its expected typed outcome; none hit the external 75-second
measurement safety limit.

| Fixture metric | Baseline | Current |
| --- | ---: | ---: |
| Accepted answer: physical requests, every run | 1 | 1 |
| Accepted answer: median last response to process exit | 514.763 ms | 264.140 ms |
| Empty completion: physical requests, every run | 6 | 4 |
| Empty completion: median start to failed process exit | 1,502.777 ms | 846.389 ms |
| Empty completion: median last response to process exit | 516.165 ms | 371.687 ms |
| Hung headers: median SIGINT to cancelled process exit | 1,076.459 ms | 986.831 ms |

The response-to-exit measurements include process teardown and are upper-bound
proxies for settlement latency, not isolated terminal-publication timings.
SIGINT-to-exit likewise differs from the job-signal latency measured above.
First-run startup varied substantially, so these three-run medians are local
comparisons, not production percentiles. Reports, raw samples, binary SHA-256
identities and medians are retained in `/tmp/hi-harness-fixture-measurement/`.

Evaluation identities deliberately changed: baseline `native_director=2` became
`native_turn_policy=1`, and `session_reducer=2` became `session_reducer=3`.
Binary, source-state and changed regression-fixture digests also differ; results
must not be pooled as if they used one evaluation identity. The HTTP fixture
inputs and commands above were identical across the two binaries.

## Session replay scaling gate

Final replay-source run on 2026-09-06 with the unoptimized test binary (debug information enabled), during an isolated runtime window with no builds or other test processes:

```sh
cargo test -p hi-agent --lib session_replay_scaling -- --ignored --nocapture
```

The final gate used the already compiled binary directly: `target/debug/deps/hi_agent-051e5502ea035aae session_replay_scaling --ignored --nocapture --test-threads=1`. Output is retained in `/tmp/hi-final-g-replay-scaling.log`.

The manual test constructs identical families of 2,000 / 4,000 / 8,000 history
items before timing. Each item has four serialized events: a 256-character user
message, transcript block open, a 256-character append, and block settlement.
Each measured pass decodes each record through the legacy/canonical translator,
applies the shared reducer in place, and consumes its restored state. Seven
passes run for each size; the median excludes fixture construction and state
destruction. Tests check final message/block counts and use a thread-local test
counter at the projection digest function to assert zero full-state hashes
during every replay pass. Snapshot hashing remains an explicit transport/cache
boundary.

| History items | Events | Median elapsed | Relative to 1x | Full-state hashes |
| ---: | ---: | ---: | ---: | ---: |
| 2,000 | 8,000 | 42.848 ms | 1.00x | 0 |
| 4,000 | 16,000 | 86.796 ms | 2.03x | 0 |
| 8,000 | 32,000 | 174.445 ms | 4.07x | 0 |

These local timings demonstrate near-linear scaling for this fixture family;
they are a regression gate, not a production throughput claim. Separate
fixtures cover interrupted execution at every snapshot/tail boundary, duplicate
settlement replay, compaction/rewind recovery preservation, version-two snapshot
migration, unsupported versions, and a 20,000-block transcript with 2,000 later
deltas. The final focused replay, wire-format and persistence gate passed 65
tests with the manual scaling test ignored; CLI session/restore parity passed
47 tests. Local-HTTP fixtures run with loopback socket access.

## Session persistence and shared tool observations

Runtime session writes now run through one retained, bounded owner (128 accepted
operations including the active operation). Async callers wait for the actual
store result. Cancelling admission does not enqueue work; cancelling an accepted
write drops only its waiter. A barrier waits for earlier work and reports failures
whose waiter disappeared. Explicit synchronous startup/public setters retain
commit-before-return compatibility. Active recovery, plan, compaction and
settlement paths use the async ownership seam. Entry fences a live Agent whose
session publication remains indeterminate; restoring authoritative state is
required before another turn can proceed.

Coverage includes a real locked JSONL file on a single-thread Tokio runtime:
cancellation stays
responsive, the dropped turn does not discard its accepted recovery write, and
the barrier observes its later commit. Separate owner tests cover bounded queue
admission, cancelled unaccepted writes, late failures and recovery after an
observed write error. The typed fast-check deadline fixture also passed outside
the outer tool sandbox, which otherwise prevents nested `sandbox-exec` startup.

All physically completed native tool branches and real nested program calls now
share one result postprocessor. It updates evidence, implementation state,
progress and telemetry once per execution. Tool checks, proactive file checks,
affected-package checks and final verification feed canonical task recovery.
Only stable-input process results can establish validation; infrastructure,
cancellation and changed-input results remain unverified. Empty program
envelopes provide no progress credit. The old ProgressTracker validation failure
map, mutation epoch, selector/coverage parser and diagnosis decision path were
removed. The behavioral regression for unchanged model-authored failure across
edits permits three corrective requests, then returns Failed/NoProgress after
four actual failed validations (eight model requests total).

## Follow-up: moderation ownership failure, 2026-09-06

A live Rust task stopped on E0382 after its final edit changed `match cmd` to
`match &cmd`. The enclosing alternative patterns still moved the command's
String fields. A direct compiler reproduction confirmed the reported failure;
borrowing those fields in the outer patterns repaired it. The affected project
then passed `cargo check --locked --offline` and all 18 existing tests.

The session contained two edit requests whose old and new text were identical,
plus one blocked repetition. No-op edits must return actionable failure without
preparing a file mutation or claiming that an edit happened.

Related harness regression coverage now requires recovery exhaustion to stop
model dispatch while allowing an eligible deterministic final check of retained
edits. That check cannot raise an explicit verification ceiling or enter review,
maintenance, or another repair cycle. Its result remains bound to the actual
workspace revision; a passing check alone does not confirm task completion.
Closeout distinguishes current failures from older evidence and renders stored
diagnostic identities without their internal NUL separator or test-detail hash.

Native default-feature Cargo checks use a shared manifest scope, so an equivalent
final check can discharge an earlier fast-check failure. Arbitrary commands,
feature flags, package selectors and workspace-wide checks retain separate
identities. Historical scopes remain readable.

These policy fixes advance `native_turn_policy` from 1 to 2. The measurements
above describe version 1; they have not been rerun as performance claims for
version 2. The three-intervention recovery default remains unchanged.

The preceding feature-construction episode also spent two interventions while
enum, parser, and handler changes were still being assembled. That is a remaining
tradeoff of charging intermediate validation failures against this shared budget;
these fixes do not add a separate construction allowance or credit arbitrary
source changes as objective progress.

The follow-up regression run passed 3,503 tests: Agent 1,694 (six existing
manual tests ignored), Tools 588, TUI 640, and CLI 581 including its integration
tests. The actual Cargo fixture first creates its lockfile before capturing the
input revision and allows exactly one final check. This preserves the production
rule that changes during validation make its evidence inapplicable. Commands,
per-suite output and build results are retained in
`/tmp/hi-chat-incident-verification/`.

All-target Clippy for the affected crates passed with the existing CI allowance
for large enum variants, as did formatting, the source-size ratchet and diff
checks. Both `hi` and `hi-smoke` were rebuilt in release mode. The release binary
passed all 23 scripted PR terminal scenarios: 22 in the complete run, then the
remaining scenario after updating only its expected closeout text. That scenario
still requires a typed non-success outcome, exactly two provider requests, an
idle next drive, and no accepted generic completion. Already-running hi processes
retain their loaded executable and require a restart to use these fixes.


## Follow-up: provider failure after passing tests, 2026-09-06

The next live episode completed five tool batches, including a filtered
`cargo test` whose output showed 18 passing tests, then failed with local physical
request exhaustion. Its plan still had unfinished implementation work. The
saved session did not retain the failed physical attempts, so it cannot establish
which underlying responses consumed that operation's allowance. Accepted tool
responses did start fresh operations; there was no evidence of one budget being
incorrectly accumulated across those successful tool rounds.

Two independent defects obscured the outcome: the terminal provider error
bypassed native verification, and local budget exhaustion was displayed as an
upstream rejection with a duplicated error chain. The filtered command could
not establish a reliable Cargo exit status because `sh` does not enable pipefail.

Terminal provider errors after tool work now enter the existing final verification
and settlement path, retaining the original cause before fallible publication.
A private settlement receipt prevents Entry from cleaning up that body twice.
Failure during later settlement retains the provider cause and secondary
diagnostics; failed final reconciliation revokes a prior pass. Checks of an
unchanged current revision use explicit admission, real configured/discovered
stages, and existing writer, revision, and verification-limit gates. No evidence
is inferred from a pipeline's final filter status, and passing checks cannot
complete an unfinished plan.

Request-limit errors carry bounded typed evidence: the limit reason, operation
identity, send count, and up to eight recent dispatch ordinals, route digests,
HTTP statuses, and fixed failure categories. They do not retain payloads,
credentials, endpoint URLs, or arbitrary API text. Their safe summary survives
ordinary persisted failure messages. Client-side rejection of an otherwise
decoded required-tool response now preserves its usage without accepting that
completion's context occupancy. Already-completed plans with concrete tool
results close on the first empty recap instead of sending recap-repair requests.

This changes `native_turn_policy` from 2 to 3. The physical-send and semantic
recovery defaults remain four and three. Earlier performance measurements retain
their original policy identities and are not new performance claims.

The new terminal regression was first run against the previous release binary.
With default limits, a real Cargo fixture passed its filtered test command, then
a fake provider's exhausted outage retries produced Failed/InfrastructureFailure
with Unverified evidence. The scenario requires current-revision Passed evidence
alongside the same non-success task disposition. Separate real-HTTP Rust tests
isolate four-send exhaustion from the semantic ceiling and require one tool
effect, one final check, one terminal receipt, and no extra provider dispatch.
Artifacts for this incident are under `/tmp/hi-post-test-incident-verification/`.


Final validation passed 3,924 tests: AI 409, Tools 588, Agent 1,706 (six existing
manual tests ignored), TUI 640, and CLI 581 including integration tests. Agent
regressions verify that terminal observations continue to record fresh evidence
without replacing the original stop cause or scheduling another correction.
Error assertions inspect the full source chain while the wrapper renders once;
the paused-drive assertion reflects the persisted recovery stop and still
requires infrastructure failure, retained edits and an unfinished plan.

All-target Clippy passed for the affected crates with the existing large-enum
allowance. Formatting, the source-size ratchet and diff checks passed. Rebuilt
release binaries passed all 24 scripted PR terminal scenarios, including the
new unchanged-workspace Cargo case that failed against the previous executable.
Its assertions require Passed verification, Failed/InfrastructureFailure for
the unfinished task, retained plan leftovers and idle automatic drive. Passing
smoke cases retain their summaries under the existing artifact policy. The
release identity is recorded in `release-identity.json` in the artifact directory;
already-running hi processes must restart to load the rebuilt executable.
