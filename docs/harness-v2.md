# Harness v2

Harness v2 coordinates tools, processes, agents, workspace durability, and the
session transcript around one publication rule:

> A result is not successful until its workspace effects and transcript record
> have reached a known settlement state.

The shared contracts live in `hi-workspace`. Local and PipeFS sessions use the
same admission and settlement lifecycle; only the durability authority differs.

## Local and PipeFS operating modes

| Concern | Local workspace | PipeFS workspace |
|---|---|---|
| Authority | Local controller journal plus complete workspace version | Remote head, manifest, lease generation, and transcript cursor |
| Foreground mutation | Reconcile local bytes, journal the result, then return | Reconcile bytes and publish through the negotiated compatibility or causal protocol |
| Write agent | Detached candidate; the parent applies verified bytes | Detached candidate; the child never receives the lease and the parent alone publishes |
| Background live writer | Local process jobs are tracked but do not serialize unrelated foreground work or other local process jobs | Disabled until pause/checkpoint/resume fencing exists |
| Ambiguous settlement | Preserve local recovery evidence; foreground work can continue only in the explicit audit-degraded case | Preserve the archive/marker and fail closed until remote proof |
| Hooks and repository MCP | Available only after folder trust | Disabled for the whole PipeFS binding |
| Repository guides | Authority-bearing only after folder trust; otherwise data-only | Start data-only; a machine-local trust grant may promote only the prompt context |
| Import/export | Ordinary local files remain authoritative | Import is preview/confirm into an empty head; export creates a fresh destination |

Both modes therefore exercise the same state machine and job callbacks. Local
mode is not a mock PipeFS implementation, and PipeFS does not turn every local
filesystem action into a bespoke remote code path.

## Native execution and recovery

One native turn loop owns model, tool, verification and settlement decisions.
The WASM host, guest, engine API and shadow NativeDirector are removed. Evaluation
identities use `native_turn_policy` version 3. Recovery exhaustion forbids further
model requests while retaining one deterministic check of the final workspace.
After tool work, a terminal provider error follows that same verification and
settlement path, preserving its original typed cause. Validation activity can
admit a check of the current revision even when this turn made no edits. An
unfiltered native check supplies the exit status that shell output filters may
hide. Passing verification does not mark an unfinished task complete.

Each inference operation has one shared request context across transport,
authentication, compatibility repair and fallback. The default allowance is
four physical dispatches including the first, with at most 60 seconds of total
backoff. Exact replays retain their request identity; changed routes or payloads
receive another request identity within the same operation. Configured MoA
children and its reserved aggregation dispatch have three shared recovery sends.
Only accepted, usable completions start another inference operation.
Local request-limit failures retain bounded, redacted attempt evidence and
remain distinct from an upstream rejection. A completed plan with concrete tool
evidence closes deterministically when its final recap is empty, without asking
the model to generate another recap.

Provider silence is measured from physical dispatch across response headers and
streaming. Advancing text, reasoning or tool arguments resets the clock;
keepalives and repeated status events do not. The default is five minutes.
Queue and explicit approval waiting are excluded. The TUI renders typed attempt
and progress events without making a second timeout decision.

Typed settings `provider.max_attempts` (default 4),
`provider.no_progress_timeout` (milliseconds, default 300000; 0 disables silence
recovery), and `recovery.max_interventions` (default 3) resolve centrally.
Existing explicit request/turn/tool/step/verification limits remain additional
ceilings. Productive work receives no new overall duration or step limit.

Automatic answer, protocol, inspection, verification and reviewer corrections
share one durable task-recovery allowance. Only a new best objective result can
replenish it. Cosmetic changes, different labels, unrelated passing checks and
regressing to a previously achieved result do not. Canonical observations bind
validation scope and stable diagnostic identities to exact input revisions.
Failure cycles stop recovery; infrastructure failures do not request code repair.
Current unresolved failures are tracked separately from historical best results.
A same-scope pass clears the obligation; changing input leaves it unverified.
Recovery schema 2 retains that current result explicitly. Development schema 1
records migrate after snapshot integrity validation; ambiguous historical
fail/pass ordering requires revalidation while keeping the session readable.
Automatic plan/goal drives and session resume retain exhausted recovery state;
explicit user retries start a fresh episode with diagnostic evidence retained.
New automatic goal completions remain conservative in intermediate saved records.
The final outcome record commits the settled goal and earned recovery credit
together; a crash before that receipt cannot skip the unfinished step on resume.

A successful check carries its immutable input revision and digest. Later
changes to canonical inputs invalidate that evidence, including prose changes;
settlement cannot transfer a pass to new bytes. Optional skill curation and
automatic coding memory run once before settlement. If they change inputs, the
existing deterministic verifier checks the final revision within its existing
ceiling, without another main-model request. Task review precedes those automatic
writes so generated memory does not trigger unrelated task-repair requests.
An applicable task review can survive those writes only when the reviewed files
remain identical and changes are limited to acknowledged metadata outputs outside
the requested task. Explicitly requested metadata and any source/external changes
invalidate review. The full workspace still requires a fresh verification pass;
changes during that recheck also invalidate the retained review.
An invalidated pass cannot become success through the unverified-work override.
The generated goal checklist is a derived view unless explicitly edited or named
as a validation input.

Selecting settlement closes the execution loop and starts its one 60-second
caller deadline. The agent runner owns writer shutdown, rollback policy,
transcript reconciliation, durable settlement and terminal publication. A typed
`TurnFailure` retains its outcome, original error, cleanup diagnostics and pending
settlement disposition. Frontend failure cancels the runner and continues polling
it without rendering. Accepted writes and ambiguous publication retain their
recovery owners beyond the caller deadline. Cancellation before commit follows
rollback policy; cancellation after commit preserves the committed result.
Missing final answers receive a deterministic closeout without another inference.

## Workspace lifecycle

Every session has an always-present `WorkspaceController`. A mutation first
receives a non-cloneable permit bound to the controller, binding ID, epoch,
complete base version, operation ID, and idempotency key. Execution produces a
separate report. Settlement turns that report into a receipt or recovery state.

Only `Ready` admits ordinary mutations. Dropping an admitted permit before
settlement records an abandoned operation and closes admission. Rebinding
increments the epoch, so callbacks and candidates from the old root cannot
publish into the new root.

Rebinds, PipeFS mode switches, ordinary session close, daemon shutdown, and
transport EOF all use the same lifecycle barrier: stop owned writers, await
native process reaping and task settlement, reconcile/checkpoint, then require
an `Exit`, `ModeSwitch`, or `Rebind` receipt from the current controller
identity and epoch. Local `--keep-background` is the deliberate exception: it
releases explicitly backgrounded processes to outlive the client, so it does
not claim an Exit-barrier receipt. PipeFS continues to reject that option.

Local sessions keep foreground work available when the audit journal fails,
but visibly enter `LocalAuditDegraded` and stop admitting resumable writers.
PipeFS fails closed on journal, lease, head, or transcript ambiguity.

A local `Process + LiveWriter` job is an observability and shutdown record, not
a workspace authority fence. If Hi restarts while one is recorded active, the
old lifecycle becomes terminal `Orphaned` and any legacy synthetic
`crashed_writer_job` marker for that same process is resolved without claiming
success, cancellation, rollback, process death, or a workspace-byte decision.
Detached candidates, hooks, unsettled foreground operations, and every PipeFS
writer retain their recovery fences.

## Jobs and compatibility commands

The unified registry represents processes, read agents, write candidates,
hooks, and compaction. Every lifecycle callback carries the controller,
binding, and epoch fence. Terminal states are written once; polling only reads
the projection.

The compatibility commands continue to work:

```text
task / wait_tasks / kill_task
bash_output / bash_kill
```

They map to the common `job status`, `job output`, `job wait`, and `job cancel`
lifecycle. A write candidate or live writer cannot become `Succeeded` directly
from `Running`: it must pass through merge/settlement and receive the required
workspace publication receipt.

Managed defaults are five minutes in queue, fifteen minutes of candidate
execution, two minutes for verification, and sixty seconds before a caller
sees durability pending. Preparation concurrency is four and total active jobs
are capped at sixteen. These are typed settings, not hidden process timeouts.
Ordinary interactive turns and foreground local workflows remain unlimited by
default, but output, concurrency, retries, subprocess ownership, cancellation,
and observability are governed independently.

## Detached write candidates

Write-capable children never receive the authoritative PipeFS lease and never
edit the parent workspace. Git roots use a private `--no-local --no-hardlinks`
clone plus an exact overlay of dirty and untracked state. Non-Git roots use the
internal snapshotter and a private Git repository. Writable files are copied;
source `.git` data is never shared.

Operations that create, read, recover, or garbage-collect one internal snapshot
store are serialized in-process and, on supported Unix hosts, by an OS advisory
lock. Candidate reports, event streams, build caches, staged executables, and
private temporary files live in a parent-created runtime sibling outside the
materialized workspace. Directory checks plus no-follow regular-file opens keep
a source-controlled `.hi/candidate-child` or executable symlink from redirecting
the parent's pre-sandbox writes.

A `VerifiedCandidate` seals its source binding, epoch, complete base version,
before/after digests, file kinds and modes, postimages, deletions, verification
evidence, bounded artifacts, and effective provider/model route with canonical
BLAKE3. Auto-apply rechecks the exact base under serialized merge capacity and
uses transactional postimages plus the exact executable verifier contract from
the child's final successful verification round. The parent shares one finite
wall-clock budget across that pipeline, checks that verification did not mutate
the destination, and publishes success only after a stable green checkpoint.
A known verifier failure rolls back to the sealed preimage; an ambiguous rollback
enters recovery-required. A binding or complete-version mismatch stays stale for
review or rerun.

Candidate children disable repository hooks, signing, imported repository MCP,
repository skills, and elevated repository instructions.

## Journal and session replay

Harness projections extend the existing `events.sqlite3`; no second database is
created. Schema v2 contains workspace bindings, operations, jobs, recoveries,
and session snapshots. Projection changes and their required `run_events`
entry commit in one `BEGIN IMMEDIATE` transaction with foreign keys enabled and
`synchronous=FULL`. Future schema versions are rejected. The schema version is
the last write of a successful migration transaction.

The versioned `SessionReducer` is authoritative for both local JSONL and remote
restoration. Readers translate old records into the same events; interrupted
workspace executions are repaired once by the reducer. Snapshot version 3 adds
persisted task-recovery state and pending execution recovery. Version 2 snapshots
are migrated after validating their original integrity; unsupported future
versions fail closed. Tail replay applies events directly and hashes only at
snapshot boundaries. The live snapshot/patch presentation transport remains a
separate consumer, controlled by `session_projection_v2`.

Asynchronous journal writes use one dedicated writer thread per canonical
`events.sqlite3` path and a bounded queue of 128 requests. The entire
read/validate/update/commit operation runs there. Transition, binding and event
updates share one transaction. An accepted request survives a dropped waiter:
its publication owner retains the identity and recovery fence until the durable
acknowledgment is resolved. Queue rejection is distinct from accepted work with
an outstanding acknowledgment.

Runtime session persistence also uses a retained owner with a bounded queue of
128 operations around the existing session sink. Active turns await commit
receipts without blocking runtime threads. Dropping an accepted waiter does not
discard its write, and a barrier reports late errors. Synchronous compatibility
commands reject admission while asynchronous persistence is outstanding. A turn
whose publication remains ambiguous requires authoritative restoration before
another turn can use that Agent. This introduces no new database or wire format.

Interactive compaction now registers a bounded, read-only `Compaction` job and
prepares every strategy from an owned transcript snapshot. Publication checks a
content-addressed session revision, writes the durable replacement boundary,
and swaps the live transcript without an intervening await. A changed revision
discards the candidate as `Stale`; a persistence failure seals `Failed` while
leaving live messages untouched. The CLI/TUI command remains awaited for
compatibility rather than becoming a free-running background command, but its
work and terminal callback use the unified job lifecycle.

Manifest evaluation identities seal the binary, Git state, fixtures, limits,
provider/model policy, tool-envelope schema, reducer/native turn-policy versions,
workspace backend, materializer, OS/architecture, MCP, and network policy.
Reports separate different identities as `incomparable_records`; they are never
reported as regressions.

## Tool and provider envelope

After dynamic tool selection, the model request receives one canonical BLAKE3
envelope containing the ordered tool set, schemas, effect/replay policy, limits,
provider capabilities and any evidence-backed actual model revision, workspace
authority/version, folder trust, and permissions. Execution rejects calls
absent from that exact envelope or whose schema changed after advertisement.
Hidden `run_program` tools have a separate sealed allowlist and are not directly
callable.

Every production request emitted by the interactive agent—including auxiliary
chat-only requests—carries an envelope. Managed RSI, team, and diff fanout
requests are also sealed for their actual target route, and MoA reseals each
private reference request with the conservative capabilities of that route. A
reusable diff-lab request template is intentionally unsealed while no target is
known; it is never sent directly and each derived request is sealed before I/O.

Each envelope records the attributable schema-token estimate for every tool.
Per-call timeline records carry queue delay, execution latency, typed success or
failure, truncation, and exact workspace effects, so tool demotion decisions can
be based on measured cost and outcomes instead of catalog position.

Provider capabilities cover tool choice, parallel calls, strict-schema
dialect, streamed arguments, request limits, structured output, modalities,
usage, reasoning replay, cancellation, and actual model revision. Fallback
routes use the conservative intersection. The registry supports finite,
audited probes with per-route/model caching; today the CLI seeds it from the
bounded live model metadata already fetched during startup. Frontends may
install another bounded probe, but ordinary agent construction performs no
additional provider I/O. Missing, expired, failed, or incomplete observations
therefore resolve to the conservative capability set.

Shell commands are parsed with tree-sitter. Parse failures, dynamic evaluation,
and opaque redirection are treated as non-replayable live writes. The existing
denylist and privileged brokers remain authoritative.

The `read` tool accepts `workspace://` directly through the workspace path
resolver. Production agent requests refresh `session://current/transcript`,
background lifecycle updates refresh `job://JOB_ID/output`, and sealed candidate
artifacts are resolved lazily from the binding's private state root after their
content-addressed record is verified. Other bounded `artifact://`, `session://`,
and `job://` bodies can be registered by their owning host.
`mcp://SERVER/RESOURCE_URI` delegates to the connected server's MCP
`resources/read` method. All routes use the same line offset/limit, output
budget, oversized-body refusal, and final secret-redaction boundary as
compatibility path reads.

## Typed settings, credentials, and diagnostics

Harness settings resolve through one typed registry in fixed order: built-in,
profile, trusted workspace, session, then one-shot command. Unknown keys,
incorrect types, out-of-range limits, and literal values for secret settings are
rejected. User-owned profile, sync, RSI, and Outcome config is migrated
best-effort from legacy `api_key`/`api_key_env` fields to `api_key_ref`; pasted
literals are sealed in the private credential store and persisted as opaque
`auth-store://` references, while environment-backed values become `env://`
references. Repository config is read without migration or write-back.

Folder trust is enforced in release and self-built binaries alike. The explicit
`HI_FOLDER_TRUST=off` operator override prints a visible warning and attempts to
append an audit record to the same control store; candidate children force the
normal trust gate back on instead of inheriting that override. Local repository
guides are wrapped as data-only context until trust is granted; `/trust on` and
`/trust off` promote or demote the active prompt context without requiring a
restart.

The versioned `ToolDiagnostic` contract includes retryability, operation
identity, sensitive fields, artifact references, local-only export policy, and a
deduplicating store. Resource routing already emits these typed diagnostics.
General tool, provider, and workspace failures still use their existing typed or
text error paths in several places, so migration of every diagnostic producer is
a remaining rollout item rather than a compatibility promise.

## Presentation transport

`hi debug tui --stdio` provides deterministic JSONL input, resize, focus,
component-tree, transcript-block, and versioned session snapshot/patch tests.
The harness already consumes the real TUI event path. Stable durable IDs and
exactly-once terminal settlement are enforced by the versioned reducer for
projected transcript-block events. With `session_projection_v2` enabled, the
production TUI reduces that same live event stream into a shadow projection and
can rebuild from integrity-checked snapshots or exact-base patches without
changing the compatibility `UiEvent` wire format. The legacy widgets remain the
rendering authority while the gate is off. CLI/plain output, remote views, and
durable JSONL/PipeFS emission of the new block lifecycle have not yet been
promoted to projection-patch consumers; those migrations remain independent
rollout steps.

## PipeFS behavior

The parent process is the sole lease holder and publisher. Lease notifications
push directly into the controller. Lost or uncertain leases stop affected
writers, retain pending archives and recovery markers, and close admission.
Once uncertainty is published it remains latched even if a later background
heartbeat renews its stored expiry; only the synchronous admission/recovery
lease proof may return the controller to `Ready`, so a coalesced status update
cannot hide the writer-stopping boundary.

Writer protocol 2 can use `causal_commit_v1` only when the server advertises it
and an atomic publisher is installed. The causal operation endpoint validates
the expected head and lease generation, CAS-updates changed bytes, appends the
operation and transcript, and advances the cursor idempotently by operation ID.
A non-replayable external effect requires a remote intent acknowledgement
before execution.

PipeFS background write candidates require both `candidate_jobs_v2` and writer
protocol 2. Synchronous detached candidates remain available through the
compatibility writer where allowed, but protocol 1 never admits a background
write candidate. Live background writers remain disabled in either protocol.

PipeFS repository guides start wrapped as untrusted, data-only context. A
successful `/trust on` is settled as an idempotent external policy mutation and
promotes only the guide text for the current machine; `/trust off` demotes it
again. Repository hooks and repository-imported MCP stay disabled for the whole
portable binding regardless of that prompt-context trust decision.

On a protocol-1 or compatibility server, workspace CAS is followed by
deterministic transcript records and `flush_through`. A flush failure becomes
`TranscriptPending`; the client does not describe that sequence as atomic.
Protocol-1 clients may read, recover, and export an upgraded session, but may
not mutate it. That compatibility floor must ultimately be enforced by a
server-owned per-session minimum writer protocol; the client's global
capability response alone cannot prove that a particular session was upgraded.

See [PipeFS](pipefs.md) for status, recovery, takeover, clean detach, explicit
import, and fresh-path export commands. PipeFS never silently imports the
launch directory, and `--keep-background` remains unsupported.

## Deterministic fault injection

Set `HI_HARNESS_FAILPOINTS` to a comma-separated list of `point=N` or
`point=always` entries. Occurrences are one-based and process-local:

```sh
HI_HARNESS_FAILPOINTS='candidate_after_apply=1,transcript_before_flush=2' hi ...
```

Supported boundaries cover admission journaling, tool start, effect completion,
archive fsync, commit acknowledgement, transcript flush, candidate apply and
rollback, rebind, lease changes, spawn/exit/cancel, import/export publication,
compaction CAS, and schema-version update. Invalid configuration fails closed.

## Rollout controls

Typed feature settings are independent:

```text
features.workspace_controller_v2
features.candidate_jobs_v2
features.pipefs_causal_commit_v1
features.session_projection_v2
```

Rollback may stop new v2 admission but must leave status, inspect, retry,
recovery export, and PipeFS export operational. Databases are not down-migrated
and recovery caches are not deleted by feature rollback.

`workspace_controller_v2` is enabled by default. Session restoration always
uses the reducer. Historical `features.session_reducer_v2` records remain
readable and no longer select a runtime. Historical NativeDirector settings
remain readable; actively enabling that removed engine returns a migration
error. Detached candidate jobs, causal PipeFS publication and presentation
promotion retain their independent gates. The client protocol, request validation, recovery
paths, and fake server tests for `causal_commit_v1` live in this repository;
the corresponding service transaction must be deployed and advertised by the
PipeFS server before the client gate can take effect. PipeFS live background
writers remain disabled; gated write agents are detached candidates and the
parent remains their sole publisher.

One recovery boundary still depends on that server rollout. If a client
process dies after an atomic remote commit is acknowledged locally but after
its pending archive has already been removed, the current server API cannot
yet look the receipt up by operation ID. The local journal remains fenced and
the evidence stays inspect/export/discard-actionable; automatic reconciliation
requires either the retained receipt or the deployed server-side proof lookup.

## Acceptance coverage

Pull-request CI is provider-free and runs the full `hi-workspace`, `hi-control`,
and fake-`hi-pipefs` suites, including normalized local/PipeFS lifecycle traces,
schema-v1-to-v2 migration, future-schema refusal, failed-migration rollback,
lease loss, causal replay, recovery retention, and import/export safety. The
Linux job also exercises process cancellation/reaping and candidate lifecycle
chaos; the scheduled macOS job runs the native process and PipeFS archive suites.
The source-size ratchet covers all new harness crates.

Live quality measurements stay separate from correctness checks. The scheduled
harness evaluation builds the immediately preceding revision and the candidate
revision, runs the paired two-binary A/B evaluator, and retains both artifact
sets. Evaluation identity differences are reported as incomparable rather than
as regressions.

The repository supplies deterministic failpoints at every named crash window
listed above and focused invariant tests across the relevant controllers and
jobs. CI does not claim that an external PipeFS service transaction has been
deployed: service canaries, per-session protocol-floor enforcement, and receipt
lookup must be added on the server side before those guarantees can be promoted.
