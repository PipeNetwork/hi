# PipeFS workspaces

PipeFS is an opt-in workspace mode for continuing the same HI session on another machine without
making the launch directory the durable source of truth. HI restores the session's latest
workspace revision from IPOP, runs existing file tools and native Git against an ordinary private
local directory, and acknowledges a mutating operation only after its resulting revision is
durable.

PipeFS is disabled by default. It requires an IPOP deployment with PipeFS enabled, authenticated
session sync, and the current session writer lease.

## Platform support

PipeFS archive version 1 is supported on Linux and macOS only. Windows is unsupported: do not
enable `[pipefs] enabled` or pass `--pipefs` on a Windows client. The v1 format accepts Unix
filenames (while rejecting traversal, control characters, and names that collide under common
macOS case/Unicode normalization), and relies on Unix symlink, mode, and atomic-rename semantics.

## Enabling it

For a new headless or one-shot session, pass `--pipefs`. To make new sessions request it by
default, add this to `hi.toml` or the user configuration file:

```toml
[pipefs]
enabled = true
```

Interactive sessions expose these commands:

```text
/pipefs on
/pipefs off
/pipefs status
/pipefs retry
/pipefs recover list
/pipefs recover inspect <recovery-id>
/pipefs recover retry <recovery-id>
/pipefs recover export <recovery-id> <destination.tar.zst>
/pipefs recover discard <recovery-id> --confirm <digest-from-inspect>
```

If PipeFS activation itself fails, the same local recovery data remains
reachable before session and provider initialization:

```text
hi workspace status --session <session-id> [--json]
hi workspace recover list --session <session-id> [--json]
hi workspace recover inspect <recovery-id> --session <session-id> [--json]
hi workspace recover retry <recovery-id> --session <session-id> [--json]
hi workspace recover export <recovery-id> --session <session-id> --to <new-archive.tar.zst>
hi workspace recover discard <recovery-id> --session <session-id> --confirm <digest-from-inspect>
hi workspace takeover --session <session-id> [--json]
hi workspace detach --session <session-id> --if-clean
hi workspace import --session <session-id> --from <path> --preview [--json]
hi workspace import --session <session-id> --from <path> --confirm <preview-digest>
hi workspace export --session <session-id> --to <new-path> [--revision HEAD]
```

Omit `--session` to inspect recovery evidence in the current local workspace,
without sync credentials or provider startup:

```text
hi workspace status [--json]
hi workspace recover list [--json]
hi workspace recover inspect <recovery-id> [--json]
hi workspace recover discard <recovery-id> --confirm <fresh-workspace-digest> --accept-current-bytes
```

Local retry is intentionally unavailable after restart because the new process
cannot prove that an old native process was reaped. Local discard means “stop
all external writers and accept the complete current byte image”; it preserves
those bytes, includes VCS metadata such as `.git` in the confirmation digest,
and closes the interrupted lifecycle as failed rather than inferring success,
cancellation, or rollback. Harness-owned runtime state is the only documented
scan exclusion.

Status, recovery list/inspect/export, and import preview are read-only and do
not acquire a writer lease. Recovery retry, takeover, clean detach, confirmed
import, and remote export contact IPOP; every remote mutation requires a
generation-fenced writer lease. Recovery listings use stable, deterministic
journal recovery IDs. Legacy cache IDs remain accepted as compatibility aliases.
Inspecting either form shows the owning cache and its whole-cache confirmation
digest: export and confirmation-discard operate on that complete retained cache,
not only on one logical journal row. Cache ownership is derived from the complete
`HI_SYNC_BASE_URL`/`HI_SYNC_API_KEY` pair, or from a trusted `[sync]` section.
Export always creates a new deterministic archive and retains the source cache;
discard is permanent and requires the content-bound digest shown by `inspect`.
Import is permitted only for an empty remote head, rescans the source after
confirmation, and never treats the launch directory as implicitly authoritative.
User-owned `[sync]` credentials may also use `api_key_ref = "env://NAME"` or a
private `auth-store://...` reference. Legacy literal/environment fields remain
readable and are migrated best-effort when a user-owned config is loaded;
repository config is never rewritten as a side effect of inspection.

The effective startup setting is chosen in this order:

1. The existing remote session setting when attaching or resuming.
2. An explicit `--pipefs` request for a new session.
3. The `[pipefs] enabled` user setting.
4. Disabled.

`/pipefs on` works from any saved interactive session, even when transcript sync was off at
startup. It first verifies server capability and authentication, enables transcript sync for the
same session identity, acquires its existing writer lease, restores an existing remote head when
present, and only then rebinds the agent. A session with no head starts in a clean empty PipeFS
directory. The original launch directory is not copied or uploaded.

`/pipefs off` checkpoints the workspace and transcript, disables the remote mode, returns the
agent to its original directory, and removes the acknowledged cache. Background jobs must be
finished first. If persistence or cleanup fails, HI retains the cache and reports an actionable
`/pipefs retry` status instead of silently returning to local operation.

## Durability and recovery

File tools report known changed paths. Shell commands, native processes, and background jobs cause
a full workspace reconciliation. During a failed or pending commit, reads remain available but
new mutations, graceful exit, and `/pipefs off` are blocked until `/pipefs retry` succeeds.

If a process dies after the server has committed an operation and the local
pending archive has already been removed, recovery remains fail-closed and
inspect/export/discard-actionable. Automatic proof by operation ID additionally
requires the causal server receipt-lookup capability; it is not guessed from a
matching workspace head.

Servers advertising `causal_commit_v1` and writer protocol 2 can acknowledge
workspace head, operation receipt, transcript records, and causal cursor in one
idempotent transaction. Older servers retain the compatibility sequence:
workspace CAS followed by deterministic transcript flush. A failed transcript
flush is reported as pending and blocks further mutations; it is never described
as an atomic publication. Non-replayable external effects require a separately
acknowledged remote intent before execution under protocol 2.

Lease status is `Valid`, `Uncertain`, or `Lost`. Either uncertain or lost status
immediately closes mutation admission, stops and reaps affected writers, and
retains pending archives and recovery markers. An uncertainty notification is
latched even if a later background heartbeat refreshes the stored expiry; only
a synchronous admission or recovery proof can return the controller to `Ready`.
This prevents a coalesced watch update from hiding the writer-stopping boundary.

Background write agents are detached candidates and require both
`candidate_jobs_v2` and writer protocol 2. Protocol 1 may run a synchronous
detached candidate only through the explicitly enabled compatibility writer; it
never admits a background write candidate. Live background writer processes are
disabled until the client can pause the complete process group, take a stable
checkpoint, and resume under the same lease fence. `--keep-background` remains
unsupported in PipeFS.

The workspace archive contains regular files, directories, safe relative symlinks, modes,
timestamps, and `.git`. Runtime state and transaction staging live outside it. Consequently normal
`git clone`, `status`, `checkout`, `add`, and `commit` continue to use the installed native Git, and
repository metadata resumes with the working tree.

Taking over a session on another machine invalidates the former writer. The new writer downloads
and verifies the bounded full-plus-delta restore chain before the agent can activate. Authentication,
network, missing-revision, corruption, or path-portability failures stop activation; HI never
continues in the launch directory as a fallback.

An interrupted process leaves a private recovery marker with its dirty cache. A later process on
the same machine can retry it only when every unresolved journal fence in the owning cache has
exact proof for the same operation, binding epoch, idempotency key, and replay class. One matching
archive cannot cover an unrelated unsettled operation or job. An unmatched recovery remains
available for inspection, whole-cache export, or confirmation-discard, but retry is rejected before
lease acquisition. The recorded base must also still be the remote head. Conflicting remote work is
never overwritten or automatically merged.

Local caches and mode hints are namespaced by the normalized IPOP origin and a non-secret
fingerprint of the authenticated credential, in addition to the session and lease generation.
Changing accounts, credentials, or IPOP deployments never exposes or automatically adopts another
authority's recovery bytes. Caches created by older clients before authority scoping remain on disk
for explicit manual salvage, but current clients do not automatically list, restore, or delete them.

## What “diskless” means

PipeFS removes the local disk as the *durable source of truth*. Native tools and Git still require
temporary local bytes while the session is active. Clean caches are removed after remote
acknowledgement. A failed or forcibly terminated process may retain a private recovery cache until
HI can prove that it is represented by the remote head or safely reconcile it.

Folder trust is evaluated independently on every machine. Restored workspace content never carries
a local trust grant, and HI does not automatically execute repository hooks during restoration.
The portable binding initially treats repository guides as data-only context.
`/trust on` may promote only those instructions after its idempotent policy
write and transcript settlement succeed on this machine; `/trust off` demotes
them again. Repository hooks and repository-imported MCP remain disabled in
PipeFS even after that explicit prompt-context trust grant.
