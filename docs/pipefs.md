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
/pipefs recover inspect <cache-id>
/pipefs recover export <cache-id> <destination.tar.zst>
/pipefs recover discard <cache-id> --confirm <same-cache-id>
```

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

The workspace archive contains regular files, directories, safe relative symlinks, modes,
timestamps, and `.git`. Runtime state and transaction staging live outside it. Consequently normal
`git clone`, `status`, `checkout`, `add`, and `commit` continue to use the installed native Git, and
repository metadata resumes with the working tree.

Taking over a session on another machine invalidates the former writer. The new writer downloads
and verifies the bounded full-plus-delta restore chain before the agent can activate. Authentication,
network, missing-revision, corruption, or path-portability failures stop activation; HI never
continues in the launch directory as a fallback.

An interrupted process leaves a private recovery marker with its dirty cache. A later process on
the same machine can retry it only when its recorded base is still the remote head. Conflicting
remote work is never overwritten or automatically merged.

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
