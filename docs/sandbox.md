# Shell sandbox (`HI_SANDBOX`)

Agent `bash` commands can run under an OS write-confine policy so a misguided
command cannot modify files outside the project. This is **in addition to** the
heuristic dangerous-command guard (`HI_ALLOW_DANGEROUS`) — not a replacement.

## Defaults (deliberate)

| Setting | Default | Why |
|--------|---------|-----|
| `HI_SANDBOX` unset / empty | **workspace** | Agent shells must not write outside the project by default. |
| `HI_SANDBOX=workspace` (or `on` / `1`) | **on** where enforced | Writes limited to the workspace root, system temp, and essential device nodes. Reads and network stay open. |
| `HI_SANDBOX=off` | no OS sandbox | Opt out when toolchains must write caches under `$HOME` (Cargo, npm, pip). |
| `HI_SANDBOX=<typo>` | **startup error** | Unknown values are rejected so a typo cannot silently disable confinement. |

**Recommendation:** keep the default for untrusted prompts and multi-tenant hosts.
Set `HI_SANDBOX=off` only when global package/tool caches under `$HOME` must stay
writable for day-to-day local development.

```bash
# Default: confined shell writes (macOS Seatbelt today):
hi "refactor the parser"

# Explicit off when home-dir caches must stay writable:
HI_SANDBOX=off hi "..."
```

## Platform support

| Platform | Enforcement | Mechanism |
|----------|-------------|-----------|
| **macOS** | Yes when policy is `workspace` | `sandbox-exec` Seatbelt profile (deny `file-write*`, re-allow workspace + temp + devices) |
| **Linux** | Yes when pipe-wrap is available | Rootless `pipe-wrap` namespaces/bind mounts; degraded warning when unavailable |
| **Windows** | Not enforced | No profile |

Check at runtime with `ProcessRunner::sandbox_enforced()`,
`sandbox_backend_name()`, and `sandbox_backend_status()`. When Linux cannot
find a capability-compatible `pipe-wrap`, `ProcessRunner::new` prints a
one-shot **warning** and preserves the existing best-effort execution behavior.
Set `HI_PIPE_WRAP=/absolute/path/to/pipe-wrap` to select a packaged artifact.

Code: `crates/hi-tools/src/sandbox.rs`, wired through `ProcessRunner`.

## What the workspace policy allows

- **Writes:** workspace root (canonicalized), `/tmp` and per-user temp roots,
  `/dev/null|stdout|stderr|tty|…`
- **Reads:** unrestricted (headers, toolchains, system libs)
- **Network:** unrestricted at the OS-sandbox layer (SSRF controls for
  `web_*` tools are separate in `web.rs`)
- **Exec:** unrestricted (the dangerous-command denylist still applies first)

## Escape hatches

| Env | Effect |
|-----|--------|
| `HI_SANDBOX=off` | No OS sandbox (default is workspace) |
| `HI_ALLOW_DANGEROUS=1` | Disables the **heuristic** denylist only — does not disable OS sandbox |
| `HI_ALLOW_PRIVATE_WEB=1` | Relaxes SSRF private-IP blocks for `web_*` tools |

## Linux build and policy mapping

The pinned source revision and reproducible build helper are:

```text
tools/pipe-wrap.lock
tools/build-pipe-wrap.sh
```

The helper builds the standalone `pipe-wrap` binary for x86_64 or aarch64;
hi intentionally does not vendor it as a Rust library. `workspace` uses
read-only host roots plus writable workspace/state/temp binds and shared
network. `strict` uses a smaller executable/readable root set and an isolated
network namespace. `readonly` keeps the filesystem read-only and isolates the
network. All modes retain `--die-with-parent`, PID/IPC/UTS isolation, sanitized
environment handling, bounded output, and process-group cancellation.

Pipe-wrap requires a Linux user-namespace-capable kernel. AppArmor/SELinux and
distribution policy can still prevent rootless namespace creation; those
conditions are surfaced as `Unavailable`/`Degraded` rather than reported as
enforced.

## Related

- Dangerous-command guard: `crates/hi-tools/src/guard.rs`
- RSI candidate host (stricter allowlisted shell): `crates/hi-tool-host`
- Architecture trust domains: [architecture.md](architecture.md)
