# Shell sandbox (`HI_SANDBOX`)

Agent `bash` commands can run under an OS write-confine policy so a misguided
command cannot modify files outside the project. This is **in addition to** the
heuristic dangerous-command guard (`HI_ALLOW_DANGEROUS`) — not a replacement.

## Defaults (deliberate)

| Setting | Default | Why |
|--------|---------|-----|
| `HI_SANDBOX` unset / empty | **workspace** | Agent shells must not write outside the project by default. |
| `HI_SANDBOX=workspace` (or `on` / `1`) | **on** where enforced | Writes limited to the workspace root, system temp, and essential device nodes. Shared home-directory caches stay read-only. Reads and network stay open. |
| `HI_SANDBOX=off` | no OS sandbox | Opt out when a tool must populate shared caches or perform other home-directory writes. |
| `HI_SANDBOX=<typo>` | **startup error** | Unknown values are rejected so a typo cannot silently disable confinement. |

**Recommendation:** keep the default for untrusted prompts and multi-tenant hosts.
Existing global caches remain readable. Point a tool's cache into the workspace,
or set `HI_SANDBOX=off`, when it must download or update shared cache content.

```bash
# Default: confined shell writes; shared toolchain caches are read-only:
hi "refactor the parser"

# Explicit off when shared caches or other home-directory paths need writes:
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
For safety, hi never discovers or probes `pipe-wrap` through `PATH`; it accepts
only that explicit absolute path or a packaged sibling outside sandbox-writable
workspace/temp roots.

Code: `crates/hi-tools/src/sandbox.rs`, wired through `ProcessRunner`.

## What the workspace policy allows

- **Writes:** workspace root (canonicalized), `/tmp` and per-user temp roots,
  `/dev/null|stdout|stderr|tty|…`
- **Shared caches and hi control state:** readable but not writable.
  Cargo/Rustup, npm, pip/uv, Go module/build caches, `$XDG_CONFIG_HOME/hi`,
  `$XDG_DATA_HOME/hi`, `$XDG_STATE_HOME/hi`, legacy `~/.hi`, and explicit
  `HI_TRUST_STORE` / `HI_ME_MD` paths receive late read-only Linux mounts and
  macOS literal+subpath denies. This prevents a command from forging trust or
  credentials and from poisoning executable source, wheels, shims, toolchains,
  workflow/session state, or build artifacts for a later process. Linux uses a
  fresh private tmpfs for `/tmp` and then restores explicit project binds, so
  absent protected paths below host temp can only be created ephemerally.
- **Over-broad roots:** filesystem root, `$HOME`, another writable ancestor
  containing protected state, or a workspace nested *inside* protected state is
  removed from writable roots entirely. Choose a normal project directory;
  this fails closed even when a protected path does not exist yet.
- **hi state:** the shared state root (`$XDG_STATE_HOME/hi` or
  `~/.local/state/hi`) is never added as a writable exception. Nested hi
  commands that need state writes must point `XDG_STATE_HOME` into the project.
  Project-local Cargo/npm/pip/uv/Go caches and hi state explicitly configured
  inside a normal workspace remain writable; credential/config/data/trust paths
  do not receive that exception.
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
read-only host roots plus writable workspace/temp binds and shared
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
