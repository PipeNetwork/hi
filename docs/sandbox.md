# Shell sandbox (`HI_SANDBOX`)

Agent `bash` commands can run under an OS write-confine policy so a misguided
command cannot modify files outside the project. This is **in addition to** the
heuristic dangerous-command guard (`HI_ALLOW_DANGEROUS`) — not a replacement.

## Defaults (deliberate)

| Setting | Default | Why |
|--------|---------|-----|
| `HI_SANDBOX` unset / empty | **off** | Toolchains often write caches under `$HOME` (Cargo, npm, pip). Confining writes by default breaks normal coding-agent workflows. |
| `HI_SANDBOX=workspace` (or `on` / `1`) | **on** where enforced | Writes limited to the workspace root, system temp, and essential device nodes. Reads and network stay open. |
| `HI_SANDBOX=<typo>` | **startup error** | Unknown values are rejected so a typo cannot silently disable confinement. |

**Recommendation:** turn the sandbox **on** for untrusted prompts, multi-tenant
hosts, or when you do not need global package/tool caches. Leave it **off** for
day-to-day local development unless you have hit an accidental out-of-tree write.

```bash
# Confined shell writes (macOS Seatbelt today):
HI_SANDBOX=workspace hi "refactor the parser"

# Explicit off (same as unset):
HI_SANDBOX=off hi "..."
```

## Platform support

| Platform | Enforcement | Mechanism |
|----------|-------------|-----------|
| **macOS** | Yes when policy is `workspace` | `sandbox-exec` Seatbelt profile (deny `file-write*`, re-allow workspace + temp + devices) |
| **Linux** | **Not yet** — policy parses, commands unchanged | Planned: Landlock (kernel ≥5.13) and/or `bubblewrap` fallback — see sketch below |
| **Windows** | Not enforced | No profile |

Check at runtime: `ProcessRunner::sandbox_enforced()` is true only when a profile
was actually installed for this OS. When `HI_SANDBOX=workspace` is set on a
platform that cannot enforce it, `ProcessRunner::new` prints a one-shot
**warning** to stderr (see `SandboxProfile::unenforced_warning`).

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
| `HI_SANDBOX=off` / unset | No OS sandbox |
| `HI_ALLOW_DANGEROUS=1` | Disables the **heuristic** denylist only — does not disable OS sandbox |
| `HI_ALLOW_PRIVATE_WEB=1` | Relaxes SSRF private-IP blocks for `web_*` tools |

## Linux enforcement sketch (follow-up)

Goal: same **semantic** policy as macOS — write-confine to workspace + temp,
open read/net — without requiring a full container runtime.

1. **Detect:** if `landlock` is available (`linux/landlock.h` / `libc` ruleset
   syscalls on kernel ≥5.13), install an exclusive ruleset:
   - `LANDLOCK_ACCESS_FS_WRITE_FILE | REMOVE_* | MAKE_*` denied globally
   - path-beneath rules granting write under workspace + `$TMPDIR` + `/tmp`
   - leave read/execute/network unconstrained by Landlock (Landlock is FS-scoped)
2. **Fallback:** when Landlock is missing, optionally wrap with `bwrap`
   (`--ro-bind / / --bind workspace workspace --bind tmp tmp --dev /dev …`) if
   `bwrap` is on `PATH` and `HI_SANDBOX_BWRAP=1`.
3. **Default remains off** until the Linux path is integration-tested against
   Cargo/npm cache layouts (or we add explicit bind-mounts for
   `~/.cargo/registry`, `~/.npm`, etc. under a `workspace+caches` policy).
4. **Tests:** mirror macOS e2e in `sandbox.rs`: write outside workspace must fail;
   write inside and read `/etc/hosts` must succeed.

Until that lands, `HI_SANDBOX=workspace` on Linux is a **no-op for enforcement**
(documented here and in the module docs). The process still starts, but stderr
gets a one-shot warning so operators are not misled.

## Related

- Dangerous-command guard: `crates/hi-tools/src/guard.rs`
- RSI candidate host (stricter allowlisted shell): `crates/hi-tool-host`
- Architecture trust domains: [architecture.md](architecture.md)
