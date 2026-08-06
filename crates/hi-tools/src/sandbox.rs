//! OS sandboxing for shell commands the agent runs.
//!
//! The `workspace` policy (the default) confines a command's *writes* to the
//! workspace (plus temp and a handful of device nodes) while leaving reads and
//! network open — so a misbehaving or misguided command cannot modify files
//! outside the project. Reads stay open because a coding agent legitimately
//! reads system headers, toolchains, and libraries everywhere.
//!
//! `strict` is deny-by-default: only explicitly listed paths (workspace, temp,
//! system roots) are readable, and writes are confined to the workspace.
//! `readonly` allows reads everywhere but denies all writes and restricts
//! child-process network access.
//!
//! Enforcement is macOS (Seatbelt via `sandbox-exec`) and Linux (the pinned
//! rootless `pipe-wrap` binary). On other platforms, or when pipe-wrap is not
//! installed/capable, the policy parses but is **not enforced** and a warning
//! is emitted by [`crate::ProcessRunner`].
//!
//! **Default is workspace** so agent shells cannot write outside the project
//! without an explicit opt-out. Set `HI_SANDBOX=off` when global tool caches
//! under `$HOME` (Cargo/npm/pip) must remain writable. Full operator docs +
//! Linux Landlock/bwrap sketch: `docs/sandbox.md`.
//!
//! Path handling learns from grok-build's hard-won lesson: Seatbelt matches on
//! *real* paths, so every writable root is canonicalized (resolving the
//! `/tmp` → `/private/tmp` firmlink) before it goes into the profile — an
//! un-canonicalized `/tmp/...` subpath silently matches nothing and denies the
//! very writes it meant to allow.

use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};

/// Runtime state of the selected operating-system sandbox backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SandboxBackendStatus {
    Off,
    Enforced,
    Unavailable,
    Degraded,
}

/// Env marker set on every child spawned under an enforced hi sandbox.
///
/// A process that sees it is already write-confined by an ancestor's
/// Seatbelt/Landlock profile — OS confinement is inherited by every
/// descendant unconditionally — and macOS additionally refuses to apply a
/// new profile inside one (`sandbox-exec: sandbox_apply: Operation not
/// permitted`, exit 71). So a nested hi resolves its policy to `Off`:
/// skipping the redundant wrapper keeps the outer confinement intact and is
/// the only way spawns work at all. Without this, `hi` could not verify its
/// own test suite from inside a sandboxed session — every spawn-based test
/// failed on the nested wrapper.
pub const NESTED_SANDBOX_ENV: &str = "HI_SANDBOXED";

/// How much of the filesystem a shell command may modify.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SandboxPolicy {
    /// No sandbox — commands run with the process's own permissions.
    /// Opt in with `HI_SANDBOX=off` when home-dir tool caches must stay writable.
    Off,
    /// Writes confined to the workspace (+ temp + device nodes); reads open.
    /// Default for agent-spawned shells.
    #[default]
    Workspace,
    /// Deny-by-default: only workspace, temp, and system roots are readable;
    /// writes confined to the workspace. Strongest filesystem isolation.
    Strict,
    /// Reads open, all writes denied, child-process network restricted.
    ReadOnly,
}

impl SandboxPolicy {
    /// Parse a policy string (case-insensitive).
    ///
    /// - `workspace` / `on` / `1` → [`SandboxPolicy::Workspace`]
    /// - `strict` → [`SandboxPolicy::Strict`]
    /// - `readonly` / `read-only` → [`SandboxPolicy::ReadOnly`]
    /// - `off` / `0` / `false` / `no` / empty → [`SandboxPolicy::Off`]
    /// - anything else → `Err` with the original token (typos must not silently disable)
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "workspace" | "on" | "1" | "true" | "yes" => Ok(SandboxPolicy::Workspace),
            "strict" => Ok(SandboxPolicy::Strict),
            "readonly" | "read-only" => Ok(SandboxPolicy::ReadOnly),
            "off" | "0" | "false" | "no" | "" => Ok(SandboxPolicy::Off),
            other => Err(other.to_string()),
        }
    }

    /// Resolve the policy from `HI_SANDBOX`.
    ///
    /// Unset / empty → [`SandboxPolicy::Workspace`] (the default). Unknown
    /// non-empty values return `Err` so callers can refuse to start rather than
    /// silently running open. Use `off` / `0` / `false` / `no` to disable.
    ///
    /// When [`NESTED_SANDBOX_ENV`] is present the policy is `Off` regardless
    /// of `HI_SANDBOX`: the process is already confined by an ancestor's
    /// profile (which children inherit), and macOS denies re-applying one.
    pub fn from_env() -> Result<Self, String> {
        Self::resolve(
            std::env::var("HI_SANDBOX").ok().as_deref(),
            std::env::var_os(NESTED_SANDBOX_ENV).is_some(),
        )
    }

    /// Pure core of [`Self::from_env`] — both environmental inputs are
    /// explicit so tests never mutate process-global env.
    fn resolve(value: Option<&str>, nested: bool) -> Result<Self, String> {
        if nested {
            return Ok(SandboxPolicy::Off);
        }
        match value {
            None => Ok(SandboxPolicy::default()),
            Some(value) if value.trim().is_empty() => Ok(SandboxPolicy::default()),
            Some(value) => Self::parse(value).map_err(|token| {
                format!(
                    "unknown HI_SANDBOX value '{token}' \
                     (expected workspace|strict|readonly|on|1 or off|0|false)"
                )
            }),
        }
    }

    /// Whether this policy restricts child-process network access.
    pub fn restricts_network(self) -> bool {
        matches!(self, SandboxPolicy::ReadOnly | SandboxPolicy::Strict)
    }
}

/// Deny-path configuration layered on top of a [`SandboxPolicy`]. Paths in
/// `deny_write` are read-only even inside an otherwise writable workspace;
/// `deny_read` paths can't be read at all. Glob patterns (e.g. `**/*.pem`)
/// are expanded at launch time on Linux and evaluated as runtime regex on macOS.
#[derive(Clone, Debug, Default)]
pub struct SandboxConfig {
    /// Paths that are writable under the base policy but should be read-only.
    pub deny_write: Vec<PathBuf>,
    /// Paths that should be completely unreadable.
    pub deny_read: Vec<PathBuf>,
    /// Glob patterns to deny (e.g. `**/*.pem`, `**/.env*`).
    pub deny_globs: Vec<String>,
}

/// A resolved sandbox profile bound to a set of writable roots. Cheap to clone.
#[derive(Clone, Debug)]
pub struct SandboxProfile {
    policy: SandboxPolicy,
    /// The Seatbelt profile text (macOS) or Landlock rule spec (Linux). Empty
    /// when the policy is off or the platform is unenforced.
    profile: String,
    /// Deny-path config used by Linux pipe-wrap/bwrap compatibility paths.
    #[cfg(target_os = "linux")]
    config: SandboxConfig,
    #[cfg(target_os = "linux")]
    writable_roots: Vec<PathBuf>,
    /// Whether child-process network should be restricted (ReadOnly/Strict).
    restrict_network: bool,
    #[cfg(target_os = "linux")]
    pipe_wrap: Option<PathBuf>,
}

impl SandboxProfile {
    /// Build a profile for `policy` whose writable roots are `writable` (e.g.
    /// the workspace root and the agent's state directory). Non-existent roots
    /// are skipped; existing ones are canonicalized so Seatbelt subpath matches
    /// hit the real filesystem path.
    pub fn new(policy: SandboxPolicy, writable: &[&Path]) -> Self {
        Self::with_config(policy, writable, SandboxConfig::default())
    }

    /// Build a profile with additional deny-path configuration.
    pub fn with_config(policy: SandboxPolicy, writable: &[&Path], config: SandboxConfig) -> Self {
        if policy == SandboxPolicy::Off {
            return Self {
                policy,
                profile: String::new(),
                #[cfg(target_os = "linux")]
                config,
                #[cfg(target_os = "linux")]
                writable_roots: Vec::new(),
                restrict_network: false,
                #[cfg(target_os = "linux")]
                pipe_wrap: None,
            };
        }
        let restrict_network = policy.restricts_network();
        // hi's own state root (transaction journals, checkpoint refs) must
        // stay writable under write-allowing policies: edit/checkpoint
        // operations journal there *before* touching workspace files, so a
        // confined hi — a nested session, or sandboxed verify running this
        // repo's own tests — fails its first mutation without it. ReadOnly
        // keeps every write denied, including these.
        let mut roots: Vec<PathBuf> = writable.iter().map(|path| path.to_path_buf()).collect();
        if matches!(policy, SandboxPolicy::Workspace | SandboxPolicy::Strict) {
            let state_root = crate::checkpoint::default_state_root();
            // Seatbelt matches real paths; the root must exist to canonicalize.
            let _ = std::fs::create_dir_all(&state_root);
            roots.push(state_root);
        }
        let root_refs: Vec<&Path> = roots.iter().map(PathBuf::as_path).collect();
        let profile = if cfg!(target_os = "macos") {
            seatbelt_profile(policy, &root_refs, &config)
        } else {
            String::new()
        };
        #[cfg(target_os = "linux")]
        let pipe_wrap = discover_pipe_wrap();
        Self {
            policy,
            profile,
            #[cfg(target_os = "linux")]
            config,
            #[cfg(target_os = "linux")]
            writable_roots: roots,
            restrict_network,
            #[cfg(target_os = "linux")]
            pipe_wrap,
        }
    }

    /// Whether this profile actually enforces anything (on this platform).
    pub fn is_enforced(&self) -> bool {
        #[cfg(target_os = "macos")]
        {
            !self.profile.is_empty()
        }
        #[cfg(target_os = "linux")]
        {
            return self.pipe_wrap.is_some();
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            false
        }
    }

    /// Backend status suitable for reports and canonical agent events.
    pub fn backend_status(&self) -> SandboxBackendStatus {
        if self.policy == SandboxPolicy::Off {
            SandboxBackendStatus::Off
        } else if self.is_enforced() {
            SandboxBackendStatus::Enforced
        } else if cfg!(target_os = "linux") {
            SandboxBackendStatus::Unavailable
        } else {
            SandboxBackendStatus::Degraded
        }
    }

    /// Stable backend label for diagnostics.
    pub fn backend_name(&self) -> &'static str {
        if cfg!(target_os = "macos") {
            "seatbelt"
        } else if cfg!(target_os = "linux") {
            "pipe-wrap"
        } else {
            "none"
        }
    }

    /// True when the operator asked for confinement but this OS cannot enforce it.
    pub fn requested_but_unenforced(&self) -> bool {
        self.policy != SandboxPolicy::Off && !self.is_enforced()
    }

    /// One-line operator warning when [`Self::requested_but_unenforced`].
    pub fn unenforced_warning() -> &'static str {
        "HI_SANDBOX is set but OS write-confinement is not enforced \
         (macOS Seatbelt or Linux pipe-wrap is unavailable — see docs/sandbox.md)"
    }

    /// Whether child-process network access should be restricted.
    pub fn restricts_child_network(&self) -> bool {
        self.restrict_network
    }

    /// Wrap a `sh -c <command>` invocation so it runs under the sandbox. Returns
    /// the program and its argument vector. When the policy is off or the
    /// platform is unenforced, returns the plain `sh -c` invocation unchanged.
    pub fn wrap(&self, command: &str) -> (String, Vec<String>) {
        let (program, args) = self.wrap_program(
            OsStr::new("sh"),
            [OsString::from("-c"), OsString::from(command)],
        );
        (
            program.to_string_lossy().into_owned(),
            args.into_iter()
                .map(|arg| arg.to_string_lossy().into_owned())
                .collect(),
        )
    }

    /// Wrap a direct executable without routing its arguments through a shell.
    pub fn wrap_program<I, S>(&self, program: &OsStr, args: I) -> (OsString, Vec<OsString>)
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        let args: Vec<OsString> = args
            .into_iter()
            .map(|arg| arg.as_ref().to_os_string())
            .collect();
        if self.policy == SandboxPolicy::Off {
            return (program.to_os_string(), args);
        }
        if cfg!(target_os = "macos") && !self.profile.is_empty() {
            let mut wrapped = vec![
                OsString::from("-p"),
                OsString::from(&self.profile),
                program.to_os_string(),
            ];
            wrapped.extend(args);
            return (OsString::from("sandbox-exec"), wrapped);
        }
        #[cfg(target_os = "linux")]
        if let Some(pipe_wrap) = &self.pipe_wrap {
            let wrapped = pipe_wrap_arguments(
                self.policy,
                &self.config,
                &self.writable_roots,
                program,
                &args,
            );
            return (pipe_wrap.clone().into_os_string(), wrapped);
        }
        #[cfg(target_os = "linux")]
        {
            // Linux: the profile is the landlock ruleset spec, but enforcement
            // happens via bwrap re-exec for deny paths. For the simple case
            // (no deny paths), we wrap with bwrap --bind / /.
            if self.config.deny_write.is_empty() && self.config.deny_read.is_empty() {
                return (program.to_os_string(), args);
            }
            // With deny paths, use bwrap to bind-over the denied paths.
            let original_args = args;
            let mut args = vec!["--bind".to_string(), "/".to_string(), "/".to_string()];
            for path in &self.config.deny_write {
                if let Some(s) = path.to_str() {
                    args.push("--ro-bind".to_string());
                    args.push(s.to_string());
                    args.push(s.to_string());
                }
            }
            for path in &self.config.deny_read {
                if let Some(s) = path.to_str() {
                    // Bind /dev/null over the path to make it unreadable.
                    args.push("--ro-bind".to_string());
                    args.push("/dev/null".to_string());
                    args.push(s.to_string());
                }
            }
            args.push("--dev-bind".to_string());
            args.push("/dev".to_string());
            args.push("/dev".to_string());
            args.push("--proc".to_string());
            args.push("/proc".to_string());
            let mut wrapped = args.into_iter().map(OsString::from).collect::<Vec<_>>();
            wrapped.push(OsString::from("--"));
            wrapped.push(program.to_os_string());
            wrapped.extend(original_args);
            return (OsString::from("bwrap"), wrapped);
        }
        (program.to_os_string(), args)
    }

    pub fn policy(&self) -> SandboxPolicy {
        self.policy
    }
}

#[cfg(target_os = "linux")]
fn discover_pipe_wrap() -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(path) = std::env::var_os("HI_PIPE_WRAP") {
        candidates.push(PathBuf::from(path));
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(parent) = exe.parent()
    {
        candidates.push(parent.join("pipe-wrap"));
    }
    if let Some(path) = std::env::var_os("PATH") {
        candidates.extend(std::env::split_paths(&path).map(|dir| dir.join("pipe-wrap")));
    }
    candidates
        .into_iter()
        .find(|candidate| candidate.is_file() && pipe_wrap_capability_probe(candidate))
}

#[cfg(target_os = "linux")]
fn pipe_wrap_capability_probe(path: &Path) -> bool {
    let Ok(output) = std::process::Command::new(path).arg("--help").output() else {
        return false;
    };
    let mut help = String::from_utf8_lossy(&output.stdout).into_owned();
    help.push_str(&String::from_utf8_lossy(&output.stderr));
    let smoke = std::process::Command::new(path)
        .args(["--unshare-all", "--ro-bind", "/", "/", "--", "true"])
        .output();
    output.status.success()
        && help.contains("--ro-bind")
        && help.contains("--unshare-all")
        && help.contains("--die-with-parent")
        && smoke.is_ok_and(|result| result.status.success())
}

#[cfg(target_os = "linux")]
fn pipe_wrap_arguments(
    policy: SandboxPolicy,
    config: &SandboxConfig,
    writable_roots: &[PathBuf],
    program: &OsStr,
    program_args: &[OsString],
) -> Vec<OsString> {
    let mut args = vec![
        OsString::from("--die-with-parent"),
        OsString::from("--unshare-all"),
    ];
    if policy == SandboxPolicy::Workspace {
        // Workspace mode keeps network access for package managers, web tools,
        // and normal coding workflows while isolating the other namespaces.
        args.push(OsString::from("--share-net"));
    }

    match policy {
        SandboxPolicy::Workspace | SandboxPolicy::ReadOnly => {
            push_flag_path(&mut args, "--ro-bind", Path::new("/"), Path::new("/"));
        }
        SandboxPolicy::Strict => {
            args.extend([OsString::from("--tmpfs"), OsString::from("/")]);
            for path in ["/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc", "/opt"] {
                let path = Path::new(path);
                if path.exists() {
                    push_flag_path(&mut args, "--ro-bind", path, path);
                }
            }
        }
        SandboxPolicy::Off => {}
    }

    if policy == SandboxPolicy::Workspace {
        for root in writable_roots {
            push_flag_path(&mut args, "--bind", root, root);
        }
        let temp = Path::new("/tmp");
        if temp.exists() {
            push_flag_path(&mut args, "--bind", temp, temp);
        }
    }
    if policy == SandboxPolicy::Strict {
        for root in writable_roots {
            push_flag_path(&mut args, "--bind", root, root);
        }
        let temp = Path::new("/tmp");
        if temp.exists() {
            push_flag_path(&mut args, "--bind", temp, temp);
        }
    }
    if Path::new("/dev").exists() {
        args.extend([OsString::from("--dev"), OsString::from("/dev")]);
    }
    if Path::new("/proc").exists() {
        args.extend([OsString::from("--proc"), OsString::from("/proc")]);
    }
    for path in &config.deny_write {
        push_flag_path(&mut args, "--ro-bind", path, path);
    }
    for path in &config.deny_read {
        push_flag_path(&mut args, "--ro-bind", Path::new("/dev/null"), path);
    }
    args.push(OsString::from("--"));
    args.push(program.to_os_string());
    args.extend(program_args.iter().cloned());
    args
}

#[cfg(target_os = "linux")]
fn push_flag_path(args: &mut Vec<OsString>, flag: &str, source: &Path, target: &Path) {
    if source.exists() && target.parent().is_some_and(Path::exists) {
        args.push(OsString::from(flag));
        args.push(source.as_os_str().to_os_string());
        args.push(target.as_os_str().to_os_string());
    }
}

/// Build a Seatbelt profile for macOS. The structure depends on the policy:
///
/// - `Workspace`: allow everything by default, deny all writes, re-allow writes
///   under writable roots + temp + devices. Deny paths get specific write
///   sub-action denies that survive last-match-wins ordering.
/// - `Strict`: deny everything by default, allow reads only for system roots +
///   workspace + temp, allow writes only for workspace + temp + devices.
/// - `ReadOnly`: allow reads, deny all writes (no writable roots), restrict
///   network.
fn seatbelt_profile(policy: SandboxPolicy, writable: &[&Path], config: &SandboxConfig) -> String {
    let mut out = String::from("(version 1)\n");
    match policy {
        SandboxPolicy::Workspace => {
            out.push_str("(allow default)\n(deny file-write*)\n");
            push_device_writes(&mut out);
            for temp in temp_roots() {
                out.push_str(&format!("(allow file-write* (subpath {}))\n", quote(&temp)));
            }
            for root in writable {
                if let Ok(canonical) = root.canonicalize()
                    && let Some(text) = canonical.to_str()
                {
                    out.push_str(&format!("(allow file-write* (subpath {}))\n", quote(text)));
                }
            }
        }
        SandboxPolicy::Strict => {
            out.push_str("(deny default)\n");
            // Allow reads from system roots, workspace, and temp.
            for readable in system_readable_roots() {
                out.push_str(&format!(
                    "(allow file-read* (subpath {}))\n",
                    quote(&readable)
                ));
            }
            for root in writable {
                if let Ok(canonical) = root.canonicalize()
                    && let Some(text) = canonical.to_str()
                {
                    out.push_str(&format!("(allow file-read* (subpath {}))\n", quote(text)));
                    out.push_str(&format!("(allow file-write* (subpath {}))\n", quote(text)));
                }
            }
            push_device_writes(&mut out);
            push_device_reads(&mut out);
            for temp in temp_roots() {
                out.push_str(&format!("(allow file-read* (subpath {}))\n", quote(&temp)));
                out.push_str(&format!("(allow file-write* (subpath {}))\n", quote(&temp)));
            }
            // Allow process execution from system paths.
            out.push_str("(allow process-exec (subpath \"/usr\"))\n");
            out.push_str("(allow process-exec (subpath \"/bin\"))\n");
            out.push_str("(allow process-exec (subpath \"/opt\"))\n");
        }
        SandboxPolicy::ReadOnly => {
            out.push_str("(allow default)\n(deny file-write*)\n");
            push_device_writes(&mut out);
            // No writable roots — temp is still needed for toolchains.
            for temp in temp_roots() {
                out.push_str(&format!("(allow file-write* (subpath {}))\n", quote(&temp)));
            }
            // Restrict network: deny all socket operations.
            out.push_str("(deny network*)\n");
        }
        SandboxPolicy::Off => {}
    }
    // Deny paths: emit specific write sub-action denies that survive
    // last-match-wins ordering even inside an allowed workspace subpath.
    for path in &config.deny_write {
        if let Ok(canonical) = path.canonicalize() {
            for alias in macos_deny_aliases(path, &canonical) {
                emit_seatbelt_deny(&mut out, &alias, false);
            }
        }
    }
    for path in &config.deny_read {
        if let Ok(canonical) = path.canonicalize() {
            for alias in macos_deny_aliases(path, &canonical) {
                emit_seatbelt_deny(&mut out, &alias, true);
            }
        }
    }
    out
}

fn push_device_writes(out: &mut String) {
    out.push_str(
        "(allow file-write*\n  (literal \"/dev/null\")\n  (literal \"/dev/stdout\")\n  \
         (literal \"/dev/stderr\")\n  (literal \"/dev/tty\")\n  (literal \"/dev/dtracehelper\")\n  \
         (literal \"/dev/zero\")\n  (subpath \"/dev/fd\"))\n",
    );
}

fn push_device_reads(out: &mut String) {
    out.push_str(
        "(allow file-read*\n  (literal \"/dev/null\")\n  (literal \"/dev/zero\")\n  \
         (literal \"/dev/urandom\")\n  (literal \"/dev/random\")\n  (subpath \"/dev/fd\"))\n",
    );
}

/// System roots that remain readable under the `strict` policy.
fn system_readable_roots() -> Vec<String> {
    let mut roots = Vec::new();
    for candidate in [
        "/usr", "/bin", "/sbin", "/lib", "/lib64", "/opt", "/etc", "/System",
    ] {
        if Path::new(candidate).exists() {
            roots.push(candidate.to_string());
        }
    }
    roots
}

/// Emit Seatbelt deny rules for a path. `deny_read` controls whether reads are
/// also denied. The 8 specific write sub-actions survive last-match-wins
/// ordering even when a broader `(allow file-write* (subpath ...))` is emitted
/// later — the specific sub-action deny is more specific and wins.
fn emit_seatbelt_deny(out: &mut String, path: &Path, deny_read: bool) {
    let Some(text) = path.to_str() else { return };
    let quoted = quote(text);
    if deny_read {
        out.push_str(&format!("(deny file-read* (literal {quoted}))\n"));
    }
    out.push_str(&format!("(deny file-write* (literal {quoted}))\n"));
    for sub in [
        "file-write-data",
        "file-write-create",
        "file-write-unlink",
        "file-write-mode",
        "file-write-owner",
        "file-write-flags",
        "file-write-times",
        "file-write-setugid",
    ] {
        out.push_str(&format!("(deny {sub} (literal {quoted}))\n"));
    }
}

/// Generate all macOS firmlink alias forms for a deny path so that a deny on
/// `/private/tmp/proj/.env` also covers `/tmp/proj/.env` and vice versa.
fn macos_deny_aliases(path: &Path, canonical: &Path) -> Vec<PathBuf> {
    let mut forms = vec![path.to_path_buf()];
    if canonical != path {
        forms.push(canonical.to_path_buf());
    }
    let mut expanded = Vec::new();
    for form in &forms {
        expanded.push(form.clone());
        if let Some(alias) = toggle_private_prefix(form) {
            expanded.push(alias);
        }
    }
    expanded
}

/// Toggle between `/private/tmp` ↔ `/tmp`, `/private/var` ↔ `/var`, etc.
fn toggle_private_prefix(path: &Path) -> Option<PathBuf> {
    let s = path.to_str()?;
    if let Some(rest) = s.strip_prefix("/private/tmp") {
        return Some(PathBuf::from(format!("/tmp{rest}")));
    }
    if let Some(rest) = s.strip_prefix("/tmp") {
        return Some(PathBuf::from(format!("/private/tmp{rest}")));
    }
    if let Some(rest) = s.strip_prefix("/private/var") {
        return Some(PathBuf::from(format!("/var{rest}")));
    }
    if let Some(rest) = s.strip_prefix("/var") {
        return Some(PathBuf::from(format!("/private/var{rest}")));
    }
    None
}

/// Canonicalized system temp roots that must stay writable.
fn temp_roots() -> Vec<String> {
    let mut roots = Vec::new();
    for candidate in [
        "/tmp",
        "/var/folders",
        "/private/tmp",
        "/private/var/folders",
    ] {
        if let Ok(canonical) = Path::new(candidate).canonicalize()
            && let Some(text) = canonical.to_str()
        {
            let text = text.to_string();
            if !roots.contains(&text) {
                roots.push(text);
            }
        }
    }
    roots
}

/// Quote a path for a Seatbelt profile string literal. Seatbelt uses
/// double-quoted strings with backslash escaping; workspace paths with a quote
/// or backslash are exotic but must not break the profile (or worse, escape it).
fn quote(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for ch in text.chars() {
        if ch == '"' || ch == '\\' {
            out.push('\\');
        }
        out.push(ch);
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_default_is_workspace() {
        // We can't safely mutate process env in parallel tests; assert the
        // pure-mapping behaviour via Default (from_env unset/empty uses it).
        assert_eq!(SandboxPolicy::default(), SandboxPolicy::Workspace);
    }

    #[test]
    fn policy_resolution_is_off_inside_an_existing_hi_sandbox() {
        // Nested: the ancestor's profile already confines every descendant,
        // and macOS refuses `sandbox_apply` inside one — re-wrapping would
        // turn every spawn into "exited with code 71". This is what lets hi
        // verify its own test suite from a sandboxed session.
        assert_eq!(
            SandboxPolicy::resolve(None, true).unwrap(),
            SandboxPolicy::Off
        );
        assert_eq!(
            SandboxPolicy::resolve(Some("workspace"), true).unwrap(),
            SandboxPolicy::Off,
            "an explicit HI_SANDBOX cannot re-apply inside an outer sandbox"
        );
        // Not nested: default and explicit values resolve as documented.
        assert_eq!(
            SandboxPolicy::resolve(None, false).unwrap(),
            SandboxPolicy::Workspace
        );
        assert_eq!(
            SandboxPolicy::resolve(Some(""), false).unwrap(),
            SandboxPolicy::Workspace
        );
        assert_eq!(
            SandboxPolicy::resolve(Some("off"), false).unwrap(),
            SandboxPolicy::Off
        );
        assert!(SandboxPolicy::resolve(Some("bogus"), false).is_err());
    }

    #[test]
    fn policy_parse_accepts_known_tokens() {
        assert_eq!(
            SandboxPolicy::parse("workspace").unwrap(),
            SandboxPolicy::Workspace
        );
        assert_eq!(
            SandboxPolicy::parse("ON").unwrap(),
            SandboxPolicy::Workspace
        );
        assert_eq!(
            SandboxPolicy::parse("strict").unwrap(),
            SandboxPolicy::Strict
        );
        assert_eq!(
            SandboxPolicy::parse("readonly").unwrap(),
            SandboxPolicy::ReadOnly
        );
        assert_eq!(
            SandboxPolicy::parse("read-only").unwrap(),
            SandboxPolicy::ReadOnly
        );
        assert_eq!(SandboxPolicy::parse("off").unwrap(), SandboxPolicy::Off);
        assert_eq!(SandboxPolicy::parse("").unwrap(), SandboxPolicy::Off);
    }

    #[test]
    fn policy_parse_rejects_unknown_tokens() {
        let err = SandboxPolicy::parse("maybe").unwrap_err();
        assert_eq!(err, "maybe");
        assert!(SandboxPolicy::parse("workspaces").is_err());
    }

    #[test]
    fn restricts_network_is_true_for_strict_and_readonly() {
        assert!(SandboxPolicy::ReadOnly.restricts_network());
        assert!(SandboxPolicy::Strict.restricts_network());
        assert!(!SandboxPolicy::Workspace.restricts_network());
        assert!(!SandboxPolicy::Off.restricts_network());
    }

    #[test]
    fn workspace_policy_reports_platform_enforcement() {
        let profile = SandboxProfile::new(SandboxPolicy::Workspace, &[]);
        if cfg!(target_os = "macos") {
            assert!(!profile.requested_but_unenforced());
            assert!(profile.is_enforced());
        } else if cfg!(target_os = "linux") {
            // Linux enforcement depends on a rootless, capability-compatible
            // pipe-wrap artifact and user-namespace policy at runtime.
            assert_eq!(!profile.requested_but_unenforced(), profile.is_enforced());
        } else {
            assert!(profile.requested_but_unenforced());
            assert!(!profile.is_enforced());
        }
        let off = SandboxProfile::new(SandboxPolicy::Off, &[]);
        assert!(!off.requested_but_unenforced());
    }

    #[test]
    fn off_policy_wraps_to_plain_sh() {
        let profile = SandboxProfile::new(SandboxPolicy::Off, &[]);
        assert!(!profile.is_enforced());
        let (prog, args) = profile.wrap("echo hi");
        assert_eq!(prog, "sh");
        assert_eq!(args, vec!["-c", "echo hi"]);
    }

    #[test]
    fn quote_escapes_quotes_and_backslashes() {
        assert_eq!(quote("/a/b"), "\"/a/b\"");
        assert_eq!(quote("/a\"b"), "\"/a\\\"b\"");
        assert_eq!(quote("/a\\b"), "\"/a\\\\b\"");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn workspace_profile_names_the_canonical_root_and_denies_writes() {
        let dir = std::env::temp_dir().join(format!("hi-sb-prof-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let profile = SandboxProfile::new(SandboxPolicy::Workspace, &[dir.as_path()]);
        assert!(profile.is_enforced(), "macOS enforces the workspace policy");
        let (prog, args) = profile.wrap("true");
        assert_eq!(prog, "sandbox-exec");
        let text = &args[1];
        assert!(text.contains("(deny file-write*)"));
        // The canonical path (with /tmp → /private/tmp resolved) must appear.
        let canonical = dir.canonicalize().unwrap();
        assert!(
            text.contains(canonical.to_str().unwrap()),
            "profile names the canonical workspace root: {text}"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// End-to-end: a command under the workspace sandbox may write inside the
    /// workspace but is denied writes elsewhere, while reads stay open.
    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn workspace_sandbox_confines_writes_but_not_reads() {
        use std::process::Command;
        if std::env::var_os(NESTED_SANDBOX_ENV).is_some() {
            eprintln!("skipped: already inside an hi sandbox — macOS denies nested sandbox_apply");
            return;
        }
        let ws = std::env::temp_dir().join(format!("hi-sb-ws-{}", std::process::id()));
        std::fs::create_dir_all(&ws).unwrap();
        let ws_canon = ws.canonicalize().unwrap();
        // "Outside" must be a non-temp, non-workspace location — temp is
        // deliberately writable, so a sibling under /var/folders would pass.
        let home = std::env::var("HOME").expect("HOME set on macOS");
        let outside = Path::new(&home).join(format!(".hi-sb-leak-{}.txt", std::process::id()));
        let _ = std::fs::remove_file(&outside);
        let profile = SandboxProfile::new(SandboxPolicy::Workspace, &[ws.as_path()]);

        let run = |command: String| {
            let (prog, args) = profile.wrap(&command);
            Command::new(prog).args(args).output().unwrap()
        };

        // Write inside the workspace: allowed.
        let inside_file = ws_canon.join("inside.txt");
        let out = run(format!("echo hi > {}", inside_file.display()));
        assert!(out.status.success(), "write inside workspace must succeed");
        assert!(inside_file.exists());

        // Write outside the workspace: denied (non-zero, file not created).
        let out = run(format!("echo leak > {}", outside.display()));
        assert!(!out.status.success(), "write outside workspace must fail");
        assert!(!outside.exists(), "no file should be created outside");

        // Read outside the workspace: allowed.
        let out = run("head -c 1 /etc/hosts >/dev/null".to_string());
        assert!(out.status.success(), "reads outside stay open");

        std::fs::remove_dir_all(&ws).unwrap();
        let _ = std::fs::remove_file(&outside);
    }

    /// The bash tool path (ProcessRunner::spawn_shell) actually applies the
    /// sandbox when `HI_SANDBOX=workspace`.
    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn process_runner_applies_sandbox_from_env() {
        if std::env::var_os(NESTED_SANDBOX_ENV).is_some() {
            eprintln!("skipped: already inside an hi sandbox — macOS denies nested sandbox_apply");
            return;
        }
        // Serialize env mutation with other env-sensitive tests via a lock.
        static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let ws = std::env::temp_dir().join(format!("hi-sb-runner-{}", std::process::id()));
        std::fs::create_dir_all(&ws).unwrap();
        let home = std::env::var("HOME").expect("HOME set on macOS");
        let outside = Path::new(&home).join(format!(".hi-sb-runner-leak-{}", std::process::id()));
        let _ = std::fs::remove_file(&outside);

        // The runner captures the profile at construction, so env only needs to
        // be set across `new` — not across the await. Build under the lock,
        // then release it before running the command.
        let runner = {
            let _guard = ENV_LOCK.lock().unwrap();
            // SAFETY: guarded by ENV_LOCK; restored before the guard drops.
            unsafe { std::env::set_var("HI_SANDBOX", "workspace") };
            let runner = crate::ProcessRunner::new(&ws).unwrap();
            unsafe { std::env::remove_var("HI_SANDBOX") };
            runner
        };
        assert!(runner.sandbox_enforced(), "runner picked up HI_SANDBOX");
        let mut sink = |_: &str| {};
        let exec = runner
            .run_shell_streaming(
                &format!("echo leak > {}", outside.display()),
                std::time::Duration::from_secs(10),
                &mut sink,
            )
            .await
            .unwrap();

        assert_ne!(
            exec.status,
            crate::ToolStatus::Succeeded,
            "a write outside the workspace must be denied by the sandbox"
        );
        assert!(!outside.exists(), "no leak file created");
        std::fs::remove_dir_all(&ws).unwrap();
    }
}
