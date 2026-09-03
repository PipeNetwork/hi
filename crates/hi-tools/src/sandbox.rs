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
//! without an explicit opt-out. Shared toolchain caches under `$HOME` stay
//! read-only: their contents can include executable source, wheels, or build
//! artifacts that would otherwise persist beyond the sandboxed command. Set
//! `HI_SANDBOX=off` (or point a tool's cache inside the workspace) when a build
//! must populate one. Full operator docs + Linux Landlock/bwrap sketch:
//! `docs/sandbox.md`.
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
/// failed on the nested wrapper. The outer profile still keeps hi's shared
/// state root read-only; a nested command that needs state writes must use a
/// project-local state directory inside the explicit workspace.
pub const NESTED_SANDBOX_ENV: &str = "HI_SANDBOXED";

/// How much of the filesystem a shell command may modify.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SandboxPolicy {
    /// No sandbox — commands run with the process's own permissions.
    /// Opt out with `HI_SANDBOX=off` when home-dir tool caches must stay writable.
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
    /// Skip broad host-temp allowances. Hermetic embedded runners can place
    /// one private temp directory beneath an explicit writable root instead.
    pub deny_host_temp: bool,
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
    #[cfg(target_os = "linux")]
    protected_roots: Vec<PathBuf>,
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
                #[cfg(target_os = "linux")]
                protected_roots: Vec::new(),
                restrict_network: false,
                #[cfg(target_os = "linux")]
                pipe_wrap: None,
            };
        }
        let restrict_network = policy.restricts_network();
        let protected_paths = shared_cache_protected_paths_for_writable_roots(writable);
        let roots: Vec<PathBuf> = writable
            .iter()
            .filter(|path| !writable_root_exposes_protected_path(path, &protected_paths))
            .map(|path| path.to_path_buf())
            .collect();
        let root_refs: Vec<&Path> = roots.iter().map(PathBuf::as_path).collect();
        let profile = if cfg!(target_os = "macos") {
            seatbelt_profile_with_protected_paths(policy, &root_refs, &config, &protected_paths)
        } else {
            String::new()
        };
        #[cfg(target_os = "linux")]
        let pipe_wrap = discover_pipe_wrap(&roots);
        Self {
            policy,
            profile,
            #[cfg(target_os = "linux")]
            config,
            #[cfg(target_os = "linux")]
            writable_roots: roots,
            #[cfg(target_os = "linux")]
            protected_roots: protected_paths,
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
            self.pipe_wrap.is_some()
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
        // `sandbox-exec` looks up a relative name with a sanitized PATH, so
        // `rg` from Cursor's bundle (or Homebrew) becomes "No such file"
        // exit 71 even though the parent process can see it. Pass an
        // absolute path when we can resolve one.
        let program = resolve_executable(program);
        if cfg!(target_os = "macos") && !self.profile.is_empty() {
            let mut wrapped = vec![
                OsString::from("-p"),
                OsString::from(&self.profile),
                program.clone(),
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
                &self.protected_roots,
                program.as_os_str(),
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
        // On Linux every cfg branch above returns, so this fallthrough is only
        // reachable on platforms without the Linux bwrap wrap. Keep it as the
        // safe default rather than an unreachable!() panic.
        #[cfg_attr(target_os = "linux", allow(unreachable_code))]
        (program.to_os_string(), args)
    }

    pub fn policy(&self) -> SandboxPolicy {
        self.policy
    }
}

/// Resolve `program` against `$PATH` so sandbox wrappers exec a real file.
/// Relative names are left unchanged when nothing matches.
fn resolve_executable(program: &OsStr) -> OsString {
    let path = Path::new(program);
    if path.is_absolute() {
        return program.to_os_string();
    }
    if path.components().count() > 1 {
        return path
            .canonicalize()
            .map(PathBuf::into_os_string)
            .unwrap_or_else(|_| program.to_os_string());
    }
    let Some(path_os) = std::env::var_os("PATH") else {
        return program.to_os_string();
    };
    for dir in std::env::split_paths(&path_os) {
        let candidate = dir.join(path);
        if executable_file(&candidate) {
            return candidate.into_os_string();
        }
    }
    program.to_os_string()
}

fn executable_file(path: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::metadata(path)
            .ok()
            .is_some_and(|meta| meta.is_file() && meta.permissions().mode() & 0o111 != 0)
    }
    #[cfg(not(unix))]
    {
        path.is_file()
    }
}

#[cfg(target_os = "linux")]
fn discover_pipe_wrap(writable_roots: &[PathBuf]) -> Option<PathBuf> {
    let explicit = std::env::var_os("HI_PIPE_WRAP");
    let current_exe = std::env::current_exe().ok();
    pipe_wrap_candidates(explicit.as_deref(), current_exe.as_deref(), writable_roots)
        .into_iter()
        .find_map(|candidate| {
            let candidate = candidate.canonicalize().ok()?;
            (candidate.is_file() && pipe_wrap_capability_probe(&candidate)).then_some(candidate)
        })
}

/// Return only wrapper locations whose selection is controlled by the
/// operator or the hi installation. `$PATH` is deliberately not an input:
/// probing an attacker-created `pipe-wrap` would execute it before any
/// sandbox exists.
#[cfg(any(test, target_os = "linux"))]
fn pipe_wrap_candidates(
    explicit: Option<&OsStr>,
    current_exe: Option<&Path>,
    writable_roots: &[PathBuf],
) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(path) = explicit.map(PathBuf::from)
        && path.is_absolute()
    {
        // HI_PIPE_WRAP is an explicit operator trust decision. Requiring an
        // absolute path prevents the current directory or PATH from deciding
        // which program the capability probe executes.
        candidates.push(path);
    }
    if let Some(sibling) = current_exe
        .and_then(Path::parent)
        .map(|parent| parent.join("pipe-wrap"))
        && !sandbox_can_write_path(&sibling, writable_roots)
    {
        candidates.push(sibling);
    }
    candidates.dedup();
    candidates
}

#[cfg(any(test, target_os = "linux"))]
fn sandbox_can_write_path(path: &Path, writable_roots: &[PathBuf]) -> bool {
    let resolved_path = resolve_path_for_comparison(path);
    writable_roots.iter().any(|root| {
        let resolved_root = resolve_path_for_comparison(root);
        resolved_path.starts_with(resolved_root)
    }) || [Path::new("/tmp"), Path::new("/var/tmp")]
        .into_iter()
        .map(resolve_path_for_comparison)
        .any(|root| resolved_path.starts_with(root))
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
        && help.contains("--tmpfs")
        && smoke.is_ok_and(|result| result.status.success())
}

#[cfg(target_os = "linux")]
fn pipe_wrap_arguments(
    policy: SandboxPolicy,
    config: &SandboxConfig,
    writable_roots: &[PathBuf],
    protected_roots: &[PathBuf],
    program: &OsStr,
    program_args: &[OsString],
) -> Vec<OsString> {
    pipe_wrap_arguments_with_protected_roots(
        policy,
        config,
        writable_roots,
        protected_roots,
        program,
        program_args,
    )
}

#[cfg(any(test, target_os = "linux"))]
fn pipe_wrap_arguments_with_protected_roots(
    policy: SandboxPolicy,
    config: &SandboxConfig,
    writable_roots: &[PathBuf],
    protected_roots: &[PathBuf],
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

    if matches!(policy, SandboxPolicy::Workspace | SandboxPolicy::Strict) {
        // A host-shared writable /tmp lets an absent configured cache/state
        // path be created and poisoned. Use a private temp mount instead, then
        // layer explicit project roots back in (so /tmp/project still works).
        // Existing protected paths under host temp are reintroduced read-only
        // by the late overlays below.
        let temp = Path::new("/tmp");
        if temp.exists() {
            args.push(OsString::from("--tmpfs"));
            args.push(temp.as_os_str().to_os_string());
        }
        if let Some(home) = std::env::var_os("HOME")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
            .filter(|home| home.exists() && path_is_under_linux_temp(home))
        {
            push_flag_path(&mut args, "--ro-bind", &home, &home);
        }
        for root in writable_roots {
            if !same_resolved_path(root, temp) {
                push_flag_path(&mut args, "--bind", root, root);
            }
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
    if matches!(policy, SandboxPolicy::Workspace | SandboxPolicy::Strict) {
        // Defense in depth when an operator selects a broad ancestor: later
        // read-only binds keep shared hi state, credentials, config, cached
        // code, and toolchain executables outside the command's write scope.
        for path in protected_roots {
            push_flag_path(&mut args, "--ro-bind", path, path);
        }
    }
    args.push(OsString::from("--"));
    args.push(program.to_os_string());
    args.extend(program_args.iter().cloned());
    args
}

#[cfg(any(test, target_os = "linux"))]
fn push_flag_path(args: &mut Vec<OsString>, flag: &str, source: &Path, target: &Path) {
    if source.exists() && target.parent().is_some_and(Path::exists) {
        args.push(OsString::from(flag));
        args.push(source.as_os_str().to_os_string());
        args.push(target.as_os_str().to_os_string());
    }
}

#[cfg(any(test, target_os = "linux"))]
fn same_resolved_path(left: &Path, right: &Path) -> bool {
    resolve_path_for_comparison(left) == resolve_path_for_comparison(right)
}

#[cfg(any(test, target_os = "linux"))]
fn path_is_under_linux_temp(path: &Path) -> bool {
    let temp = Path::new("/tmp");
    path.starts_with(temp)
        || resolve_path_for_comparison(path).starts_with(resolve_path_for_comparison(temp))
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
fn seatbelt_profile_with_protected_paths(
    policy: SandboxPolicy,
    writable: &[&Path],
    config: &SandboxConfig,
    protected_paths: &[PathBuf],
) -> String {
    let mut out = String::from("(version 1)\n");
    match policy {
        SandboxPolicy::Workspace => {
            out.push_str("(allow default)\n(deny file-write*)\n");
            push_device_writes(&mut out);
            if !config.deny_host_temp {
                for temp in temp_roots() {
                    out.push_str(&format!("(allow file-write* (subpath {}))\n", quote(&temp)));
                }
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
            if !config.deny_host_temp {
                for temp in temp_roots() {
                    out.push_str(&format!("(allow file-read* (subpath {}))\n", quote(&temp)));
                    out.push_str(&format!("(allow file-write* (subpath {}))\n", quote(&temp)));
                }
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
            if !config.deny_host_temp {
                for temp in temp_roots() {
                    out.push_str(&format!("(allow file-write* (subpath {}))\n", quote(&temp)));
                }
            }
            // Restrict network: deny all socket operations.
            out.push_str("(deny network*)\n");
        }
        SandboxPolicy::Off => {}
    }
    if matches!(policy, SandboxPolicy::Workspace | SandboxPolicy::Strict) {
        // These denies are deliberately emitted after the broad writable-root
        // allows above. Subpath rules protect executable cache contents as
        // well as config/credential files, including paths not created yet.
        for path in protected_paths {
            let resolved = resolve_path_for_comparison(path);
            for alias in macos_deny_aliases(path, &resolved) {
                emit_seatbelt_subpath_deny(&mut out, &alias);
            }
        }
    }
    if policy == SandboxPolicy::Workspace {
        for path in toolchain_secret_deny_paths() {
            emit_seatbelt_deny(&mut out, &path, false);
            if let Ok(canonical) = path.canonicalize()
                && canonical != path
            {
                emit_seatbelt_deny(&mut out, &canonical, false);
            }
        }
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

/// Shared hi state and home caches can contain durable approvals, signing
/// keys, executable source, wheels, shims, toolchains, and build artifacts.
/// Include paths that do not exist yet so an over-broad workspace cannot
/// create and poison them.
fn default_shared_cache_paths() -> Vec<PathBuf> {
    let home = std::env::var_os("HOME")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from);
    let mut paths = Vec::new();
    if let Some(home) = home {
        for relative in [
            ".cargo",
            ".rustup",
            ".npm",
            ".config/hi",
            ".cache/pip",
            ".cache/uv",
            ".cache/go-build",
            ".hi",
            ".local/share/hi",
            "Library/Caches/pip",
            "Library/Caches/uv",
            "Library/Caches/go-build",
            "go/pkg",
            ".local/state/hi",
        ] {
            let path = home.join(relative);
            if path.is_absolute() && !paths.contains(&path) {
                paths.push(path);
            }
        }
    }
    paths
}

fn configured_shared_cache_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    let mut push = |path: PathBuf| {
        if path.is_absolute() && !paths.contains(&path) {
            paths.push(path);
        }
    };

    for name in [
        "CARGO_HOME",
        "RUSTUP_HOME",
        "NPM_CONFIG_CACHE",
        "npm_config_cache",
        "PIP_CACHE_DIR",
        "UV_CACHE_DIR",
        "GOCACHE",
        "GOMODCACHE",
    ] {
        if let Some(path) = std::env::var_os(name)
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
        {
            push(path);
        }
    }
    if let Some(gopath) = std::env::var_os("GOPATH") {
        for root in std::env::split_paths(&gopath) {
            push(root.join("pkg"));
        }
    }
    if let Some(xdg_cache) = std::env::var_os("XDG_CACHE_HOME")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
    {
        for relative in ["pip", "uv", "go-build"] {
            push(xdg_cache.join(relative));
        }
    }
    if let Some(state_root) = std::env::var_os("HI_STATE_ROOT")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
    {
        push(state_root);
    }
    if let Some(xdg_state) = std::env::var_os("XDG_STATE_HOME")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
    {
        push(xdg_state.join("hi"));
    }
    paths
}

/// Config/data roots and explicit standing-rule/trust files are control-plane
/// inputs, not disposable caches. They remain protected even when their env
/// override points beneath a writable workspace.
fn configured_sensitive_paths() -> Vec<PathBuf> {
    sensitive_paths_from_inputs(
        std::env::var_os("XDG_CONFIG_HOME")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from),
        std::env::var_os("XDG_DATA_HOME")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from),
        std::env::var_os("HI_TRUST_STORE")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from),
        std::env::var_os("HI_ME_MD")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from),
        std::env::current_dir().ok().as_deref(),
    )
}

fn sensitive_paths_from_inputs(
    config_home: Option<PathBuf>,
    data_home: Option<PathBuf>,
    trust_store: Option<PathBuf>,
    me_md: Option<PathBuf>,
    current_dir: Option<&Path>,
) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    let mut push = |path: PathBuf| {
        let path = if path.is_absolute() {
            path
        } else if let Some(current) = current_dir {
            current.join(path)
        } else {
            return;
        };
        if !paths.contains(&path) {
            paths.push(path);
        }
    };
    if let Some(config) = config_home {
        push(config.join("hi"));
    }
    if let Some(data) = data_home {
        push(data.join("hi"));
    }
    for path in [trust_store, me_md].into_iter().flatten() {
        push(path);
    }
    paths
}

fn shared_cache_protected_paths_for_writable_roots(writable: &[&Path]) -> Vec<PathBuf> {
    let mut protected = scoped_shared_cache_protected_paths(
        default_shared_cache_paths(),
        configured_shared_cache_paths(),
        writable,
    );
    for path in configured_sensitive_paths() {
        if !protected.contains(&path) {
            protected.push(path);
        }
    }
    protected
}

fn scoped_shared_cache_protected_paths(
    defaults: Vec<PathBuf>,
    configured: Vec<PathBuf>,
    writable: &[&Path],
) -> Vec<PathBuf> {
    let normal_writable_roots = writable
        .iter()
        .filter(|root| !writable_root_exposes_protected_path(root, &defaults))
        .map(|root| resolve_path_for_comparison(root))
        .collect::<Vec<_>>();
    let mut protected = defaults;
    for path in configured {
        let resolved = resolve_path_for_comparison(&path);
        let intentionally_in_workspace = normal_writable_roots
            .iter()
            .any(|root| resolved.starts_with(root));
        if !intentionally_in_workspace && !protected.contains(&path) {
            protected.push(path);
        }
    }
    protected
}

fn writable_root_exposes_protected_path(root: &Path, protected_paths: &[PathBuf]) -> bool {
    let root = resolve_path_for_comparison(root);
    root.parent().is_none()
        || protected_paths
            .iter()
            .map(|path| resolve_path_for_comparison(path))
            .any(|path| path.starts_with(&root) || root.starts_with(&path))
}

/// Canonicalize through the nearest existing ancestor so comparisons remain
/// sound for a protected cache that has not been created yet.
fn resolve_path_for_comparison(path: &Path) -> PathBuf {
    if let Ok(canonical) = path.canonicalize() {
        return canonical;
    }
    let mut cursor = path;
    let mut missing = Vec::new();
    while !cursor.exists() {
        let Some(name) = cursor.file_name() else {
            return path.to_path_buf();
        };
        missing.push(name.to_os_string());
        let Some(parent) = cursor.parent() else {
            return path.to_path_buf();
        };
        cursor = parent;
    }
    let Ok(mut resolved) = cursor.canonicalize() else {
        return path.to_path_buf();
    };
    for component in missing.into_iter().rev() {
        resolved.push(component);
    }
    resolved
}

fn toolchain_secret_deny_paths() -> Vec<PathBuf> {
    let cargo_home = std::env::var_os("CARGO_HOME")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .filter(|value| !value.is_empty())
                .map(|home| PathBuf::from(home).join(".cargo"))
        });
    let mut paths = cargo_home
        .into_iter()
        .flat_map(|cargo_home| {
            ["config", "config.toml", "credentials", "credentials.toml"]
                .into_iter()
                .map(move |name| cargo_home.join(name))
        })
        .collect::<Vec<_>>();
    if let Some(rustup_home) = std::env::var_os("RUSTUP_HOME")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .filter(|value| !value.is_empty())
                .map(|home| PathBuf::from(home).join(".rustup"))
        })
    {
        paths.push(rustup_home.join("settings.toml"));
    }
    paths
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

fn emit_seatbelt_subpath_deny(out: &mut String, path: &Path) {
    let Some(text) = path.to_str() else { return };
    let quoted = quote(text);
    out.push_str(&format!("(deny file-write* (literal {quoted}))\n"));
    out.push_str(&format!("(deny file-write* (subpath {quoted}))\n"));
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
        out.push_str(&format!("(deny {sub} (subpath {quoted}))\n"));
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

    #[cfg(target_os = "macos")]
    static SANDBOX_EXEC_TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

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

    #[test]
    fn hermetic_profile_omits_broad_host_temp_write_rules() {
        let root = tempfile::tempdir().unwrap();
        let config = SandboxConfig {
            deny_host_temp: true,
            ..SandboxConfig::default()
        };
        let profile = seatbelt_profile_with_protected_paths(
            SandboxPolicy::Workspace,
            &[root.path()],
            &config,
            &[],
        );
        let canonical_root = root.path().canonicalize().unwrap();
        assert!(profile.contains(canonical_root.to_str().unwrap()));
        for temp in temp_roots() {
            if Path::new(&temp) != canonical_root {
                assert!(
                    !profile.contains(&format!("(allow file-write* (subpath {}))", quote(&temp))),
                    "hermetic profile exposed host temp {temp}: {profile}"
                );
            }
        }
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

    #[cfg(target_os = "macos")]
    #[test]
    fn sandbox_wrap_resolves_relative_programs_to_absolute_paths() {
        let dir = std::env::temp_dir().join(format!("hi-sb-which-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let profile = SandboxProfile::new(SandboxPolicy::Workspace, &[dir.as_path()]);
        let (prog, args) = profile.wrap_program(OsStr::new("sh"), [] as [&OsStr; 0]);
        assert_eq!(prog, "sandbox-exec");
        let wrapped = args.last().expect("sandbox-exec argv includes the program");
        assert!(
            Path::new(wrapped).is_absolute(),
            "relative sh must be resolved, got {wrapped:?}"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn pipe_wrap_candidates_never_use_path_or_writable_siblings() {
        let writable = PathBuf::from("/work/project");
        let candidates = pipe_wrap_candidates(
            Some(OsStr::new("pipe-wrap")),
            Some(Path::new("/work/project/target/debug/hi")),
            std::slice::from_ref(&writable),
        );
        assert!(
            candidates.is_empty(),
            "relative HI_PIPE_WRAP and a workspace sibling are untrusted"
        );

        let candidates = pipe_wrap_candidates(
            Some(OsStr::new("/operator/pipe-wrap")),
            Some(Path::new("/opt/hi/bin/hi")),
            std::slice::from_ref(&writable),
        );
        assert_eq!(
            candidates,
            [
                PathBuf::from("/operator/pipe-wrap"),
                PathBuf::from("/opt/hi/bin/pipe-wrap")
            ]
        );
        assert!(
            !candidates.contains(&PathBuf::from("/untrusted/path/pipe-wrap")),
            "PATH is deliberately not an input to candidate discovery"
        );

        assert!(
            pipe_wrap_candidates(None, Some(Path::new("/tmp/hi")), &[]).is_empty(),
            "the sandbox can replace a sibling in its writable temp mount"
        );
    }

    #[test]
    fn linux_mount_plan_reprotects_shared_caches_after_broad_bind() {
        let temp = tempfile::tempdir().unwrap();
        let home = temp.path().join("home");
        let cargo = home.join(".cargo");
        let rustup = home.join(".rustup");
        let npm = home.join(".npm");
        let pip = home.join(".cache/pip");
        let uv = home.join(".cache/uv");
        let go = home.join("go/pkg");
        std::fs::create_dir_all(&cargo).unwrap();
        std::fs::create_dir_all(&rustup).unwrap();
        std::fs::create_dir_all(&npm).unwrap();
        std::fs::create_dir_all(&pip).unwrap();
        std::fs::create_dir_all(&uv).unwrap();
        std::fs::create_dir_all(&go).unwrap();

        let args = pipe_wrap_arguments_with_protected_roots(
            SandboxPolicy::Workspace,
            &SandboxConfig::default(),
            std::slice::from_ref(&home),
            &[
                cargo.clone(),
                rustup.clone(),
                npm.clone(),
                pip.clone(),
                uv.clone(),
                go.clone(),
            ],
            OsStr::new("true"),
            &[],
        );
        let bind_position = |flag: &str, path: &Path| {
            args.windows(3).position(|window| {
                window[0] == OsStr::new(flag)
                    && window[1] == path.as_os_str()
                    && window[2] == path.as_os_str()
            })
        };
        let broad = bind_position("--bind", &home).expect("broad home bind");
        for protected in [&cargo, &rustup, &npm, &pip, &uv, &go] {
            let readonly = bind_position("--ro-bind", protected)
                .unwrap_or_else(|| panic!("missing read-only overlay for {protected:?}: {args:?}"));
            assert!(
                readonly > broad,
                "read-only overlay must follow and override the broad bind"
            );
        }
    }

    #[test]
    fn linux_mount_plan_uses_private_temp_for_absent_protected_paths() {
        let protected = PathBuf::from(format!(
            "/tmp/hi-sandbox-absent-{}/nested/cache",
            std::process::id()
        ));
        assert!(!protected.exists(), "test path must start absent");
        let args = pipe_wrap_arguments_with_protected_roots(
            SandboxPolicy::Workspace,
            &SandboxConfig::default(),
            &[],
            std::slice::from_ref(&protected),
            OsStr::new("true"),
            &[],
        );
        assert!(
            args.windows(2).any(|window| {
                window[0] == OsStr::new("--tmpfs") && window[1] == OsStr::new("/tmp")
            }),
            "Linux temp must be namespace-private: {args:?}"
        );
        assert!(
            !args.windows(3).any(|window| {
                window[0] == OsStr::new("--bind")
                    && window[1] == OsStr::new("/tmp")
                    && window[2] == OsStr::new("/tmp")
            }),
            "host /tmp must never be exposed as a broad writable bind: {args:?}"
        );
        assert!(
            !args
                .iter()
                .any(|argument| argument == protected.as_os_str()),
            "an absent host path needs no mount inside a private /tmp"
        );
        assert!(
            !protected.exists(),
            "building the plan must not touch the host"
        );

        let external = PathBuf::from(format!(
            "/var/lib/hi-sandbox-absent-{}/cache",
            std::process::id()
        ));
        let args = pipe_wrap_arguments_with_protected_roots(
            SandboxPolicy::Workspace,
            &SandboxConfig::default(),
            &[],
            std::slice::from_ref(&external),
            OsStr::new("true"),
            &[],
        );
        assert!(
            !args.iter().any(|argument| argument == external.as_os_str()),
            "ordinary absent external paths must not be materialized"
        );

        let project = PathBuf::from(format!("/tmp/hi-sandbox-project-{}", std::process::id()));
        std::fs::create_dir_all(&project).unwrap();
        let args = pipe_wrap_arguments_with_protected_roots(
            SandboxPolicy::Workspace,
            &SandboxConfig::default(),
            std::slice::from_ref(&project),
            &[],
            OsStr::new("true"),
            &[],
        );
        let tmpfs = args
            .windows(2)
            .position(|window| {
                window[0] == OsStr::new("--tmpfs") && window[1] == OsStr::new("/tmp")
            })
            .expect("private temp mount");
        let project_bind = args
            .windows(3)
            .position(|window| {
                window[0] == OsStr::new("--bind")
                    && window[1] == project.as_os_str()
                    && window[2] == project.as_os_str()
            })
            .expect("explicit project bind");
        assert!(
            project_bind > tmpfs,
            "an explicit /tmp project must be restored after private temp"
        );
        std::fs::remove_dir_all(project).unwrap();
    }

    #[test]
    fn overbroad_writable_roots_that_contain_shared_caches_fail_closed() {
        let home = PathBuf::from("/users/alice");
        let protected = vec![
            home.join(".cargo"),
            home.join(".npm"),
            home.join(".cache/pip"),
            home.join("go/pkg"),
        ];
        assert!(writable_root_exposes_protected_path(&home, &protected));
        assert!(writable_root_exposes_protected_path(
            &home.join(".cache"),
            &protected
        ));
        assert!(writable_root_exposes_protected_path(
            Path::new("/"),
            &protected
        ));
        assert!(
            writable_root_exposes_protected_path(&home.join(".cargo/project"), &protected),
            "a writable slice nested inside protected state is equally unsafe"
        );
        assert!(
            writable_root_exposes_protected_path(&home.join(".cargo"), &protected),
            "the exact protected root must be rejected"
        );
        assert!(
            !writable_root_exposes_protected_path(&home.join("projects/repo"), &protected),
            "a normal project root does not contain a shared cache"
        );
    }

    #[test]
    fn config_data_legacy_and_explicit_control_paths_are_protected() {
        let home = PathBuf::from("/users/alice");
        let defaults = vec![
            home.join(".config/hi"),
            home.join(".local/share/hi"),
            home.join(".local/state/hi"),
            home.join(".hi"),
        ];
        for path in &defaults {
            assert!(writable_root_exposes_protected_path(path, &defaults));
            assert!(writable_root_exposes_protected_path(&home, &defaults));
        }

        let cwd = Path::new("/work/repo");
        let sensitive = sensitive_paths_from_inputs(
            Some(PathBuf::from("/config")),
            Some(PathBuf::from("/data")),
            Some(PathBuf::from("relative/trust.toml")),
            Some(PathBuf::from("/rules/me.md")),
            Some(cwd),
        );
        assert!(sensitive.contains(&PathBuf::from("/config/hi")));
        assert!(sensitive.contains(&PathBuf::from("/data/hi")));
        assert!(sensitive.contains(&cwd.join("relative/trust.toml")));
        assert!(sensitive.contains(&PathBuf::from("/rules/me.md")));
    }

    #[test]
    fn configured_cache_inside_normal_workspace_remains_writable() {
        let temp = tempfile::tempdir().unwrap();
        let home = temp.path().join("home");
        let workspace = home.join("projects/repo");
        let project_cache = workspace.join(".tool-cache/cargo");
        let outside_cache = home.join("custom-cache");
        std::fs::create_dir_all(&project_cache).unwrap();
        std::fs::create_dir_all(&outside_cache).unwrap();
        let defaults = vec![home.join(".cargo"), home.join(".npm")];
        let protected = scoped_shared_cache_protected_paths(
            defaults.clone(),
            vec![project_cache.clone(), outside_cache.clone()],
            &[workspace.as_path()],
        );

        assert!(protected.contains(&defaults[0]));
        assert!(protected.contains(&outside_cache));
        assert!(
            !protected.contains(&project_cache),
            "an explicitly configured project-local cache is in workspace scope"
        );
        assert!(
            !writable_root_exposes_protected_path(&workspace, &protected),
            "the project root must not be filtered because of its local cache"
        );

        let profile = seatbelt_profile_with_protected_paths(
            SandboxPolicy::Workspace,
            &[workspace.as_path()],
            &SandboxConfig::default(),
            &protected,
        );
        let project_cache_deny = format!(
            "(deny file-write* (subpath {}))",
            quote(project_cache.to_string_lossy().as_ref())
        );
        assert!(
            !profile.contains(&project_cache_deny),
            "project-local cache must not receive a Seatbelt deny"
        );
        let args = pipe_wrap_arguments_with_protected_roots(
            SandboxPolicy::Workspace,
            &SandboxConfig::default(),
            std::slice::from_ref(&workspace),
            &protected,
            OsStr::new("true"),
            &[],
        );
        assert!(
            !args.windows(3).any(|window| {
                window[0] == OsStr::new("--ro-bind")
                    && window[1] == project_cache.as_os_str()
                    && window[2] == project_cache.as_os_str()
            }),
            "project-local cache must not receive a Linux read-only overlay"
        );
    }

    #[test]
    fn home_workspace_is_removed_from_profile_writable_roots() {
        let home = PathBuf::from(std::env::var_os("HOME").expect("HOME"));
        let profile = SandboxProfile::new(SandboxPolicy::Workspace, &[home.as_path()]);

        #[cfg(target_os = "linux")]
        assert!(
            !profile
                .writable_roots
                .iter()
                .any(|root| resolve_path_for_comparison(root) == resolve_path_for_comparison(&home)),
            "HOME must fail closed instead of exposing absent shared caches"
        );
        #[cfg(target_os = "macos")]
        {
            let home = resolve_path_for_comparison(&home);
            let allow = format!(
                "(allow file-write* (subpath {}))",
                quote(home.to_string_lossy().as_ref())
            );
            assert!(
                !profile.profile.contains(&allow),
                "HOME must not receive a broad Seatbelt write allow"
            );
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        let _ = (home, profile);
    }

    #[test]
    fn shared_hi_state_root_is_not_implicitly_writable() {
        let state = crate::checkpoint::default_state_root();
        let profile = SandboxProfile::new(SandboxPolicy::Workspace, &[]);

        #[cfg(target_os = "linux")]
        assert!(
            !profile.writable_roots.iter().any(
                |root| resolve_path_for_comparison(root) == resolve_path_for_comparison(&state)
            ),
            "shared hi state must not be exposed to sandboxed commands"
        );
        #[cfg(target_os = "macos")]
        {
            let state = resolve_path_for_comparison(&state);
            let allow = format!(
                "(allow file-write* (subpath {}))",
                quote(state.to_string_lossy().as_ref())
            );
            assert!(
                !profile.profile.contains(&allow),
                "shared hi state must not receive a Seatbelt write allow"
            );
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        let _ = (state, profile);
    }

    #[test]
    fn seatbelt_reprotects_shared_cache_subpaths_after_broad_allow() {
        let temp = tempfile::tempdir().unwrap();
        let home = temp.path().join("home");
        std::fs::create_dir_all(&home).unwrap();
        let protected = [
            home.join(".cargo"),
            home.join(".rustup"),
            home.join(".npm"),
            home.join(".cache/pip"),
            home.join(".cache/uv"),
            home.join("go/pkg"),
        ];
        let profile = seatbelt_profile_with_protected_paths(
            SandboxPolicy::Workspace,
            &[home.as_path()],
            &SandboxConfig::default(),
            &protected,
        );
        let home = home.canonicalize().unwrap();
        let allow = format!(
            "(allow file-write* (subpath {}))",
            quote(home.to_string_lossy().as_ref())
        );
        let allow_position = profile.find(&allow).expect("broad home allow");
        for path in protected {
            let literal_deny = format!(
                "(deny file-write* (literal {}))",
                quote(path.to_string_lossy().as_ref())
            );
            let subpath_deny = format!(
                "(deny file-write* (subpath {}))",
                quote(path.to_string_lossy().as_ref())
            );
            for deny in [literal_deny, subpath_deny] {
                let deny_position = profile
                    .find(&deny)
                    .unwrap_or_else(|| panic!("missing shared-cache deny for {path:?}: {profile}"));
                assert!(
                    deny_position > allow_position,
                    "cache literal/subpath deny must follow the broad allow"
                );
            }
        }
    }

    #[test]
    fn shared_home_caches_and_hi_control_roots_are_not_implicit_writable_roots() {
        let home = PathBuf::from(std::env::var_os("HOME").expect("HOME"));
        let shared_caches = [
            home.join(".cargo"),
            home.join(".rustup"),
            home.join(".npm"),
            home.join(".config/hi"),
            home.join(".cache/pip"),
            home.join(".cache/uv"),
            home.join(".cache/go-build"),
            home.join(".hi"),
            home.join(".local/share/hi"),
            home.join(".local/state/hi"),
            home.join("Library/Caches/pip"),
            home.join("Library/Caches/uv"),
            home.join("go/pkg"),
            home.join("Library/Caches/go-build"),
        ];
        let profile = SandboxProfile::new(SandboxPolicy::Workspace, &[]);

        #[cfg(target_os = "linux")]
        for cache in &shared_caches {
            assert!(
                !profile.writable_roots.iter().any(|root| root == cache),
                "shared cache must not be writable by default: {cache:?}"
            );
        }
        #[cfg(target_os = "macos")]
        for cache in &shared_caches {
            let allow = format!(
                "(allow file-write* (subpath {}))",
                quote(cache.to_string_lossy().as_ref())
            );
            assert!(
                !profile.profile.contains(&allow),
                "shared cache must not be writable by default: {cache:?}"
            );
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        let _ = (profile, shared_caches);
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
        let _sandbox_guard = SANDBOX_EXEC_TEST_LOCK.lock().await;
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
        assert!(
            out.status.success(),
            "write inside workspace must succeed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
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

    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn workspace_sandbox_denies_writes_to_shared_cargo_home() {
        use std::process::Command;
        if std::env::var_os(NESTED_SANDBOX_ENV).is_some() {
            eprintln!("skipped: already inside an hi sandbox — macOS denies nested sandbox_apply");
            return;
        }
        let _sandbox_guard = SANDBOX_EXEC_TEST_LOCK.lock().await;
        let ws = std::env::temp_dir().join(format!("hi-sb-cargo-{}", std::process::id()));
        std::fs::create_dir_all(&ws).unwrap();
        let home = std::env::var("HOME").expect("HOME set on macOS");
        let cargo = Path::new(&home).join(".cargo");
        std::fs::create_dir_all(&cargo).unwrap();
        let probe = cargo.join(format!("hi-sb-probe-{}", std::process::id()));
        let _ = std::fs::remove_file(&probe);
        let profile = SandboxProfile::new(SandboxPolicy::Workspace, &[ws.as_path()]);
        let (prog, args) = profile.wrap(&format!("echo ok > {}", probe.display()));
        let out = Command::new(prog).args(args).output().unwrap();
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            !out.status.success(),
            "write to shared cargo home must fail under workspace sandbox"
        );
        assert!(
            !stderr.contains("sandbox_apply"),
            "Seatbelt profile must apply before testing the denial: {stderr}"
        );
        assert!(!probe.exists(), "sandbox must not create the probe file");
        let _ = std::fs::remove_file(&probe);
        std::fs::remove_dir_all(&ws).unwrap();
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
        let _sandbox_guard = SANDBOX_EXEC_TEST_LOCK.lock().await;
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
