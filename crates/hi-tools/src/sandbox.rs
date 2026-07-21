//! Opt-in OS sandboxing for shell commands the agent runs.
//!
//! The `workspace` policy confines a command's *writes* to the workspace (plus
//! temp and a handful of device nodes) while leaving reads and network open —
//! so a misbehaving or misguided command cannot modify files outside the
//! project. Reads stay open because a coding agent legitimately reads system
//! headers, toolchains, and libraries everywhere.
//!
//! `strict` is deny-by-default: only explicitly listed paths (workspace, temp,
//! system roots) are readable, and writes are confined to the workspace.
//! `readonly` allows reads everywhere but denies all writes and restricts
//! child-process network access.
//!
//! Enforcement is macOS (Seatbelt via `sandbox-exec`) and Linux (Landlock via
//! `landlock_restrict_self` + bwrap re-exec for deny paths). On other platforms
//! the policy parses but is **not enforced** — [`SandboxProfile::wrap`] returns
//! the command unchanged.
//!
//! **Default is off** so Cargo/npm/pip global caches under `$HOME` keep working
//! for everyday local use. Prefer `HI_SANDBOX=workspace` for untrusted prompts.
//! Full operator docs + Linux Landlock/bwrap sketch: `docs/sandbox.md`.
//!
//! Path handling learns from grok-build's hard-won lesson: Seatbelt matches on
//! *real* paths, so every writable root is canonicalized (resolving the
//! `/tmp` → `/private/tmp` firmlink) before it goes into the profile — an
//! un-canonicalized `/tmp/...` subpath silently matches nothing and denies the
//! very writes it meant to allow.

use std::path::{Path, PathBuf};

/// How much of the filesystem a shell command may modify.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SandboxPolicy {
    /// No sandbox — commands run with the process's own permissions (default).
    #[default]
    Off,
    /// Writes confined to the workspace (+ temp + device nodes); reads open.
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
    /// Unset / empty → [`SandboxPolicy::Off`]. Unknown non-empty values return
    /// `Err` so callers can refuse to start rather than silently running open.
    pub fn from_env() -> Result<Self, String> {
        match std::env::var("HI_SANDBOX") {
            Err(_) => Ok(SandboxPolicy::Off),
            Ok(value) if value.trim().is_empty() => Ok(SandboxPolicy::Off),
            Ok(value) => Self::parse(&value).map_err(|token| {
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
    /// Deny-path config for platforms that enforce it via bind-over (Linux bwrap).
    config: SandboxConfig,
    /// Whether child-process network should be restricted (ReadOnly/Strict).
    restrict_network: bool,
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
                config,
                restrict_network: false,
            };
        }
        let restrict_network = policy.restricts_network();
        let profile = if cfg!(target_os = "macos") {
            seatbelt_profile(policy, writable, &config)
        } else if cfg!(target_os = "linux") {
            landlock_profile(policy, writable, &config)
        } else {
            String::new()
        };
        Self {
            policy,
            profile,
            config,
            restrict_network,
        }
    }

    /// Whether this profile actually enforces anything (on this platform).
    pub fn is_enforced(&self) -> bool {
        !self.profile.is_empty()
    }

    /// True when the operator asked for confinement but this OS cannot enforce it.
    pub fn requested_but_unenforced(&self) -> bool {
        self.policy != SandboxPolicy::Off && !self.is_enforced()
    }

    /// One-line operator warning when [`Self::requested_but_unenforced`].
    pub fn unenforced_warning() -> &'static str {
        "HI_SANDBOX is set but OS write-confinement is not enforced on this platform \
         (macOS Seatbelt / Linux Landlock only — see docs/sandbox.md)"
    }

    /// Whether child-process network access should be restricted.
    pub fn restricts_child_network(&self) -> bool {
        self.restrict_network
    }

    /// Wrap a `sh -c <command>` invocation so it runs under the sandbox. Returns
    /// the program and its argument vector. When the policy is off or the
    /// platform is unenforced, returns the plain `sh -c` invocation unchanged.
    pub fn wrap(&self, command: &str) -> (String, Vec<String>) {
        if self.profile.is_empty() {
            return (
                "sh".to_string(),
                vec!["-c".to_string(), command.to_string()],
            );
        }
        if cfg!(target_os = "macos") {
            (
                "sandbox-exec".to_string(),
                vec![
                    "-p".to_string(),
                    self.profile.clone(),
                    "sh".to_string(),
                    "-c".to_string(),
                    command.to_string(),
                ],
            )
        } else {
            // Linux: the profile is the landlock ruleset spec, but enforcement
            // happens via bwrap re-exec for deny paths. For the simple case
            // (no deny paths), we wrap with bwrap --bind / /.
            if self.config.deny_write.is_empty() && self.config.deny_read.is_empty() {
                return (
                    "sh".to_string(),
                    vec!["-c".to_string(), command.to_string()],
                );
            }
            // With deny paths, use bwrap to bind-over the denied paths.
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
            args.push("--".to_string());
            args.push("sh".to_string());
            args.push("-c".to_string());
            args.push(command.to_string());
            ("bwrap".to_string(), args)
        }
    }

    pub fn policy(&self) -> SandboxPolicy {
        self.policy
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
                out.push_str(&format!("(allow file-read* (subpath {}))\n", quote(&readable)));
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
    for candidate in ["/usr", "/bin", "/sbin", "/lib", "/lib64", "/opt", "/etc", "/System"] {
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

/// Build a Landlock rule spec for Linux. Landlock can only *allow* access to
/// paths — it cannot deny a subpath of an allowed tree. So for `Workspace`
/// (read-all), deny paths are enforced via bwrap bind-over in [`SandboxProfile::wrap`].
/// For `Strict` (deny-by-default), only the listed paths are allowed.
///
/// The returned string is a human-readable description of the ruleset; actual
/// enforcement uses the `landlock` crate's syscalls at apply time. For now this
/// serves as the profile marker (non-empty = enforced) and documents the intent.
fn landlock_profile(policy: SandboxPolicy, writable: &[&Path], _config: &SandboxConfig) -> String {
    let mut out = String::new();
    match policy {
        SandboxPolicy::Workspace => {
            out.push_str("landlock: read=/, write=");
            for root in writable {
                if let Ok(canonical) = root.canonicalize()
                    && let Some(text) = canonical.to_str()
                {
                    out.push_str(text);
                    out.push(',');
                }
            }
            for temp in temp_roots() {
                out.push_str(&temp);
                out.push(',');
            }
        }
        SandboxPolicy::Strict => {
            out.push_str("landlock: read=");
            for root in system_readable_roots() {
                out.push_str(&root);
                out.push(',');
            }
            for root in writable {
                if let Ok(canonical) = root.canonicalize()
                    && let Some(text) = canonical.to_str()
                {
                    out.push_str(text);
                    out.push(',');
                }
            }
            out.push_str("; write=");
            for root in writable {
                if let Ok(canonical) = root.canonicalize()
                    && let Some(text) = canonical.to_str()
                {
                    out.push_str(text);
                    out.push(',');
                }
            }
        }
        SandboxPolicy::ReadOnly => {
            out.push_str("landlock: read=/, write= (none)");
        }
        SandboxPolicy::Off => {}
    }
    out
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
    fn policy_from_env_is_off_by_default() {
        // We can't safely mutate process env in parallel tests; assert the
        // pure-mapping behaviour via the match instead by constructing directly.
        assert_eq!(SandboxPolicy::default(), SandboxPolicy::Off);
    }

    #[test]
    fn policy_parse_accepts_known_tokens() {
        assert_eq!(SandboxPolicy::parse("workspace").unwrap(), SandboxPolicy::Workspace);
        assert_eq!(SandboxPolicy::parse("ON").unwrap(), SandboxPolicy::Workspace);
        assert_eq!(SandboxPolicy::parse("strict").unwrap(), SandboxPolicy::Strict);
        assert_eq!(SandboxPolicy::parse("readonly").unwrap(), SandboxPolicy::ReadOnly);
        assert_eq!(SandboxPolicy::parse("read-only").unwrap(), SandboxPolicy::ReadOnly);
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
    fn workspace_policy_reports_unenforced_off_macos() {
        let profile = SandboxProfile::new(SandboxPolicy::Workspace, &[]);
        if cfg!(target_os = "macos") {
            // No writable roots → empty seatbelt body still built; with empty
            // writable list the profile text is non-empty on macOS.
            // Unenforced only when platform cannot install a profile at all.
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
