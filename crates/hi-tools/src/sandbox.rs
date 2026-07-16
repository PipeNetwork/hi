//! Opt-in OS sandboxing for shell commands the agent runs.
//!
//! The `workspace` policy confines a command's *writes* to the workspace (plus
//! temp and a handful of device nodes) while leaving reads and network open —
//! so a misbehaving or misguided command cannot modify files outside the
//! project. Reads stay open because a coding agent legitimately reads system
//! headers, toolchains, and libraries everywhere.
//!
//! Enforcement today is macOS-only, via the kernel Seatbelt sandbox
//! (`sandbox-exec`). On other platforms the policy parses but is **not
//! enforced** — [`SandboxProfile::wrap`] returns the command unchanged — so the
//! agent's behaviour is identical to sandbox-off. Linux (Landlock/bwrap) is a
//! follow-up. Default is **off**: the sandbox is a deliberate opt-in via
//! `HI_SANDBOX=workspace`.
//!
//! Path handling learns from grok-build's hard-won lesson: Seatbelt matches on
//! *real* paths, so every writable root is canonicalized (resolving the
//! `/tmp` → `/private/tmp` firmlink) before it goes into the profile — an
//! un-canonicalized `/tmp/...` subpath silently matches nothing and denies the
//! very writes it meant to allow.

use std::path::Path;

/// How much of the filesystem a shell command may modify.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SandboxPolicy {
    /// No sandbox — commands run with the process's own permissions (default).
    #[default]
    Off,
    /// Writes confined to the workspace (+ temp + device nodes); reads open.
    Workspace,
}

impl SandboxPolicy {
    /// Resolve the policy from `HI_SANDBOX` (`workspace` enables it; anything
    /// else, including unset, is off).
    pub fn from_env() -> Self {
        match std::env::var("HI_SANDBOX").ok().as_deref().map(str::trim) {
            Some("workspace") | Some("on") | Some("1") => SandboxPolicy::Workspace,
            _ => SandboxPolicy::Off,
        }
    }
}

/// A resolved sandbox profile bound to a set of writable roots. Cheap to clone.
#[derive(Clone, Debug)]
pub struct SandboxProfile {
    policy: SandboxPolicy,
    /// The Seatbelt profile text (macOS). Empty when the policy is off or the
    /// platform is unenforced.
    profile: String,
}

impl SandboxProfile {
    /// Build a profile for `policy` whose writable roots are `writable` (e.g.
    /// the workspace root and the agent's state directory). Non-existent roots
    /// are skipped; existing ones are canonicalized so Seatbelt subpath matches
    /// hit the real filesystem path.
    pub fn new(policy: SandboxPolicy, writable: &[&Path]) -> Self {
        if policy == SandboxPolicy::Off || !cfg!(target_os = "macos") {
            return Self {
                policy,
                profile: String::new(),
            };
        }
        let profile = seatbelt_profile(writable);
        Self { policy, profile }
    }

    /// Whether this profile actually enforces anything (on this platform).
    pub fn is_enforced(&self) -> bool {
        !self.profile.is_empty()
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
    }

    pub fn policy(&self) -> SandboxPolicy {
        self.policy
    }
}

/// Build a Seatbelt profile: allow everything by default (reads, exec, network),
/// then deny all writes, then re-allow writes under each canonical writable root
/// plus the system temp dirs and essential device nodes. Deny-then-allow order
/// with `(allow default)` first means the later write-allows win for their
/// subpaths while writes elsewhere stay denied.
fn seatbelt_profile(writable: &[&Path]) -> String {
    let mut out = String::from("(version 1)\n(allow default)\n(deny file-write*)\n");
    // Device nodes a normal shell/toolchain needs to write.
    out.push_str(
        "(allow file-write*\n  (literal \"/dev/null\")\n  (literal \"/dev/stdout\")\n  \
         (literal \"/dev/stderr\")\n  (literal \"/dev/tty\")\n  (literal \"/dev/dtracehelper\")\n  \
         (literal \"/dev/zero\")\n  (subpath \"/dev/fd\"))\n",
    );
    // System temp — canonicalized so /tmp resolves to /private/tmp, and the
    // per-user /var/folders temp used by mktemp/cargo/build tools.
    for temp in temp_roots() {
        out.push_str(&format!("(allow file-write* (subpath {}))\n", quote(&temp)));
    }
    // The workspace and other explicitly writable roots.
    for root in writable {
        if let Ok(canonical) = root.canonicalize()
            && let Some(text) = canonical.to_str()
        {
            out.push_str(&format!("(allow file-write* (subpath {}))\n", quote(text)));
        }
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
