//! Trusted capability and filesystem enforcement for candidate tool requests.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Component, Path, PathBuf},
    process::Stdio,
    time::Duration,
};

use anyhow::{Context, Result, bail, ensure};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
#[cfg(unix)]
use std::os::unix::fs::FileTypeExt;
use tokio::{process::Command, time::timeout};

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SideEffect {
    None,
    WorkspaceRead,
    WorkspaceWrite,
    Process,
    Network,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ToolDescriptor {
    pub name: String,
    pub input_schema: Value,
    pub output_schema: Value,
    pub required_capabilities: BTreeSet<String>,
    pub side_effect: SideEffect,
    pub maximum_output_bytes: u64,
    pub timeout_ms: u64,
    pub replayable: bool,
}

#[derive(Clone, Debug)]
pub struct ToolContext {
    pub run_id: String,
    pub candidate_id: String,
    pub stage: String,
    pub workspace: PathBuf,
    pub capabilities: BTreeSet<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ToolResponse {
    pub succeeded: bool,
    pub output: Value,
    pub stdout_hash: Option<String>,
    pub stderr_hash: Option<String>,
    pub truncated: bool,
}

#[async_trait]
pub trait ToolExecutor: Send + Sync {
    fn descriptor(&self) -> &ToolDescriptor;
    async fn execute(&self, context: &ToolContext, request: Value) -> Result<ToolResponse>;
}

#[derive(Default)]
pub struct ToolHost {
    executors: BTreeMap<String, Box<dyn ToolExecutor>>,
}

impl ToolHost {
    pub fn register(&mut self, executor: Box<dyn ToolExecutor>) -> Result<()> {
        let descriptor = executor.descriptor();
        validate_descriptor(descriptor)?;
        ensure!(
            !self.executors.contains_key(&descriptor.name),
            "duplicate tool {}",
            descriptor.name
        );
        self.executors.insert(descriptor.name.clone(), executor);
        Ok(())
    }

    pub fn descriptors(&self) -> Vec<ToolDescriptor> {
        self.executors
            .values()
            .map(|e| e.descriptor().clone())
            .collect()
    }

    pub async fn execute(
        &self,
        name: &str,
        context: &ToolContext,
        request: Value,
    ) -> Result<ToolResponse> {
        ensure!(
            serde_json::to_vec(&request)?.len() <= 256 * 1024,
            "tool request exceeds 256 KiB"
        );
        let executor = self
            .executors
            .get(name)
            .with_context(|| format!("unknown tool {name}"))?;
        let descriptor = executor.descriptor();
        ensure!(
            descriptor
                .required_capabilities
                .is_subset(&context.capabilities),
            "tool capability denied"
        );
        timeout(
            Duration::from_millis(descriptor.timeout_ms),
            executor.execute(context, request),
        )
        .await
        .context("tool deadline exceeded")?
    }
}

fn validate_descriptor(descriptor: &ToolDescriptor) -> Result<()> {
    ensure!(
        !descriptor.name.is_empty() && descriptor.name.len() <= 128,
        "invalid tool name"
    );
    ensure!(
        descriptor.maximum_output_bytes > 0 && descriptor.maximum_output_bytes <= 64 * 1024 * 1024,
        "invalid tool output limit"
    );
    ensure!(
        descriptor.timeout_ms > 0 && descriptor.timeout_ms <= 15 * 60 * 1_000,
        "invalid tool timeout"
    );
    if matches!(
        descriptor.side_effect,
        SideEffect::WorkspaceWrite | SideEffect::Process | SideEffect::Network
    ) {
        ensure!(
            !descriptor.replayable,
            "side-effecting tools require recorded mutation artifacts, not response replay"
        );
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct PathPolicy {
    root: PathBuf,
    protected: Vec<PathBuf>,
    allow_lockfiles: bool,
}

impl PathPolicy {
    pub fn new(root: &Path, protected: Vec<PathBuf>, allow_lockfiles: bool) -> Result<Self> {
        let root = root.canonicalize().context("canonicalizing workspace")?;
        ensure!(root.is_dir(), "workspace is not a directory");
        Ok(Self {
            root,
            protected,
            allow_lockfiles,
        })
    }

    pub fn authorize(&self, relative: &Path, mutation: bool) -> Result<PathBuf> {
        ensure!(
            !relative.as_os_str().is_empty() && !relative.is_absolute(),
            "tool path must be relative"
        );
        ensure!(
            relative
                .components()
                .all(|c| matches!(c, Component::Normal(_))),
            "path traversal rejected"
        );
        if mutation && !self.allow_lockfiles {
            ensure!(
                !matches!(
                    relative.file_name().and_then(|n| n.to_str()),
                    Some("Cargo.lock" | "package-lock.json" | "pnpm-lock.yaml" | "yarn.lock")
                ),
                "lockfile mutation is not authorized"
            );
        }
        let target = self.root.join(relative);
        ensure!(
            !self.protected.iter().any(|p| target.starts_with(p)),
            "protected path denied"
        );
        let mut cursor = self.root.clone();
        for component in relative.components() {
            if let Component::Normal(name) = component {
                cursor.push(name);
                match fs::symlink_metadata(&cursor) {
                    Ok(meta) => {
                        ensure!(!meta.file_type().is_symlink(), "symlink path rejected");
                        ensure!(
                            !meta.file_type().is_fifo() && !meta.file_type().is_socket(),
                            "special file rejected"
                        );
                        #[cfg(unix)]
                        {
                            use std::os::unix::fs::{FileTypeExt, MetadataExt};
                            ensure!(
                                !meta.file_type().is_block_device()
                                    && !meta.file_type().is_char_device(),
                                "device path rejected"
                            );
                            let root_dev = fs::metadata(&self.root)?.dev();
                            ensure!(meta.dev() == root_dev, "mount boundary escape rejected");
                        }
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => break,
                    Err(error) => return Err(error.into()),
                }
            }
        }
        if let Some(parent) = target.parent() {
            let existing = nearest_existing(parent)?;
            let canonical = existing.canonicalize()?;
            ensure!(
                canonical.starts_with(&self.root),
                "workspace escape rejected"
            );
        }
        Ok(target)
    }
}

fn nearest_existing(path: &Path) -> Result<&Path> {
    let mut current = path;
    while !current.exists() {
        current = current.parent().context("path has no existing parent")?;
    }
    Ok(current)
}

#[derive(Clone, Debug)]
pub struct ShellPolicy {
    pub allowed_programs: BTreeSet<String>,
    pub maximum_output_bytes: usize,
    pub timeout: Duration,
}

impl ShellPolicy {
    pub async fn run(
        &self,
        workspace: &Path,
        program: &str,
        arguments: &[String],
    ) -> Result<ToolResponse> {
        ensure!(
            self.allowed_programs.contains(program),
            "shell program denied"
        );
        ensure!(
            arguments.len() <= 128
                && arguments
                    .iter()
                    .all(|v| v.len() <= 16 * 1024 && !v.contains('\0')),
            "invalid shell arguments"
        );
        let mut command = Command::new(program);
        command
            .args(arguments)
            .current_dir(workspace)
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("HOME", "/nonexistent")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let output = timeout(self.timeout, command.output())
            .await
            .context("shell command deadline exceeded")??;
        let (stdout, stdout_truncated) = bounded(&output.stdout, self.maximum_output_bytes);
        let (stderr, stderr_truncated) = bounded(&output.stderr, self.maximum_output_bytes);
        Ok(ToolResponse {
            succeeded: output.status.success(),
            output: serde_json::json!({"exit_code": output.status.code(), "stdout": String::from_utf8_lossy(stdout), "stderr": String::from_utf8_lossy(stderr)}),
            stdout_hash: Some(blake3::hash(&output.stdout).to_hex().to_string()),
            stderr_hash: Some(blake3::hash(&output.stderr).to_hex().to_string()),
            truncated: stdout_truncated || stderr_truncated,
        })
    }
}

fn bounded(bytes: &[u8], limit: usize) -> (&[u8], bool) {
    if bytes.len() > limit {
        (&bytes[..limit], true)
    } else {
        (bytes, false)
    }
}

pub fn validate_patch_paths(patch: &str, policy: &PathPolicy) -> Result<Vec<PathBuf>> {
    ensure!(
        patch.len() <= 16 * 1024 * 1024,
        "patch exceeds size ceiling"
    );
    let mut paths = Vec::new();
    for line in patch
        .lines()
        .filter(|line| line.starts_with("+++ ") || line.starts_with("--- "))
    {
        let raw = line[4..].split_whitespace().next().unwrap_or("");
        if raw == "/dev/null" {
            continue;
        }
        let raw = raw
            .strip_prefix("a/")
            .or_else(|| raw.strip_prefix("b/"))
            .unwrap_or(raw);
        if raw.is_empty() {
            bail!("patch has empty path");
        }
        paths.push(policy.authorize(Path::new(raw), true)?);
    }
    ensure!(!paths.is_empty(), "patch contains no file paths");
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_traversal_symlinks_and_unauthorized_lockfiles() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("file"), "ok").unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink("/etc/passwd", tmp.path().join("link")).unwrap();
        let policy = PathPolicy::new(tmp.path(), vec![], false).unwrap();
        assert!(policy.authorize(Path::new("../escape"), true).is_err());
        assert!(policy.authorize(Path::new("Cargo.lock"), true).is_err());
        #[cfg(unix)]
        assert!(policy.authorize(Path::new("link"), false).is_err());
        assert!(policy.authorize(Path::new("src/new.rs"), true).is_ok());
    }

    /// Mirror of the interactive `hi-tools` capability → side-effect matrix.
    ///
    /// When registering RSI host tools, descriptors must use these classes so
    /// candidate capabilities stay aligned with workstation tool semantics.
    /// Source of truth for *which tools exist* is `hi_tools::TOOL_CATALOG`;
    /// this table documents the shared side-effect vocabulary.
    fn interactive_tool_side_effect(name: &str) -> Option<SideEffect> {
        match name {
            "update_plan" | "record_decision" => Some(SideEffect::None),
            "read" | "list" | "grep" | "glob" | "repo_map" | "find_symbol" | "diff"
            | "diagnostics" | "definition" | "references" | "hover" => {
                Some(SideEffect::WorkspaceRead)
            }
            "write" | "edit" | "multi_edit" | "apply_patch" | "web_download" => {
                Some(SideEffect::WorkspaceWrite)
            }
            "bash" | "bash_output" | "bash_kill" | "explore" | "delegate" => {
                Some(SideEffect::Process)
            }
            "web_search" | "web_fetch" => Some(SideEffect::Network),
            _ => None,
        }
    }

    #[test]
    fn side_effect_matrix_covers_interactive_catalog_names() {
        let expected = [
            "update_plan",
            "record_decision",
            "read",
            "list",
            "grep",
            "glob",
            "repo_map",
            "find_symbol",
            "diff",
            "diagnostics",
            "definition",
            "references",
            "hover",
            "write",
            "edit",
            "multi_edit",
            "apply_patch",
            "web_download",
            "bash",
            "bash_output",
            "bash_kill",
            "explore",
            "delegate",
            "web_search",
            "web_fetch",
        ];
        for name in expected {
            assert!(
                interactive_tool_side_effect(name).is_some(),
                "missing side-effect mapping for {name}"
            );
        }
    }

    #[test]
    fn side_effecting_tools_cannot_be_marked_replayable() {
        for side_effect in [
            SideEffect::WorkspaceWrite,
            SideEffect::Process,
            SideEffect::Network,
        ] {
            let descriptor = ToolDescriptor {
                name: "sample".into(),
                input_schema: serde_json::json!({}),
                output_schema: serde_json::json!({}),
                required_capabilities: BTreeSet::new(),
                side_effect,
                maximum_output_bytes: 1024,
                timeout_ms: 1000,
                replayable: true,
            };
            assert!(
                validate_descriptor(&descriptor).is_err(),
                "{side_effect:?} must reject replayable"
            );
        }
        let ok = ToolDescriptor {
            name: "sample".into(),
            input_schema: serde_json::json!({}),
            output_schema: serde_json::json!({}),
            required_capabilities: BTreeSet::new(),
            side_effect: SideEffect::WorkspaceRead,
            maximum_output_bytes: 1024,
            timeout_ms: 1000,
            replayable: true,
        };
        assert!(validate_descriptor(&ok).is_ok());
    }

    #[test]
    fn write_process_network_require_non_replayable_in_host_policy() {
        // Pins the shared policy: mutating/network/process host tools always
        // record artifacts (mirrors interactive non-read_only tools).
        assert!(matches!(
            interactive_tool_side_effect("write"),
            Some(SideEffect::WorkspaceWrite)
        ));
        assert!(matches!(
            interactive_tool_side_effect("bash"),
            Some(SideEffect::Process)
        ));
        assert!(matches!(
            interactive_tool_side_effect("web_search"),
            Some(SideEffect::Network)
        ));
        assert!(matches!(
            interactive_tool_side_effect("read"),
            Some(SideEffect::WorkspaceRead)
        ));
    }
}
