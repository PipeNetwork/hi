//! Local tool host: advertise a small catalog and execute via `hi-tools`.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use hi_ai::ToolSpec;
use hi_lsp::LspManager;
use hi_tools::shell_policy::classify_shell_tool_arguments;
use hi_tools::{
    BackgroundRegistry, PlanStep, ProcessRunner, ReadCache, RepoMapCache, TOOL_SPECS, ToolEffects,
    ToolOutcome, ToolStatus, TruncationState, execute_prepared_in_runtime,
    execute_streaming_in_runtime_with_runner, is_filesystem_mutating,
    prepare_mutation_in_with_state,
};

use crate::liveness::tool_fingerprint;
use crate::ui::{ConfirmationRequest, ConfirmationResult, PermissionMode, Ui};

const ADVERTISED: &[&str] = &[
    "read",
    "write",
    "edit",
    "multi_edit",
    "apply_patch",
    "bash",
    "bash_output",
    "bash_kill",
    "list",
    "grep",
    "glob",
    "update_plan",
    "repo_map",
    "diagnostics",
    "definition",
    "references",
    "hover",
];

pub fn advertised_tools() -> Vec<ToolSpec> {
    TOOL_SPECS
        .iter()
        .filter(|spec| ADVERTISED.contains(&spec.name.as_str()))
        .cloned()
        .collect()
}

/// Cloneable handle that can kill in-flight foreground/background tools
/// while a turn future still borrows [`ToolHost`].
#[derive(Clone)]
pub struct ToolInterrupt {
    foreground: hi_tools::ForegroundProcessRegistry,
    background: Arc<BackgroundRegistry>,
}

impl ToolInterrupt {
    pub fn interrupt(&self) {
        self.foreground.kill_current();
        self.background.kill_all();
    }
}

pub struct ToolHost {
    root: PathBuf,
    state_root: PathBuf,
    runner: ProcessRunner,
    lsp: Arc<LspManager>,
    background: Arc<BackgroundRegistry>,
    read_cache: Mutex<ReadCache>,
    repo_map: Mutex<RepoMapCache>,
    liveness: hi_liveness::Publisher,
}

impl ToolHost {
    pub fn new(root: PathBuf, state_root: PathBuf) -> Result<Self> {
        Self::new_with_runner(root.clone(), state_root, ProcessRunner::new(&root)?)
    }

    pub fn new_with_runner(
        root: PathBuf,
        state_root: PathBuf,
        runner: ProcessRunner,
    ) -> Result<Self> {
        std::fs::create_dir_all(&state_root)?;
        let lsp = Arc::new(LspManager::new(&root)?);
        Ok(Self {
            root,
            state_root,
            runner,
            lsp,
            background: Arc::new(BackgroundRegistry::default()),
            read_cache: Mutex::new(ReadCache::new()),
            repo_map: Mutex::new(RepoMapCache::new()),
            liveness: hi_liveness::Publisher::new(),
        })
    }

    pub fn set_liveness(&mut self, liveness: hi_liveness::Publisher) {
        self.liveness = liveness;
    }

    pub fn liveness(&self) -> hi_liveness::Publisher {
        self.liveness.clone()
    }

    pub fn interrupt_handle(&self) -> ToolInterrupt {
        ToolInterrupt {
            foreground: self.runner.foreground_registry(),
            background: Arc::clone(&self.background),
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn state_root(&self) -> &Path {
        &self.state_root
    }

    pub fn interrupt(&self) {
        self.runner.foreground_registry().kill_current();
        self.background.kill_all();
    }

    pub fn reset_bash_repeats(&self) {
        self.runner.reset_bash_repeats();
    }

    pub fn sandbox_enforced(&self) -> bool {
        self.runner.sandbox_enforced()
    }

    pub fn sandbox_backend_name(&self) -> &'static str {
        self.runner.sandbox_backend_name()
    }

    pub(crate) fn runner_handle(&self) -> &ProcessRunner {
        &self.runner
    }

    pub async fn execute(
        &self,
        id: &str,
        name: &str,
        arguments: &str,
        permission: impl Fn() -> PermissionMode,
        ui: &mut dyn Ui,
    ) -> ToolOutcome {
        if let Some(denied) = self
            .confirm_if_needed(name, arguments, permission, ui)
            .await
        {
            return denied;
        }
        self.liveness.note_tool_start(id, name);
        self.liveness
            .note_tool_fingerprint(&tool_fingerprint(name, arguments));
        self.publish_pgids();
        ui.tool_started_id(id, name, arguments);
        let outcome = if matches!(name, "write" | "edit" | "multi_edit" | "apply_patch") {
            match prepare_mutation_in_with_state(&self.root, &self.state_root, name, arguments)
                .await
            {
                Ok(prepared) => {
                    execute_prepared_in_runtime(&self.lsp, &self.read_cache, prepared).await
                }
                Err(error) => failed_outcome(format!("Error: {error:#}")),
            }
        } else {
            let liveness = self.liveness.clone();
            let mut on_line = |line: &str| {
                liveness.note_progress();
                ui.tool_stream(name, line);
            };
            execute_streaming_in_runtime_with_runner(
                &self.runner,
                &self.root,
                &self.state_root,
                &self.lsp,
                self.background.as_ref(),
                &self.read_cache,
                &self.repo_map,
                name,
                arguments,
                &mut on_line,
            )
            .await
        };
        if outcome.status == ToolStatus::Failed {
            self.liveness.note_tool_error(&outcome.content);
        }
        let leaked = !self.runner.detached_descendants_preserved()
            && !run_in_background(name, arguments)
            && self.runner.foreground_registry().active_count() > 0;
        self.liveness.note_tool_end(leaked);
        self.publish_pgids();
        outcome
    }

    fn publish_pgids(&self) {
        let fg = self.runner.foreground_registry().active_pgids();
        let current = fg.first().copied();
        let mut all = fg;
        for (id, _, status) in self.background.snapshot() {
            if status == "running"
                && let Some(pgid) = self.background.os_pid(&id)
                && !all.contains(&pgid)
            {
                all.push(pgid);
            }
        }
        self.liveness.set_child_pgids(all, current);
    }

    async fn confirm_if_needed(
        &self,
        name: &str,
        arguments: &str,
        permission: impl Fn() -> PermissionMode,
        ui: &mut dyn Ui,
    ) -> Option<ToolOutcome> {
        if permission() == PermissionMode::Always {
            return None;
        }
        let request = if is_filesystem_mutating(name) {
            match prepare_mutation_in_with_state(&self.root, &self.state_root, name, arguments)
                .await
            {
                Ok(prepared) => ConfirmationRequest::FileEdit {
                    path: prepared
                        .single_target_path()
                        .unwrap_or_else(|| "(multiple files)".into()),
                    diff: prepared.preview(),
                },
                Err(_) => ConfirmationRequest::FileEdit {
                    path: path_from_args(arguments).unwrap_or_else(|| name.to_string()),
                    diff: arguments.to_string(),
                },
            }
        } else if name == "bash" && !classify_shell_tool_arguments(arguments).is_proven_read_only()
        {
            ConfirmationRequest::ShellMutation {
                command: command_from_args(arguments),
                cwd: self.root.display().to_string(),
            }
        } else {
            return None;
        };
        // Re-read after the (possibly slow) preview so `/yolo` typed while
        // the diff was being prepared is honored without a prompt.
        let mode = permission();
        if mode == PermissionMode::Always {
            return None;
        }
        if mode == PermissionMode::Auto && request.safe_for_auto() {
            return None;
        }
        self.liveness
            .set_state(hi_liveness::HarnessState::AwaitingConfirmation);
        self.liveness
            .emit(hi_liveness::EventCode::ConfirmShown, None, None, None);
        self.liveness.note_progress();
        let mut guard = ConfirmGuard {
            liveness: self.liveness.clone(),
            answered: false,
        };
        let result = ui.confirm(request).await;
        guard.answered = true;
        self.liveness
            .emit(hi_liveness::EventCode::ConfirmAnswered, None, None, None);
        self.liveness.note_progress();
        match result {
            ConfirmationResult::Approved => None,
            denied => {
                self.liveness
                    .set_state(hi_liveness::HarnessState::AwaitingModel);
                match denied {
                    ConfirmationResult::Rejected => Some(denied_outcome("rejected by user")),
                    _ => Some(denied_outcome("confirmation unavailable")),
                }
            }
        }
    }
}

struct ConfirmGuard {
    liveness: hi_liveness::Publisher,
    answered: bool,
}

impl Drop for ConfirmGuard {
    fn drop(&mut self) {
        if !self.answered {
            hi_liveness::report_invariant(
                &self.liveness,
                hi_liveness::InvariantCode::ConfirmUnanswered,
            );
        }
    }
}

fn run_in_background(name: &str, arguments: &str) -> bool {
    if name != "bash" {
        return false;
    }
    serde_json::from_str::<serde_json::Value>(arguments)
        .ok()
        .and_then(|value| value.get("run_in_background")?.as_bool())
        .unwrap_or(false)
}

fn path_from_args(arguments: &str) -> Option<String> {
    serde_json::from_str::<serde_json::Value>(arguments)
        .ok()?
        .get("path")?
        .as_str()
        .map(str::to_string)
}

fn command_from_args(arguments: &str) -> String {
    serde_json::from_str::<serde_json::Value>(arguments)
        .ok()
        .and_then(|v| v.get("command")?.as_str().map(str::to_string))
        .unwrap_or_else(|| arguments.to_string())
}

pub fn plan_from_outcome(outcome: &ToolOutcome) -> Option<Vec<PlanStep>> {
    outcome.plan.clone()
}

pub fn interrupted_outcome() -> ToolOutcome {
    denied_outcome("interrupted by user")
}

fn failed_outcome(content: impl Into<String>) -> ToolOutcome {
    ToolOutcome {
        content: content.into(),
        display: None,
        plan: None,
        status: ToolStatus::Failed,
        process: None,
        background: None,
        effects: ToolEffects::default(),
        truncation: TruncationState::Complete,
        images: Vec::new(),
    }
}

fn denied_outcome(content: impl Into<String>) -> ToolOutcome {
    ToolOutcome {
        status: ToolStatus::Denied,
        ..failed_outcome(content)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::TestUi;
    use hi_tools::ProcessRunner;
    use hi_tools::sandbox::SandboxPolicy;

    fn host(root: std::path::PathBuf) -> ToolHost {
        let state = root.join(".hi");
        let runner = ProcessRunner::new_with_policy(&root, SandboxPolicy::Off).unwrap();
        ToolHost::new_with_runner(root, state, runner).unwrap()
    }

    #[tokio::test]
    async fn bash_sed_dump_is_rewritten_to_read() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("lib.rs"), "fn main() {}\n").unwrap();
        let host = host(dir.path().to_path_buf());
        let args = serde_json::json!({"command": "sed -n '1,2p' lib.rs"}).to_string();
        let mut ui = TestUi::default();
        let out = host
            .execute("c1", "bash", &args, || PermissionMode::Always, &mut ui)
            .await;
        assert!(
            !out.content.contains("Reserve bash"),
            "sed dump should rewrite, not reject: {}",
            out.content
        );
        assert!(
            out.content.contains("fn main"),
            "numbered read of lib.rs: {}",
            out.content
        );
        assert_eq!(out.status, ToolStatus::Succeeded);
    }

    #[tokio::test]
    async fn bash_cargo_still_runs() {
        let dir = tempfile::tempdir().unwrap();
        let host = host(dir.path().to_path_buf());
        let args = serde_json::json!({"command": "echo ok"}).to_string();
        let mut ui = TestUi::default();
        let out = host
            .execute("c1", "bash", &args, || PermissionMode::Always, &mut ui)
            .await;
        assert!(
            out.content.contains("ok"),
            "echo should run: {}",
            out.content
        );
    }

    #[test]
    fn tool_unclosed_sets_invariant_without_panicking() {
        let dir = tempfile::tempdir().unwrap();
        let host = host(dir.path().to_path_buf());
        host.liveness.note_tool_start("c1", "bash");
        host.liveness.note_tool_start("c2", "bash");
        let snap = host.liveness.snapshot();
        assert_eq!(
            snap.invariant.expect("sticky invariant").code,
            hi_liveness::InvariantCode::ToolUnclosed
        );
    }
}
