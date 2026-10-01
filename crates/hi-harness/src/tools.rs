//! Local tool host: advertise a small catalog and execute via `hi-tools`.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use hi_ai::ToolSpec;
use hi_lsp::LspManager;
use hi_tools::shell_policy::classify_shell_tool_arguments;
use hi_tools::{
    BackgroundRegistry, FailedMutationLedger, InspectRepeatLedger, PlanStep, ProcessRunner,
    ReadCache, RepoMapCache, RerunLedger, TOOL_SPECS, ToolEffects, ToolOutcome, ToolStatus,
    TruncationState, VerifyRepeatLedger, execute_prepared_in_runtime,
    execute_streaming_in_runtime_with_runner, is_filesystem_mutating, is_probe_refusal,
    prepare_mutation_in_with_state,
};

use crate::TurnCancellation;
use crate::liveness::tool_fingerprint;
use crate::ui::{AutoHint, ConfirmationRequest, ConfirmationResult, PermissionMode, Ui};

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
    inspect_repeats: Mutex<InspectRepeatLedger>,
    verify_repeats: Mutex<VerifyRepeatLedger>,
    /// Foreground `bash` commands that are neither probes nor verify runs.
    reruns: Mutex<RerunLedger>,
    failed_mutations: Mutex<FailedMutationLedger>,
    /// Bumped on every applied file change; part of a verify run's storm key.
    mutation_epoch: std::sync::atomic::AtomicU64,
    liveness: hi_liveness::Publisher,
}

/// The ledger key for a verify command (`cargo test`, `pytest`, …), or `None`
/// for anything else.
fn verify_command_key(name: &str, arguments: &str) -> Option<String> {
    if !crate::completion::looks_like_verify(name, arguments) {
        return None;
    }
    let command = serde_json::from_str::<serde_json::Value>(arguments)
        .ok()?
        .get("command")?
        .as_str()?
        .to_string();
    Some(hi_tools::verify_key(&command))
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
        let host = Self {
            root,
            state_root,
            runner,
            lsp,
            background: Arc::new(BackgroundRegistry::default()),
            read_cache: Mutex::new(ReadCache::new()),
            repo_map: Mutex::new(RepoMapCache::new()),
            inspect_repeats: Mutex::new(InspectRepeatLedger::new()),
            verify_repeats: Mutex::new(VerifyRepeatLedger::new()),
            reruns: Mutex::new(RerunLedger::new()),
            failed_mutations: Mutex::new(FailedMutationLedger::new()),
            mutation_epoch: std::sync::atomic::AtomicU64::new(0),
            liveness: hi_liveness::Publisher::new(),
        };
        host.bind_pgid_source();
        Ok(host)
    }

    pub fn set_liveness(&mut self, liveness: hi_liveness::Publisher) {
        self.liveness = liveness;
        self.bind_pgid_source();
    }

    fn bind_pgid_source(&self) {
        let foreground = self.runner.foreground_registry();
        let background = Arc::clone(&self.background);
        self.liveness.set_pgid_source(Arc::new(move || {
            collect_child_pgids(&foreground, &background)
        }));
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
        if let Ok(mut cache) = self.read_cache.lock() {
            cache.reset_turn_full_reads();
        }
        if let Ok(mut ledger) = self.inspect_repeats.lock() {
            ledger.reset();
        }
        if let Ok(mut ledger) = self.verify_repeats.lock() {
            ledger.reset();
        }
        if let Ok(mut ledger) = self.reruns.lock() {
            ledger.reset();
        }
    }

    pub(crate) fn forget_inspect_keys(&self, keys: &[String]) {
        if keys.is_empty() {
            return;
        }
        let mut ledger = self
            .inspect_repeats
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for key in keys {
            ledger.forget(key);
        }
    }

    fn admit_inspect_repeat(&self, name: &str, fingerprint: &str) -> Option<String> {
        self.inspect_repeats
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .admit(name, fingerprint)
    }

    fn admit_verify_repeat(&self, key: Option<&str>) -> Option<String> {
        self.verify_repeats
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .admit(key?)
    }

    fn admit_rerun(&self, fingerprint: &str, command: Option<&str>) -> Option<String> {
        self.reruns
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .admit(fingerprint, command?)
    }

    fn record_rerun_output(&self, fingerprint: &str, command: Option<&str>, outcome: &ToolOutcome) {
        if command.is_none() || outcome.status == ToolStatus::Denied {
            return;
        }
        self.reruns
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .record_output(fingerprint, &outcome.content);
    }

    fn admit_failed_mutation(&self, name: &str, fingerprint: &str) -> Option<String> {
        if !matches!(name, "write" | "edit" | "multi_edit" | "apply_patch") {
            return None;
        }
        self.failed_mutations
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .admit(fingerprint)
    }

    fn record_verify_output(&self, key: Option<&str>, output: &str) {
        let Some(key) = key else {
            return;
        };
        if is_probe_refusal(output) {
            return;
        }
        self.verify_repeats
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .record_output(key, output);
    }

    fn record_inspect_output(&self, name: &str, fingerprint: &str, output: &str) {
        self.inspect_repeats
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .record_output(name, fingerprint, output);
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
        self.execute_with_auto(id, name, arguments, permission, AutoHint::Heuristic, ui)
            .await
    }

    pub async fn execute_with_auto(
        &self,
        id: &str,
        name: &str,
        arguments: &str,
        permission: impl Fn() -> PermissionMode,
        auto: AutoHint,
        ui: &mut dyn Ui,
    ) -> ToolOutcome {
        let never = TurnCancellation::new();
        self.execute_cancellable(id, name, arguments, permission, auto, ui, &never)
            .await
    }

    /// [`Self::execute_with_auto`] that stops at `cancel`.
    ///
    /// Esc used to reach only `ProcessRunner` children through
    /// [`ToolInterrupt`]. An MCP call, hook, LSP request, or web fetch kept
    /// the turn attached until the remote side answered, which users saw as
    /// a stall that ignored Esc. Dropping the tool future ends those awaits;
    /// process groups still die through their drop guards. The liveness
    /// record closes normally so a user interrupt never reads as
    /// `ToolUnclosed` or `ConfirmUnanswered` to the supervisor.
    #[allow(clippy::too_many_arguments)]
    pub async fn execute_cancellable(
        &self,
        id: &str,
        name: &str,
        arguments: &str,
        permission: impl Fn() -> PermissionMode,
        auto: AutoHint,
        ui: &mut dyn Ui,
        cancel: &TurnCancellation,
    ) -> ToolOutcome {
        if cancel.is_cancelled() {
            return interrupted_outcome();
        }
        if let Some(denied) = self
            .confirm_if_needed(name, arguments, permission, auto, ui, cancel)
            .await
        {
            return denied;
        }
        self.liveness.note_tool_start(id, name);
        let fingerprint = tool_fingerprint(name, arguments);
        ui.tool_started_id(id, name, arguments);
        if let Some(message) = self.admit_inspect_repeat(name, &fingerprint) {
            // Do not count a refusal toward identical-tool storm: a single
            // model round of eight greps would otherwise hit the storm
            // invariant before the harness can stop on probe refusals.
            self.liveness.note_tool_end(false);
            return failed_outcome(message);
        }
        let verify = verify_command_key(name, arguments);
        if let Some(message) = self.admit_verify_repeat(verify.as_deref()) {
            self.liveness.note_tool_end(false);
            return failed_outcome(message);
        }
        if let Some(message) = self.admit_failed_mutation(name, &fingerprint) {
            self.liveness.note_tool_end(false);
            return failed_outcome(message);
        }
        // Refused before the storm count, like the ledgers above: the third
        // identical output gets a replay instead of ending the turn.
        let rerun = rerun_command(name, arguments, verify.is_some());
        if let Some(message) = self.admit_rerun(&fingerprint, rerun.as_deref()) {
            self.liveness.note_tool_end(false);
            return failed_outcome(message);
        }
        // Edit, test, edit, test is progress, not a storm: a verify run after a
        // file change counts as a new call. The verify ledger above still
        // refuses a rerun when nothing changed.
        let storm_key = if verify.is_some() {
            format!(
                "{fingerprint}@{}",
                self.mutation_epoch
                    .load(std::sync::atomic::Ordering::Relaxed)
            )
        } else {
            fingerprint.clone()
        };
        self.liveness.note_tool_fingerprint(&storm_key);
        if self
            .liveness
            .snapshot()
            .invariant
            .is_some_and(|inv| inv.code == hi_liveness::InvariantCode::IdenticalToolStorm)
        {
            // The call that crosses the cap must not start. A full `cargo test`
            // here is what made the storm look like a stall.
            self.liveness.note_tool_end(false);
            return failed_outcome(
                "identical tool storm; this call was not started. Stop repeating it.",
            );
        }
        let run = async {
            if matches!(name, "write" | "edit" | "multi_edit" | "apply_patch") {
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
            }
        };
        let outcome = tokio::select! {
            outcome = run => outcome,
            _ = cancel.cancelled() => interrupted_outcome(),
        };
        if outcome.effects.mutation_applied {
            // Exact grep/read repeats are stale once the tree changed.
            if let Ok(mut ledger) = self.inspect_repeats.lock() {
                ledger.reset();
            }
            if let Ok(mut ledger) = self.verify_repeats.lock() {
                ledger.note_mutation();
            }
            if let Ok(mut ledger) = self.reruns.lock() {
                ledger.note_mutation();
            }
            if let Ok(mut ledger) = self.failed_mutations.lock() {
                ledger.note_mutation();
            }
            self.mutation_epoch
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        } else if outcome.status == ToolStatus::Failed
            && matches!(name, "write" | "edit" | "multi_edit" | "apply_patch")
            && let Ok(mut ledger) = self.failed_mutations.lock()
        {
            ledger.note_failure(&fingerprint, &outcome.content);
        }
        self.record_inspect_output(name, &fingerprint, &outcome.content);
        self.record_verify_output(verify.as_deref(), &outcome.content);
        self.record_rerun_output(&fingerprint, rerun.as_deref(), &outcome);
        if outcome.status == ToolStatus::Failed {
            // A suite that keeps failing the same way across edits (drme's
            // pre-existing CABAC test) must not read as nine identical errors.
            if verify.is_some() {
                self.liveness
                    .note_tool_error(&format!("{storm_key} {}", outcome.content));
            } else {
                self.liveness.note_tool_error(&outcome.content);
            }
        }
        let leaked = !self.runner.detached_descendants_preserved()
            && !run_in_background(name, arguments)
            && self.runner.foreground_registry().active_count() > 0;
        self.liveness.note_tool_end(leaked);
        outcome
    }

    pub(crate) async fn confirmation_request(
        &self,
        name: &str,
        arguments: &str,
    ) -> Option<ConfirmationRequest> {
        if is_filesystem_mutating(name) {
            return Some(
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
                },
            );
        }
        if name == "bash" && !classify_shell_tool_arguments(arguments).is_proven_read_only() {
            return Some(ConfirmationRequest::ShellMutation {
                command: command_from_args(arguments),
                cwd: self.root.display().to_string(),
            });
        }
        None
    }

    async fn confirm_if_needed(
        &self,
        name: &str,
        arguments: &str,
        permission: impl Fn() -> PermissionMode,
        auto: AutoHint,
        ui: &mut dyn Ui,
        cancel: &TurnCancellation,
    ) -> Option<ToolOutcome> {
        if permission() == PermissionMode::Always {
            return None;
        }
        let request = self.confirmation_request(name, arguments).await?;
        // Re-read after the (possibly slow) preview so `/yolo` typed while
        // the diff was being prepared is honored without a prompt.
        let mode = permission();
        if mode == PermissionMode::Always {
            return None;
        }
        let skip = mode == PermissionMode::Auto
            && match auto {
                AutoHint::Approve => !request.blocks_auto_expand(),
                AutoHint::Confirm => false,
                AutoHint::Heuristic => request.safe_for_auto(),
            };
        if skip {
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
        // A turn cancelled while the overlay is up counts as answered:
        // the user chose to stop, which is not an unanswered confirm.
        let result = tokio::select! {
            result = ui.confirm(request) => result,
            _ = cancel.cancelled() => ConfirmationResult::Cancelled,
        };
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
                    ConfirmationResult::Cancelled => Some(interrupted_outcome()),
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

fn collect_child_pgids(
    foreground: &hi_tools::ForegroundProcessRegistry,
    background: &BackgroundRegistry,
) -> (Vec<i32>, Option<i32>) {
    let fg = foreground.active_pgids();
    let current = fg.first().copied();
    let mut all = fg;
    for pgid in background.running_pgids() {
        if !all.contains(&pgid) {
            all.push(pgid);
        }
    }
    (all, current)
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

/// The command of a foreground `bash` call the verify ledger does not
/// already cover.
fn rerun_command(name: &str, arguments: &str, verify: bool) -> Option<String> {
    if name != "bash" || verify || run_in_background(name, arguments) {
        return None;
    }
    Some(command_from_args(arguments))
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
    use crate::ui::{ConfirmationResult, TestUi};
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

    /// Live ~/drme: the fourth `cargo test` after real edits was refused, and
    /// the model dodged the cap by dropping `timeout`. A rerun is refused only
    /// when no file changed, whatever the pager or tool options.
    #[tokio::test]
    async fn verify_rerun_is_refused_only_when_nothing_changed() {
        let dir = tempfile::tempdir().unwrap();
        let host = host(dir.path().to_path_buf());
        let mut ui = TestUi::default();
        let run = |command: &str, timeout: bool| {
            if timeout {
                serde_json::json!({"command": command, "timeout": 600}).to_string()
            } else {
                serde_json::json!({"command": command}).to_string()
            }
        };
        let first = host
            .execute(
                "t0",
                "bash",
                &run("cargo test 2>&1 | tail -40", true),
                || PermissionMode::Always,
                &mut ui,
            )
            .await;
        assert!(
            !hi_tools::is_probe_refusal(&first.content),
            "{}",
            first.content
        );
        let dodged = host
            .execute(
                "t1",
                "bash",
                &run("cargo test 2>&1 | tail -30", false),
                || PermissionMode::Always,
                &mut ui,
            )
            .await;
        assert!(
            hi_tools::is_probe_refusal(&dodged.content),
            "same command with another pager and no timeout is the same run: {}",
            dodged.content
        );
        for round in 0..10 {
            let write_args = serde_json::json!({
                "path": "touched.rs",
                "content": format!("pub fn touched() -> u32 {{ {round} }}\n")
            })
            .to_string();
            let wrote = host
                .execute(
                    &format!("w{round}"),
                    "write",
                    &write_args,
                    || PermissionMode::Always,
                    &mut ui,
                )
                .await;
            assert_eq!(wrote.status, ToolStatus::Succeeded, "{}", wrote.content);
            let rerun = host
                .execute(
                    &format!("v{round}"),
                    "bash",
                    &run("cargo test 2>&1 | tail -40", true),
                    || PermissionMode::Always,
                    &mut ui,
                )
                .await;
            assert!(
                !hi_tools::is_probe_refusal(&rerun.content),
                "run {round} after an edit must start: {}",
                rerun.content
            );
        }
        assert!(
            host.liveness.snapshot().invariant.is_none(),
            "a suite failing the same way after each edit is not a tool storm: {:?}",
            host.liveness.snapshot().invariant
        );
    }

    /// Live ~/drme: one `python3 << 'EOF'` script ran eight times with the
    /// same output and no edit, and the storm cap ended the turn. The third
    /// identical output is refused with a replay before the storm counts it.
    #[tokio::test]
    async fn identical_bash_output_is_refused_before_the_storm() {
        async fn bash(host: &ToolHost, ui: &mut TestUi, id: &str, args: &str) -> ToolOutcome {
            host.execute(id, "bash", args, || PermissionMode::Always, ui)
                .await
        }
        let dir = tempfile::tempdir().unwrap();
        let host = host(dir.path().to_path_buf());
        let mut ui = TestUi::default();
        let script = serde_json::json!({
            "command": "printf 'obu_size mismatch at 0x1f\\n'\nprintf 'bits_for(74) = 7\\n'"
        })
        .to_string();
        for id in ["p0", "p1"] {
            let out = bash(&host, &mut ui, id, &script).await;
            assert!(!is_probe_refusal(&out.content), "{id}: {}", out.content);
            assert!(out.content.contains("obu_size mismatch"), "{}", out.content);
        }
        let refused = bash(&host, &mut ui, "p2", &script).await;
        assert_eq!(refused.status, ToolStatus::Failed);
        assert!(is_probe_refusal(&refused.content), "{}", refused.content);
        assert!(
            refused.content.contains("obu_size mismatch at 0x1f"),
            "the refusal replays the last output: {}",
            refused.content
        );
        for round in 3..12 {
            let out = bash(&host, &mut ui, &format!("p{round}"), &script).await;
            assert!(is_probe_refusal(&out.content), "{}", out.content);
        }
        assert!(
            host.liveness.snapshot().invariant.is_none(),
            "refused reruns must not add up to a tool storm: {:?}",
            host.liveness.snapshot().invariant
        );

        let write =
            serde_json::json!({"path": "obu.rs", "content": "pub fn obu() {}\n"}).to_string();
        let wrote = host
            .execute("w", "write", &write, || PermissionMode::Always, &mut ui)
            .await;
        assert_eq!(wrote.status, ToolStatus::Succeeded, "{}", wrote.content);
        for id in ["after-edit", "after-edit2"] {
            let out = bash(&host, &mut ui, id, &script).await;
            assert!(!is_probe_refusal(&out.content), "{id}: {}", out.content);
        }
        host.reset_bash_repeats();
        let out = bash(&host, &mut ui, "next-turn", &script).await;
        assert!(!is_probe_refusal(&out.content), "{}", out.content);

        let changing = serde_json::json!({"command": "echo pid $$"}).to_string();
        for round in 0..6 {
            let out = bash(&host, &mut ui, &format!("e{round}"), &changing).await;
            assert!(!is_probe_refusal(&out.content), "{}", out.content);
        }
    }

    #[tokio::test]
    async fn second_identical_list_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let host = host(dir.path().to_path_buf());
        let args = serde_json::json!({"path": "."}).to_string();
        let mut ui = TestUi::default();
        let first = host
            .execute("c1", "list", &args, || PermissionMode::Always, &mut ui)
            .await;
        assert_eq!(first.status, ToolStatus::Succeeded);
        let second = host
            .execute("c2", "list", &args, || PermissionMode::Always, &mut ui)
            .await;
        assert_eq!(second.status, ToolStatus::Failed);
        assert!(
            second.content.contains("already ran this turn"),
            "{}",
            second.content
        );
        assert!(
            hi_tools::is_probe_refusal(&second.content),
            "refusals must trip the same-turn stop: {}",
            second.content
        );
    }

    #[tokio::test]
    async fn inspect_repeat_resets_after_a_successful_write() {
        let dir = tempfile::tempdir().unwrap();
        let host = host(dir.path().to_path_buf());
        let list_args = serde_json::json!({"path": "."}).to_string();
        let mut ui = TestUi::default();
        let first = host
            .execute("c1", "list", &list_args, || PermissionMode::Always, &mut ui)
            .await;
        assert_eq!(first.status, ToolStatus::Succeeded);
        let refused = host
            .execute("c2", "list", &list_args, || PermissionMode::Always, &mut ui)
            .await;
        assert!(
            hi_tools::is_probe_refusal(&refused.content),
            "{}",
            refused.content
        );
        let write_args = serde_json::json!({
            "path": "added.rs",
            "content": "pub const MAX_PASSWORD: u32 = 64;\n"
        })
        .to_string();
        let wrote = host
            .execute(
                "c3",
                "write",
                &write_args,
                || PermissionMode::Always,
                &mut ui,
            )
            .await;
        assert_eq!(wrote.status, ToolStatus::Succeeded, "{}", wrote.content);
        assert!(wrote.effects.mutation_applied);
        let again = host
            .execute("c4", "list", &list_args, || PermissionMode::Always, &mut ui)
            .await;
        assert_eq!(again.status, ToolStatus::Succeeded, "{}", again.content);
        assert!(
            !hi_tools::is_probe_refusal(&again.content),
            "post-write list must see the new tree, got {}",
            again.content
        );
        assert!(
            again.content.contains("added.rs"),
            "list after write should include the new file: {}",
            again.content
        );
    }

    /// Esc mid-tool must end the await, not just signal a process group,
    /// and the liveness record must close so the supervisor never reads a
    /// user interrupt as `ToolUnclosed`.
    #[tokio::test]
    async fn cancel_mid_tool_returns_interrupted_and_closes_liveness() {
        let dir = tempfile::tempdir().unwrap();
        let host = host(dir.path().to_path_buf());
        let args = serde_json::json!({"command": "sleep 30"}).to_string();
        let mut ui = TestUi::default();
        let cancel = TurnCancellation::new();
        let trigger = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            trigger.cancel();
        });
        let started = std::time::Instant::now();
        let out = host
            .execute_cancellable(
                "c1",
                "bash",
                &args,
                || PermissionMode::Always,
                AutoHint::Heuristic,
                &mut ui,
                &cancel,
            )
            .await;
        assert!(
            started.elapsed() < std::time::Duration::from_secs(10),
            "cancel must end the tool await promptly"
        );
        assert_eq!(out.status, ToolStatus::Denied);
        assert!(
            out.content.contains("interrupted by user"),
            "{}",
            out.content
        );
        let beat = host.liveness().snapshot();
        assert_eq!(beat.invariant, None, "{:?}", beat.invariant);
        // The next tool must not trip ToolUnclosed on the interrupted one.
        let echo = serde_json::json!({"command": "echo after"}).to_string();
        let next = host
            .execute("c2", "bash", &echo, || PermissionMode::Always, &mut ui)
            .await;
        assert_eq!(next.status, ToolStatus::Succeeded, "{}", next.content);
        let beat = host.liveness().snapshot();
        assert_eq!(beat.invariant, None, "{:?}", beat.invariant);
    }

    /// A confirm overlay abandoned by turn cancellation counts as answered.
    #[tokio::test]
    async fn cancel_during_confirm_is_interrupted_not_unanswered() {
        struct NeverAnswers;
        impl Ui for NeverAnswers {
            fn assistant_text(&mut self, _: &str) {}
            fn assistant_reasoning(&mut self, _: &str) {}
            fn assistant_end(&mut self) {}
            fn tool_call(&mut self, _: &str, _: &str) {}
            fn tool_result(&mut self, _: &str, _: &str) {}
            fn status(&mut self, _: &str) {}
            fn turn_end(&mut self, _: &str) {}
            fn turn_error(&mut self, _: &str, _: &str, _: &str) {}
            fn confirm(&mut self, _: ConfirmationRequest) -> crate::ConfirmationFuture<'_> {
                Box::pin(std::future::pending())
            }
        }
        let dir = tempfile::tempdir().unwrap();
        let host = host(dir.path().to_path_buf());
        let args = serde_json::json!({"command": "touch never"}).to_string();
        let mut ui = NeverAnswers;
        let cancel = TurnCancellation::new();
        let trigger = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            trigger.cancel();
        });
        let out = host
            .execute_cancellable(
                "c1",
                "bash",
                &args,
                || PermissionMode::Ask,
                AutoHint::Confirm,
                &mut ui,
                &cancel,
            )
            .await;
        assert!(
            out.content.contains("interrupted by user"),
            "{}",
            out.content
        );
        assert!(!dir.path().join("never").exists());
        let beat = host.liveness().snapshot();
        assert_eq!(beat.invariant, None, "{:?}", beat.invariant);
    }

    #[tokio::test]
    async fn auto_hint_approve_skips_overlay_for_gray_shell() {
        let dir = tempfile::tempdir().unwrap();
        let host = host(dir.path().to_path_buf());
        let args = serde_json::json!({"command": "touch jev-ok"}).to_string();
        let mut ui = TestUi {
            confirm: ConfirmationResult::Rejected,
            ..TestUi::default()
        };
        let out = host
            .execute_with_auto(
                "c1",
                "bash",
                &args,
                || PermissionMode::Auto,
                AutoHint::Approve,
                &mut ui,
            )
            .await;
        assert_eq!(out.status, ToolStatus::Succeeded, "{}", out.content);
        assert!(dir.path().join("jev-ok").exists());
    }

    #[tokio::test]
    async fn auto_hint_cannot_expand_force_push() {
        let dir = tempfile::tempdir().unwrap();
        let host = host(dir.path().to_path_buf());
        let args = serde_json::json!({"command": "git push --force origin main"}).to_string();
        let mut ui = TestUi {
            confirm: ConfirmationResult::Rejected,
            ..TestUi::default()
        };
        let out = host
            .execute_with_auto(
                "c1",
                "bash",
                &args,
                || PermissionMode::Auto,
                AutoHint::Approve,
                &mut ui,
            )
            .await;
        assert_eq!(out.status, ToolStatus::Denied, "{}", out.content);
        assert!(out.content.contains("rejected by user"), "{}", out.content);
    }

    #[tokio::test]
    async fn auto_hint_confirm_withholds_heuristic_safe_file() {
        let dir = tempfile::tempdir().unwrap();
        let host = host(dir.path().to_path_buf());
        let args = serde_json::json!({
            "path": "lib.rs",
            "content": "pub fn ok() {}\n"
        })
        .to_string();
        let mut ui = TestUi {
            confirm: ConfirmationResult::Rejected,
            ..TestUi::default()
        };
        let withheld = host
            .execute_with_auto(
                "c1",
                "write",
                &args,
                || PermissionMode::Auto,
                AutoHint::Confirm,
                &mut ui,
            )
            .await;
        assert_eq!(withheld.status, ToolStatus::Denied, "{}", withheld.content);
        let allowed = host
            .execute_with_auto(
                "c2",
                "write",
                &args,
                || PermissionMode::Auto,
                AutoHint::Heuristic,
                &mut ui,
            )
            .await;
        assert_eq!(allowed.status, ToolStatus::Succeeded, "{}", allowed.content);
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

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn silent_child_pgids_are_visible_mid_flight() {
        let dir = tempfile::tempdir().unwrap();
        let host = host(dir.path().to_path_buf());
        let liveness = host.liveness();
        let interrupt = host.interrupt_handle();
        let args = serde_json::json!({"command": "sleep 30"}).to_string();
        let mut ui = TestUi::default();
        let exec = host.execute("c1", "bash", &args, || PermissionMode::Always, &mut ui);
        tokio::pin!(exec);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        let mut saw_pgids = false;
        loop {
            tokio::select! {
                _ = &mut exec => break,
                _ = tokio::time::sleep(std::time::Duration::from_millis(15)) => {
                    let snap = liveness.snapshot();
                    if !snap.child_pgids.is_empty() {
                        saw_pgids = true;
                        interrupt.interrupt();
                    }
                    if std::time::Instant::now() > deadline {
                        interrupt.interrupt();
                    }
                }
            }
        }
        assert!(
            saw_pgids,
            "heartbeat must show live child_pgids during a silent tool"
        );
    }
}
