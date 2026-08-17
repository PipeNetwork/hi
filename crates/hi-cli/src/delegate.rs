//! The CLI's write-`delegate` subagent runner.
//!
//! A delegate works in an isolated Git worktree based on an immutable snapshot
//! of the parent's current tree. Only a typed successful child outcome with a
//! non-empty, independently verified diff is eligible for transactional merge.

use std::ffi::OsString;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};
use async_trait::async_trait;
use hi_agent::{DelegateOutcome, DelegateProgress, DelegateRunner};
use hi_tools::ToolStatus;

use crate::candidate_gate::{
    independently_verify_candidate_cached, inspect_child_report, is_destination_verify_cancelled,
    is_verifier_cancelled, repository_root, same_paths, staged_candidate_diff,
};
use crate::candidate_merge::apply_candidate_and_reverify_cancellable;
use crate::delegate_events;
use crate::resource_governor::{self, ResourceClass};

const DELEGATE_TIMEOUT_SECS: u64 = 600;
const DELEGATE_QUEUE_TIMEOUT_SECS: u64 = 600;
const DEFAULT_GLOBAL_DELEGATE_CONCURRENCY: usize = 4;
const MAX_GLOBAL_DELEGATE_CONCURRENCY: usize = 16;

/// Cross-process delegate capacity lease. Atomic create-new slot files prevent
/// independent `hi` processes from oversubscribing the provider/build machine.
/// A lease is reclaimed only when its recorded PID is no longer alive.
struct DelegateLease {
    path: PathBuf,
    _file: File,
}

impl Drop for DelegateLease {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn acquire_delegate_lease(
    state_root: &Path,
    timeout: Duration,
    stop: &dyn Fn() -> bool,
) -> Result<DelegateLease> {
    let limit = std::env::var("HI_GLOBAL_DELEGATE_CONCURRENCY")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_GLOBAL_DELEGATE_CONCURRENCY)
        .clamp(1, MAX_GLOBAL_DELEGATE_CONCURRENCY);
    let lease_root = state_root.join("delegate-leases");
    std::fs::create_dir_all(&lease_root)
        .with_context(|| format!("creating delegate lease directory {}", lease_root.display()))?;
    let started = Instant::now();
    loop {
        if stop() {
            anyhow::bail!("cancelled waiting for a global delegate concurrency slot");
        }
        for slot in 0..limit {
            let path = lease_root.join(format!("slot-{slot}.lease"));
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(mut file) => {
                    use std::io::Write;
                    writeln!(file, "{}", std::process::id())?;
                    return Ok(DelegateLease { path, _file: file });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    if lease_is_stale(&path) {
                        let _ = std::fs::remove_file(&path);
                    }
                }
                Err(error) => return Err(error).context("acquiring delegate concurrency lease"),
            }
        }
        if started.elapsed() >= timeout {
            anyhow::bail!("timed out waiting for a global delegate concurrency slot");
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

fn lease_is_stale(path: &Path) -> bool {
    let Ok(text) = std::fs::read_to_string(path) else {
        return false;
    };
    let Ok(pid) = text.trim().parse::<u32>() else {
        return true;
    };
    #[cfg(unix)]
    {
        !std::path::Path::new("/proc").join(pid.to_string()).exists()
            && std::process::Command::new("kill")
                .args(["-0", &pid.to_string()])
                .status()
                .is_ok_and(|status| !status.success())
    }
    #[cfg(not(unix))]
    {
        let _ = pid;
        false
    }
}

pub struct CliDelegateRunner {
    exe: PathBuf,
    provider: String,
    model: String,
    base_url: String,
    api_key: String,
    default_verify: Option<String>,
    max_steps: Option<u32>,
    max_verify: u32,
    workspace_root: PathBuf,
    state_root: PathBuf,
    counter: AtomicU32,
}

impl CliDelegateRunner {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        exe: PathBuf,
        provider: String,
        model: String,
        base_url: String,
        api_key: String,
        default_verify: Option<String>,
        max_steps: Option<u32>,
        max_verify: u32,
        workspace_root: PathBuf,
        state_root: PathBuf,
    ) -> Result<Self> {
        let workspace_root = canonical_directory(&workspace_root, "delegate workspace root")?;
        std::fs::create_dir_all(&state_root)
            .with_context(|| format!("creating delegate state root {}", state_root.display()))?;
        let state_root = canonical_directory(&state_root, "delegate state root")?;
        ensure!(
            state_root != workspace_root && !workspace_root.starts_with(&state_root),
            "delegate state root must not equal or contain the workspace root"
        );
        // No configured verify pipeline used to mean "delegate unavailable".
        // Known project types have an obvious build gate, and a delegate whose
        // work must compile is strictly safer than no delegate at all — the
        // child also gets its verify-repair loop, so gate failures feed the
        // compiler error back to the model before anything is rejected.
        let default_verify = default_verify
            .filter(|command| !command.trim().is_empty())
            .or_else(|| derive_default_verify(&workspace_root));
        Ok(Self {
            exe,
            provider,
            model,
            base_url,
            api_key,
            default_verify,
            max_steps,
            max_verify,
            workspace_root,
            state_root,
            counter: AtomicU32::new(0),
        })
    }
}

#[async_trait]
impl DelegateRunner for CliDelegateRunner {
    async fn run_cancellable(
        &self,
        task: &str,
        verify: Option<&str>,
        cancellation: hi_agent::TurnCancellation,
    ) -> DelegateOutcome {
        self.run_with_route(
            task,
            verify,
            &hi_agent::SubagentRoute::default(),
            None,
            cancellation,
        )
        .await
    }

    async fn run_routed(
        &self,
        task: &str,
        verify: Option<&str>,
        route: &hi_agent::SubagentRoute,
        cancellation: hi_agent::TurnCancellation,
    ) -> DelegateOutcome {
        self.run_with_route(task, verify, route, None, cancellation)
            .await
    }

    async fn run_routed_with_progress(
        &self,
        task: &str,
        verify: Option<&str>,
        route: &hi_agent::SubagentRoute,
        cancellation: hi_agent::TurnCancellation,
        progress: Option<Arc<dyn DelegateProgress>>,
    ) -> DelegateOutcome {
        self.run_with_route(task, verify, route, progress, cancellation)
            .await
    }

    async fn run(&self, task: &str, verify: Option<&str>) -> DelegateOutcome {
        self.run_with_route(
            task,
            verify,
            &hi_agent::SubagentRoute::default(),
            None,
            hi_agent::TurnCancellation::new(),
        )
        .await
    }
}

impl CliDelegateRunner {
    /// Resolve the child's provider route: team-role overrides win over the
    /// runner's defaults. An endpoint override implies the generic
    /// OpenAI-compatible provider — local servers (MLX, Ollama, llama.cpp)
    /// all speak it.
    pub(crate) fn effective_route(
        &self,
        route: &hi_agent::SubagentRoute,
    ) -> (String, String, String, String) {
        let model = route.model.clone().unwrap_or_else(|| self.model.clone());
        match route.base_url.as_deref() {
            Some(url) => (
                "openai".to_string(),
                model,
                url.to_string(),
                route.api_key.clone().unwrap_or_default(),
            ),
            None => (
                self.provider.clone(),
                model,
                self.base_url.clone(),
                self.api_key.clone(),
            ),
        }
    }

    async fn run_with_route(
        &self,
        task: &str,
        verify: Option<&str>,
        route: &hi_agent::SubagentRoute,
        progress: Option<Arc<dyn DelegateProgress>>,
        cancellation: hi_agent::TurnCancellation,
    ) -> DelegateOutcome {
        if cancellation.is_cancelled() {
            return outcome(ToolStatus::Cancelled, "delegate cancelled before setup");
        }
        let Some(verify_cmd) = verify
            .map(str::to_string)
            .or_else(|| self.default_verify.clone())
            .filter(|command| !command.trim().is_empty())
        else {
            return outcome(
                ToolStatus::Denied,
                "delegate unavailable: no verification pipeline was resolved; nothing was run.",
            );
        };

        let repo_root = match repository_root(&self.workspace_root)
            .and_then(|root| canonical_directory(&root, "delegate repository root"))
        {
            Ok(root) => root,
            Err(error) => {
                return outcome(
                    ToolStatus::Denied,
                    &format!("delegate unavailable: not in a git repository: {error:#}"),
                );
            }
        };
        if !hi_tools::worktree::in_git_repo(&self.workspace_root) {
            return outcome(
                ToolStatus::Denied,
                "delegate unavailable: not in a git repository.",
            );
        }
        // Start from the exact parent state, including uncommitted files.
        let checkpoint = match hi_tools::checkpoint::create_detailed_with_state(
            &self.workspace_root,
            &self.state_root,
        )
        .await
        {
            hi_tools::checkpoint::CreateResult::Created(sha) => sha,
            hi_tools::checkpoint::CreateResult::Unavailable(reason)
            | hi_tools::checkpoint::CreateResult::Failed(reason) => {
                return outcome(
                    ToolStatus::Denied,
                    &format!("delegate unavailable: couldn't snapshot the working tree: {reason}"),
                );
            }
        };
        let workspace_relative = match self.workspace_root.strip_prefix(&repo_root) {
            Ok(relative) => relative.to_path_buf(),
            Err(error) => {
                return outcome(
                    ToolStatus::Failed,
                    &format!("delegate workspace is outside its repository root: {error}"),
                );
            }
        };

        let idx = self.counter.fetch_add(1, Ordering::Relaxed);
        let exe = self.exe.clone();
        let (provider, model, base_url, api_key) = self.effective_route(route);
        let max_steps = self.max_steps;
        let max_verify = self.max_verify;
        let task = task.to_string();
        let workspace_root = self.workspace_root.clone();
        let state_root = self.state_root.clone();
        if cancellation.is_cancelled() {
            return outcome(ToolStatus::Cancelled, "delegate cancelled before setup");
        }

        tokio::task::spawn_blocking(move || {
            run_blocking(
                &exe,
                &provider,
                &model,
                &base_url,
                &api_key,
                &task,
                &verify_cmd,
                max_steps,
                max_verify,
                &checkpoint,
                idx,
                &repo_root,
                &workspace_relative,
                &workspace_root,
                &state_root,
                progress,
                cancellation,
            )
        })
        .await
        .unwrap_or_else(|error| {
            outcome(
                ToolStatus::Failed,
                &format!("delegate task failed to run: {error}"),
            )
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn run_blocking(
    exe: &Path,
    provider: &str,
    model: &str,
    base_url: &str,
    api_key: &str,
    task: &str,
    verify_cmd: &str,
    max_steps: Option<u32>,
    max_verify: u32,
    checkpoint: &str,
    idx: u32,
    repo_root: &Path,
    workspace_relative: &Path,
    workspace_root: &Path,
    state_root: &Path,
    progress: Option<Arc<dyn DelegateProgress>>,
    cancellation: hi_agent::TurnCancellation,
) -> DelegateOutcome {
    if let Some(out) = stop_if_cancelled(&cancellation, None, None) {
        return out;
    }
    let queue_started = Instant::now();
    report_progress(progress.as_deref(), "waiting for capacity");
    let _lease = match acquire_delegate_lease(
        state_root,
        Duration::from_secs(delegate_queue_timeout_secs()),
        &|| cancellation.is_cancelled(),
    ) {
        Ok(lease) => lease,
        Err(error) => {
            if let Some(out) = stop_if_cancelled(&cancellation, None, None) {
                return out;
            }
            return outcome(
                ToolStatus::Failed,
                &format!("delegate could not acquire global capacity: {error:#}"),
            );
        }
    };
    let queue_wait_ms = queue_started.elapsed().as_millis();
    let setup_queue_started = Instant::now();
    let setup_lease = match resource_governor::acquire_while(
        state_root,
        ResourceClass::Setup,
        Duration::from_secs(delegate_queue_timeout_secs()),
        &|| cancellation.is_cancelled(),
    ) {
        Ok(lease) => lease,
        Err(error) => {
            if let Some(out) = stop_if_cancelled(&cancellation, None, None) {
                return out;
            }
            return outcome(
                ToolStatus::Failed,
                &format!("delegate could not acquire setup capacity: {error:#}"),
            );
        }
    };
    let setup_queue_ms = setup_queue_started.elapsed().as_millis();
    let setup_started = Instant::now();
    report_progress(progress.as_deref(), "creating worktree");
    if let Some(out) = stop_if_cancelled(&cancellation, None, None) {
        return out;
    }
    let worktree_root = hi_tools::worktree::worktree_path("delegate", idx);
    if let Err(error) = hi_tools::worktree::add_worktree(repo_root, &worktree_root, checkpoint) {
        return outcome(
            ToolStatus::Failed,
            &format!("delegate failed to create an isolated worktree: {error}"),
        );
    }
    let worktree = worktree_root.join(workspace_relative);
    if !worktree.is_dir() {
        hi_tools::worktree::cleanup(repo_root, &[worktree_root]);
        return outcome(
            ToolStatus::Failed,
            "delegate failed to resolve its scoped workspace in the isolated worktree.",
        );
    }

    let artifact_dir = delegate_artifacts_dir(state_root, idx);
    if let Err(error) = std::fs::create_dir_all(&artifact_dir) {
        hi_tools::worktree::cleanup(repo_root, &[worktree_root]);
        return outcome(
            ToolStatus::Failed,
            &format!("delegate failed to create artifact directory: {error}"),
        );
    }
    let report_path = artifact_dir.join("report.json");
    let log_path = artifact_dir.join("child.log");
    let events_path = artifact_dir.join("events.jsonl");

    let prompt = child_prompt(task, verify_cmd);
    let mut arguments = vec![
        OsString::from("--subagent"),
        OsString::from("--provider"),
        OsString::from(provider),
        OsString::from("--model"),
        OsString::from(model),
        OsString::from("--base-url"),
        OsString::from(base_url),
        OsString::from("--no-save"),
        OsString::from("--temperature"),
        OsString::from("0"),
        OsString::from("--verify"),
        OsString::from(verify_cmd),
        OsString::from("--max-verify-repairs"),
        OsString::from(max_verify.to_string()),
        OsString::from("--review"),
        OsString::from("always"),
        OsString::from("--report"),
        report_path.as_os_str().to_os_string(),
    ];
    let mut event_tailer = None;
    if let Some(progress) = progress.clone()
        && std::fs::write(&events_path, []).is_ok()
    {
        arguments.push("--events-jsonl".into());
        arguments.push(events_path.as_os_str().into());
        event_tailer = delegate_events::start_event_tailer(events_path.clone(), progress);
    }
    if let Some(max_steps) = max_steps {
        arguments.push("--max-steps".into());
        arguments.push(max_steps.to_string().into());
    }
    arguments.push(prompt.into());

    let worktree_setup_ms = setup_started.elapsed().as_millis();
    drop(setup_lease);
    if let Some(out) =
        stop_if_cancelled(&cancellation, Some(repo_root), Some(worktree_root.clone()))
    {
        return out;
    }
    let model_queue_started = Instant::now();
    let process_lease = match resource_governor::acquire_while(
        state_root,
        ResourceClass::Model,
        Duration::from_secs(delegate_queue_timeout_secs()),
        &|| cancellation.is_cancelled(),
    ) {
        Ok(lease) => lease,
        Err(error) => {
            if let Some(out) =
                stop_if_cancelled(&cancellation, Some(repo_root), Some(worktree_root.clone()))
            {
                return out;
            }
            hi_tools::worktree::cleanup(repo_root, &[worktree_root]);
            return outcome(
                ToolStatus::Failed,
                &format!("delegate could not acquire shared model capacity: {error:#}"),
            );
        }
    };
    let model_queue_ms = model_queue_started.elapsed().as_millis();
    let child_started = Instant::now();
    report_progress(progress.as_deref(), "running");
    let execution = crate::child_process::run_maybe_cancelled(
        &worktree,
        exe,
        arguments,
        vec![
            ("HI_FORCE_API_KEY".into(), api_key.into()),
            ("HI_API_KEY".into(), api_key.into()),
            (
                "CARGO_TARGET_DIR".into(),
                state_root.join("build-cache/cargo-target").into_os_string(),
            ),
            (
                "SCCACHE_DIR".into(),
                state_root.join("build-cache/sccache").into_os_string(),
            ),
        ],
        Duration::from_secs(delegate_timeout_secs()),
        &log_path,
        Some(cancellation.clone()),
    );
    let child_runtime_ms = child_started.elapsed().as_millis();
    drop(process_lease);
    if let Some(tailer) = event_tailer {
        tailer.finish();
    }
    if let Some(out) =
        stop_if_cancelled(&cancellation, Some(repo_root), Some(worktree_root.clone()))
    {
        return out;
    }
    let decision_started = Instant::now();
    report_progress(progress.as_deref(), "verifying");
    let mut result = match execution {
        Ok(execution) if execution.status == ToolStatus::Succeeded => decide(
            &worktree,
            checkpoint,
            verify_cmd,
            &report_path,
            &artifact_dir,
            workspace_root,
            state_root,
            &cancellation,
        ),
        Ok(execution) if execution.status == ToolStatus::Cancelled => {
            hi_tools::worktree::cleanup(repo_root, &[worktree_root.clone()]);
            outcome(
                ToolStatus::Cancelled,
                "delegate cancelled — child process stopped; nothing was applied.",
            )
        }
        Ok(execution) => {
            let status = execution.status;
            if matches!(status, ToolStatus::TimedOut | ToolStatus::Failed) {
                let _ = resource_governor::record_overload(state_root, ResourceClass::Model);
            }
            outcome(
                status,
                &format!(
                    "delegate child ended with {status:?} (exit {:?}); its partial changes were discarded. Artifacts: {}",
                    execution.outcome.exit_code,
                    artifact_dir.display()
                ),
            )
        }
        Err(error) => outcome(
            ToolStatus::Failed,
            &format!(
                "delegate couldn't run the hardened child process: {error:#}. Artifacts: {}",
                artifact_dir.display()
            ),
        ),
    };
    let decision_ms = decision_started.elapsed().as_millis();
    result.summary.push_str(&format!(
        " [timing: delegate_queue={queue_wait_ms}ms setup_queue={setup_queue_ms}ms setup={worktree_setup_ms}ms model_queue={model_queue_ms}ms child={child_runtime_ms}ms verify_apply={decision_ms}ms total={}ms]",
        queue_started.elapsed().as_millis(),
    ));
    if result.applied {
        record_verified_merge(state_root, task, &result.changed_files);
    }
    hi_tools::worktree::cleanup(repo_root, &[worktree_root]);
    result
}

/// Journal of merges that passed independent verification — ground truth about
/// what "native" code looks like in this workspace. Raw material for `/learn`
/// (convention distillation into a SKILL.md); never read on the hot path.
fn record_verified_merge(state_root: &Path, task: &str, changed_files: &[String]) {
    const JOURNAL_CAP_LINES: usize = 1_000;
    let dir = state_root.join("learning");
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let path = dir.join("verified-merges.jsonl");
    let at_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    let task_head: String = task.chars().take(240).collect();
    let record = serde_json::json!({
        "at_ms": at_ms,
        "task": task_head,
        "files": changed_files,
    });
    let mut lines = std::fs::read_to_string(&path)
        .map(|text| text.lines().map(str::to_string).collect::<Vec<_>>())
        .unwrap_or_default();
    lines.push(record.to_string());
    if lines.len() > JOURNAL_CAP_LINES {
        lines.drain(..lines.len() - JOURNAL_CAP_LINES / 2);
    }
    let _ = std::fs::write(&path, lines.join("\n") + "\n");
}

fn report_progress(progress: Option<&dyn DelegateProgress>, activity: &str) {
    if let Some(progress) = progress {
        progress.progress(activity, None);
    }
}

fn stop_if_cancelled(
    cancellation: &hi_agent::TurnCancellation,
    repo_root: Option<&Path>,
    worktree_root: Option<PathBuf>,
) -> Option<DelegateOutcome> {
    if !cancellation.is_cancelled() {
        return None;
    }
    if let (Some(repo), Some(worktree)) = (repo_root, worktree_root) {
        hi_tools::worktree::cleanup(repo, &[worktree]);
    }
    Some(outcome(
        ToolStatus::Cancelled,
        "delegate cancelled; nothing was applied.",
    ))
}

fn delegate_artifacts_dir(state_root: &Path, idx: u32) -> PathBuf {
    let pid = std::process::id();
    state_root
        .join("delegate-artifacts")
        .join(pid.to_string())
        .join(idx.to_string())
}

fn delegate_queue_timeout_secs() -> u64 {
    std::env::var("HI_DELEGATE_QUEUE_TIMEOUT_SECS")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|&seconds| seconds > 0)
        .unwrap_or(DELEGATE_QUEUE_TIMEOUT_SECS)
}

fn delegate_timeout_secs() -> u64 {
    std::env::var("HI_DELEGATE_TIMEOUT_SECS")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|&seconds| seconds > 0)
        .unwrap_or(DELEGATE_TIMEOUT_SECS)
}

fn decide(
    worktree: &Path,
    checkpoint: &str,
    verify_cmd: &str,
    report_path: &Path,
    artifact_dir: &Path,
    destination: &Path,
    state_root: &Path,
    cancellation: &hi_agent::TurnCancellation,
) -> DelegateOutcome {
    let child = match inspect_child_report(report_path) {
        Ok(child) => child,
        Err(error) => {
            return outcome(
                ToolStatus::Failed,
                &format!(
                    "delegate child did not produce an eligible typed outcome: {error:#}. Nothing was applied. Artifacts: {}",
                    artifact_dir.display()
                ),
            );
        }
    };

    let before_check = match staged_candidate_diff(worktree, checkpoint) {
        Ok(diff) => diff,
        Err(error) => {
            return outcome(
                ToolStatus::Failed,
                &format!("delegate diff could not be resolved: {error:#}; nothing was applied."),
            );
        }
    };
    if before_check.paths.is_empty() {
        return outcome(
            ToolStatus::Failed,
            "delegate made no changes; nothing was applied.",
        );
    }
    // The staged diff excludes regenerable artifacts (target/, caches, .hi
    // state); compare against the child report filtered the same way so a
    // build inside the worktree can't fail the exact-match gate.
    let reported: Vec<String> = child
        .changed_files
        .iter()
        .filter(|path| crate::candidate_gate::merge_eligible_path(path))
        .cloned()
        .collect();
    if !same_paths(&reported, &before_check.display_paths) {
        return outcome(
            ToolStatus::Failed,
            &format!(
                "delegate report did not match its exact worktree diff; nothing was applied. \
                 reported: [{}] diff: [{}] child_dir: {}",
                reported.join(", "),
                before_check.display_paths.join(", "),
                worktree.display()
            ),
        );
    }

    if cancellation.is_cancelled() {
        return outcome(
            ToolStatus::Cancelled,
            "delegate cancelled during verification; nothing was applied.",
        );
    }

    let after_check = match independently_verify_candidate_cached(
        worktree,
        checkpoint,
        verify_cmd,
        &state_root.join("delegate-verification-cache"),
        Some(cancellation.clone()),
    ) {
        Ok(diff) => diff,
        Err(error) if is_verifier_cancelled(&error) => {
            return outcome(
                ToolStatus::Cancelled,
                "delegate cancelled during verification; nothing was applied.",
            );
        }
        Err(error) => {
            return outcome(
                ToolStatus::Failed,
                &format!(
                    "delegate rolled back before merge — independent verification failed: {error:#}. The working tree is unchanged."
                ),
            );
        }
    };
    if !same_paths(&reported, &after_check.display_paths) {
        return outcome(
            ToolStatus::Failed,
            "delegate's independently verified diff no longer matched its report; nothing was applied.",
        );
    }

    if cancellation.is_cancelled() {
        return outcome(
            ToolStatus::Cancelled,
            "delegate cancelled during verification; nothing was applied.",
        );
    }

    match apply_candidate_and_reverify_cancellable(
        worktree,
        checkpoint,
        destination,
        state_root,
        verify_cmd,
        Some(cancellation.clone()),
    ) {
        Ok(applied) => DelegateOutcome {
            status: ToolStatus::Succeeded,
            applied: true,
            summary: format!(
                "delegate applied — {} file(s) changed · child outcome passed · independent and destination verification passed: {} [merge_queue={}ms apply={}ms verifier_queue={}ms verifier={}ms]",
                applied.changes.len(),
                applied.changes.join(", "),
                applied.timings.merge_queue_ms,
                applied.timings.apply_ms,
                applied.timings.verifier_queue_ms,
                applied.timings.verifier_ms,
            ),
            changed_files: applied.changes,
        },
        Err(error) if is_destination_verify_cancelled(&error) => outcome(
            ToolStatus::Cancelled,
            "delegate cancelled during destination verification; destination was rolled back.",
        ),
        Err(error) if is_verifier_cancelled(&error) => outcome(
            ToolStatus::Cancelled,
            "delegate cancelled during verification; nothing was applied.",
        ),
        Err(error) => outcome(
            ToolStatus::Failed,
            &format!(
                "delegate changes were not accepted: {error:#}. See artifacts at {}",
                artifact_dir.display()
            ),
        ),
    }
}

impl CliDelegateRunner {
    /// Test-only view of the effective default verify pipeline.
    #[cfg(test)]
    pub(crate) fn default_verify_for_tests(&self) -> Option<String> {
        self.default_verify.clone()
    }
}

/// The build gate a workspace's project type implies when the session
/// configures no verify pipeline. Conservative: only ecosystems whose
/// standard check command is safe, non-interactive, and meaningful.
/// `None` keeps delegate unavailable, exactly as before.
pub(crate) fn derive_default_verify(workspace_root: &Path) -> Option<String> {
    if workspace_root.join("Cargo.toml").is_file() {
        return Some("cargo check --workspace --all-targets".to_string());
    }
    if workspace_root.join("go.mod").is_file() {
        return Some("go build ./...".to_string());
    }
    None
}

fn canonical_directory(path: &Path, label: &str) -> Result<PathBuf> {
    let path = path
        .canonicalize()
        .with_context(|| format!("canonicalizing {label} {}", path.display()))?;
    ensure!(
        path.is_dir(),
        "{label} is not a directory: {}",
        path.display()
    );
    Ok(path)
}

fn outcome(status: ToolStatus, summary: &str) -> DelegateOutcome {
    DelegateOutcome {
        status,
        applied: false,
        changed_files: Vec::new(),
        summary: summary.to_string(),
    }
}

fn child_prompt(task: &str, verify: &str) -> String {
    format!(
        "Implement this self-contained subtask by editing files and running commands as needed. \
         Do not report completion until `{verify}` passes on the final revision.\n\nTask: {task}"
    )
}
