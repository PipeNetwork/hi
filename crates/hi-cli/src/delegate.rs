//! The CLI's write-`delegate` subagent runner.
//!
//! A delegate works in a detached candidate with private Git metadata and an
//! exact snapshot of the parent's current tree. Only a typed successful child
//! outcome with a non-empty, independently verified diff is eligible for a
//! fenced transactional merge.

use std::ffi::OsString;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};
use async_trait::async_trait;
use hi_agent::{DelegateOutcome, DelegateProgress, DelegateRunner};
use hi_tools::ToolStatus;

use crate::candidate_gate::{
    independently_verify_candidate_cached, inspect_child_report, is_destination_verify_cancelled,
    is_verifier_cancelled, same_paths, staged_candidate_diff,
};
use crate::candidate_merge::apply_candidate_and_reverify_cancellable_at_base;
use crate::delegate_events;
use crate::resource_governor::{self, ResourceClass};

const DELEGATE_SETTLEMENT_GRACE_SECS: u64 = 60;
const DEFAULT_DELEGATE_QUEUE_TIMEOUT_SECS: u64 = 5 * 60;
const DEFAULT_DELEGATE_TIMEOUT_SECS: u64 = 15 * 60;
const DEFAULT_GLOBAL_DELEGATE_CONCURRENCY: usize = 4;
const MAX_GLOBAL_DELEGATE_CONCURRENCY: usize = 16;

/// Cross-process delegate capacity lease. Atomic create-new slot files prevent
/// independent `hi` processes from oversubscribing the provider/build machine.
/// The unique owner token prevents an old guard from removing a replacement
/// file after fault recovery.
#[derive(Debug)]
struct DelegateLease {
    path: PathBuf,
    owner: String,
    _file: File,
}

impl Drop for DelegateLease {
    fn drop(&mut self) {
        remove_delegate_lease_if_owner(&self.path, &self.owner, &self._file);
    }
}

fn delegate_owner_token() -> String {
    uuid::Uuid::new_v4().to_string()
}

fn delegate_lease_record(owner: &str) -> String {
    let birth = resource_governor::current_process_birth_identity().unwrap_or("unknown");
    format!("owner={owner}\npid={}\nbirth={birth}\n", std::process::id())
}

fn delegate_lease_owner(path: &Path) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()?
        .lines()
        .find_map(|line| {
            line.strip_prefix("owner=")
                .map(str::trim)
                .filter(|owner| !owner.is_empty())
                .map(str::to_owned)
        })
}

fn remove_delegate_lease_if_owner(path: &Path, owner: &str, file: &File) {
    if delegate_lease_owner(path).as_deref() == Some(owner)
        && file_still_names_open_inode(path, file)
    {
        let _ = std::fs::remove_file(path);
    }
}

#[cfg(unix)]
fn try_lock_delegate_file(file: &File) -> std::io::Result<bool> {
    use std::os::fd::AsRawFd;

    let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if result == 0 {
        return Ok(true);
    }
    let error = std::io::Error::last_os_error();
    if matches!(
        error.raw_os_error(),
        Some(code) if code == libc::EWOULDBLOCK || code == libc::EAGAIN
    ) {
        Ok(false)
    } else {
        Err(error)
    }
}

#[cfg(not(unix))]
fn try_lock_delegate_file(_file: &File) -> std::io::Result<bool> {
    Ok(true)
}

#[cfg(unix)]
fn file_still_names_open_inode(path: &Path, file: &File) -> bool {
    use std::os::unix::fs::MetadataExt;

    let Ok(open) = file.metadata() else {
        return false;
    };
    let Ok(named) = std::fs::metadata(path) else {
        return false;
    };
    open.dev() == named.dev() && open.ino() == named.ino()
}

#[cfg(not(unix))]
fn file_still_names_open_inode(path: &Path, _file: &File) -> bool {
    path.is_file()
}

/// Reclaim only while holding the old inode's exclusive advisory lock. Every
/// v2 owner retains this lock for the lease lifetime, preventing two waiters
/// from deciding an old path is stale and then unlinking a newly-created slot.
fn reclaim_stale_delegate_lease(path: &Path) -> bool {
    let Ok(file) = OpenOptions::new().read(true).write(true).open(path) else {
        return false;
    };
    if !matches!(try_lock_delegate_file(&file), Ok(true))
        || !lease_is_stale(path)
        || !file_still_names_open_inode(path, &file)
    {
        return false;
    }
    std::fs::remove_file(path).is_ok()
}

fn acquire_delegate_lease(
    state_root: &Path,
    timeout: Option<Duration>,
    stop: &dyn Fn() -> bool,
) -> Result<DelegateLease> {
    let limit = std::env::var("HI_GLOBAL_DELEGATE_CONCURRENCY")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_GLOBAL_DELEGATE_CONCURRENCY)
        .clamp(1, MAX_GLOBAL_DELEGATE_CONCURRENCY);
    acquire_delegate_lease_with_limit(state_root, timeout, stop, limit)
}

fn acquire_delegate_lease_with_limit(
    state_root: &Path,
    timeout: Option<Duration>,
    stop: &dyn Fn() -> bool,
    limit: usize,
) -> Result<DelegateLease> {
    let limit = limit.clamp(1, MAX_GLOBAL_DELEGATE_CONCURRENCY);
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
                    let owner = delegate_owner_token();
                    if !try_lock_delegate_file(&file)
                        .context("locking delegate concurrency lease owner")?
                    {
                        anyhow::bail!("new delegate concurrency lease was already locked");
                    }
                    if let Err(error) = file
                        .write_all(delegate_lease_record(&owner).as_bytes())
                        .and_then(|()| file.sync_all())
                    {
                        // Only unlink a record that durably identifies this
                        // guard. A partial/empty file is recovered after the
                        // shared incomplete-record grace instead of risking
                        // deletion of a replacement path.
                        remove_delegate_lease_if_owner(&path, &owner, &file);
                        return Err(error).context("recording delegate concurrency lease owner");
                    }
                    return Ok(DelegateLease {
                        path,
                        owner,
                        _file: file,
                    });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    reclaim_stale_delegate_lease(&path);
                }
                Err(error) => return Err(error).context("acquiring delegate concurrency lease"),
            }
        }
        if queue_wait_timed_out(started.elapsed(), timeout) {
            anyhow::bail!("timed out waiting for a global delegate concurrency slot");
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

fn lease_is_stale(path: &Path) -> bool {
    let text = std::fs::read_to_string(path).ok();
    let Some(text) = text.as_deref() else {
        return resource_governor::owner_record_is_stale(path, None, None);
    };

    // v2 records are complete only when every field exists. This distinction
    // matters when `create_new` succeeded but the owner write was interrupted:
    // a partial PID must not masquerade as a live, permanent lease.
    if text.lines().any(|line| line.starts_with("owner=")) {
        let owner = text
            .lines()
            .find_map(|line| line.strip_prefix("owner=").map(str::trim))
            .filter(|owner| !owner.is_empty());
        let pid = text.lines().find_map(|line| {
            line.strip_prefix("pid=")?
                .trim()
                .parse::<u32>()
                .ok()
                .filter(|pid| *pid > 0)
        });
        let birth_field = text
            .lines()
            .find_map(|line| line.strip_prefix("birth=").map(str::trim))
            .filter(|birth| !birth.is_empty());
        if owner.is_none() || pid.is_none() || birth_field.is_none() {
            return resource_governor::owner_record_is_stale(path, None, None);
        }
        let birth = birth_field.filter(|birth| *birth != "unknown");
        return resource_governor::owner_record_is_stale(path, pid, birth);
    }

    // Backward compatibility for the original single-PID record. The shared
    // stale check compares file age with process uptime, so PID reuse cannot
    // wedge an unlimited queue.
    let legacy_pid = text
        .split_whitespace()
        .next()
        .and_then(|value| value.parse::<u32>().ok())
        .filter(|pid| *pid > 0);
    resource_governor::owner_record_is_stale(path, legacy_pid, None)
}

pub struct CliDelegateRunner {
    exe: PathBuf,
    provider: String,
    model: String,
    base_url: String,
    api_key: String,
    configured_verify: Option<String>,
    /// `0` means no explicit cap; positive values are forwarded to children.
    /// Atomic because `/config steps` can change it after this runner is
    /// attached to the long-lived interactive Agent.
    max_steps: AtomicU32,
    /// `u64::MAX` means no explicit cap; every `u32` value (including a
    /// managed zero budget) is forwarded losslessly to children.
    max_tool_calls: AtomicU64,
    max_verify: u32,
    workspace: RwLock<DelegateWorkspaceBinding>,
    counter: AtomicU32,
}

#[derive(Clone)]
struct DelegateWorkspaceBinding {
    workspace_root: PathBuf,
    state_root: PathBuf,
    default_verify: Option<String>,
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
        max_tool_calls: Option<u32>,
        max_verify: u32,
        workspace_root: PathBuf,
        state_root: PathBuf,
    ) -> Result<Self> {
        let configured_verify = default_verify.filter(|command| !command.trim().is_empty());
        let workspace =
            delegate_workspace_binding(&workspace_root, &state_root, configured_verify.as_deref())?;
        // Derive a conservative build gate when no verifier was configured.
        Ok(Self {
            exe,
            provider,
            model,
            base_url,
            api_key,
            configured_verify,
            max_steps: AtomicU32::new(max_steps.unwrap_or(0)),
            max_tool_calls: AtomicU64::new(encode_optional_u32(max_tool_calls)),
            max_verify,
            workspace: RwLock::new(workspace),
            counter: AtomicU32::new(0),
        })
    }

    pub(crate) fn configured_max_steps(&self) -> Option<u32> {
        match self.max_steps.load(Ordering::Relaxed) {
            0 => None,
            value => Some(value),
        }
    }

    pub(crate) fn configured_max_tool_calls(&self) -> Option<u32> {
        decode_optional_u32(self.max_tool_calls.load(Ordering::Relaxed))
    }
}

#[async_trait]
impl DelegateRunner for CliDelegateRunner {
    fn bind_workspace(&self, workspace_root: &Path, state_root: &Path) -> bool {
        let Ok(binding) = delegate_workspace_binding(
            workspace_root,
            state_root,
            self.configured_verify.as_deref(),
        ) else {
            return false;
        };
        *self
            .workspace
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = binding;
        true
    }

    fn is_bound_to_workspace(&self, workspace_root: &Path, state_root: &Path) -> bool {
        let binding = self
            .workspace
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        std::fs::canonicalize(workspace_root).ok().as_ref() == Some(&binding.workspace_root)
            && std::fs::canonicalize(state_root).ok().as_ref() == Some(&binding.state_root)
    }

    fn set_max_steps(&self, max_steps: Option<u32>) {
        self.max_steps
            .store(max_steps.unwrap_or(0), Ordering::Relaxed);
    }

    fn set_max_tool_calls(&self, max_tool_calls: Option<u32>) {
        self.max_tool_calls
            .store(encode_optional_u32(max_tool_calls), Ordering::Relaxed);
    }

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
        let workspace = self
            .workspace
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        let Some(verify_cmd) = verify
            .map(str::to_string)
            .or_else(|| workspace.default_verify.clone())
            .filter(|command| !command.trim().is_empty())
        else {
            return outcome(
                ToolStatus::Denied,
                "delegate unavailable: no verification pipeline was resolved; nothing was run.",
            );
        };

        let idx = self.counter.fetch_add(1, Ordering::Relaxed);
        let exe = self.exe.clone();
        let (provider, model, base_url, api_key) = self.effective_route(route);
        let max_steps = self.configured_max_steps();
        let max_tool_calls = self.configured_max_tool_calls();
        let max_verify = self.max_verify;
        let task = task.to_string();
        let workspace_root = workspace.workspace_root;
        let state_root = workspace.state_root;
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
                max_tool_calls,
                max_verify,
                idx,
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
    max_tool_calls: Option<u32>,
    max_verify: u32,
    idx: u32,
    workspace_root: &Path,
    state_root: &Path,
    progress: Option<Arc<dyn DelegateProgress>>,
    cancellation: hi_agent::TurnCancellation,
) -> DelegateOutcome {
    if let Some(out) = stop_if_cancelled(&cancellation) {
        return out;
    }
    let queue_started = Instant::now();
    report_progress(progress.as_deref(), "waiting for capacity");
    let _lease = match acquire_delegate_lease(state_root, delegate_queue_timeout(), &|| {
        cancellation.is_cancelled()
    }) {
        Ok(lease) => lease,
        Err(error) => {
            if let Some(out) = stop_if_cancelled(&cancellation) {
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
    let setup_lease = match resource_governor::acquire_while_optional(
        state_root,
        ResourceClass::Setup,
        delegate_queue_timeout(),
        &|| cancellation.is_cancelled(),
    ) {
        Ok(lease) => lease,
        Err(error) => {
            if let Some(out) = stop_if_cancelled(&cancellation) {
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
    report_progress(progress.as_deref(), "creating detached candidate");
    if let Some(out) = stop_if_cancelled(&cancellation) {
        return out;
    }
    let worktree_root = hi_tools::worktree::worktree_path("delegate", idx);
    let candidate = match hi_tools::candidate_workspace::CandidateWorkspace::create(
        workspace_root,
        state_root,
        &worktree_root,
    ) {
        Ok(candidate) => candidate,
        Err(error) => {
            return outcome(
                ToolStatus::Failed,
                &format!("delegate failed to create a detached candidate: {error:#}"),
            );
        }
    };
    let worktree = candidate.root().to_path_buf();
    let checkpoint = candidate.baseline_commit().to_string();
    let source_snapshot_id = candidate.source_snapshot_id().to_string();

    let artifact_dir = delegate_artifacts_dir(state_root, idx);
    if let Err(error) = std::fs::create_dir_all(&artifact_dir) {
        return outcome(
            ToolStatus::Failed,
            &format!("delegate failed to create artifact directory: {error}"),
        );
    }
    let report_path = artifact_dir.join("report.json");
    let log_path = artifact_dir.join("child.log");
    let events_path = artifact_dir.join("events.jsonl");
    let child_paths = match crate::child_process::CandidateChildPaths::prepare(&candidate) {
        Ok(paths) => paths,
        Err(error) => {
            return outcome(
                ToolStatus::Failed,
                &format!("delegate failed to create isolated child artifacts: {error}"),
            );
        }
    };
    let child_report_path = child_paths.report();
    let child_events_path = child_paths.events();
    let prompt = child_prompt(task, verify_cmd);
    let child_timeout_secs = delegate_timeout_secs();
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
        OsString::from("--review"),
        OsString::from("always"),
        OsString::from("--report"),
        child_report_path.as_os_str().to_os_string(),
    ];
    if max_verify != hi_agent::UNLIMITED_REPAIR_CYCLES {
        arguments.push(OsString::from("--max-verify-repairs"));
        arguments.push(OsString::from(max_verify.to_string()));
    }
    arguments.extend(delegate_child_budget_arguments(
        max_steps,
        max_tool_calls,
        child_timeout_secs,
    ));
    let mut event_tailer = None;
    if let Some(progress) = progress.clone()
        && std::fs::write(&child_events_path, []).is_ok()
    {
        arguments.push("--events-jsonl".into());
        arguments.push(child_events_path.as_os_str().into());
        event_tailer = delegate_events::start_event_tailer(child_events_path.clone(), progress);
    }
    arguments.push(prompt.into());

    let worktree_setup_ms = setup_started.elapsed().as_millis();
    drop(setup_lease);
    if let Some(out) = stop_if_cancelled(&cancellation) {
        return out;
    }
    let model_queue_started = Instant::now();
    let process_lease = match resource_governor::acquire_while_optional(
        state_root,
        ResourceClass::Model,
        delegate_queue_timeout(),
        &|| cancellation.is_cancelled(),
    ) {
        Ok(lease) => lease,
        Err(error) => {
            if let Some(out) = stop_if_cancelled(&cancellation) {
                return out;
            }
            return outcome(
                ToolStatus::Failed,
                &format!("delegate could not acquire shared model capacity: {error:#}"),
            );
        }
    };
    let model_queue_ms = model_queue_started.elapsed().as_millis();
    let child_started = Instant::now();
    report_progress(progress.as_deref(), "running");
    let execution =
        crate::child_process::run_maybe_cancelled(crate::child_process::CandidateChildLaunch {
            workspace_root: &worktree,
            runtime_root: child_paths.runtime_root(),
            executable: exe,
            arguments,
            environment: child_paths.delegate_environment(api_key),
            timeout: child_timeout_secs.map(Duration::from_secs),
            log_path: &log_path,
            cancellation: Some(cancellation.clone()),
            isolation: crate::child_process::CandidateProcessIsolation::new(
                workspace_root,
                state_root,
            ),
        });
    let child_runtime_ms = child_started.elapsed().as_millis();
    drop(process_lease);
    if let Some(tailer) = event_tailer {
        tailer.finish();
    }
    child_paths.retain(&report_path, Some(&events_path));
    if let Some(out) = stop_if_cancelled(&cancellation) {
        return out;
    }
    let decision_started = Instant::now();
    report_progress(progress.as_deref(), "verifying");
    let mut result = match execution {
        Ok(execution) if execution.status == ToolStatus::Succeeded => decide(
            &worktree,
            &checkpoint,
            &source_snapshot_id,
            verify_cmd,
            &report_path,
            &artifact_dir,
            workspace_root,
            state_root,
            &cancellation,
        ),
        Ok(execution) if execution.status == ToolStatus::Cancelled => outcome(
            ToolStatus::Cancelled,
            "delegate cancelled — child process stopped; nothing was applied.",
        ),
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

fn stop_if_cancelled(cancellation: &hi_agent::TurnCancellation) -> Option<DelegateOutcome> {
    if !cancellation.is_cancelled() {
        return None;
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

fn delegate_queue_timeout() -> Option<Duration> {
    delegate_queue_timeout_secs_from_value(
        std::env::var("HI_DELEGATE_QUEUE_TIMEOUT_SECS")
            .ok()
            .as_deref(),
    )
    .map(Duration::from_secs)
}

/// Resolve the managed delegate capacity wait. Detached work is finite by
/// default; a valid positive override replaces the five-minute default.
pub(crate) fn delegate_queue_timeout_secs_from_value(configured: Option<&str>) -> Option<u64> {
    Some(
        configured
            .and_then(|value| value.parse().ok())
            .filter(|&seconds| seconds > 0)
            .unwrap_or(DEFAULT_DELEGATE_QUEUE_TIMEOUT_SECS),
    )
}

pub(crate) fn queue_wait_timed_out(elapsed: Duration, timeout: Option<Duration>) -> bool {
    timeout.is_some_and(|timeout| elapsed >= timeout)
}

fn delegate_timeout_secs() -> Option<u64> {
    let configured = std::env::var("HI_DELEGATE_TIMEOUT_SECS").ok();
    delegate_timeout_secs_from_value(configured.as_deref())
}

/// Resolve the detached delegate wall-clock deadline. A valid positive
/// override is exact; unset, invalid, and zero use the fifteen-minute managed
/// default rather than creating an unbounded unattended process.
pub(crate) fn delegate_timeout_secs_from_value(configured: Option<&str>) -> Option<u64> {
    Some(
        configured
            .and_then(|value| value.parse().ok())
            .filter(|&seconds| seconds > 0)
            .unwrap_or(DEFAULT_DELEGATE_TIMEOUT_SECS),
    )
}

/// Child soft budget plus explicit model-round/tool-call caps. This stays a pure argv
/// builder so the parent/child boundary is regression-testable without spawning
/// a real agent process.
pub(crate) fn delegate_child_budget_arguments(
    max_steps: Option<u32>,
    max_tool_calls: Option<u32>,
    outer_timeout_secs: Option<u64>,
) -> Vec<OsString> {
    let mut arguments = Vec::new();
    // Leave enough time for the child to settle and write its typed report
    // before an explicitly requested outer kill. A one-second opt-in has no
    // useful earlier integer deadline, so it relies on the outer timeout.
    if let Some(outer_timeout_secs) = outer_timeout_secs.filter(|seconds| *seconds > 1) {
        let turn_deadline_secs = outer_timeout_secs
            .saturating_sub(DELEGATE_SETTLEMENT_GRACE_SECS)
            .max(1)
            .min(outer_timeout_secs - 1);
        arguments.push(OsString::from("--turn-deadline"));
        arguments.push(OsString::from(turn_deadline_secs.to_string()));
    }
    if let Some(max_steps) = max_steps {
        arguments.push(OsString::from("--max-steps"));
        arguments.push(OsString::from(max_steps.to_string()));
    }
    if let Some(max_tool_calls) = max_tool_calls {
        arguments.push(OsString::from("--max-tool-calls"));
        arguments.push(OsString::from(max_tool_calls.to_string()));
    }
    arguments
}

const OPTIONAL_U32_NONE: u64 = u64::MAX;

fn encode_optional_u32(value: Option<u32>) -> u64 {
    value.map(u64::from).unwrap_or(OPTIONAL_U32_NONE)
}

fn decode_optional_u32(value: u64) -> Option<u32> {
    (value != OPTIONAL_U32_NONE).then_some(value as u32)
}

#[allow(clippy::too_many_arguments)]
fn decide(
    worktree: &Path,
    checkpoint: &str,
    source_snapshot_id: &str,
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

    match apply_candidate_and_reverify_cancellable_at_base(
        worktree,
        checkpoint,
        destination,
        state_root,
        verify_cmd,
        Some(source_snapshot_id),
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
        self.workspace
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .default_verify
            .clone()
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

fn delegate_workspace_binding(
    workspace_root: &Path,
    state_root: &Path,
    configured_verify: Option<&str>,
) -> Result<DelegateWorkspaceBinding> {
    let workspace_root = canonical_directory(workspace_root, "delegate workspace root")?;
    std::fs::create_dir_all(state_root)
        .with_context(|| format!("creating delegate state root {}", state_root.display()))?;
    let state_root = canonical_directory(state_root, "delegate state root")?;
    ensure!(
        state_root != workspace_root && !workspace_root.starts_with(&state_root),
        "delegate state root must not equal or contain the workspace root"
    );
    Ok(DelegateWorkspaceBinding {
        default_verify: configured_verify
            .map(str::to_owned)
            .or_else(|| derive_default_verify(&workspace_root)),
        workspace_root,
        state_root,
    })
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

#[cfg(test)]
mod lease_tests {
    use super::*;

    fn lease_dir(label: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "hi-delegate-lease-{label}-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    #[cfg(unix)]
    #[test]
    fn dropping_old_guard_does_not_remove_replacement_slot() {
        let root = lease_dir("replacement");
        let path = root.join("slot-0.lease");
        let owner = delegate_owner_token();
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        file.write_all(delegate_lease_record(&owner).as_bytes())
            .unwrap();
        let lease = DelegateLease {
            path: path.clone(),
            owner,
            _file: file,
        };

        std::fs::remove_file(&path).unwrap();
        let replacement_owner = delegate_owner_token();
        std::fs::write(&path, delegate_lease_record(&replacement_owner)).unwrap();
        drop(lease);

        assert_eq!(
            delegate_lease_owner(&path).as_deref(),
            Some(replacement_owner.as_str())
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn stale_cleanup_cannot_unlink_a_locked_active_slot() {
        let root = lease_dir("locked-race");
        let path = root.join("slot-0.lease");
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        assert!(try_lock_delegate_file(&file).unwrap());
        file.write_all(
            format!("owner=stale\npid={}\nbirth=unreachable-owner\n", u32::MAX).as_bytes(),
        )
        .unwrap();

        // Keep this test independent of an external `ps` lookup. Under a
        // saturated process table that lookup may conservatively fail, which
        // correctly leaves a live-PID record alone but is unrelated to the
        // lock/inode race exercised here. This PID cannot name a supported
        // process, so the unlocked record is deterministically stale.
        assert!(lease_is_stale(&path));

        assert!(!reclaim_stale_delegate_lease(&path));
        assert!(path.exists());

        drop(file);
        // Acquisition retries reclamation on every queue pass. Assert the
        // same eventual behavior here: Darwin can briefly continue reporting
        // advisory-lock contention immediately after the owning descriptor is
        // closed under heavy process pressure.
        let reclaim_deadline = Instant::now() + Duration::from_secs(2);
        while !reclaim_stale_delegate_lease(&path) && Instant::now() < reclaim_deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(!path.exists());
        let _ = std::fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn dropping_current_guard_removes_its_own_slot() {
        let root = lease_dir("own");
        let path = root.join("slot-0.lease");
        let owner = delegate_owner_token();
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        file.write_all(delegate_lease_record(&owner).as_bytes())
            .unwrap();
        let lease = DelegateLease {
            path: path.clone(),
            owner,
            _file: file,
        };

        drop(lease);

        assert!(!path.exists());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn incomplete_delegate_slot_uses_conservative_fault_recovery_grace() {
        let root = lease_dir("incomplete");
        let path = root.join("slot-0.lease");
        std::fs::write(&path, "owner=partially-written\n").unwrap();
        assert!(!lease_is_stale(&path));

        let file = OpenOptions::new().write(true).open(&path).unwrap();
        let old = std::fs::FileTimes::new().set_modified(std::time::SystemTime::UNIX_EPOCH);
        file.set_times(old).unwrap();
        assert!(lease_is_stale(&path));

        let _ = std::fs::remove_dir_all(root);
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn delegate_slot_rejects_reused_live_pid_birth_identity() {
        let root = lease_dir("pid-reuse");
        let path = root.join("slot-0.lease");
        std::fs::write(
            &path,
            format!(
                "owner=forged\npid={}\nbirth=not-this-process\n",
                std::process::id()
            ),
        )
        .unwrap();
        assert!(lease_is_stale(&path));

        let birth = resource_governor::current_process_birth_identity()
            .expect("supported platforms expose process birth identity");
        std::fs::write(
            &path,
            format!("owner=active\npid={}\nbirth={birth}\n", std::process::id()),
        )
        .unwrap();
        assert!(!lease_is_stale(&path));

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn delegate_slot_wait_has_no_default_deadline_but_observes_cancellation() {
        let root = lease_dir("unlimited-wait");
        let held = acquire_delegate_lease_with_limit(&root, None, &|| false, 1).unwrap();
        let started = Instant::now();
        let error = acquire_delegate_lease_with_limit(
            &root,
            None,
            &|| started.elapsed() >= Duration::from_millis(75),
            1,
        )
        .expect_err("cancellation must stop an otherwise-unbounded capacity wait");

        assert!(format!("{error:#}").contains("cancelled"));
        assert!(started.elapsed() >= Duration::from_millis(75));
        drop(held);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn explicit_delegate_slot_wait_timeout_is_enforced() {
        let root = lease_dir("explicit-wait");
        let held = acquire_delegate_lease_with_limit(&root, None, &|| false, 1).unwrap();
        let started = Instant::now();
        let error =
            acquire_delegate_lease_with_limit(&root, Some(Duration::from_millis(75)), &|| false, 1)
                .expect_err("an explicit capacity timeout must remain enforceable");

        assert!(format!("{error:#}").contains("timed out"));
        assert!(started.elapsed() < Duration::from_secs(2));
        drop(held);
        let _ = std::fs::remove_dir_all(root);
    }
}
