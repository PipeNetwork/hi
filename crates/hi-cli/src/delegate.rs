//! The CLI's write-`delegate` subagent runner.
//!
//! Isolates the child in a git worktree checked out to a snapshot of the parent's
//! *current* tree (so it sees uncommitted work), runs it as a `hi --subagent`
//! subprocess with the session's provider credentials, verifies in the worktree,
//! and applies only the verified diff back to the real tree. On failure nothing
//! touches the real tree — the worktree (and any git state the child created) is
//! thrown away.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use hi_agent::{DelegateOutcome, DelegateRunner};

/// Wall-clock cap on one delegate subagent subprocess. If the child hangs (e.g. a
/// stuck provider request), it's killed so the parent can't block forever. A
/// legitimate write+verify subtask finishes well within this; overridable via
/// `HI_DELEGATE_TIMEOUT_SECS`.
const DELEGATE_TIMEOUT_SECS: u64 = 600;

pub struct CliDelegateRunner {
    exe: PathBuf,
    provider: String,
    model: String,
    base_url: String,
    api_key: String,
    default_verify: Option<String>,
    max_steps: u32,
    max_verify: u32,
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
        max_steps: u32,
        max_verify: u32,
    ) -> Self {
        Self {
            exe,
            provider,
            model,
            base_url,
            api_key,
            default_verify,
            max_steps,
            max_verify,
            counter: AtomicU32::new(0),
        }
    }
}

#[async_trait]
impl DelegateRunner for CliDelegateRunner {
    async fn run(&self, task: &str, verify: Option<&str>) -> DelegateOutcome {
        let verify_cmd = verify
            .map(str::to_string)
            .or_else(|| self.default_verify.clone());

        if !hi_tools::worktree::in_git_repo() {
            return outcome(false, "delegate unavailable: not in a git repository.");
        }
        // Snapshot the parent's current tree (incl. uncommitted work) as the base
        // the child branches from and we diff against.
        let checkpoint = match hi_tools::checkpoint::create(Path::new(".")).await {
            Some(sha) => sha,
            None => {
                return outcome(
                    false,
                    "delegate unavailable: couldn't snapshot the working tree.",
                );
            }
        };

        // The rest is blocking git + a long-running subprocess — run it off the
        // async executor.
        let idx = self.counter.fetch_add(1, Ordering::Relaxed);
        let exe = self.exe.clone();
        let provider = self.provider.clone();
        let model = self.model.clone();
        let base_url = self.base_url.clone();
        let api_key = self.api_key.clone();
        let max_steps = self.max_steps;
        let max_verify = self.max_verify;
        let task = task.to_string();

        tokio::task::spawn_blocking(move || {
            run_blocking(
                &exe,
                &provider,
                &model,
                &base_url,
                &api_key,
                &task,
                verify_cmd,
                max_steps,
                max_verify,
                &checkpoint,
                idx,
            )
        })
        .await
        .unwrap_or_else(|err| outcome(false, &format!("delegate task failed to run: {err}")))
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
    verify_cmd: Option<String>,
    max_steps: u32,
    max_verify: u32,
    checkpoint: &str,
    idx: u32,
) -> DelegateOutcome {
    let worktree = hi_tools::worktree::worktree_path("delegate", idx);
    if let Err(err) = hi_tools::worktree::add_worktree(&worktree, checkpoint) {
        return outcome(
            false,
            &format!("delegate failed to create an isolated worktree: {err}"),
        );
    }

    // Run the child `hi` in the worktree. `--subagent` forbids it from spawning
    // further subagents (depth ≤ 1) and implies no session save.
    let prompt = child_prompt(task, verify_cmd.as_deref());
    let mut cmd = Command::new(exe);
    cmd.current_dir(&worktree)
        .env("HI_API_KEY", api_key)
        // No pipes: we gate on the ground-truth verify + the worktree diff, not the
        // child's stdout — and unread pipes would deadlock the timeout wait.
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .args([
            "--subagent",
            "--provider",
            provider,
            "--model",
            model,
            "--base-url",
            base_url,
            "--no-save",
            "--temperature",
            "0",
            "--max-steps",
            &max_steps.to_string(),
        ]);
    if let Some(v) = &verify_cmd {
        cmd.args(["--verify", v, "--max-verify", &max_verify.to_string()]);
    }
    cmd.arg(&prompt);

    let result = match run_with_timeout(cmd, delegate_timeout_secs()) {
        Err(err) => outcome(false, &format!("delegate {err}")),
        Ok(()) => decide(&worktree, checkpoint, verify_cmd.as_deref()),
    };
    // Always tear the worktree down — the real tree only ever sees an applied diff.
    hi_tools::worktree::cleanup(&[worktree]);
    result
}

fn delegate_timeout_secs() -> u64 {
    std::env::var("HI_DELEGATE_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&s| s > 0)
        .unwrap_or(DELEGATE_TIMEOUT_SECS)
}

/// Run the child to completion, or kill it after `secs`. `Err(msg)` on launch
/// failure or timeout (the worktree is discarded either way, so nothing is applied).
fn run_with_timeout(mut cmd: Command, secs: u64) -> Result<(), String> {
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("couldn't launch the subagent: {e}"))?;
    let deadline = Instant::now() + Duration::from_secs(secs);
    loop {
        match child.try_wait() {
            Ok(Some(_status)) => return Ok(()),
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(format!(
                        "timed out after {secs}s (subagent killed); nothing applied. Try a smaller \
                         task or check the provider."
                    ));
                }
                std::thread::sleep(Duration::from_millis(250));
            }
            Err(e) => return Err(format!("subagent wait failed: {e}")),
        }
    }
}

/// Ground-truth gate (re-run verify ourselves) + apply-back.
fn decide(worktree: &Path, checkpoint: &str, verify_cmd: Option<&str>) -> DelegateOutcome {
    let passed = match verify_cmd {
        Some(v) => hi_tools::worktree::verify_passes(worktree, v),
        None => true,
    };
    let changed = hi_tools::worktree::changed_files(worktree, checkpoint);

    if !passed {
        return outcome(
            false,
            &format!(
                "delegate rolled back — verification (`{}`) did not pass; the working tree is \
                 unchanged. Refine the task or implement it directly.",
                verify_cmd.unwrap_or_default()
            ),
        );
    }
    if changed.is_empty() {
        return outcome(false, "delegate made no changes; nothing to apply.");
    }
    match hi_tools::worktree::apply_changes(worktree, checkpoint) {
        Ok(_) => {
            let note = if verify_cmd.is_some() {
                " · verification passed"
            } else {
                ""
            };
            DelegateOutcome {
                applied: true,
                summary: format!(
                    "delegate applied — {} file(s) changed{note}: {}",
                    changed.len(),
                    changed.join(", ")
                ),
                changed_files: changed,
            }
        }
        Err(err) => outcome(
            false,
            &format!(
                "delegate produced changes that failed to apply back cleanly: {err}. The working \
                 tree is unchanged."
            ),
        ),
    }
}

fn outcome(applied: bool, summary: &str) -> DelegateOutcome {
    DelegateOutcome {
        applied,
        changed_files: Vec::new(),
        summary: summary.to_string(),
    }
}

fn child_prompt(task: &str, verify: Option<&str>) -> String {
    let check = match verify {
        Some(v) => format!(" Your changes must make `{v}` pass."),
        None => String::new(),
    };
    format!(
        "Implement this self-contained subtask by editing files and running commands as needed, then \
         confirm it works.{check}\n\nTask: {task}"
    )
}
