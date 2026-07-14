//! The CLI's write-`delegate` subagent runner.
//!
//! A delegate works in an isolated Git worktree based on an immutable snapshot
//! of the parent's current tree. Only a typed successful child outcome with a
//! non-empty, independently verified diff is eligible for transactional merge.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use async_trait::async_trait;
use hi_agent::{DelegateOutcome, DelegateRunner};
use hi_tools::ToolStatus;

use crate::candidate_gate::{
    independently_verify_candidate, inspect_child_report, repository_root, same_paths,
    staged_candidate_diff,
};
use crate::candidate_merge::apply_candidate_and_reverify;

const DELEGATE_TIMEOUT_SECS: u64 = 600;

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
    async fn run(&self, task: &str, verify: Option<&str>) -> DelegateOutcome {
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
        let provider = self.provider.clone();
        let model = self.model.clone();
        let base_url = self.base_url.clone();
        let api_key = self.api_key.clone();
        let max_steps = self.max_steps;
        let max_verify = self.max_verify;
        let task = task.to_string();
        let workspace_root = self.workspace_root.clone();
        let state_root = self.state_root.clone();

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
) -> DelegateOutcome {
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
    let report_path = artifact_dir.join("report.json");
    let log_path = artifact_dir.join("child.log");

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
    if let Some(max_steps) = max_steps {
        arguments.push("--max-steps".into());
        arguments.push(max_steps.to_string().into());
    }
    arguments.push(prompt.into());

    let execution = crate::child_process::run(
        &worktree,
        exe,
        arguments,
        vec![
            ("HI_FORCE_API_KEY".into(), api_key.into()),
            ("HI_API_KEY".into(), api_key.into()),
        ],
        Duration::from_secs(delegate_timeout_secs()),
        &log_path,
    );
    let result = match execution {
        Ok(execution) if execution.status == ToolStatus::Succeeded => decide(
            &worktree,
            checkpoint,
            verify_cmd,
            &report_path,
            &artifact_dir,
            workspace_root,
            state_root,
        ),
        Ok(execution) => {
            let status = execution.status;
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
    hi_tools::worktree::cleanup(repo_root, &[worktree_root]);
    result
}

fn delegate_artifacts_dir(state_root: &Path, idx: u32) -> PathBuf {
    let pid = std::process::id();
    state_root
        .join("delegate-artifacts")
        .join(pid.to_string())
        .join(idx.to_string())
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
    if !same_paths(&child.changed_files, &before_check.display_paths) {
        return outcome(
            ToolStatus::Failed,
            "delegate report did not match its exact worktree diff; nothing was applied.",
        );
    }

    let after_check = match independently_verify_candidate(worktree, checkpoint, verify_cmd) {
        Ok(diff) => diff,
        Err(error) => {
            return outcome(
                ToolStatus::Failed,
                &format!(
                    "delegate rolled back before merge — independent verification failed: {error:#}. The working tree is unchanged."
                ),
            );
        }
    };
    if !same_paths(&child.changed_files, &after_check.display_paths) {
        return outcome(
            ToolStatus::Failed,
            "delegate's independently verified diff no longer matched its report; nothing was applied.",
        );
    }

    match apply_candidate_and_reverify(worktree, checkpoint, destination, state_root, verify_cmd) {
        Ok(applied) => DelegateOutcome {
            status: ToolStatus::Succeeded,
            applied: true,
            summary: format!(
                "delegate applied — {} file(s) changed · child outcome passed · independent and destination verification passed: {}",
                applied.len(),
                applied.join(", ")
            ),
            changed_files: applied,
        },
        Err(error) => outcome(
            ToolStatus::Failed,
            &format!(
                "delegate changes were not accepted: {error:#}. See artifacts at {}",
                artifact_dir.display()
            ),
        ),
    }
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
