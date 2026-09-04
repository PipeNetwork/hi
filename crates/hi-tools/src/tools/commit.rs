//! Typed implementation of the compatibility `/commit` helper.

use std::path::Path;

use crate::ToolStatus;

/// Machine-readable outcome for a session-scoped Git commit.
///
/// Frontends must not infer success from [`Self::content`]. A failed commit can
/// still have changed the index or run repository filters/hooks, so the two
/// effect flags intentionally remain independent from [`Self::status`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommitOutcome {
    pub status: ToolStatus,
    pub content: String,
    pub workspace_may_have_changed: bool,
    pub external_effect_may_have_occurred: bool,
}

impl CommitOutcome {
    fn failed(content: impl Into<String>, mutation_started: bool) -> Self {
        Self {
            status: ToolStatus::Failed,
            content: content.into(),
            workspace_may_have_changed: mutation_started,
            // Git attributes, filters, and hooks can execute arbitrary code.
            // Once a mutating Git command starts, losing its response is not
            // safe grounds to replay the operation.
            external_effect_may_have_occurred: mutation_started,
        }
    }

    fn succeeded(content: String) -> Self {
        Self {
            status: ToolStatus::Succeeded,
            content,
            workspace_may_have_changed: true,
            external_effect_may_have_occurred: true,
        }
    }

    pub fn succeeded_typed(&self) -> bool {
        self.status == ToolStatus::Succeeded
    }
}

/// Compatibility entry point retained for callers that only render output.
/// New mutation paths must use [`commit_in_typed`] and inspect its status.
pub async fn commit_in(root: &Path, paths: &[String]) -> String {
    commit_in_typed(root, paths).await.content
}

/// Stage session-touched paths and commit them with a generated message.
///
/// This never runs `git add -A`; an empty or unsafe path set is rejected. The
/// returned status is authoritative even when its human-facing content looks
/// benign (for example, "nothing to commit").
pub async fn commit_in_typed(root: &Path, paths: &[String]) -> CommitOutcome {
    let in_tree = match super::run_git_operation(
        root,
        vec!["rev-parse".into(), "--is-inside-work-tree".into()],
    )
    .await
    {
        Ok(output) => {
            output.status == ToolStatus::Succeeded && output.outcome.stdout_summary.trim() == "true"
        }
        Err(error) => {
            return CommitOutcome::failed(format!("git not available: {error}"), false);
        }
    };
    if !in_tree {
        return CommitOutcome::failed("not a git repository", false);
    }

    let staged_paths = sanitize_commit_paths(root, paths);
    if staged_paths.is_empty() {
        return CommitOutcome::failed(
            "nothing this session changed — stage files yourself.",
            false,
        );
    }

    let mut add_args = vec!["add".into(), "--".into()];
    add_args.extend(staged_paths.iter().cloned());
    let add = match super::run_git_operation(root, add_args).await {
        Ok(output) => output,
        Err(error) => {
            return CommitOutcome::failed(format!("git add failed: {error}"), true);
        }
    };
    if add.status != ToolStatus::Succeeded {
        return CommitOutcome::failed(
            format!("git add failed: {}", add.model_content().trim()),
            true,
        );
    }

    match staged_diff_raw(root) {
        Ok(cached) if secret_in_staged_diff(&cached) => {
            unstage_paths(root, &staged_paths).await;
            return CommitOutcome::failed(
                "refusing to commit: staged diff looks like it contains secrets",
                true,
            );
        }
        Ok(_) => {}
        Err(error) => {
            unstage_paths(root, &staged_paths).await;
            return CommitOutcome::failed(format!("git diff failed: {error}"), true);
        }
    }

    let stat = match super::run_git_operation(
        root,
        vec![
            "--no-pager".into(),
            "diff".into(),
            "--cached".into(),
            "--name-only".into(),
        ],
    )
    .await
    {
        Ok(output) if output.status == ToolStatus::Succeeded => output.outcome.stdout_summary,
        Ok(output) => {
            unstage_paths(root, &staged_paths).await;
            return CommitOutcome::failed(
                format!("git diff failed: {}", output.model_content().trim()),
                true,
            );
        }
        Err(error) => {
            unstage_paths(root, &staged_paths).await;
            return CommitOutcome::failed(format!("git diff failed: {error}"), true);
        }
    };
    let files: Vec<&str> = stat
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect();
    if files.is_empty() {
        return CommitOutcome::failed("nothing to commit (working tree clean)", true);
    }

    let count = files.len();
    let subject = if count == 1 {
        format!("update {}", files[0])
    } else {
        format!("update {count} files")
    };
    const MAX_FILES_IN_BODY: usize = 40;
    let mut body = String::new();
    for file in files.iter().take(MAX_FILES_IN_BODY) {
        body.push_str("  - ");
        body.push_str(file);
        body.push('\n');
    }
    if count > MAX_FILES_IN_BODY {
        body.push_str(&format!("  - … and {} more\n", count - MAX_FILES_IN_BODY));
    }
    let message = if body.trim().is_empty() {
        subject.clone()
    } else {
        format!("{subject}\n\n{body}", body = body.trim_end())
    };

    let commit =
        match super::run_git_operation(root, vec!["commit".into(), "-m".into(), message]).await {
            Ok(output) => output,
            Err(error) => {
                return CommitOutcome::failed(format!("git commit failed: {error}"), true);
            }
        };
    if commit.status != ToolStatus::Succeeded {
        let detail = if commit.outcome.stderr_summary.trim().is_empty() {
            commit.outcome.stdout_summary.trim()
        } else {
            commit.outcome.stderr_summary.trim()
        };
        return CommitOutcome::failed(format!("git commit failed: {detail}"), true);
    }

    CommitOutcome::succeeded(format!(
        "staged {count} file{}\ncommitted: \"{subject}\"",
        if count == 1 { "" } else { "s" }
    ))
}

fn sanitize_commit_paths(root: &Path, paths: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for raw in paths {
        let trimmed = raw.trim();
        if trimmed.is_empty() || matches!(trimmed, "." | "./" | "-A") {
            continue;
        }
        let candidate = if Path::new(trimmed).is_absolute() {
            match Path::new(trimmed).strip_prefix(root) {
                Ok(relative) => relative.to_path_buf(),
                Err(_) => continue,
            }
        } else {
            Path::new(trimmed).to_path_buf()
        };
        if candidate
            .components()
            .any(|component| matches!(component, std::path::Component::ParentDir))
        {
            continue;
        }
        let normalized = candidate.to_string_lossy().replace('\\', "/");
        if !normalized.is_empty() && !out.contains(&normalized) {
            out.push(normalized);
        }
    }
    out
}

fn secret_in_staged_diff(diff: &str) -> bool {
    !matches!(
        hi_secrets::redact_secrets(diff),
        std::borrow::Cow::Borrowed(_)
    )
}

const MAX_STAGED_DIFF_BYTES: usize = 1_048_576;

fn staged_diff_raw(root: &Path) -> Result<String, String> {
    let output = std::process::Command::new("git")
        .args(["--no-pager", "diff", "--cached"])
        .current_dir(root)
        .output()
        .map_err(|error| error.to_string())?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_string());
    }
    if output.stdout.len() > MAX_STAGED_DIFF_BYTES {
        let prefix = String::from_utf8_lossy(&output.stdout[..MAX_STAGED_DIFF_BYTES]);
        if secret_in_staged_diff(&prefix) {
            return Ok(prefix.into_owned());
        }
        return Err("staged diff too large to scan for secrets".into());
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

async fn unstage_paths(root: &Path, paths: &[String]) {
    if paths.is_empty() {
        return;
    }
    let mut args = vec!["reset".into(), "HEAD".into(), "--".into()];
    args.extend(paths.iter().cloned());
    let _ = super::run_git_operation(root, args).await;
}
