//! Compact git identity for the per-turn volatile context and `/btw`.
//!
//! Prompt injection is **workspace-strict**: a nested directory under some
//! other checkout must not inherit that checkout's GitHub origin. `/btw`
//! still reports an enclosing repo so side questions keep working.

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Hard cap for the volatile `[Git identity]` block. Individual fields are
/// already tiny; this stops a pathological origin or branch name from
/// crowding out memory / task context.
const MAX_IDENTITY_CHARS: usize = 400;

const MAX_REPORTED_DIRTY_PATHS: usize = 1_000;

/// Sanitized `host/owner/repo` (lowercased, no `.git`, no userinfo).
///
/// Matches the portal fingerprint helper in `hi-cli`: credentials in
/// `https://user:token@host/path.git` never survive.
pub fn normalize_git_remote(remote: &str) -> Option<String> {
    let value = remote.trim().trim_end_matches('/').trim_end_matches(".git");
    let host_path = if let Some(rest) = value.split_once("://").map(|(_, rest)| rest) {
        let rest = rest.rsplit_once('@').map(|(_, rest)| rest).unwrap_or(rest);
        rest.split_once('/')?
    } else if let Some((user_host, path)) = value.split_once(':') {
        let host = user_host
            .rsplit_once('@')
            .map(|(_, host)| host)
            .unwrap_or(user_host);
        (host, path)
    } else {
        return None;
    };
    Some(format!(
        "{}/{}",
        host_path.0.to_ascii_lowercase(),
        host_path.1.trim_matches('/').to_ascii_lowercase()
    ))
}

/// Prompt fragment for the volatile context. `None` when `root` is not the
/// git toplevel (nested dirs, non-repos, missing git).
pub fn prompt_section(root: &Path) -> Option<String> {
    if !is_workspace_git_toplevel(root) {
        return None;
    }
    let mut lines = vec!["[Git identity]".to_string()];
    push_identity_lines(&mut lines, root, /*strict_origin=*/ true);
    if lines.len() == 1 {
        return None;
    }
    Some(clip_identity(&lines.join("\n")))
}

/// Cheap git facts for `/btw`. Failures are silent — a missing git binary or
/// non-repo workspace just omits the lines. Unlike [`prompt_section`], this
/// reports an enclosing repository (same as `git` from a nested cwd).
pub(crate) fn btw_lines(root: &Path) -> Vec<String> {
    if !is_inside_work_tree(root) {
        return vec!["- git: not a repository".into()];
    }
    let mut lines = Vec::new();
    push_identity_lines(&mut lines, root, /*strict_origin=*/ false);
    push_btw_history_lines(&mut lines, root);
    lines
}

fn push_identity_lines(lines: &mut Vec<String>, root: &Path, strict_origin: bool) {
    if let Some(branch) = git_stdout(root, &["rev-parse", "--abbrev-ref", "HEAD"])
        && !branch.is_empty()
    {
        lines.push(format!("- git branch: {branch}"));
    }
    if let Some(head) = git_stdout(root, &["rev-parse", "--short", "HEAD"])
        && !head.is_empty()
    {
        lines.push(format!("- git HEAD: {head}"));
    }
    match git_dirty_summary(root) {
        Some(status) if status == "0 path(s)" => lines.push("- git dirty: clean".into()),
        Some(status) => lines.push(format!("- git dirty: {status}")),
        None => {}
    }
    if let Some(origin) = sanitized_origin(root) {
        lines.push(format!("- git origin: {origin}"));
        if let Some(url) = public_https_url(&origin) {
            lines.push(format!("- git origin url: {url}"));
        }
    } else if !strict_origin {
        // `/btw` can mention a missing origin; the prompt must not invent one.
    }
    if is_linked_worktree(root) {
        lines.push("- git worktree: linked".into());
    }
}

fn push_btw_history_lines(lines: &mut Vec<String>, root: &Path) {
    // Oldest root commit — answers "how old is this project?" without tools.
    // `log --reverse -n1` is wrong (max-count applies before reverse); walk
    // root commits instead and pick the earliest by author date.
    if let Some(roots) = git_stdout(
        root,
        &[
            "log",
            "--max-parents=0",
            "--format=%aI%x00%h %ad %s",
            "--date=short",
            "HEAD",
        ],
    ) {
        let mut best: Option<(String, String)> = None; // (sort_key, display)
        for record in roots.lines().filter(|line| !line.is_empty()) {
            let Some((sort_key, display)) = record.split_once('\0') else {
                continue;
            };
            if display.is_empty() {
                continue;
            }
            match &best {
                Some((prev, _)) if sort_key >= prev.as_str() => {}
                _ => best = Some((sort_key.to_string(), display.to_string())),
            }
        }
        if let Some((_, display)) = best {
            lines.push(format!("- git first commit: {display}"));
        }
    }
    if let Some(latest) = git_stdout(root, &["log", "-1", "--format=%h %ad %s", "--date=short"])
        && !latest.is_empty()
    {
        lines.push(format!("- git latest commit: {latest}"));
    }
}

fn sanitized_origin(root: &Path) -> Option<String> {
    let raw = git_stdout(root, &["config", "--get", "remote.origin.url"])?;
    if raw.is_empty() {
        return None;
    }
    normalize_git_remote(&raw)
}

fn public_https_url(origin: &str) -> Option<String> {
    let host = origin.split('/').next().unwrap_or("");
    if !is_public_git_host(host) {
        return None;
    }
    Some(format!("https://{origin}"))
}

fn is_public_git_host(host: &str) -> bool {
    let host = host.trim().trim_end_matches('.').to_ascii_lowercase();
    host == "github.com"
        || host == "gitlab.com"
        || host == "bitbucket.org"
        || host == "codeberg.org"
        || host == "sr.ht"
        || host.ends_with(".github.com")
        || host.ends_with(".gitlab.com")
}

fn is_workspace_git_toplevel(root: &Path) -> bool {
    if !is_inside_work_tree(root) {
        return false;
    }
    let Some(toplevel) = git_stdout(root, &["rev-parse", "--show-toplevel"]) else {
        return false;
    };
    same_path(Path::new(&toplevel), root)
}

fn is_inside_work_tree(root: &Path) -> bool {
    git_stdout(root, &["rev-parse", "--is-inside-work-tree"])
        .map(|s| s == "true")
        .unwrap_or(false)
}

fn is_linked_worktree(root: &Path) -> bool {
    let Some(git_dir) = git_stdout(root, &["rev-parse", "--git-dir"]) else {
        return false;
    };
    let git_dir = PathBuf::from(&git_dir);
    let resolved = if git_dir.is_absolute() {
        git_dir
    } else {
        root.join(git_dir)
    };
    !same_path(&resolved, &root.join(".git"))
}

fn same_path(left: &Path, right: &Path) -> bool {
    match (canonicalize_or_clone(left), canonicalize_or_clone(right)) {
        (Some(a), Some(b)) => a == b,
        _ => left == right,
    }
}

fn canonicalize_or_clone(path: &Path) -> Option<PathBuf> {
    std::fs::canonicalize(path).ok().or_else(|| {
        if path.as_os_str().is_empty() {
            None
        } else {
            Some(path.to_path_buf())
        }
    })
}

fn clip_identity(text: &str) -> String {
    if text.chars().count() <= MAX_IDENTITY_CHARS {
        return text.to_string();
    }
    let clipped: String = text
        .chars()
        .take(MAX_IDENTITY_CHARS.saturating_sub(1))
        .collect();
    format!("{clipped}…")
}

fn git_command(root: &Path) -> Command {
    let mut cmd = Command::new("git");
    cmd.current_dir(root)
        .env("GIT_OPTIONAL_LOCKS", "0")
        .env("GIT_TERMINAL_PROMPT", "0")
        .env_remove("GIT_DIR")
        .env_remove("GIT_WORK_TREE")
        .env_remove("GIT_COMMON_DIR")
        .env_remove("GIT_INDEX_FILE");
    cmd
}

/// Run `git` in `root`. Returns `Some` on success (stdout may be empty).
fn git_stdout(root: &Path, args: &[&str]) -> Option<String> {
    let output = git_command(root).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// Count dirty paths without buffering the whole porcelain listing. A large
/// generated tree can contain hundreds of thousands of paths; the snapshot
/// only needs a compact count and should not retain that complete listing.
fn git_dirty_summary(root: &Path) -> Option<String> {
    let mut child = git_command(root)
        .args(["status", "--porcelain"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let stdout = child.stdout.take()?;
    let mut reader = BufReader::new(stdout);
    let mut count = 0usize;
    let mut line = String::new();
    while count < MAX_REPORTED_DIRTY_PATHS {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => count += 1,
            Err(_) => return None,
        }
    }
    let capped = count == MAX_REPORTED_DIRTY_PATHS && !line.is_empty();
    if capped {
        let _ = child.kill();
    }
    let status = child.wait().ok()?;
    if !status.success() && !capped {
        return None;
    }
    Some(if capped {
        format!("{MAX_REPORTED_DIRTY_PATHS}+ path(s)")
    } else {
        format!("{count} path(s)")
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_root(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "hi-git-id-{tag}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ))
    }

    struct TempRepo {
        root: PathBuf,
    }

    impl TempRepo {
        fn new(tag: &str) -> Self {
            let root = unique_root(tag);
            let _ = std::fs::remove_dir_all(&root);
            std::fs::create_dir_all(&root).unwrap();
            Self { root }
        }

        fn run(&self, args: &[&str]) {
            let out = git_env(Command::new("git").args(args).current_dir(&self.root))
                .output()
                .expect("git");
            assert!(
                out.status.success(),
                "git {args:?} failed: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        }

        fn commit(&self, msg: &str, when: &str) {
            let out = git_env(
                Command::new("git")
                    .args(["commit", "-q", "-m", msg])
                    .current_dir(&self.root),
            )
            .env("GIT_AUTHOR_DATE", when)
            .env("GIT_COMMITTER_DATE", when)
            .output()
            .expect("git commit");
            assert!(
                out.status.success(),
                "commit failed: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
    }

    impl Drop for TempRepo {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    fn git_env(cmd: &mut Command) -> &mut Command {
        cmd.env("GIT_AUTHOR_NAME", "btw")
            .env("GIT_AUTHOR_EMAIL", "btw@example.com")
            .env("GIT_COMMITTER_NAME", "btw")
            .env("GIT_COMMITTER_EMAIL", "btw@example.com")
            .env("GIT_OPTIONAL_LOCKS", "0")
            .env("GIT_TERMINAL_PROMPT", "0")
            .env_remove("GIT_DIR")
            .env_remove("GIT_WORK_TREE")
    }

    #[test]
    fn normalize_strips_userinfo_and_git_suffix() {
        assert_eq!(
            normalize_git_remote(
                "https://user:ghp_notareal@github.com/Example-Org/Quality-Fixture.git"
            ),
            Some("github.com/example-org/quality-fixture".into())
        );
        assert_eq!(
            normalize_git_remote("git@github.com:PipeNetwork/hi.git"),
            Some("github.com/pipenetwork/hi".into())
        );
        assert_eq!(
            normalize_git_remote("ssh://git@gitlab.com/org/repo"),
            Some("gitlab.com/org/repo".into())
        );
    }

    #[test]
    fn prompt_section_omits_non_repo() {
        let repo = TempRepo::new("none");
        assert_eq!(prompt_section(&repo.root), None);
        assert_eq!(
            btw_lines(&repo.root),
            vec!["- git: not a repository".to_string()]
        );
    }

    #[test]
    fn prompt_section_strips_credentialed_origin() {
        let repo = TempRepo::new("creds");
        repo.run(&["init", "-q", "-b", "main"]);
        std::fs::write(repo.root.join("README"), "hi\n").unwrap();
        repo.run(&["add", "README"]);
        repo.commit("initial commit", "2020-01-15T12:00:00");
        repo.run(&[
            "remote",
            "add",
            "origin",
            "https://user:ghp_notareal@github.com/Example-Org/Quality-Fixture.git",
        ]);

        let section = prompt_section(&repo.root).expect("identity section");
        assert!(
            section.contains("github.com/example-org/quality-fixture"),
            "sanitized origin missing: {section}"
        );
        assert!(
            section.contains("https://github.com/example-org/quality-fixture"),
            "public url missing: {section}"
        );
        assert!(
            !section.contains("user") && !section.contains("ghp_") && !section.contains("notareal"),
            "credential leaked: {section}"
        );
        assert!(
            !section.contains("https://user:"),
            "raw url leaked: {section}"
        );
        assert!(section.contains("- git branch: main"), "{section}");
        assert!(section.contains("- git dirty: clean"), "{section}");
        assert!(section.starts_with("[Git identity]\n"), "{section}");
    }

    #[test]
    fn prompt_section_omits_nested_dir_inside_parent_repo() {
        let repo = TempRepo::new("nested");
        repo.run(&["init", "-q", "-b", "main"]);
        std::fs::write(repo.root.join("README"), "hi\n").unwrap();
        repo.run(&["add", "README"]);
        repo.commit("initial commit", "2020-01-15T12:00:00");
        repo.run(&[
            "remote",
            "add",
            "origin",
            "https://github.com/Example-Org/Quality-Fixture.git",
        ]);
        let nested = repo.root.join("pkg");
        std::fs::create_dir_all(&nested).unwrap();

        assert!(
            prompt_section(&nested).is_none(),
            "nested dir must not inherit parent origin"
        );
        assert!(
            prompt_section(&repo.root).is_some(),
            "toplevel should still emit identity"
        );
    }

    #[test]
    fn btw_lines_includes_first_and_latest_commit() {
        let repo = TempRepo::new("btw");
        repo.run(&["init", "-q", "-b", "main"]);
        std::fs::write(repo.root.join("README"), "hi\n").unwrap();
        repo.run(&["add", "README"]);
        repo.commit("initial commit", "2020-01-15T12:00:00");
        std::fs::write(repo.root.join("README"), "hi again\n").unwrap();
        repo.run(&["add", "README"]);
        repo.commit("second commit", "2021-06-01T12:00:00");

        let joined = btw_lines(&repo.root).join("\n");
        assert!(
            joined.contains("- git branch: main"),
            "branch missing: {joined}"
        );
        assert!(joined.contains("- git HEAD:"), "HEAD missing: {joined}");
        assert!(
            joined.contains("- git first commit:") && joined.contains("initial commit"),
            "first commit missing: {joined}"
        );
        assert!(
            joined.contains("- git latest commit:") && joined.contains("second commit"),
            "latest commit missing: {joined}"
        );
        assert!(
            joined.contains("- git dirty: clean"),
            "expected clean tree: {joined}"
        );
        assert!(
            !joined.contains("[Git identity]"),
            "btw lines should stay bullet-only: {joined}"
        );
    }
}
