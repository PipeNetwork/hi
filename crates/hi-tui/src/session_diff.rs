//! Synchronous git helpers behind the Changes pane (Ctrl-G).
//!
//! The pane paints its own colours via `diff_lines`, so everything here is
//! plain `--no-color` text. `git diff HEAD` never lists untracked files, yet
//! a file the agent just created is exactly what a session review must show,
//! so new files get an add-only diff synthesised from their contents. All
//! calls are synchronous: they run from key handlers and tool-result events,
//! and `git diff` over a handful of paths is fast.

use std::path::Path;
use std::process::Command;

/// Largest new file whose contents are inlined into the synthesised diff.
const MAX_SYNTH_FILE_BYTES: u64 = 256 * 1024;
/// Untracked files appended when showing the whole working tree; a fresh
/// clone with a large untracked build directory must not stall the UI.
const MAX_UNTRACKED_TREE_FILES: usize = 25;

fn git_stdout(root: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

/// `git diff HEAD` for `paths` (all tracked changes when empty), falling back
/// to an index diff in a repository with no commits yet.
fn tracked_diff(root: &Path, paths: &[String]) -> String {
    let mut with_head: Vec<&str> = vec!["--no-pager", "diff", "--no-color", "HEAD"];
    let mut without_head: Vec<&str> = vec!["--no-pager", "diff", "--no-color"];
    if !paths.is_empty() {
        with_head.push("--");
        without_head.push("--");
        for path in paths {
            with_head.push(path);
            without_head.push(path);
        }
    }
    git_stdout(root, &with_head)
        .or_else(|| git_stdout(root, &without_head))
        .unwrap_or_default()
}

/// Untracked, non-ignored files under `root`, optionally limited to `paths`.
fn untracked_files(root: &Path, paths: &[String]) -> Vec<String> {
    let mut args: Vec<&str> = vec!["ls-files", "--others", "--exclude-standard", "-z"];
    if !paths.is_empty() {
        args.push("--");
        args.extend(paths.iter().map(String::as_str));
    }
    git_stdout(root, &args)
        .unwrap_or_default()
        .split('\0')
        .filter(|p| !p.is_empty())
        .map(str::to_string)
        .collect()
}

/// An add-only unified diff for a file git does not know yet. Binary and
/// oversized files keep their header and a one-line note so the pane still
/// lists them.
pub(crate) fn synth_new_file_diff(root: &Path, path: &str) -> Option<String> {
    let absolute = root.join(path);
    let meta = std::fs::symlink_metadata(&absolute).ok()?;
    if !meta.is_file() {
        return None;
    }
    let mut out = format!(
        "diff --git a/{path} b/{path}\nnew file mode 100644\n--- /dev/null\n+++ b/{path}\n"
    );
    if meta.len() > MAX_SYNTH_FILE_BYTES {
        out.push_str(&format!(
            "({} KiB new file; preview omitted)\n",
            meta.len() / 1024
        ));
        return Some(out);
    }
    let bytes = std::fs::read(&absolute).ok()?;
    let Ok(text) = std::str::from_utf8(&bytes) else {
        out.push_str("(binary file)\n");
        return Some(out);
    };
    let lines: Vec<&str> = text
        .split_inclusive('\n')
        .map(|l| l.strip_suffix('\n').unwrap_or(l))
        .map(|l| l.strip_suffix('\r').unwrap_or(l))
        .collect();
    if lines.is_empty() {
        return Some(out);
    }
    out.push_str(&format!("@@ -0,0 +1,{} @@\n", lines.len()));
    for line in lines {
        out.push('+');
        out.push_str(line);
        out.push('\n');
    }
    Some(out)
}

fn append_new_files(root: &Path, out: &mut String, paths: &[String], limit: usize) {
    for path in paths.iter().take(limit) {
        if let Some(diff) = synth_new_file_diff(root, path) {
            if !out.is_empty() && !out.ends_with('\n') {
                out.push('\n');
            }
            out.push_str(&diff);
        }
    }
}

/// Every uncommitted change in the working tree: tracked edits plus (a
/// bounded number of) new files. Empty outside a git repository.
pub(crate) fn working_tree_diff_sync(root: &Path) -> String {
    let mut out = tracked_diff(root, &[]);
    let new_files = untracked_files(root, &[]);
    append_new_files(root, &mut out, &new_files, MAX_UNTRACKED_TREE_FILES);
    out
}

/// Uncommitted changes limited to `files` (workspace-relative): the running
/// diff of what a session (or one turn) edited, new files included.
pub(crate) fn session_diff_sync(root: &Path, files: &[String]) -> String {
    if files.is_empty() {
        return String::new();
    }
    let mut out = tracked_diff(root, files);
    let new_files = untracked_files(root, files);
    append_new_files(root, &mut out, &new_files, usize::MAX);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(label: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "hi-tui-session-diff-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn synthesised_new_file_diff_is_add_only() {
        let dir = temp_dir("text");
        std::fs::write(dir.join("new.rs"), "fn a() {}\nfn b() {}\n").unwrap();
        let diff = synth_new_file_diff(&dir, "new.rs").unwrap();
        assert_eq!(
            diff,
            "diff --git a/new.rs b/new.rs\nnew file mode 100644\n--- /dev/null\n+++ b/new.rs\n@@ -0,0 +1,2 @@\n+fn a() {}\n+fn b() {}\n"
        );
        let hunks = crate::review::parse_review_hunks(&diff);
        assert_eq!(hunks.len(), 1);
        assert_eq!(hunks[0].path, "new.rs");
        assert_eq!((hunks[0].start, hunks[0].end), (Some(1), Some(2)));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn synthesised_diff_notes_binary_and_missing_files() {
        let dir = temp_dir("binary");
        std::fs::write(dir.join("blob.bin"), [0xff, 0xfe, 0x00, 0x01]).unwrap();
        let diff = synth_new_file_diff(&dir, "blob.bin").unwrap();
        assert!(diff.ends_with("+++ b/blob.bin\n(binary file)\n"), "{diff}");
        assert!(synth_new_file_diff(&dir, "nope.txt").is_none());
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        assert!(
            synth_new_file_diff(&dir, "sub").is_none(),
            "directories are not files"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn outside_a_repo_the_diffs_are_empty() {
        let dir = temp_dir("norepo");
        // `git -C <tmp>` may still find an enclosing repository on a developer
        // machine; only assert on the pure-filter case, which never matches.
        assert_eq!(session_diff_sync(&dir, &[]), "");
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn git(dir: &Path, args: &[&str]) -> bool {
        Command::new("git")
            .arg("-C")
            .arg(dir)
            .args(args)
            .env("GIT_AUTHOR_NAME", "t")
            .env("GIT_AUTHOR_EMAIL", "t@example.com")
            .env("GIT_COMMITTER_NAME", "t")
            .env("GIT_COMMITTER_EMAIL", "t@example.com")
            .output()
            .is_ok_and(|o| o.status.success())
    }

    /// A tracked edit plus a brand-new file: `git diff HEAD` alone would miss
    /// the new file, which is exactly what a session review must show.
    #[test]
    fn session_diff_covers_tracked_edits_and_new_files() {
        let dir = temp_dir("repo");
        if !git(&dir, &["init", "-q"]) {
            eprintln!("git unavailable; skipping");
            return;
        }
        std::fs::write(dir.join("a.rs"), "one\ntwo\n").unwrap();
        std::fs::write(dir.join("ignored.log"), "noise\n").unwrap();
        std::fs::write(dir.join(".gitignore"), "*.log\n").unwrap();
        assert!(git(&dir, &["add", "."]));
        assert!(git(&dir, &["commit", "-q", "-m", "init"]));

        std::fs::write(dir.join("a.rs"), "one\nTWO\n").unwrap();
        std::fs::write(dir.join("b.rs"), "fresh\n").unwrap();
        std::fs::write(dir.join("untouched.rs"), "elsewhere\n").unwrap();

        let session = session_diff_sync(&dir, &["a.rs".into(), "b.rs".into()]);
        let doc = crate::review::ReviewDoc::build(
            session.trim(),
            crate::review::ReviewSource::Session,
            0,
        );
        let paths: Vec<&str> = doc.files.iter().map(|f| f.path.as_str()).collect();
        assert_eq!(paths, vec!["a.rs", "b.rs"], "{session}");
        assert_eq!(doc.files[0].status, crate::review::FileStatus::Modified);
        assert_eq!((doc.files[0].additions, doc.files[0].deletions), (1, 1));
        assert_eq!(doc.files[1].status, crate::review::FileStatus::Added);
        assert_eq!((doc.files[1].additions, doc.files[1].deletions), (1, 0));
        assert!(
            !session.contains("untouched.rs"),
            "session scope excludes files the session never edited"
        );

        let tree = working_tree_diff_sync(&dir);
        assert!(tree.contains("+++ b/a.rs") && tree.contains("+++ b/b.rs"));
        assert!(
            tree.contains("+++ b/untouched.rs"),
            "the working tree view lists every new file"
        );
        assert!(
            !tree.contains("ignored.log"),
            "ignored files stay out of the diff"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
