//! The recent-work scope: what git reports as changed, with line counts,
//! and how a big scope is grouped into turns.

use super::*;
use std::fs;

/// `git …` in `root`; panics on failure so a broken fixture is loud.
pub(crate) fn run_git(root: &Path, args: &[&str]) {
    let output = Command::new("git")
        .args(["-c", "user.email=t@example.com", "-c", "user.name=t"])
        .args(args)
        .current_dir(root)
        .output()
        .expect("git runs");
    assert!(
        output.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

pub(crate) fn init_repo(root: &Path) {
    run_git(root, &["init", "-q"]);
    run_git(root, &["config", "commit.gpgsign", "false"]);
}

pub(crate) fn commit_all(root: &Path, message: &str) {
    run_git(root, &["add", "-A"]);
    run_git(root, &["commit", "-q", "-m", message]);
}

fn populate(root: &Path, files: usize) {
    fs::create_dir_all(root.join("src")).unwrap();
    for index in 0..files {
        fs::write(root.join(format!("src/f{index}.rs")), "").unwrap();
    }
}

#[test]
fn uncommitted_work_else_the_last_commit_with_line_counts() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    init_repo(root);
    populate(root, 5);
    fs::write(root.join("README.md"), "# App\nline\n").unwrap();

    // Nothing committed yet: everything is untracked work, counted by lines.
    let scope = git_scope(root).expect("untracked files are recent work");
    assert_eq!(scope.source, GitSource::Uncommitted);
    assert!(scope.files.contains(&"README.md".to_string()));
    assert!(scope.files.contains(&"src/f0.rs".to_string()));
    assert_eq!(scope.files.len(), 6, "five sources and the README");
    let readme = scope.numstat_for("README.md").unwrap();
    assert_eq!((readme.added, readme.removed), (Some(2), Some(0)));
    assert!(readme.untracked);
    assert_eq!(readme.label(), "untracked, 2 lines");
    assert_eq!(scope.summary(), "6 uncommitted files (git status, +2 −0)");

    commit_all(root, "initial import");
    let scope = git_scope(root).expect("a clean tree audits the last commit");
    match &scope.source {
        GitSource::LastCommit {
            short_hash,
            subject,
        } => {
            assert_eq!(subject, "initial import");
            assert!(!short_hash.is_empty());
            assert_eq!(
                scope.summary(),
                format!("last commit {short_hash} (6 files, +2 −0)")
            );
        }
        other => panic!("expected the last commit, got {other:?}"),
    }
    assert_eq!(
        scope.numstat_for("README.md").map(GitNumstat::label),
        Some("+2 −0".into())
    );

    // A modified tracked file, a rename, a new file, and a deletion: the
    // uncommitted set wins, the old rename path and the deleted file are
    // not listed.
    fs::write(root.join("src/f1.rs"), "one\ntwo\nthree\n").unwrap();
    run_git(root, &["mv", "src/f2.rs", "src/moved.rs"]);
    fs::write(root.join("src/new.rs"), "fn new() {}").unwrap();
    fs::remove_file(root.join("src/f3.rs")).unwrap();
    let scope = git_scope(root).unwrap();
    assert_eq!(scope.source, GitSource::Uncommitted);
    assert_eq!(
        scope.files,
        vec![
            "src/f1.rs".to_string(),
            "src/moved.rs".to_string(),
            "src/new.rs".to_string()
        ]
    );
    assert_eq!(
        scope.numstat_for("src/f1.rs").map(GitNumstat::label),
        Some("+3 −0".into())
    );
    assert_eq!(
        scope.numstat_for("src/moved.rs").map(GitNumstat::label),
        Some("+0 −0".into()),
        "a rename is counted under its new name"
    );
    assert_eq!(
        scope.numstat_for("src/new.rs").map(GitNumstat::label),
        Some("untracked, 1 lines".into()),
        "a file without a trailing newline still has its last line"
    );
    assert!(
        scope.numstat_for("src/f3.rs").is_none(),
        "deleted: not on disk"
    );
    assert_eq!(scope.summary(), "3 uncommitted files (git status, +4 −0)");
    assert_eq!(scope.chunk_files(0), scope.files, "one turn: every file");

    // A commit that only deletes leaves nothing to audit.
    commit_all(root, "changes");
    fs::remove_file(root.join("src/new.rs")).unwrap();
    commit_all(root, "drop new.rs");
    assert_eq!(git_scope(root), Err(GitEmpty::Nothing));

    // Binary content is not line-counted.
    fs::write(root.join("blob.bin"), [0u8, 1, 2, 3]).unwrap();
    let scope = git_scope(root).unwrap();
    assert_eq!(scope.files, vec!["blob.bin".to_string()]);
    assert!(scope.numstat_for("blob.bin").is_none());
    assert_eq!(scope.summary(), "1 uncommitted file (git status)");
}

#[test]
fn workspace_inside_a_repository_gets_workspace_relative_paths() {
    let dir = tempfile::tempdir().unwrap();
    let repo = dir.path();
    init_repo(repo);
    let workspace = repo.join("app");
    fs::create_dir_all(workspace.join("src")).unwrap();
    fs::write(workspace.join("src/lib.rs"), "").unwrap();
    fs::write(repo.join("elsewhere.txt"), "").unwrap();
    let scope = git_scope(&workspace).unwrap();
    assert_eq!(
        scope.files,
        vec!["src/lib.rs".to_string()],
        "porcelain paths are repo-relative; the audit sees workspace-relative ones and nothing outside"
    );
    assert_eq!(
        churn_by_path(&workspace),
        HashMap::new(),
        "no commits: no churn"
    );
    commit_all(repo, "init");
    let churn = churn_by_path(&workspace);
    assert_eq!(churn.get("src/lib.rs"), Some(&1));
    assert_eq!(churn.get("elsewhere.txt"), None, "outside the workspace");
}

#[cfg(unix)]
#[test]
fn the_scope_never_runs_the_checkouts_fsmonitor_hook() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("repo");
    fs::create_dir_all(&root).unwrap();
    init_repo(&root);
    fs::write(root.join("a.rs"), "a\n").unwrap();
    commit_all(&root, "init");
    let marker = dir.path().join("ran");
    let hook = dir.path().join("hook.sh");
    fs::write(&hook, format!("#!/bin/sh\ntouch '{}'\n", marker.display())).unwrap();
    fs::set_permissions(&hook, fs::Permissions::from_mode(0o755)).unwrap();
    let hook = hook.to_string_lossy().into_owned();
    run_git(&root, &["config", "core.fsmonitor", hook.as_str()]);
    fs::write(root.join("a.rs"), "a\nb\n").unwrap();

    let scope = git_scope(&root).expect("the edit is recent work");
    assert_eq!(scope.files, vec!["a.rs".to_string()]);
    assert_eq!(
        scope.numstat_for("a.rs").and_then(|stat| stat.added),
        Some(1)
    );
    assert!(
        !marker.exists(),
        "`git status` ran the repository's fsmonitor hook"
    );
}

#[test]
fn status_paths_skip_rename_sources_and_short_fields() {
    let status = "R  src/moved.rs\0src/f2.rs\0 M src/f1.rs\0?? src/new.rs\0 D src/f3.rs\0\0";
    assert_eq!(
        status_paths(status).collect::<Vec<_>>(),
        vec!["src/moved.rs", "src/f1.rs", "src/new.rs", "src/f3.rs"]
    );
    assert_eq!(status_paths("").count(), 0);
}

#[test]
fn numstat_entries_read_counts_binaries_and_renames() {
    // `git diff --numstat -z`: `added\tremoved\tpath\0`, `-` for a binary,
    // and a rename as `added\tremoved\t\0old\0new\0`.
    let listed = "3\t1\tsrc/a.rs\0-\t-\timg.png\u{0}0\t0\t\0old.rs\0new.rs\0";
    assert_eq!(
        numstat_entries_for_test(listed),
        vec![
            (Some(3), Some(1), "src/a.rs".to_string()),
            (None, None, "img.png".to_string()),
            (Some(0), Some(0), "new.rs".to_string()),
        ]
    );
    assert!(numstat_entries_for_test("").is_empty());
    assert!(
        numstat_entries_for_test("garbage").is_empty(),
        "a field without two tabs is skipped"
    );
}

#[test]
fn group_splits_a_big_scope_by_directory_or_crate_within_the_budget() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let mut files = Vec::new();
    for file in ["Cargo.toml", "README.md"] {
        fs::write(root.join(file), "").unwrap();
        files.push(file.to_string());
    }
    for crate_name in ["hi-a", "hi-b"] {
        for index in 0..5 {
            let path = format!("crates/{crate_name}/src/m{index}.rs");
            fs::create_dir_all(root.join(&path).parent().unwrap()).unwrap();
            fs::write(root.join(&path), "").unwrap();
            files.push(path);
        }
    }
    for index in 0..7 {
        let path = format!("docs/d{index}.md");
        fs::create_dir_all(root.join("docs")).unwrap();
        fs::write(root.join(&path), "").unwrap();
        files.push(path);
    }
    files.sort();
    let mut scope = GitScope {
        source: GitSource::Uncommitted,
        files: files.clone(),
        numstat: Vec::new(),
        groups: Vec::new(),
    };
    assert!(scope.group(root, 20).is_empty(), "19 files fit one turn");
    assert!(scope.groups.is_empty());

    let labels = scope.group(root, 8);
    assert_eq!(
        labels,
        vec![
            "., crates/hi-a".to_string(),
            "crates/hi-b".to_string(),
            "docs".to_string()
        ],
        "crates/ is a container so its crates are the units; the two root files pack with hi-a (7 <= 8), \
         hi-b would overflow and docs fills a turn of its own"
    );
    assert_eq!(scope.groups.len(), 3);
    assert_eq!(
        scope.groups[0][..2],
        ["Cargo.toml".to_string(), "README.md".to_string()]
    );
    assert_eq!(scope.groups[0].len(), 7);
    assert_eq!(scope.groups[1].len(), 5);
    assert_eq!(scope.groups[2].len(), 7);
    assert_eq!(scope.chunk_files(2), scope.groups[2]);
    assert!(scope.chunk_files(9).is_empty());

    let labels = scope.group(root, 12);
    assert_eq!(
        labels,
        vec![
            "., crates/hi-a, crates/hi-b".to_string(),
            "docs".to_string()
        ],
        "with room for twelve the root files and both crates pack into one turn"
    );
    assert_eq!(scope.groups[0].len(), 12);
    assert_eq!(scope.groups[1].len(), 7);

    let labels = scope.group(root, 4);
    assert_eq!(
        labels,
        vec![
            ".".to_string(),
            "crates/hi-a (1/2)".to_string(),
            "crates/hi-a (2/2)".to_string(),
            "crates/hi-b (1/2)".to_string(),
            "crates/hi-b (2/2)".to_string(),
            "docs (1/2)".to_string(),
            "docs (2/2)".to_string(),
        ],
        "over-budget groups are sliced; slices of different groups never merge past the budget"
    );
    assert_eq!(
        scope.groups.iter().map(Vec::len).sum::<usize>(),
        files.len(),
        "every file is in exactly one turn"
    );
    assert!(scope.groups.iter().all(|group| group.len() <= 4));
}
