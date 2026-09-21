use super::*;
use crate::review::{ReviewInput, ReviewInputs};
use std::fs;

fn populate(root: &Path, files: usize) {
    fs::create_dir_all(root.join("src")).unwrap();
    for index in 0..files {
        fs::write(root.join(format!("src/f{index}.rs")), "").unwrap();
    }
}

fn readme_inputs() -> ReviewInputs {
    ReviewInputs {
        files: vec![ReviewInput {
            path: "README.md".into(),
            kind: InputKind::Readme,
        }],
        ..ReviewInputs::default()
    }
}

fn spec_inputs() -> ReviewInputs {
    ReviewInputs {
        files: vec![ReviewInput {
            path: "docs/spec.md".into(),
            kind: InputKind::Spec,
        }],
        ..ReviewInputs::default()
    }
}

/// `git …` in `root`; panics on failure so a broken fixture is loud.
fn run_git(root: &Path, args: &[&str]) {
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

fn init_repo(root: &Path) {
    run_git(root, &["init", "-q"]);
    run_git(root, &["config", "commit.gpgsign", "false"]);
}

fn commit_all(root: &Path, message: &str) {
    run_git(root, &["add", "-A"]);
    run_git(root, &["commit", "-q", "-m", message]);
}

#[test]
fn counts_stop_at_the_limit_and_skip_ignored_files() {
    let dir = tempfile::tempdir().unwrap();
    populate(dir.path(), 5);
    assert!(!workspace_exceeds(dir.path(), 5));
    assert!(workspace_exceeds(dir.path(), 4));
    fs::write(dir.path().join(".gitignore"), "src/\n").unwrap();
    assert!(
        !workspace_exceeds(dir.path(), 1),
        "gitignored files do not count"
    );
}

#[test]
fn small_workspace_and_spec_and_scope_keep_the_whole_tree() {
    let dir = tempfile::tempdir().unwrap();
    populate(dir.path(), 3);
    let mut small = readme_inputs();
    assert_eq!(resolve_target(dir.path(), &mut small, false), Ok(()));
    assert_eq!(small, readme_inputs(), "small: README fallback untouched");

    populate(dir.path(), LARGE_WORKSPACE_FILES + 1);
    let mut with_spec = spec_inputs();
    assert_eq!(resolve_target(dir.path(), &mut with_spec, false), Ok(()));
    assert_eq!(with_spec, spec_inputs(), "a spec covers the whole tree");

    let mut scoped = readme_inputs();
    scoped.scope.push("src".into());
    let before = scoped.clone();
    assert_eq!(resolve_target(dir.path(), &mut scoped, false), Ok(()));
    assert_eq!(scoped, before, "an explicit directory is the scope");
}

#[test]
fn large_workspace_outside_git_is_refused_and_says_what_to_pass() {
    let dir = tempfile::tempdir().unwrap();
    populate(dir.path(), LARGE_WORKSPACE_FILES + 1);
    // A tempdir may sit inside a checkout: make sure git cannot see one.
    let inner = dir.path().join("work");
    fs::create_dir_all(&inner).unwrap();
    fs::rename(dir.path().join("src"), inner.join("src")).unwrap();
    let mut inputs = readme_inputs();
    let refusal = match git_scope(&inner) {
        Err(GitEmpty::NotARepo) => resolve_target(&inner, &mut inputs, false).unwrap_err(),
        other => {
            // Inside somebody's repository this tempdir is tracked work;
            // the refusal wording for that case is covered below.
            eprintln!("tempdir is inside a git repository ({other:?}); skipping");
            return;
        }
    };
    assert!(refusal.contains("more than 200 files"), "{refusal}");
    assert!(
        refusal.contains("this is not a git repository"),
        "{refusal}"
    );
    assert!(refusal.contains("`/review audit all`"), "{refusal}");
    assert!(refusal.contains("`/review audit <dir>`"), "{refusal}");
    assert!(refusal.contains("write plan.md"), "{refusal}");
    assert_eq!(inputs, readme_inputs(), "a refusal changes nothing");
}

#[test]
fn large_workspace_audits_uncommitted_work_else_the_last_commit() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    init_repo(root);
    populate(root, LARGE_WORKSPACE_FILES + 1);
    fs::write(root.join("README.md"), "# App\n").unwrap();

    // Nothing committed yet: everything is untracked work.
    let scope = git_scope(root).expect("untracked files are recent work");
    assert_eq!(scope.source, GitSource::Uncommitted);
    assert!(scope.files.contains(&"README.md".to_string()));
    assert!(scope.files.contains(&"src/f0.rs".to_string()));
    assert_eq!(scope.files.len(), LARGE_WORKSPACE_FILES + 2);
    assert_eq!(
        scope.summary(),
        format!(
            "{} uncommitted files (git status)",
            LARGE_WORKSPACE_FILES + 2
        )
    );

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
                format!(
                    "last commit {short_hash} ({} files)",
                    LARGE_WORKSPACE_FILES + 2
                )
            );
        }
        other => panic!("expected the last commit, got {other:?}"),
    }

    // A modified tracked file, a rename, a new file, and a deletion: the
    // uncommitted set wins, the old rename path and the deleted file are
    // not listed.
    fs::write(root.join("src/f1.rs"), "changed").unwrap();
    run_git(root, &["mv", "src/f2.rs", "src/moved.rs"]);
    fs::write(root.join("src/new.rs"), "").unwrap();
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

    let mut inputs = readme_inputs();
    assert_eq!(resolve_target(root, &mut inputs, false), Ok(()));
    assert!(
        inputs.files.is_empty(),
        "README is not a spec for recent work: defects only"
    );
    assert!(inputs.defects_only());
    assert_eq!(inputs.git, Some(scope));
    assert_eq!(
        inputs.notice.as_deref(),
        Some(
            "no plan.md or spec.md found; auditing recent work for defects: 3 uncommitted files (git status) · `/review audit all` audits the whole repo in chunks"
        )
    );
    assert_eq!(
        inputs.scope_summary().as_deref(),
        Some("3 uncommitted files (git status)")
    );

    // A commit that only deletes leaves nothing to audit: refuse.
    commit_all(root, "changes");
    fs::remove_file(root.join("src/new.rs")).unwrap();
    commit_all(root, "drop new.rs");
    assert_eq!(git_scope(root), Err(GitEmpty::Nothing));
    let refusal = resolve_target(root, &mut readme_inputs(), false).unwrap_err();
    assert!(
        refusal.contains("no commit whose files are still in the tree"),
        "{refusal}"
    );
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
fn chunks_are_top_level_dirs_with_containers_split_and_root_files_last() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    for file in [
        "crates/hi-a/src/lib.rs",
        "crates/hi-b/Cargo.toml",
        "docs/guide.md",
        "src/main.rs",
        "src/nested/deep.rs",
        "target/debug/out",
        "empty/.keep",
        "Cargo.toml",
    ] {
        let path = root.join(file);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, "").unwrap();
    }
    fs::create_dir_all(root.join("nothing/here")).unwrap();
    fs::write(root.join(".gitignore"), "target/\n").unwrap();
    assert_eq!(
        chunks(root),
        vec![
            "crates/hi-a".to_string(),
            "crates/hi-b".to_string(),
            "docs".to_string(),
            "src".to_string(),
            ".".to_string(),
        ],
        "crates/ holds no files of its own so its crates are the chunks; target/ is ignored; \
         empty/ has only a hidden file and nothing/ no files at all"
    );
    assert_eq!(chunk_label("."), "top-level files");
    assert_eq!(chunk_label("crates/hi-a"), "crates/hi-a");

    let mut inputs = readme_inputs();
    assert_eq!(resolve_target(root, &mut inputs, true), Ok(()));
    assert_eq!(inputs.chunks.len(), 5);
    assert!(inputs.files.is_empty(), "README dropped: defects only");
    assert_eq!(
        inputs.notice.as_deref(),
        Some(
            "no plan.md or spec.md found; auditing the whole workspace for defects in 5 chunks, one turn each: \
crates/hi-a, crates/hi-b, docs, src, top-level files"
        )
    );
    assert_eq!(inputs.scope_summary().as_deref(), Some("5 chunks"));

    let mut with_spec = spec_inputs();
    with_spec.scope = vec!["src".into(), "docs".into()];
    assert_eq!(resolve_target(root, &mut with_spec, true), Ok(()));
    assert_eq!(
        with_spec.chunks,
        vec!["src".to_string(), "docs".to_string()],
        "`all` with directories: the directories are the chunks, in order"
    );
    assert!(with_spec.scope.is_empty());
    assert_eq!(with_spec.files, spec_inputs().files, "a real spec stays");
    assert!(
        with_spec
            .notice
            .as_deref()
            .unwrap()
            .starts_with("auditing docs/spec.md against the whole workspace in 2 chunks"),
        "{:?}",
        with_spec.notice
    );

    let empty = tempfile::tempdir().unwrap();
    assert_eq!(
        resolve_target(empty.path(), &mut ReviewInputs::default(), true),
        Err("spec review: the workspace has no files to audit".to_string())
    );
}
