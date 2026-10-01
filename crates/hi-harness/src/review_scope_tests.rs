use super::*;
use crate::review::{ReviewInput, ReviewInputs};
use crate::review_git::GitSource;
use crate::review_git::tests::{commit_all, init_repo, run_git};
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

fn settings(chunk_files: usize) -> ReviewSettings {
    ReviewSettings {
        chunk_files,
        ..ReviewSettings::default()
    }
}

fn resolve(root: &Path, inputs: &mut ReviewInputs, all: bool) -> Result<(), String> {
    resolve_target(root, inputs, all, &ReviewSettings::default())
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
    assert_eq!(resolve(dir.path(), &mut small, false), Ok(()));
    assert_eq!(
        small,
        readme_inputs(),
        "small: one turn, README fallback untouched"
    );

    populate(dir.path(), LARGE_WORKSPACE_FILES + 1);
    let mut with_spec = spec_inputs();
    assert_eq!(resolve(dir.path(), &mut with_spec, false), Ok(()));
    assert_eq!(with_spec, spec_inputs(), "a spec covers the whole tree");

    let mut scoped = spec_inputs();
    scoped.scope.push("src".into());
    let before = scoped.clone();
    assert_eq!(resolve(dir.path(), &mut scoped, false), Ok(()));
    assert_eq!(
        scoped, before,
        "an explicit directory with a spec is the scope"
    );
}

#[test]
fn small_workspace_over_the_budget_is_cut_into_turns() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    populate(root, 40);
    fs::write(root.join("README.md"), "# App\n").unwrap();
    let mut inputs = readme_inputs();
    assert_eq!(
        resolve_target(root, &mut inputs, false, &settings(16)),
        Ok(())
    );
    assert!(inputs.files.is_empty(), "README dropped: defects only");
    assert_eq!(
        inputs.chunks,
        vec![
            "src (1/3)".to_string(),
            "src (2/3)".to_string(),
            "src (3/3), .".to_string()
        ],
        "40 files in src/ slice into three turns; the one root file packs into the last"
    );
    let files = chunk_files(root, "src (3/3), .", &[]);
    assert_eq!(files.len(), 13, "the last slice (12) plus README.md");
    assert!(files.contains(&"README.md".to_string()));
    let first = chunk_files(root, "src (1/3)", &[]);
    assert_eq!(first.len(), 14);
    assert_eq!(first[0], "src/f0.rs");
    assert!(
        inputs
            .notice
            .as_deref()
            .unwrap()
            .starts_with("no plan.md or spec.md found; auditing the workspace for defects in 3 chunks, one turn each: src (1/3); src (2/3); src (3/3), top-level files"),
        "{:?}",
        inputs.notice
    );

    // Named directories over the budget are cut the same way; within it
    // they stay one turn.
    let mut scoped = readme_inputs();
    scoped.scope.push("src".into());
    assert_eq!(
        resolve_target(root, &mut scoped, false, &settings(16)),
        Ok(())
    );
    assert_eq!(scoped.chunks.len(), 3);
    assert!(scoped.scope.is_empty(), "the chunks replace the scope");
    assert!(scoped.files.is_empty());
    let mut fits = readme_inputs();
    fits.scope.push("src".into());
    assert_eq!(
        resolve_target(root, &mut fits, false, &settings(64)),
        Ok(())
    );
    assert!(fits.chunks.is_empty());
    assert_eq!(fits.scope, vec!["src".to_string()]);
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
        Err(GitEmpty::NotARepo) => resolve(&inner, &mut inputs, false).unwrap_err(),
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
fn large_workspace_audits_recent_work_in_turns_when_it_is_big() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    init_repo(root);
    populate(root, LARGE_WORKSPACE_FILES + 1);
    fs::write(root.join("README.md"), "# App\n").unwrap();
    commit_all(root, "initial import");

    // Three changed files fit one turn.
    fs::write(root.join("src/f1.rs"), "changed\n").unwrap();
    run_git(root, &["mv", "src/f2.rs", "src/moved.rs"]);
    fs::write(root.join("src/new.rs"), "fn new() {}\n").unwrap();
    fs::remove_file(root.join("src/f3.rs")).unwrap();
    let scope = git_scope(root).unwrap();
    let mut inputs = readme_inputs();
    assert_eq!(resolve(root, &mut inputs, false), Ok(()));
    assert!(
        inputs.files.is_empty(),
        "README is not a spec for recent work: defects only"
    );
    assert!(inputs.defects_only());
    assert_eq!(inputs.git, Some(scope));
    assert!(inputs.chunks.is_empty());
    assert_eq!(
        inputs.notice.as_deref(),
        Some(
            "no plan.md or spec.md found; auditing recent work for defects: 3 uncommitted files (git status, +2 −0) · `/review audit all` audits the whole repo in chunks"
        )
    );
    assert_eq!(
        inputs.scope_summary().as_deref(),
        Some("3 uncommitted files (git status, +2 −0)")
    );

    // Forty changed files across two directories: turns of at most the
    // budget, grouped by top-level directory.
    fs::create_dir_all(root.join("docs")).unwrap();
    for index in 0..20 {
        fs::write(root.join(format!("src/f{index}.rs")), "changed\n").unwrap();
        fs::write(root.join(format!("docs/d{index}.md")), "# d\n").unwrap();
    }
    let mut many = readme_inputs();
    assert_eq!(
        resolve_target(root, &mut many, false, &settings(16)),
        Ok(())
    );
    let git = many.git.as_ref().unwrap();
    assert_eq!(git.source, GitSource::Uncommitted);
    assert_eq!(
        many.chunks,
        vec![
            "docs (1/2)".to_string(),
            "docs (2/2)".to_string(),
            "src (1/2)".to_string(),
            "src (2/2)".to_string()
        ],
        "{:?}",
        many.chunks
    );
    assert_eq!(git.groups.len(), 4);
    assert!(git.groups.iter().all(|group| group.len() <= 16));
    assert_eq!(
        git.groups.iter().map(Vec::len).sum::<usize>(),
        git.files.len()
    );
    assert_eq!(git.chunk_files(0), git.groups[0]);
    assert!(
        many.notice
            .as_deref()
            .unwrap()
            .contains(" in 4 turns of at most 16 files: docs (1/2); docs (2/2); src (1/2); src (2/2) · `/review audit all`"),
        "{:?}",
        many.notice
    );

    // A commit that only deletes leaves nothing to audit: refuse.
    commit_all(root, "changes");
    fs::remove_file(root.join("src/new.rs")).unwrap();
    commit_all(root, "drop new.rs");
    assert_eq!(git_scope(root), Err(GitEmpty::Nothing));
    let refusal = resolve(root, &mut readme_inputs(), false).unwrap_err();
    assert!(
        refusal.contains("no commit whose files are still in the tree"),
        "{refusal}"
    );
}

#[test]
fn recent_work_leaves_out_excluded_directories() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    init_repo(root);
    populate(root, LARGE_WORKSPACE_FILES + 1);
    commit_all(root, "initial import");
    fs::create_dir_all(root.join("bench/fixtures")).unwrap();
    fs::write(root.join("bench/fixtures/big.json"), "{}\n").unwrap();
    fs::write(root.join("src/f1.rs"), "changed\n").unwrap();
    let exclude = ReviewSettings {
        exclude: vec!["fixtures".into()],
        ..ReviewSettings::default()
    };

    let mut inputs = readme_inputs();
    assert_eq!(resolve_target(root, &mut inputs, false, &exclude), Ok(()));
    let git = inputs.git.as_ref().unwrap();
    assert_eq!(git.files, vec!["src/f1.rs".to_string()]);
    assert!(git.numstat_for("bench/fixtures/big.json").is_none());
    assert_eq!(
        inputs.scope_summary().as_deref(),
        Some("1 uncommitted file (git status, +1 −0)")
    );

    fs::write(root.join("src/f1.rs"), "").unwrap();
    let refusal = resolve_target(root, &mut readme_inputs(), false, &exclude).unwrap_err();
    assert!(refusal.contains("under `[review] exclude`"), "{refusal}");
    assert!(refusal.contains("`/review audit all`"), "{refusal}");
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
    let dirs = top_level_dirs(root, &[]);
    assert_eq!(
        dirs,
        vec![
            "crates".to_string(),
            "docs".to_string(),
            "empty".to_string(),
            "nothing".to_string(),
            "src".to_string()
        ],
        "target/ is ignored"
    );
    // A budget of one file: an over-budget directory with subdirectories
    // descends (its own files become `dir/.`), nothing packs.
    assert_eq!(
        chunk_plan(root, &dirs, true, 1, &[]),
        vec![
            "crates/hi-a".to_string(),
            "crates/hi-b".to_string(),
            "docs".to_string(),
            "src/.".to_string(),
            "src/nested".to_string(),
            ".".to_string(),
        ],
        "crates/ holds no files of its own so its crates are the chunks; src/ has a file of its own and one \
         in nested/; empty/ has only a hidden file and nothing/ no files at all"
    );
    assert_eq!(
        chunk_plan(root, &dirs, true, 16, &[]),
        vec!["crates, docs, src, .".to_string()],
        "everything fits one turn: one packed chunk, crates/ kept whole"
    );
    assert_eq!(
        chunk_plan(root, &dirs, true, 2, &["docs".into()]),
        vec!["crates".to_string(), "src".to_string(), ".".to_string()],
        "docs excluded; crates (2) and src (2) each fill a turn of two, the root file cannot join either"
    );
    assert_eq!(
        chunk_plan(root, &dirs, true, 3, &["docs".into()]),
        vec!["crates".to_string(), "src, .".to_string()],
        "with a turn of three the root file packs with src"
    );
    assert_eq!(chunk_label("."), "top-level files");
    assert_eq!(chunk_label("crates/hi-a"), "crates/hi-a");
    assert_eq!(chunk_label("src/., docs"), "src (own files), docs");
    assert_eq!(chunk_parts("a, b (1/2), ."), vec!["a", "b (1/2)", "."]);
    assert_eq!(
        chunk_files(root, "crates, docs, src, .", &[]),
        vec![
            "crates/hi-a/src/lib.rs".to_string(),
            "crates/hi-b/Cargo.toml".to_string(),
            "docs/guide.md".to_string(),
            "src/main.rs".to_string(),
            "src/nested/deep.rs".to_string(),
            "Cargo.toml".to_string()
        ]
    );
    assert_eq!(
        chunk_files(root, "src (2/2)", &[]),
        vec!["src/nested/deep.rs".to_string()],
        "a slice label derives the same slice"
    );
    assert_eq!(
        chunk_files(root, "src/.", &[]),
        vec!["src/main.rs".to_string()],
        "own files exclude the subdirectory"
    );
    assert!(chunk_files(root, "missing", &[]).is_empty());

    let mut inputs = readme_inputs();
    assert_eq!(
        resolve_target(root, &mut inputs, true, &settings(1)),
        Ok(())
    );
    assert_eq!(inputs.chunks.len(), 6);
    assert!(inputs.files.is_empty(), "README dropped: defects only");
    assert_eq!(
        inputs.notice.as_deref(),
        Some(
            "no plan.md or spec.md found; auditing the whole workspace for defects in 6 chunks, one turn each: \
crates/hi-a; crates/hi-b; docs; src (own files); src/nested; top-level files"
        )
    );
    assert_eq!(inputs.scope_summary().as_deref(), Some("6 chunks"));
    assert_eq!(inputs.exclude, Vec::<String>::new());

    let mut with_spec = spec_inputs();
    with_spec.scope = vec!["src".into(), "docs".into()];
    assert_eq!(
        resolve_target(root, &mut with_spec, true, &settings(1)),
        Ok(())
    );
    assert_eq!(
        with_spec.chunks,
        vec![
            "src/.".to_string(),
            "src/nested".to_string(),
            "docs".to_string()
        ],
        "`all` with directories: the directories are the chunks, in order"
    );
    assert!(with_spec.scope.is_empty());
    assert_eq!(with_spec.files, spec_inputs().files, "a real spec stays");
    assert!(
        with_spec
            .notice
            .as_deref()
            .unwrap()
            .starts_with("auditing docs/spec.md against src, docs in 3 chunks"),
        "{:?}",
        with_spec.notice
    );

    let mut dot = readme_inputs();
    dot.scope = vec![".".into()];
    assert_eq!(resolve_target(root, &mut dot, true, &settings(16)), Ok(()));
    assert_eq!(
        dot.chunks,
        vec!["crates, docs, src, .".to_string()],
        "`all .` is `all`"
    );
    assert!(
        dot.notice
            .as_deref()
            .unwrap()
            .contains("auditing the whole workspace for defects in 1 chunks, one turn each: crates, docs, src, top-level files"),
        "{:?}",
        dot.notice
    );

    let mut excluded = readme_inputs();
    let exclude = ReviewSettings {
        chunk_files: 16,
        exclude: vec!["docs".into(), "crates/hi-b".into()],
        ..ReviewSettings::default()
    };
    assert_eq!(resolve_target(root, &mut excluded, true, &exclude), Ok(()));
    assert_eq!(excluded.chunks, vec!["crates, src, .".to_string()]);
    assert_eq!(
        excluded.exclude, exclude.exclude,
        "kept for chunk_files after a reload"
    );
    assert_eq!(
        chunk_files(root, "crates, src, .", &excluded.exclude),
        vec![
            "crates/hi-a/src/lib.rs".to_string(),
            "src/main.rs".to_string(),
            "src/nested/deep.rs".to_string(),
            "Cargo.toml".to_string()
        ]
    );

    let empty = tempfile::tempdir().unwrap();
    assert_eq!(
        resolve(empty.path(), &mut ReviewInputs::default(), true),
        Err("spec review: the workspace has no files to audit".to_string())
    );
}

#[test]
fn chunks_are_ordered_by_recent_git_churn() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    init_repo(root);
    for file in ["a/one.rs", "b/two.rs", "c/three.rs"] {
        let path = root.join(file);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, "").unwrap();
    }
    commit_all(root, "init");
    fs::write(root.join("c/three.rs"), "changed\n").unwrap();
    commit_all(root, "touch c");
    fs::write(root.join("c/three.rs"), "changed again\n").unwrap();
    fs::write(root.join("b/two.rs"), "changed\n").unwrap();
    commit_all(root, "touch b and c");
    let dirs = top_level_dirs(root, &[]);
    assert_eq!(
        chunk_plan(root, &dirs, true, 1, &[]),
        vec!["c".to_string(), "b".to_string(), "a".to_string()],
        "c was touched by three commits, b by two, a by one"
    );
    fs::create_dir_all(root.join("d")).unwrap();
    fs::write(root.join("d/four.rs"), "").unwrap();
    assert_eq!(
        chunk_plan(root, &top_level_dirs(root, &[]), true, 1, &[]),
        vec![
            "c".to_string(),
            "b".to_string(),
            "a".to_string(),
            "d".to_string()
        ],
        "an uncommitted directory has no churn and comes last"
    );
}

#[test]
fn pack_units_slices_big_units_and_merges_small_ones_in_order() {
    let unit = |label: &str, count: usize| Unit {
        label: label.into(),
        files: (0..count).map(|i| format!("{label}/{i}")).collect(),
        churn: 0,
    };
    let packed = pack_units(
        vec![unit("a", 2), unit("b", 1), unit("c", 9), unit("d", 1)],
        4,
    );
    let labels: Vec<&str> = packed.iter().map(|u| u.label.as_str()).collect();
    assert_eq!(
        labels,
        vec!["a, b", "c (1/3)", "c (2/3)", "c (3/3), d"],
        "a+b is 3 <= 4; c is 9 files in 3 slices of 3, none of which fits after a+b; d rides with the last slice"
    );
    assert_eq!(packed[0].files, vec!["a/0", "a/1", "b/0"]);
    assert_eq!(packed[1].files, vec!["c/0", "c/1", "c/2"]);
    assert_eq!(packed[3].files, vec!["c/6", "c/7", "c/8", "d/0"]);
    let big_first = pack_units(vec![unit("a", 3), unit("b", 3), unit("c", 9)], 4);
    let labels: Vec<&str> = big_first.iter().map(|u| u.label.as_str()).collect();
    assert_eq!(
        labels,
        vec!["a", "b", "c (1/3)", "c (2/3)", "c (3/3)"],
        "3 + 3 > 4: nothing packs"
    );
    assert!(pack_units(Vec::new(), 4).is_empty());
    assert_eq!(
        pack_units(vec![unit("x", 3)], 0).len(),
        3,
        "a zero budget is one file per turn"
    );
}

#[test]
fn an_explicit_dot_is_the_whole_workspace_with_or_without_all() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    for file in ["a/one.rs", "b/two.rs", "top.rs"] {
        let path = root.join(file);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, "").unwrap();
    }
    let absolute = root.display().to_string();
    for raw in [".", "./", absolute.as_str()] {
        let discovered = crate::review::discover_inputs(root, &[raw.to_string()]);
        assert_eq!(discovered.scope, vec![".".to_string()], "{raw}");
        for all in [true, false] {
            let mut inputs = discovered.clone();
            assert_eq!(
                resolve_target(root, &mut inputs, all, &settings(1)),
                Ok(()),
                "{raw} all={all}"
            );
            assert_eq!(
                inputs.chunks,
                vec!["a".to_string(), "b".to_string(), ".".to_string()],
                "{raw} all={all}"
            );
            assert!(inputs.scope.is_empty());
            assert!(
                inputs
                    .notice
                    .as_deref()
                    .unwrap()
                    .contains("auditing the whole workspace for defects"),
                "{:?}",
                inputs.notice
            );
        }
    }
}

#[test]
fn excluded_matches_paths_prefixes_and_bare_names() {
    let exclude = vec![
        "bench/fixtures".to_string(),
        "node_modules".to_string(),
        " ".to_string(),
    ];
    assert!(excluded("bench/fixtures", &exclude));
    assert!(excluded("bench/fixtures/big.json", &exclude));
    assert!(!excluded("bench/fixtures2/x", &exclude));
    assert!(!excluded("bench", &exclude));
    assert!(excluded("web/node_modules/pkg/index.js", &exclude));
    assert!(excluded("node_modules", &exclude));
    assert!(!excluded("src/node_modules_shim.rs", &exclude));
    assert!(
        !excluded("src/main.rs", &exclude),
        "a blank entry excludes nothing"
    );
}
