use std::sync::atomic::{AtomicU64, Ordering};

use super::*;

fn root(label: &str) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let root = std::env::temp_dir().join(format!(
        "hi-ledger-{label}-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    root
}

#[test]
fn external_changes_advance_revision_and_merge() {
    let root = root("external");
    std::fs::write(root.join("a.txt"), "one").unwrap();
    let mut ledger = ChangeLedger::new(&root).unwrap();
    let baseline = ledger.revision();
    std::fs::write(root.join("a.txt"), "two").unwrap();
    ledger.reconcile().unwrap();
    std::fs::write(root.join("a.txt"), "three").unwrap();
    ledger.reconcile().unwrap();
    let changes = ledger.changes_since(baseline);
    assert_eq!(changes.len(), 1);
    assert_eq!(changes[0].before_len, Some(3));
    assert_eq!(changes[0].after_len, Some(5));
    assert_eq!(ledger.revision(), 2);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn startup_scan_skips_large_artifacts_and_named_virtualenvs() {
    let root = root("bounded-startup");
    let large = std::fs::File::create(root.join("model.safetensors")).unwrap();
    large
        .set_len(MAX_AUTOMATIC_FILE_BYTES.saturating_add(1))
        .unwrap();
    std::fs::create_dir_all(root.join(".venv-wan/lib/python")).unwrap();
    std::fs::write(
        root.join(".venv-wan/lib/python/generated.py"),
        "value = 1\n",
    )
    .unwrap();
    std::fs::write(root.join("main.py"), "value = 2\n").unwrap();

    let ledger = ChangeLedger::new(&root).unwrap();

    assert!(ledger.observed.contains_key("main.py"));
    assert!(!ledger.observed.contains_key("model.safetensors"));
    assert!(
        !ledger
            .observed
            .contains_key(".venv-wan/lib/python/generated.py")
    );
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn nested_project_model_caches_are_pruned_but_source_models_dirs_are_not() {
    // A workspace containing several checkouts (root/proj with its own
    // .git) must not walk proj/models or proj/.hi/models weight caches —
    // the root-relative prune alone left multi-hundred-GB model trees in
    // scan scope. Source directories merely named models stay tracked.
    let root = root("nested-model-caches");
    std::fs::create_dir_all(root.join("proj/.git")).unwrap();
    std::fs::create_dir_all(root.join("proj/models")).unwrap();
    std::fs::create_dir_all(root.join("proj/.hi/models")).unwrap();
    std::fs::create_dir_all(root.join("proj/src/models")).unwrap();
    std::fs::write(root.join("proj/models/weights.json"), "w\n").unwrap();
    std::fs::write(root.join("proj/.hi/models/cache.json"), "c\n").unwrap();
    std::fs::write(root.join("proj/src/models/user.rs"), "struct U;\n").unwrap();

    let ledger = ChangeLedger::new(&root).unwrap();

    assert!(!ledger.observed.contains_key("proj/models/weights.json"));
    assert!(!ledger.observed.contains_key("proj/.hi/models/cache.json"));
    assert!(ledger.observed.contains_key("proj/src/models/user.rs"));
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn generated_build_caches_are_pruned_even_when_nested() {
    let root = root("nested-build-cache");
    std::fs::create_dir_all(root.join(".cargo-home/registry/src/dep")).unwrap();
    std::fs::create_dir_all(root.join(".hi/state/cargo-home/registry/src/dep")).unwrap();
    std::fs::create_dir_all(root.join("bench/.build-arm64/release")).unwrap();
    std::fs::create_dir_all(root.join("bench/src/build-tools")).unwrap();
    std::fs::write(
        root.join("bench/.build-arm64/release/fingerprint"),
        "generated\n",
    )
    .unwrap();
    std::fs::create_dir_all(root.join("bench/terminal-bench/jobs/2026-01-01")).unwrap();
    std::fs::write(
        root.join("bench/terminal-bench/jobs/2026-01-01/output.py"),
        "generated = True\n",
    )
    .unwrap();
    std::fs::write(root.join("bench/src/build-tools/main.rs"), "fn main() {}\n").unwrap();
    std::fs::write(
        root.join(".cargo-home/registry/src/dep/lib.rs"),
        "pub fn dependency() {}\n",
    )
    .unwrap();
    std::fs::write(
        root.join(".hi/state/cargo-home/registry/src/dep/lib.rs"),
        "pub fn runtime_dependency() {}\n",
    )
    .unwrap();

    let ledger = ChangeLedger::new(&root).unwrap();

    assert!(
        !ledger
            .observed
            .contains_key("bench/.build-arm64/release/fingerprint")
    );
    assert!(
        !ledger
            .observed
            .contains_key("bench/terminal-bench/jobs/2026-01-01/output.py")
    );
    assert!(
        ledger
            .observed
            .contains_key("bench/src/build-tools/main.rs")
    );
    assert!(
        !ledger
            .observed
            .contains_key(".cargo-home/registry/src/dep/lib.rs")
    );
    assert!(
        !ledger
            .observed
            .contains_key(".hi/state/cargo-home/registry/src/dep/lib.rs")
    );
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn touched_paths_and_mutation_events_survive_net_zero_effects() {
    let root = root("net-zero");
    let mut ledger = ChangeLedger::new(&root).unwrap();
    let baseline = ledger.revision();
    std::fs::write(root.join("temporary.rs"), "x\n").unwrap();
    ledger.reconcile().unwrap();
    std::fs::remove_file(root.join("temporary.rs")).unwrap();
    ledger.reconcile().unwrap();

    assert!(ledger.changes_since(baseline).is_empty());
    assert_eq!(ledger.touched_paths_since(baseline), vec!["temporary.rs"]);
    assert!(ledger.had_mutation_since(baseline));

    let before_empty_effect = ledger.revision();
    ledger
        .record_tool_effects(&ToolEffects {
            mutation_attempted: true,
            mutation_applied: true,
            file_changes: Vec::new(),
        })
        .unwrap();
    assert!(ledger.had_mutation_since(before_empty_effect));
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn revision_evidence_survives_more_than_512_distinct_mutations() {
    let root = root("long-running-revision-evidence");
    let mut ledger = ChangeLedger::new(&root).unwrap();
    let baseline = ledger.revision();

    for index in 0..513 {
        ledger.push_event(vec![FileChange {
            path: format!("generated/{index}.rs"),
            kind: FileChangeKind::Create,
            before_digest: None,
            after_digest: Some(format!("sha256:{index}")),
            before_len: None,
            after_len: Some(index),
            before_mode: None,
            after_mode: Some(0o644),
        }]);
    }

    let changes = ledger.changes_since(baseline);
    assert_eq!(changes.len(), 513);
    assert_eq!(
        changes.first().map(|change| change.path.as_str()),
        Some("generated/0.rs")
    );
    assert_eq!(
        changes.last().map(|change| change.path.as_str()),
        Some("generated/99.rs"),
        "BTreeMap ordering is lexical, but every mutation must remain represented"
    );
    assert!(
        changes
            .iter()
            .any(|change| change.path == "generated/512.rs"),
        "settlement must retain evidence beyond the former 512-event window"
    );
    assert_eq!(ledger.touched_paths_since(baseline).len(), 513);
    assert!(ledger.had_mutation_since(baseline));
    assert_eq!(ledger.events.len(), MAX_REVISION_EVENTS);
    assert_eq!(ledger.dropped_event_count(), 1);

    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn compacted_events_preserve_active_turn_and_verification_baselines() {
    let root = root("compacted-active-baselines");
    let mut ledger = ChangeLedger::new(&root).unwrap();
    ledger.push_event(vec![FileChange {
        path: "before-turn.rs".into(),
        kind: FileChangeKind::Create,
        before_digest: None,
        after_digest: Some("old".into()),
        before_len: None,
        after_len: Some(1),
        before_mode: None,
        after_mode: Some(0o644),
    }]);
    let turn_baseline = ledger.begin_turn_retention_window();

    for index in 0..700 {
        ledger.push_event(vec![FileChange {
            path: "repeated.rs".into(),
            kind: FileChangeKind::Modify,
            before_digest: Some(format!("digest-{index}")),
            after_digest: Some(format!("digest-{}", index + 1)),
            before_len: Some(index),
            after_len: Some(index + 1),
            before_mode: Some(0o644),
            after_mode: Some(0o644),
        }]);
    }
    let verification_baseline = ledger.revision();
    ledger.retain_verification_baseline(verification_baseline);
    for index in 700..1_400 {
        ledger.push_event(vec![FileChange {
            path: "repeated.rs".into(),
            kind: FileChangeKind::Modify,
            before_digest: Some(format!("digest-{index}")),
            after_digest: Some(format!("digest-{}", index + 1)),
            before_len: Some(index),
            after_len: Some(index + 1),
            before_mode: Some(0o644),
            after_mode: Some(0o644),
        }]);
    }
    ledger.push_event(vec![FileChange {
        path: "temporary.rs".into(),
        kind: FileChangeKind::Create,
        before_digest: None,
        after_digest: Some("temporary".into()),
        before_len: None,
        after_len: Some(1),
        before_mode: None,
        after_mode: Some(0o644),
    }]);
    ledger.push_event(vec![FileChange {
        path: "temporary.rs".into(),
        kind: FileChangeKind::Delete,
        before_digest: Some("temporary".into()),
        after_digest: None,
        before_len: Some(1),
        after_len: None,
        before_mode: Some(0o644),
        after_mode: None,
    }]);

    assert_eq!(ledger.events.len(), MAX_REVISION_EVENTS);
    assert!(ledger.dropped_event_count() > 512);
    let turn_changes = ledger.changes_since(turn_baseline);
    assert_eq!(turn_changes.len(), 1);
    assert_eq!(turn_changes[0].path, "repeated.rs");
    assert_eq!(turn_changes[0].before_digest.as_deref(), Some("digest-0"));
    assert_eq!(turn_changes[0].after_digest.as_deref(), Some("digest-1400"));
    assert_eq!(
        ledger.touched_paths_since(turn_baseline),
        vec!["repeated.rs", "temporary.rs"]
    );
    assert!(ledger.had_mutation_since(turn_baseline));

    let verification_changes = ledger.changes_since(verification_baseline);
    assert_eq!(verification_changes.len(), 1);
    assert_eq!(
        verification_changes[0].before_digest.as_deref(),
        Some("digest-700")
    );
    assert_eq!(
        verification_changes[0].after_digest.as_deref(),
        Some("digest-1400")
    );
    assert_eq!(
        ledger.touched_paths_since(verification_baseline),
        vec!["repeated.rs", "temporary.rs"]
    );
    assert!(ledger.had_mutation_since(verification_baseline));
    assert!(
        ledger
            .touched_paths_since(0)
            .contains(&"before-turn.rs".to_string())
    );

    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn independent_roots_never_share_state() {
    let first = root("first");
    let second = root("second");
    let mut left = ChangeLedger::new(&first).unwrap();
    let mut right = ChangeLedger::new(&second).unwrap();
    std::fs::write(first.join("only-left"), "x").unwrap();
    left.reconcile().unwrap();
    right.reconcile().unwrap();
    assert_eq!(left.changed_paths_since(0), vec!["only-left"]);
    assert!(right.changed_paths_since(0).is_empty());
    assert_ne!(left.workspace_revision(), right.workspace_revision());
    let _ = std::fs::remove_dir_all(first);
    let _ = std::fs::remove_dir_all(second);
}

#[test]
fn ignored_config_and_explicit_pruned_paths_remain_authoritative() {
    let root = root("ignored-explicit");
    std::fs::write(root.join(".gitignore"), ".env\ntarget/\n").unwrap();
    let state_root = root.join(".hi/state");
    std::fs::create_dir_all(&state_root).unwrap();
    let mut ledger = ChangeLedger::new_with_state(&root, Some(&state_root)).unwrap();
    let baseline = ledger.revision();

    std::fs::write(root.join(".env"), "TOKEN=test\n").unwrap();
    std::fs::create_dir_all(root.join(".hi")).unwrap();
    std::fs::write(root.join(".hi/config.toml"), "[quality]\n").unwrap();
    ledger.reconcile().unwrap();

    let generated = root.join("target/generated.txt");
    std::fs::create_dir_all(generated.parent().unwrap()).unwrap();
    std::fs::write(&generated, "generated\n").unwrap();
    let after = read_state(&generated, None).unwrap().unwrap();
    ledger
        .record_tool_effects(&ToolEffects {
            mutation_attempted: true,
            mutation_applied: true,
            file_changes: vec![FileChange {
                path: "target/generated.txt".into(),
                kind: FileChangeKind::Create,
                before_digest: None,
                after_digest: Some(after.digest),
                before_len: None,
                after_len: Some(after.len),
                before_mode: None,
                after_mode: Some(after.mode),
            }],
        })
        .unwrap();

    // A full scan prunes target/, but the typed exact path must supplement
    // it rather than manufacturing a deletion that cancels the create.
    ledger.reconcile().unwrap();
    let paths = ledger.changed_paths_since(baseline);
    assert!(paths.contains(&".env".to_string()), "{paths:?}");
    assert!(paths.contains(&".hi/config.toml".to_string()), "{paths:?}");
    assert!(
        paths.contains(&"target/generated.txt".to_string()),
        "{paths:?}"
    );

    std::fs::write(state_root.join("journal"), "runtime-only").unwrap();
    assert!(ledger.reconcile().unwrap().is_empty());
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn weight_cache_trees_are_pruned_but_src_models_are_not() {
    let root = root("weight-cache");
    std::fs::create_dir_all(root.join("models/shard")).unwrap();
    std::fs::write(root.join("models/shard/a.bin"), "weights").unwrap();
    std::fs::create_dir_all(root.join(".hi/models/shard")).unwrap();
    std::fs::write(root.join(".hi/models/shard/b.bin"), "weights").unwrap();
    std::fs::create_dir_all(root.join("src/models")).unwrap();
    std::fs::write(root.join("src/models/user.rs"), "struct User;\n").unwrap();

    let ledger = ChangeLedger::new(&root).unwrap();
    assert!(ledger.observed.contains_key("src/models/user.rs"));
    assert!(!ledger.observed.contains_key("models/shard/a.bin"));
    assert!(!ledger.observed.contains_key(".hi/models/shard/b.bin"));
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn goal_export_is_metadata_until_a_task_explicitly_edits_it() {
    let root = root("goal-export-input");
    let mut ledger = ChangeLedger::new(&root).unwrap();
    let baseline = ledger.workspace_revision();
    let mut goal = crate::Goal::new("runtime view", vec!["step one".into(), "step two".into()]);
    goal.export_markdown_to(&root).unwrap();
    goal.advance();
    goal.export_markdown_to(&root).unwrap();
    assert!(ledger.reconcile().unwrap().is_empty());
    assert_eq!(ledger.workspace_revision(), baseline);
    for path in ["src.rs", ".hi/config.toml", ".hi/memory.md"] {
        let before = ledger.workspace_revision();
        std::fs::write(root.join(path), "requested change").unwrap();
        assert!(
            ledger
                .reconcile()
                .unwrap()
                .iter()
                .any(|change| change.path == path)
        );
        assert_ne!(
            ledger.workspace_revision(),
            before,
            "{path} is a task input"
        );
    }

    let path = crate::goal::GOAL_EXPORT_PATH;
    let before = read_state(&root.join(path), None).unwrap().unwrap();
    std::fs::write(root.join(path), "explicitly requested content").unwrap();
    let after = read_state(&root.join(path), None).unwrap().unwrap();
    let before_edit = ledger.revision();
    ledger
        .record_tool_effects(&ToolEffects {
            mutation_attempted: true,
            mutation_applied: true,
            file_changes: vec![FileChange {
                path: path.into(),
                kind: FileChangeKind::Modify,
                before_digest: Some(before.digest),
                after_digest: Some(after.digest),
                before_len: Some(before.len),
                after_len: Some(after.len),
                before_mode: Some(before.mode),
                after_mode: Some(after.mode),
            }],
        })
        .unwrap();
    ledger.reconcile().unwrap();
    assert!(
        ledger
            .changed_paths_since(before_edit)
            .contains(&path.to_owned())
    );
    let requested_revision = ledger.workspace_revision();
    goal.export_markdown_to(&root).unwrap();
    assert!(
        ledger
            .reconcile()
            .unwrap()
            .iter()
            .any(|change| change.path == path)
    );
    assert_ne!(
        ledger.workspace_revision(),
        requested_revision,
        "an explicitly edited goal export is now validation input"
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn explicitly_checked_goal_export_is_an_input_without_a_fake_mutation() {
    let root = root("read-goal-export-input");
    let mut goal = crate::Goal::new("runtime view", vec!["one".into(), "two".into()]);
    goal.export_markdown_to(&root).unwrap();
    let mut ledger = ChangeLedger::new(&root).unwrap();
    let before = ledger.workspace_revision();
    let revision = ledger.revision();
    let prepared = ledger
        .prepare_goal_validation_input_cancellable(&CancellationToken::new())
        .unwrap();
    ledger.register_goal_validation_input();
    let (changes, retired) = ledger.commit_prepared_reconcile(prepared);
    retired.discard();
    assert!(changes.is_empty());
    assert_eq!(ledger.revision(), revision);
    assert!(ledger.changed_paths_since(revision).is_empty());
    assert_ne!(
        ledger.workspace_revision(),
        before,
        "the check now fingerprints its named input"
    );
    goal.advance();
    goal.export_markdown_to(&root).unwrap();
    assert!(
        ledger
            .reconcile()
            .unwrap()
            .iter()
            .any(|change| change.path == crate::goal::GOAL_EXPORT_PATH)
    );
    std::fs::remove_dir_all(root).unwrap();
}
