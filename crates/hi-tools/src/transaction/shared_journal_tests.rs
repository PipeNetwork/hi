use super::*;

#[test]
fn another_workspace_cannot_prune_a_preparing_writers_journal_parent() {
    let directory = tempfile::tempdir().unwrap();
    let state = directory.path().join("state");
    let first_root = directory.path().join("first");
    let second_root = directory.path().join("second");
    fs::create_dir_all(&first_root).unwrap();
    fs::create_dir_all(&second_root).unwrap();
    let first_journal = transaction_journal_dir(&first_root.canonicalize().unwrap(), &state);
    let journal_parent = first_journal.parent().unwrap().to_path_buf();
    let (ready, waiting) = std::sync::mpsc::channel();
    let (release, released) = std::sync::mpsc::channel();
    let writer_state = state.clone();
    let writer_root = first_root.clone();
    let writer = std::thread::spawn(move || {
        // Expose the recursive directory creation window deterministically:
        // its shared parent exists, but this writer's child does not yet.
        fs::create_dir_all(first_journal.parent().unwrap()).unwrap();
        ready.send(()).unwrap();
        released.recv().unwrap();
        fs::create_dir(&first_journal)
            .expect("a concurrent transaction must not remove this mkdir's parent");
        MutationPlan::new_with_state(
            &writer_root,
            &writer_state,
            vec![PlannedFileMutation::add("first.txt", b"first".to_vec())],
        )
        .unwrap()
        .commit()
        .unwrap();
    });
    waiting.recv().unwrap();
    let second_journal = transaction_journal_dir(&second_root.canonicalize().unwrap(), &state);
    MutationPlan::new_with_state(
        &second_root,
        &state,
        vec![PlannedFileMutation::add("second.txt", b"second".to_vec())],
    )
    .unwrap()
    .commit()
    .unwrap();
    let parent_retained = journal_parent.is_dir();
    release.send(()).unwrap();
    let result = writer.join();
    assert!(
        parent_retained,
        "only the second workspace's hash directory may be pruned"
    );
    result.unwrap();
    assert!(
        !second_journal.exists(),
        "per-workspace cleanup remains active"
    );
    assert_eq!(fs::read(first_root.join("first.txt")).unwrap(), b"first");
    assert_eq!(fs::read(second_root.join("second.txt")).unwrap(), b"second");
}
