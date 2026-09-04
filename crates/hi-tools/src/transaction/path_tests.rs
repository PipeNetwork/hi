use super::*;

#[cfg(unix)]
#[test]
fn mutation_parent_components_follow_directory_symlinks() {
    let root = tempfile::tempdir().unwrap();
    let state = tempfile::tempdir().unwrap();
    fs::create_dir_all(root.path().join("deep/sub")).unwrap();
    fs::write(root.path().join("config"), b"root original").unwrap();
    fs::write(root.path().join("deep/config"), b"nested original").unwrap();
    std::os::unix::fs::symlink("deep/sub", root.path().join("alias")).unwrap();

    let requested = root.path().join("alias/../config");
    assert_eq!(fs::read(&requested).unwrap(), b"nested original");
    let plan = MutationPlan::new_with_state(
        root.path(),
        state.path(),
        vec![PlannedFileMutation::update(&requested, b"updated".to_vec())],
    )
    .unwrap();
    assert_eq!(plan.single_target_path().as_deref(), Some("deep/config"));
    plan.commit().unwrap();
    assert_eq!(fs::read(&requested).unwrap(), b"updated");
    assert_eq!(
        fs::read(root.path().join("config")).unwrap(),
        b"root original"
    );
}

#[cfg(unix)]
#[test]
fn new_mutation_paths_resolve_symlink_parent_before_missing_tail() {
    let root = tempfile::tempdir().unwrap();
    let state = tempfile::tempdir().unwrap();
    fs::create_dir_all(root.path().join("deep/sub")).unwrap();
    std::os::unix::fs::symlink("deep/sub", root.path().join("alias")).unwrap();
    let plan = MutationPlan::new_with_state(
        root.path(),
        state.path(),
        vec![PlannedFileMutation::add(
            "alias/../new/../created/file",
            b"new".to_vec(),
        )],
    )
    .unwrap();
    assert_eq!(
        plan.single_target_path().as_deref(),
        Some("deep/created/file")
    );
    plan.commit().unwrap();
    assert_eq!(
        fs::read(root.path().join("deep/created/file")).unwrap(),
        b"new"
    );
    assert!(!root.path().join("created").exists());
}

#[cfg(unix)]
#[test]
fn symlink_parent_cannot_bypass_workspace_containment() {
    let root = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    fs::create_dir(outside.path().join("sub")).unwrap();
    std::os::unix::fs::symlink(outside.path().join("sub"), root.path().join("alias")).unwrap();
    let error = resolve_workspace_target(root.path(), Path::new("alias/../file")).unwrap_err();
    assert!(error.to_string().contains("outside workspace"), "{error:#}");
}

#[test]
fn file_parent_is_not_collapsed_into_a_different_target() {
    let root = tempfile::tempdir().unwrap();
    fs::write(root.path().join("file"), b"original").unwrap();
    let error = resolve_workspace_target(root.path(), Path::new("file/../other")).unwrap_err();
    assert!(error.to_string().contains("not a directory"), "{error:#}");
    assert!(!root.path().join("other").exists());
}

#[cfg(unix)]
#[test]
fn unresolved_symlink_keeps_final_identity_but_cannot_be_traversed() {
    let root = tempfile::tempdir().unwrap();
    std::os::unix::fs::symlink("missing", root.path().join("dangling")).unwrap();
    std::os::unix::fs::symlink("cycle", root.path().join("cycle")).unwrap();
    for path in ["dangling", "cycle"] {
        assert_eq!(
            resolve_workspace_target(root.path(), Path::new(path)).unwrap(),
            root.path().canonicalize().unwrap().join(path)
        );
    }
    for path in ["dangling/child", "cycle/child"] {
        assert!(resolve_workspace_target(root.path(), Path::new(path)).is_err());
    }
}
