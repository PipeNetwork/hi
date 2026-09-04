use super::*;

#[test]
fn checkpoint_restores_a_link_that_was_retargeted_to_an_existing_file() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("workspace");
    let state = temp.path().join("state");
    fs::create_dir_all(&root).unwrap();
    fs::write(root.join("original"), "original contents").unwrap();
    fs::write(root.join("current"), "current contents").unwrap();
    std::os::unix::fs::symlink("original", root.join("link")).unwrap();
    let checkpoint = crate::internal_snapshot::create(&root, &state).unwrap();
    fs::remove_file(root.join("link")).unwrap();
    std::os::unix::fs::symlink("current", root.join("link")).unwrap();

    crate::internal_snapshot::restore(&root, &state, &checkpoint).unwrap();

    assert_eq!(
        fs::read_link(root.join("link")).unwrap(),
        PathBuf::from("original")
    );
    assert!(
        !fs::symlink_metadata(root.join("current"))
            .unwrap()
            .is_symlink()
    );
    assert_eq!(
        fs::read_to_string(root.join("current")).unwrap(),
        "current contents"
    );
    assert_eq!(
        fs::read_to_string(root.join("original")).unwrap(),
        "original contents"
    );
}

#[test]
fn restore_addresses_symlink_entry_without_touching_its_referent() {
    for target_kind in ["internal", "external", "dangling"] {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("workspace");
        let state = temp.path().join("state");
        fs::create_dir_all(&root).unwrap();
        let referent = if target_kind == "external" {
            temp.path().join("external-file")
        } else {
            root.join("referent")
        };
        if target_kind != "dangling" {
            fs::write(&referent, "keep referent contents").unwrap();
        }
        std::os::unix::fs::symlink(&referent, root.join("link")).unwrap();
        let plan = MutationPlan::new_restore_with_state(
            &root,
            &state,
            vec![RestoreMutation {
                path: "link".into(),
                postimage: Some(RestoreNode::File {
                    bytes: b"checkpoint contents".to_vec(),
                    mode: 0o644,
                }),
            }],
        )
        .unwrap();
        assert_eq!(
            plan.file_changes()[0].path,
            "link",
            "restore selected {target_kind} referent instead of symlink entry"
        );
        plan.commit().unwrap();
        assert!(
            !fs::symlink_metadata(root.join("link"))
                .unwrap()
                .is_symlink()
        );
        assert_eq!(
            fs::read_to_string(root.join("link")).unwrap(),
            "checkpoint contents"
        );
        if target_kind == "dangling" {
            assert!(!referent.exists());
        } else {
            assert_eq!(
                fs::read_to_string(&referent).unwrap(),
                "keep referent contents"
            );
        }
    }
}

#[test]
fn restore_reinstates_symlink_target_and_keeps_both_referents_unchanged() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("workspace");
    fs::create_dir_all(&root).unwrap();
    fs::write(root.join("original"), "original contents").unwrap();
    fs::write(root.join("current"), "current contents").unwrap();
    std::os::unix::fs::symlink("current", root.join("link")).unwrap();
    MutationPlan::new_restore_with_state(
        &root,
        temp.path().join("state"),
        vec![RestoreMutation {
            path: "link".into(),
            postimage: Some(RestoreNode::Symlink {
                target: "original".into(),
            }),
        }],
    )
    .unwrap()
    .commit()
    .unwrap();
    assert_eq!(
        fs::read_link(root.join("link")).unwrap(),
        PathBuf::from("original")
    );
    assert!(
        !fs::symlink_metadata(root.join("current"))
            .unwrap()
            .is_symlink()
    );
    assert_eq!(
        fs::read_to_string(root.join("current")).unwrap(),
        "current contents"
    );
    assert_eq!(
        fs::read_to_string(root.join("original")).unwrap(),
        "original contents"
    );
}

#[test]
fn restore_rejects_escaping_parent_symlinks_and_workspace_root() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("workspace");
    let outside = temp.path().join("outside");
    fs::create_dir_all(&root).unwrap();
    fs::create_dir_all(&outside).unwrap();
    fs::write(outside.join("file"), "untouched").unwrap();
    std::os::unix::fs::symlink(&outside, root.join("alias")).unwrap();
    std::os::unix::fs::symlink("missing-parent", root.join("dangling")).unwrap();
    for path in ["alias/file", "dangling/file", ".", "..", "../outside/file"] {
        assert!(
            MutationPlan::new_restore_with_state(
                &root,
                temp.path().join("state"),
                vec![RestoreMutation {
                    path: path.into(),
                    postimage: None,
                }]
            )
            .is_err(),
            "unsafe restore target was accepted: {path}"
        );
    }
    assert_eq!(
        fs::read_to_string(outside.join("file")).unwrap(),
        "untouched"
    );
}
