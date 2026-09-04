use super::*;
use crate::tests::test_app;

#[test]
fn path_grants_reject_traversal_and_unknown_multi_file_targets() {
    let mut app = test_app("custom", "test-model");
    app.permission_mode = hi_agent::PermissionMode::Ask;
    app.add_auto_approve_path("src/lib.rs");
    assert!(app.path_auto_approved("src/other.rs"));

    for path in [
        "src/../.env",
        "src/../../outside.txt",
        "src\\..\\.env",
        "(unknown)",
        "(multiple files)",
        "",
        ".",
    ] {
        let request = ConfirmationRequest::FileEdit {
            path: path.into(),
            diff: "+ changed\n".into(),
        };
        app.add_auto_approve_path(path);
        assert!(!app.should_auto_approve(&request), "path={path:?}");
        assert!(
            !perm_actions(&request).contains(&PermAction::AlwaysPath),
            "path={path:?}"
        );
        assert!(matches!(
            handle_key(
                &mut app,
                &KeyEvent::new(KeyCode::Char('p'), KeyModifiers::NONE),
                &request,
            ),
            ConfirmDecision::Redraw
        ));
        assert!(matches!(
            handle_key(
                &mut app,
                &KeyEvent::new(KeyCode::Char('y'), KeyModifiers::NONE),
                &request,
            ),
            ConfirmDecision::Approve
        ));
    }
    assert_eq!(app.auto_approve_paths, vec!["src"]);
}

#[test]
fn legacy_malformed_prefix_cannot_authorize_parent_path() {
    let mut app = test_app("custom", "test-model");
    app.auto_approve_paths = vec!["src/..".into(), "(unknown)".into()];
    assert!(!app.path_auto_approved("src/../.env"));
    assert!(!app.path_auto_approved("(unknown)"));
}
