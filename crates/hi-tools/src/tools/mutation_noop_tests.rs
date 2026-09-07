use super::prepare_mutation_in_with_state;
use crate::ToolStatus;
use serde_json::json;

#[tokio::test]
async fn identical_edits_fail_without_writes_transactions_or_applied_effects() {
    let workspace = tempfile::tempdir().unwrap();
    let state = tempfile::tempdir().unwrap();
    let target = workspace.path().join("notes.txt");
    let original = "alpha\nbeta\n";
    std::fs::write(&target, original).unwrap();
    let modified = std::fs::metadata(&target).unwrap().modified().unwrap();
    let cases = [
        (
            "edit",
            json!({"path": "notes.txt", "old_string": "beta", "new_string": "beta"}),
        ),
        (
            "edit",
            json!({"path": "notes.txt", "old_string": "beta", "new_string": "beta", "replace_all": true}),
        ),
        (
            "multi_edit",
            json!({"path": "notes.txt", "edits": [
                {"old_string": "beta", "new_string": "beta"}
            ]}),
        ),
        (
            "multi_edit",
            json!({"path": "notes.txt", "edits": [
                {"old_string": "alpha", "new_string": "ALPHA"},
                {"old_string": "beta", "new_string": "beta"}
            ]}),
        ),
    ];

    for (name, arguments) in cases {
        let arguments = arguments.to_string();
        let error =
            prepare_mutation_in_with_state(workspace.path(), state.path(), name, &arguments)
                .await
                .expect_err("an identical replacement must fail before transaction preparation");
        let message = format!("{error:#}");
        assert!(message.contains("old_string and new_string are identical"));
        assert!(message.contains("Provide a different replacement"));
        assert_eq!(std::fs::read_dir(state.path()).unwrap().count(), 0);

        let outcome = super::super::execute_in(workspace.path(), name, &arguments).await;
        assert_eq!(outcome.status, ToolStatus::Failed);
        assert!(outcome.content.contains("no edit was made"));
        assert!(!outcome.effects.mutation_applied);
        assert!(outcome.effects.file_changes.is_empty());
        assert_eq!(std::fs::read_to_string(&target).unwrap(), original);
        assert_eq!(
            std::fs::metadata(&target).unwrap().modified().unwrap(),
            modified,
            "even an earlier genuine hunk in a rejected batch must not be written"
        );
    }
}

#[tokio::test]
async fn genuine_single_replace_all_and_batched_edits_still_commit() {
    let workspace = tempfile::tempdir().unwrap();
    let target = workspace.path().join("notes.txt");
    let cases = [
        (
            "edit",
            json!({"path": "notes.txt", "old_string": "alpha", "new_string": "ALPHA"}),
            "ALPHA\nbeta\nbeta\n",
        ),
        (
            "edit",
            json!({"path": "notes.txt", "old_string": "beta", "new_string": "BETA", "replace_all": true}),
            "alpha\nBETA\nBETA\n",
        ),
        (
            "multi_edit",
            json!({"path": "notes.txt", "edits": [
                {"old_string": "alpha", "new_string": "ALPHA"},
                {"old_string": "beta\nbeta", "new_string": "BETA\nBETA"}
            ]}),
            "ALPHA\nBETA\nBETA\n",
        ),
    ];
    for (name, arguments, expected) in cases {
        std::fs::write(&target, "alpha\nbeta\nbeta\n").unwrap();
        let outcome =
            super::super::execute_in(workspace.path(), name, &arguments.to_string()).await;
        assert_eq!(outcome.status, ToolStatus::Succeeded, "{outcome:?}");
        assert!(outcome.effects.mutation_applied);
        assert_eq!(outcome.effects.file_changes.len(), 1);
        assert_eq!(std::fs::read_to_string(&target).unwrap(), expected);
    }
}
