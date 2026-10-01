use crate::ToolStatus;
use serde_json::json;

fn plain(text: &str) -> String {
    let mut out = String::new();
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            for c in chars.by_ref() {
                if c == 'm' {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn numbered_lines(count: usize) -> String {
    (1..=count)
        .map(|n| format!("    let value_{n} = compute_{n}(input, {n});\n"))
        .collect()
}

/// Live ~/drme: `src/decoder/mod.rs` is 3,145 lines, so every edit's preview
/// was "preview omitted" and the UI drew `(+0 -0)` for a 45-line rewrite.
#[tokio::test]
async fn large_file_edit_preview_counts_every_changed_line() {
    let workspace = tempfile::tempdir().unwrap();
    let original = numbered_lines(3_000);
    assert!(original.len() > 128 * 1024, "{} bytes", original.len());
    std::fs::write(workspace.path().join("big.rs"), &original).unwrap();
    let old_string: String = (2_001..=2_003)
        .map(|n| format!("    let value_{n} = compute_{n}(input, {n});\n"))
        .collect();
    let arguments = json!({
        "path": "big.rs",
        "old_string": old_string,
        "new_string": "    let merged = compute_all(input);\n",
    })
    .to_string();

    let outcome = super::super::execute_in(workspace.path(), "edit", &arguments).await;

    assert_eq!(outcome.status, ToolStatus::Succeeded, "{outcome:?}");
    let display = plain(outcome.display.as_deref().expect("edit shows a preview"));
    assert!(
        display.contains("1 addition, 3 deletions"),
        "the summary counts the change however large the file is: {display}"
    );
    assert!(!display.contains("omitted"), "{display}");
    assert!(
        display.contains("2001 - ") && display.contains("2001 + "),
        "the gutter numbers the edited lines by their place in the file: {display}"
    );
}

#[tokio::test]
async fn truncated_previews_still_count_every_file() {
    let workspace = tempfile::tempdir().unwrap();
    std::fs::write(workspace.path().join("small.txt"), "alpha\nbeta\n").unwrap();
    let huge: String = numbered_lines(20_000)
        .lines()
        .map(|line| format!("+{line}\n"))
        .collect();
    let patch = format!(
        "*** Begin Patch\n*** Add File: generated.rs\n{huge}*** Update File: small.txt\n-beta\n+BETA\n*** End Patch"
    );
    let arguments = json!({ "patch": patch }).to_string();

    let outcome = super::super::execute_in(workspace.path(), "apply_patch", &arguments).await;

    assert_eq!(outcome.status, ToolStatus::Succeeded, "{outcome:?}");
    let display = plain(
        outcome
            .display
            .as_deref()
            .expect("apply_patch shows its diff"),
    );
    assert!(
        display.contains("20000 additions, 0 deletions"),
        "the cut-off file keeps its full count: {}",
        &display[..display.len().min(400)]
    );
    assert!(display.contains("(diff truncated)"), "{}", display.len());
    assert!(
        display.contains("+++ small.txt") && display.contains("1 addition, 1 deletion"),
        "a file after the size budget still reports its change: {}",
        &display[display.len().saturating_sub(400)..]
    );
    assert!(display.len() < 20_000, "{} chars", display.len());
    assert!(
        outcome.content.contains("updated small.txt"),
        "the model still gets the terse summary: {}",
        outcome.content
    );
}

#[tokio::test]
async fn edits_that_change_nothing_are_refused_as_already_in_the_file() {
    let workspace = tempfile::tempdir().unwrap();
    let target = workspace.path().join("notes.txt");
    let cases = [
        (
            "trailing whitespace",
            "alpha  \nbeta\n",
            "edit",
            json!({"path": "notes.txt", "old_string": "alpha\nbeta", "new_string": "alpha  \nbeta"}),
        ),
        (
            "line endings",
            "alpha\r\nbeta\r\n",
            "edit",
            json!({"path": "notes.txt", "old_string": "alpha\nbeta\n", "new_string": "alpha\r\nbeta\r\n"}),
        ),
        (
            "indentation",
            "fn f() {\n    body();\n}\n",
            "edit",
            json!({"path": "notes.txt", "old_string": "fn f() {\n  body();\n}", "new_string": "fn f() {\n    body();\n}"}),
        ),
        (
            "edits that undo each other",
            "alpha\nbeta\n",
            "multi_edit",
            json!({"path": "notes.txt", "edits": [
                {"old_string": "alpha", "new_string": "ALPHA"},
                {"old_string": "ALPHA", "new_string": "alpha"}
            ]}),
        ),
        (
            "identical write",
            "alpha\nbeta\n",
            "write",
            json!({"path": "notes.txt", "content": "alpha\nbeta\n"}),
        ),
        (
            "patch that re-adds a line",
            "alpha\nbeta\n",
            "apply_patch",
            json!({"patch": "*** Begin Patch\n*** Update File: notes.txt\n-beta\n+beta\n*** End Patch"}),
        ),
    ];
    for (label, original, name, arguments) in cases {
        std::fs::write(&target, original).unwrap();
        let modified = std::fs::metadata(&target).unwrap().modified().unwrap();

        let outcome =
            super::super::execute_in(workspace.path(), name, &arguments.to_string()).await;

        assert_eq!(outcome.status, ToolStatus::Failed, "{label}: {outcome:?}");
        assert!(
            outcome.content.contains("no change:")
                && outcome.content.contains("already in this file")
                && outcome.content.contains("Nothing was written"),
            "{label}: {}",
            outcome.content
        );
        assert!(!outcome.effects.mutation_applied, "{label}");
        assert!(outcome.effects.file_changes.is_empty(), "{label}");
        assert_eq!(
            std::fs::read_to_string(&target).unwrap(),
            original,
            "{label}"
        );
        assert_eq!(
            std::fs::metadata(&target).unwrap().modified().unwrap(),
            modified,
            "{label}: the file is not rewritten"
        );
    }
}

#[tokio::test]
async fn a_failed_edit_names_the_file_once() {
    let workspace = tempfile::tempdir().unwrap();
    std::fs::write(workspace.path().join("notes.txt"), "alpha\nbeta\n").unwrap();
    let arguments =
        json!({"path": "notes.txt", "old_string": "gamma", "new_string": "delta"}).to_string();

    let outcome = super::super::execute_in(workspace.path(), "edit", &arguments).await;

    assert_eq!(outcome.status, ToolStatus::Failed);
    assert_eq!(
        outcome.content.matches("editing notes.txt").count(),
        1,
        "{}",
        outcome.content
    );
    assert!(
        outcome.content.contains("old_string not found"),
        "{}",
        outcome.content
    );
}
