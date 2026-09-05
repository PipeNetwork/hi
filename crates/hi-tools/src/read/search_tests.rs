use super::*;

fn has_ripgrep() -> bool {
    std::process::Command::new("rg")
        .arg("--version")
        .output()
        .is_ok_and(|output| output.status.success())
}

#[tokio::test]
async fn grep_searches_hidden_project_configuration() {
    if !has_ripgrep() {
        return;
    }
    let root = tempfile::tempdir().unwrap();
    for dir in [
        ".github/workflows",
        ".git",
        ".cargo-home/registry",
        ".hi/state/cargo-home/registry",
    ] {
        std::fs::create_dir_all(root.path().join(dir)).unwrap();
    }
    std::fs::write(
        root.path().join(".github/workflows/test.yml"),
        "needle: useful workflow\n",
    )
    .unwrap();
    std::fs::write(
        root.path().join(".git/config"),
        "needle: private vcs metadata\n",
    )
    .unwrap();
    std::fs::write(
        root.path().join(".cargo-home/registry/vendor.rs"),
        "needle: downloaded cache\n",
    )
    .unwrap();
    std::fs::write(
        root.path().join(".hi/state/cargo-home/registry/vendor.rs"),
        "needle: default downloaded cache\n",
    )
    .unwrap();
    let runner =
        ProcessRunner::new_with_policy(root.path(), crate::sandbox::SandboxPolicy::Off).unwrap();
    for arguments in [
        r#"{"pattern":"needle"}"#,
        r#"{"pattern":"needle","glob":"**"}"#,
    ] {
        let output = run_grep_with_runner(root.path(), Some(&runner), arguments)
            .await
            .unwrap();
        assert!(
            output.content.contains("useful workflow"),
            "{}",
            output.content
        );
        assert!(
            !output.content.contains("private vcs metadata")
                && !output.content.contains("downloaded cache"),
            "{}",
            output.content
        );
    }
    let fallback = run_grep_fallback_sync(
        root.path(),
        root.path().to_str().unwrap(),
        "needle",
        None,
        0,
    )
    .unwrap();
    assert!(fallback.content.contains("useful workflow"));
    assert!(!fallback.content.contains("downloaded cache"));
}

#[tokio::test]
async fn grep_does_not_condense_source_matches_as_a_test_log() {
    if !has_ripgrep() {
        return;
    }
    let root = tempfile::tempdir().unwrap();
    let mut source = String::from("// needle: test result: ok. 20 passed; 0 failed\n");
    for index in 1..=20 {
        source.push_str(&format!("// needle: relevant_evidence_{index:02}\n"));
    }
    std::fs::write(root.path().join("source.rs"), source).unwrap();
    let runner =
        ProcessRunner::new_with_policy(root.path(), crate::sandbox::SandboxPolicy::Off).unwrap();
    let output = run_grep_with_runner(root.path(), Some(&runner), r#"{"pattern":"needle"}"#)
        .await
        .unwrap();
    assert!(
        output.content.contains("relevant_evidence_10"),
        "{}",
        output.content
    );
    assert!(
        !output.content.contains("lines omitted"),
        "{}",
        output.content
    );
}

#[test]
fn fallback_grep_merges_overlapping_context_without_repeating_lines() {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(
        root.path().join("source.rs"),
        "before\nneedle one\nbridge\nneedle two\nafter\n",
    )
    .unwrap();
    let output = run_grep_fallback_sync(
        root.path(),
        root.path().to_str().unwrap(),
        "needle",
        None,
        2,
    )
    .unwrap();
    assert_eq!(
        output.content.matches("bridge").count(),
        1,
        "{}",
        output.content
    );
    assert!(output.content.contains("source.rs:2: needle one"));
    assert!(output.content.contains("source.rs:4: needle two"));
}

#[test]
fn dense_fallback_grep_context_fixture() {
    let root = tempfile::tempdir().unwrap();
    let mut source = String::new();
    for index in 0..40 {
        source.push_str(&format!(
            "needle item_{index:02}\ncontext detail_{index:02}\n"
        ));
    }
    std::fs::write(root.path().join("source.rs"), source).unwrap();
    let output = run_grep_fallback_sync(
        root.path(),
        root.path().to_str().unwrap(),
        "needle",
        None,
        3,
    )
    .unwrap();
    eprintln!(
        "dense fallback context fixture: output={} chars; unique item markers={}",
        output.content.chars().count(),
        (0..40)
            .filter(|index| output.content.contains(&format!("item_{index:02}")))
            .count()
    );
    assert_eq!(
        output.content.matches("item_20").count(),
        1,
        "{}",
        output.content
    );
    for index in 0..40 {
        assert_eq!(
            output.content.matches(&format!("item_{index:02}")).count(),
            1
        );
    }
    assert_eq!(output.truncation, crate::TruncationState::Complete);
}

#[test]
fn fallback_grep_separates_disjoint_context_groups() {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(
        root.path().join("source.rs"),
        "before\nneedle one\nafter\nomit A\nomit B\nbefore\nneedle two\nafter\n",
    )
    .unwrap();
    let output = run_grep_fallback_sync(
        root.path(),
        root.path().to_str().unwrap(),
        "needle",
        None,
        1,
    )
    .unwrap();
    assert_eq!(
        output.content.matches("--\n").count(),
        1,
        "{}",
        output.content
    );
    assert!(!output.content.contains("omit"), "{}", output.content);
    assert!(
        output.content.contains("source.rs:2: needle one")
            && output.content.contains("source.rs:7: needle two")
    );
}

#[test]
fn fallback_grep_reports_budget_truncation_in_metadata() {
    let root = tempfile::tempdir().unwrap();
    let source = (0..100)
        .map(|index| format!("needle {index}: {}\n", "detail".repeat(50)))
        .collect::<String>();
    std::fs::write(root.path().join("source.rs"), source).unwrap();
    let output = run_grep_fallback_sync(
        root.path(),
        root.path().to_str().unwrap(),
        "needle",
        None,
        0,
    )
    .unwrap();
    assert!(matches!(
        output.truncation,
        crate::TruncationState::Truncated { .. }
    ));
    assert!(output.content.contains("truncated"));
}

#[tokio::test]
async fn grep_does_not_silently_stop_at_200_matches_inside_a_small_result() {
    if !has_ripgrep() {
        return;
    }
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("source.rs"), "needle\n".repeat(250)).unwrap();
    let runner =
        ProcessRunner::new_with_policy(root.path(), crate::sandbox::SandboxPolicy::Off).unwrap();
    let output = run_grep_with_runner(
        root.path(),
        Some(&runner),
        r#"{"pattern":"needle","path":"source.rs"}"#,
    )
    .await
    .unwrap();
    assert!(output.content.contains("250:needle"), "{}", output.content);
    assert_eq!(output.content.matches("needle").count(), 250);
    assert_eq!(output.truncation, crate::TruncationState::Complete);
}
