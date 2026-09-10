use super::*;

#[test]
fn evidence_cycle_diagnostics_are_bounded_without_blocking_new_inspections() {
    let mut evidence = EvidenceTracker::default();
    for index in 0..5_000 {
        evidence.record_signature(format!("signature-{index}"));
    }

    assert_eq!(
        evidence.seen_signatures.len(),
        EvidenceTracker::SIGNATURE_LIMIT
    );
    assert_eq!(
        evidence.seen_signature_set.len(),
        EvidenceTracker::SIGNATURE_LIMIT
    );
    assert_eq!(evidence.seen_signatures_dropped, 904);
    assert!(!evidence.has_seen_signature("signature-0"));
    assert!(evidence.has_seen_signature("signature-4999"));

    for index in 0..2_100 {
        evidence.record_success(
            "read",
            &serde_json::json!({"path": format!("src/{index}.rs")}).to_string(),
            "1\tcontents\n",
        );
    }
    assert_eq!(evidence.inspected_paths.len(), EvidenceTracker::PATH_LIMIT);
    assert_eq!(
        evidence.completed_read_paths.len(),
        EvidenceTracker::PATH_LIMIT
    );
    assert_eq!(
        evidence.inspected_paths.front().map(String::as_str),
        Some("src/52.rs")
    );
    assert_eq!(
        evidence.inspected_paths.back().map(String::as_str),
        Some("src/2099.rs")
    );
}

#[test]
fn successful_validation_is_recorded_without_a_mutation() {
    let mut tracker = ImplementationTracker::default();
    tracker.record_tool_result(
        "bash",
        r#"{"command":"cargo test --quiet"}"#,
        "",
        true,
        false,
    );
    assert!(tracker.validation_seen);
    assert!(!tracker.validation_after_last_mutation);
}

#[test]
fn validation_that_updates_a_lockfile_still_counts_as_validation() {
    let mut tracker = ImplementationTracker::default();
    tracker.record_tool_result(
        "bash",
        r#"{"command":"cargo test --quiet"}"#,
        "",
        true,
        true,
    );
    assert!(tracker.mutation_seen);
    assert!(tracker.validation_seen);
    assert!(tracker.validation_after_last_mutation);
}

#[test]
fn elided_completed_read_is_reopened_for_paging() {
    let mut evidence = EvidenceTracker::default();
    evidence.record_success(
        "read",
        r#"{"path":"crates/hi-tui/src/lib.rs"}"#,
        "   1\tfn main() {}\n",
    );
    assert!(
        evidence.rereads_only_completed_files(&[(
            "c".into(),
            "read".into(),
            r#"{"path":"crates/hi-tui/src/lib.rs","offset":560}"#.into(),
        )]),
        "a full read should block extra pages"
    );
    let messages = vec![
        Message::assistant(vec![Content::ToolCall {
            id: "r1".into(),
            name: "read".into(),
            arguments: r#"{"path":"crates/hi-tui/src/lib.rs"}"#.into(),
        }]),
        Message::tool_result("r1", "[elided read output — was 1289 lines]"),
    ];
    evidence.reopen_elided_reads(&messages);
    assert!(
        !evidence.rereads_only_completed_files(&[(
            "c".into(),
            "read".into(),
            r#"{"path":"crates/hi-tui/src/lib.rs","offset":560}"#.into(),
        )]),
        "elided contents are no longer 'returned in full'"
    );
    assert!(
        evidence.round_adds_evidence(&[(
            "c".into(),
            "read".into(),
            r#"{"path":"crates/hi-tui/src/lib.rs","offset":560}"#.into(),
        )]),
        "an extra page of an elided file is new evidence"
    );
}

#[test]
fn truncated_file_paging_crosses_the_legacy_eight_page_boundary() {
    let mut evidence = EvidenceTracker::default();
    for page in 0..9 {
        let offset = page * 100 + 1;
        evidence.record_success(
            "read",
            &serde_json::json!({"path": "src/large.rs", "offset": offset}).to_string(),
            &format!(
                "{offset}\tcontent\n— read more with offset {}",
                offset + 100
            ),
        );
    }

    assert!(evidence.round_adds_evidence(&[(
        "next".into(),
        "read".into(),
        serde_json::json!({"path": "src/large.rs", "offset": 901}).to_string(),
    )]));
}
