use super::{format_read_with_budget, render_read_with_budget};

#[test]
fn tiny_budgets_remain_bounded_with_explicit_truncation() {
    for budget in 0..128 {
        let rendered =
            render_read_with_budget("x".repeat(1_000).as_str(), None, None, Some(budget));
        assert!(rendered.content.chars().count() <= budget);
        assert!(rendered.truncated);
    }
}

#[test]
fn first_long_line_respects_budget_and_reports_missing_content() {
    let fixture = "é".repeat(100_000);
    let rendered = format_read_with_budget(&fixture, None, None, Some(512));
    eprintln!(
        "single-line fixture: raw={} chars; model result={} chars",
        fixture.chars().count(),
        rendered.chars().count()
    );
    assert!(rendered.chars().count() <= 512);
    assert!(rendered.contains("[line truncated]"));
    assert!(rendered.contains("   1\t"));
    assert!(rendered.contains("bounded"));
}

#[test]
fn long_first_line_keeps_accurate_next_line_offset() {
    let fixture = format!("{}\nsecond\nthird\n", "x".repeat(5_000));
    let rendered = format_read_with_budget(&fixture, None, None, Some(256));
    assert!(rendered.chars().count() <= 256);
    assert!(rendered.contains("[line truncated]"));
    assert!(rendered.contains("lines 1-1 of 3"));
    assert!(rendered.contains("read more with offset 2"));
    assert!(!rendered.contains("second"));
}

#[tokio::test]
async fn single_long_line_has_authoritative_truncation_metadata() {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("minified.json"), "x".repeat(300_000)).unwrap();
    let cache = std::sync::Mutex::new(crate::ReadCache::new());
    let outcome = crate::read::run_read(root.path(), &cache, r#"{"path":"minified.json"}"#)
        .await
        .unwrap();
    assert!(outcome.content.chars().count() <= crate::read::read_output_budget());
    assert!(matches!(
        outcome.truncation,
        crate::TruncationState::Truncated { .. }
    ));
}
