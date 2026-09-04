use super::condense_diagnostics;

fn passing_failure_regression_log() -> String {
    let mut log = String::from("running 800 tests\n");
    for index in 0..800 {
        log.push_str(&format!(
            "test parser::rejects_expected_error_case_{index:04} ... ok\n"
        ));
    }
    log.push_str("\ntest result: ok. 800 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.10s\n");
    log
}

#[test]
fn passing_test_names_are_not_failure_evidence() {
    let fixture = passing_failure_regression_log();
    let condensed = condense_diagnostics(&fixture, 5_000);
    eprintln!(
        "passing libtest fixture: raw={} chars; model result={} chars",
        fixture.chars().count(),
        condensed.chars().count()
    );
    assert!(condensed.contains("800 passed; 0 failed"));
    assert!(condensed.contains("lines omitted"));
    assert!(
        condensed.chars().count() < 1_000,
        "{} chars",
        condensed.chars().count()
    );
}

#[test]
fn passing_names_do_not_hide_real_failures_or_their_details() {
    let mut fixture = passing_failure_regression_log();
    fixture.push_str("\nrunning 1 test\ntest parser::actual_failure ... FAILED\n\nfailures:\n\n---- parser::actual_failure stdout ----\nthread 'parser::actual_failure' panicked at src/parser.rs:42:7:\nassertion failed: parse(input).is_ok()\ninput detail line 1\ninput detail line 2\ninput detail line 3: crucial malformed field\ninput detail line 4\n\nfailures:\n    parser::actual_failure\n\ntest result: FAILED. 0 passed; 1 failed; 0 ignored\n");
    let condensed = condense_diagnostics(&fixture, 5_000);
    assert!(condensed.contains("actual_failure ... FAILED"));
    assert!(condensed.contains("src/parser.rs:42:7"));
    assert!(condensed.contains("input detail line 3: crucial malformed field"));
    assert!(condensed.contains("test result: FAILED. 0 passed; 1 failed"));
}
