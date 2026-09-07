use super::*;

#[test]
fn stored_diagnostics_render_without_internal_separators() {
    let digest = digest_failure(
        Path::new("."),
        "error[E0382]: borrow of partially moved value: `cmd`\n --> src/server.rs:483:36\n",
    )
    .unwrap();
    let identity = digest.signature.first().unwrap();
    assert!(identity.contains('\0'));
    assert_eq!(
        display_signature_item(identity),
        "error[E0382] borrow of partially moved value `cmd` (src/server.rs)"
    );
    assert_eq!(
        display_signature_item("test:moderation::kick\0detail:1234"),
        "Failing test: moderation::kick"
    );
}

const CARGO_OUTPUT: &str = r#"   Compiling foo v0.1.0
error[E0308]: mismatched types
  --> crates/foo/src/lib.rs:4:9
   |
 4 |         "text"
   |         ^^^^^^ expected `u32`, found `&str`

error[E0308]: mismatched types
  --> crates/foo/src/lib.rs:4:9
   |
duplicate of the same error

error[E0425]: cannot find value `missing` in this scope
  --> crates/foo/src/other.rs:9:5

error: aborting due to 2 previous errors
error: could not compile `foo` (lib) due to 2 previous errors
"#;

const TEST_OUTPUT: &str = r#"running 3 tests
test tests::works ... ok
test tests::breaks ... FAILED
test tests::also_breaks ... FAILED

failures:

---- tests::breaks stdout ----
thread 'tests::breaks' panicked at crates/foo/src/lib.rs:20:5:
assertion `left == right` failed
  left: 1
 right: 2

---- tests::also_breaks stdout ----
thread 'tests::also_breaks' panicked at crates/foo/src/lib.rs:30:5:
boom

failures:
    tests::breaks
    tests::also_breaks

test result: FAILED. 1 passed; 2 failed
"#;

#[test]
fn diagnostics_dedupe_cascades_and_skip_meta_lines() {
    let diagnostics = parse_diagnostics(CARGO_OUTPUT);
    assert_eq!(diagnostics.len(), 2, "{diagnostics:?}");
    assert!(diagnostics[0].headline.starts_with("error[E0308]"));
    assert_eq!(
        diagnostics[0].location.as_deref(),
        Some("crates/foo/src/lib.rs:4:9")
    );
    assert!(diagnostics[1].headline.starts_with("error[E0425]"));
}

#[test]
fn failing_tests_capture_names_and_excerpts() {
    let tests = parse_failing_tests(TEST_OUTPUT);
    assert_eq!(tests.len(), 2);
    assert_eq!(tests[0].0, "tests::breaks");
    assert!(tests[0].1.iter().any(|line| line.contains("assertion")));
    assert_eq!(tests[1].0, "tests::also_breaks");
}

#[test]
fn changed_assertion_detail_changes_the_test_failure_signature() {
    let changed = TEST_OUTPUT.replace("  left: 1\n right: 2", "  left: 3\n right: 4");
    let first = digest_failure(Path::new("/nonexistent"), TEST_OUTPUT).unwrap();
    let repeated = digest_failure(Path::new("/nonexistent"), TEST_OUTPUT).unwrap();
    let changed = digest_failure(Path::new("/nonexistent"), &changed).unwrap();

    assert_eq!(first.signature, repeated.signature);
    assert_ne!(first.signature, changed.signature);
}

#[test]
fn quiet_format_failures_digest_too() {
    // hi's own verify stages run `cargo test --quiet`, whose failure
    // lines are `name --- FAILED` with progress dots — no `test ` prefix.
    // This exact shape went undigested in live runs (6/6 unstructured).
    let quiet = r#"running 300 tests
...................... 22/300
background::tests::kill_started_after_reaps_auto_backgrounded --- FAILED
........ 31/300
edit::tests::apply_multi_patch_adds_updates_and_deletes --- FAILED

failures:

---- background::tests::kill_started_after_reaps_auto_backgrounded stdout ----
thread 'background::tests::kill_started_after_reaps_auto_backgrounded' panicked at crates/hi-tools/src/background.rs:873:9:
got: "[sh_1: exited with code 71]"

failures:
    background::tests::kill_started_after_reaps_auto_backgrounded
    edit::tests::apply_multi_patch_adds_updates_and_deletes

test result: FAILED. 298 passed; 2 failed; 0 ignored
"#;
    let tests = parse_failing_tests(quiet);
    assert_eq!(tests.len(), 2, "{tests:?}");
    assert_eq!(
        tests[0].0,
        "background::tests::kill_started_after_reaps_auto_backgrounded"
    );
    assert!(tests[0].1.iter().any(|line| line.contains("panicked")));
    assert_eq!(
        tests[1].0,
        "edit::tests::apply_multi_patch_adds_updates_and_deletes"
    );
}

#[test]
fn digest_includes_source_region_when_file_exists() {
    let dir = std::env::temp_dir().join(format!("hi-digest-{}", std::process::id()));
    std::fs::create_dir_all(dir.join("crates/foo/src")).unwrap();
    std::fs::write(
        dir.join("crates/foo/src/lib.rs"),
        "fn f() -> u32 {\n    let x = 1;\n    let y = 2;\n        \"text\"\n}\n",
    )
    .unwrap();
    let digest = digest_failure(&dir, CARGO_OUTPUT).unwrap();
    assert!(digest.text.contains("2 distinct compiler error(s)"));
    assert!(
        digest.text.contains("source (crates/foo/src/lib.rs:"),
        "{}",
        digest.text
    );
    assert!(digest.text.contains(">    4 |"), "{}", digest.text);
    assert_eq!(digest.failure_count, 2);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn digest_is_none_for_unstructured_output() {
    assert!(digest_failure(Path::new("/nonexistent"), "some shell noise\n").is_none());
}

#[test]
fn pytest_failures_come_from_summary_with_failures_section_excerpts() {
    let raw = "\
____________ test_json_string ____________
    def test_json_string():\n>       assert out == '[]'\nE       AssertionError: assert None == '[]'
tests/test_output.py:37: AssertionError
=========================== short test summary info ============================
FAILED tests/test_output.py::test_json_string - AssertionError: assert None
ERROR tests/providers/test_memset.py
";
    let tests = parse_pytest_failures(raw);
    assert_eq!(tests.len(), 2);
    assert_eq!(tests[0].0, "tests/test_output.py::test_json_string");
    assert!(
        tests[0].1.iter().any(|l| l.contains("AssertionError")),
        "{:?}",
        tests[0].1
    );
    assert_eq!(tests[1].0, "tests/providers/test_memset.py");
    let digest = digest_failure(Path::new("/nonexistent"), raw).unwrap();
    assert!(digest.text.contains("2 failing test(s)"));
}

#[test]
fn nextest_output_digests_via_panic_excerpt_without_phantom_errors() {
    // cargo-nextest embeds libtest lines but routes panics through
    // per-test `stderr ───` sections and ends with `error: test run failed`.
    let raw = "\
    test tests::fails ... FAILED
  stderr ───
    thread 'tests::fails' (42) panicked at src/lib.rs:11:18:
    assertion `left == right` failed: two should be three
      left: 2
     right: 3
    note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
     Summary [   0.008s] 2 tests run: 1 passed, 1 failed, 0 skipped
        FAIL [   0.007s] (2/2) rust-fail tests::fails
error: test run failed
";
    let digest = digest_failure(Path::new("/nonexistent"), raw).unwrap();
    assert!(
        digest.text.contains("1 failing test(s): tests::fails"),
        "{}",
        digest.text
    );
    assert!(
        digest.text.contains("panicked at src/lib.rs:11:18"),
        "{}",
        digest.text
    );
    assert!(digest.text.contains("left: 2"), "{}", digest.text);
    assert!(
        !digest.text.contains("compiler error"),
        "the nextest wrapper line must not become a phantom diagnostic: {}",
        digest.text
    );
}

#[test]
fn go_failures_capture_name_and_indented_detail() {
    let raw = "--- FAIL: TestParse (0.01s)\n    parse_test.go:12: got 1, want 2\nFAIL\n";
    let tests = parse_go_failures(raw);
    assert_eq!(tests.len(), 1);
    assert_eq!(tests[0].0, "TestParse");
    assert!(tests[0].1[0].contains("parse_test.go:12"));
}

#[test]
fn python_traceback_fallback_extracts_dotted_exception_and_frame() {
    let raw = "\
Traceback (most recent call last):
  File \"/app/cli.py\", line 102, in main
    results = client.execute()
  File \"/app/client.py\", line 64, in execute
    raise sa_exc.ArgumentError(msg)
sqlalchemy.exc.ArgumentError: could not assemble any primary key columns
";
    let diagnostics = parse_python_tracebacks(raw);
    assert_eq!(diagnostics.len(), 1);
    assert!(
        diagnostics[0]
            .headline
            .starts_with("sqlalchemy.exc.ArgumentError:")
    );
    assert_eq!(
        diagnostics[0].location.as_deref(),
        Some("/app/client.py:64")
    );
    // The fallback stays out of the way when pytest already named failures.
    let with_pytest = format!("FAILED tests/a.py::t - boom\n{raw}");
    let digest = digest_failure(Path::new("/nonexistent"), &with_pytest).unwrap();
    assert!(digest.text.contains("1 failing test(s)"));
    assert!(!digest.text.contains("compiler error"), "{}", digest.text);
}

/// Corpus harness for tuning against real-world failure logs (e.g. agent
/// trajectories from Hugging Face). Reporting-only, never fails:
/// `HI_DIGEST_CORPUS=<dir-of-.log-files> cargo test -p hi-agent --lib \
///  digest_corpus -- --ignored --nocapture`
#[test]
#[ignore = "set HI_DIGEST_CORPUS to a directory of failure logs"]
fn digest_corpus_coverage() {
    let Some(dir) = std::env::var_os("HI_DIGEST_CORPUS") else {
        return;
    };
    let mut total = 0usize;
    let mut digested = 0usize;
    let mut misses: Vec<String> = Vec::new();
    for entry in std::fs::read_dir(dir).expect("corpus dir").flatten() {
        let path = entry.path();
        if path.extension().is_none_or(|e| e != "log") {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        total += 1;
        match digest_failure(Path::new("/nonexistent-root"), &text) {
            Some(_) => digested += 1,
            None => misses.push(path.display().to_string()),
        }
    }
    println!("digest corpus coverage: {digested}/{total}");
    for miss in misses.iter().take(10) {
        println!("  miss: {miss}");
    }
    // HI_DIGEST_SHOW=<file> renders one digest for eyeballing quality.
    if let Some(show) = std::env::var_os("HI_DIGEST_SHOW")
        && let Ok(text) = std::fs::read_to_string(&show)
        && let Some(digest) = digest_failure(Path::new("/nonexistent-root"), &text)
    {
        println!(
            "--- digest for {} ---\n{}",
            show.to_string_lossy(),
            digest.text
        );
    }
}

/// Corpus harness over real consecutive test-run pairs (agent
/// trajectories): validates that the failure signature is stable enough
/// for thrashing detection. Invariant: a byte-identical rerun MUST read
/// as "no progress" — a violation means volatile content (tmp paths,
/// thread ids, timings) leaked into the signature. Reporting-only:
/// `HI_CONVERGENCE_CORPUS=<pairs.jsonl> cargo test -p hi-agent --lib \
///  convergence_corpus -- --ignored --nocapture`
#[test]
#[ignore = "set HI_CONVERGENCE_CORPUS to a jsonl of {first, second, identical} pairs"]
fn convergence_corpus_signature_stability() {
    let Some(path) = std::env::var_os("HI_CONVERGENCE_CORPUS") else {
        return;
    };
    let text = std::fs::read_to_string(path).expect("corpus file");
    let root = Path::new("/nonexistent-root");
    let (mut pairs, mut both_parsed, mut violations) = (0usize, 0usize, Vec::new());
    let mut verdicts: std::collections::BTreeMap<&str, usize> = Default::default();
    for line in text.lines() {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        let (Some(first), Some(second)) = (
            value.get("first").and_then(|v| v.as_str()),
            value.get("second").and_then(|v| v.as_str()),
        ) else {
            continue;
        };
        let identical = value
            .get("identical")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        pairs += 1;
        let (Some(a), Some(b)) = (digest_failure(root, first), digest_failure(root, second)) else {
            continue;
        };
        both_parsed += 1;
        let previous = (a.failure_count, a.signature.clone());
        let note = convergence_note(Some(&previous), &b);
        let verdict = if note.contains("No progress") {
            "no-progress"
        } else if note.contains("Progress:") {
            "progress"
        } else if note.contains("Regression:") {
            "regression"
        } else {
            "changed"
        };
        *verdicts.entry(verdict).or_default() += 1;
        if identical && verdict != "no-progress" {
            violations.push(format!(
                "identical rerun read as {verdict}: sig_a={:?} sig_b={:?}",
                a.signature.iter().take(3).collect::<Vec<_>>(),
                b.signature.iter().take(3).collect::<Vec<_>>(),
            ));
        }
    }
    println!("convergence corpus: {pairs} pairs · {both_parsed} with both sides digested");
    println!("verdicts: {verdicts:?}");
    println!("identical-rerun invariant violations: {}", violations.len());
    for v in violations.iter().take(5) {
        println!("  VIOLATION {v}");
    }
}

#[test]
fn convergence_notes_cover_progress_stall_and_regression() {
    let digest = |keys: &[&str]| FailureDigest {
        text: String::new(),
        signature: keys.iter().map(|k| k.to_string()).collect(),
        failure_count: keys.len(),
    };
    let current = digest(&["diag:a", "diag:b"]);
    assert_eq!(convergence_note(None, &current), "");
    let same = (2, current.signature.clone());
    assert!(convergence_note(Some(&same), &current).contains("No progress"));
    let bigger = (
        4,
        digest(&["diag:a", "diag:b", "diag:c", "diag:d"]).signature,
    );
    assert!(convergence_note(Some(&bigger), &current).contains("Progress: 4 → 2"));
    let smaller = (1, digest(&["diag:z"]).signature);
    assert!(convergence_note(Some(&smaller), &current).contains("Regression: 1 → 2"));
}
