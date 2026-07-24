//! Failure digests for verification stages.
//!
//! A failed `cargo check`/`cargo test` hands the model a wall of raw output in
//! which one root cause often fans out into dozens of cascade errors, and the
//! lines that matter (the failing span) aren't in context. The digest
//! restructures that evidence: distinct root-cause diagnostics first with the
//! spanned source region inlined, failing test names with their panic
//! excerpts, and a stable signature the repair loop compares across rounds to
//! tell converging from thrashing.

use std::collections::BTreeSet;
use std::path::Path;

/// Distinct diagnostics to list in full (with source spans for the first few).
const MAX_LISTED_ERRORS: usize = 8;
/// Diagnostics that get their source region inlined.
const MAX_SPANNED_ERRORS: usize = 3;
/// Context lines either side of a diagnostic's line in the inlined region.
const SPAN_CONTEXT_LINES: usize = 4;
/// Failing tests that get their panic/stdout excerpt inlined.
const MAX_TEST_EXCERPTS: usize = 3;
/// Lines kept from a failing test's `---- name stdout ----` section.
const MAX_TEST_EXCERPT_LINES: usize = 12;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Diagnostic {
    /// The `error[E0308]: mismatched types` headline.
    pub headline: String,
    /// `path:line:col` from the following `--> ` line, when present.
    pub location: Option<String>,
}

#[derive(Debug, Default)]
pub(crate) struct FailureDigest {
    /// Rendered digest text, ready to prepend to the raw stage output.
    pub text: String,
    /// Stable identity of this failure set, for cross-round comparison.
    pub signature: BTreeSet<String>,
    /// Distinct compiler diagnostics + failing tests.
    pub failure_count: usize,
}

/// Headlines that summarize other errors rather than being one.
fn is_meta_error_line(line: &str) -> bool {
    const META: &[&str] = &[
        "error: aborting due to",
        "error: could not compile",
        "error: test failed",
        "error: test run failed",
        "error: doctest failed",
        "error: build failed",
        "error: process didn't exit successfully",
        "error: 1 target failed",
    ];
    META.iter().any(|prefix| line.starts_with(prefix))
}

/// Parse rustc-style diagnostics: an `error…` headline followed (within a few
/// lines) by an ` --> path:line:col` location.
fn parse_diagnostics(raw: &str) -> Vec<Diagnostic> {
    let lines: Vec<&str> = raw.lines().collect();
    let mut out: Vec<Diagnostic> = Vec::new();
    let mut seen = BTreeSet::new();
    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim_start();
        let is_error = trimmed.starts_with("error[") || trimmed.starts_with("error:");
        if !is_error || is_meta_error_line(trimmed) {
            continue;
        }
        let location = lines[i + 1..]
            .iter()
            .take(3)
            .find_map(|next| next.trim_start().strip_prefix("--> "))
            .map(|loc| loc.trim().to_string());
        let headline = trimmed.to_string();
        let key = format!("{headline}\0{}", location.as_deref().unwrap_or(""));
        if seen.insert(key) {
            out.push(Diagnostic { headline, location });
        }
    }
    out
}

/// Failing pytest cases from the short-summary section (`FAILED
/// tests/x.py::name - Reason` / `ERROR tests/x.py`), with each failure's
/// `____ name ____` FAILURES-section body as the excerpt.
fn parse_pytest_failures(raw: &str) -> Vec<(String, Vec<String>)> {
    let mut names: Vec<(String, Option<String>)> = Vec::new();
    let mut seen = BTreeSet::new();
    for line in raw.lines() {
        let trimmed = line.trim();
        let Some(rest) = trimmed
            .strip_prefix("FAILED ")
            .or_else(|| trimmed.strip_prefix("ERROR "))
        else {
            continue;
        };
        let (name, reason) = match rest.split_once(" - ") {
            Some((name, reason)) => (name.trim(), Some(reason.trim().to_string())),
            None => (rest.trim(), None),
        };
        if !name.contains(".py") || name.contains(' ') {
            continue;
        }
        if seen.insert(name.to_string()) {
            names.push((name.to_string(), reason));
        }
    }
    let mut out = Vec::new();
    for (name, reason) in names {
        // The FAILURES-section header uses the short name (`____ test_x ____`);
        // match on the last `::` segment, else the file path (collection errors).
        let short = name.rsplit("::").next().unwrap_or(&name);
        let excerpt = raw
            .lines()
            .skip_while(|line| {
                !(line.starts_with('_') && line.contains(short) && line.ends_with('_'))
            })
            .skip(1)
            .take_while(|line| !line.starts_with("____") && !line.starts_with("===="))
            .take(MAX_TEST_EXCERPT_LINES)
            .map(str::to_string)
            .collect::<Vec<_>>();
        let excerpt = if excerpt.is_empty() {
            reason.map(|reason| vec![reason]).unwrap_or_default()
        } else {
            excerpt
        };
        out.push((name, excerpt));
    }
    out
}

/// Failing `go test` cases: `--- FAIL: TestName` with the indented
/// `file.go:NN: message` lines as the excerpt.
fn parse_go_failures(raw: &str) -> Vec<(String, Vec<String>)> {
    let lines: Vec<&str> = raw.lines().collect();
    let mut out = Vec::new();
    let mut seen = BTreeSet::new();
    for (i, line) in lines.iter().enumerate() {
        let Some(rest) = line.trim_start().strip_prefix("--- FAIL: ") else {
            continue;
        };
        let name = rest.split_whitespace().next().unwrap_or(rest).to_string();
        if name.is_empty() || !seen.insert(name.clone()) {
            continue;
        }
        let excerpt = lines[i + 1..]
            .iter()
            .take_while(|next| next.starts_with(' ') || next.starts_with('\t'))
            .take(MAX_TEST_EXCERPT_LINES)
            .map(|next| next.trim_end().to_string())
            .collect();
        out.push((name, excerpt));
    }
    out
}

/// Uncaught Python exceptions: each `Traceback (most recent call last):`
/// block's final `SomeError: message` line becomes the headline, with the
/// last stack frame as the location. Fallback only — pytest output already
/// names its failures.
fn parse_python_tracebacks(raw: &str) -> Vec<Diagnostic> {
    let mut out: Vec<Diagnostic> = Vec::new();
    let mut seen = BTreeSet::new();
    for block in raw.split("Traceback (most recent call last):").skip(1) {
        let mut location = None;
        let mut headline = None;
        for line in block.lines().take(60) {
            let trimmed = line.trim();
            if let Some(frame) = trimmed.strip_prefix("File \"") {
                if let Some((path, rest)) = frame.split_once("\", line ") {
                    let line_no: String =
                        rest.chars().take_while(char::is_ascii_digit).collect();
                    if !line_no.is_empty() {
                        location = Some(format!("{path}:{line_no}"));
                    }
                }
                continue;
            }
            // The first non-frame, non-source line shaped like `Error: msg`
            // (including dotted classes: `sqlalchemy.exc.ArgumentError: msg`)
            // ends the block.
            if !trimmed.is_empty()
                && !line.starts_with("    ")
                && trimmed.split_once(':').is_some_and(|(kind, _)| {
                    !kind.contains(char::is_whitespace)
                        && kind
                            .rsplit('.')
                            .next()
                            .and_then(|last| last.chars().next())
                            .is_some_and(char::is_uppercase)
                })
            {
                headline = Some(trimmed.to_string());
                break;
            }
        }
        if let Some(headline) = headline {
            let key = format!("{headline}\0{}", location.as_deref().unwrap_or(""));
            if seen.insert(key) {
                out.push(Diagnostic { headline, location });
            }
        }
    }
    out
}

/// Failing libtest cases: `test name ... FAILED` lines, plus each failing
/// test's `---- name stdout ----` section for the excerpt.
fn parse_failing_tests(raw: &str) -> Vec<(String, Vec<String>)> {
    let mut names: Vec<String> = Vec::new();
    let mut seen = BTreeSet::new();
    for line in raw.lines() {
        if let Some(rest) = line.trim().strip_prefix("test ")
            && let Some(name) = rest.strip_suffix("... FAILED")
        {
            let name = name.trim().to_string();
            if !name.is_empty() && seen.insert(name.clone()) {
                names.push(name);
            }
        }
    }
    let mut out = Vec::new();
    for name in names {
        let marker = format!("---- {name} stdout ----");
        let excerpt = raw.split(&marker).nth(1).map(|section| {
            section
                .lines()
                .skip_while(|line| line.trim().is_empty())
                .take_while(|line| !line.trim_start().starts_with("---- "))
                .take(MAX_TEST_EXCERPT_LINES)
                .map(str::to_string)
                .collect::<Vec<_>>()
        });
        out.push((name, excerpt.unwrap_or_default()));
    }
    out
}

/// The ±context source region around `path:line:col`, resolved under `root`.
/// Returns `None` when the path doesn't resolve to a readable workspace file.
fn source_region(root: &Path, location: &str) -> Option<String> {
    let mut parts = location.split(':');
    let path = parts.next()?;
    let line: usize = parts.next()?.parse().ok()?;
    let relative = Path::new(path);
    if relative.is_absolute()
        || relative
            .components()
            .any(|c| !matches!(c, std::path::Component::Normal(_)))
    {
        return None;
    }
    let text = std::fs::read_to_string(root.join(relative)).ok()?;
    let lines: Vec<&str> = text.lines().collect();
    if line == 0 || line > lines.len() {
        return None;
    }
    let start = line.saturating_sub(SPAN_CONTEXT_LINES + 1);
    let end = (line + SPAN_CONTEXT_LINES).min(lines.len());
    let mut region = format!("   source ({path}:{}-{end}):\n", start + 1);
    for (offset, content) in lines[start..end].iter().enumerate() {
        let number = start + offset + 1;
        let marker = if number == line { ">" } else { " " };
        region.push_str(&format!("   {marker}{number:>5} | {content}\n"));
    }
    Some(region)
}

/// Digest a failed stage's raw output. Returns `None` when nothing structured
/// was recognized — the caller then keeps the raw output alone.
/// Name-anchored panic excerpt: the `thread '<name>' panicked at …` block for
/// a failing test whose runner didn't provide a `---- name stdout ----`
/// section (cargo-nextest routes panics through per-test `stderr ───`
/// sections instead).
fn panic_excerpt(raw: &str, name: &str) -> Vec<String> {
    let marker = format!("'{name}'");
    raw.lines()
        .skip_while(|line| !(line.contains("panicked at") && line.contains(&marker)))
        .take_while(|line| {
            let trimmed = line.trim();
            !trimmed.is_empty() && !trimmed.starts_with("note: run with")
        })
        .take(MAX_TEST_EXCERPT_LINES)
        .map(|line| line.trim_end().to_string())
        .collect()
}

pub(crate) fn digest_failure(root: &Path, raw: &str) -> Option<FailureDigest> {
    let mut diagnostics = parse_diagnostics(raw);
    let mut failing_tests = parse_failing_tests(raw);
    for (name, excerpt) in &mut failing_tests {
        if excerpt.is_empty() {
            *excerpt = panic_excerpt(raw, name);
        }
    }
    failing_tests.extend(parse_pytest_failures(raw));
    failing_tests.extend(parse_go_failures(raw));
    if diagnostics.is_empty() && failing_tests.is_empty() {
        // Nothing test-framework-shaped: an uncaught exception (a crashed
        // script or harness) is still a structurable root cause.
        diagnostics = parse_python_tracebacks(raw);
    }
    if diagnostics.is_empty() && failing_tests.is_empty() {
        return None;
    }

    let mut signature = BTreeSet::new();
    let mut text = String::from("── failure digest ──\n");

    if !diagnostics.is_empty() {
        text.push_str(&format!(
            "{} distinct compiler error(s) — cascade duplicates removed; fix these root causes:\n",
            diagnostics.len()
        ));
        for (i, diagnostic) in diagnostics.iter().take(MAX_LISTED_ERRORS).enumerate() {
            signature.insert(format!(
                "diag:{}\0{}",
                diagnostic.headline,
                diagnostic.location.as_deref().unwrap_or("")
            ));
            match &diagnostic.location {
                Some(location) => {
                    text.push_str(&format!("{}. {} — {location}\n", i + 1, diagnostic.headline));
                    if i < MAX_SPANNED_ERRORS
                        && let Some(region) = source_region(root, location)
                    {
                        text.push_str(&region);
                    }
                }
                None => text.push_str(&format!("{}. {}\n", i + 1, diagnostic.headline)),
            }
        }
        // Every diagnostic contributes to the signature even when not listed.
        for diagnostic in diagnostics.iter().skip(MAX_LISTED_ERRORS) {
            signature.insert(format!(
                "diag:{}\0{}",
                diagnostic.headline,
                diagnostic.location.as_deref().unwrap_or("")
            ));
        }
        if diagnostics.len() > MAX_LISTED_ERRORS {
            text.push_str(&format!(
                "… and {} more distinct error(s) in the full output below.\n",
                diagnostics.len() - MAX_LISTED_ERRORS
            ));
        }
    }

    if !failing_tests.is_empty() {
        let names: Vec<&str> = failing_tests
            .iter()
            .map(|(name, _)| name.as_str())
            .collect();
        text.push_str(&format!(
            "{} failing test(s): {}\n",
            names.len(),
            names.join(", ")
        ));
        for (name, excerpt) in failing_tests.iter().take(MAX_TEST_EXCERPTS) {
            signature.insert(format!("test:{name}"));
            if !excerpt.is_empty() {
                text.push_str(&format!("---- {name} ----\n"));
                for line in excerpt {
                    text.push_str(&format!("   {line}\n"));
                }
            }
        }
        for (name, _) in failing_tests.iter().skip(MAX_TEST_EXCERPTS) {
            signature.insert(format!("test:{name}"));
        }
    }

    let failure_count = signature.len();
    Some(FailureDigest {
        text,
        signature,
        failure_count,
    })
}

/// Compare this round's failure set with the previous round's for the same
/// stage, rendering a convergence note the model can act on.
pub(crate) fn convergence_note(
    previous: Option<&(usize, BTreeSet<String>)>,
    digest: &FailureDigest,
) -> String {
    let Some((previous_count, previous_signature)) = previous else {
        return String::new();
    };
    if *previous_signature == digest.signature {
        return "\nNo progress since the previous repair attempt: the same failure(s) persist. \
                Re-read the failing code and reconsider the approach instead of re-applying a \
                similar patch.\n"
            .to_string();
    }
    match digest.failure_count.cmp(previous_count) {
        std::cmp::Ordering::Less => format!(
            "\nProgress: {previous_count} → {} distinct failure(s) since the previous attempt.\n",
            digest.failure_count
        ),
        std::cmp::Ordering::Greater => format!(
            "\nRegression: {previous_count} → {} distinct failure(s) since the previous attempt — \
             the last change introduced new breakage; review it before continuing.\n",
            digest.failure_count
        ),
        std::cmp::Ordering::Equal => {
            "\nThe failure set changed but did not shrink since the previous attempt.\n".to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(digest.text.contains("source (crates/foo/src/lib.rs:"), "{}", digest.text);
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
        assert!(tests[0].1.iter().any(|l| l.contains("AssertionError")), "{:?}", tests[0].1);
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
        assert!(digest.text.contains("1 failing test(s): tests::fails"), "{}", digest.text);
        assert!(digest.text.contains("panicked at src/lib.rs:11:18"), "{}", digest.text);
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
        assert!(diagnostics[0].headline.starts_with("sqlalchemy.exc.ArgumentError:"));
        assert_eq!(diagnostics[0].location.as_deref(), Some("/app/client.py:64"));
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
            println!("--- digest for {} ---\n{}", show.to_string_lossy(), digest.text);
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
        let bigger = (4, digest(&["diag:a", "diag:b", "diag:c", "diag:d"]).signature);
        assert!(convergence_note(Some(&bigger), &current).contains("Progress: 4 → 2"));
        let smaller = (1, digest(&["diag:z"]).signature);
        assert!(convergence_note(Some(&smaller), &current).contains("Regression: 1 → 2"));
    }
}
