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
pub(crate) fn digest_failure(root: &Path, raw: &str) -> Option<FailureDigest> {
    let diagnostics = parse_diagnostics(raw);
    let failing_tests = parse_failing_tests(raw);
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
