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

use sha2::{Digest, Sha256};

#[path = "verify_digest_source.rs"]
mod source;
use source::source_region;

/// Distinct diagnostics to list in full (with source spans for the first few).
const MAX_LISTED_ERRORS: usize = 8;
/// Diagnostics that get their source region inlined.
const MAX_SPANNED_ERRORS: usize = 3;
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

/// Render a stored identity without exposing its internal separators or hashes.
/// Keep the original identity intact for recovery comparisons and session replay.
pub(crate) fn display_signature_item(identity: &str) -> String {
    let (headline, detail) = identity.split_once('\0').unwrap_or((identity, ""));
    let text = if let Some(name) = headline.strip_prefix("test:") {
        format!("Failing test: {name}")
    } else {
        let headline = headline.strip_prefix("diag:").unwrap_or(headline);
        if detail.is_empty() {
            headline.to_owned()
        } else {
            format!("{headline} ({detail})")
        }
    };
    text.chars()
        .map(|character| {
            if character.is_control() {
                ' '
            } else {
                character
            }
        })
        .collect()
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
                    let line_no: String = rest.chars().take_while(char::is_ascii_digit).collect();
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

/// Failing libtest cases, plus each failing test's `---- name stdout ----`
/// section for the excerpt. Two libtest formats exist: the verbose
/// `test name ... FAILED`, and the `--quiet` form `name --- FAILED` — which
/// is what hi's own verify stages emit (`cargo test --quiet`). Missing the
/// quiet form meant every failing verify fed the model a raw wall instead
/// of the structured digest (observed live: 6/6 recent verify failures
/// unstructured).
fn parse_failing_tests(raw: &str) -> Vec<(String, Vec<String>)> {
    let mut names: Vec<String> = Vec::new();
    let mut seen = BTreeSet::new();
    for line in raw.lines() {
        let trimmed = line.trim();
        let name = if let Some(rest) = trimmed.strip_prefix("test ") {
            rest.strip_suffix("... FAILED")
        } else {
            trimmed.strip_suffix("--- FAILED")
        };
        if let Some(name) = name {
            let name = name.trim().to_string();
            // Quiet-format guard: a libtest case path, not prose that happens
            // to end in "FAILED" (e.g. "test result: FAILED. 1 passed…").
            let plausible = !name.is_empty()
                && !name.contains(' ')
                && name
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, ':' | '_' | '-'));
            if plausible && seen.insert(name.clone()) {
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

/// A failing test name alone is too coarse for repair convergence: the same
/// test can move from one assertion/state mismatch to another. Bind the name
/// to its bounded failure detail while discarding Rust's source-location-only
/// panic line, which commonly shifts after an otherwise equivalent edit.
fn failing_test_signature(name: &str, excerpt: &[String]) -> String {
    let stable_detail = excerpt
        .iter()
        .filter(|line| !line.contains(" panicked at "))
        .flat_map(|line| line.split_whitespace())
        .collect::<Vec<_>>()
        .join(" ");
    if stable_detail.is_empty() {
        return format!("test:{name}");
    }
    format!(
        "test:{name}\0detail:{:x}",
        Sha256::digest(crate::recovery::normalize_diagnostic(&stable_detail).as_bytes())
    )
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
                crate::recovery::normalize_diagnostic(&diagnostic.headline),
                crate::recovery::normalize_diagnostic(diagnostic.location.as_deref().unwrap_or(""))
            ));
            match &diagnostic.location {
                Some(location) => {
                    text.push_str(&format!(
                        "{}. {} — {location}\n",
                        i + 1,
                        diagnostic.headline
                    ));
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
                crate::recovery::normalize_diagnostic(&diagnostic.headline),
                crate::recovery::normalize_diagnostic(diagnostic.location.as_deref().unwrap_or(""))
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
            .take(MAX_LISTED_ERRORS)
            .map(|(name, _)| name.as_str())
            .collect();
        text.push_str(&format!(
            "{} failing test(s): {}\n",
            failing_tests.len(),
            names.join(", ")
        ));
        if failing_tests.len() > names.len() {
            text.push_str(&format!(
                "… and {} more failing test(s) in the full output below.\n",
                failing_tests.len() - names.len()
            ));
        }
        for (name, excerpt) in failing_tests.iter().take(MAX_TEST_EXCERPTS) {
            signature.insert(failing_test_signature(name, excerpt));
            if !excerpt.is_empty() {
                text.push_str(&format!("---- {name} ----\n"));
                for line in excerpt {
                    text.push_str(&format!("   {line}\n"));
                }
            }
        }
        for (name, excerpt) in failing_tests.iter().skip(MAX_TEST_EXCERPTS) {
            signature.insert(failing_test_signature(name, excerpt));
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
#[path = "verify_digest_tests.rs"]
mod tests;
