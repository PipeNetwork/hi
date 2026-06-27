use std::sync::LazyLock;

/// Per-result character budget so a single read or noisy command can't blow the
/// context. Overridable via `HI_TOOL_RESULT_CHARS` — lower it for a tight local
/// window, raise it when the model has room. Read once, at first use. The
/// default is intentionally tight for remote agent loops: ~5k chars is ~1.3k
/// tokens, and repeated tool rounds resend this history.
pub(crate) static MAX_OUTPUT_CHARS: LazyLock<usize> = LazyLock::new(|| {
    std::env::var("HI_TOOL_RESULT_CHARS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n >= 1_000)
        .unwrap_or(5_000)
});

/// Clip output to the configured character budget ([`MAX_OUTPUT_CHARS`]).
pub(crate) fn truncate(s: &str) -> String {
    truncate_to(s, *MAX_OUTPUT_CHARS)
}

/// Whether the diagnostic condenser is enabled. Off falls back to plain head+tail
/// truncation — the knob that lets the eval harness A/B the condenser's value.
/// Read once, at first use.
static CONDENSE_ENABLED: LazyLock<bool> =
    LazyLock::new(|| condense_enabled(std::env::var("HI_CONDENSE").ok().as_deref()));

/// Parse the `HI_CONDENSE` toggle: on by default; `0`/`off`/`false`/`no` disable.
pub(crate) fn condense_enabled(var: Option<&str>) -> bool {
    !matches!(var, Some("0" | "off" | "false" | "no"))
}

/// Condense output to the configured budget (test/diagnostic-aware), unless the
/// condenser is disabled — then just head+tail clip.
pub(crate) fn condense(s: &str) -> String {
    if *CONDENSE_ENABLED {
        condense_diagnostics(s, *MAX_OUTPUT_CHARS)
    } else {
        truncate(s)
    }
}

/// Condense test-runner and compiler output: keep the head, the summary, and
/// every failure — test failures with a little surrounding context, and whole
/// compiler-diagnostic blocks (the `error[...]` line plus its `-->` location,
/// code frame, and notes) — while dropping the long runs of passing `... ok`
/// noise. A 4,000-line green run with three failures collapses to those three
/// failures plus the count. Output that doesn't look like a test/diagnostic run,
/// or that has nothing of interest in its body, falls back to plain head+tail
/// [`truncate_to`] (don't mangle a file dump).
///
/// Deliberately biased to *over*-keep: a stray noise line surviving is harmless,
/// but dropping a real failure would send the model after the wrong thing. This
/// is the "Tier 0" deterministic extractor — no model call, reproducible on the
/// eval harness.
pub fn condense_diagnostics(s: &str, max: usize) -> String {
    if !looks_like_diagnostics(s) {
        return truncate_to(s, max);
    }
    let lines: Vec<&str> = s.lines().collect();
    let n = lines.len();
    const HEAD: usize = 4; // the "running N tests" / session preamble
    const TAIL: usize = 6; // the summary line(s) live at the very end
    const CONTEXT: usize = 2; // lines kept on each side of a test-failure line
    const MAX_BLOCK: usize = 40; // cap on a single compiler-diagnostic block

    let mut keep = vec![false; n];
    // Whether anything in the *body* (past the head, before the tail) was worth
    // keeping. If not, detection was a false positive — clip normally instead of
    // emitting a misleading "everything omitted" digest.
    let mut matched = false;
    for i in 0..n {
        let line = lines[i];
        if starts_diagnostic_block(line) {
            // Keep the whole multi-line block: rustc/gcc/clang print the location,
            // code frame, and notes under the `error:` line, ending at a blank
            // line. Bounded so a pathological block can't run away.
            matched = true;
            let end = (i + MAX_BLOCK).min(n);
            for (off, l) in lines[i..end].iter().enumerate() {
                if off > 0 && l.trim().is_empty() {
                    break;
                }
                keep[i + off] = true;
            }
        } else if is_signal_line(line) {
            matched = true;
            let lo = i.saturating_sub(CONTEXT);
            let hi = (i + CONTEXT + 1).min(n);
            for slot in keep.iter_mut().take(hi).skip(lo) {
                *slot = true;
            }
        }
        // Always keep the head preamble and the trailing summary, wherever the
        // failures fall — but signals are detected everywhere, so errors sitting
        // in the tail still mark this as real diagnostics (not a false positive).
        if i < HEAD || i + TAIL >= n {
            keep[i] = true;
        }
    }
    if !matched {
        return truncate_to(s, max);
    }
    // When almost everything is a signal (a wall of failures), there's no green
    // noise to drop — head+tail clip the original rather than pepper it with
    // omission markers.
    let kept = keep.iter().filter(|&&k| k).count();
    if kept * 10 >= n * 9 {
        return truncate_to(s, max);
    }

    let mut out = String::new();
    let mut i = 0;
    while i < n {
        if keep[i] {
            out.push_str(lines[i]);
            out.push('\n');
            i += 1;
        } else {
            let start = i;
            while i < n && !keep[i] {
                i += 1;
            }
            // A tiny gap (e.g. a blank line between two error blocks) is cheaper
            // to show than to announce, and keeps the output readable.
            let gap = i - start;
            if gap <= 2 {
                for l in &lines[start..i] {
                    out.push_str(l);
                    out.push('\n');
                }
            } else {
                out.push_str(&format!("… {gap} lines omitted …\n"));
            }
        }
    }
    // Even the condensed view honours the char budget.
    truncate_to(out.trim_end(), max)
}

/// Whether output looks like a test run or compiler diagnostics, and so is worth
/// condensing rather than blind-clipping. Specific markers keep this from firing
/// on an ordinary command dump.
pub(crate) fn looks_like_diagnostics(s: &str) -> bool {
    let l = s.to_ascii_lowercase();
    // Test runners.
    l.contains("test result:")                             // rust libtest summary
        || l.contains("=== failures ===")                  // pytest
        || l.contains("short test summary")                // pytest
        || l.contains("collected ")                        // pytest "collected N items"
        || (l.contains("running ") && l.contains(" test")) // libtest "running N tests"
        || (l.contains(" passed") && (l.contains(" failed") || l.contains(" error")))
        || l.contains("--- fail:")                         // go test
        || l.contains("fail:")                             // go test (case varies)
        // Compilers.
        || l.contains("error[")            // rustc, with an error code
        || l.contains("could not compile") // cargo
        || l.contains("error ts")          // tsc: "error TS2322"
        || l.contains(": error:")          // gcc/clang/go: file:line:col: error:
        || l.contains(": warning:")
}

/// Whether a line *begins* a multi-line compiler diagnostic whose whole block
/// (location, code frame, notes) should be kept together.
pub(crate) fn starts_diagnostic_block(line: &str) -> bool {
    let t = line.trim_start();
    t.starts_with("error[")          // rustc:  error[E0308]: mismatched types
        || t.starts_with("error:")   // rustc/cargo:  error: cannot find value …
        || t.starts_with("warning:") // rustc:  warning: unused variable …
        || line.contains(": error:") // gcc/clang/go:  src/x.c:4:9: error: …
        || line.contains(": warning:")
        || line.contains("error TS") // tsc:  x.ts(1,2): error TS2322: …
}

/// Whether a line carries failure/summary signal worth keeping (test-runner
/// output). Broad on purpose — see [`condense_diagnostics`]'s over-keep bias.
pub(crate) fn is_signal_line(line: &str) -> bool {
    // pytest prints assertion detail on lines that start with `E ` but may carry
    // no keyword of their own.
    if line.starts_with("E ") {
        return true;
    }
    let l = line.to_ascii_lowercase();
    const SIGNALS: [&str; 14] = [
        "fail",     // "test x ... FAILED", "failures:", "N failed"
        "error",    // "error[E..]", "error:", pytest "ERROR"
        "panic",    // rust panic
        "assert",   // assertion failed / pytest assert
        "thread '", // rust panic location
        "left:",    // assert_eq! diff
        "right:",
        "exception", // python
        "traceback", // python
        "expected",  // assertions
        "could not compile",
        "test result:", // libtest summary
        "short test summary",
        "=====", // pytest section rules (FAILURES / summary)
    ];
    SIGNALS.iter().any(|p| l.contains(p))
}

/// Clip to `max` chars keeping *both* ends. For command and test output the tail
/// — failures, summaries, `error[...]` lines — is usually the most useful part,
/// so head-only truncation drops the signal. Splits the budget ~60% head / ~40%
/// tail and notes how much of the middle went. Split out so tests can set `max`.
pub(crate) fn truncate_to(s: &str, max: usize) -> String {
    let total = s.chars().count();
    if total <= max {
        return s.to_string();
    }
    let head_budget = max * 6 / 10;
    let tail_budget = max - head_budget;
    let head: String = s.chars().take(head_budget).collect();
    let tail: String = s.chars().skip(total - tail_budget).collect();
    let dropped = total - head_budget - tail_budget;
    format!("{head}\n… [truncated {dropped} characters] …\n{tail}")
}

#[cfg(test)]
mod tests {
    use super::{condense_diagnostics, condense_enabled, truncate_to};

    #[test]
    fn condense_toggle_defaults_on_and_parses_off_values() {
        assert!(condense_enabled(None), "default on when unset");
        assert!(condense_enabled(Some("1")), "any other value is on");
        assert!(condense_enabled(Some("on")));
        for off in ["0", "off", "false", "no"] {
            assert!(!condense_enabled(Some(off)), "{off} disables");
        }
    }

    /// A realistic libtest run: a `header`, `passing` "... ok" lines, an injected
    /// failure block somewhere in the middle, and the final summary.
    fn cargo_log(passing: usize, fail_at: usize) -> String {
        let mut out = format!("\nrunning {} tests\n", passing + 1);
        for i in 0..passing {
            if i == fail_at {
                out.push_str("test mod::middle_case ... FAILED\n");
            }
            out.push_str(&format!("test mod::case_{i:04} ... ok\n"));
        }
        out.push_str(
            "\nfailures:\n\n---- mod::middle_case stdout ----\n\
             thread 'mod::middle_case' panicked at src/lib.rs:42:5:\n\
             assertion `left == right` failed\n  left: 3\n right: 4\n\n\
             failures:\n    mod::middle_case\n\n\
             test result: FAILED. {passing} passed; 1 failed; 0 ignored\n",
        );
        out
    }

    #[test]
    fn condense_keeps_cargo_failure_and_summary_drops_green() {
        let log = cargo_log(400, 200);
        let out = condense_diagnostics(&log, 50_000);
        // The failure, its panic detail, and the summary all survive…
        assert!(
            out.contains("middle_case ... FAILED"),
            "keeps the failing test"
        );
        assert!(out.contains("left: 3"), "keeps the assertion detail");
        assert!(out.contains("test result: FAILED"), "keeps the summary");
        // …while the green noise is collapsed.
        assert!(out.contains("lines omitted"), "drops passing lines: {out}");
        assert!(
            out.len() < log.len() / 3,
            "much smaller: {} vs {}",
            out.len(),
            log.len()
        );
    }

    #[test]
    fn condense_beats_head_tail_when_failure_is_in_the_middle() {
        // The money case: one failure buried in the middle of a long green run,
        // with a budget tight enough that blind head+tail would clip it out.
        let log = cargo_log(1000, 500);
        let budget = 8_000;
        assert!(
            !truncate_to(&log, budget).contains("middle_case ... FAILED"),
            "head+tail drops the middle failure"
        );
        assert!(
            condense_diagnostics(&log, budget).contains("middle_case ... FAILED"),
            "condense preserves it"
        );
    }

    #[test]
    fn condense_keeps_pytest_failures() {
        let log = "\
collected 50 items

tests/test_a.py ..........................................         [ 96%]
tests/test_b.py F                                                  [100%]

=================================== FAILURES ===================================
________________________________ test_parsing _________________________________

    def test_parsing():
>       assert parse('1+1') == 3
E       assert 2 == 3

tests/test_b.py:12: AssertionError
=========================== short test summary info ============================
FAILED tests/test_b.py::test_parsing - assert 2 == 3
========================= 1 failed, 49 passed in 0.42s =========================
";
        let out = condense_diagnostics(log, 50_000);
        assert!(out.contains("test_parsing"), "keeps the failing test name");
        assert!(
            out.contains("assert 2 == 3"),
            "keeps the assertion (E line)"
        );
        assert!(out.contains("1 failed, 49 passed"), "keeps the summary");
    }

    #[test]
    fn condense_passes_through_non_test_output() {
        // A plain file/command dump (no test markers) is left untouched when it
        // fits — condense must not mangle ordinary output.
        let dump = "fn main() {\n    println!(\"hello\");\n}\n";
        assert_eq!(condense_diagnostics(dump, 50_000), dump);
    }

    #[test]
    fn condense_keeps_whole_rustc_error_block() {
        // A `cargo build` run: a wall of "Compiling …" noise, one multi-line
        // rustc diagnostic (location + code frame + note), then the summary.
        let mut log = String::new();
        for i in 0..40 {
            log.push_str(&format!("   Compiling crate_{i} v0.1.0 (/tmp/crate_{i})\n"));
        }
        log.push_str(
            "error[E0308]: mismatched types\n  \
             --> src/lib.rs:42:18\n   |\n\
             42 |     let x: u32 = \"hi\";\n   \
             |            ---   ^^^^ expected `u32`, found `&str`\n   |\n   \
             = note: expected type `u32`\n\n\
             error: could not compile `app` (lib) due to 1 previous error\n",
        );
        let out = condense_diagnostics(&log, 50_000);
        // The entire diagnostic block survives — code, the caret line, the note…
        assert!(out.contains("error[E0308]"), "keeps the error line");
        assert!(out.contains("--> src/lib.rs:42:18"), "keeps the location");
        assert!(
            out.contains("expected `u32`, found `&str`"),
            "keeps the code frame / caret line"
        );
        assert!(out.contains("= note: expected type"), "keeps the note");
        assert!(out.contains("could not compile"), "keeps the summary");
        // …while the "Compiling …" noise is dropped.
        assert!(
            out.contains("lines omitted"),
            "drops the compile noise: {out}"
        );
    }

    #[test]
    fn condense_keeps_tsc_errors() {
        let mut log = String::from("> tsc --noEmit\n\n");
        for i in 0..40 {
            log.push_str(&format!("  checking module_{i:02}.ts\n"));
        }
        log.push_str(
            "src/index.ts(10,7): error TS2322: Type 'string' is not assignable to type 'number'.\n\
             src/index.ts(15,3): error TS2554: Expected 1 arguments, but got 0.\n\n\
             Found 2 errors in the same file, starting at: src/index.ts:10\n",
        );
        let out = condense_diagnostics(&log, 50_000);
        assert!(out.contains("error TS2322"), "keeps the first tsc error");
        assert!(out.contains("error TS2554"), "keeps the second tsc error");
        assert!(out.contains("Found 2 errors"), "keeps the summary");
        assert!(out.contains("lines omitted"), "drops the checking noise");
    }

    #[test]
    fn truncate_keeps_head_and_tail() {
        // 300 chars, budget 100 → keep 60 head + 40 tail, drop the 200 middle.
        let s = format!("{}{}{}", "A".repeat(100), "M".repeat(100), "Z".repeat(100));
        let out = truncate_to(&s, 100);
        assert!(out.starts_with(&"A".repeat(60)), "keeps the head");
        assert!(out.trim_end().ends_with(&"Z".repeat(40)), "keeps the tail");
        assert!(!out.contains('M'), "drops the middle");
        assert!(
            out.contains("truncated 200 characters"),
            "notes the gap: {out}"
        );
        // Under budget passes through untouched.
        assert_eq!(truncate_to("short", 100), "short");
    }
}
