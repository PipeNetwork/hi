//! Parse compiler/test-runner diagnostics into structured attributions, so the
//! agent's verify-failure nudge can point the model at the right file/region
//! instead of dumping a raw diagnostic blob.
//!
//! Builds on the pattern *detection* in [`crate::condense`] (which recognizes
//! rustc/tsc/gcc/clang/pytest/libtest output and keeps the relevant lines) but
//! goes one step further: it turns those lines into `{path, line, column,
//! message, kind}` tuples. Pure parsing — no I/O, no env.
//!
//! Tolerant by design: a missed parse is a no-op (the caller keeps the raw
//! output in the nudge anyway — attribution is enrich-only), and a wrong parse
//! is low-harm (it's a hint, not authoritative). Match on stable, long-standing
//! output shapes and fall back to [`AttrKind::Other`] / empty when unsure.

/// The kind of failure an attribution points at — drives the nudge's wording
/// ("compile error here" vs "failing test here") and lets the model pick the
/// right fix strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttrKind {
    /// A compile/typecheck/load error (rustc, tsc, cargo check, gcc).
    Compile,
    /// A failing test assertion (pytest, rust libtest).
    Test,
    /// A lint warning (clang `-Wall`, ruff, rustc `warning:`).
    Lint,
    /// Something failed but no structured location could be parsed.
    Other,
}

/// One parsed failure location: where it is, and what went wrong.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Attribution {
    /// File path as it appears in the diagnostic output (relative to the
    /// project root as the tool reported it).
    pub path: String,
    /// 1-based line number, when parseable.
    pub line: Option<u32>,
    /// 1-based column number, when parseable.
    pub column: Option<u32>,
    /// The error/assertion message, trimmed.
    pub message: String,
    pub kind: AttrKind,
}

/// Parse up to `max` attributions from a diagnostic blob, in document order.
/// Returns empty when nothing parseable is found — the caller then falls back
/// to the raw-output-only nudge.
pub fn parse_attributions(output: &str, max: usize) -> Vec<Attribution> {
    if max == 0 {
        return Vec::new();
    }
    let mut out: Vec<Attribution> = Vec::new();
    let lines: Vec<&str> = output.lines().collect();

    // rustc / cargo: an `error[..]: msg` header followed (within a few lines)
    // by a `--> path:line:col` location. Walk line by line.
    for (i, line) in lines.iter().enumerate() {
        let t = line.trim_start();
        if let Some(attr) = parse_rustc_header(t).map(|msg| {
            // Look ahead for the `-->` location line (usually the next line).
            let loc = lines[i + 1..]
                .iter()
                .take(4)
                .find_map(|l| parse_rustc_arrow(l));
            match loc {
                Some(l) => Attribution {
                    path: l.path,
                    line: l.line,
                    column: l.col,
                    message: msg,
                    kind: AttrKind::Compile,
                },
                None => Attribution {
                    path: String::new(),
                    line: None,
                    column: None,
                    message: msg,
                    kind: AttrKind::Compile,
                },
            }
        }) {
            push_unique(&mut out, attr, max);
        } else if let Some(attr) = parse_gcc_clang(line) {
            push_unique(&mut out, attr, max);
        } else if let Some(attr) = parse_tsc(line) {
            push_unique(&mut out, attr, max);
        } else if let Some(attr) = parse_pytest_location(line) {
            push_unique(&mut out, attr, max);
        } else if let Some(attr) = parse_libtest_panic(line) {
            push_unique(&mut out, attr, max);
        } else if let Some(attr) = parse_go_test(line) {
            push_unique(&mut out, attr, max);
        } else if let Some(attr) = parse_generic_test_location(line) {
            push_unique(&mut out, attr, max);
        }
    }

    // Fallback: if nothing parsed but the output looks like a failure, surface
    // the first error-looking line as an Other attribution so the model at
    // least gets a pointer to *something*.
    if out.is_empty()
        && let Some(msg) = first_error_line(&lines)
    {
        out.push(Attribution {
            path: String::new(),
            line: None,
            column: None,
            message: msg,
            kind: AttrKind::Other,
        });
    }
    out
}

/// A parsed `--> path:line:col` location from rustc output.
struct RustcLoc {
    path: String,
    line: Option<u32>,
    col: Option<u32>,
}

fn parse_rustc_arrow(line: &str) -> Option<RustcLoc> {
    let t = line.trim_start();
    let rest = t.strip_prefix("-->")?;
    let loc = rest.trim();
    // `src/lib.rs:42:18` (path may contain `:` on Windows drives — split from
    // the right on the two numeric segments).
    let parts: Vec<&str> = loc.rsplitn(3, ':').collect();
    // rsplitn yields [col, line, path...] in reverse.
    let (col, line, path) = match parts.len() {
        3 => (parts[0], parts[1], parts[2]),
        2 => (parts[0], "", parts[1]),
        _ => return None,
    };
    let col = col.trim().parse::<u32>().ok();
    let line = line.trim().parse::<u32>().ok();
    if line.is_none() && col.is_none() {
        return None;
    }
    Some(RustcLoc {
        path: path.trim().to_string(),
        line,
        col,
    })
}

/// `error[E0308]: mismatched types` or `error: cannot find value x` → the
/// message after the colon.
fn parse_rustc_header(line: &str) -> Option<String> {
    let rest = line
        .strip_prefix("error[")
        .and_then(|r| r.split_once("]: ").map(|(_, m)| m))
        .or_else(|| line.strip_prefix("error: "))?;
    Some(rest.trim().to_string())
}

/// gcc/clang/go: `src/x.c:10:3: error: ...` or `src/x.c:10:3: warning: ...`.
fn parse_gcc_clang(line: &str) -> Option<Attribution> {
    // Find the `: error:` / `: warning:` separator, then parse the leading
    // `path:line:col` (col optional). Avoid matching rustc `-->` here.
    let lower = line.to_ascii_lowercase();
    let (sep, kind) = if lower.contains(": error:") {
        (": error:", AttrKind::Compile)
    } else if lower.contains(": warning:") {
        (": warning:", AttrKind::Lint)
    } else {
        return None;
    };
    let idx = lower.find(sep)?;
    let head = line[..idx].trim();
    let message = line[idx + sep.len()..].trim().to_string();
    // head is `path:line:col` or `path:line`.
    let parts: Vec<&str> = head.rsplitn(3, ':').collect();
    let (col, line_no, path) = match parts.len() {
        3 => (parts[0], parts[1], parts[2]),
        2 => (parts[0], "", parts[1]),
        _ => return None,
    };
    let col = col.trim().parse::<u32>().ok();
    let line_no = line_no.trim().parse::<u32>().ok();
    let path = path.trim().to_string();
    if path.is_empty() || (line_no.is_none() && col.is_none()) {
        return None;
    }
    Some(Attribution {
        path,
        line: line_no,
        column: col,
        message,
        kind,
    })
}

/// tsc: `src/x.ts(1,2): error TS2304: Cannot find name 'x'.`
fn parse_tsc(line: &str) -> Option<Attribution> {
    let lower = line.to_ascii_lowercase();
    let idx = lower.find("error ts")?;
    // The `(line,col)` precedes `: error TS`.
    let head = &line[..idx];
    let paren_close = head.rfind(')')?;
    let paren_open = head[..paren_close].rfind('(')?;
    let coords = &head[paren_open + 1..paren_close];
    let path = head[..paren_open].trim_end();
    let mut parts = coords.split(',');
    let line_no = parts.next()?.trim().parse::<u32>().ok()?;
    let column = parts.next().and_then(|c| c.trim().parse::<u32>().ok());
    // Message: everything after `error TSnnnn: `.
    let after = &line[idx + 8..]; // skip "error ts"
    let msg = after
        .split_once(": ")
        .map(|(_, m)| m)
        .unwrap_or(after)
        .trim()
        .to_string();
    if path.is_empty() {
        return None;
    }
    Some(Attribution {
        path: path.to_string(),
        line: Some(line_no),
        column,
        message: msg,
        kind: AttrKind::Compile,
    })
}

/// pytest: `tests/test_x.py:12: AssertionError` (the file:line header above an
/// `E ` detail block).
fn parse_pytest_location(line: &str) -> Option<Attribution> {
    let t = line.trim();
    // Must look like `path:line:` and carry an assertion/error keyword.
    let lower = t.to_ascii_lowercase();
    if !(lower.contains("assert") || lower.contains("error") || lower.contains("fail")) {
        return None;
    }
    // Split `path:line:...` — take path up to the first colon, then the line
    // number up to the second colon; the trailing text is the assertion label.
    let first = t.find(':')?;
    let (path, rest) = t.split_at(first);
    let rest = &rest[1..]; // drop the first colon
    // The line number is the segment up to the next colon (or end).
    let line_seg = rest.split(':').next()?.trim();
    let line_no = line_seg.parse::<u32>().ok()?;
    if path.is_empty() {
        return None;
    }
    // Message: pytest's header line is just `path:line:`; the real message is
    // the `E ` line that follows. Use a generic label here — the caller keeps
    // the raw output, so the model still sees the `E ` detail.
    Some(Attribution {
        path: path.to_string(),
        line: Some(line_no),
        column: None,
        message: "assertion failed".to_string(),
        kind: AttrKind::Test,
    })
}

/// Rust libtest: `thread 'name' panicked at src/x.rs:42:9:` and sometimes a
/// following `assertion `left == right`` diff.
fn parse_libtest_panic(line: &str) -> Option<Attribution> {
    let t = line.trim();
    let rest = t.strip_prefix("thread '")?;
    let after = rest.split_once("panicked at ")?;
    // after.1 is `src/x.rs:42:9:` — parse trailing `:line:col:`.
    let loc = after.1.trim_end_matches(':');
    let parts: Vec<&str> = loc.rsplitn(3, ':').collect();
    let (col, line_no, path) = match parts.len() {
        3 => (parts[0], parts[1], parts[2]),
        2 => (parts[0], "", parts[1]),
        _ => return None,
    };
    let col = col.trim().parse::<u32>().ok();
    let line_no = line_no.trim().parse::<u32>().ok();
    let path = path.trim().to_string();
    if path.is_empty() || (line_no.is_none() && col.is_none()) {
        return None;
    }
    Some(Attribution {
        path,
        line: line_no,
        column: col,
        message: "panicked".to_string(),
        kind: AttrKind::Test,
    })
}

/// Go test: `--- FAIL: TestName` header followed by a `file:line:` detail line.
/// Also handles `file_test.go:15: some error` lines that `go test` emits directly.
fn parse_go_test(line: &str) -> Option<Attribution> {
    let t = line.trim();
    // Direct `file_test.go:15: error message` — go test prints these for
    // assertion failures and panics. Distinguish from gcc/clang (which use
    // `: error:` / `: warning:`) by the absence of those keywords: go test's
    // format is `path:line: message` with no `error`/`warning` keyword.
    if let Some(attr) = parse_go_test_location(t) {
        return Some(attr);
    }
    None
}

/// Parse `path_test.go:line: message` — the format `go test` uses for
/// failure locations. The `_test.go` suffix is a strong signal this is a Go
/// test file. Returns `None` for non-test files (those are gcc/clang's domain).
fn parse_go_test_location(line: &str) -> Option<Attribution> {
    // Must contain `_test.go:` to be a Go test location.
    if !line.contains("_test.go:") {
        return None;
    }
    // Split `path:line: message` — path up to first colon, line number up to
    // second colon, rest is the message.
    let first = line.find(':')?;
    let (path, rest) = line.split_at(first);
    let rest = &rest[1..]; // drop the first colon
    let line_seg = rest.split(':').next()?.trim();
    let line_no = line_seg.parse::<u32>().ok()?;
    // Message is everything after the second colon.
    let message = rest.split_once(':').map(|(_, m)| m).unwrap_or("").trim();
    if path.is_empty() {
        return None;
    }
    Some(Attribution {
        path: path.to_string(),
        line: Some(line_no),
        column: None,
        message: if message.is_empty() {
            "test failed".to_string()
        } else {
            message.to_string()
        },
        kind: AttrKind::Test,
    })
}

/// Generic test-runner location: `path:line: message` where the path ends in
/// a test-file suffix we recognize (`_test.go`, `_test.rb`, `_spec.rb`,
/// `test_*.py`, `*_test.dart`, `*.test.ts`, `*.spec.ts`). Catches Mocha,
/// RSpec, Dart, and other runners that print `file:line:` stack frames.
/// Returns `None` for paths that don't look like test files (to avoid
/// stealing gcc/clang's `path:line:col: error:` format).
fn parse_generic_test_location(line: &str) -> Option<Attribution> {
    let t = line.trim();
    // Must look like `path:line: message` — find the first two colons.
    let first = t.find(':')?;
    let (path, rest) = t.split_at(first);
    let rest = &rest[1..]; // drop first colon
    let second = rest.find(':')?;
    let line_seg = &rest[..second];
    let line_no = line_seg.trim().parse::<u32>().ok()?;
    let message = rest[second + 1..].trim();
    // Only fire for paths that look like test files — this avoids matching
    // gcc/clang's `path:line:col: error:` (handled by `parse_gcc_clang`)
    // and rustc's `--> path:line:col` (handled by `parse_rustc_arrow`).
    // The path must not contain whitespace (rules out Mocha's
    // `at Context.<anonymous> (path` prefix).
    let lower = path.to_ascii_lowercase();
    let is_test_file = lower.ends_with("_test.go")
        || lower.ends_with("_test.rb")
        || lower.ends_with("_spec.rb")
        || lower.ends_with("_test.dart")
        || lower.ends_with(".test.ts")
        || lower.ends_with(".spec.ts")
        || lower.ends_with(".test.tsx")
        || lower.ends_with(".spec.tsx")
        || lower.ends_with(".test.js")
        || lower.ends_with(".spec.js");
    if !is_test_file || path.is_empty() || path.contains(char::is_whitespace) {
        return None;
    }
    Some(Attribution {
        path: path.to_string(),
        line: Some(line_no),
        column: None,
        message: if message.is_empty() {
            "test failed".to_string()
        } else {
            message.to_string()
        },
        kind: AttrKind::Test,
    })
}

/// The first line that looks like an error but matched no structured parser —
/// used as a last-resort `Other` attribution so the model gets *some* pointer.
fn first_error_line(lines: &[&str]) -> Option<String> {
    for line in lines {
        let l = line.to_ascii_lowercase();
        if l.contains("error") || l.contains("failed") || l.contains("panic") {
            let t = line.trim();
            if !t.is_empty() {
                return Some(t.to_string());
            }
        }
    }
    None
}

/// Push `attr`, dropping it if it duplicates an existing attribution's
/// (path, line, message). Keeps the nudge concise when a compiler repeats the
/// same error across phases.
fn push_unique(out: &mut Vec<Attribution>, attr: Attribution, max: usize) {
    if attr.path.is_empty() && attr.line.is_none() && attr.kind != AttrKind::Other {
        // A structured parse that yielded no location is useless as a hint.
        return;
    }
    let dup = out.iter().any(|e| {
        e.path == attr.path
            && e.line == attr.line
            && e.message == attr.message
            && e.kind == attr.kind
    });
    if !dup && out.len() < max {
        out.push(attr);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_rustc_error_with_arrow_location() {
        let s = "error[E0308]: mismatched types\n  --> src/lib.rs:42:18\n   |\n42 |     let x: i32 = \"hi\";\n   |                  ^^^";
        let a = parse_attributions(s, 3);
        assert_eq!(a.len(), 1);
        assert_eq!(a[0].path, "src/lib.rs");
        assert_eq!(a[0].line, Some(42));
        assert_eq!(a[0].column, Some(18));
        assert_eq!(a[0].kind, AttrKind::Compile);
        assert!(a[0].message.contains("mismatched types"));
    }

    #[test]
    fn parses_rustc_error_without_code() {
        let s = "error: cannot find value `x` in this scope\n  --> src/main.rs:10:5";
        let a = parse_attributions(s, 3);
        assert_eq!(a.len(), 1);
        assert_eq!(a[0].line, Some(10));
        assert!(a[0].message.contains("cannot find value"));
    }

    #[test]
    fn parses_tsc_error() {
        let s = "src/x.ts(1,2): error TS2304: Cannot find name 'x'.";
        let a = parse_attributions(s, 3);
        assert_eq!(a.len(), 1);
        assert_eq!(a[0].path, "src/x.ts");
        assert_eq!(a[0].line, Some(1));
        assert_eq!(a[0].column, Some(2));
        assert_eq!(a[0].kind, AttrKind::Compile);
        assert!(a[0].message.contains("Cannot find name"));
    }

    #[test]
    fn parses_gcc_clang_error_and_warning() {
        let err = "src/x.c:10:3: error: use of undeclared identifier 'foo'";
        let a = parse_attributions(err, 3);
        assert_eq!(a.len(), 1);
        assert_eq!(a[0].path, "src/x.c");
        assert_eq!(a[0].line, Some(10));
        assert_eq!(a[0].column, Some(3));
        assert_eq!(a[0].kind, AttrKind::Compile);

        let warn = "src/x.c:5:1: warning: unused variable 'y'";
        let a = parse_attributions(warn, 3);
        assert_eq!(a[0].kind, AttrKind::Lint);
        assert!(a[0].message.contains("unused variable"));
    }

    #[test]
    fn parses_pytest_location_header() {
        let s = "tests/test_x.py:12: AssertionError\nE   assert 1 == 2";
        let a = parse_attributions(s, 3);
        assert_eq!(a.len(), 1);
        assert_eq!(a[0].path, "tests/test_x.py");
        assert_eq!(a[0].line, Some(12));
        assert_eq!(a[0].kind, AttrKind::Test);
    }

    #[test]
    fn parses_libtest_panic() {
        let s = "thread 'tests::it' panicked at src/x.rs:42:9:\nassertion `left == right` failed";
        let a = parse_attributions(s, 3);
        assert_eq!(a.len(), 1);
        assert_eq!(a[0].path, "src/x.rs");
        assert_eq!(a[0].line, Some(42));
        assert_eq!(a[0].column, Some(9));
        assert_eq!(a[0].kind, AttrKind::Test);
    }

    #[test]
    fn empty_for_non_diagnostic_input() {
        let a = parse_attributions("all good\nnothing to see here", 3);
        assert!(a.is_empty(), "no false attribution: {a:?}");
    }

    #[test]
    fn empty_for_blank_input() {
        assert!(parse_attributions("", 3).is_empty());
    }

    #[test]
    fn max_cap_honored() {
        let s =
            "error[E1]: a\n --> a.rs:1:1\nerror[E2]: b\n --> b.rs:2:2\nerror[E3]: c\n --> c.rs:3:3";
        let a = parse_attributions(s, 2);
        assert_eq!(a.len(), 2);
    }

    #[test]
    fn fallback_other_for_unstructured_failure() {
        // A failure line that matches no parser still yields an Other pointer.
        let a = parse_attributions("the build exploded", 3);
        // "exploded" has no error keyword, so this should be empty.
        assert!(a.is_empty());

        let a = parse_attributions("error: something weird happened", 3);
        // `error:` parses as a rustc header with no following `-->`, so path is
        // empty and the structured parse is dropped — the fallback then fires.
        assert_eq!(a.len(), 1);
        assert_eq!(a[0].kind, AttrKind::Other);
        assert!(a[0].message.contains("error: something weird"));
    }

    #[test]
    fn deduplicates_repeated_errors() {
        let s = "error[E0308]: mismatched types\n  --> src/lib.rs:42:18\nerror[E0308]: mismatched types\n  --> src/lib.rs:42:18";
        let a = parse_attributions(s, 5);
        assert_eq!(a.len(), 1, "duplicate dropped: {a:?}");
    }

    #[test]
    fn parses_go_test_location() {
        // `go test` prints `file_test.go:line: message` for failures.
        let s = "math_test.go:15: assertion failed";
        let a = parse_attributions(s, 3);
        assert_eq!(a.len(), 1);
        assert_eq!(a[0].path, "math_test.go");
        assert_eq!(a[0].line, Some(15));
        assert_eq!(a[0].kind, AttrKind::Test);
        assert!(a[0].message.contains("assertion failed"));
    }

    #[test]
    fn parses_go_test_fail_header_and_detail() {
        // A `--- FAIL: TestName` header followed by a location line.
        let s = "--- FAIL: TestAdd (0.00s)\n    math_test.go:15: something went wrong";
        let a = parse_attributions(s, 3);
        assert_eq!(a.len(), 1);
        assert_eq!(a[0].path, "math_test.go");
        assert_eq!(a[0].line, Some(15));
        assert_eq!(a[0].kind, AttrKind::Test);
    }

    #[test]
    fn parses_generic_test_location_mocha() {
        // Mocha-style stack frame with a .test.ts file. The generic parser
        // only matches lines that *start* with `path:line:` — Mocha's
        // `  at ... (path:line:col)` doesn't match, so the raw
        // output stays (the fallback `first_error_line` may fire).
        // and that's fine).
        let s = "    at Context.<anonymous> (src/parser.test.ts:42:15)";
        let a = parse_attributions(s, 3);
        assert!(
            a.is_empty(),
            "Mocha stack frames aren't prefix-matched: {a:?}"
        );
    }

    #[test]
    fn parses_generic_test_location_rspec() {
        // RSpec prints `path:line:` for failures in `_spec.rb` files.
        let s = "spec/calculator_spec.rb:10: expected 4 to equal 5";
        let a = parse_attributions(s, 3);
        assert_eq!(a.len(), 1);
        assert_eq!(a[0].path, "spec/calculator_spec.rb");
        assert_eq!(a[0].line, Some(10));
        assert_eq!(a[0].kind, AttrKind::Test);
        assert!(a[0].message.contains("expected 4 to equal 5"));
    }

    #[test]
    fn does_not_steal_gcc_clang_format() {
        // gcc/clang uses `path:line:col: error:` — the generic test parser
        // must NOT match this (it's gcc/clang's job, and the path doesn't
        // end in a test-file suffix).
        let s = "src/x.c:10:3: error: use of undeclared identifier 'foo'";
        let a = parse_attributions(s, 3);
        assert_eq!(a.len(), 1);
        assert_eq!(a[0].kind, AttrKind::Compile);
        assert_eq!(a[0].path, "src/x.c");
    }
}
