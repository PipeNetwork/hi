//! Deterministic seeding of a `/review` audit: the verify command (or
//! `cargo check`) runs before the audit and each re-audit, and its result
//! is handed to the model as candidate defects instead of letting the audit
//! start cold. When it fails at a file in the workspace and the model still
//! reports no P0/P1, the failure itself becomes a P1 finding so the fix
//! loop has something to fix and the exit code says the tree is not clean.
//! A failure that names no workspace file (no toolchain, no network, a
//! read-only sandbox) is reported, not seeded: no fix pass can reach it.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::review::{Finding, Severity};

/// Longest output tail quoted in a prompt or kept in the drive.
const SEED_TAIL_CHARS: usize = 3_000;

/// Longest first-failure line kept for a finding title.
const SEED_HEAD_CHARS: usize = 160;

/// One run of the seed command, kept on the drive for the report.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifySeed {
    pub command: String,
    pub passed: bool,
    /// Turn it ran before: `audit`, `re-audit 1`.
    pub before: String,
    /// Tail of the output (bounded), for the prompt and the report.
    pub tail: String,
    /// First line that names a failure, for the synthetic finding.
    #[serde(default)]
    pub head: Option<String>,
    /// `path:line` in the workspace that the output points at, if any.
    #[serde(default)]
    pub location: Option<String>,
}

impl VerifySeed {
    /// Summarize a run of `command` that printed `output` (stdout and
    /// stderr as the runner joined them) before the `before` turn.
    pub fn from_output(
        root: &Path,
        command: &str,
        passed: bool,
        output: &str,
        before: &str,
    ) -> Self {
        let head = (!passed).then(|| failure_head(output)).flatten();
        let location = (!passed).then(|| output_location(root, output)).flatten();
        Self {
            command: command.to_string(),
            passed,
            before: before.to_string(),
            tail: tail(output, SEED_TAIL_CHARS),
            head,
            location,
        }
    }

    /// `cargo test --offline · failed before audit`.
    pub fn summary(&self) -> String {
        format!(
            "{} · {} before {}",
            self.command,
            if self.passed { "passed" } else { "failed" },
            self.before
        )
    }

    /// The prompt section: the result, the failing output, and what to do
    /// with it.
    pub fn prompt_section(&self) -> String {
        if self.passed {
            return format!(
                "\nVerify before this turn: `{}` passed. A defect you report must be something it does not cover.\n",
                self.command
            );
        }
        let mut out = format!(
            "\nVerify before this turn: `{}` FAILED. Output tail:\n```\n{}\n```\n\
Each failure is a candidate defect: find its cause in the code under audit and report it as a `finding:` row with the \
`path:line` of the cause (not of the test). A failure whose cause is outside the code under audit is not a finding; \
name it in `residual:`.\n",
            self.command,
            self.tail.trim_end()
        );
        if let Some(location) = &self.location {
            out.push_str(&format!("The output points at {location}.\n"));
        }
        out
    }

    /// The P1 that stands in for a failing verify the model did not report.
    pub fn synthetic_finding(&self) -> Finding {
        let mut title = format!("`{}` fails", self.command);
        if let Some(head) = &self.head {
            title.push_str(": ");
            title.push_str(head);
        }
        Finding {
            severity: Severity::P1,
            title,
            location: self.location.clone(),
            verified: true,
            feature_gap: false,
            from_verify: true,
        }
    }
}

fn tail(text: &str, chars: usize) -> String {
    let count = text.chars().count();
    if count <= chars {
        return text.to_string();
    }
    let skipped: String = text.chars().skip(count - chars).collect();
    match skipped.find('\n') {
        Some(newline) if newline + 1 < skipped.len() => {
            format!("…{}", &skipped[newline..])
        }
        _ => format!("…{skipped}"),
    }
}

/// The first line that reads like a failure: a compiler `error`, a test
/// `FAILED`/`panicked`, an assertion, or a non-zero exit note. A passing
/// binary's `test result: ok. 3 passed; 0 failed` is not one.
fn failure_head(output: &str) -> Option<String> {
    const MARKERS: &[&str] = &[
        "error[",
        "error:",
        "panicked at",
        "FAILED",
        "failed",
        "assertion",
        "Error:",
        "Traceback",
        "[exit code",
    ];
    output
        .lines()
        .map(|line| strip_ansi(line).trim().to_string())
        .find(|line| {
            !line.is_empty()
                && MARKERS.iter().any(|marker| line.contains(marker))
                && !line.starts_with("warning")
                && !line.starts_with("test result: ok")
        })
        .map(|line| {
            if line.chars().count() > SEED_HEAD_CHARS {
                let cut: String = line.chars().take(SEED_HEAD_CHARS).collect();
                format!("{cut}…")
            } else {
                line
            }
        })
}

/// The first `path:line` in `output` whose path is a file in the workspace
/// under `root` (`--> src/a.rs:12:5`, `src/a.rs:12`; `File "x.py", line 3`
/// is not matched: Python's form has no colon). A location that belongs to
/// a compiler warning is skipped when a later error names a file, so the
/// fix pass is not aimed at the warning. A file outside the workspace
/// (`~/.cargo/registry/…`, `../other/x.rs`) is not a place a fix pass can
/// reach.
fn output_location(root: &Path, output: &str) -> Option<String> {
    let lines: Vec<String> = output.lines().map(strip_ansi).collect();
    // A warning's `--> path:line` comes before the error that failed the
    // run. Citing that file sends the fix pass at the warning.
    location_after(root, &lines, false).or_else(|| location_after(root, &lines, true))
}

/// The first workspace `path:line` in `lines`. With `warnings` false, skip
/// a warning and the location lines that follow it until a blank line or an
/// error. With `warnings` true, take the first location at all (a
/// warnings-as-errors run may have nothing else).
fn location_after(root: &Path, lines: &[String], warnings: bool) -> Option<String> {
    let mut after_warning = false;
    for line in lines {
        let trimmed = line.trim();
        if !warnings {
            if trimmed.starts_with("warning") {
                after_warning = true;
                continue;
            }
            if trimmed.is_empty()
                || trimmed.starts_with("error")
                || trimmed.contains("FAILED")
                || trimmed.contains("panicked at")
            {
                after_warning = false;
            }
            if after_warning {
                continue;
            }
        }
        if let Some(location) = line_location(root, line) {
            return Some(location);
        }
    }
    None
}

fn line_location(root: &Path, line: &str) -> Option<String> {
    line.split(|ch: char| ch.is_whitespace() || matches!(ch, '(' | ')' | ',' | '\'' | '"' | '`'))
        .find_map(|token| {
            let (path, rest) = token.split_once(':')?;
            let line_no: String = rest.chars().take_while(char::is_ascii_digit).collect();
            if line_no.is_empty() || path.is_empty() {
                return None;
            }
            crate::review_reads::workspace_file(root, path).map(|path| format!("{path}:{line_no}"))
        })
}

fn strip_ansi(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\u{1b}' {
            if chars.peek() == Some(&'[') {
                chars.next();
                for next in chars.by_ref() {
                    if next.is_ascii_alphabetic() {
                        break;
                    }
                }
            }
            continue;
        }
        out.push(ch);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failed_run_keeps_head_location_and_tail_and_seeds_a_p1() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/main.rs"), "fn main() {}\n").unwrap();
        let output = "   Compiling x\nwarning: unused variable\nrunning 3 tests\ntest welcome ... FAILED\n\nfailures:\n\n---- welcome stdout ----\nthread 'welcome' panicked at tests/integration.rs:40:5:\nno Welcome\n  --> \u{1b}[1msrc/main.rs:66:9\u{1b}[0m\n[exit code 101]";
        let seed =
            VerifySeed::from_output(dir.path(), "cargo test --offline", false, output, "audit");
        assert_eq!(seed.head.as_deref(), Some("test welcome ... FAILED"));
        assert_eq!(
            seed.location.as_deref(),
            Some("src/main.rs:66"),
            "tests/integration.rs is not in the tree, src/main.rs is"
        );
        assert_eq!(seed.summary(), "cargo test --offline · failed before audit");
        let finding = seed.synthetic_finding();
        assert_eq!(finding.severity, Severity::P1);
        assert_eq!(
            finding.title,
            "`cargo test --offline` fails: test welcome ... FAILED"
        );
        assert_eq!(finding.location.as_deref(), Some("src/main.rs:66"));
        assert!(finding.from_verify);
        let section = seed.prompt_section();
        assert!(section.contains("`cargo test --offline` FAILED"));
        assert!(section.contains("[exit code 101]"));
        assert!(section.contains("The output points at src/main.rs:66."));
        assert!(section.contains("`path:line` of the cause (not of the test)"));
    }

    #[test]
    fn head_skips_a_passing_summary_and_the_location_stays_in_the_workspace() {
        let outer = tempfile::tempdir().unwrap();
        let root = outer.path().join("ws");
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("src/lib.rs"), "\n").unwrap();
        std::fs::write(outer.path().join("dep.rs"), "\n").unwrap();
        let registry = outer.path().join("dep.rs").display().to_string();
        let output = format!(
            "test result: ok. 3 passed; 0 failed; 0 ignored\ntest parse ... FAILED\n\
thread 'parse' panicked at {registry}:7:1:\n --> ../dep.rs:7:1\n --> src/lib.rs:12:5\n"
        );
        let seed = VerifySeed::from_output(&root, "cargo test", false, &output, "audit");
        assert_eq!(seed.head.as_deref(), Some("test parse ... FAILED"));
        assert_eq!(seed.location.as_deref(), Some("src/lib.rs:12"));

        std::fs::write(root.join("src/other.rs"), "\n").unwrap();
        let warned = "warning: unused import\n --> src/other.rs:1:5\n  |\nerror[E0308]: mismatched types\n --> src/lib.rs:4:1\n";
        let seed = VerifySeed::from_output(&root, "cargo check", false, warned, "audit");
        assert_eq!(
            seed.location.as_deref(),
            Some("src/lib.rs:4"),
            "a warning's path is not the failure"
        );

        let missing = VerifySeed::from_output(
            &root,
            "cargo check",
            false,
            "sh: 1: cargo: not found\n[exit code 127]",
            "audit",
        );
        assert_eq!(missing.location, None, "nothing in the workspace to fix");
    }

    #[test]
    fn passed_run_has_no_head_and_a_one_line_section() {
        let dir = tempfile::tempdir().unwrap();
        let seed = VerifySeed::from_output(dir.path(), "make test", true, "ok\n", "re-audit 1");
        assert_eq!(seed.head, None);
        assert_eq!(seed.location, None);
        assert_eq!(seed.summary(), "make test · passed before re-audit 1");
        assert_eq!(
            seed.prompt_section(),
            "\nVerify before this turn: `make test` passed. A defect you report must be something it does not cover.\n"
        );
    }

    #[test]
    fn tail_is_bounded_at_a_line_start() {
        let long: String = (0..500).map(|i| format!("line {i}\n")).collect();
        let kept = tail(&long, 100);
        assert!(kept.starts_with("…\nline "), "{kept:?}");
        assert!(kept.chars().count() <= 102);
        assert_eq!(tail("short", 100), "short");
    }
}
