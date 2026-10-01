//! The drive's read-coverage accounting (continuation turns, partial
//! audits), the seeded finding from a failing verify command, the
//! uncited-change warning, and the fuzzy no-progress stop.

use super::*;
use crate::review::{
    CoverageRow, CoverageState, GitScope, GitSource, InputKind, REVIEW_DEFECTS_FORMAT_BLOCK,
    ReviewInput, Severity,
};

fn finding(severity: Severity, title: &str, location: &str) -> Finding {
    Finding {
        severity,
        title: title.into(),
        location: Some(location.into()),
        verified: true,
        feature_gap: false,
        from_verify: false,
    }
}

fn defects(findings: Vec<Finding>) -> ReviewVerdict {
    let clean = !findings.iter().any(|f| f.severity.is_blocking());
    ReviewVerdict {
        complete: clean,
        coverage: Vec::new(),
        findings,
        residual: None,
        stated_complete: Some(clean),
        unparsed_coverage_rows: 0,
    }
}

fn spec_verdict(rows: &[(CoverageState, &str)], findings: Vec<Finding>) -> ReviewVerdict {
    let complete = rows
        .iter()
        .all(|(state, _)| *state == CoverageState::Implemented);
    ReviewVerdict {
        complete,
        coverage: rows
            .iter()
            .map(|(state, item)| CoverageRow {
                state: *state,
                item: (*item).into(),
                evidence: None,
            })
            .collect(),
        findings,
        residual: None,
        stated_complete: Some(complete),
        unparsed_coverage_rows: 0,
    }
}

/// A completed turn that read `read` of the `scope` files.
fn audited(verdict: Option<ReviewVerdict>, read: &[&str], scope: &[&str]) -> ReviewTurnFacts {
    ReviewTurnFacts {
        stop_reason: TurnStopReason::Completed,
        verdict,
        read_files: read.iter().map(|s| s.to_string()).collect(),
        scope_files: Some(scope.iter().map(|s| s.to_string()).collect()),
        ..ReviewTurnFacts::default()
    }
}

fn fixed(changed: &[&str]) -> ReviewTurnFacts {
    ReviewTurnFacts {
        stop_reason: TurnStopReason::Completed,
        changed_files: changed.iter().map(|s| s.to_string()).collect(),
        ..ReviewTurnFacts::default()
    }
}

fn git_inputs(files: &[&str]) -> ReviewInputs {
    ReviewInputs {
        git: Some(GitScope {
            source: GitSource::Uncommitted,
            files: files.iter().map(|s| s.to_string()).collect(),
            numstat: Vec::new(),
            groups: Vec::new(),
        }),
        ..ReviewInputs::default()
    }
}

fn plan_inputs() -> ReviewInputs {
    ReviewInputs {
        files: vec![ReviewInput {
            path: "plan.md".into(),
            kind: InputKind::Plan,
        }],
        ..ReviewInputs::default()
    }
}

fn prompt_of(step: &ReviewStep) -> &str {
    match step {
        ReviewStep::RunPrompt(prompt) => prompt,
        other => panic!("expected another turn, got {other:?}"),
    }
}

#[test]
fn defects_only_audit_asks_for_unread_files_then_reports_partial() {
    let scope = ["src/a.rs", "src/b.rs", "src/c.rs", "src/d.rs"];
    let mut drive = ReviewDrive::start(git_inputs(&scope), true, 3);
    assert!(
        drive.pending_seed,
        "the first audit turn runs the seed command"
    );

    let step = drive.next_step(&audited(
        Some(defects(vec![])),
        &["src/a.rs", "src/b.rs"],
        &scope,
    ));
    let prompt = prompt_of(&step);
    assert!(
        prompt.starts_with("[hi:review] Defect audit, continued (1/2). The last turn left 2 of the files under audit unread"),
        "{prompt}"
    );
    assert!(
        prompt.contains("Unread files (read every one):\n- src/c.rs\n- src/d.rs\n"),
        "{prompt}"
    );
    assert!(
        !prompt.contains("src/a.rs"),
        "read files are not re-listed: {prompt}"
    );
    assert!(prompt.contains(REVIEW_DEFECTS_FORMAT_BLOCK));
    assert_eq!(
        crate::review::transcript_label(prompt).as_deref(),
        Some("/review · defect audit, continued")
    );
    assert_eq!(drive.phase, ReviewPhase::Audit);
    assert_eq!(drive.continuations, 1);
    assert_eq!(drive.scope_count, 4);
    assert_eq!(
        drive.read_files,
        vec!["src/a.rs".to_string(), "src/b.rs".to_string()]
    );
    assert!(
        drive.status_line().contains("· continued 1/2 ·"),
        "{}",
        drive.status_line()
    );
    assert_eq!(drive.last_verdict, None, "the audit is not over");

    // The continuation reads one more file and finds a P2; the last one is
    // asked for once more, with the finding carried along.
    let p2 = finding(Severity::P2, "Log unknown commands", "src/c.rs:9");
    let step = drive.next_step(&audited(
        Some(defects(vec![p2.clone()])),
        &["src/c.rs"],
        &scope,
    ));
    let prompt = prompt_of(&step);
    assert!(
        prompt.starts_with("[hi:review] Defect audit, continued (2/2)."),
        "{prompt}"
    );
    assert!(
        prompt.contains("Unread files (read every one):\n- src/d.rs\n"),
        "{prompt}"
    );
    assert!(!prompt.contains("- src/c.rs\n"), "{prompt}");
    assert!(
        prompt.contains("Findings already recorded from the files read so far")
            && prompt.contains("[P2] Log unknown commands — src/c.rs:9"),
        "{prompt}"
    );
    assert_eq!(drive.continuations, 2);
    assert_eq!(
        drive.scope_count, 4,
        "a continuation does not recount the scope"
    );

    // Still unread after the last continuation: partial, and the merged
    // verdict keeps the P2.
    let step = drive.next_step(&audited(Some(defects(vec![])), &[], &scope));
    match &step {
        ReviewStep::Done(text) => assert_eq!(
            text,
            "spec review partial: 4 uncommitted files (git status) audited, no plan/spec items · no P0/P1 defects · partial audit: read 3/4 files in scope; 1 unread · 1 P2/P3 reported"
        ),
        other => panic!("expected Done, got {other:?}"),
    }
    assert_eq!(drive.phase, ReviewPhase::Done);
    assert_eq!(drive.continuations, 0);
    assert_eq!(drive.unread_files, vec!["src/d.rs".to_string()]);
    assert_eq!(drive.unread_count, 1);
    assert!(drive.is_partial());
    assert!(
        !drive.coverage_complete(),
        "partial is not complete: exit 3, not 0"
    );
    assert_eq!(
        drive.read_coverage().as_deref(),
        Some("read 3/4 files in scope")
    );
    assert_eq!(drive.last_verdict.as_ref().unwrap().findings, vec![p2]);
    let lines = drive.report_lines();
    assert!(
        lines.contains(
            &"read 3/4 files in scope; unread after 2 continuation turn(s) per chunk: src/d.rs"
                .to_string()
        ),
        "{lines:?}"
    );
    assert!(
        lines
            .last()
            .unwrap()
            .starts_with("next: the audit is partial (1 file(s) in scope were never read)"),
        "{lines:?}"
    );
}

#[test]
fn fully_read_scope_is_complete_and_the_report_says_so() {
    let scope = ["src/a.rs", "src/b.rs"];
    let mut drive = ReviewDrive::start(git_inputs(&scope), true, 3);
    let step = drive.next_step(&audited(
        Some(defects(vec![])),
        &["src/b.rs", "src/a.rs"],
        &scope,
    ));
    match &step {
        ReviewStep::Done(text) => assert_eq!(
            text,
            "spec review complete: 2 uncommitted files (git status) audited, no plan/spec items · no P0/P1 defects · read 2/2 files in scope"
        ),
        other => panic!("expected Done, got {other:?}"),
    }
    assert!(!drive.is_partial());
    assert!(drive.coverage_complete());
    assert!(
        drive
            .report_lines()
            .contains(&"read 2/2 files in scope".to_string())
    );
    assert!(drive.unread_files.is_empty());
}

#[test]
fn a_spec_audit_reports_unread_files_without_asking_again() {
    let mut inputs = plan_inputs();
    inputs.scope = vec!["src".into()];
    let mut drive = ReviewDrive::start(inputs, true, 3);
    let verdict = spec_verdict(&[(CoverageState::Implemented, "Welcome")], vec![]);
    let step = drive.next_step(&audited(
        Some(verdict),
        &["src/a.rs"],
        &["src/a.rs", "src/b.rs"],
    ));
    match &step {
        ReviewStep::Done(text) => assert!(
            text.starts_with("spec review complete: all 1 plan/spec items implemented · no P0/P1 defects · read 1/2 files in scope"),
            "{text}"
        ),
        other => panic!("a spec audit reads what grounds its items; got {other:?}"),
    }
    assert!(
        !drive.is_partial(),
        "unread files do not make a spec audit partial"
    );
    assert!(drive.coverage_complete());
    assert_eq!(drive.unread_count, 1, "but they are counted");
    assert!(
        drive
            .report_lines()
            .contains(&"read 1/2 files in scope; unread: src/b.rs".to_string()),
        "{:?}",
        drive.report_lines()
    );
    assert_eq!(drive.next_action(), None);
}

#[test]
fn chunked_defects_audit_sums_the_scope_over_its_chunks() {
    let inputs = ReviewInputs {
        chunks: vec!["crates/a".into(), "crates/b".into()],
        ..ReviewInputs::default()
    };
    let mut drive = ReviewDrive::start(inputs, true, 3);
    let first = ["crates/a/src/lib.rs", "crates/a/src/main.rs"];
    let step = drive.next_step(&audited(Some(defects(vec![])), &first, &first));
    assert!(prompt_of(&step).contains("chunk 2/2: crates/b"));
    assert_eq!(drive.scope_count, 2);
    let second = ["crates/b/src/lib.rs"];
    let step = drive.next_step(&audited(Some(defects(vec![])), &second, &second));
    assert!(
        matches!(&step, ReviewStep::Done(text) if text.contains("read 3/3 files in scope")),
        "{step:?}"
    );
    assert_eq!(drive.scope_count, 3);
    assert_eq!(drive.read_files.len(), 3);
}

#[test]
fn an_unbounded_scope_leaves_read_coverage_untracked() {
    let mut drive = ReviewDrive::start(plan_inputs(), true, 3);
    let verdict = spec_verdict(&[(CoverageState::Implemented, "Welcome")], vec![]);
    let facts = ReviewTurnFacts {
        stop_reason: TurnStopReason::Completed,
        verdict: Some(verdict),
        read_files: vec!["src/a.rs".into()],
        scope_files: None,
        ..ReviewTurnFacts::default()
    };
    match drive.next_step(&facts) {
        ReviewStep::Done(text) => assert!(!text.contains("files in scope"), "{text}"),
        other => panic!("expected Done, got {other:?}"),
    }
    assert_eq!(drive.scope_count, 0);
    assert_eq!(drive.read_coverage(), None);
    assert_eq!(
        drive.read_files,
        vec!["src/a.rs".to_string()],
        "still recorded for the report"
    );
}

fn failed_seed(dir: &std::path::Path) -> VerifySeed {
    std::fs::create_dir_all(dir.join("src")).unwrap();
    std::fs::write(dir.join("src/main.rs"), "fn main() {}\n").unwrap();
    VerifySeed::from_output(
        dir,
        "cargo test --offline",
        false,
        "test welcome ... FAILED\nthread 'welcome' panicked at src/main.rs:66:9:\nno Welcome\n[exit code 101]",
        "audit",
    )
}

#[test]
fn failing_seed_becomes_a_p1_when_the_model_reports_no_defect() {
    let dir = tempfile::tempdir().unwrap();
    let scope = ["src/main.rs"];
    let mut drive = ReviewDrive::start(git_inputs(&scope), false, 3);
    drive.verify = Some(failed_seed(dir.path()));
    drive.pending_seed = false;
    assert!(
        drive
            .resume_prompt()
            .unwrap()
            .contains("`cargo test --offline` FAILED. Output tail:"),
        "a resumed audit turn carries the seed result"
    );

    let step = drive.next_step(&audited(Some(defects(vec![])), &scope, &scope));
    let prompt = prompt_of(&step);
    assert!(prompt.contains("Fix pass 1/3"), "{prompt}");
    assert!(
        prompt.contains(
            "1. [P1] `cargo test --offline` fails: test welcome ... FAILED — src/main.rs:66"
        ),
        "{prompt}"
    );
    assert_eq!(drive.phase, ReviewPhase::Fix);
    assert!(drive.findings_to_fix[0].from_verify);
    assert!(!drive.pending_seed, "a fix turn is not seeded");

    // The fix pass changes the cited file; the re-audit is seeded again
    // (here: it passes) and the model reports nothing: done, clean.
    let step = drive.next_step(&fixed(&["src/main.rs"]));
    assert!(prompt_of(&step).contains("Re-audit after fix pass 1/3"));
    assert!(
        drive.pending_seed,
        "the re-audit runs the seed command first"
    );
    assert_eq!(drive.seed_label(), "re-audit 1");
    drive.verify = Some(VerifySeed::from_output(
        dir.path(),
        "cargo test --offline",
        true,
        "ok\n",
        "re-audit 1",
    ));
    drive.pending_seed = false;
    let step = drive.next_step(&audited(Some(defects(vec![])), &scope, &scope));
    match &step {
        ReviewStep::Done(text) => assert!(
            text.contains("no P0/P1 defects after 1 fix pass(es)") && !text.contains("seeded"),
            "{text}"
        ),
        other => panic!("expected Done, got {other:?}"),
    }
    assert!(
        drive
            .report_lines()
            .contains(&"verify: cargo test --offline · passed before re-audit 1".to_string()),
        "{:?}",
        drive.report_lines()
    );
    assert!(drive.uncited_changes.is_empty(), "src/main.rs was cited");
}

#[test]
fn failing_seed_is_reported_not_synthesized_when_plan_items_are_missing() {
    let dir = tempfile::tempdir().unwrap();
    let mut drive = ReviewDrive::start(plan_inputs(), false, 3);
    drive.verify = Some(failed_seed(dir.path()));
    let verdict = spec_verdict(
        &[
            (CoverageState::Implemented, "Welcome"),
            (CoverageState::Missing, "/topic"),
        ],
        vec![],
    );
    let step = drive.next_step(&ReviewTurnFacts {
        stop_reason: TurnStopReason::Completed,
        verdict: Some(verdict),
        ..ReviewTurnFacts::default()
    });
    match &step {
        ReviewStep::Done(text) => assert!(
            text.contains("1/2 plan/spec items implemented · no P0/P1 defects"),
            "the failing tests are presumably the missing item's: {text}"
        ),
        other => panic!("expected Done, got {other:?}"),
    }
    assert!(drive.open_blocking().is_empty());
    assert!(
        drive
            .report_lines()
            .contains(&"verify: cargo test --offline · failed before audit".to_string())
    );

    // Audit-only with nothing missing: the seeded P1 stays open (exit 4).
    let mut audit_only = ReviewDrive::start(git_inputs(&["src/main.rs"]), true, 3);
    audit_only.verify = Some(failed_seed(dir.path()));
    let step = audit_only.next_step(&audited(
        Some(defects(vec![])),
        &["src/main.rs"],
        &["src/main.rs"],
    ));
    match &step {
        ReviewStep::Done(text) => assert!(
            text.contains("1 P0/P1 defect(s) still open (audit only)")
                && text.ends_with("· 1 P1 seeded from the failing verify command"),
            "{text}"
        ),
        other => panic!("expected Done, got {other:?}"),
    }
    assert_eq!(audit_only.open_blocking().len(), 1);
    assert!(
        audit_only.report_lines().iter().any(|line| line.contains(
            "[P1] `cargo test --offline` fails: test welcome ... FAILED — src/main.rs:66 (seeded: the verify command fails and the model reported no defect)"
        )),
        "{:?}",
        audit_only.report_lines()
    );

    // A model that reports the cause itself gets no duplicate.
    let mut reported = ReviewDrive::start(git_inputs(&["src/main.rs"]), true, 3);
    reported.verify = Some(failed_seed(dir.path()));
    let p1 = finding(Severity::P1, "Welcome banner never sent", "src/main.rs:60");
    reported.next_step(&audited(
        Some(defects(vec![p1.clone()])),
        &["src/main.rs"],
        &["src/main.rs"],
    ));
    assert_eq!(reported.last_verdict.as_ref().unwrap().findings, vec![p1]);
}

#[test]
fn fix_pass_changes_no_finding_cited_are_flagged_for_the_reaudit() {
    let mut drive = ReviewDrive::start(plan_inputs(), false, 3);
    let p0 = finding(Severity::P0, "Reject empty nicknames", "src/server.rs:88");
    let verdict = spec_verdict(&[(CoverageState::Implemented, "Welcome")], vec![p0]);
    drive.next_step(&ReviewTurnFacts {
        stop_reason: TurnStopReason::Completed,
        verdict: Some(verdict),
        ..ReviewTurnFacts::default()
    });
    let step = drive.next_step(&fixed(&[
        "src/server.rs",
        "src/unrelated.rs",
        "tests/server_test.rs",
        "Cargo.lock",
    ]));
    let prompt = prompt_of(&step);
    assert!(
        prompt.contains("Of those, files no finding cited")
            && prompt.contains("- src/unrelated.rs\n"),
        "{prompt}"
    );
    assert!(
        !prompt
            .split("Of those, files no finding cited")
            .nth(1)
            .unwrap()
            .split("Prior findings")
            .next()
            .unwrap()
            .contains("tests/server_test.rs"),
        "a regression test is an expected side change: {prompt}"
    );
    assert_eq!(drive.uncited_changes, vec!["src/unrelated.rs".to_string()]);
    let verdict = spec_verdict(&[(CoverageState::Implemented, "Welcome")], vec![]);
    drive.next_step(&ReviewTurnFacts {
        stop_reason: TurnStopReason::Completed,
        verdict: Some(verdict),
        ..ReviewTurnFacts::default()
    });
    assert!(
        drive.report_lines().contains(
            &"warning: a fix pass changed files no finding cited (the re-audit was pointed at them): src/unrelated.rs".to_string()
        ),
        "{:?}",
        drive.report_lines()
    );
}

#[test]
fn a_reworded_finding_at_the_same_place_is_no_progress() {
    let mut drive = ReviewDrive::start(plan_inputs(), false, 3);
    let p0 = finding(Severity::P0, "Reject empty nicknames", "src/server.rs:88");
    let verdict = spec_verdict(&[(CoverageState::Implemented, "Welcome")], vec![p0]);
    drive.next_step(&ReviewTurnFacts {
        stop_reason: TurnStopReason::Completed,
        verdict: Some(verdict),
        ..ReviewTurnFacts::default()
    });
    drive.next_step(&fixed(&["src/server.rs"]));
    let reworded = finding(
        Severity::P1,
        "Empty nickname accepted by NICK handler",
        "src/server.rs:92",
    );
    let verdict = spec_verdict(&[(CoverageState::Implemented, "Welcome")], vec![reworded]);
    let step = drive.next_step(&ReviewTurnFacts {
        stop_reason: TurnStopReason::Completed,
        verdict: Some(verdict),
        ..ReviewTurnFacts::default()
    });
    match step {
        ReviewStep::Stopped(text) => assert!(
            text.contains("no progress: the same 1 P0/P1 finding(s) remain after fix pass 1"),
            "{text}"
        ),
        other => panic!("expected Stopped, got {other:?}"),
    }

    // A different defect in the same file is progress.
    let mut moved_on = ReviewDrive::start(plan_inputs(), false, 3);
    let p0 = finding(Severity::P0, "Reject empty nicknames", "src/server.rs:88");
    moved_on.next_step(&ReviewTurnFacts {
        stop_reason: TurnStopReason::Completed,
        verdict: Some(spec_verdict(
            &[(CoverageState::Implemented, "Welcome")],
            vec![p0],
        )),
        ..ReviewTurnFacts::default()
    });
    moved_on.next_step(&fixed(&["src/server.rs"]));
    let other = finding(
        Severity::P1,
        "Drop kicked members from the channel list",
        "src/server.rs:300",
    );
    let step = moved_on.next_step(&ReviewTurnFacts {
        stop_reason: TurnStopReason::Completed,
        verdict: Some(spec_verdict(
            &[(CoverageState::Implemented, "Welcome")],
            vec![other],
        )),
        ..ReviewTurnFacts::default()
    });
    assert!(prompt_of(&step).contains("Fix pass 2/3"));
}

#[test]
fn a_stop_between_audit_turns_keeps_what_they_found() {
    let scope = ["src/a.rs", "src/b.rs"];
    let p0 = finding(Severity::P0, "Reject empty nicknames", "src/a.rs:9");
    let mut drive = ReviewDrive::start(git_inputs(&scope), false, 3);
    let step = drive.next_step(&audited(
        Some(defects(vec![p0.clone()])),
        &["src/a.rs"],
        &scope,
    ));
    assert!(prompt_of(&step).contains("continued (1/2)"));
    let step = drive.next_step(&ReviewTurnFacts {
        stop_reason: TurnStopReason::Error,
        error: Some("provider overloaded".into()),
        ..ReviewTurnFacts::default()
    });
    assert!(matches!(step, ReviewStep::Stopped(_)), "{step:?}");
    assert_eq!(drive.open_blocking(), vec![p0.clone()]);
    assert!(
        drive
            .report_lines()
            .iter()
            .any(|line| line.contains("Reject empty nicknames")),
        "{:?}",
        drive.report_lines()
    );

    // `/review stop` on a paused continuation keeps them too.
    let mut paused = ReviewDrive::start(git_inputs(&scope), false, 3);
    paused.next_step(&audited(
        Some(defects(vec![p0.clone()])),
        &["src/a.rs"],
        &scope,
    ));
    paused.next_step(&ReviewTurnFacts {
        stop_reason: TurnStopReason::Cancelled,
        ..ReviewTurnFacts::default()
    });
    assert!(paused.paused);
    paused.stop("stopped by /review stop");
    assert_eq!(paused.open_blocking(), vec![p0]);
}

#[test]
fn a_seed_failure_that_names_no_workspace_file_is_reported_not_seeded() {
    let dir = tempfile::tempdir().unwrap();
    let scope = ["src/main.rs"];
    let mut drive = ReviewDrive::start(git_inputs(&scope), false, 3);
    drive.verify = Some(VerifySeed::from_output(
        dir.path(),
        "cargo check --workspace --all-targets",
        false,
        "error: failed to download from `https://index.crates.io/config.json`\n[exit code 101]",
        "audit",
    ));
    let step = drive.next_step(&audited(Some(defects(vec![])), &scope, &scope));
    assert!(matches!(step, ReviewStep::Done(_)), "{step:?}");
    assert!(drive.open_blocking().is_empty());
    assert!(
        drive.report_lines().contains(
            &"verify: cargo check --workspace --all-targets · failed before audit".to_string()
        ),
        "{:?}",
        drive.report_lines()
    );
}
