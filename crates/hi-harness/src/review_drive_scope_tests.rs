//! The drive on a narrowed audit: a git-scoped defects-only run, and the
//! chunked whole-repo run (`all`) that folds one verdict per chunk.

use super::*;
use crate::review::{CoverageRow, CoverageState, GitScope, GitSource, Severity};
use crate::session::JsonlSession;

fn finding(severity: Severity, title: &str, location: &str) -> Finding {
    Finding {
        severity,
        title: title.into(),
        location: Some(location.into()),
        verified: true,
        feature_gap: false,
    }
}

/// A defects-only verdict: no coverage rows, the word says P0/P1 or not.
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

fn completed(verdict: Option<ReviewVerdict>, unverified: &[&str]) -> ReviewTurnFacts {
    ReviewTurnFacts {
        stop_reason: TurnStopReason::Completed,
        error: None,
        changed_files: Vec::new(),
        verdict,
        unverified_citations: unverified.iter().map(|s| s.to_string()).collect(),
    }
}

fn git_inputs() -> ReviewInputs {
    ReviewInputs {
        git: Some(GitScope {
            source: GitSource::Uncommitted,
            files: vec!["src/a.rs".into(), "src/b.rs".into()],
        }),
        notice: Some("no plan.md or spec.md found; auditing recent work for defects: 2 uncommitted files (git status)".into()),
        ..ReviewInputs::default()
    }
}

fn chunked_inputs() -> ReviewInputs {
    ReviewInputs {
        chunks: vec!["crates/a".into(), "crates/b".into(), ".".into()],
        ..ReviewInputs::default()
    }
}

#[test]
fn git_scoped_audit_reports_defects_with_no_coverage_and_points_wider() {
    let mut drive = ReviewDrive::start(git_inputs(), true, 3);
    assert_eq!(
        drive.status_line(),
        "spec review · audit only · 2 uncommitted files (git status) · no plan/spec (defects only)"
    );
    assert!(
        drive
            .resume_prompt()
            .unwrap()
            .contains("Code under audit: the 2 uncommitted file(s)")
    );
    let p2 = finding(Severity::P2, "Log unknown commands", "src/a.rs:9");
    let step = drive.next_step(&completed(Some(defects(vec![p2])), &["src/gone.rs"]));
    match step {
        ReviewStep::Done(text) => assert_eq!(
            text,
            "spec review complete: 2 uncommitted files (git status) audited, no plan/spec items · no P0/P1 defects · 1 P2/P3 reported"
        ),
        other => panic!("expected Done, got {other:?}"),
    }
    assert!(
        drive.coverage_complete(),
        "no plan/spec: nothing to cover, so coverage is not what blocks exit 0"
    );
    let lines = drive.report_lines();
    assert_eq!(
        lines[0],
        "coverage: none (no plan/spec items; write plan.md for a coverage audit)"
    );
    assert_eq!(lines[1], "findings:");
    assert_eq!(
        lines.last().map(String::as_str),
        Some(
            "next: this audited the uncommitted changes only; write plan.md (a `- [ ]` checklist of what should exist) or pass one for a coverage audit: `/review docs/spec.md [dir]`; `/review audit all` audits the whole repo in chunks"
        )
    );
    assert_eq!(drive.unverified_citations, vec!["src/gone.rs".to_string()]);
    assert_eq!(
        drive.status_line(),
        "spec review · done · 2 uncommitted files (git status) · no plan/spec (defects only)"
    );

    let mut last_commit = ReviewDrive::start(
        ReviewInputs {
            git: Some(GitScope {
                source: GitSource::LastCommit {
                    short_hash: "a1b2c3d".into(),
                    subject: "fix".into(),
                },
                files: vec!["src/a.rs".into()],
            }),
            ..ReviewInputs::default()
        },
        false,
        3,
    );
    assert!(
        last_commit
            .status_line()
            .contains("· last commit a1b2c3d (1 file) ·"),
        "{}",
        last_commit.status_line()
    );
    last_commit.next_step(&completed(Some(defects(vec![])), &[]));
    assert!(
        last_commit
            .next_action()
            .unwrap()
            .starts_with("next: this audited the last commit only;")
    );
}

#[test]
fn chunked_audit_runs_one_turn_per_chunk_then_merges_and_fixes() {
    let mut drive = ReviewDrive::start(chunked_inputs(), false, 3);
    assert_eq!(
        drive.status_line(),
        "spec review · audit · up to 3 fix passes · chunk 1/3 · crates/a · no plan/spec (defects only)"
    );
    assert!(
        drive
            .resume_prompt()
            .unwrap()
            .contains("chunk 1/3: crates/a")
    );

    let p0 = finding(
        Severity::P0,
        "Reject empty nicknames",
        "crates/a/src/lib.rs:9",
    );
    let p2 = finding(
        Severity::P2,
        "Log unknown commands",
        "crates/a/src/lib.rs:20",
    );
    let step = drive.next_step(&completed(
        Some(defects(vec![p0.clone(), p2.clone()])),
        &["crates/a/src/old.rs"],
    ));
    match &step {
        ReviewStep::RunPrompt(prompt) => {
            assert!(
                prompt.contains("Defect audit, chunk 2/3: crates/b"),
                "{prompt}"
            );
            assert!(prompt.contains("Scope: chunk 2/3 of the workspace: `crates/b/`"));
        }
        other => panic!("the next chunk runs, got {other:?}"),
    }
    assert_eq!(drive.phase, ReviewPhase::Audit, "still auditing");
    assert_eq!(drive.chunk, 1);
    assert_eq!(
        drive.last_verdict, None,
        "no verdict until every chunk is in"
    );
    assert_eq!(drive.chunk_verdict.as_ref().unwrap().findings.len(), 2);
    assert_eq!(
        drive.status_line(),
        "spec review · audit · up to 3 fix passes · chunk 2/3 · crates/b · no plan/spec (defects only)"
    );
    assert_eq!(drive.pass, 0);

    // A chunk with no block gets the defects-only re-ask, then counts.
    let step = drive.next_step(&completed(None, &[]));
    match &step {
        ReviewStep::RunPrompt(prompt) => {
            assert!(prompt.contains("Format re-ask"), "{prompt}");
            assert!(
                prompt.contains("no `coverage:` rows (there are no plan/spec items)"),
                "{prompt}"
            );
        }
        other => panic!("re-ask expected, got {other:?}"),
    }
    assert_eq!(drive.chunk, 1, "a re-ask is the same chunk");
    let p1 = finding(
        Severity::P1,
        "Drop kicked members",
        "crates/b/src/lib.rs:30",
    );
    let step = drive.next_step(&completed(
        Some(defects(vec![p0.clone(), p1.clone()])),
        &["crates/b/src/gone.rs"],
    ));
    match &step {
        ReviewStep::RunPrompt(prompt) => {
            assert!(prompt.contains("chunk 3/3: top-level files"), "{prompt}");
        }
        other => panic!("the last chunk runs, got {other:?}"),
    }
    assert_eq!(drive.chunk, 2);
    assert!(!drive.format_retry_used, "reset per chunk");
    assert_eq!(
        drive.status_line(),
        "spec review · audit · up to 3 fix passes · chunk 3/3 · top-level files · no plan/spec (defects only)"
    );

    // Last chunk in: the merged verdict has three distinct findings and the
    // two P0/P1 go to a fix pass.
    let step = drive.next_step(&completed(Some(defects(vec![p2.clone()])), &[]));
    match &step {
        ReviewStep::RunPrompt(prompt) => assert!(prompt.contains("Fix pass 1/3"), "{prompt}"),
        other => panic!("fix pass expected, got {other:?}"),
    }
    assert_eq!(drive.phase, ReviewPhase::Fix);
    assert_eq!(drive.chunk, 3, "past the last chunk");
    assert_eq!(drive.chunk_verdict, None);
    let merged = drive.last_verdict.as_ref().unwrap();
    assert_eq!(merged.findings.len(), 3, "{:?}", merged.findings);
    assert_eq!(drive.findings_to_fix, vec![p0.clone(), p1.clone()]);
    assert_eq!(
        drive.unverified_citations,
        vec![
            "crates/a/src/old.rs".to_string(),
            "crates/b/src/gone.rs".to_string()
        ],
        "citations accumulate across chunks"
    );
    assert_eq!(
        drive.status_line(),
        "spec review · fix pass 1/3 · 2 P0/P1 · 3 chunks · no plan/spec (defects only)"
    );

    // The re-audit is scoped by the fix pass, not the chunks.
    let step = drive.next_step(&ReviewTurnFacts {
        stop_reason: TurnStopReason::Completed,
        error: None,
        changed_files: vec!["crates/a/src/lib.rs".into()],
        verdict: None,
        unverified_citations: Vec::new(),
    });
    match &step {
        ReviewStep::RunPrompt(prompt) => {
            assert!(prompt.contains("Re-audit after fix pass 1/3"), "{prompt}");
            assert!(!prompt.contains("Scope: chunk"), "{prompt}");
            assert!(prompt.contains("- crates/a/src/lib.rs\n"), "{prompt}");
        }
        other => panic!("re-audit expected, got {other:?}"),
    }
    assert_eq!(drive.phase, ReviewPhase::Reaudit);
    let step = drive.next_step(&completed(Some(defects(vec![p2])), &[]));
    match step {
        ReviewStep::Done(text) => assert_eq!(
            text,
            "spec review complete: 3 chunks audited, no plan/spec items · no P0/P1 defects after 1 fix pass(es) · 1 P2/P3 reported"
        ),
        other => panic!("expected Done, got {other:?}"),
    }
    assert_eq!(drive.chunk, 3, "a re-audit does not touch the chunk cursor");
    assert!(
        drive
            .next_action()
            .unwrap()
            .starts_with("next: this audited the whole repo for defects;")
    );
}

#[test]
fn chunked_audit_with_a_spec_merges_coverage_by_best_state() {
    let mut inputs = chunked_inputs();
    inputs.files.push(crate::review::ReviewInput {
        path: "plan.md".into(),
        kind: crate::review::InputKind::Plan,
    });
    inputs.chunks.truncate(2);
    let mut drive = ReviewDrive::start(inputs, true, 3);
    let row = |state: CoverageState, item: &str, evidence: Option<&str>| CoverageRow {
        state,
        item: item.into(),
        evidence: evidence.map(str::to_string),
    };
    let first = ReviewVerdict {
        complete: false,
        coverage: vec![
            row(
                CoverageState::Implemented,
                "Welcome",
                Some("crates/a/src/lib.rs:1"),
            ),
            row(CoverageState::Missing, "/topic", None),
        ],
        findings: vec![],
        residual: None,
        stated_complete: Some(false),
        unparsed_coverage_rows: 0,
    };
    assert!(matches!(
        drive.next_step(&completed(Some(first), &[])),
        ReviewStep::RunPrompt(_)
    ));
    let second = ReviewVerdict {
        complete: false,
        coverage: vec![
            row(CoverageState::Missing, "Welcome", None),
            row(
                CoverageState::Implemented,
                "/topic",
                Some("crates/b/src/lib.rs:7"),
            ),
        ],
        findings: vec![],
        residual: None,
        stated_complete: Some(false),
        unparsed_coverage_rows: 0,
    };
    match drive.next_step(&completed(Some(second), &[])) {
        ReviewStep::Done(text) => {
            assert!(text.contains("all 2 plan/spec items implemented"), "{text}")
        }
        other => panic!("audit-only finishes after the last chunk, got {other:?}"),
    }
    let verdict = drive.last_verdict.as_ref().unwrap();
    assert!(verdict.complete, "each item was implemented in some chunk");
    assert_eq!(
        verdict.coverage[1].evidence.as_deref(),
        Some("crates/b/src/lib.rs:7")
    );
    assert!(drive.coverage_complete());
    assert_eq!(drive.next_action(), None);
    assert!(
        drive.report_lines().iter().any(|line| line
            .starts_with("note: the model wrote INCOMPLETE with every coverage row implemented")),
        "{:?}",
        drive.report_lines()
    );
}

#[test]
fn a_spec_with_no_coverage_rows_is_not_complete_coverage() {
    let mut inputs = ReviewInputs::default();
    inputs.files.push(crate::review::ReviewInput {
        path: "plan.md".into(),
        kind: crate::review::InputKind::Plan,
    });
    let mut drive = ReviewDrive::start(inputs, true, 3);
    drive.next_step(&completed(Some(defects(vec![])), &[]));
    assert!(
        !drive.coverage_complete(),
        "the model skipped the plan's items"
    );
    assert_eq!(
        drive.report_lines()[0],
        "coverage: no rows reported for plan.md (the model skipped its items)"
    );
}

#[test]
fn chunk_progress_round_trips_through_the_session_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("s.jsonl");
    let mut session = JsonlSession::create(&path).unwrap();
    let mut drive = ReviewDrive::start(chunked_inputs(), false, 3);
    let p0 = finding(
        Severity::P0,
        "Reject empty nicknames",
        "crates/a/src/lib.rs:9",
    );
    drive.next_step(&completed(Some(defects(vec![p0])), &[]));
    assert_eq!(drive.chunk, 1);
    drive.paused = true;
    session.record_review_drive(&drive).unwrap();
    let loaded = JsonlSession::load(&path).unwrap().review_drive;
    assert_eq!(loaded, drive);
    assert_eq!(loaded.inputs.chunks.len(), 3);
    assert!(
        loaded
            .resume_prompt()
            .unwrap()
            .contains("chunk 2/3: crates/b"),
        "a resumed chunked audit continues with the next chunk"
    );

    let mut git = ReviewDrive::start(git_inputs(), true, 3);
    git.paused = true;
    session.record_review_drive(&git).unwrap();
    assert_eq!(JsonlSession::load(&path).unwrap().review_drive, git);
}
