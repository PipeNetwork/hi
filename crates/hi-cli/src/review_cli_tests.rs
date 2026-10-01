use super::*;
use crate::config::Cli;
use clap::Parser;
use hi_harness::{
    CoverageRow, CoverageState, Finding, InputKind, ReviewInput, ReviewInputs, ReviewTurnFacts,
    ReviewVerdict, Severity, TurnStopReason,
};

fn inputs() -> ReviewInputs {
    ReviewInputs {
        files: vec![ReviewInput {
            path: "plan.md".into(),
            kind: InputKind::Plan,
        }],
        ..ReviewInputs::default()
    }
}

fn verdict(complete: bool, findings: Vec<Finding>) -> ReviewVerdict {
    ReviewVerdict {
        complete,
        coverage: vec![CoverageRow {
            state: if complete {
                CoverageState::Implemented
            } else {
                CoverageState::Missing
            },
            item: "/topic".into(),
            evidence: None,
        }],
        findings,
        residual: None,
        stated_complete: Some(complete),
        unparsed_coverage_rows: 0,
    }
}

fn completed(verdict: Option<ReviewVerdict>) -> ReviewTurnFacts {
    ReviewTurnFacts {
        stop_reason: TurnStopReason::Completed,
        verdict,
        ..ReviewTurnFacts::default()
    }
}

fn p0() -> Finding {
    Finding {
        severity: Severity::P0,
        title: "Reject empty nicknames".into(),
        location: Some("src/server.rs:88".into()),
        verified: true,
        feature_gap: false,
        from_verify: false,
    }
}

fn git_inputs(files: &[&str]) -> ReviewInputs {
    ReviewInputs {
        git: Some(hi_harness::GitScope {
            source: hi_harness::GitSource::Uncommitted,
            files: files.iter().map(|f| f.to_string()).collect(),
            numstat: Vec::new(),
            groups: Vec::new(),
        }),
        ..ReviewInputs::default()
    }
}

#[test]
fn clap_parses_spec_review_flags() {
    let cli = Cli::try_parse_from(["hi", "--spec-review"]).unwrap();
    assert_eq!(cli.spec_review, Some(Vec::new()));
    assert_eq!(cli.spec_review_passes, None);
    assert!(!cli.spec_review_audit_only);

    let cli = Cli::try_parse_from([
        "hi",
        "--spec-review",
        "docs/spec.md",
        "plan.md",
        "--spec-review-passes",
        "2",
        "--spec-review-audit-only",
        "--report",
        "r.json",
    ])
    .unwrap();
    assert_eq!(
        cli.spec_review,
        Some(vec!["docs/spec.md".into(), "plan.md".into()])
    );
    assert_eq!(cli.spec_review_passes, Some(2));
    assert!(cli.spec_review_audit_only);
    assert_eq!(cli.report.as_deref(), Some(std::path::Path::new("r.json")));

    // The independent-review policy keeps its flag; both can appear.
    let cli = Cli::try_parse_from(["hi", "--spec-review", "--review", "off"]).unwrap();
    assert_eq!(cli.spec_review, Some(Vec::new()));
    assert!(cli.review.is_some());
    assert!(cli.spec_review_passes.is_none());
}

#[test]
fn clap_rejects_spec_review_with_a_prompt_or_orphan_knobs() {
    assert!(Cli::try_parse_from(["hi", "fix the tests", "--spec-review"]).is_err());
    assert!(Cli::try_parse_from(["hi", "--spec-review-passes", "2"]).is_err());
    assert!(Cli::try_parse_from(["hi", "--spec-review-audit-only"]).is_err());
    assert!(Cli::try_parse_from(["hi", "--spec-review", "--spec-review-passes", "x"]).is_err());
}

#[test]
fn exit_codes_follow_the_documented_table() {
    let mut clean = ReviewDrive::start(inputs(), false, 3);
    clean.next_step(&completed(Some(verdict(true, vec![]))));
    assert_eq!(exit_code(&clean), EXIT_COMPLETE);

    let mut incomplete = ReviewDrive::start(inputs(), false, 3);
    incomplete.next_step(&completed(Some(verdict(false, vec![]))));
    assert_eq!(exit_code(&incomplete), EXIT_INCOMPLETE_COVERAGE);

    let mut open = ReviewDrive::start(inputs(), true, 3);
    open.next_step(&completed(Some(verdict(false, vec![p0()]))));
    assert_eq!(
        exit_code(&open),
        EXIT_OPEN_DEFECTS,
        "open P0 outranks coverage"
    );

    let mut stopped = ReviewDrive::start(inputs(), false, 3);
    stopped.next_step(&completed(None));
    stopped.next_step(&completed(None));
    assert_eq!(exit_code(&stopped), EXIT_STOPPED);

    let running = ReviewDrive::start(inputs(), false, 3);
    assert_eq!(exit_code(&running), EXIT_HARNESS_ERROR);
    assert_eq!(exit_code(&ReviewDrive::default()), EXIT_HARNESS_ERROR);
}

#[test]
fn review_run_into_result_maps_codes_to_review_exit() {
    let mut clean = ReviewDrive::start(inputs(), false, 3);
    clean.next_step(&completed(Some(verdict(true, vec![]))));
    let run = ReviewRun {
        drive: clean,
        last_outcome: None,
        turns: 1,
    };
    assert!(run.into_result().is_ok());

    let mut incomplete = ReviewDrive::start(inputs(), false, 3);
    incomplete.next_step(&completed(Some(verdict(false, vec![]))));
    let run = ReviewRun {
        drive: incomplete,
        last_outcome: None,
        turns: 1,
    };
    let error = run.into_result().unwrap_err();
    let exit = error.downcast_ref::<ReviewExit>().expect("ReviewExit");
    assert_eq!(exit.code, EXIT_INCOMPLETE_COVERAGE);
    assert!(
        exit.summary.contains("0/1 plan/spec items implemented"),
        "{exit}"
    );
    assert!(exit.to_string().ends_with("(exit 3)"));
}

#[test]
fn report_object_carries_inputs_verdict_findings_and_exit_code() {
    let mut drive = ReviewDrive::start(inputs(), true, 3);
    drive.next_step(&completed(Some(verdict(false, vec![p0()]))));
    drive.unverified_citations = vec!["src/gone.rs:1".into()];
    let report = review_report_json(&drive);
    assert_eq!(report["inputs"]["files"][0]["path"], "plan.md");
    assert_eq!(report["inputs"]["files"][0]["kind"], "plan");
    assert_eq!(report["phase"], "done");
    assert_eq!(report["audit_only"], true);
    assert_eq!(report["passes"], 0);
    assert_eq!(report["max_passes"], 3);
    assert_eq!(report["verdict"], "incomplete");
    assert_eq!(report["coverage_complete"], false);
    assert_eq!(report["coverage"][0]["state"], "missing");
    assert_eq!(report["findings"][0]["severity"], "P0");
    assert_eq!(report["findings"][0]["title"], "Reject empty nicknames");
    assert_eq!(report["open_blocking"], 1);
    assert_eq!(report["unverified_citations"][0], "src/gone.rs:1");
    assert_eq!(report["exit_code"], EXIT_OPEN_DEFECTS);
    assert!(
        report["summary"]
            .as_str()
            .unwrap()
            .starts_with("spec review")
    );

    assert_eq!(
        report["read"]["scope_total"], 0,
        "untracked scope: nothing counted"
    );
    assert_eq!(report["read"]["partial"], false);
    assert!(report["verify"].is_null());
    assert_eq!(report["uncited_changes"].as_array().unwrap().len(), 0);

    let idle = review_report_json(&ReviewDrive::default());
    assert_eq!(idle["phase"], "idle");
    assert!(idle["verdict"].is_null());
    assert_eq!(idle["coverage"].as_array().unwrap().len(), 0);
    assert!(idle["inputs"]["git"].is_null());
    assert_eq!(idle["inputs"]["chunks"].as_array().unwrap().len(), 0);
}

/// A defects-only audit (recent git work, or `all` without a plan/spec) has
/// no coverage rows: a clean one exits 0, not 3, and the report names the
/// scope. `all` is a value of `--spec-review`, not a path.
#[test]
fn defects_only_scope_exits_clean_and_reports_its_scope() {
    let mut drive = ReviewDrive::start(git_inputs(&["src/a.rs"]), true, 3);
    let mut clean = verdict(true, vec![]);
    clean.coverage.clear();
    drive.next_step(&completed(Some(clean)));
    assert_eq!(exit_code(&drive), EXIT_COMPLETE);
    let report = review_report_json(&drive);
    assert_eq!(report["inputs"]["git"]["source"]["kind"], "uncommitted");
    assert_eq!(report["inputs"]["git"]["files"][0], "src/a.rs");
    assert_eq!(report["coverage_complete"], true);
    assert_eq!(report["coverage"].as_array().unwrap().len(), 0);
    assert!(
        report["summary"]
            .as_str()
            .unwrap()
            .contains("1 uncommitted file (git status)"),
        "{report}"
    );

    // With a real plan the same empty coverage is the model's omission.
    let mut skipped = ReviewDrive::start(inputs(), true, 3);
    let mut none = verdict(true, vec![]);
    none.coverage.clear();
    skipped.next_step(&completed(Some(none)));
    assert_eq!(exit_code(&skipped), EXIT_INCOMPLETE_COVERAGE);
    let run = ReviewRun {
        drive: skipped,
        last_outcome: None,
        turns: 1,
    };
    let error = run.into_result().unwrap_err();
    assert!(
        error
            .to_string()
            .starts_with("spec review: no coverage rows reported for plan.md"),
        "{error}"
    );

    let mut paths = vec!["all".to_string(), "docs/spec.md".to_string()];
    assert!(review_take_all(&mut paths));
    assert_eq!(paths, vec!["docs/spec.md".to_string()]);
    let cli = Cli::try_parse_from(["hi", "--spec-review", "all"]).unwrap();
    assert_eq!(cli.spec_review, Some(vec!["all".into()]));
}

/// A defects-only audit that never read part of its scope, even after its
/// continuation turns, is partial: exit 3 with the read counts in the
/// summary and the report, not a clean 0.
#[test]
fn partial_read_coverage_exits_incomplete_with_the_counts() {
    let scope = ["src/a.rs", "src/b.rs", "src/c.rs"];
    let mut drive = ReviewDrive::start(git_inputs(&scope), true, 3);
    let mut clean = verdict(true, vec![]);
    clean.coverage.clear();
    let turn = |read: &[&str]| ReviewTurnFacts {
        read_files: read.iter().map(|f| f.to_string()).collect(),
        scope_files: Some(scope.iter().map(|f| f.to_string()).collect()),
        ..completed(Some(clean.clone()))
    };
    // First turn reads one file: continuation. Two continuations later the
    // third file is still unread.
    assert!(matches!(
        drive.next_step(&turn(&["src/a.rs"])),
        ReviewStep::RunPrompt(_)
    ));
    assert!(matches!(
        drive.next_step(&turn(&["src/b.rs"])),
        ReviewStep::RunPrompt(_)
    ));
    let ReviewStep::Done(summary) = drive.next_step(&turn(&[])) else {
        panic!("continuations are bounded");
    };
    assert_eq!(
        summary,
        "spec review partial: 3 uncommitted files (git status) audited, no plan/spec items · no P0/P1 defects · partial audit: read 2/3 files in scope; 1 unread"
    );
    assert_eq!(exit_code(&drive), EXIT_INCOMPLETE_COVERAGE);
    let report = review_report_json(&drive);
    assert_eq!(report["read"]["scope_total"], 3);
    assert_eq!(report["read"]["unread_count"], 1);
    assert_eq!(report["read"]["unread"][0], "src/c.rs");
    assert_eq!(report["read"]["files"].as_array().unwrap().len(), 2);
    assert_eq!(report["read"]["partial"], true);
    assert_eq!(report["coverage_complete"], false);
    assert_eq!(report["exit_code"], EXIT_INCOMPLETE_COVERAGE);
    let run = ReviewRun {
        drive,
        last_outcome: None,
        turns: 3,
    };
    let error = run.into_result().unwrap_err();
    assert_eq!(
        error.to_string(),
        "spec review: partial audit: read 2/3 files in scope; 1 unread (exit 3)"
    );
}

/// A failing seed command with no finding from the model becomes a
/// synthetic P1: exit 4, flagged in the report.
#[test]
fn seeded_finding_from_a_failing_verify_exits_with_open_defects() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("src")).unwrap();
    std::fs::write(dir.path().join("src/a.rs"), "fn a() {}\n").unwrap();
    let mut drive = ReviewDrive::start(git_inputs(&["src/a.rs"]), true, 3);
    drive.verify = Some(hi_harness::VerifySeed::from_output(
        dir.path(),
        "cargo test",
        false,
        "error[E0308]: mismatched types\n --> src/a.rs:4:5\n",
        "audit",
    ));
    let mut clean = verdict(true, vec![]);
    clean.coverage.clear();
    drive.next_step(&completed(Some(clean)));
    assert_eq!(exit_code(&drive), EXIT_OPEN_DEFECTS);
    let report = review_report_json(&drive);
    assert_eq!(report["findings"][0]["severity"], "P1");
    assert_eq!(report["findings"][0]["from_verify"], true);
    assert_eq!(report["verify"]["command"], "cargo test");
    assert_eq!(report["verify"]["passed"], false);
    assert_eq!(report["open_blocking"], 1);
}
