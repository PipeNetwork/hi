use super::*;
use crate::completion::{Decision, Demand, Event, TurnFacts, decide};
use crate::review::{CoverageRow, CoverageState, InputKind, ReviewInput};
use crate::session::{JsonlSession, LoadedSession};

fn inputs() -> ReviewInputs {
    ReviewInputs {
        files: vec![
            ReviewInput {
                path: "plan.md".into(),
                kind: InputKind::Plan,
            },
            ReviewInput {
                path: "docs/spec.md".into(),
                kind: InputKind::Spec,
            },
        ],
        ..ReviewInputs::default()
    }
}

fn finding(severity: crate::review::Severity, title: &str, location: &str) -> Finding {
    Finding {
        severity,
        title: title.into(),
        location: Some(location.into()),
        verified: true,
        feature_gap: false,
    }
}

fn row(text: &str, checked: Option<bool>) -> ChecklistItem {
    ChecklistItem {
        text: text.into(),
        checked,
    }
}

fn verdict(complete: bool, findings: Vec<Finding>) -> ReviewVerdict {
    ReviewVerdict {
        complete,
        coverage: vec![
            CoverageRow {
                state: CoverageState::Implemented,
                item: "Welcome".into(),
                evidence: Some("src/main.rs:1".into()),
            },
            CoverageRow {
                state: if complete {
                    CoverageState::Implemented
                } else {
                    CoverageState::Missing
                },
                item: "/topic".into(),
                evidence: None,
            },
        ],
        findings,
        residual: None,
        stated_complete: Some(complete),
        unparsed_coverage_rows: 0,
    }
}

fn completed(verdict: Option<ReviewVerdict>, changed: &[&str]) -> ReviewTurnFacts {
    ReviewTurnFacts {
        stop_reason: TurnStopReason::Completed,
        error: None,
        changed_files: changed.iter().map(|s| s.to_string()).collect(),
        verdict,
        unverified_citations: Vec::new(),
    }
}

fn p0() -> Finding {
    finding(
        crate::review::Severity::P0,
        "Reject empty nicknames",
        "src/server.rs:88",
    )
}

fn p1() -> Finding {
    finding(
        crate::review::Severity::P1,
        "KICK requires operator",
        "src/commands.rs:40",
    )
}

fn p2() -> Finding {
    finding(
        crate::review::Severity::P2,
        "Log unknown commands",
        "src/commands.rs:9",
    )
}

#[test]
fn clean_audit_is_done_without_a_fix_pass() {
    let mut drive = ReviewDrive::start(inputs(), false, 3);
    assert!(drive.is_active());
    assert_eq!(
        drive.status_line(),
        "spec review · audit · up to 3 fix passes · plan.md + docs/spec.md"
    );
    let step = drive.next_step(&completed(Some(verdict(true, vec![p2()])), &[]));
    match step {
        ReviewStep::Done(text) => {
            assert!(text.contains("all 2 plan/spec items implemented"), "{text}");
            assert!(text.contains("no P0/P1 defects"), "{text}");
            assert!(text.contains("1 P2/P3 reported"), "{text}");
        }
        other => panic!("expected Done, got {other:?}"),
    }
    assert_eq!(drive.phase, ReviewPhase::Done);
    assert!(!drive.is_active());
    assert!(drive.coverage_complete());
    assert_eq!(drive.next_step(&completed(None, &[])), ReviewStep::Idle);
}

#[test]
fn incomplete_coverage_without_defects_is_done_and_reported() {
    let mut drive = ReviewDrive::start(inputs(), false, 3);
    let step = drive.next_step(&completed(Some(verdict(false, vec![])), &[]));
    assert!(
        matches!(step, ReviewStep::Done(ref t) if t.contains("1/2 plan/spec items implemented"))
    );
    assert!(!drive.coverage_complete());
    assert_eq!(drive.coverage_counts(), (1, 2));
    let lines = drive.report_lines();
    assert_eq!(lines[0], "coverage (incomplete):");
    assert!(
        lines
            .iter()
            .any(|l| l.contains("missing") && l.contains("/topic"))
    );
    assert!(lines.contains(&"findings: none".to_string()));
    assert_eq!(
        lines.last().map(String::as_str),
        Some(
            "next: 1 plan/spec item(s) are missing or partial (above); build them, then `/review` again"
        )
    );
}

/// Live run: a re-audit built the missing `/topic` feature and then reported
/// it missing. Write tools are denied in audit turns; whatever still slips
/// through (a shell redirection) is recorded and shown, so the verdict is
/// read knowing the tree moved under it.
#[test]
fn files_changed_by_an_audit_turn_are_recorded_and_warned_about() {
    let mut drive = ReviewDrive::start(inputs(), false, 3);
    drive.next_step(&completed(Some(verdict(false, vec![p0()])), &[]));
    assert_eq!(drive.phase, ReviewPhase::Fix);
    drive.next_step(&completed(None, &["src/server.rs"]));
    assert_eq!(drive.phase, ReviewPhase::Reaudit);
    let step = drive.next_step(&completed(
        Some(verdict(false, vec![])),
        &["src/main.rs", "tests/integration.rs"],
    ));
    assert!(matches!(step, ReviewStep::Done(_)), "{step:?}");
    assert_eq!(
        drive.audit_changed_files,
        vec![
            "src/main.rs".to_string(),
            "tests/integration.rs".to_string()
        ]
    );
    assert_eq!(
        drive.all_changed_files,
        vec!["src/server.rs".to_string()],
        "audit changes are not fix-pass changes"
    );
    let lines = drive.report_lines();
    assert!(
        lines.iter().any(|line| line
            == "warning: an audit turn changed files (it is meant to be read-only; /undo restores that turn): src/main.rs, tests/integration.rs"),
        "{lines:?}"
    );

    let clean = ReviewDrive::start(inputs(), false, 3);
    assert!(clean.audit_changed_files.is_empty());

    let mut stopped = ReviewDrive::start(inputs(), false, 3);
    let step = stopped.next_step(&errored(
        "audit turn kept calling write tools after they were denied; stopping the turn",
        &["src/main.rs"],
    ));
    assert!(matches!(step, ReviewStep::Stopped(_)), "{step:?}");
    assert_eq!(
        stopped.audit_changed_files,
        vec!["src/main.rs".to_string()],
        "an audit turn that errored still records what it changed"
    );
}

/// Live run: `/review audit` on a repo with only a README ended with the
/// raw block, a summary, and nothing that said what to do now. The report
/// ends with the next action, and says when the model's verdict word was
/// overruled by its own rows.
#[test]
fn report_ends_with_the_next_action_and_notes_a_contradicted_verdict() {
    let readme = ReviewInputs {
        files: vec![ReviewInput {
            path: "README.md".into(),
            kind: InputKind::Readme,
        }],
        ..ReviewInputs::default()
    };
    let mut drive = ReviewDrive::start(readme, true, 3);
    let mut contradicted = verdict(true, vec![]);
    contradicted.stated_complete = Some(false);
    let step = drive.next_step(&completed(Some(contradicted), &[]));
    assert!(
        matches!(step, ReviewStep::Done(ref t) if t.contains("all 2 plan/spec items implemented")),
        "{step:?}"
    );
    let lines = drive.report_lines();
    assert_eq!(lines[0], "coverage (complete):");
    assert!(
        lines.contains(
            &"note: the model wrote INCOMPLETE with every coverage row implemented; verdict taken from the rows"
                .to_string()
        ),
        "{lines:?}"
    );
    assert_eq!(
        lines.last().map(String::as_str),
        Some(
            "next: this audited README.md, not a plan or spec; write plan.md (a `- [ ]` checklist of what should exist) or pass one for a coverage audit: `/review docs/spec.md [dir]`"
        )
    );

    let mut audit_only = ReviewDrive::start(inputs(), true, 3);
    audit_only.next_step(&completed(Some(verdict(true, vec![p0()])), &[]));
    assert_eq!(
        audit_only.next_action().as_deref(),
        Some("next: `/review` (without `audit`) fixes the 1 open P0/P1 finding(s)")
    );

    let mut defects_only = ReviewDrive::start(ReviewInputs::default(), false, 3);
    let mut bare = verdict(true, vec![]);
    bare.coverage.clear();
    defects_only.next_step(&completed(Some(bare), &[]));
    assert!(
        defects_only
            .next_action()
            .is_some_and(|next| next.starts_with("next: this audited for defects only;")),
    );

    let mut clean = ReviewDrive::start(inputs(), false, 3);
    clean.next_step(&completed(Some(verdict(true, vec![])), &[]));
    assert_eq!(clean.next_action(), None, "nothing left to point at");
    assert!(!clean.report_lines().iter().any(|l| l.starts_with("next:")));

    let mut running = ReviewDrive::start(inputs(), false, 3);
    running.next_step(&completed(Some(verdict(false, vec![p0()])), &[]));
    assert_eq!(running.phase, ReviewPhase::Fix);
    assert!(
        !running
            .report_lines()
            .iter()
            .any(|l| l.starts_with("next:")),
        "mid-loop status has no next action"
    );
}

/// Live run: `finding: P1 | Implement TOPIC …` for the unchecked `/topic`
/// row sent the fix pass off to build the feature. With the checklist on
/// the drive that row is a gap: the fix pass takes only the real defect,
/// the report still lists the gap, and a gap-only audit is simply done.
#[test]
fn unchecked_plan_rows_filed_as_findings_are_reported_not_fixed() {
    let topic = finding(
        crate::review::Severity::P1,
        "Implement TOPIC with broadcast and replay on join",
        "src/main.rs:80",
    );
    let checklist = vec![
        row("Welcome line is sent first on connect", Some(true)),
        row(
            "/topic: TOPIC command with broadcast and replay on join",
            Some(false),
        ),
    ];

    let mut drive = ReviewDrive::start(inputs(), false, 3);
    drive.checklist_items = checklist.clone();
    let step = drive.next_step(&completed(
        Some(verdict(false, vec![p0(), topic.clone()])),
        &[],
    ));
    let ReviewStep::RunPrompt(fix) = step else {
        panic!("the P0 still opens a fix pass, got {step:?}");
    };
    assert!(fix.contains("[P0] Reject empty nicknames"), "{fix}");
    assert!(!fix.contains("TOPIC"), "the gap is not fix work: {fix}");
    assert_eq!(drive.findings_to_fix.len(), 1);
    assert_eq!(
        drive.status_line(),
        "spec review · fix pass 1/3 · 1 P0/P1 · plan.md + docs/spec.md"
    );
    let stored = drive.last_verdict.as_ref().unwrap();
    assert_eq!(stored.findings.len(), 2, "the gap stays in the verdict");
    assert!(stored.findings[1].feature_gap);
    assert_eq!(drive.open_blocking().len(), 1);
    assert!(
        drive
            .report_lines()
            .iter()
            .any(|l| l.contains("[P1] Implement TOPIC")
                && l.contains("(unchecked plan item: reported, not fixed)")),
        "{:?}",
        drive.report_lines()
    );

    let mut gap_only = ReviewDrive::start(inputs(), false, 3);
    gap_only.checklist_items = checklist;
    match gap_only.next_step(&completed(Some(verdict(false, vec![topic])), &[])) {
        ReviewStep::Done(text) => {
            assert!(text.contains("no P0/P1 defects"), "{text}");
            assert!(
                text.contains("1 P0/P1 row(s) are unchecked plan items, reported not fixed"),
                "{text}"
            );
        }
        other => panic!("a gap-only verdict needs no fix pass, got {other:?}"),
    }
    assert_eq!(gap_only.open_blocking().len(), 0);
    assert_eq!(gap_only.pass, 0);
}

#[test]
fn blocking_findings_seed_fix_then_reaudit_then_done() {
    let mut drive = ReviewDrive::start(inputs(), false, 3);
    let step = drive.next_step(&completed(
        Some(verdict(false, vec![p0(), p1(), p2()])),
        &[],
    ));
    let ReviewStep::RunPrompt(fix) = step else {
        panic!("expected fix prompt");
    };
    assert!(fix.contains("Fix pass 1/3"));
    assert!(fix.contains("[P0] Reject empty nicknames"));
    assert!(fix.contains("[P1] KICK requires operator"));
    assert!(
        !fix.contains("Log unknown commands"),
        "P2 stays out of the fix loop"
    );
    assert_eq!(drive.phase, ReviewPhase::Fix);
    assert_eq!(drive.pass, 1);
    assert_eq!(drive.findings_to_fix.len(), 2);
    assert_eq!(
        drive.status_line(),
        "spec review · fix pass 1/3 · 2 P0/P1 · plan.md + docs/spec.md"
    );
    assert_eq!(drive.intent_for_prompt(&fix), Some(Intent::Fix));
    assert_eq!(drive.intent_for_prompt("user typed this"), None);

    let step = drive.next_step(&completed(None, &["src/server.rs", "src/commands.rs"]));
    let ReviewStep::RunPrompt(reaudit) = step else {
        panic!("expected re-audit prompt");
    };
    assert!(reaudit.contains("Re-audit after fix pass 1/3"));
    assert!(reaudit.contains("- src/server.rs\n"));
    assert!(reaudit.contains("[P0] Reject empty nicknames"));
    assert_eq!(drive.phase, ReviewPhase::Reaudit);
    assert_eq!(drive.intent_for_prompt(&reaudit), Some(Intent::Review));
    assert_eq!(drive.all_changed_files.len(), 2);

    let step = drive.next_step(&completed(Some(verdict(false, vec![p2()])), &[]));
    assert!(
        matches!(step, ReviewStep::Done(ref t) if t.contains("no P0/P1 defects after 1 fix pass(es)"))
    );
    assert_eq!(drive.phase, ReviewPhase::Done);
}

#[test]
fn audit_only_reports_blocking_findings_and_stops() {
    let mut drive = ReviewDrive::start(inputs(), true, 3);
    assert_eq!(
        drive.status_line(),
        "spec review · audit only · plan.md + docs/spec.md"
    );
    let step = drive.next_step(&completed(Some(verdict(false, vec![p0()])), &[]));
    assert!(
        matches!(step, ReviewStep::Done(ref t) if t.contains("1 P0/P1 defect(s) still open (audit only)"))
    );
    assert_eq!(drive.phase, ReviewPhase::Done);
    assert_eq!(drive.pass, 0);
    assert_eq!(drive.open_blocking().len(), 1);
}

#[test]
fn identical_findings_after_reaudit_stop_for_no_progress() {
    let mut drive = ReviewDrive::start(inputs(), false, 3);
    drive.next_step(&completed(Some(verdict(false, vec![p0()])), &[]));
    drive.next_step(&completed(None, &["src/server.rs"]));
    let mut moved = p0();
    moved.location = Some("src/server.rs:120".into());
    let step = drive.next_step(&completed(Some(verdict(false, vec![moved])), &[]));
    match step {
        ReviewStep::Stopped(text) => assert!(text.contains("no progress"), "{text}"),
        other => panic!("expected Stopped, got {other:?}"),
    }
    assert_eq!(drive.phase, ReviewPhase::Stopped);
    assert!(drive.stop_reason.as_deref().unwrap().contains("fix pass 1"));
    assert!(drive.status_line().contains("stopped: no progress"));
}

#[test]
fn different_findings_continue_until_the_pass_cap() {
    let mut drive = ReviewDrive::start(inputs(), false, 2);
    assert!(matches!(
        drive.next_step(&completed(Some(verdict(false, vec![p0()])), &[])),
        ReviewStep::RunPrompt(_)
    ));
    assert!(matches!(
        drive.next_step(&completed(None, &["a.rs"])),
        ReviewStep::RunPrompt(_)
    ));
    let step = drive.next_step(&completed(Some(verdict(false, vec![p1()])), &[]));
    assert!(matches!(step, ReviewStep::RunPrompt(ref p) if p.contains("Fix pass 2/2")));
    assert_eq!(drive.pass, 2);
    assert!(matches!(
        drive.next_step(&completed(None, &["b.rs"])),
        ReviewStep::RunPrompt(_)
    ));
    let step = drive.next_step(&completed(Some(verdict(false, vec![p0()])), &[]));
    match step {
        ReviewStep::Stopped(text) => assert!(text.contains("pass cap reached"), "{text}"),
        other => panic!("expected Stopped, got {other:?}"),
    }
    assert_eq!(drive.pass, 2);
}

#[test]
fn missing_block_reasks_once_then_stops() {
    let mut drive = ReviewDrive::start(inputs(), false, 3);
    let step = drive.next_step(&completed(None, &[]));
    assert_eq!(
        step,
        ReviewStep::RunPrompt(crate::review::REVIEW_FORMAT_HINT.to_string()),
        "no checklist and no verdict yet: the generic template"
    );
    assert_eq!(drive.phase, ReviewPhase::Audit);
    assert!(drive.format_retry_used);
    let step = drive.next_step(&completed(None, &[]));
    assert!(
        matches!(step, ReviewStep::Stopped(ref t) if t.contains("no parseable <review> block"))
    );
    assert_eq!(drive.phase, ReviewPhase::Stopped);

    let mut drive = ReviewDrive::start(inputs(), false, 3);
    drive.next_step(&completed(None, &[]));
    let step = drive.next_step(&completed(Some(verdict(true, vec![])), &[]));
    assert!(matches!(step, ReviewStep::Done(_)));
    assert!(
        !drive.format_retry_used,
        "a parsed verdict resets the re-ask"
    );
}

/// Live run: the re-audit answered with a "Build Next" plan for the missing
/// item instead of the block, then repeated it verbatim to the generic
/// re-ask. The re-ask is now a form with the item column pre-filled.
#[test]
fn reask_and_reaudit_carry_the_item_column() {
    let mut drive = ReviewDrive::start(inputs(), false, 3);
    drive.checklist_items = vec![
        row("Welcome banner", Some(true)),
        row("/topic command", Some(false)),
    ];
    let step = drive.next_step(&completed(None, &[]));
    let ReviewStep::RunPrompt(reask) = step else {
        panic!("expected the re-ask, got {step:?}");
    };
    assert!(reask.starts_with("[hi:review] Format re-ask."), "{reask}");
    assert!(
        reask.contains("do not write a plan or next steps"),
        "{reask}"
    );
    assert!(
        reask.contains("coverage: <state> | Welcome banner | <path:line or ->\n"),
        "{reask}"
    );
    assert!(
        reask.contains("coverage: <state> | /topic command | <path:line or ->\n"),
        "{reask}"
    );
    assert!(!reask.contains("Prior findings"), "first audit has none");
    assert_eq!(
        drive.coverage_items(),
        vec!["Welcome banner".to_string(), "/topic command".to_string()]
    );
    assert_eq!(drive.unchecked_items(), vec!["/topic command".to_string()]);

    // Once a verdict exists its rows name the items; the re-audit and a
    // re-ask during it list those and the findings under re-check.
    drive.next_step(&completed(Some(verdict(false, vec![p0()])), &[]));
    assert_eq!(
        drive.coverage_items(),
        vec!["Welcome".to_string(), "/topic".to_string()]
    );
    let ReviewStep::RunPrompt(reaudit) = drive.next_step(&completed(None, &["src/main.rs"])) else {
        panic!("re-audit expected");
    };
    assert!(
        reaudit.contains(
            "Plan/spec items (one `coverage:` row each, in this order):\n- Welcome\n- /topic\n"
        ),
        "{reaudit}"
    );
    assert!(reaudit.contains("Report; do not plan."), "{reaudit}");
    let ReviewStep::RunPrompt(reask) = drive.next_step(&completed(None, &[])) else {
        panic!("re-ask expected");
    };
    assert!(
        reask.contains("coverage: <state> | Welcome | <path:line or ->\n"),
        "{reask}"
    );
    assert!(
        reask.contains("Prior findings, each listed again only if it is still present:\n- [P0] Reject empty nicknames — src/server.rs:88\n"),
        "{reask}"
    );
}

#[test]
fn cancel_pauses_and_resume_prompt_replays_the_phase() {
    let mut drive = ReviewDrive::start(inputs(), false, 3);
    let cancelled = ReviewTurnFacts {
        stop_reason: TurnStopReason::Cancelled,
        error: None,
        changed_files: Vec::new(),
        verdict: None,
        unverified_citations: Vec::new(),
    };
    let step = drive.next_step(&cancelled);
    assert!(matches!(step, ReviewStep::Paused(ref t) if t.contains("paused during audit")));
    assert!(drive.paused);
    assert!(drive.is_active());
    assert!(drive.status_line().contains("paused"));
    assert!(
        drive
            .resume_prompt()
            .unwrap()
            .contains("Spec-coverage audit")
    );

    drive.paused = false;
    drive.next_step(&completed(Some(verdict(false, vec![p0()])), &[]));
    drive.next_step(&cancelled);
    assert_eq!(drive.phase, ReviewPhase::Fix);
    assert!(drive.resume_prompt().unwrap().contains("Fix pass 1/3"));
    drive.paused = false;
    drive.next_step(&completed(None, &["src/server.rs"]));
    drive.next_step(&cancelled);
    assert_eq!(drive.phase, ReviewPhase::Reaudit);
    assert!(
        drive
            .resume_prompt()
            .unwrap()
            .contains("Re-audit after fix pass 1/3")
    );
    assert_eq!(drive.stop_reason, None);
}

fn errored(error: &str, changed: &[&str]) -> ReviewTurnFacts {
    ReviewTurnFacts {
        stop_reason: TurnStopReason::Error,
        error: Some(error.into()),
        changed_files: changed.iter().map(|s| s.to_string()).collect(),
        verdict: None,
        unverified_citations: Vec::new(),
    }
}

#[test]
fn audit_turn_error_stops_the_loop_with_the_error() {
    let mut drive = ReviewDrive::start(inputs(), false, 3);
    let step = drive.next_step(&errored("identical tool storm; stopping the turn", &[]));
    match step {
        ReviewStep::Stopped(text) => {
            assert!(text.contains("audit turn failed"), "{text}");
            assert!(text.contains("tool storm"), "{text}");
        }
        other => panic!("expected Stopped, got {other:?}"),
    }
    assert!(!drive.is_active());
    assert_eq!(drive.resume_prompt(), None);
}

/// Live run: the defects were already fixed when the fix turn started, the
/// model could not close the seeded plan (the edit hint forbids
/// `update_plan` without edits) and the turn ended as a policy error even
/// though `cargo test` was green. The re-audit judges that tree.
#[test]
fn fix_turn_error_reaudits_the_tree_it_left() {
    let mut drive = ReviewDrive::start(inputs(), false, 3);
    drive.next_step(&completed(Some(verdict(false, vec![p0()])), &[]));
    assert_eq!(drive.phase, ReviewPhase::Fix);
    let step = drive.next_step(&errored(
        "plan still has unfinished steps",
        &["src/server.rs"],
    ));
    match step {
        ReviewStep::RunPrompt(prompt) => {
            assert!(prompt.contains("Re-audit after fix pass 1/3"), "{prompt}");
            assert!(prompt.contains("- src/server.rs\n"), "{prompt}");
        }
        other => panic!("a failed fix turn still re-audits, got {other:?}"),
    }
    assert_eq!(drive.phase, ReviewPhase::Reaudit);
    assert_eq!(
        drive.fix_turn_error.as_deref(),
        Some("plan still has unfinished steps")
    );
    assert!(
        drive
            .status_line()
            .contains("fix turn ended early: plan still has unfinished steps"),
        "{}",
        drive.status_line()
    );
    assert_eq!(drive.all_changed_files, vec!["src/server.rs".to_string()]);

    let step = drive.next_step(&completed(Some(verdict(true, vec![])), &[]));
    assert!(matches!(step, ReviewStep::Done(_)), "{step:?}");
    assert_eq!(drive.fix_turn_error, None, "cleared once a verdict is in");

    // The same finding surviving the failed pass is still "no progress".
    let mut stuck = ReviewDrive::start(inputs(), false, 3);
    stuck.next_step(&completed(Some(verdict(false, vec![p0()])), &[]));
    stuck.next_step(&errored("identical tool storm; stopping the turn", &[]));
    match stuck.next_step(&completed(Some(verdict(false, vec![p0()])), &[])) {
        ReviewStep::Stopped(text) => assert!(text.contains("no progress"), "{text}"),
        other => panic!("expected Stopped, got {other:?}"),
    }
}

#[test]
fn forced_review_intent_with_open_plan_never_demands_edit() {
    let mut facts = TurnFacts::new(Intent::Review, true);
    facts.used_tools = true;
    facts.visible_answer = true;
    assert_eq!(decide(Event::ModelStop, &mut facts), Decision::Complete);
    assert!(!facts.wants_edit());
    assert!(!facts.wants_verify());
}

#[test]
fn forced_fix_intent_with_seeded_plan_demands_verify_then_edit() {
    let mut facts = TurnFacts::new(Intent::Fix, true);
    facts.used_tools = true;
    facts.visible_answer = true;
    facts.mutated = true;
    match decide(Event::ModelStop, &mut facts) {
        Decision::Continue { demand, .. } => assert_eq!(demand, Demand::Verify),
        other => panic!("expected verify demand, got {other:?}"),
    }
    facts.note_continue(Demand::Verify);
    facts.ran_verify = true;
    match decide(Event::ModelStop, &mut facts) {
        Decision::Continue { demand, .. } => assert_eq!(demand, Demand::Edit),
        other => panic!("expected edit demand while steps remain, got {other:?}"),
    }
    facts.plan_open = false;
    assert_eq!(decide(Event::ModelStop, &mut facts), Decision::Complete);
}

#[test]
fn review_drive_round_trips_through_the_session_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("s.jsonl");
    let mut session = JsonlSession::create(&path).unwrap();
    let mut drive = ReviewDrive::start(inputs(), false, 3);
    drive.next_step(&completed(Some(verdict(false, vec![p0(), p2()])), &[]));
    drive.paused = true;
    drive.unverified_citations = vec!["src/gone.rs:1".into()];
    session.record_review_drive(&drive).unwrap();
    let loaded = JsonlSession::load(&path).unwrap();
    assert_eq!(loaded.review_drive, drive);
    assert_eq!(loaded.review_drive.phase, ReviewPhase::Fix);
    assert!(loaded.review_drive.paused);

    session
        .rewrite(&LoadedSession {
            review_drive: drive.clone(),
            ..LoadedSession::default()
        })
        .unwrap();
    assert_eq!(JsonlSession::load(&path).unwrap().review_drive, drive);

    session.rewrite(&LoadedSession::default()).unwrap();
    let raw = std::fs::read_to_string(&path).unwrap();
    assert!(!raw.contains("review_drive"), "idle drives are not written");
    assert_eq!(
        JsonlSession::load(&path).unwrap().review_drive,
        ReviewDrive::default()
    );
}
