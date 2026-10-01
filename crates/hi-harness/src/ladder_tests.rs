use super::*;

const VERIFY: &str = "cargo test --quiet";
const TASK: &str = "make parse_port reject 0";

fn rungs(specs: &[&str]) -> Vec<LadderRung> {
    specs
        .iter()
        .map(|spec| LadderRung::parse(spec).unwrap())
        .collect()
}

fn three_rungs() -> Vec<LadderRung> {
    rungs(&[
        "pipe/deepseek-v4-flash low",
        "pipe/deepseek-v4-pro medium",
        "pipe/gpt-6-astra high",
    ])
}

fn ladder(attempts_per_rung: u32) -> Ladder {
    Ladder::start(
        TASK,
        "pipe/base",
        three_rungs(),
        attempts_per_rung,
        VERIFY,
        0,
    )
}

fn usage(input: u64, output: u64) -> Usage {
    Usage {
        input_tokens: input,
        output_tokens: output,
        ..Usage::default()
    }
}

fn completed(verification: &str, changed: &[&str]) -> LadderTurnFacts {
    LadderTurnFacts {
        stop_reason: TurnStopReason::Completed,
        changed_files: changed.iter().map(|file| file.to_string()).collect(),
        verification: Some(verification.into()),
        usage: usage(1_000, 100),
        ..LadderTurnFacts::default()
    }
}

fn failed(output: &str, changed: &[&str]) -> LadderTurnFacts {
    LadderTurnFacts {
        verify_output: Some(output.into()),
        ..completed("failed", changed)
    }
}

fn prompt_of(step: LadderStep) -> String {
    match step {
        LadderStep::RunPrompt(prompt) => prompt,
        other => panic!("expected another attempt, got {other:?}"),
    }
}

#[test]
fn rung_parse_takes_model_syntax() {
    assert_eq!(
        LadderRung::parse("pipe/deepseek-v4-flash low").unwrap(),
        LadderRung {
            model: Some("pipe/deepseek-v4-flash".into()),
            effort: Some(EffortArg::Level(ReasoningEffort::Low)),
        }
    );
    assert_eq!(
        LadderRung::parse("  high ").unwrap(),
        LadderRung {
            model: None,
            effort: Some(EffortArg::Level(ReasoningEffort::High)),
        }
    );
    assert_eq!(
        LadderRung::parse("pipe/gpt-6-astra").unwrap(),
        LadderRung {
            model: Some("pipe/gpt-6-astra".into()),
            effort: None,
        }
    );
    assert_eq!(
        LadderRung::parse("max").unwrap().effort,
        Some(EffortArg::Level(ReasoningEffort::Xhigh))
    );
    assert_eq!(
        LadderRung::parse("pipe/kimi-3 off").unwrap().effort,
        Some(EffortArg::Off)
    );
    assert!(LadderRung::parse("   ").is_err());
    let typo = LadderRung::parse("pipe/kimi-3 hihg").unwrap_err();
    assert!(typo.contains("pipe/kimi-3 hihg"), "{typo}");
}

#[test]
fn rung_labels_fall_back_to_the_base_model() {
    let rungs = rungs(&["high", "pipe/x", "pipe/y low"]);
    assert_eq!(rungs[0].label("pipe/base"), "pipe/base · high");
    assert_eq!(rungs[1].label("pipe/base"), "pipe/x");
    assert_eq!(rungs[2].label("pipe/base"), "pipe/y · low");
}

#[test]
fn default_settings_are_an_effort_ladder_on_the_session_model() {
    let settings = LadderSettings::default();
    assert_eq!(settings.attempts_per_rung, DEFAULT_ATTEMPTS_PER_RUNG);
    assert!(settings.jev_start);
    let resolved = settings.resolved_rungs();
    assert_eq!(resolved.len(), DEFAULT_EFFORT_RUNGS.len());
    assert!(resolved.iter().all(|rung| rung.model.is_none()));
    assert_eq!(
        resolved
            .iter()
            .map(|rung| rung.label("pipe/base"))
            .collect::<Vec<_>>(),
        ["pipe/base · low", "pipe/base · medium", "pipe/base · high"]
    );
    let configured = LadderSettings {
        rungs: three_rungs(),
        ..LadderSettings::default()
    };
    assert_eq!(configured.resolved_rungs(), three_rungs());
}

#[test]
fn start_clamps_the_start_rung_and_attempts() {
    let high = Ladder::start(TASK, "pipe/base", three_rungs(), 0, VERIFY, 9);
    assert_eq!(high.rung, 2);
    assert_eq!(high.attempts_per_rung, 1);
    assert!(high.is_active());
    let many = Ladder::start(TASK, "pipe/base", three_rungs(), 99, VERIFY, 0);
    assert_eq!(many.attempts_per_rung, MAX_ATTEMPTS_PER_RUNG);
}

#[test]
fn a_pass_on_the_cheapest_rung_ends_verified() {
    let mut ladder = ladder(1);
    let step = ladder.after_turn(&completed("passed", &["src/port.rs"]));
    let LadderStep::Verified(summary) = step else {
        panic!("expected verified, got {step:?}");
    };
    assert_eq!(ladder.phase, LadderPhase::Verified);
    assert_eq!(ladder.rung, 0);
    assert!(summary.contains("rung 1/3"), "{summary}");
    assert!(summary.contains(VERIFY), "{summary}");
    assert!(!summary.contains("no attempt changed files"), "{summary}");
    assert_eq!(ladder.attempts.len(), 1);
    assert_eq!(ladder.attempts[0].result, AttemptResult::Verified);
    assert_eq!(
        ladder.rung_effort(),
        None,
        "a finished ladder owns no effort"
    );
}

#[test]
fn a_pass_with_no_changes_is_flagged() {
    let mut ladder = ladder(1);
    ladder.after_turn(&completed("passed", &[]));
    assert!(
        ladder.status_line().contains("no attempt changed files"),
        "{}",
        ladder.status_line()
    );
}

#[test]
fn a_failed_check_climbs_with_its_output_and_the_task() {
    let mut ladder = ladder(1);
    let prompt = prompt_of(ladder.after_turn(&failed(
        "test port::zero ... FAILED\nassertion failed: parse_port(0).is_err()",
        &["src/port.rs"],
    )));
    assert_eq!(ladder.rung, 1);
    assert_eq!(ladder.attempts_on_rung, 0);
    assert_eq!(ladder.rung_model(), Some("pipe/deepseek-v4-pro"));
    assert_eq!(
        ladder.rung_effort(),
        Some(EffortArg::Level(ReasoningEffort::Medium))
    );
    assert!(prompt.starts_with(LADDER_PREFIX), "{prompt}");
    assert!(
        prompt.contains("attempt 2 · rung 2/3 pipe/deepseek-v4-pro · medium"),
        "{prompt}"
    );
    assert!(
        prompt.contains("assertion failed: parse_port(0).is_err()"),
        "{prompt}"
    );
    assert!(prompt.contains(&format!("`{VERIFY}` failed")), "{prompt}");
    assert!(prompt.ends_with(TASK), "{prompt}");
    assert!(!prompt.contains("changed no files"), "{prompt}");
}

#[test]
fn attempts_per_rung_retries_before_climbing() {
    let mut ladder = ladder(2);
    let retry = prompt_of(ladder.after_turn(&failed("FAILED", &["src/port.rs"])));
    assert_eq!(ladder.rung, 0, "one attempt left on this rung");
    assert_eq!(ladder.attempts_on_rung, 1);
    assert!(retry.contains("rung 1/3"), "{retry}");
    assert!(
        ladder.status_line().contains("(2/2 on this rung)"),
        "{}",
        ladder.status_line()
    );
    prompt_of(ladder.after_turn(&failed("FAILED", &["src/port.rs"])));
    assert_eq!(ladder.rung, 1);
    assert_eq!(ladder.attempts_on_rung, 0);
}

#[test]
fn an_attempt_that_changed_nothing_climbs_at_once() {
    let mut ladder = ladder(3);
    let prompt = prompt_of(ladder.after_turn(&failed("FAILED", &[])));
    assert_eq!(
        ladder.rung, 1,
        "a retry on the same rung would repeat itself"
    );
    assert!(prompt.contains("changed no files"), "{prompt}");
}

#[test]
fn a_turn_error_counts_as_a_failed_attempt() {
    let mut ladder = ladder(1);
    let prompt = prompt_of(ladder.after_turn(&LadderTurnFacts {
        stop_reason: TurnStopReason::Error,
        error: Some("404 model not found: pipe/deepseek-v4-flash".into()),
        ..LadderTurnFacts::default()
    }));
    assert_eq!(ladder.rung, 1, "an unavailable rung is skipped");
    assert!(prompt.contains("404 model not found"), "{prompt}");
    assert!(!prompt.contains("```text"), "no check output to quote");
    assert_eq!(
        ladder.attempts[0].result.label(),
        "error: 404 model not found: pipe/deepseek-v4-flash"
    );
}

#[test]
fn failing_the_top_rung_exhausts_the_ladder() {
    let mut ladder = Ladder::start(
        TASK,
        "pipe/base",
        rungs(&["pipe/a", "pipe/b"]),
        1,
        VERIFY,
        0,
    );
    prompt_of(ladder.after_turn(&failed("FAILED", &["a.rs"])));
    let step = ladder.after_turn(&failed("FAILED", &["a.rs"]));
    let LadderStep::Stopped(summary) = step else {
        panic!("expected a stop, got {step:?}");
    };
    assert_eq!(ladder.phase, LadderPhase::Exhausted);
    assert!(summary.starts_with("ladder exhausted"), "{summary}");
    assert!(summary.contains("pipe/b"), "{summary}");
    assert!(summary.contains("2 attempt(s)"), "{summary}");
    assert_eq!(
        ladder.after_turn(&completed("passed", &[])),
        LadderStep::Idle
    );
}

#[test]
fn a_cancelled_turn_or_check_stops_the_ladder() {
    let mut turn = ladder(1);
    let step = turn.after_turn(&LadderTurnFacts {
        stop_reason: TurnStopReason::Cancelled,
        ..LadderTurnFacts::default()
    });
    assert!(matches!(step, LadderStep::Stopped(_)), "{step:?}");
    assert_eq!(turn.phase, LadderPhase::Stopped);
    assert_eq!(turn.attempts[0].result, AttemptResult::Cancelled);

    let mut check = ladder(1);
    check.after_turn(&completed("cancelled", &["a.rs"]));
    assert_eq!(check.phase, LadderPhase::Stopped);
}

#[test]
fn a_turn_without_a_check_stops_the_ladder() {
    let mut ladder = ladder(1);
    let step = ladder.after_turn(&LadderTurnFacts {
        stop_reason: TurnStopReason::Completed,
        changed_files: vec!["a.rs".into()],
        ..LadderTurnFacts::default()
    });
    let LadderStep::Stopped(summary) = step else {
        panic!("expected a stop, got {step:?}");
    };
    assert!(summary.contains("no verify command ran"), "{summary}");
}

#[test]
fn an_idle_ladder_ignores_turns_and_stops() {
    let mut ladder = Ladder::default();
    assert_eq!(
        ladder.after_turn(&completed("failed", &[])),
        LadderStep::Idle
    );
    assert_eq!(ladder.stop("stopped by /ladder stop"), LadderStep::Idle);
    assert_eq!(ladder.status_line(), "ladder: not running");
    assert!(ladder.report_lines().is_empty());
}

#[test]
fn stop_ends_a_running_ladder_once() {
    let mut ladder = ladder(1);
    let step = ladder.stop("stopped by /ladder stop");
    let LadderStep::Stopped(summary) = step else {
        panic!("expected a stop, got {step:?}");
    };
    assert!(summary.contains("stopped by /ladder stop"), "{summary}");
    assert_eq!(ladder.stop("again"), LadderStep::Idle);
}

#[test]
fn long_check_output_keeps_its_tail() {
    let mut ladder = ladder(1);
    let head = "é".repeat(FAILURE_TAIL_CHARS * 2);
    let output = format!("{head}\nerror[E0308]: mismatched types at src/port.rs:9");
    let prompt = prompt_of(ladder.after_turn(&failed(&output, &["src/port.rs"])));
    assert!(
        prompt.contains("mismatched types at src/port.rs:9"),
        "tail kept"
    );
    assert!(
        prompt.contains(&format!("(last {FAILURE_TAIL_CHARS} characters)")),
        "cut is announced"
    );
    assert!(
        prompt.chars().filter(|ch| *ch == 'é').count() < FAILURE_TAIL_CHARS,
        "the head is dropped, not quoted"
    );
    assert_eq!(tail_chars("short", 10), ("short", false));
    assert_eq!(tail_chars("ééé", 2), ("éé", true));
}

#[test]
fn report_lists_each_attempt_and_the_token_total() {
    let mut ladder = ladder(1);
    ladder.after_turn(&failed("FAILED", &["src/port.rs"]));
    ladder.after_turn(&LadderTurnFacts {
        usage: usage(2_500, 300),
        ..completed("passed", &["src/port.rs", "tests/port.rs"])
    });
    let lines = ladder.report_lines();
    assert_eq!(lines.len(), 3, "{lines:?}");
    assert!(
        lines[0].contains("rung 1 pipe/deepseek-v4-flash · low — verify failed · 1 file(s)"),
        "{lines:?}"
    );
    assert!(
        lines[1].contains("rung 2 pipe/deepseek-v4-pro · medium — verified · 2 file(s)"),
        "{lines:?}"
    );
    assert_eq!(ladder.total_tokens(), (3_500, 400));
    assert!(lines[2].starts_with("  total:"), "{lines:?}");
}

#[test]
fn transcript_label_is_the_first_line_of_ladder_prompts_only() {
    let mut ladder = ladder(1);
    let prompt = prompt_of(ladder.after_turn(&failed("FAILED", &["a.rs"])));
    assert_eq!(
        transcript_label(&prompt).as_deref(),
        Some("[ladder] attempt 2 · rung 2/3 pipe/deepseek-v4-pro · medium")
    );
    assert_eq!(transcript_label(TASK), None);
}

#[test]
fn a_model_only_rung_leaves_effort_to_the_session() {
    let ladder = Ladder::start(TASK, "pipe/base", rungs(&["pipe/a"]), 1, VERIFY, 0);
    assert_eq!(ladder.rung_model(), Some("pipe/a"));
    assert_eq!(ladder.rung_effort(), None);
    let effort_only = Ladder::start(TASK, "pipe/base", rungs(&["low"]), 1, VERIFY, 0);
    assert_eq!(effort_only.rung_model(), Some("pipe/base"));
}
