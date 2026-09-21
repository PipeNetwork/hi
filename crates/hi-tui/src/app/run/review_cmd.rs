//! `/review` in the TUI: start the spec audit, chain fix -> re-audit turns,
//! and answer `/review status` while a turn runs.
//!
//! Everything user-facing here says "spec review": the Ctrl-G diff review in
//! `crate::review` already owns the bare word.

use anyhow::Result;
use crossterm::event::Event;
use hi_harness::{Harness, ReviewCommand, ReviewStep, TurnOutcome};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::text::Line;
use std::io;
use tokio::sync::mpsc;

use crate::App;
use crate::event::UiEvent;
use crate::render::dim;

/// Run review turns back to back while the drive asks for them. `outcome`
/// is the turn that just ended (a user prompt, `/review`, or a resumed turn).
pub(super) async fn chain(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    input_rx: &mut mpsc::UnboundedReceiver<Event>,
    ticker: &mut tokio::time::Interval,
    app: &mut App,
    harness: &mut Harness,
    mut outcome: Option<TurnOutcome>,
) -> Result<()> {
    while let Some(next) = after_turn(app, harness, outcome.as_ref()) {
        outcome = super::session::run_turn(terminal, input_rx, ticker, app, harness, &next, false)
            .await?;
    }
    Ok(())
}

/// `/review [audit|status|stop] [path...]`. Returns the prompt to run when
/// the command starts or resumes a review turn.
pub(super) fn handle(app: &mut App, harness: &mut Harness, arg: &str) -> Option<String> {
    match harness.review_command(arg) {
        ReviewCommand::Start { prompt, notes } => {
            for note in notes {
                app.push(Line::styled(note, dim()));
            }
            app.spec_review_status = Some(harness.review_drive().status_line());
            app.plan = harness.current_plan().to_vec();
            app.clear_suggested_prompt();
            Some(prompt)
        }
        // `/review stop` drops the fix checklist from the harness plan; the
        // panel renders `app.plan`, so refresh it here too (found by the
        // first live `/review audit` of this feature against its own doc).
        ReviewCommand::Message(text) => {
            push_lines(app, &text);
            app.plan = harness.current_plan().to_vec();
            sync_status(app, harness);
            None
        }
    }
}

/// Advance the review loop after a turn. Returns the next prompt to run, if
/// the drive wants another turn. `None` outcome means the turn future failed
/// before producing one (request error); the loop stops.
pub(super) fn after_turn(
    app: &mut App,
    harness: &mut Harness,
    outcome: Option<&TurnOutcome>,
) -> Option<String> {
    if !harness.review_drive().is_active() {
        app.spec_review_status = None;
        return None;
    }
    let Some(outcome) = outcome else {
        let text = harness.stop_review();
        push_lines(app, &text);
        app.spec_review_status = None;
        return None;
    };
    let mut sink = hi_harness::TestUi::default();
    let step = harness.review_next_step(outcome, &mut sink);
    if let Some(steps) = sink.plans.pop() {
        app.apply(UiEvent::Plan { steps });
    }
    for text in sink.statuses {
        app.apply(UiEvent::Status { text });
    }
    let next = match step {
        ReviewStep::RunPrompt(prompt) => Some(prompt),
        // A stop after a fix pass still has a verdict: the report says what
        // is open and what to do; before any verdict it is empty.
        ReviewStep::Done(summary) | ReviewStep::Stopped(summary) => {
            app.push(Line::styled(summary, dim()));
            for line in harness.review_drive().report_lines() {
                app.push(Line::styled(line, dim()));
            }
            None
        }
        ReviewStep::Paused(text) => {
            app.push(Line::styled(text, dim()));
            for prompt in sink.suggested_prompts {
                app.apply(UiEvent::SuggestedPrompt { text: prompt });
            }
            None
        }
        ReviewStep::Idle => None,
    };
    sync_status(app, harness);
    next
}

/// `/review status` while a turn is running: the harness is busy, so answer
/// from the last status line the loop published.
pub(super) fn running_status(app: &App) -> String {
    app.spec_review_status
        .clone()
        .unwrap_or_else(|| "spec review: not running".into())
}

/// Short transcript echo for review-injected prompts instead of the full
/// multi-paragraph instruction block.
pub(super) fn transcript_echo(prompt: &str) -> String {
    hi_harness::review_transcript_label(prompt).unwrap_or_else(|| prompt.to_string())
}

fn sync_status(app: &mut App, harness: &Harness) {
    let drive = harness.review_drive();
    app.spec_review_status = drive.is_active().then(|| drive.status_line());
}

fn push_lines(app: &mut App, text: &str) {
    for line in text.lines() {
        app.push(Line::styled(line.to_string(), dim()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::test_app;
    use hi_harness::{HarnessConfig, LoadedSession, ReviewDrive, ReviewPhase, ToolHost};
    use hi_tools::sandbox::SandboxPolicy;
    use hi_tools::{PlanStatus, PlanStep, ProcessRunner};
    use std::path::Path;

    fn harness(root: &Path) -> Harness {
        let state = root.join(".hi");
        let runner = ProcessRunner::new_with_policy(root, SandboxPolicy::Off).expect("runner");
        let tools = ToolHost::new_with_runner(root.to_path_buf(), state.clone(), runner).unwrap();
        let mut config = HarnessConfig::pipe(root.to_path_buf(), "pk_test");
        config.state_root = state;
        Harness::new_with_tools(config, tools).unwrap()
    }

    /// Found by the first live `/review audit` of this feature against its
    /// own doc: `/review stop` cleared the harness checklist, but the panel
    /// renders `app.plan`, which kept showing the `[P0]` steps.
    #[test]
    fn stop_clears_the_plan_panel_with_the_harness_checklist() {
        let dir = tempfile::tempdir().unwrap();
        let mut harness = harness(dir.path());
        let checklist = vec![PlanStep {
            title: "[P0] Reject empty nicknames — src/server.rs:1".into(),
            status: PlanStatus::Pending,
        }];
        harness.apply_loaded_session(LoadedSession {
            plan: checklist.clone(),
            review_drive: ReviewDrive {
                phase: ReviewPhase::Fix,
                pass: 1,
                ..ReviewDrive::default()
            },
            ..LoadedSession::default()
        });
        let mut app = test_app("pipe", "m");
        app.plan = checklist;
        app.spec_review_status = Some(harness.review_drive().status_line());

        assert_eq!(handle(&mut app, &mut harness, "stop"), None);
        assert_eq!(harness.review_drive().phase, ReviewPhase::Stopped);
        assert!(harness.current_plan().is_empty());
        assert!(
            app.plan.is_empty(),
            "the panel mirrors the cleared checklist, got {:?}",
            app.plan
        );
        assert_eq!(app.spec_review_status, None);

        assert_eq!(handle(&mut app, &mut harness, "stop"), None);
        assert!(app.plan.is_empty(), "a second stop has nothing to restore");
    }
}
