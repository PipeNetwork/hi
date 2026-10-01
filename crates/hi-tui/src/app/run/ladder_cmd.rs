//! `/ladder` in the TUI: start a ladder, chain retry and climb attempts
//! until the verify command passes or the top rung fails, and answer
//! `/ladder status` while a turn runs. Esc cancels the running attempt,
//! which ends the ladder.

use anyhow::Result;
use crossterm::event::Event;
use hi_harness::{Harness, LadderCommand, LadderStep, TurnOutcome};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::text::Line;
use std::io;
use tokio::sync::mpsc;

use crate::App;
use crate::event::UiEvent;
use crate::render::dim;

/// Stop reason when a turn fails before it produces an outcome.
const TURN_FAILED: &str = "the turn failed before it finished";

/// Everything that chains off a finished turn: spec-review passes, then
/// ladder attempts. At most one of the two is active.
pub(super) async fn chain_after_turn(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    input_rx: &mut mpsc::UnboundedReceiver<Event>,
    ticker: &mut tokio::time::Interval,
    app: &mut App,
    harness: &mut Harness,
    outcome: Option<TurnOutcome>,
) -> Result<()> {
    super::review_cmd::chain(terminal, input_rx, ticker, app, harness, outcome.clone()).await?;
    chain(terminal, input_rx, ticker, app, harness, outcome).await
}

/// Run attempts back to back while the ladder asks for them. `outcome` is
/// the turn that just ended.
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

/// `/ladder [status|stop|<task>]`. Returns the prompt to run when the
/// command starts a ladder.
pub(super) async fn handle(app: &mut App, harness: &mut Harness, arg: &str) -> Option<String> {
    match harness.ladder_command(arg).await {
        LadderCommand::Start { prompt, notes } => {
            push_lines(app, &notes.join("\n"));
            sync_status(app, harness);
            app.clear_suggested_prompt();
            Some(prompt)
        }
        LadderCommand::Message(text) => {
            push_lines(app, &text);
            sync_status(app, harness);
            None
        }
    }
}

/// Advance the ladder after a turn. Returns the next attempt's prompt, if
/// the ladder wants one. A `None` outcome means the turn future failed
/// before producing one; the ladder stops.
pub(super) fn after_turn(
    app: &mut App,
    harness: &mut Harness,
    outcome: Option<&TurnOutcome>,
) -> Option<String> {
    if !harness.ladder().is_active() {
        app.ladder_status = None;
        return None;
    }
    let Some(outcome) = outcome else {
        push_lines(app, &harness.stop_ladder(TURN_FAILED));
        sync_status(app, harness);
        return None;
    };
    let mut sink = hi_harness::TestUi::default();
    let step = harness.ladder_next_step(outcome, &mut sink);
    for text in sink.statuses {
        app.apply(UiEvent::Status { text });
    }
    let next = match step {
        LadderStep::RunPrompt(prompt) => Some(prompt),
        LadderStep::Verified(summary) | LadderStep::Stopped(summary) => {
            app.push(Line::styled(summary, dim()));
            for line in harness.ladder().report_lines() {
                app.push(Line::styled(line, dim()));
            }
            None
        }
        LadderStep::Idle => None,
    };
    sync_status(app, harness);
    next
}

/// `/ladder` while a turn is running: the harness is busy, so `status`
/// answers from the last status line the ladder published and anything
/// else says how a ladder starts and stops.
pub(super) fn running_reply(app: &mut App, arg: &str) {
    let text = if arg.trim().eq_ignore_ascii_case("status") {
        running_status(app)
    } else {
        "ladder: starts between turns · Esc cancels the running attempt and ends a running ladder"
            .into()
    };
    app.push(Line::styled(text, dim()));
}

fn running_status(app: &App) -> String {
    app.ladder_status
        .clone()
        .unwrap_or_else(|| "ladder: not running".into())
}

/// The harness switches models between rungs and back when the ladder
/// ends; the status line and the model the status bar shows follow it.
fn sync_status(app: &mut App, harness: &Harness) {
    let ladder = harness.ladder();
    app.ladder_status = ladder.is_active().then(|| ladder.status_line());
    let model = harness.model();
    if app.model != model {
        super::session::show_model(app, model);
    }
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
    use hi_harness::{HarnessConfig, LadderRung, LadderSettings, ToolHost, TurnStopReason};
    use hi_tools::ProcessRunner;
    use hi_tools::sandbox::SandboxPolicy;
    use std::path::Path;

    fn harness(root: &Path) -> Harness {
        let state = root.join(".hi");
        let runner = ProcessRunner::new_with_policy(root, SandboxPolicy::Off).expect("runner");
        let tools = ToolHost::new_with_runner(root.to_path_buf(), state.clone(), runner).unwrap();
        let mut config = HarnessConfig::pipe(root.to_path_buf(), "pk_test");
        config.state_root = state;
        let mut harness = Harness::new_with_tools(config, tools).unwrap();
        harness.set_ladder_settings(LadderSettings {
            rungs: vec![
                LadderRung::parse("pipe/cheap low").unwrap(),
                LadderRung::parse("pipe/strong high").unwrap(),
            ],
            verify: Some("cargo test --quiet".into()),
            ..LadderSettings::default()
        });
        harness
    }

    fn failed_check() -> TurnOutcome {
        TurnOutcome {
            stop_reason: TurnStopReason::Completed,
            usage: hi_ai::Usage::default(),
            changed_files: vec!["src/lib.rs".into()],
            error: None,
            verification: Some("failed".into()),
            attempted_mutation: false,
        }
    }

    fn served(id: &str, window: u32, price: (f64, f64)) -> hi_ai::ServedModel {
        serde_json::from_value(serde_json::json!({
            "id": id,
            "context_window": window,
            "price": [price.0, price.1],
        }))
        .unwrap()
    }

    #[tokio::test]
    async fn start_climb_and_stop_keep_the_status_line_in_sync() {
        let dir = tempfile::tempdir().unwrap();
        let mut harness = harness(dir.path());
        let mut app = test_app("pipe", "m");
        app.served.insert(
            "pipe/strong".into(),
            served("pipe/strong", 400_000, (3.0, 15.0)),
        );
        let session_model = harness.model();

        let prompt = handle(&mut app, &mut harness, "make the test pass").await;
        assert_eq!(prompt.as_deref(), Some("make the test pass"));
        assert!(
            running_status(&app).starts_with("ladder · rung 1/2 pipe/cheap · low"),
            "{}",
            running_status(&app)
        );
        assert_eq!(app.model, "pipe/cheap", "the status bar shows the rung");

        let next = after_turn(&mut app, &mut harness, Some(&failed_check()));
        let next = next.expect("a failed check earns another attempt");
        assert!(next.starts_with("[ladder] attempt 2"), "{next}");
        assert!(
            running_status(&app).starts_with("ladder · rung 2/2 pipe/strong · high"),
            "{}",
            running_status(&app)
        );
        assert_eq!(harness.model(), "pipe/strong");
        assert_eq!(app.model, "pipe/strong");
        assert_eq!(app.usage_pricing, Some((3.0, 15.0)));
        assert_eq!(app.context_window, Some(400_000));

        assert_eq!(after_turn(&mut app, &mut harness, None), None);
        assert_eq!(app.ladder_status, None);
        assert_eq!(running_status(&app), "ladder: not running");
        assert_eq!(
            harness.model(),
            session_model,
            "the session model comes back"
        );
        assert_eq!(app.model, session_model);
    }

    /// The ladder would switch away from a model picked mid-attempt and
    /// then restore the one it started from, so it says so instead.
    #[tokio::test]
    async fn a_model_picked_mid_ladder_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let mut harness = harness(dir.path());
        let mut app = test_app("pipe", "m");
        handle(&mut app, &mut harness, "make the test pass").await;

        super::super::session::apply_running_model(&mut app, &harness.live(), "pipe/other");
        assert_eq!(harness.model(), "pipe/cheap");
        assert_eq!(app.model, "pipe/cheap");
        assert!(
            app.transcript_text()
                .contains("ladder: it picks each attempt's model"),
            "{}",
            app.transcript_text()
        );
    }

    #[tokio::test]
    async fn a_refusal_is_shown_and_starts_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let mut harness = harness(dir.path());
        harness.set_ladder_settings(LadderSettings::default());
        let mut app = test_app("pipe", "m");
        assert_eq!(handle(&mut app, &mut harness, "fix it").await, None);
        assert_eq!(app.ladder_status, None);
        assert!(!harness.ladder().is_active());
    }
}
