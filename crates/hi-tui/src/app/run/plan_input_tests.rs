use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;

use crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};
use ratatui::{Terminal, backend::TestBackend};
use tokio::sync::mpsc;

use super::super::drive::drive;
use super::handle_working_plan_approval_key;
use crate::{App, plan_approval::PlanApprovalFocus};

fn app_with_card() -> App {
    let mut app = crate::tests::test_app("custom", "test-model");
    app.plan_mode = true;
    app.plan = vec![hi_agent::PlanStep {
        title: "Implement scheduler".into(),
        status: hi_agent::PlanStatus::Pending,
    }];
    app.open_plan_approval();
    app
}

async fn drive_event(app: &mut App, event: Event) -> bool {
    let mut terminal = Terminal::new(TestBackend::new(100, 30)).unwrap();
    let (input_tx, mut input_rx) = mpsc::unbounded_channel();
    input_tx.send(event).unwrap();
    let (ui_tx, ui_rx) = mpsc::unbounded_channel();
    let (_confirmation_tx, confirmation_rx) = mpsc::unbounded_channel();
    let mut ticker = tokio::time::interval(Duration::from_secs(60));
    let result = drive(
        &mut terminal,
        &mut input_rx,
        &mut ticker,
        app,
        ui_rx,
        confirmation_rx,
        async {
            tokio::time::sleep(Duration::from_millis(30)).await;
            Ok(())
        },
        false,
        None,
        None,
        ui_tx,
        None,
        Arc::new(hi_tools::BackgroundTaskRegistry::new()),
    )
    .await
    .unwrap();
    result.cancelled
}

#[tokio::test]
async fn escape_parks_live_plan_card_without_cancelling_turn_or_tool() {
    for running_tool in [false, true] {
        let mut app = app_with_card();
        let interrupt = Arc::new(AtomicBool::new(false));
        app.interrupt = Some(interrupt.clone());
        if running_tool {
            app.current_tool = Some("run".into());
        }
        let cancelled = drive_event(
            &mut app,
            Event::Key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)),
        )
        .await;
        assert!(!cancelled);
        assert!(!interrupt.load(Ordering::Relaxed));
        assert!(app.plan_approval.as_ref().unwrap().parked);
    }
}

#[tokio::test]
async fn escape_cancels_comment_edit_without_clearing_composer() {
    let mut app = app_with_card();
    app.input.set("preserve queued draft");
    let card = app.plan_approval.as_mut().unwrap();
    card.focus = PlanApprovalFocus::Commenting;
    card.comment_draft = "unsaved comment".into();
    let cancelled = drive_event(
        &mut app,
        Event::Key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)),
    )
    .await;
    assert!(!cancelled);
    assert_eq!(app.input.text(), "preserve queued draft");
    let card = app.plan_approval.as_ref().unwrap();
    assert!(!card.parked);
    assert_eq!(card.focus, PlanApprovalFocus::Preview);
    assert!(card.comment_draft.is_empty());
}

#[tokio::test]
async fn ctrl_c_still_cancels_turn_with_plan_card_visible() {
    let mut app = app_with_card();
    assert!(
        drive_event(
            &mut app,
            Event::Key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL))
        )
        .await
    );
}

#[tokio::test]
async fn paste_in_live_comment_editor_is_included_in_revision_feedback() {
    let mut app = app_with_card();
    app.input.set("Keep public API stable.");
    app.plan_approval.as_mut().unwrap().focus = PlanApprovalFocus::Commenting;
    assert!(
        !drive_event(
            &mut app,
            Event::Paste("Add cancellation tests.\r\nCover retries.".into())
        )
        .await
    );
    assert_eq!(app.input.text(), "Keep public API stable.");
    assert_eq!(
        app.plan_approval.as_ref().unwrap().comment_draft,
        "Add cancellation tests.\nCover retries."
    );
    assert!(handle_working_plan_approval_key(
        &mut app,
        &KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)
    ));
    assert!(handle_working_plan_approval_key(
        &mut app,
        &KeyEvent::new(KeyCode::Char('r'), KeyModifiers::NONE)
    ));
    let feedback = app.input.text();
    assert!(feedback.contains("Add cancellation tests.\nCover retries."));
    assert!(feedback.contains("Keep public API stable."));
}

#[test]
fn parked_comment_editor_leaves_paste_for_composer() {
    let mut app = app_with_card();
    let card = app.plan_approval.as_mut().unwrap();
    card.focus = PlanApprovalFocus::Commenting;
    card.parked = true;
    assert!(!app.paste_plan_comment("next prompt"));
    assert!(app.plan_approval.as_ref().unwrap().comment_draft.is_empty());
}

#[test]
fn covered_plan_card_cannot_take_approval_keys() {
    let mut app = app_with_card();
    app.tutorial = Some(crate::tutorial::TutorialOverlay::fresh());
    assert!(!handle_working_plan_approval_key(
        &mut app,
        &KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE)
    ));
    assert!(app.plan_mode);
    assert!(app.plan_approval_capturing());
    assert!(app.queue.is_empty());
    app.tutorial = None;
    app.mode = crate::mode::UiMode::Review;
    assert!(!handle_working_plan_approval_key(
        &mut app,
        &KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE)
    ));
    assert!(app.plan_mode);
    assert!(app.queue.is_empty());
}

#[test]
fn pending_tool_question_is_rendered_ahead_of_plan_approval() {
    let mut app = app_with_card();
    app.confirmation = Some(hi_agent::ConfirmationRequest::AskUser {
        question: "Which transport should the public API use?".into(),
        options: vec!["REST".into(), "gRPC".into()],
    });
    let mut terminal = Terminal::new(TestBackend::new(100, 30)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let screen: String = terminal
        .backend()
        .buffer()
        .content
        .iter()
        .map(|cell| cell.symbol())
        .collect();
    assert!(
        screen.contains("Which transport"),
        "visible question must own its keys: {screen}"
    );
    assert!(!screen.contains("Review the plan before execution."));
    assert!(!handle_working_plan_approval_key(
        &mut app,
        &KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE)
    ));
    assert!(app.plan_mode);
    assert!(app.plan_approval_capturing());
}

#[test]
fn tool_question_temporarily_overrides_fullscreen_and_local_picker_surfaces() {
    for covered_by_review in [false, true] {
        let mut app = app_with_card();
        if covered_by_review {
            app.mode = crate::mode::UiMode::Review;
        } else {
            app.tutorial = Some(crate::tutorial::TutorialOverlay::fresh());
        }
        app.local_directory_prompt = Some("/tmp/model-directory".into());
        app.confirmation = Some(hi_agent::ConfirmationRequest::AskUser {
            question: "Which transport should the public API use?".into(),
            options: vec!["REST".into(), "gRPC".into()],
        });
        let mut terminal = Terminal::new(TestBackend::new(100, 30)).unwrap();
        terminal.draw(|frame| app.render(frame)).unwrap();
        let screen: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect();
        assert!(
            screen.contains("Which transport"),
            "active question must be visible: {screen}"
        );
        assert!(!screen.contains("Review the plan before execution."));
        assert_eq!(
            app.local_directory_prompt.as_deref(),
            Some("/tmp/model-directory")
        );
        assert_eq!(app.mode.is_review(), covered_by_review);
        assert_eq!(app.tutorial.is_some(), !covered_by_review);
    }
}
