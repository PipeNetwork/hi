use super::*;
use ratatui::backend::TestBackend;

#[tokio::test]
async fn escape_during_running_tool_signals_whole_turn_cancellation() {
    let mut terminal = Terminal::new(TestBackend::new(100, 30)).unwrap();
    let (input_tx, mut input_rx) = mpsc::unbounded_channel();
    input_tx
        .send(Event::Key(crossterm::event::KeyEvent::new(
            KeyCode::Esc,
            KeyModifiers::NONE,
        )))
        .unwrap();
    let mut ticker = tokio::time::interval(std::time::Duration::from_secs(60));
    let mut app = crate::tests::test_app("openai", "gpt-4o");
    app.current_tool = Some("bash: curl http://127.0.0.1:8080/api".into());
    let interrupt = Arc::new(std::sync::atomic::AtomicBool::new(false));
    app.interrupt = Some(interrupt.clone());
    let (ui_tx, ui_rx) = mpsc::unbounded_channel();
    let (_confirmation_tx, confirmation_rx) = mpsc::unbounded_channel();
    let cancellation = hi_agent::TurnCancellation::new();
    let future_cancellation = cancellation.clone();
    let future = async move {
        while !future_cancellation.is_cancelled() {
            tokio::task::yield_now().await;
        }
        Ok(())
    };

    let result = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        drive(
            &mut terminal,
            &mut input_rx,
            &mut ticker,
            &mut app,
            ui_rx,
            confirmation_rx,
            future,
            false,
            None,
            None,
            ui_tx,
            Some(cancellation.clone()),
            Arc::new(hi_tools::BackgroundTaskRegistry::new()),
        ),
    )
    .await
    .expect("Esc cancellation must not wait for the foreground tool")
    .unwrap();

    assert!(result.cancelled);
    assert!(result.value.is_some());
    assert!(cancellation.is_cancelled());
    assert!(
        interrupt.load(std::sync::atomic::Ordering::Relaxed),
        "turn cancellation also wakes cooperative batch checks"
    );
}

#[tokio::test]
async fn clicking_working_stop_signals_cancellation_and_parks_the_queue() {
    let mut terminal = Terminal::new(TestBackend::new(100, 30)).unwrap();
    let (input_tx, mut input_rx) = mpsc::unbounded_channel();
    let mut ticker = tokio::time::interval(std::time::Duration::from_secs(60));
    let mut app = crate::tests::test_app("openai", "gpt-4o");
    app.working = true;
    app.queue.push_back("keep this follow-up".into());
    terminal.draw(|frame| app.render(frame)).unwrap();
    let stop = app.turn_status_rect;
    input_tx
        .send(Event::Mouse(crossterm::event::MouseEvent {
            kind: crossterm::event::MouseEventKind::Down(crossterm::event::MouseButton::Left),
            column: stop.x + stop.width - 1,
            row: stop.y,
            modifiers: KeyModifiers::NONE,
        }))
        .unwrap();
    let (ui_tx, ui_rx) = mpsc::unbounded_channel();
    let (_confirmation_tx, confirmation_rx) = mpsc::unbounded_channel();
    let cancellation = hi_agent::TurnCancellation::new();
    let future_cancellation = cancellation.clone();
    let future = async move {
        while !future_cancellation.is_cancelled() {
            tokio::task::yield_now().await;
        }
        Ok(())
    };

    let result = drive(
        &mut terminal,
        &mut input_rx,
        &mut ticker,
        &mut app,
        ui_rx,
        confirmation_rx,
        future,
        false,
        None,
        None,
        ui_tx,
        Some(cancellation.clone()),
        Arc::new(hi_tools::BackgroundTaskRegistry::new()),
    )
    .await
    .unwrap();

    assert!(result.cancelled);
    assert!(cancellation.is_cancelled());
    assert!(app.queue_paused);
    assert!(
        app.quit_notice.is_some(),
        "an exit escalation arriving just after settlement must still be honored"
    );
    assert!(!app.exit_requested);
}

#[tokio::test]
async fn quit_submitted_during_a_turn_cancels_without_entering_the_queue() {
    for quit in ["/quit", "/exit", "/q"] {
        let mut terminal = Terminal::new(TestBackend::new(100, 30)).unwrap();
        let (input_tx, mut input_rx) = mpsc::unbounded_channel();
        input_tx
            .send(Event::Key(crossterm::event::KeyEvent::new(
                KeyCode::Enter,
                KeyModifiers::NONE,
            )))
            .unwrap();
        let mut ticker = tokio::time::interval(std::time::Duration::from_secs(60));
        let mut app = crate::tests::test_app("openai", "gpt-4o");
        app.working = true;
        app.input.set(quit);
        let (ui_tx, ui_rx) = mpsc::unbounded_channel();
        let (_confirmation_tx, confirmation_rx) = mpsc::unbounded_channel();
        let cancellation = hi_agent::TurnCancellation::new();
        let future_cancellation = cancellation.clone();
        let future = async move {
            while !future_cancellation.is_cancelled() {
                tokio::task::yield_now().await;
            }
            Ok(())
        };

        let result = drive(
            &mut terminal,
            &mut input_rx,
            &mut ticker,
            &mut app,
            ui_rx,
            confirmation_rx,
            future,
            false,
            None,
            None,
            ui_tx,
            Some(cancellation.clone()),
            Arc::new(hi_tools::BackgroundTaskRegistry::new()),
        )
        .await
        .unwrap();

        assert!(result.cancelled, "{quit}");
        assert!(cancellation.is_cancelled(), "{quit}");
        assert!(app.exit_requested, "{quit}");
        assert!(app.queue.is_empty(), "{quit} must never become queued work");
    }
}

#[tokio::test]
async fn second_ctrl_c_during_cancellation_requests_exit_after_settlement() {
    let mut terminal = Terminal::new(TestBackend::new(100, 30)).unwrap();
    let (input_tx, mut input_rx) = mpsc::unbounded_channel();
    for _ in 0..2 {
        input_tx
            .send(Event::Key(crossterm::event::KeyEvent::new(
                KeyCode::Char('c'),
                KeyModifiers::CONTROL,
            )))
            .unwrap();
    }
    let mut ticker = tokio::time::interval(std::time::Duration::from_secs(60));
    let mut app = crate::tests::test_app("openai", "gpt-4o");
    app.working = true;
    let (ui_tx, ui_rx) = mpsc::unbounded_channel();
    let (_confirmation_tx, confirmation_rx) = mpsc::unbounded_channel();
    let cancellation = hi_agent::TurnCancellation::new();
    let future_cancellation = cancellation.clone();
    let future = async move {
        while !future_cancellation.is_cancelled() {
            tokio::task::yield_now().await;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        Ok(())
    };

    let result = drive(
        &mut terminal,
        &mut input_rx,
        &mut ticker,
        &mut app,
        ui_rx,
        confirmation_rx,
        future,
        false,
        None,
        None,
        ui_tx,
        Some(cancellation),
        Arc::new(hi_tools::BackgroundTaskRegistry::new()),
    )
    .await
    .unwrap();

    assert!(result.cancelled);
    assert!(app.exit_requested);
}

#[tokio::test]
async fn typed_settlement_controls_consumed_steering_even_when_cancel_key_races() {
    for status in [
        hi_agent::TurnStatus::Completed,
        hi_agent::TurnStatus::Cancelled,
        hi_agent::TurnStatus::Blocked,
        hi_agent::TurnStatus::Failed,
    ] {
        for frontend_cancelled in [false, true] {
            let mut terminal = Terminal::new(TestBackend::new(100, 30)).unwrap();
            let (input_tx, mut input_rx) = mpsc::unbounded_channel();
            if frontend_cancelled {
                input_tx
                    .send(Event::Key(crossterm::event::KeyEvent::new(
                        KeyCode::Char('c'),
                        KeyModifiers::CONTROL,
                    )))
                    .unwrap();
            }
            let mut ticker = tokio::time::interval(std::time::Duration::from_secs(60));
            let mut app = crate::tests::test_app("openai", "gpt-4o");
            let (ui_tx, ui_rx) = mpsc::unbounded_channel();
            let (_confirmation_tx, confirmation_rx) = mpsc::unbounded_channel();
            let inbox = hi_agent::InterjectionInbox::default();
            app.queue.push_back("preserve the public API".into());
            app.mid_turn_offered
                .push_back("preserve the public API".into());
            // The model drained this instruction before its terminal result.
            inbox.push("preserve the public API");
            inbox.drain();
            let cancellation = hi_agent::TurnCancellation::new();
            let future_cancellation = cancellation.clone();
            let future = async move {
                while frontend_cancelled && !future_cancellation.is_cancelled() {
                    tokio::task::yield_now().await;
                }
                let mut outcome =
                    hi_agent::TurnOutcome::infrastructure_failure("test-model", None, Vec::new());
                outcome.status = status;
                if status == hi_agent::TurnStatus::Cancelled {
                    future_cancellation.cancel();
                    outcome.stop_reason = hi_agent::TurnStopReason::Cancelled;
                }
                Ok(outcome)
            };

            let result = tokio::time::timeout(
                std::time::Duration::from_secs(2),
                drive(
                    &mut terminal,
                    &mut input_rx,
                    &mut ticker,
                    &mut app,
                    ui_rx,
                    confirmation_rx,
                    future,
                    true,
                    Some(inbox),
                    None,
                    ui_tx,
                    Some(cancellation),
                    Arc::new(hi_tools::BackgroundTaskRegistry::new()),
                ),
            )
            .await
            .unwrap()
            .unwrap();

            assert_eq!(result.value.as_ref().unwrap().status, status);
            assert_eq!(result.cancelled, frontend_cancelled);
            assert_eq!(
                app.queue.front().map(String::as_str),
                (status == hi_agent::TurnStatus::Cancelled).then_some("preserve the public API"),
                "status={status:?}, frontend_cancelled={frontend_cancelled}"
            );
            assert!(app.mid_turn_offered.is_empty());
        }
    }
}

#[tokio::test]
async fn closed_terminal_input_is_reported_instead_of_silently_exiting() {
    let mut terminal = Terminal::new(TestBackend::new(100, 30)).unwrap();
    let (input_tx, mut input_rx) = mpsc::unbounded_channel();
    drop(input_tx);
    let mut ticker = tokio::time::interval(std::time::Duration::from_secs(60));
    let mut app = crate::tests::test_app("openai", "gpt-4o");
    let (ui_tx, ui_rx) = mpsc::unbounded_channel();
    let (_confirmation_tx, confirmation_rx) = mpsc::unbounded_channel();

    let result = drive(
        &mut terminal,
        &mut input_rx,
        &mut ticker,
        &mut app,
        ui_rx,
        confirmation_rx,
        std::future::pending::<Result<()>>(),
        false,
        None,
        None,
        ui_tx,
        None,
        Arc::new(hi_tools::BackgroundTaskRegistry::new()),
    )
    .await;
    let error = match result {
        Ok(_) => panic!("closed input must be visible to the caller"),
        Err(error) => error,
    };

    assert_eq!(
        error.to_string(),
        "terminal input reader stopped unexpectedly; the active operation was cancelled"
    );
}

#[test]
fn late_cancel_preserves_committed_turn() {
    assert_eq!(
        settle_turn_cancellation(true, true, Some(hi_agent::TurnStatus::Completed)),
        TurnCancellationSettlement {
            cancelled: false,
            agent_already_cleaned: false,
        }
    );
}

#[test]
fn typed_cancel_skips_frontend_cleanup_but_missing_result_needs_it() {
    assert_eq!(
        settle_turn_cancellation(true, true, Some(hi_agent::TurnStatus::Cancelled)),
        TurnCancellationSettlement {
            cancelled: true,
            agent_already_cleaned: true,
        }
    );
    assert_eq!(
        settle_turn_cancellation(true, true, None),
        TurnCancellationSettlement {
            cancelled: true,
            agent_already_cleaned: false,
        }
    );
}

#[test]
fn timeout_returning_the_body_error_keeps_failure_semantics() {
    assert_eq!(
        settle_turn_cancellation(false, true, None),
        TurnCancellationSettlement {
            cancelled: false,
            agent_already_cleaned: false,
        }
    );
}

#[test]
fn trio_reviews_only_completed_turns() {
    assert_eq!(
        trio_non_reviewable_status(hi_agent::TurnStatus::Completed),
        None
    );
    assert_eq!(
        trio_non_reviewable_status(hi_agent::TurnStatus::Blocked),
        Some("blocked")
    );
    assert_eq!(
        trio_non_reviewable_status(hi_agent::TurnStatus::Failed),
        Some("failed")
    );
    assert_eq!(
        trio_non_reviewable_status(hi_agent::TurnStatus::Cancelled),
        Some("cancelled")
    );
}

#[test]
fn trio_default_never_settles_from_a_round_count() {
    assert!(!trio_round_cap_reached(0, None));
    assert!(!trio_round_cap_reached(3, None));
    assert!(!trio_round_cap_reached(u64::MAX, None));
    assert_eq!(trio_round_label(4, None), "4");
}

#[test]
fn trio_explicit_round_cap_still_settles_at_the_boundary() {
    assert!(!trio_round_cap_reached(2, Some(3)));
    assert!(trio_round_cap_reached(3, Some(3)));
    assert!(trio_round_cap_reached(4, Some(3)));
    assert_eq!(trio_round_label(2, Some(3)), "2/3");
}
