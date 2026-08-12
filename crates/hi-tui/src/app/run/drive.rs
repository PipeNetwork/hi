//! Drive an agent future while keeping the TUI live (redraw, scroll, cancel, interject).

use std::io;
use std::time::Instant;

use anyhow::Result;
use crossterm::event::{Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::prelude::*;
use tokio::sync::mpsc;

use super::{ChordPipeline, reconcile_queue_with_interjections, run_chord_pipeline};
use crate::event::{ConfirmationControl, UiEvent};
use crate::{App, TurnState, dim, watchdog_stuck_timeout};
use hi_agent::{Command, command};

fn apply_ui_event(app: &mut App, event: UiEvent) {
    if let Some(tap) = &app.remote_event_tap {
        tap(&event);
    }
    app.apply(event);
}

fn drain_ui_events(app: &mut App, rx: &mut mpsc::UnboundedReceiver<UiEvent>, limit: usize) {
    let mut pending: Option<UiEvent> = None;
    for _ in 0..limit {
        let Ok(event) = rx.try_recv() else { break };
        if let Some(tap) = &app.remote_event_tap {
            tap(&event);
        }
        let merged = match (&mut pending, event) {
            (Some(UiEvent::Text { text }), UiEvent::Text { text: next })
            | (Some(UiEvent::Reasoning { text }), UiEvent::Reasoning { text: next }) => {
                text.push_str(&next);
                true
            }
            (
                Some(UiEvent::ToolStream { name, line }),
                UiEvent::ToolStream {
                    name: next_name,
                    line: next_line,
                },
            ) if *name == next_name => {
                line.push('\n');
                line.push_str(&next_line);
                true
            }
            (_, event) => {
                if let Some(previous) = pending.take() {
                    app.apply(previous);
                }
                pending = Some(event);
                true
            }
        };
        debug_assert!(merged);
    }
    if let Some(event) = pending {
        app.apply(event);
    }
}

/// Drive a model future (a turn or a compaction) to completion while keeping
/// the UI live: redraw + spin every tick, drain the agent's events, let the
/// user scroll/queue/cancel. Successful values are preserved so typed turn
/// outcomes, rather than UI prose, can drive final presentation.
pub(crate) struct DriveCompletion<T> {
    pub(crate) cancelled: bool,
    pub(crate) value: Option<T>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn drive<T>(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    input: &mut mpsc::UnboundedReceiver<Event>,
    ticker: &mut tokio::time::Interval,
    app: &mut App,
    mut rx: mpsc::UnboundedReceiver<UiEvent>,
    mut confirmations: mpsc::UnboundedReceiver<ConfirmationControl>,
    fut: impl std::future::Future<Output = Result<T>>,
    expect_turn_end: bool,
    // When set, plain-text lines submitted while the turn runs are injected
    // into the *current* turn (mid-turn steering) instead of queued for the
    // next one. Slash-commands always queue.
    interject: Option<hi_agent::InterjectionInbox>,
    // Immediate `/btw` launcher — fires own model call(s) without waiting for
    // the main turn's next model-round boundary.
    btw: Option<hi_agent::BtwDispatcher>,
    // Clone of the ChannelUi sender so `/btw` side events can join the same
    // UiEvent stream the pane already drains.
    ui_tx: mpsc::UnboundedSender<UiEvent>,
    // Whole-turn cancel signal shared with `run_turn_cancellable`. When the
    // user hits Ctrl+C / Esc-to-cancel, we fire this and keep polling `fut`
    // so the agent can settle tool_results cooperatively instead of only
    // dropping the turn future.
    turn_cancel: Option<hi_agent::TurnCancellation>,
) -> Result<DriveCompletion<T>> {
    tokio::pin!(fut);
    let mut cancelled = false;
    let mut value = None;
    let mut last_activity = Instant::now();
    let mut watchdog_stuck = false;
    let watchdog_timeout = watchdog_stuck_timeout();
    let mut pending_confirmation: Option<ConfirmationControl> = None;
    let mut confirmations_open = true;
    let signal_turn_cancel = |app: &mut App, cancelled_flag: &mut bool| {
        *cancelled_flag = true;
        if let Some(cancel) = turn_cancel.as_ref() {
            cancel.cancel();
        }
        if let Some(flag) = app.interrupt.as_ref() {
            flag.store(true, std::sync::atomic::Ordering::Release);
        }
    };
    loop {
        // After a cancel request with a shared TurnCancellation, keep the
        // drive loop alive until the turn future settles (or fails). Breaking
        // early would drop the future before cooperative tool cleanup runs.
        if cancelled && turn_cancel.is_some() && value.is_none() {
            // Fall through to select! so `fut` can complete.
        }
        terminal.draw(|f| app.render(f))?;
        tokio::select! {
            result = &mut fut => {
                drain_ui_events(app, &mut rx, 1024);
                match result {
                    Ok(result) => value = Some(result),
                    Err(err) => {
                        let (kind, guidance) = hi_agent::classify_error(&err);
                        if !matches!(app.last_turn_state, TurnState::Failed(_)) {
                            app.note_turn_failed(&format!("{err:#}"), kind, guidance);
                        }
                        if hi_agent::ui::error_counts_as_model_issue(&err) {
                            app.record_model_issue();
                        }
                    }
                }
                break;
            }
            Some(event) = rx.recv() => {
                last_activity = Instant::now();
                apply_ui_event(app, event);
                // Batch a bounded number of already queued stream chunks before
                // redrawing, while preserving select-loop fairness for input.
                drain_ui_events(app, &mut rx, 64);
            }
            request = confirmations.recv(), if pending_confirmation.is_none() && confirmations_open => {
                match request {
                    Some(request) => {
                        // Session-wide `a` or path-scoped `p` auto-approve.
                        if app.should_auto_approve(&request.request) {
                            let _ = request.response.send(hi_agent::ConfirmationResult::Approved);
                        } else {
                            app.confirmation = Some(request.request.clone());
                            app.confirmation_scroll = 0;
                            if matches!(
                                request.request,
                                hi_agent::ConfirmationRequest::AskUser { .. }
                            ) {
                                app.ask_user_draft.clear();
                            }
                            pending_confirmation = Some(request);
                        }
                    }
                    None => confirmations_open = false,
                }
            }
            _ = ticker.tick() => {
                app.spinner = app.spinner.wrapping_add(1);
                app.drain_loops();
                app.drain_voice();
                let idle = last_activity.elapsed();
                app.waiting_for = Some(idle);
                // Only notify about a quiet backend while no tool is legitimately
                // running. This is only a soft wait notice.
                if expect_turn_end
                    && !watchdog_stuck
                    && app.current_tool.is_none()
                    && idle >= watchdog_timeout
                {
                    watchdog_stuck = true;
                    app.note_backend_waiting(idle, watchdog_timeout);
                }
            },
            maybe = input.recv() => {
                match maybe {
                    Some(Event::Mouse(mouse)) => app.handle_mouse(mouse),
                    Some(Event::Paste(text))
                        if pending_confirmation.as_ref().is_some_and(|request| {
                            matches!(
                                request.request,
                                hi_agent::ConfirmationRequest::AskUser { .. }
                            )
                        }) =>
                    {
                        app.ask_user_draft.push_str(&text);
                    }
                    Some(Event::Paste(text)) if pending_confirmation.is_none() => {
                        app.input.insert_str(&text)
                    }
                    Some(Event::Paste(_)) => {}
                    Some(Event::Key(key)) if key.kind == KeyEventKind::Press => {
                        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                        if let Some(request) = pending_confirmation.take() {
                            if let hi_agent::ConfirmationRequest::AskUser { options, .. } =
                                &request.request
                            {
                                let options = options.clone();
                                match key.code {
                                    KeyCode::Char(c)
                                        if !ctrl && c.is_ascii_digit() && c != '0' =>
                                    {
                                        let picked =
                                            options.get((c as u8 - b'1') as usize).cloned();
                                        if let Some(option) = picked {
                                            let _ = request.response.send(
                                                hi_agent::ConfirmationResult::Answer(option),
                                            );
                                            app.confirmation = None;
                                            app.ask_user_draft.clear();
                                        } else {
                                            app.ask_user_draft.push(c);
                                            pending_confirmation = Some(request);
                                        }
                                    }
                                    KeyCode::Char(c) if !ctrl => {
                                        app.ask_user_draft.push(c);
                                        pending_confirmation = Some(request);
                                    }
                                    KeyCode::Backspace => {
                                        app.ask_user_draft.pop();
                                        pending_confirmation = Some(request);
                                    }
                                    KeyCode::Enter => {
                                        let answer = app.ask_user_draft.trim().to_string();
                                        if answer.is_empty() {
                                            pending_confirmation = Some(request);
                                        } else {
                                            let _ = request.response.send(
                                                hi_agent::ConfirmationResult::Answer(answer),
                                            );
                                            app.confirmation = None;
                                            app.ask_user_draft.clear();
                                        }
                                    }
                                    KeyCode::Esc => {
                                        let _ = request.response.send(
                                            hi_agent::ConfirmationResult::Cancelled,
                                        );
                                        app.confirmation = None;
                                        app.ask_user_draft.clear();
                                    }
                                    KeyCode::Char('c') if ctrl => {
                                        let _ = request.response.send(
                                            hi_agent::ConfirmationResult::Cancelled,
                                        );
                                        app.confirmation = None;
                                        app.ask_user_draft.clear();
                                        signal_turn_cancel(app, &mut cancelled);
                                        if turn_cancel.is_none() {
                                            break;
                                        }
                                        continue;
                                    }
                                    KeyCode::Up => {
                                        app.confirmation_scroll =
                                            app.confirmation_scroll.saturating_sub(1);
                                        pending_confirmation = Some(request);
                                    }
                                    KeyCode::Down => {
                                        app.confirmation_scroll =
                                            app.confirmation_scroll.saturating_add(1);
                                        pending_confirmation = Some(request);
                                    }
                                    KeyCode::PageUp => {
                                        app.confirmation_scroll =
                                            app.confirmation_scroll.saturating_sub(10);
                                        pending_confirmation = Some(request);
                                    }
                                    KeyCode::PageDown => {
                                        app.confirmation_scroll =
                                            app.confirmation_scroll.saturating_add(10);
                                        pending_confirmation = Some(request);
                                    }
                                    _ => pending_confirmation = Some(request),
                                }
                                continue;
                            }
                            match key.code {
                                KeyCode::Char('y') if !ctrl => {
                                    let _ = request.response.send(hi_agent::ConfirmationResult::Approved);
                                    app.confirmation = None;
                                }
                                KeyCode::Char('n') if !ctrl => {
                                    let _ = request.response.send(hi_agent::ConfirmationResult::Rejected);
                                    app.confirmation = None;
                                }
                                // "Always allow this session": approve this
                                // request AND auto-approve all subsequent ones
                                // without showing the modal. Removes the y-y-y-y
                                // fatigue during a heavy edit session.
                                KeyCode::Char('a') if !ctrl => {
                                    let _ = request.response.send(hi_agent::ConfirmationResult::Approved);
                                    app.auto_approve_session = true;
                                    app.confirmation = None;
                                    app.push(Line::styled(
                                        "auto-approve on for this session (approvals suppressed until quit)",
                                        Style::default().fg(crate::theme::theme().accent_success),
                                    ));
                                }
                                // Path-scoped auto-approve for file edits (`p`).
                                KeyCode::Char('p') if !ctrl => {
                                    if let hi_agent::ConfirmationRequest::FileEdit { path, .. } =
                                        &request.request
                                    {
                                        let prefix = App::auto_approve_prefix_for(path);
                                        app.add_auto_approve_path(path);
                                        let _ = request
                                            .response
                                            .send(hi_agent::ConfirmationResult::Approved);
                                        app.confirmation = None;
                                        app.push(Line::styled(
                                            format!(
                                                "auto-approve path '{prefix}/' for this session"
                                            ),
                                            Style::default()
                                                .fg(crate::theme::theme().accent_success),
                                        ));
                                    } else {
                                        // Not a file edit — keep the modal open.
                                        pending_confirmation = Some(request);
                                    }
                                }
                                KeyCode::Esc => {
                                    let _ = request.response.send(hi_agent::ConfirmationResult::Rejected);
                                    app.confirmation = None;
                                }
                                KeyCode::Char('c') if ctrl => {
                                    let _ = request.response.send(hi_agent::ConfirmationResult::Cancelled);
                                    app.confirmation = None;
                                    signal_turn_cancel(app, &mut cancelled);
                                    if turn_cancel.is_none() {
                                        break;
                                    }
                                    continue;
                                }
                                KeyCode::Up => {
                                    app.confirmation_scroll = app.confirmation_scroll.saturating_sub(1);
                                    pending_confirmation = Some(request);
                                }
                                KeyCode::Down => {
                                    app.confirmation_scroll = app.confirmation_scroll.saturating_add(1);
                                    pending_confirmation = Some(request);
                                }
                                KeyCode::PageUp => {
                                    app.confirmation_scroll = app.confirmation_scroll.saturating_sub(10);
                                    pending_confirmation = Some(request);
                                }
                                KeyCode::PageDown => {
                                    app.confirmation_scroll = app.confirmation_scroll.saturating_add(10);
                                    pending_confirmation = Some(request);
                                }
                                _ => pending_confirmation = Some(request),
                            }
                            continue;
                        }
                        match key.code {
                            KeyCode::Char('c') if ctrl => {
                                signal_turn_cancel(app, &mut cancelled);
                                if turn_cancel.is_none() {
                                    break;
                                }
                                continue;
                            }
                            // Esc clears a half-typed queued command, or — when the
                            // input is empty — interrupts the current tool call
                            // (if one is running) or cancels the whole turn.
                            KeyCode::Esc if app.input.is_empty() => {
                                if app.current_tool.is_some() {
                                    // A tool is running: signal interrupt to skip
                                    // just this tool call, not the whole turn.
                                    if let Some(flag) = &app.interrupt {
                                        flag.store(true, std::sync::atomic::Ordering::Relaxed);
                                    }
                                } else {
                                    signal_turn_cancel(app, &mut cancelled);
                                    if turn_cancel.is_none() {
                                        break;
                                    }
                                    continue;
                                }
                            }
                            KeyCode::Esc => app.input.clear(),
                            // A line submitted while a turn runs: `/copy` reads the
                            // selection synchronously; everything else joins the
                            // visible next-turn queue. Plain text is *also* offered
                            // to the in-flight turn as mid-turn steering when an
                            // inbox is available (slash-commands only queue).
                            _ => {
                            // Shared palette + action/mode dispatch (in-turn path).
                            match run_chord_pipeline(app, &key) {
                                Some(ChordPipeline::Continue) => continue,
                                Some(ChordPipeline::OpenPalette) => {
                                    app.palette = Some(crate::palette::CommandPalette::open());
                                    continue;
                                }
                                Some(ChordPipeline::PaletteAccept(cmd)) => {
                                    let _ = app.enqueue_prompt(cmd);
                                    continue;
                                }
                                None => {}
                            }
                            if let Some(submitted) = app.edit_key(&key) {
                                match command::parse(&submitted) {
                                    Some(Command::Copy(arg)) => app.copy(&arg),
                                    Some(Command::Btw(question)) => {
                                        // Immediate side-channel answer: own model
                                        // call(s) via BtwDispatcher — do NOT wait
                                        // for the main turn's next model round.
                                        let question = question.trim();
                                        if question.is_empty() {
                                            app.push(Line::styled(
                                                "usage: /btw <question>".to_string(),
                                                dim(),
                                            ));
                                        } else if let Some(dispatch) = btw.as_ref().filter(|d| d.is_enabled()) {
                                            // Pane only — no main-transcript tool/Q spam.
                                            app.btw_note_question(question);
                                            let (side_tx, mut side_rx) =
                                                mpsc::unbounded_channel::<hi_agent::BtwSideEvent>();
                                            if dispatch.ask(question, side_tx) {
                                                let forward = ui_tx.clone();
                                                tokio::spawn(async move {
                                                    while let Some(ev) = side_rx.recv().await {
                                                        let mapped = match ev {
                                                            hi_agent::BtwSideEvent::Question(q) => {
                                                                UiEvent::BtwQuestion { question: q }
                                                            }
                                                            hi_agent::BtwSideEvent::Answer(text) => {
                                                                UiEvent::BtwAnswer { text }
                                                            }
                                                            hi_agent::BtwSideEvent::ToolStarted {
                                                                name,
                                                                arguments,
                                                            } => UiEvent::BtwToolStarted {
                                                                name,
                                                                arguments,
                                                            },
                                                            hi_agent::BtwSideEvent::ToolResult {
                                                                name,
                                                                result,
                                                            } => UiEvent::BtwToolResult {
                                                                name,
                                                                result,
                                                            },
                                                            // Side-loop provider chatter stays out of the main transcript;
                                                            // the BTW pane already shows tools/answers.
                                                            hi_agent::BtwSideEvent::Status(_) => {
                                                                continue;
                                                            }
                                                            hi_agent::BtwSideEvent::End => {
                                                                UiEvent::BtwEnd
                                                            }
                                                        };
                                                        if forward.send(mapped).is_err() {
                                                            break;
                                                        }
                                                    }
                                                });
                                            } else if let Some(inbox) = interject.as_ref() {
                                                // Dispatcher refused — fall back to inbox.
                                                inbox.push(format!(
                                                    "{}{}",
                                                    hi_agent::BTW_INTERJECTION_PREFIX,
                                                    question
                                                ));
                                            }
                                            app.follow();
                                        } else if let Some(inbox) = interject.as_ref() {
                                            // Fallback: queue for next model boundary.
                                            app.btw_note_question(question);
                                            inbox.push(format!(
                                                "{}{}",
                                                hi_agent::BTW_INTERJECTION_PREFIX,
                                                question
                                            ));
                                            app.follow();
                                        } else {
                                            app.push(Line::styled(
                                                "/btw is mid-turn only — nothing is running"
                                                    .to_string(),
                                                dim(),
                                            ));
                                        }
                                    }
                                    other => {
                                        // Always queue so the line shows under the
                                        // prompt and runs after this turn if it was
                                        // not consumed as mid-turn steering.
                                        if !app.enqueue_prompt(submitted.clone()) {
                                            continue;
                                        }
                                        let plain = other.is_none();
                                        if plain
                                            && let Some(inbox) = interject.as_ref() {
                                                inbox.push(submitted.clone());
                                                app.mid_turn_offered.push_back(submitted.clone());
                                            }
                                        app.follow();
                                    }
                                }
                            }
                            }
                        }
                    }
                    Some(Event::FocusGained) => app.set_focus(true),
                    Some(Event::FocusLost) => app.set_focus(false),
                    None => {
                        signal_turn_cancel(app, &mut cancelled);
                        if turn_cancel.is_none() {
                            break;
                        }
                        continue;
                    }
                    _ => {}
                }
            }
        }
    }
    app.waiting_for = None;
    app.confirmation = None;
    // Never leave the agent blocked on a oneshot that the modal dropped.
    // Turn completion, input EOF, and plain Ctrl+C can exit while a confirm
    // is still outstanding — resolve it explicitly so tool code sees Cancelled
    // rather than a disconnected channel.
    if let Some(request) = pending_confirmation.take() {
        let _ = request
            .response
            .send(hi_agent::ConfirmationResult::Cancelled);
    }
    // Reconcile the visible queue with mid-turn steering: drop entries the
    // agent already injected, and keep anything still pending in the inbox
    // (turn ended before the next Model phase) for the next turn.
    if let Some(inbox) = interject.as_ref() {
        reconcile_queue_with_interjections(app, inbox);
    } else {
        app.mid_turn_offered.clear();
    }
    Ok(DriveCompletion { cancelled, value })
}
