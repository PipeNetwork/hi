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
use crate::{App, TurnState, watchdog_stuck_timeout};
use hi_agent::{Command, command};

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
) -> Result<DriveCompletion<T>> {
    tokio::pin!(fut);
    let mut cancelled = false;
    let mut value = None;
    let mut last_activity = Instant::now();
    let mut watchdog_stuck = false;
    let watchdog_timeout = watchdog_stuck_timeout();
    let mut pending_confirmation: Option<ConfirmationControl> = None;
    let mut confirmations_open = true;
    loop {
        terminal.draw(|f| app.render(f))?;
        tokio::select! {
            result = &mut fut => {
                while let Ok(event) = rx.try_recv() {
                    if let Some(tap) = &app.remote_event_tap {
                        tap(&event);
                    }
                    app.apply(event);
                }
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
                if let Some(tap) = &app.remote_event_tap {
                    tap(&event);
                }
                app.apply(event);
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
                    Some(Event::Paste(text)) if pending_confirmation.is_none() => app.input.insert_str(&text),
                    Some(Event::Paste(_)) => {}
                    Some(Event::Key(key)) if key.kind == KeyEventKind::Press => {
                        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                        if let Some(request) = pending_confirmation.take() {
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
                                        app.add_auto_approve_path(&path);
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
                                    cancelled = true;
                                    break;
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
                            KeyCode::Char('c') if ctrl => { cancelled = true; break; }
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
                                    cancelled = true;
                                    break;
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
                                    app.queue.push_back(cmd);
                                    app.clamp_queue_selection();
                                    continue;
                                }
                                None => {}
                            }
                            if let Some(submitted) = app.edit_key(&key) {
                                match command::parse(&submitted) {
                                    Some(Command::Copy(arg)) => app.copy(&arg),
                                    Some(Command::Btw(question)) => {
                                        // A `/btw` side question goes straight to the
                                        // running turn as a question — not the next-turn
                                        // queue. Tag it so the loop frames it as
                                        // "answer briefly, then continue" rather than
                                        // steering. Falls back to a normal queued turn
                                        // when no inbox (nothing running) is attached.
                                        if let Some(inbox) = interject.as_ref() {
                                            inbox.push(format!(
                                                "{}{}",
                                                hi_agent::BTW_INTERJECTION_PREFIX,
                                                question
                                            ));
                                            app.follow();
                                        } else {
                                            app.queue.push_back(submitted.clone());
                                            app.clamp_queue_selection();
                                            app.follow();
                                        }
                                    }
                                    other => {
                                        // Always queue so the line shows under the
                                        // prompt and runs after this turn if it was
                                        // not consumed as mid-turn steering.
                                        app.queue.push_back(submitted.clone());
                                        app.clamp_queue_selection();
                                        let plain = other.is_none();
                                        if plain {
                                            if let Some(inbox) = interject.as_ref() {
                                                inbox.push(submitted.clone());
                                                app.mid_turn_offered.push_back(submitted.clone());
                                            }
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
                        cancelled = true;
                        break;
                    }
                    _ => {}
                }
            }
        }
    }
    app.waiting_for = None;
    app.confirmation = None;
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
