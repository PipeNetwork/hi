//! Drive an agent future while keeping the TUI live (redraw, scroll, cancel, interject).

use std::time::Instant;

use anyhow::Result;
use crossterm::event::{Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::Terminal;
use ratatui::prelude::*;
use std::sync::Arc;
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

/// Turn outcomes own transcript settlement. A typed cancellation rewinds
/// consumed steering, while completed/blocked/failed turns keep their messages
/// even if a late frontend interrupt raced their committed result.
pub(crate) trait DriveResult {
    fn interjections_committed(&self, frontend_cancelled: bool) -> bool;
}

impl DriveResult for hi_agent::TurnOutcome {
    fn interjections_committed(&self, _frontend_cancelled: bool) -> bool {
        self.status != hi_agent::TurnStatus::Cancelled
    }
}

impl DriveResult for () {
    fn interjections_committed(&self, frontend_cancelled: bool) -> bool {
        !frontend_cancelled
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TurnCancellationSettlement {
    /// The frontend should present cancellation and, if needed, clean up.
    pub(crate) cancelled: bool,
    /// The agent returned its typed Cancelled outcome and already performed
    /// rollback/cleanup; the frontend must not repeat either operation.
    pub(crate) agent_already_cleaned: bool,
}

/// Resolve a cancel-key/result race. A committed result that wins before the
/// cancellation token is observed remains committed. Only an explicit
/// frontend cancellation may use the no-result fallback: an internal hard
/// timeout can also leave the token set while returning the turn body's
/// original error, which must retain failure (rather than rollback) semantics.
pub(crate) fn settle_turn_cancellation(
    frontend_cancel_requested: bool,
    token_cancelled: bool,
    status: Option<hi_agent::TurnStatus>,
) -> TurnCancellationSettlement {
    let agent_already_cleaned = token_cancelled && status == Some(hi_agent::TurnStatus::Cancelled);
    TurnCancellationSettlement {
        cancelled: agent_already_cleaned || (frontend_cancel_requested && status.is_none()),
        agent_already_cleaned,
    }
}

/// Trio must not ask its skeptic to "approve" an execution that was blocked,
/// failed, or cancelled.
pub(crate) fn trio_non_reviewable_status(status: hi_agent::TurnStatus) -> Option<&'static str> {
    match status {
        hi_agent::TurnStatus::Failed => Some("failed"),
        hi_agent::TurnStatus::Blocked => Some("blocked"),
        hi_agent::TurnStatus::Cancelled => Some("cancelled"),
        hi_agent::TurnStatus::Completed => None,
    }
}

/// Whether a trio run has exhausted an explicitly configured round cap.
/// `None` is the ordinary unlimited mode and can never settle due to count.
pub(crate) fn trio_round_cap_reached(rounds_completed: u64, max_rounds: Option<u64>) -> bool {
    max_rounds.is_some_and(|max| rounds_completed >= max)
}

/// Compact progress label for trio status lines. An unlimited run displays its
/// current round without implying a denominator or hidden ceiling.
pub(crate) fn trio_round_label(round: u64, max_rounds: Option<u64>) -> String {
    max_rounds.map_or_else(|| round.to_string(), |max| format!("{round}/{max}"))
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn drive<T, B>(
    terminal: &mut Terminal<B>,
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
    // Cloneable task registry so `/tasks` can kill a background subagent
    // without borrowing the agent the turn future already holds.
    bg_tasks: Arc<hi_tools::BackgroundTaskRegistry>,
) -> Result<DriveCompletion<T>>
where
    T: DriveResult,
    B: Backend,
    B::Error: Send + Sync + 'static,
{
    tokio::pin!(fut);
    let mut cancelled = false;
    let mut value = None;
    let mut last_activity = Instant::now();
    let mut watchdog_stuck = false;
    let mut input_closed = false;
    let watchdog_timeout = watchdog_stuck_timeout();
    let mut pending_confirmation: Option<ConfirmationControl> = None;
    let mut confirm_queue: std::collections::VecDeque<ConfirmationControl> =
        std::collections::VecDeque::new();
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
        app.check_tui_event_trace()?;
        // After a cancel request with a shared TurnCancellation, keep the
        // drive loop alive until the turn future settles (or fails). Breaking
        // early would drop the future before cooperative tool cleanup runs.
        if cancelled && turn_cancel.is_some() && value.is_none() {
            // Fall through to select! so `fut` can complete.
        }
        if pending_confirmation.is_none() {
            while let Some(request) = confirm_queue.pop_front() {
                if app.should_auto_approve(&request.request) {
                    app.trace_approval_decided(
                        crate::tui_event_trace::approval_kind(&request.request),
                        "auto_approved",
                    );
                    let _ = request
                        .response
                        .send(hi_agent::ConfirmationResult::Approved);
                    continue;
                }
                app.confirmation = Some(request.request.clone());
                app.confirmation_scroll = 0;
                app.confirmation_selected = 0;
                app.confirm_focus = crate::confirm_overlay::ConfirmFocus::Options;
                app.ask_user_draft.clear();
                app.trace_approval_shown(crate::tui_event_trace::approval_kind(&request.request))?;
                pending_confirmation = Some(request);
                break;
            }
        }
        app.confirmation_waiting = confirm_queue.len();
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
            request = confirmations.recv(), if confirmations_open => {
                match request {
                    Some(request) => {
                        confirm_queue.push_back(request);
                    }
                    None => confirmations_open = false,
                }
            }
            _ = ticker.tick() => {
                if let Some(broker) = &app.x402_broker
                    && let Some(prompt) = broker.take()
                {
                    confirm_queue.push_back(x402_prompt_to_control(prompt));
                }
                app.spinner = app.spinner.wrapping_add(1);
                app.drain_loops();
                app.drain_voice();
                let idle = last_activity.elapsed();
                app.waiting_for = Some(idle);
                if expect_turn_end
                    && !watchdog_stuck
                    && app.current_tool.is_none()
                    && idle >= watchdog_timeout
                {
                    watchdog_stuck = true;
                    app.note_backend_waiting(idle, watchdog_timeout);
                }
            },
            maybe = input.recv(), if !input_closed => {
                match maybe {
                    Some(Event::Resize(width, height)) => {
                        // Keep resize synchronization available while a turn is
                        // active too. The harness must not have to guess whether
                        // SIGWINCH landed in the idle or drive input loop.
                        app.trace_resized(width, height)?;
                    }
                    Some(Event::Mouse(mouse)) => app.handle_mouse(mouse),
                    Some(Event::Paste(text))
                        if pending_confirmation.as_ref().is_some_and(|_| {
                            app.confirm_focus == crate::confirm_overlay::ConfirmFocus::Followup
                                || app.confirmation.as_ref().is_some_and(|request| {
                                    matches!(
                                        request,
                                        hi_agent::ConfirmationRequest::AskUser { .. }
                                    )
                                })
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
                            use crate::confirm_overlay::ConfirmDecision;
                            match crate::confirm_overlay::handle_key(
                                app,
                                &key,
                                &request.request,
                            ) {
                                ConfirmDecision::Redraw | ConfirmDecision::Unhandled => {
                                    pending_confirmation = Some(request);
                                }
                                ConfirmDecision::Approve => {
                                    app.trace_approval_decided(
                                        crate::tui_event_trace::approval_kind(&request.request),
                                        "approved",
                                    );
                                    let _ = request
                                        .response
                                        .send(hi_agent::ConfirmationResult::Approved);
                                    app.confirmation = None;
                                    app.ask_user_draft.clear();
                                }
                                ConfirmDecision::AlwaysSession => {
                                    app.trace_approval_decided(
                                        crate::tui_event_trace::approval_kind(&request.request),
                                        "always_session",
                                    );
                                    if let Some((server, tool)) =
                                        request.request.mcp_standing_grant()
                                    {
                                        let server = server.to_string();
                                        let tool = tool.to_string();
                                        app.add_auto_approve_mcp(server.clone(), tool.clone());
                                        let _ = request
                                            .response
                                            .send(hi_agent::ConfirmationResult::Approved);
                                        app.confirmation = None;
                                        app.ask_user_draft.clear();
                                        app.push(Line::styled(
                                            format!(
                                                "auto-approve MCP {server}.{tool} for this session"
                                            ),
                                            Style::default()
                                                .fg(crate::theme::theme().accent_success),
                                        ));
                                    } else {
                                        let _ = request
                                            .response
                                            .send(hi_agent::ConfirmationResult::Approved);
                                        app.auto_approve_session = true;
                                        app.confirmation = None;
                                        app.ask_user_draft.clear();
                                        app.push(Line::styled(
                                            "auto-approve file edits on for this session",
                                            Style::default()
                                                .fg(crate::theme::theme().accent_success),
                                        ));
                                    }
                                }
                                ConfirmDecision::AlwaysPath => {
                                    if let hi_agent::ConfirmationRequest::FileEdit { path, .. } =
                                        &request.request
                                    {
                                        let prefix = App::auto_approve_prefix_for(path);
                                        app.add_auto_approve_path(path);
                                        app.trace_approval_decided(
                                            crate::tui_event_trace::approval_kind(&request.request),
                                            "always_path",
                                        );
                                        let _ = request
                                            .response
                                            .send(hi_agent::ConfirmationResult::Approved);
                                        app.confirmation = None;
                                        app.ask_user_draft.clear();
                                        app.push(Line::styled(
                                            format!(
                                                "auto-approve path '{prefix}/' for this session"
                                            ),
                                            Style::default()
                                                .fg(crate::theme::theme().accent_success),
                                        ));
                                    } else {
                                        pending_confirmation = Some(request);
                                    }
                                }
                                ConfirmDecision::Reject => {
                                    app.trace_approval_decided(
                                        crate::tui_event_trace::approval_kind(&request.request),
                                        "rejected",
                                    );
                                    let _ = request
                                        .response
                                        .send(hi_agent::ConfirmationResult::Rejected);
                                    app.confirmation = None;
                                    app.ask_user_draft.clear();
                                }
                                ConfirmDecision::RejectFollowup(text) => {
                                    app.trace_approval_decided(
                                        crate::tui_event_trace::approval_kind(&request.request),
                                        "rejected_with_follow_up",
                                    );
                                    let _ = request
                                        .response
                                        .send(hi_agent::ConfirmationResult::Rejected);
                                    app.confirmation = None;
                                    app.ask_user_draft.clear();
                                    if !text.trim().is_empty() {
                                        let _ = app.enqueue_prompt_front(text);
                                    }
                                }
                                ConfirmDecision::Cancel => {
                                    app.trace_approval_decided(
                                        crate::tui_event_trace::approval_kind(&request.request),
                                        "cancelled",
                                    );
                                    let _ = request
                                        .response
                                        .send(hi_agent::ConfirmationResult::Cancelled);
                                    app.confirmation = None;
                                    app.ask_user_draft.clear();
                                    if ctrl {
                                        signal_turn_cancel(app, &mut cancelled);
                                        if turn_cancel.is_none() {
                                            break;
                                        }
                                    }
                                }
                                ConfirmDecision::Ask(answer) => {
                                    app.trace_approval_decided(
                                        crate::tui_event_trace::approval_kind(&request.request),
                                        "answered",
                                    );
                                    let _ = request
                                        .response
                                        .send(hi_agent::ConfirmationResult::Answer(answer));
                                    app.confirmation = None;
                                    app.ask_user_draft.clear();
                                }
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
                            KeyCode::Esc if app.dismiss_btw_overlay() => {
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
                                    let open_tasks = matches!(
                                        command::parse(&cmd),
                                        Some(Command::Tasks(_))
                                    ) || matches!(
                                        command::parse(&cmd),
                                        Some(Command::Queue(arg)) if arg.trim() == "tasks"
                                    );
                                    if open_tasks {
                                        crate::subagent_overlay::open_tasks(
                                            app,
                                            &[],
                                            &bg_tasks.list_now(),
                                        );
                                    } else {
                                        let _ = app.enqueue_prompt(cmd);
                                    }
                                    continue;
                                }
                                Some(ChordPipeline::KillTask(id)) => {
                                    let message = match bg_tasks.kill(&id).await {
                                        Some(outcome) => {
                                            crate::subagent_overlay::mark_cancelled(app, &id);
                                            format!("Task {} cancelled.", outcome.id)
                                        }
                                        None => format!(
                                            "kill_task error: no task with id \"{id}\""
                                        ),
                                    };
                                    app.push(Line::styled(message, dim()));
                                    continue;
                                }
                                Some(ChordPipeline::CycleSessionMode) => {
                                    app.cycle_session_face();
                                    continue;
                                }
                                Some(ChordPipeline::PlanApprove) => {
                                    if app.plan_approval_capturing() && app.plan_has_leftover() {
                                        app.apply_plan_approve_local();
                                        let prompt = if app.goal.as_ref().is_some_and(hi_agent::Goal::has_drive_work) {
                                            hi_agent::GOAL_CONTINUE_PROMPT
                                        } else {
                                            hi_agent::PLAN_DRIVE_PROMPT
                                        };
                                        let _ = app.enqueue_prompt_front(
                                            prompt.to_string(),
                                        );
                                    }
                                    continue;
                                }
                                Some(ChordPipeline::PlanPark) => {
                                    app.park_plan_approval_local();
                                    continue;
                                }
                                Some(ChordPipeline::PlanRequestChanges) => {
                                    app.apply_plan_request_changes_local();
                                    continue;
                                }
                                Some(ChordPipeline::PlanQuit) => {
                                    app.apply_plan_quit_local();
                                    continue;
                                }
                                None => {}
                            }
                            if let Some(submitted) = app.edit_key(&key) {
                                match command::parse(&submitted) {
                                    Some(Command::Copy(arg)) => app.copy(&arg),
                                    Some(Command::Tasks(_)) => {
                                        crate::subagent_overlay::open_tasks(
                                            app,
                                            &[],
                                            &bg_tasks.list_now(),
                                        );
                                    }
                                    Some(Command::Queue(arg)) if arg.trim() == "tasks" => {
                                        crate::subagent_overlay::open_tasks(
                                            app,
                                            &[],
                                            &bg_tasks.list_now(),
                                        );
                                    }
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
                        input_closed = true;
                        signal_turn_cancel(app, &mut cancelled);
                        if turn_cancel.is_none() {
                            break;
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    app.waiting_for = None;
    app.confirmation = None;
    app.confirmation_waiting = 0;
    // Never leave the agent blocked on a oneshot that the modal dropped.
    // Turn completion, input EOF, and plain Ctrl+C can exit while a confirm
    // is still outstanding — resolve it explicitly so tool code sees Cancelled
    // rather than a disconnected channel.
    if let Some(request) = pending_confirmation.take() {
        app.trace_approval_decided(
            crate::tui_event_trace::approval_kind(&request.request),
            "cancelled_on_turn_end",
        );
        let _ = request
            .response
            .send(hi_agent::ConfirmationResult::Cancelled);
    }
    while let Some(request) = confirm_queue.pop_front() {
        app.trace_approval_decided(
            crate::tui_event_trace::approval_kind(&request.request),
            "cancelled_on_turn_end",
        );
        let _ = request
            .response
            .send(hi_agent::ConfirmationResult::Cancelled);
    }
    // Reconcile the visible queue with mid-turn steering: drop entries the
    // agent injected only after a result that retained the turn transcript.
    // Errors and typed cancellation retain offered user work for the next turn.
    if let Some(inbox) = interject.as_ref() {
        let committed = value
            .as_ref()
            .is_some_and(|value| value.interjections_committed(cancelled));
        reconcile_queue_with_interjections(app, inbox, committed);
    } else {
        app.mid_turn_offered.clear();
    }
    if input_closed {
        anyhow::bail!(
            "terminal input reader stopped unexpectedly; the active operation was cancelled"
        );
    }
    Ok(DriveCompletion { cancelled, value })
}

fn x402_prompt_to_control(prompt: hi_ai::X402UserPrompt) -> ConfirmationControl {
    match prompt {
        hi_ai::X402UserPrompt::Confirm { quote, reply } => {
            let (response, answer) = tokio::sync::oneshot::channel();
            tokio::spawn(async move {
                let approved = match answer.await {
                    Ok(hi_agent::ConfirmationResult::Approved) => true,
                    Ok(hi_agent::ConfirmationResult::Answer(text)) => {
                        let text = text.trim().to_ascii_lowercase();
                        matches!(text.as_str(), "pay this quote" | "1" | "y" | "yes" | "pay")
                            || text.starts_with("pay ")
                    }
                    _ => false,
                };
                let _ = reply.send(approved);
            });
            ConfirmationControl {
                request: hi_agent::ConfirmationRequest::AskUser {
                    question: quote.prompt_text(),
                    options: vec!["Pay this quote".into(), "Cancel".into()],
                },
                response,
            }
        }
        hi_ai::X402UserPrompt::PasteSignature { reply } => {
            let (response, answer) = tokio::sync::oneshot::channel();
            tokio::spawn(async move {
                let signature = match answer.await {
                    Ok(hi_agent::ConfirmationResult::Answer(text)) => {
                        let text = text.trim().to_string();
                        (!text.is_empty()).then_some(text)
                    }
                    _ => None,
                };
                let _ = reply.send(signature);
            });
            ConfirmationControl {
                request: hi_agent::ConfirmationRequest::AskUser {
                    question: "Paste the Solana transaction signature for this x402 quote".into(),
                    options: Vec::new(),
                },
                response,
            }
        }
    }
}

#[cfg(test)]
mod cancellation_settlement_tests {
    use super::*;
    use ratatui::backend::TestBackend;

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
                    let mut outcome = hi_agent::TurnOutcome::infrastructure_failure(
                        "test-model",
                        None,
                        Vec::new(),
                    );
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
                    (status == hi_agent::TurnStatus::Cancelled)
                        .then_some("preserve the public API"),
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
}
