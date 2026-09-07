//! One interactive agent turn: drive, cancellation settlement, and auto-drive follow-up.

use std::io;
use std::path::Path;

use anyhow::Result;
use crossterm::event::Event;
use hi_agent::{Agent, AgentModelState};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::style::Style;
use ratatui::text::Line;
use tokio::sync::mpsc;

use crate::App;
use crate::event::ChannelUi;

use super::drive;

#[allow(clippy::too_many_arguments)]
pub(super) async fn run_agent_turn(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    input_rx: &mut mpsc::UnboundedReceiver<Event>,
    ticker: &mut tokio::time::Interval,
    app: &mut App,
    agent: &mut Agent,
    run_line: &str,
    mut restore_model_state: Option<AgentModelState>,
    mut restore_app_model: Option<(String, Option<u32>)>,
    loops_file: Option<&Path>,
) -> Result<()> {
    // --- Turn phase: run the agent behind a channel, staying responsive. ---
    // Flush any mode change made while another operation owned the Agent
    // before snapshotting mode or constructing the provider request.
    let drive_kind = hi_agent::DriveKind::from_prompt(run_line);
    app.push_session_face(agent);
    app.begin_plan_draft(agent.plan_mode());
    if agent.prepare_plan_drive_for_turn(drive_kind)? {
        // The transition must be visible before the future starts. Otherwise
        // a resumed user/plan turn works while the title bar still says paused.
        app.refresh_goal(agent);
    }
    let goal_drive_turn = drive_kind == hi_agent::DriveKind::Goal;
    let plan_drive_turn = drive_kind == hi_agent::DriveKind::Plan;
    // Agent::begin_drive_turn performs the matching Always→Auto demotion once
    // the future is first polled. Synchronize the frontend before `drive`
    // handles a confirmation so safe FileEdit requests are auto-approved.
    let drive_permission_restore =
        app.sync_synthetic_drive_permission(drive_kind, agent.permission_mode());
    let chrome = hi_agent::drive_chrome_line(
        run_line,
        agent.next_plan_step_title(),
        agent
            .structured_goal()
            .and_then(|goal| goal.active_sub_goal())
            .map(|step| step.description.as_str()),
    );
    app.push_user_prompt(ratatui::text::Line::styled(
        chrome.unwrap_or_else(|| format!("❯ {run_line}")),
        ratatui::style::Style::default().fg(crate::theme::theme().accent_user),
    ));
    app.set_working(true);
    app.follow();
    let checkpoint = agent.messages().len();
    app.last_turn_start = checkpoint;
    app.last_prompt = Some(run_line.to_string());
    // Long-horizon auto-drive bookkeeping: whether this is a synthetic drive
    // turn, and the goal state going in — any change by turn end (advance,
    // retry note, plan growth) counts as progress; no change is a stall.
    let goal_before = agent.structured_goal().cloned();
    let started_in_plan_mode = agent.plan_mode();
    let plan_step_before = agent.next_plan_step_title().map(str::to_owned);
    let turn_snapshot = agent.state_snapshot();
    app.last_turn_snapshot = Some(turn_snapshot.clone());
    app.trace_turn_started(agent, run_line)?;
    // Reset the per-turn tool-call counter for the observability panel.
    app.turn_tool_calls = 0;
    app.turn_rounds = 0;
    // Keep the batch interrupt paired with whole-turn cancellation. Esc and
    // Ctrl-C use the turn token so an awaited foreground process is killed and
    // reaped; the flag promptly wakes any cooperative batch boundary too.
    app.interrupt = Some(agent.interrupt_handle());
    let turn_cancel = hi_agent::TurnCancellation::new();
    let (tx, rx) = mpsc::unbounded_channel();
    let (confirm_tx, confirm_rx) = mpsc::unbounded_channel();
    let mut sink = ChannelUi {
        tx: tx.clone(),
        confirmations: confirm_tx,
        event_sink: app.event_sink.clone(),
        approval_store: app.approval_store.clone(),
    };
    // A line typed during an autonomous drive is explicit user work, so it
    // owns the next turn ahead of any newly synthesized continuation. Feeding
    // it into the active drive used to let the model consume it as an
    // interjection; queue reconciliation then removed the line and the TUI
    // immediately started another plan drive, with no user-origin turn ever
    // starting. Ordinary user turns retain live mid-turn steering.
    let interject = accepts_mid_turn_interjections(drive_kind).then(|| agent.interjection_inbox());
    let btw = agent.btw_dispatcher();
    let driven = {
        let bg_tasks = agent.background_task_registry();
        let fut = agent.run_turn_cancellable(run_line, &mut sink, turn_cancel.clone());
        drive(
            terminal,
            input_rx,
            ticker,
            app,
            rx,
            confirm_rx,
            fut,
            true,
            interject,
            Some(btw),
            tx,
            Some(turn_cancel.clone()),
            bg_tasks,
        )
        .await?
    };
    // The returned value or failure carries this turn's settled outcome.
    let stop_requested = driven.cancelled || turn_cancel.is_cancelled();
    let outcome = driven.value.as_ref().or(driven.failure.as_ref());
    let cancelled =
        outcome.is_some_and(|outcome| outcome.status == hi_agent::TurnStatus::Cancelled);
    if let Some(outcome) = outcome {
        app.note_turn_outcome(outcome);
    }
    if cancelled {
        // Keep the next-turn queue. `drive` already reconciled mid-turn
        // steers (consumed → removed; leftovers stay queued). Wiping the
        // backlog on interrupt was the main way a large prompt queue was
        // lost after stopping a stuck turn.
        app.mid_turn_offered.clear();
        let kept = app.queue.len();
        app.queue_paused = kept > 0;
        let stop_label = if driven.cancelled {
            "^C interrupted"
        } else {
            "turn deadline reached"
        };
        let msg = if kept > 0 {
            format!("{stop_label}; turn discarded ({kept} queued command(s) kept and paused)")
        } else {
            format!("{stop_label}; turn discarded")
        };
        app.push(Line::styled(
            msg,
            Style::default().fg(crate::theme::theme().warning),
        ));
        // Interrupting a drive turn is an explicit "stop": pause the goal so
        // the drive doesn't restart on the next message. Progress is held;
        // `/goal resume` continues.
        let pause_reason = if driven.cancelled {
            hi_agent::GoalPauseReason::User
        } else {
            hi_agent::GoalPauseReason::Infra
        };
        if goal_drive_turn && agent.set_goal_pause_reason(pause_reason) {
            let reason = if driven.cancelled { "user" } else { "deadline" };
            app.push(Line::styled(
                format!("goal drive interrupted — paused ({reason}); /goal resume to continue"),
                Style::default().fg(crate::theme::theme().warning),
            ));
        }
        if plan_drive_turn {
            agent.pause_plan_drive_until_user_input()?;
            app.refresh_goal(agent);
            app.push(Line::styled(
                "plan drive interrupted — paused; reply to steer and resume, or use /plan resume"
                    .to_string(),
                Style::default().fg(crate::theme::theme().warning),
            ));
        }
    } else {
        // The turn committed — ping if you've likely stepped away.
        app.maybe_notify_done();
        // Capture which files this turn changed, so the "changed: …" line
        // above the input reflects the latest turn. The agent already
        // computed this for verify gating; reuse it rather than re-walking.
        app.last_changed_files = agent.last_changed_files().to_vec();
        app.accumulate_session_files();
        // Capture the turn's trajectory telemetry for the observability
        // panel (verify rounds, recovery retries, nudges, stalls).
        app.last_telemetry = Some(agent.last_turn_telemetry().clone());
        app.last_turn_phase = Some(agent.turn_phase().label());
        // A new turn's edits supersede any open diff panel's snapshot.
        app.diff_text = None;
    }
    if stop_requested && !cancelled {
        // The body either committed or returned its original error just
        // before Ctrl-C/deadline won the frontend race. Preserve that
        // settlement, but honor "stop" by preventing autonomous drive.
        let stop_label = if driven.cancelled {
            "interrupt"
        } else {
            "deadline"
        };
        let settlement = if driven.value.is_some() {
            "arrived after the turn committed; result kept"
        } else {
            "reached while the turn error settled; failure kept"
        };
        app.push(Line::styled(
            format!("{stop_label} {settlement}"),
            Style::default().fg(crate::theme::theme().warning),
        ));
        let pause_reason = if driven.cancelled {
            hi_agent::GoalPauseReason::User
        } else {
            hi_agent::GoalPauseReason::Infra
        };
        if goal_drive_turn && agent.set_goal_pause_reason(pause_reason) {
            app.push(Line::styled(
                "goal drive paused; /goal resume to continue".to_string(),
                Style::default().fg(crate::theme::theme().warning),
            ));
        }
        if plan_drive_turn {
            agent.pause_plan_drive_until_user_input()?;
            app.refresh_goal(agent);
            app.push(Line::styled(
                "plan drive paused; reply to steer and resume, or use /plan resume".to_string(),
                Style::default().fg(crate::theme::theme().warning),
            ));
        }
    }
    if stop_requested && !app.queue.is_empty() {
        app.queue_paused = true;
    }
    if let Some(state) = restore_model_state.take() {
        agent.restore_model_state(state);
    }
    if let Some((model, context_window)) = restore_app_model.take() {
        app.model = model;
        app.context_window = context_window;
    }
    // The agent's reasoning effort may have changed during the turn
    // (e.g. repair escalation) — mirror it back for the title bar.
    app.reasoning_effort = agent.reasoning_effort();
    // The goal driver (`goal_turn_end`) may have advanced/failed a sub-goal
    // this turn — mirror the new state so the pinned block + header reflect it.
    // Restore the transient synthetic-drive face only when the user did not
    // make a mid-turn Shift-Tab choice. Push a real choice before refresh so
    // the Agent's own drive restoration cannot clobber it.
    app.restore_synthetic_drive_permission(drive_permission_restore);
    app.push_session_face(agent);
    app.refresh_goal(agent);
    // Record a main /goal that just reached a terminal state to the activity
    // feed (→ /digest), so the interactive autonomous producer joins loops +
    // fleet there instead of being the one hole.
    if let Some(before) = &goal_before
        && before.status == hi_agent::GoalStatus::Active
        && let Some(after) = agent.structured_goal()
        && matches!(
            after.status,
            hi_agent::GoalStatus::Done | hi_agent::GoalStatus::Failed
        )
        && let Some(lf) = loops_file
    {
        let verb = if after.status == hi_agent::GoalStatus::Done {
            "goal complete"
        } else {
            "goal failed"
        };
        let at_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        crate::activity::append(
            &crate::activity::activity_path(lf),
            &crate::activity::ActivityEntry {
                at_ms,
                loop_id: 0,
                source: "goal".into(),
                text: format!("{verb}: {}", after.objective),
                event_id: None,
                group_key: None,
                state: None,
                detail: None,
            },
        );
    }
    // Long-horizon auto-drive: keep pulling toward leftover work between
    // turns. Drive turns that change nothing count toward a stall park; any
    // user turn resets it. Queued user input always wins — the drive prompt
    // is only queued into an empty queue.
    if !cancelled && !stop_requested {
        if goal_drive_turn {
            let made_progress = agent.goal_drive_turn_made_progress(goal_before.as_ref());
            let progress = agent.note_goal_drive_progress(made_progress);
            match progress {
                hi_agent::GoalDriveProgress::Skipped { failed, next } => {
                    app.push(Line::styled(
                        hi_agent::goal_drive_skip_message(&failed, next.as_deref()),
                        Style::default().fg(crate::theme::theme().warning),
                    ));
                }
                hi_agent::GoalDriveProgress::Parked => {
                    app.push(Line::styled(
                        hi_agent::goal_drive_park_message(agent.leftover_work().as_deref()),
                        Style::default().fg(crate::theme::theme().warning),
                    ));
                }
                _ => {}
            }
        }
        if let Some(count) = agent.take_goal_requeue_notice() {
            app.push(Line::styled(
                hi_agent::goal_drive_requeue_message(count),
                Style::default().fg(crate::theme::theme().warning),
            ));
        }
        if plan_drive_turn {
            let made_progress = agent.plan_drive_turn_made_progress(plan_step_before.as_deref());
            agent.note_plan_drive_progress(made_progress);
            if agent.plan_drive_status() == "parked" {
                app.push(Line::styled(
                    hi_agent::plan_drive_park_message(agent.plan_leftover_work().as_deref()),
                    Style::default().fg(crate::theme::theme().warning),
                ));
            }
        }
        app.refresh_goal(agent);
        app.finish_plan_draft(started_in_plan_mode, driven.value.as_ref());
        app.push_session_face(agent);
        app.maybe_queue_drive(agent, driven.value.as_ref());
    }
    app.trace_turn_settled(agent, outcome)?;
    if cancelled && !app.exit_requested {
        // Start the escalation window after cleanup, not at the original key
        // press, so a second Ctrl-C queued during slow settlement still exits.
        app.quit_notice = Some(std::time::Instant::now() + std::time::Duration::from_millis(1800));
    }
    app.set_working(false);
    // Flush any pending live events from the TUI's /sync on RemoteUi.
    // Spawn as a background task so a slow/unreachable ipop doesn't block
    // the TUI event loop (5s timeout). Errors are silent — events are
    // re-buffered on failure and retried on the next flush.
    if let Some(rui) = &app.sync_remote_ui {
        let rui = rui.clone();
        tokio::spawn(async move {
            let _ = rui.flush().await;
        });
    }
    // Flush the startup RemoteUi (created in main.rs) so live events are
    // actually streamed during the session, not just buffered until exit.
    if let Some(cb) = &app.remote_flush_callback {
        cb();
    }
    // No follow() at turn end: if the user scrolled up to read mid-turn, leave
    // them there (the "↓ N new" hint shows the summary is below). A new turn
    // re-pins to the bottom.
    Ok(())
}

fn accepts_mid_turn_interjections(drive_kind: hi_agent::DriveKind) -> bool {
    !drive_kind.is_drive()
}

#[cfg(test)]
mod tests {
    use super::accepts_mid_turn_interjections;

    #[test]
    fn synthetic_drives_reserve_mid_turn_input_for_a_user_turn() {
        assert!(accepts_mid_turn_interjections(hi_agent::DriveKind::User));
        assert!(!accepts_mid_turn_interjections(hi_agent::DriveKind::Plan));
        assert!(!accepts_mid_turn_interjections(hi_agent::DriveKind::Goal));
    }
}
