//! `/dashboard` — control a fleet, not an agent.
//!
//! A full-screen mode over the same terminal: a table of dispatched agents
//! (one row each), a dispatch box that always spawns a *new* session on Enter,
//! and a peek panel for the selected row — its latest output plus a live reply
//! input, so you can answer an agent's question with a single keystroke
//! (`1`–`9`) or queue a follow-up without opening the full conversation.
//! `Ctrl+S` dispatches *and* attaches (a full-screen focus view of that row).
//!
//! Isolation: every row gets its **own git worktree**, checked out to a
//! snapshot of your tree at dispatch (uncommitted work included). Each turn is
//! a child `hi` run *in that worktree*, resuming the row's own session file.
//! On a successful turn the row's diff is **auto-merged** back into your real
//! tree — gated by the session verify (when set) and held visibly when it
//! overlaps another row's files (`m` forces). Failed or abandoned rows never
//! touch your tree, and their sessions stay resumable with `--resume`.

use std::collections::VecDeque;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::process::Stdio;
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use crossterm::event::{Event, KeyCode, KeyEventKind, KeyModifiers};
use futures_util::StreamExt;
use futures_util::stream::FuturesUnordered;
use hi_tools::worktree;
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Paragraph};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::{mpsc, oneshot};

use crate::dashboard_goal::{RowGoal, next_drive_stall, parse_report, should_retry_goal_turn};
use crate::input::InputLine;
use crate::render::dim;
use crate::{App, FleetLauncher, SPINNER};

/// Lines of output kept per row for the peek/attach panels.
const TAIL_CAP: usize = 200;
/// Max table rows shown before the list scrolls with the selection.
const TABLE_ROWS: usize = 8;

/// A dispatched fleet agent: one row, one worktree, one session on disk.
pub(crate) struct FleetRow {
    /// Display id (stable, 1-based, never reused within a session).
    pub(crate) id: usize,
    /// The dispatch prompt (shown truncated as the row title).
    pub(crate) title: String,
    /// The row's isolated git worktree (every turn runs in here).
    pub(crate) worktree: PathBuf,
    /// The snapshot commit the worktree branched from (diff/merge base).
    pub(crate) base: String,
    /// The row's session file (parent-owned; child appends via --session-file).
    pub(crate) session: PathBuf,
    pub(crate) state: RowState,
    /// What the merge gate concluded after the last completed turn.
    pub(crate) merge: MergeState,
    /// Files changed vs the base, from the last merge check.
    pub(crate) changed: Vec<String>,
    /// Live activity lead while working (last output line).
    pub(crate) activity: String,
    /// Recent output lines (peek/attach panel body).
    pub(crate) tail: Vec<String>,
    /// Follow-ups typed while a turn was running; dispatched FIFO on idle.
    pub(crate) pending: VecDeque<String>,
    /// The per-row reply input (peek panel).
    pub(crate) reply: InputLine,
    /// Kills the in-flight child turn when fired.
    pub(crate) kill: Option<oneshot::Sender<()>>,
    /// Current turn start (for the elapsed column).
    pub(crate) started: Option<Instant>,
    pub(crate) turns: u32,
    /// Session-cumulative tokens, from the child's per-turn report.
    pub(crate) usage: u64,
    /// Long-horizon goal progress, from the report's `goal` block.
    pub(crate) goal: Option<RowGoal>,
    /// Objective for a `/goal` dispatch — consumed by the row's *first* turn
    /// (the child plans it via `--goal`; later turns drive the session's goal).
    pub(crate) goal_objective: Option<String>,
    /// The last report's raw goal JSON, for drive-stall comparison.
    pub(crate) last_goal_json: Option<String>,
    /// Whether the in-flight turn is a synthetic drive turn (not user input).
    pub(crate) driving: bool,
    /// Consecutive drive turns with an unchanged goal — parks the drive at
    /// [`hi_agent::GOAL_DRIVE_STALL_LIMIT`]; any user reply resets it.
    pub(crate) drive_stall: u32,
    /// The real tree has advanced (another row merged) since this row's base.
    pub(crate) stale: bool,
    /// The row is waiting on the user (question, held merge, failure, parked
    /// drive) — badge + ping.
    pub(crate) attention: bool,
    /// When this row was spawned by a workflow `SpawnAgent` request, this holds
    /// the reply sender the engine is waiting on. When the child turn completes,
    /// `finish_turn` sends the `AgentResult` back so the workflow can continue.
    /// `None` for rows dispatched directly from the dashboard dispatch box.
    pub(crate) workflow_reply:
        Option<oneshot::Sender<Result<hi_workflow::AgentResult, hi_workflow::HostError>>>,
    /// The phase this row's agent belongs to (from `AgentOpts.phase`), used to
    /// group rows under phase headers in the workflow run view.
    pub(crate) workflow_phase: Option<String>,
    /// Stable label assigned by the workflow, used in the fleet/detail views.
    pub(crate) workflow_label: Option<String>,
    /// Typed workflow child state, independent of the generic row state.
    pub(crate) workflow_status: Option<WorkflowJobStatus>,
    /// The workflow requested `output_schema`-shaped output: the reply's
    /// `assistant_response` is parsed back into JSON before it reaches the
    /// engine (fleet children only produce text).
    pub(crate) workflow_expects_json: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WorkflowJobStatus {
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// Goal progress mirrored from the child's report.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum RowState {
    /// A child turn (or its merge check) is in flight.
    Working,
    Idle,
    Failed,
    /// Closed by the user: worktree cleaned up, row kept for reference.
    Closed,
}

#[derive(Clone, PartialEq, Eq)]
pub(crate) enum MergeState {
    /// No completed turn yet, or the turn changed nothing.
    None,
    /// The row's diff has been applied to the real tree.
    Merged(usize),
    /// Diff ready but overlaps other rows' files — `m` forces it.
    Held(Vec<usize>),
    /// The verify gate failed in the worktree — not merged (`m` forces).
    VerifyFailed,
}

/// An active workflow run inside the dashboard. The engine runs in a
/// `spawn_blocking` thread; host requests arrive on `host_rx` and are
/// serviced by the dashboard's `select!` loop. When the engine finishes,
/// `join_handle` resolves with the `WorkflowOutcome`.
pub(crate) struct WorkflowRun {
    /// The workflow's stable persisted run ID.
    pub(crate) run_id: String,
    /// The workflow's display name (from `WorkflowMeta.name`).
    pub(crate) name: String,
    /// The objective/description shown in the dashboard header.
    pub(crate) objective: String,
    /// The declared phases (from `WorkflowMeta.phases`), with their current
    /// state: `"active"`, `"done"`, or `"pending"`.
    pub(crate) phases: Vec<(String, String)>,
    /// The index of the currently active phase, or `None` before the first
    /// `phase()` call.
    pub(crate) current_phase: Option<usize>,
    /// Receiver for host requests from the engine thread. Taken out of the
    /// run and polled directly in the dashboard's `select!` loop to avoid
    /// double-borrowing `app.workflow_run`.
    pub(crate) host_rx: Option<mpsc::UnboundedReceiver<hi_workflow::WorkflowHostRequest>>,
    /// The join handle for the engine thread — resolves with the outcome.
    pub(crate) join_handle: Option<tokio::task::JoinHandle<hi_workflow::WorkflowOutcome>>,
    /// The cancellation token — firing it cancels the workflow.
    pub(crate) cancel: tokio_util::sync::CancellationToken,
    /// The outcome once the engine thread has joined.
    pub(crate) outcome: Option<hi_workflow::WorkflowOutcome>,
    /// Log lines emitted by the workflow (phase markers, log messages).
    pub(crate) log: Vec<String>,
    /// Total agent invocations admitted for this run.
    pub(crate) agent_budget: u64,
    /// Agent invocations that have completed and been charged.
    pub(crate) agent_spent: u64,
    /// Agent invocations reserved by an upcoming `parallel()` call.
    pub(crate) agent_reserved: u64,
}

impl WorkflowRun {
    /// Update the phase trail when a `Phase` host request arrives.
    fn on_phase(&mut self, title: &str) {
        // Mark the previous active phase as done.
        if let Some(idx) = self.current_phase {
            if idx < self.phases.len() {
                self.phases[idx].1 = "done".into();
            }
        }
        // Find or add the new phase.
        if let Some(idx) = self.phases.iter().position(|(t, _)| t == title) {
            self.phases[idx].1 = "active".into();
            self.current_phase = Some(idx);
        } else {
            self.phases.push((title.into(), "active".into()));
            self.current_phase = Some(self.phases.len() - 1);
        }
        self.log.push(format!("phase: {title}"));
    }

}

impl FleetRow {
    fn push_line(&mut self, line: String) {
        self.tail.push(line);
        if self.tail.len() > TAIL_CAP {
            let drop = self.tail.len() - TAIL_CAP;
            self.tail.drain(..drop);
        }
    }

    /// Ingest one child output line: strip ANSI, keep the tail + activity lead.
    fn push_output(&mut self, raw: &str) {
        let line = strip_ansi(raw);
        let line = line.trim_end();
        if line.trim().is_empty() {
            return;
        }
        self.activity = truncate(line.trim_start(), 64);
        self.push_line(line.to_string());
    }
}

/// What a completed in-flight future reports back.
pub(crate) enum RowDone {
    /// The child turn exited.
    Turn { ok: bool, killed: bool },
    /// The off-thread merge check finished (diff vs base + verify verdict).
    MergeCheck {
        changed: Vec<String>,
        verified: bool,
    },
    /// Post-merge: combined-tree verify verdict (None = no verify configured)
    /// + the refreshed base the worktree was reset onto (None = refresh failed).
    PostVerify {
        verify_ok: Option<bool>,
        new_base: Option<String>,
    },
}

pub(crate) type RowFut = Pin<Box<dyn Future<Output = (usize, RowDone)>>>;

/// Which input owns keystrokes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Focus {
    /// The bottom dispatch box (default) — Enter spawns a new agent.
    Dispatch,
    /// The selected row's reply input (peek panel).
    Reply,
    /// Full-screen view of the selected row (bigger tail + reply input).
    Attach,
}

/// Run the fleet dashboard until the user leaves it. Rows persist on
/// `app.fleet` across open/close. Leaving with turns in flight requires a
/// second Esc and kills the children (their sessions stay resumable).
pub(crate) async fn run_dashboard(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    input_rx: &mut mpsc::UnboundedReceiver<Event>,
    ticker: &mut tokio::time::Interval,
    app: &mut App,
    launcher: &FleetLauncher,
    adopt: Option<crate::FleetResumeInfo>,
) -> Result<()> {
    let (line_tx, mut line_rx) = mpsc::unbounded_channel::<(usize, String)>();
    let mut in_flight: FuturesUnordered<RowFut> = FuturesUnordered::new();
    let mut selected: usize = app.fleet.len().saturating_sub(1);
    // `/fleet resume [id]`: re-adopt a past session as a row before the loop
    // starts (needs the loop's channels for its first drive turn).
    let mut adopt_flash: Option<String> = None;
    if let Some(info) = adopt {
        match adopt_session(app, info, launcher, &line_tx, &mut in_flight).await {
            Ok(idx) => selected = idx,
            Err(err) => adopt_flash = Some(format!("resume failed: {err:#}")),
        }
    }
    let mut focus = Focus::Dispatch;
    let mut dispatch = InputLine::default();
    let mut exit_armed = false;
    let mut flash: Option<String> = adopt_flash.take();
    // Peek scrollback: lines back from the live tail (0 = follow).
    let mut peek_offset: usize = 0;
    // Take the workflow join handle out of app.workflow_run so the select!
    // loop can poll it without double-borrowing app.workflow_run (the host_rx
    // branch also needs app.workflow_run.as_mut()).
    let mut wf_join_handle: Option<tokio::task::JoinHandle<hi_workflow::WorkflowOutcome>> = None;
    if let Some(run) = app.workflow_run.as_mut() {
        wf_join_handle = run.join_handle.take();
    }

    loop {
        terminal.draw(|f| {
            render_dashboard(
                f,
                app,
                selected,
                focus,
                &dispatch,
                in_flight.len(),
                exit_armed,
                flash.as_deref(),
                peek_offset,
            )
        })?;

        tokio::select! {
            Some((idx, done)) = in_flight.next(), if !in_flight.is_empty() => {
                match done {
                    RowDone::Turn { ok, killed } => {
                        finish_turn(app, idx, ok, killed, launcher, &line_tx, &mut in_flight);
                    }
                    RowDone::MergeCheck { changed, verified } => {
                        finish_merge_check(app, idx, changed, verified, launcher, &line_tx, &mut in_flight);
                    }
                    RowDone::PostVerify { verify_ok, new_base } => {
                        finish_post_verify(app, idx, verify_ok, new_base, launcher, &line_tx, &mut in_flight);
                    }
                }
            }
            Some((idx, line)) = line_rx.recv() => {
                if let Some(row) = app.fleet.get_mut(idx) {
                    row.push_output(&line);
                }
                // Drain the burst so a chatty child can't starve the render loop.
                while let Ok((idx, line)) = line_rx.try_recv() {
                    if let Some(row) = app.fleet.get_mut(idx) {
                        row.push_output(&line);
                    }
                }
            }
            _ = ticker.tick() => {
                app.spinner = app.spinner.wrapping_add(1);
            }
            // Workflow host request arrived — service it (SpawnAgent creates a
            // real FleetRow, Phase/Log update the run state, etc.).
            Some(req) = async {
                if let Some(run) = app.workflow_run.as_mut()
                    && let Some(rx) = run.host_rx.as_mut()
                {
                    rx.recv().await
                } else {
                    std::future::pending().await
                }
            }, if app.workflow_run.as_ref().is_some_and(|r| r.host_rx.is_some()) => {
                handle_workflow_host_request(app, req, launcher, &line_tx, &mut in_flight).await;
            }
            // Workflow engine thread finished — await the outcome.
            outcome = async {
                if let Some(handle) = wf_join_handle.as_mut() {
                    match handle.await {
                        Ok(outcome) => Some(outcome),
                        Err(_) => Some(hi_workflow::WorkflowOutcome::Failed {
                            error: "workflow engine thread panicked".into(),
                        }),
                    }
                } else {
                    std::future::pending().await
                }
            }, if wf_join_handle.is_some() => {
                wf_join_handle = None;
                if let Some(outcome) = outcome
                    && let Some(run) = app.workflow_run.as_mut()
                {
                    run.outcome = Some(outcome.clone());
                    if let Some(root) = std::env::var_os("XDG_STATE_HOME")
                        .map(std::path::PathBuf::from)
                        .or_else(|| std::env::var_os("HOME").map(|home| std::path::PathBuf::from(home).join(".local/state")))
                    {
                        let store = hi_workflow::WorkflowRunStore::new(root.join("hi/workflow-runs"));
                        if let Ok(mut stored) = store.load(&run.run_id).map(|stored| stored.manifest) {
                            stored.finish(outcome.clone());
                            let _ = store.persist(&stored);
                        }
                    }
                    flash = Some(format!("workflow {}: {}", run.name, workflow_outcome_summary(&run.outcome.as_ref().unwrap())));
                }
            }
            maybe = input_rx.recv() => {
                let Some(event) = maybe else { return Ok(()) };
                match event {
                    Event::Paste(text) => {
                        flash = None;
                        if let Some(input) = focused_input(app, selected, focus, &mut dispatch) { input.insert_str(&text) }
                    }
                    Event::Key(key) if key.kind == KeyEventKind::Press => {
                        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                        flash = None;
                        if !matches!(key.code, KeyCode::Esc) {
                            exit_armed = false;
                        }
                        match key.code {
                            KeyCode::Esc | KeyCode::Char('c') if matches!(key.code, KeyCode::Esc) || ctrl => {
                                if focus != Focus::Dispatch {
                                    focus = Focus::Dispatch;
                                    exit_armed = false;
                                    continue;
                                }
                                if in_flight.is_empty() {
                                    return Ok(());
                                }
                                if exit_armed {
                                    if let Some(run) = &app.workflow_run {
                                        run.cancel.cancel();
                                    }
                                    // Kill in-flight children (kill_on_drop backs
                                    // this up when the futures drop with the
                                    // loop); the sessions stay resumable.
                                    for row in app.fleet.iter_mut() {
                                        if row.state == RowState::Working {
                                            if let Some(kill) = row.kill.take() {
                                                let _ = kill.send(());
                                            }
                                            row.state = RowState::Failed;
                                            row.started = None;
                                            row.activity.clear();
                                            row.push_line(
                                                "⚠ killed on exit — session remains resumable"
                                                    .to_string(),
                                            );
                                        }
                                    }
                                    return Ok(());
                                }
                                exit_armed = true;
                            }
                            KeyCode::Up => {
                                selected = selected.saturating_sub(1);
                                peek_offset = 0;
                            }
                            KeyCode::Down => {
                                if !app.fleet.is_empty() {
                                    selected = (selected + 1).min(app.fleet.len() - 1);
                                }
                                peek_offset = 0;
                            }
                            KeyCode::Tab => {
                                focus = match focus {
                                    Focus::Dispatch if !app.fleet.is_empty() => Focus::Reply,
                                    _ => Focus::Dispatch,
                                };
                                // Focusing a row's reply acknowledges it.
                                if focus != Focus::Dispatch
                                    && let Some(row) = app.fleet.get_mut(selected)
                                {
                                    row.attention = false;
                                }
                            }
                            // Peek scrollback through the row's output tail.
                            KeyCode::PageUp => {
                                if let Some(row) = app.fleet.get(selected) {
                                    peek_offset =
                                        (peek_offset + 10).min(row.tail.len().saturating_sub(1));
                                }
                            }
                            KeyCode::PageDown => {
                                peek_offset = peek_offset.saturating_sub(10);
                            }
                            // r: rebase an idle row's worktree onto a fresh
                            // snapshot of the real tree (clears the stale badge).
                            KeyCode::Char('r')
                                if focus == Focus::Dispatch
                                    && app.fleet.get(selected).is_some_and(|r| {
                                        r.state != RowState::Working && r.state != RowState::Closed
                                    }) =>
                            {
                                let base =
                                    hi_tools::checkpoint::create(std::path::Path::new(".")).await;
                                rebase_row(app, selected, base, &mut flash);
                            }
                            // Ctrl+S: dispatch AND attach (or attach the selected
                            // row when the dispatch box is empty).
                            KeyCode::Char('s') if ctrl => {
                                let text = dispatch.submit();
                                let text = text.trim().to_string();
                                if !text.is_empty() {
                                    match dispatch_new(app, text, launcher, &line_tx, &mut in_flight).await {
                                        Ok(idx) => {
                                            selected = idx;
                                            focus = Focus::Attach;
                                        }
                                        Err(err) => flash = Some(format!("dispatch failed: {err:#}")),
                                    }
                                } else if !app.fleet.is_empty() {
                                    focus = Focus::Attach;
                                }
                            }
                            KeyCode::Enter => match focus {
                                Focus::Dispatch => {
                                    let text = dispatch.submit();
                                    let text = text.trim().to_string();
                                    if !text.is_empty() {
                                        match dispatch_new(app, text, launcher, &line_tx, &mut in_flight).await {
                                            Ok(idx) => selected = idx,
                                            Err(err) => {
                                                flash = Some(format!("dispatch failed: {err:#}"))
                                            }
                                        }
                                    }
                                }
                                Focus::Reply | Focus::Attach => {
                                    if let Some(row) = app.fleet.get_mut(selected) {
                                        let text = row.reply.submit().trim().to_string();
                                        if !text.is_empty() {
                                            peek_offset = 0;
                                            send_reply(app, selected, text, launcher, &line_tx, &mut in_flight);
                                        }
                                    }
                                }
                            },
                            // Single-keystroke answer: on an idle row with an
                            // empty reply box, 1–9 replies with that digit —
                            // enough to answer "1) do X or 2) do Y?" instantly.
                            KeyCode::Char(c @ '1'..='9')
                                if focus != Focus::Dispatch
                                    && app
                                        .fleet
                                        .get(selected)
                                        .is_some_and(|r| r.reply.is_empty() && r.state != RowState::Working) =>
                            {
                                send_reply(app, selected, c.to_string(), launcher, &line_tx, &mut in_flight);
                            }
                            // m: force-merge the selected row's diff (held or
                            // verify-failed) into the real tree.
                            KeyCode::Char('m')
                                if focus == Focus::Dispatch
                                    && app.fleet.get(selected).is_some_and(|r| {
                                        r.state != RowState::Working && r.state != RowState::Closed
                                    }) =>
                            {
                                force_merge(app, selected, &mut flash);
                            }
                            // x: close an idle/failed row — clean its worktree
                            // up; the session file stays resumable.
                            KeyCode::Char('x')
                                if focus == Focus::Dispatch
                                    && app.fleet.get(selected).is_some_and(|r| {
                                        r.state != RowState::Working && r.state != RowState::Closed
                                    }) =>
                            {
                                if let Some(row) = app.fleet.get_mut(selected) {
                                    worktree::cleanup(
                                        &app.workspace_root,
                                        std::slice::from_ref(&row.worktree),
                                    );
                                    row.state = RowState::Closed;
                                    row.activity.clear();
                                    row.push_line(
                                        "row closed — worktree removed; session remains resumable"
                                            .to_string(),
                                    );
                                }
                            }
                            // Ctrl+K: kill the selected row's in-flight turn.
                            KeyCode::Char('k') if ctrl => {
                                if let Some(row) = app.fleet.get_mut(selected)
                                    && row.state == RowState::Working
                                    && let Some(kill) = row.kill.take()
                                {
                                    let _ = kill.send(());
                                }
                            }
                            KeyCode::Char('u') if ctrl => {
                                focused_input(app, selected, focus, &mut dispatch).map(InputLine::kill_to_start);
                            }
                            KeyCode::Char('a') if ctrl => {
                                focused_input(app, selected, focus, &mut dispatch).map(InputLine::home);
                            }
                            KeyCode::Char('e') if ctrl => {
                                focused_input(app, selected, focus, &mut dispatch).map(InputLine::end);
                            }
                            KeyCode::Char(c) if !ctrl => {
                                if let Some(input) = focused_input(app, selected, focus, &mut dispatch) { input.insert(c) }
                            }
                            KeyCode::Backspace => {
                                focused_input(app, selected, focus, &mut dispatch).map(InputLine::backspace);
                            }
                            KeyCode::Left => {
                                focused_input(app, selected, focus, &mut dispatch).map(InputLine::left);
                            }
                            KeyCode::Right => {
                                focused_input(app, selected, focus, &mut dispatch).map(InputLine::right);
                            }
                            KeyCode::Home => {
                                focused_input(app, selected, focus, &mut dispatch).map(InputLine::home);
                            }
                            KeyCode::End => {
                                focused_input(app, selected, focus, &mut dispatch).map(InputLine::end);
                            }
                            _ => {}
                        }
                    }
                    // Keep focus state live so attention pings fire only when
                    // you're actually away.
                    Event::FocusGained => app.set_focus(true),
                    Event::FocusLost => app.set_focus(false),
                    _ => {}
                }
            }
        }
    }
}

/// Remove every remaining fleet worktree (called at TUI shutdown).
pub(crate) fn cleanup_fleet(app: &mut App) {
    // Cancel any active workflow run so the engine thread stops.
    if let Some(run) = &app.workflow_run {
        if run.outcome.is_none() {
            run.cancel.cancel();
        }
    }
    let paths: Vec<PathBuf> = app
        .fleet
        .iter()
        .filter(|r| r.state != RowState::Closed)
        .map(|r| r.worktree.clone())
        .collect();
    if !paths.is_empty() {
        worktree::cleanup(&app.workspace_root, &paths);
    }
}

/// The input that currently owns typed characters.
fn focused_input<'a>(
    app: &'a mut App,
    selected: usize,
    focus: Focus,
    dispatch: &'a mut InputLine,
) -> Option<&'a mut InputLine> {
    match focus {
        Focus::Dispatch => Some(dispatch),
        Focus::Reply | Focus::Attach => app.fleet.get_mut(selected).map(|r| &mut r.reply),
    }
}

/// Create a new row: snapshot the tree, add its worktree, allocate its session
/// file, and start the first turn. Returns the new row's index.
async fn dispatch_new(
    app: &mut App,
    prompt: String,
    launcher: &FleetLauncher,
    line_tx: &mpsc::UnboundedSender<(usize, String)>,
    in_flight: &mut FuturesUnordered<RowFut>,
) -> Result<usize> {
    if !worktree::in_git_repo(&app.workspace_root) {
        return Err(anyhow!(
            "not in a git repository (fleet rows need worktrees)"
        ));
    }
    // A `/goal <objective>` dispatch makes the row goal-driven: the child
    // plans the objective (via --goal) and the parent auto-continues while
    // the goal stays active.
    let (goal_objective, prompt) = split_goal_dispatch(prompt);
    let title = goal_objective
        .as_ref()
        .cloned()
        .unwrap_or_else(|| prompt.clone());
    let first_prompt = if goal_objective.is_some() {
        hi_agent::GOAL_CONTINUE_PROMPT.to_string()
    } else {
        prompt
    };
    // Snapshot the current tree (incl. uncommitted work) as the row's base.
    let base = hi_tools::checkpoint::create(&app.workspace_root)
        .await
        .context("couldn't snapshot the working tree")?;
    app.fleet_next_id += 1;
    let id = app.fleet_next_id;
    let path = worktree::worktree_path("fleet", id as u32);
    worktree::add_worktree(&app.workspace_root, &path, &base)?;
    let session = (launcher.session_path)()?;
    let row = FleetRow {
        id,
        title,
        worktree: path,
        base,
        session,
        state: RowState::Idle,
        merge: MergeState::None,
        changed: Vec::new(),
        activity: String::new(),
        tail: Vec::new(),
        pending: VecDeque::new(),
        reply: InputLine::default(),
        kill: None,
        started: None,
        turns: 0,
        usage: 0,
        goal: None,
        goal_objective,
        last_goal_json: None,
        driving: false,
        drive_stall: 0,
        stale: false,
        attention: false,
        workflow_reply: None,
        workflow_phase: None,
        workflow_label: None,
        workflow_status: None,
        workflow_expects_json: false,
    };
    app.fleet.push(row);
    let idx = app.fleet.len() - 1;
    start_turn(app, idx, first_prompt, launcher, line_tx, in_flight);
    Ok(idx)
}

fn collect_workflow_phases(steps: &[hi_workflow::DeclarativeStep], phases: &mut Vec<String>) {
    for step in steps {
        match step {
            hi_workflow::DeclarativeStep::Phase { title } if !phases.contains(title) => phases.push(title.clone()),
            hi_workflow::DeclarativeStep::IfAgentSuccess { then_steps, else_steps, .. } => {
                collect_workflow_phases(then_steps, phases);
                collect_workflow_phases(else_steps, phases);
            }
            _ => {}
        }
    }
}

/// Launch a workflow run inside the dashboard. The engine runs in a
/// `spawn_blocking` thread; host requests arrive on the returned channel and
/// are serviced by the dashboard's `select!` loop. `SpawnAgent` requests create
/// real `FleetRow`s with worktree-isolated child `hi` turns — the full host
/// bridge, not a stub.
pub(crate) async fn start_workflow_run(
    app: &mut App,
    script: String,
    args: serde_json::Value,
) -> Result<()> {
    use hi_workflow::{DeclarativeRunParams, DeclarativeWorkflow};

    // Declarative `.workflow.json` definitions and Rhai scripts both run here:
    // they speak the same `WorkflowHostRequest` channel, so the dashboard's
    // host bridge (fleet rows, phases, budget, scratch files) serves either.
    let declarative = if script.trim_start().starts_with('{') {
        let definition = DeclarativeWorkflow::from_json(&script)
            .context("invalid declarative .workflow.json definition")?;
        definition.validate().map_err(|e| anyhow::anyhow!(e.to_string()))?;
        Some(definition)
    } else {
        None
    };
    let (workflow_name, workflow_description, phase_names) = match &declarative {
        Some(definition) => {
            let mut names = Vec::new();
            collect_workflow_phases(&definition.steps, &mut names);
            (
                definition.metadata.name.clone(),
                definition.metadata.description.clone(),
                names,
            )
        }
        None => {
            let meta = hi_workflow::extract_meta(&script)
                .map_err(|e| anyhow::anyhow!("invalid workflow script: {e}"))?;
            let names = meta.phases.iter().map(|p| p.title.clone()).collect();
            (meta.name, meta.description, names)
        }
    };
    let phases: Vec<(String, String)> = phase_names
        .into_iter()
        .map(|title| (title, "pending".to_string()))
        .collect();

    let (host_tx, host_rx) = mpsc::unbounded_channel::<hi_workflow::WorkflowHostRequest>();
    let cancel = tokio_util::sync::CancellationToken::new();
    let journal_path = std::env::var_os("XDG_STATE_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| std::path::PathBuf::from(home).join(".local/state")))
        .map(|base| base.join("hi/workflow-runs"));
    let run_id = format!(
        "run-{}-{}",
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis(),
        std::process::id(),
    );
    let journal = if let Some(root) = journal_path {
        let store = hi_workflow::WorkflowRunStore::new(root);
        let manifest = hi_workflow::WorkflowRunManifest::new(
            run_id.clone(), workflow_name.clone(), hi_workflow::DEFAULT_AGENT_BUDGET,
        )?;
        store.register(&manifest, &script, &args)?;
        hi_workflow::Journal::load(store.journal_path(&run_id)?)?
    } else {
        hi_workflow::Journal::new(None)
    };

    let join_handle = match declarative {
        Some(definition) => {
            let params = DeclarativeRunParams {
                workflow: definition,
                args,
                host_tx,
                cancel: cancel.clone(),
            };
            tokio::spawn(async move {
                match hi_workflow::run_declarative_workflow(params).await {
                    hi_workflow::DeclarativeOutcome::Completed { result, .. } => hi_workflow::WorkflowOutcome::Completed { result },
                    hi_workflow::DeclarativeOutcome::Paused { kind, message, .. } => hi_workflow::WorkflowOutcome::Paused { kind, message },
                    hi_workflow::DeclarativeOutcome::Cancelled { .. } => hi_workflow::WorkflowOutcome::Cancelled,
                    hi_workflow::DeclarativeOutcome::BudgetExceeded { message, .. } => hi_workflow::WorkflowOutcome::BudgetExceeded { message },
                    hi_workflow::DeclarativeOutcome::Failed { error, .. } => hi_workflow::WorkflowOutcome::Failed { error: error.to_string() },
                }
            })
        }
        None => {
            // The Rhai engine is synchronous (host calls block on the reply
            // channel the dashboard loop services), so it runs on a blocking
            // thread. The journal makes the run replayable/resumable.
            let params = hi_workflow::WorkflowRunParams {
                script: script.clone(),
                args,
                journal,
                host_tx,
                cancel: cancel.clone(),
                max_ops: hi_workflow::WorkflowRunParams::DEFAULT_MAX_OPS,
            };
            tokio::task::spawn_blocking(move || hi_workflow::run_workflow(params))
        }
    };

    app.workflow_run = Some(WorkflowRun {
        run_id,
        name: workflow_name,
        objective: workflow_description,
        phases,
        current_phase: None,
        host_rx: Some(host_rx),
        join_handle: Some(join_handle),
        cancel,
        outcome: None,
        log: Vec::new(),
        agent_budget: hi_workflow::DEFAULT_AGENT_BUDGET,
        agent_spent: 0,
        agent_reserved: 0,
    });

    Ok(())
}

/// Service a `WorkflowHostRequest` that arrived from the engine thread. Called
/// from the dashboard's `select!` loop. Returns `true` if the request was
/// handled, `false` if it was a `SpawnAgent` that needs a turn to complete
/// (the reply is stored on the row and sent in `finish_turn`).
pub(crate) async fn handle_workflow_host_request(
    app: &mut App,
    req: hi_workflow::WorkflowHostRequest,
    launcher: &FleetLauncher,
    line_tx: &mpsc::UnboundedSender<(usize, String)>,
    in_flight: &mut FuturesUnordered<RowFut>,
) {
    use hi_workflow::WorkflowHostRequest as R;
    match req {
        R::SpawnAgent { opts, reply } => {
            // Create a FleetRow for this agent, start a turn, and store the
            // reply sender. When the turn completes, finish_turn sends the
            // AgentResult back so the workflow can continue.
            spawn_workflow_agent(app, opts, reply, launcher, line_tx, in_flight).await;
        }
        R::Phase { title, replayed } => {
            if let Some(run) = &mut app.workflow_run {
                if !replayed {
                    run.on_phase(&title);
                }
            }
        }
        R::Log { message, replayed } => {
            if let Some(run) = &mut app.workflow_run
                && !replayed
            {
                run.log.push(message.clone());
            }
        }
        R::Telemetry { name, fields, replayed } => {
            if let Some(run) = &mut app.workflow_run
                && !replayed
            {
                run.log.push(format!("telemetry: {name} {fields}"));
            }
        }
        R::BudgetQuery { reply } => {
            let state = app.workflow_run.as_ref().map(|run| hi_workflow::BudgetState {
                total: Some(run.agent_budget),
                spent: run.agent_spent,
                reserved: run.agent_reserved,
                remaining: Some(
                    run.agent_budget
                        .saturating_sub(run.agent_spent.saturating_add(run.agent_reserved)),
                ),
            });
            let _ = reply.send(state.ok_or_else(|| {
                hi_workflow::HostError::Failed("workflow run is no longer active".into())
            }));
        }
        R::ReserveAgentCalls { count, reply } => {
            let result = app.workflow_run.as_mut().ok_or_else(|| {
                hi_workflow::HostError::Failed("workflow run is no longer active".into())
            }).and_then(|run| {
                let requested = run.agent_spent
                    .saturating_add(run.agent_reserved)
                    .saturating_add(count);
                if requested > run.agent_budget {
                    Err(hi_workflow::HostError::AgentCallQuotaExceeded {
                        requested,
                        maximum: run.agent_budget,
                    })
                } else {
                    run.agent_reserved += count;
                    Ok(())
                }
            });
            let _ = reply.send(result);
        }
        R::ReleaseAgentCalls { count, reply } => {
            let result = app.workflow_run.as_mut().ok_or_else(|| {
                hi_workflow::HostError::Failed("workflow run is no longer active".into())
            }).and_then(|run| {
                if count > run.agent_reserved {
                    return Err(hi_workflow::HostError::Failed(format!(
                        "cannot release {count} agent calls; only {} are reserved",
                        run.agent_reserved
                    )));
                }
                run.agent_reserved -= count;
                Ok(())
            });
            let _ = reply.send(result);
        }
        R::RenderTemplate { reply, .. } => {
            let _ = reply.send(Err(hi_workflow::HostError::Unsupported(
                "render_template not available in dashboard mode".into(),
            )));
        }
        R::WriteScratchFile { name, content, reply } => {
            let result = workflow_scratch_path(app, &name).and_then(|path| {
                if content.len() > 1024 * 1024 { return Err(hi_workflow::HostError::Failed("scratch file exceeds 1 MiB".into())); }
                if let Some(parent) = path.parent() { std::fs::create_dir_all(parent).map_err(|e| hi_workflow::HostError::Failed(e.to_string()))?; }
                std::fs::write(&path, &content).map_err(|e| hi_workflow::HostError::Failed(e.to_string()))?;
                Ok(path.display().to_string())
            });
            let _ = reply.send(result);
        }
        R::ReadScratchFile { name, reply } => {
            let result = workflow_scratch_path(app, &name).and_then(|path| {
                let meta = std::fs::metadata(&path).map_err(|e| hi_workflow::HostError::Failed(e.to_string()))?;
                if meta.len() > 1024 * 1024 { return Err(hi_workflow::HostError::Failed("scratch file exceeds 1 MiB".into())); }
                std::fs::read_to_string(path).map_err(|e| hi_workflow::HostError::Failed(e.to_string()))
            });
            let _ = reply.send(result);
        }
        R::GitDiffSince { commit, reply } => {
            let valid = !commit.is_empty() && commit.len() <= 128 && commit.bytes().all(|b| b.is_ascii_hexdigit());
            let result = if !valid { Err(hi_workflow::HostError::Failed("invalid commit id".into())) } else {
                std::process::Command::new("git").args(["-C", app.workspace_root.to_string_lossy().as_ref(), "diff", "--no-ext-diff", &commit, "--"])
                    .output().map_err(|e| hi_workflow::HostError::Failed(e.to_string())).and_then(|out| {
                        if !out.status.success() { Err(hi_workflow::HostError::Failed(String::from_utf8_lossy(&out.stderr).into_owned())) }
                        else if out.stdout.len() > 256 * 1024 { Err(hi_workflow::HostError::Failed("git diff exceeds 256 KiB".into())) }
                        else { Ok(String::from_utf8_lossy(&out.stdout).into_owned()) }
                    })
            };
            let _ = reply.send(result);
        }
    }
}

fn workflow_scratch_path(app: &App, name: &str) -> Result<std::path::PathBuf, hi_workflow::HostError> {
    if name.is_empty() || name.len() > 255 || name.contains('/') || name.contains('\\') || name == "." || name == ".." {
        return Err(hi_workflow::HostError::Failed("invalid scratch file name".into()));
    }
    let run_id = app.workflow_run.as_ref().ok_or_else(|| hi_workflow::HostError::Cancelled)?.run_id.clone();
    Ok(std::env::temp_dir().join("hi-workflows").join(run_id).join(name))
}

/// Create a `FleetRow` for a workflow `SpawnAgent` request, start the child
/// turn, and store the reply sender so `finish_turn` can send the result back.
async fn spawn_workflow_agent(
    app: &mut App,
    opts: hi_workflow::AgentOpts,
    reply: oneshot::Sender<Result<hi_workflow::AgentResult, hi_workflow::HostError>>,
    launcher: &FleetLauncher,
    line_tx: &mpsc::UnboundedSender<(usize, String)>,
    in_flight: &mut FuturesUnordered<RowFut>,
) {
    if !worktree::in_git_repo(&app.workspace_root) {
        let _ = reply.send(Err(hi_workflow::HostError::Failed(
            "not in a git repository (workflow agents need worktrees)".into(),
        )));
        return;
    }

    let mut prompt = opts.prompt.clone();
    let expects_json = opts.output_schema.is_some();
    if let Some(schema) = &opts.output_schema {
        prompt.push_str(
            "\n\nRespond with ONLY a single JSON object matching this JSON Schema — \
             no prose before or after it, no markdown code fences:\n",
        );
        prompt.push_str(&serde_json::to_string_pretty(schema).unwrap_or_default());
    }
    let title = opts
        .label
        .clone()
        .unwrap_or_else(|| truncate(&prompt, 48).to_string());
    let phase = opts.phase.clone();
    let label = opts.label.clone();

    // Snapshot the tree and create a worktree for this agent.
    let base = match hi_tools::checkpoint::create(&app.workspace_root).await {
        Some(commit) => commit,
        None => {
            let _ = reply.send(Err(hi_workflow::HostError::Failed(
                "couldn't snapshot the working tree".into(),
            )));
            return;
        }
    };
    // Use a blocking spawn for the worktree creation since it's sync.
    app.fleet_next_id += 1;
    let id = app.fleet_next_id;
    let path = worktree::worktree_path("fleet", id as u32);
    if let Err(err) = worktree::add_worktree(&app.workspace_root, &path, &base) {
        let _ = reply.send(Err(hi_workflow::HostError::Failed(format!(
            "couldn't create worktree: {err}"
        ))));
        return;
    }
    let session = match (launcher.session_path)() {
        Ok(s) => s,
        Err(err) => {
            let _ = std::fs::remove_dir_all(&path);
            let _ = reply.send(Err(hi_workflow::HostError::Failed(format!(
                "couldn't allocate session: {err}"
            ))));
            return;
        }
    };

    let mut row = FleetRow {
        id,
        title,
        worktree: path,
        base,
        session,
        state: RowState::Idle,
        merge: MergeState::None,
        changed: Vec::new(),
        activity: String::new(),
        tail: Vec::new(),
        pending: VecDeque::new(),
        reply: InputLine::default(),
        kill: None,
        started: None,
        turns: 0,
        usage: 0,
        goal: None,
        goal_objective: None,
        last_goal_json: None,
        driving: false,
        drive_stall: 0,
        stale: false,
        attention: false,
        workflow_reply: Some(reply),
        workflow_phase: phase,
        workflow_label: label,
        workflow_status: Some(WorkflowJobStatus::Running),
        workflow_expects_json: expects_json,
    };
    row.push_line(format!("› {prompt}"));
    app.fleet.push(row);
    let idx = app.fleet.len() - 1;
    start_turn(app, idx, prompt, launcher, line_tx, in_flight);
}

/// Re-adopt a past fleet session as a live row: fresh worktree off the current
/// tree, the old session file continues, its transcript preloads the peek tail,
/// and an active goal resumes driving immediately.
pub(crate) async fn adopt_session(
    app: &mut App,
    info: crate::FleetResumeInfo,
    launcher: &FleetLauncher,
    line_tx: &mpsc::UnboundedSender<(usize, String)>,
    in_flight: &mut FuturesUnordered<RowFut>,
) -> Result<usize> {
    if !worktree::in_git_repo(&app.workspace_root) {
        return Err(anyhow!(
            "not in a git repository (fleet rows need worktrees)"
        ));
    }
    let base = hi_tools::checkpoint::create(&app.workspace_root)
        .await
        .context("couldn't snapshot the working tree")?;
    app.fleet_next_id += 1;
    let id = app.fleet_next_id;
    let path = worktree::worktree_path("fleet", id as u32);
    worktree::add_worktree(&app.workspace_root, &path, &base)?;
    let goal = (info.goal_total > 0).then_some(RowGoal {
        done: info.goal_done,
        total: info.goal_total,
        active: info.goal_active,
        paused: false,
        phases: Vec::new(),
    });
    // Preload the peek tail with the session's conversation so attach shows
    // history immediately, before any new turn runs.
    let tail = load_transcript(&info.path, TAIL_CAP);
    let mut row = FleetRow {
        id,
        title: info.title,
        worktree: path,
        base,
        session: info.path,
        state: RowState::Idle,
        merge: MergeState::None,
        changed: Vec::new(),
        activity: String::new(),
        tail,
        pending: VecDeque::new(),
        reply: InputLine::default(),
        kill: None,
        started: None,
        turns: 0,
        usage: 0,
        goal,
        goal_objective: None,
        last_goal_json: None,
        driving: false,
        drive_stall: 0,
        stale: false,
        attention: false,
        workflow_reply: None,
        workflow_phase: None,
        workflow_label: None,
        workflow_status: None,
        workflow_expects_json: false,
    };
    row.push_line(format!("⟲ resumed session {}", info.id));
    let goal_active = row.goal.as_ref().is_some_and(|g| g.active);
    app.fleet.push(row);
    let idx = app.fleet.len() - 1;
    if goal_active {
        start_turn(
            app,
            idx,
            hi_agent::GOAL_CONTINUE_PROMPT.to_string(),
            launcher,
            line_tx,
            in_flight,
        );
    }
    Ok(idx)
}

/// Render a session file's conversation as plain display lines (last `cap`):
/// user prompts as `› …`, assistant text verbatim, tool calls as `⚙ label`.
fn load_transcript(path: &std::path::Path, cap: usize) -> Vec<String> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let mut lines: Vec<String> = Vec::new();
    for line in text.lines() {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if value.get("type").is_some() {
            continue; // session meta (usage/goal/compaction/…)
        }
        let Ok(msg) = serde_json::from_value::<hi_ai::Message>(value) else {
            continue;
        };
        match msg.role {
            hi_ai::Role::User => {
                for c in &msg.content {
                    if let hi_ai::Content::Text(t) = c {
                        let first = t.trim().lines().next().unwrap_or("").trim();
                        if !first.is_empty() {
                            lines.push(format!("› {}", truncate(first, 100)));
                        }
                    }
                }
            }
            hi_ai::Role::Assistant => {
                for c in &msg.content {
                    match c {
                        hi_ai::Content::Text(t) => lines.extend(
                            t.lines()
                                .map(str::trim_end)
                                .filter(|l| !l.trim().is_empty())
                                .map(str::to_string),
                        ),
                        hi_ai::Content::ToolCall {
                            name, arguments, ..
                        } => lines.push(format!("⚙ {}", hi_agent::ui::tool_label(name, arguments))),
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
    if lines.len() > cap {
        let drop = lines.len() - cap;
        lines.drain(..drop);
    }
    lines
}

/// Send `text` to the selected row: run it now if idle, else queue it.
fn send_reply(
    app: &mut App,
    idx: usize,
    text: String,
    launcher: &FleetLauncher,
    line_tx: &mpsc::UnboundedSender<(usize, String)>,
    in_flight: &mut FuturesUnordered<RowFut>,
) {
    let Some(row) = app.fleet.get_mut(idx) else {
        return;
    };
    row.attention = false;
    row.drive_stall = 0; // a user reply resets the drive-park guard
    if row.state == RowState::Working {
        row.push_line(format!("⧗ queued: {text}"));
        row.pending.push_back(text);
    } else if row.state != RowState::Closed {
        start_turn(app, idx, text, launcher, line_tx, in_flight);
    }
}

/// Spawn one child `hi` turn in the row's worktree, resuming its session.
fn start_turn(
    app: &mut App,
    idx: usize,
    prompt: String,
    launcher: &FleetLauncher,
    line_tx: &mpsc::UnboundedSender<(usize, String)>,
    in_flight: &mut FuturesUnordered<RowFut>,
) {
    let Some(row) = app.fleet.get_mut(idx) else {
        return;
    };
    row.state = RowState::Working;
    row.started = Some(Instant::now());
    row.activity = "starting…".to_string();
    row.driving = prompt == hi_agent::GOAL_CONTINUE_PROMPT;
    if row.driving {
        row.push_line("⟳ goal drive".to_string());
    } else {
        row.push_line(format!("› {prompt}"));
    }

    let mut cmd = tokio::process::Command::new(&launcher.exe);
    cmd.current_dir(&row.worktree)
        // Force the parent's resolved key (not a re-resolved default-profile
        // literal). Env, not argv, so it isn't exposed in `ps`.
        .env("HI_FORCE_API_KEY", &launcher.api_key)
        .env("HI_API_KEY", &launcher.api_key)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        // If this loop (or the whole TUI) drops mid-turn, take the child with
        // us rather than leaving an orphan writing to the worktree.
        .kill_on_drop(true)
        .args([
            "--provider",
            &launcher.provider,
            "--model",
            &launcher.model,
            "--base-url",
            &launcher.base_url,
        ]);
    if launcher.max_steps > 0 {
        cmd.args(["--max-steps", &launcher.max_steps.to_string()]);
    }
    cmd.arg("--session-file").arg(&row.session);
    // Per-turn ground truth: tokens, verify, changed files, goal progress.
    cmd.arg("--report").arg(report_path(row));
    // First turn of a /goal dispatch: the child plans the objective.
    if let Some(objective) = row.goal_objective.take() {
        cmd.arg("--goal").arg(objective);
    }
    if let Some(v) = &launcher.verify {
        cmd.args([
            "--verify",
            v,
            "--max-verify-repairs",
            &launcher.max_verify.to_string(),
        ]);
    }
    cmd.arg(&prompt);

    let mut child = match cmd.spawn() {
        Ok(child) => child,
        Err(err) => {
            row.state = RowState::Failed;
            row.started = None;
            row.push_line(format!("✗ couldn't launch the agent: {err}"));
            return;
        }
    };
    // Pump child output into the shared line stream (tagged with the row).
    if let Some(stdout) = child.stdout.take() {
        let tx = line_tx.clone();
        tokio::spawn(async move {
            let mut lines = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                if tx.send((idx, line)).is_err() {
                    break;
                }
            }
        });
    }
    if let Some(stderr) = child.stderr.take() {
        let tx = line_tx.clone();
        tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                if tx.send((idx, line)).is_err() {
                    break;
                }
            }
        });
    }
    let (kill_tx, mut kill_rx) = oneshot::channel::<()>();
    row.kill = Some(kill_tx);
    in_flight.push(Box::pin(async move {
        tokio::select! {
            status = child.wait() => {
                let ok = status.map(|s| s.success()).unwrap_or(false);
                (idx, RowDone::Turn { ok, killed: false })
            }
            _ = &mut kill_rx => {
                let _ = child.start_kill();
                let _ = child.wait().await;
                (idx, RowDone::Turn { ok: false, killed: true })
            }
        }
    }));
}

/// A child turn exited: on success, kick off the off-thread merge check (diff
/// vs base + verify gate) so the render loop never blocks on a slow verify.
fn finish_turn(
    app: &mut App,
    idx: usize,
    ok: bool,
    killed: bool,
    launcher: &FleetLauncher,
    line_tx: &mpsc::UnboundedSender<(usize, String)>,
    in_flight: &mut FuturesUnordered<RowFut>,
) {
    let Some(row) = app.fleet.get_mut(idx) else {
        return;
    };
    row.kill = None;
    row.turns += 1;
    // Ingest the child's report: session-cumulative tokens, goal progress, and
    // the drive-stall comparison (an unchanged goal across a drive turn counts
    // toward parking the drive).
    let was_driving = row.driving;
    row.driving = false;
    let mut retry_goal_turn = false;
    if let Some(report) = std::fs::read_to_string(report_path(row))
        .ok()
        .and_then(|t| parse_report(&t))
    {
        if report.total_tokens > 0 {
            row.usage = report.total_tokens;
        }
        row.drive_stall = next_drive_stall(
            was_driving,
            &row.last_goal_json,
            &report.goal_raw,
            row.drive_stall,
        );
        let was_active = row.goal.as_ref().is_some_and(|g| g.active);
        row.last_goal_json = report.goal_raw;
        row.goal = report.goal;
        retry_goal_turn = should_retry_goal_turn(
            was_driving,
            report.outcome_status.as_deref(),
            row.goal.as_ref(),
        );
        if was_active
            && row
                .goal
                .as_ref()
                .is_some_and(|g| !g.active && g.done == g.total)
        {
            row.push_line("◎ goal complete".to_string());
            record_fleet(launcher, row.id, &row.title, "goal complete");
        }
    }
    if killed {
        row.state = RowState::Failed;
        row.started = None;
        row.activity.clear();
        row.push_line("⚠ turn killed".to_string());
        // If this row was spawned by a workflow, send the failure back so the
        // engine can handle it (the workflow may pause or fail).
        if let Some(reply) = row.workflow_reply.take() {
            row.workflow_status = Some(WorkflowJobStatus::Cancelled);
            let _ = reply.send(Ok(hi_workflow::AgentResult {
                agent_id: format!("#{}", row.id),
                success: false,
                output: serde_json::json!({"summary": "turn killed"}),
                cancelled: true,
                tokens_used: row.usage,
                duration_ms: row
                    .started
                    .map(|t| t.elapsed().as_millis() as u64)
                    .unwrap_or(0),
            }));
        }
        flag_attention(app, idx);
        return;
    }
    if !ok {
        if retry_goal_turn {
            row.state = RowState::Idle;
            row.started = None;
            row.activity.clear();
            row.push_line("↻ goal turn incomplete — retrying the active sub-goal".to_string());
            continue_row(app, idx, launcher, line_tx, in_flight);
            return;
        }
        row.state = RowState::Failed;
        row.started = None;
        row.activity.clear();
        row.push_line("✗ agent run failed (see output above)".to_string());
        // Send the failure back to the workflow engine.
        if let Some(reply) = row.workflow_reply.take() {
            row.workflow_status = Some(WorkflowJobStatus::Failed);
            let _ = reply.send(Ok(hi_workflow::AgentResult {
                agent_id: format!("#{}", row.id),
                success: false,
                output: serde_json::json!({"summary": "agent run failed"}),
                cancelled: false,
                tokens_used: row.usage,
                duration_ms: row
                    .started
                    .map(|t| t.elapsed().as_millis() as u64)
                    .unwrap_or(0),
            }));
        }
        flag_attention(app, idx);
        return;
    }
    // Success: workflow rows use the same verify/diff/merge/post-merge gate as
    // ordinary fleet rows. Their host reply is retained until that lifecycle
    // reaches a terminal result.
    // Success: verify + diff in the worktree, off the render thread.
    row.activity = "merge check…".to_string();
    let worktree_path = row.worktree.clone();
    let base = row.base.clone();
    let verify = launcher.verify.clone();
    in_flight.push(Box::pin(async move {
        let outcome = tokio::task::spawn_blocking(move || {
            let changed = worktree::changed_files(&worktree_path, &base);
            let verified = match &verify {
                Some(v) if !changed.is_empty() => worktree::verify_passes(&worktree_path, v),
                _ => true,
            };
            (changed, verified)
        })
        .await
        .unwrap_or_else(|_| (Vec::new(), false));
        (
            idx,
            RowDone::MergeCheck {
                changed: outcome.0,
                verified: outcome.1,
            },
        )
    }));
}

/// The merge check landed: auto-merge when clean, hold when it overlaps other
/// rows' unmerged-or-merged files, then start any queued follow-up.
/// Record a notable fleet event (a verified merge, a combined-tree verify
/// failure, a goal completion) to the shared activity feed, so `/digest` is one
/// pane for every autonomous producer — loops, fleet rows, and goal drives.
fn record_fleet(launcher: &FleetLauncher, id: usize, title: &str, text: &str) {
    if let Some(lf) = &launcher.loops_file {
        let at_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        crate::activity::append(
            &crate::activity::activity_path(lf),
            &crate::activity::ActivityEntry {
                at_ms,
                loop_id: 0,
                source: format!("fleet#{id} {}", truncate_title(title, 40)),
                text: text.to_string(),
            },
        );
    }
}

fn finish_workflow_agent(row: &mut FleetRow, success: bool, summary: String) -> bool {
    let Some(reply) = row.workflow_reply.take() else { return false };
    row.workflow_status = Some(if success { WorkflowJobStatus::Completed } else { WorkflowJobStatus::Failed });
    let mut output = std::fs::read_to_string(report_path(row))
        .ok().and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        .and_then(|report| report.get("assistant_response").cloned())
        .unwrap_or_else(|| serde_json::json!({"summary": summary}));
    if row.workflow_expects_json {
        output = coerce_structured_output(output);
    }
    let _ = reply.send(Ok(hi_workflow::AgentResult {
        agent_id: format!("#{}", row.id), success, output, cancelled: false,
        tokens_used: row.usage, duration_ms: 0,
    }));
    true
}

/// A workflow agent was asked for schema-shaped output, but fleet children
/// reply with free text: recover the JSON object from the reply when possible
/// (the whole reply, or the outermost `{…}` span when prose or code fences
/// surround it). Scripts treat unrecoverable replies as failed structured
/// output, so the original string is returned unchanged on a parse failure.
fn coerce_structured_output(value: serde_json::Value) -> serde_json::Value {
    let serde_json::Value::String(text) = &value else { return value };
    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(text.trim()) {
        return parsed;
    }
    if let (Some(start), Some(end)) = (text.find('{'), text.rfind('}'))
        && start < end
        && let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&text[start..=end])
    {
        return parsed;
    }
    value
}

fn finish_merge_check(
    app: &mut App,
    idx: usize,
    changed: Vec<String>,
    verified: bool,
    launcher: &FleetLauncher,
    line_tx: &mpsc::UnboundedSender<(usize, String)>,
    in_flight: &mut FuturesUnordered<RowFut>,
) {
    // Overlap: any other open row whose changed files intersect ours — merged
    // rows included (re-applying an older base over their files would clobber).
    let overlaps: Vec<usize> = app
        .fleet
        .iter()
        .enumerate()
        .filter(|(i, other)| {
            *i != idx
                && other.state != RowState::Closed
                && other.changed.iter().any(|f| changed.contains(f))
        })
        .map(|(_, other)| other.id)
        .collect();
    let Some(row) = app.fleet.get_mut(idx) else {
        return;
    };
    row.state = RowState::Idle;
    row.started = None;
    row.activity.clear();
    row.changed = changed;
    if row.changed.is_empty() {
        row.merge = MergeState::None;
        if finish_workflow_agent(row, true, "completed without workspace changes".into()) {
            if let Some(run) = app.workflow_run.as_mut() {
                run.agent_reserved = run.agent_reserved.saturating_sub(1);
                run.agent_spent = run.agent_spent.saturating_add(1);
            }
            return;
        }
    } else if !verified {
        row.merge = MergeState::VerifyFailed;
        row.push_line("⇡ verify failed in the worktree — not merged (m forces)".to_string());
        if finish_workflow_agent(row, false, "worktree verification failed; changes were not merged".into()) {
            return;
        }
    } else if !overlaps.is_empty() {
        row.merge = MergeState::Held(overlaps.clone());
        row.push_line(format!(
            "⇡ merge held — overlaps #{} (m forces)",
            overlaps
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(", #")
        ));
        if finish_workflow_agent(row, false, format!("merge held because files overlap rows {overlaps:?}")) {
            return;
        }
    } else {
        match worktree::apply_changes_to(&row.worktree, &row.base, &app.workspace_root) {
            Ok(_) => {
                row.merge = MergeState::Merged(row.changed.len());
                row.push_line(format!(
                    "✓ merged {} file(s) into your tree: {}",
                    row.changed.len(),
                    row.changed.join(", ")
                ));
                record_fleet(
                    launcher,
                    row.id,
                    &row.title,
                    &format!(
                        "merged {} file(s): {}",
                        row.changed.len(),
                        row.changed.join(", ")
                    ),
                );
                mark_others_stale(app, idx);
                // Post-merge, off the render thread: verify the *combined* real
                // tree (a diff can pass in its worktree yet break the combine),
                // then refresh this row's base to a fresh snapshot so future
                // diffs are minimal. The row stays Working until this lands —
                // the worktree must not run a turn during the reset.
                let verify = launcher.verify.clone();
                let worktree_path = app.fleet[idx].worktree.clone();
                let Some(row) = app.fleet.get_mut(idx) else {
                    return;
                };
                row.state = RowState::Working;
                row.activity = "post-merge check…".to_string();
                in_flight.push(Box::pin(async move {
                    let verify_ok = match &verify {
                        Some(v) => {
                            let v = v.clone();
                            Some(
                                tokio::task::spawn_blocking(move || {
                                    worktree::verify_passes(std::path::Path::new("."), &v)
                                })
                                .await
                                .unwrap_or(false),
                            )
                        }
                        None => None,
                    };
                    let new_base = hi_tools::checkpoint::create(std::path::Path::new(".")).await;
                    let new_base = match new_base {
                        Some(base) => {
                            let wt = worktree_path.clone();
                            let sha = base.clone();
                            let reset_ok = tokio::task::spawn_blocking(move || {
                                worktree::reset_to(&wt, &sha).is_ok()
                            })
                            .await
                            .unwrap_or(false);
                            reset_ok.then_some(base)
                        }
                        None => None,
                    };
                    (
                        idx,
                        RowDone::PostVerify {
                            verify_ok,
                            new_base,
                        },
                    )
                }));
                return;
            }
            Err(err) => {
                row.merge = MergeState::VerifyFailed;
                row.push_line(format!("✗ merge failed: {err:#} (m retries)"));
                if finish_workflow_agent(row, false, format!("merge failed: {err:#}")) {
                    return;
                }
            }
        }
    }
    continue_row(app, idx, launcher, line_tx, in_flight);
}

/// The post-merge check landed: record the combined-tree verify verdict, adopt
/// the refreshed base, then continue the row (queued reply or goal drive).
fn finish_post_verify(
    app: &mut App,
    idx: usize,
    verify_ok: Option<bool>,
    new_base: Option<String>,
    launcher: &FleetLauncher,
    line_tx: &mpsc::UnboundedSender<(usize, String)>,
    in_flight: &mut FuturesUnordered<RowFut>,
) {
    let Some(row) = app.fleet.get_mut(idx) else {
        return;
    };
    row.state = RowState::Idle;
    row.started = None;
    row.activity.clear();
    if let Some(base) = new_base {
        // The fresh snapshot contains this row's merged diff, so the worktree
        // is now clean against it.
        row.base = base;
        row.changed.clear();
        row.stale = false;
    }
    if verify_ok == Some(false) {
        row.push_line("⚠ combined-tree verify failed after merge — inspect your tree".to_string());
        record_fleet(
            launcher,
            row.id,
            &row.title,
            "combined-tree verify failed after merge — inspect your tree",
        );
        row.attention = true;
    }
    let success = verify_ok != Some(false);
    let summary = if success {
        "verified changes merged into the workspace".to_string()
    } else {
        "combined-tree verification failed after merge".to_string()
    };
    if finish_workflow_agent(row, success, summary) {
        if let Some(run) = app.workflow_run.as_mut() {
            run.agent_reserved = run.agent_reserved.saturating_sub(1);
            run.agent_spent = run.agent_spent.saturating_add(1);
        }
        return;
    }
    continue_row(app, idx, launcher, line_tx, in_flight);
}

/// After a turn fully settles: run the next queued reply, else keep a goal
/// drive going, else the row is waiting on the user (attention).
fn continue_row(
    app: &mut App,
    idx: usize,
    launcher: &FleetLauncher,
    line_tx: &mpsc::UnboundedSender<(usize, String)>,
    in_flight: &mut FuturesUnordered<RowFut>,
) {
    let Some(row) = app.fleet.get_mut(idx) else {
        return;
    };
    if row.state != RowState::Idle {
        return;
    }
    if let Some(next) = row.pending.pop_front() {
        row.drive_stall = 0; // user input resets the stall guard
        start_turn(app, idx, next, launcher, line_tx, in_flight);
        return;
    }
    let drive = row.goal.as_ref().is_some_and(|g| g.active && !g.paused);
    if drive {
        if row.drive_stall >= hi_agent::GOAL_DRIVE_STALL_LIMIT {
            row.push_line(
                "⏸ drive parked — no progress for 2 turns; reply to steer and resume".to_string(),
            );
            flag_attention(app, idx);
            return;
        }
        start_turn(
            app,
            idx,
            hi_agent::GOAL_CONTINUE_PROMPT.to_string(),
            launcher,
            line_tx,
            in_flight,
        );
        return;
    }
    // Idle with nothing to do: the agent is waiting on the user.
    flag_attention(app, idx);
}

/// Mark the row as needing the user; ping the terminal when it's unfocused.
fn flag_attention(app: &mut App, idx: usize) {
    let unfocused = app.focus_known && !app.focused;
    if let Some(row) = app.fleet.get_mut(idx)
        && !row.attention
    {
        row.attention = true;
        if unfocused {
            crate::util::notify_done();
        }
    }
}

/// After a row's diff lands in the real tree, every other open row is building
/// against a snapshot that no longer matches it.
fn mark_others_stale(app: &mut App, idx: usize) {
    for (i, other) in app.fleet.iter_mut().enumerate() {
        if i != idx && other.state != RowState::Closed {
            other.stale = true;
        }
    }
}

/// The row's per-turn report file (next to its session, outside any repo).
fn report_path(row: &FleetRow) -> PathBuf {
    row.session.with_extension("report.json")
}

/// Split a dispatch-box entry: a `/goal <objective>` prefix makes the row
/// goal-driven (objective doubles as the first prompt and the row title).
fn split_goal_dispatch(prompt: String) -> (Option<String>, String) {
    match prompt.strip_prefix("/goal ") {
        Some(objective) => {
            let objective = objective.trim().to_string();
            (Some(objective.clone()), objective)
        }
        None => (None, prompt),
    }
}

/// `m`: apply the selected row's diff to the real tree regardless of holds.
fn force_merge(app: &mut App, idx: usize, flash: &mut Option<String>) {
    let Some(row) = app.fleet.get_mut(idx) else {
        return;
    };
    // Recompute cheaply — the row may have been edited since the last check.
    let changed = worktree::changed_files(&row.worktree, &row.base);
    if changed.is_empty() {
        *flash = Some(format!("#{}: nothing to merge", row.id));
        return;
    }
    match worktree::apply_changes_to(&row.worktree, &row.base, &app.workspace_root) {
        Ok(_) => {
            row.changed = changed;
            row.merge = MergeState::Merged(row.changed.len());
            row.attention = false;
            row.push_line(format!(
                "✓ merged {} file(s) into your tree (forced)",
                row.changed.len()
            ));
            mark_others_stale(app, idx);
        }
        Err(err) => {
            *flash = Some(format!("#{}: merge failed: {err:#}", row.id));
        }
    }
}

/// `r`: rebase an idle row's worktree onto a fresh snapshot of the real tree.
/// Refused while the row has unmerged changes (merge or close first).
fn rebase_row(app: &mut App, idx: usize, new_base: Option<String>, flash: &mut Option<String>) {
    let Some(row) = app.fleet.get_mut(idx) else {
        return;
    };
    let unmerged = !row.changed.is_empty() && !matches!(row.merge, MergeState::Merged(_));
    if unmerged {
        *flash = Some(format!(
            "#{}: unmerged changes — merge (m) or close (x) first",
            row.id
        ));
        return;
    }
    let Some(base) = new_base else {
        *flash = Some(format!("#{}: couldn't snapshot the tree", row.id));
        return;
    };
    match worktree::reset_to(&row.worktree, &base) {
        Ok(()) => {
            row.base = base;
            row.changed.clear();
            row.stale = false;
            row.attention = false;
            row.push_line("⟳ rebased onto the current tree".to_string());
        }
        Err(err) => *flash = Some(format!("#{}: rebase failed: {err:#}", row.id)),
    }
}

/// Strip ANSI escape sequences (CSI/OSC) so child output renders as plain rows.
fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '\u{1b}' {
            out.push(c);
            continue;
        }
        match chars.peek() {
            // CSI: ESC [ … final byte in @-~
            Some('[') => {
                chars.next();
                for c in chars.by_ref() {
                    if ('\u{40}'..='\u{7e}').contains(&c) {
                        break;
                    }
                }
            }
            // OSC: ESC ] … BEL (or ESC \)
            Some(']') => {
                chars.next();
                while let Some(c) = chars.next() {
                    if c == '\u{7}' {
                        break;
                    }
                    if c == '\u{1b}' && chars.peek() == Some(&'\\') {
                        chars.next();
                        break;
                    }
                }
            }
            _ => {
                chars.next();
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn render_dashboard(
    frame: &mut ratatui::Frame,
    app: &App,
    selected: usize,
    focus: Focus,
    dispatch: &InputLine,
    working: usize,
    exit_armed: bool,
    flash: Option<&str>,
    peek_offset: usize,
) {
    let area = frame.area();
    let attach = focus == Focus::Attach;
    let table_height = if attach {
        0
    } else {
        (app.fleet.len() + workflow_phase_header_count(app)).clamp(1, TABLE_ROWS) as u16 + 2
    };
    let rows = Layout::vertical([
        Constraint::Length(1),            // header
        Constraint::Length(table_height), // fleet table (hidden in attach)
        Constraint::Min(3),               // peek / attach panel
        Constraint::Length(3),            // focused input
        Constraint::Length(1),            // footer hints
    ])
    .split(area);

    let title = if let Some(run) = &app.workflow_run {
        let phase_trail = if run.phases.is_empty() {
            String::new()
        } else {
            run.phases
                .iter()
                .map(|(title, state)| {
                    let mark = match state.as_str() {
                        "done" => "✓",
                        "active" => "●",
                        _ => "○",
                    };
                    format!("{title} {mark}")
                })
                .collect::<Vec<_>>()
                .join(" · ")
        };
        let status = if let Some(outcome) = &run.outcome {
            workflow_outcome_summary(outcome)
        } else {
            "running".to_string()
        };
        let objective = if run.objective.is_empty() {
            String::new()
        } else {
            format!(" — {}", truncate(&run.objective, 48))
        };
        format!(
            " hi workflow · {}{objective} · {} · {} agent(s){} ",
            run.name,
            if phase_trail.is_empty() {
                &status
            } else {
                &phase_trail
            },
            app.fleet.len(),
            if exit_armed {
                " — turns in flight! Esc again kills them (sessions stay resumable)"
            } else {
                ""
            },
        )
    } else {
        format!(
            " hi fleet · {} agent(s) · {} working{} ",
            app.fleet.len(),
            working,
            if exit_armed {
                " — turns in flight! Esc again kills them (sessions stay resumable)"
            } else {
                ""
            },
        )
    };
    let header_style = if exit_armed {
        Style::default()
            .fg(crate::theme::theme().warning)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default()
            .fg(crate::theme::theme().accent_assistant)
            .add_modifier(Modifier::BOLD)
    };
    frame.render_widget(Paragraph::new(Line::styled(title, header_style)), rows[0]);

    if !attach {
        render_table(frame, app, selected, rows[1]);
    }
    render_peek(frame, app, selected, rows[2], attach, peek_offset);
    render_input(frame, app, selected, focus, dispatch, rows[3]);

    let hint = match flash {
        Some(msg) => Line::styled(msg.to_string(), Style::default().fg(crate::theme::theme().warning)),
        None => Line::styled(
            match focus {
                Focus::Dispatch => {
                    "Enter dispatch (/goal <obj> = driven) · Ctrl+S +attach · ↑↓ · Tab reply · m merge · r rebase · x close · Ctrl+K kill · PgUp scroll · Esc"
                }
                Focus::Reply => {
                    "Enter send · 1-9 quick answer · ↑↓ select · Tab dispatch · Esc back"
                }
                Focus::Attach => "Enter send · 1-9 quick answer · Esc table",
            }
            .to_string(),
            dim(),
        ),
    };
    frame.render_widget(Paragraph::new(hint), rows[4]);
}

fn merge_badge(row: &FleetRow) -> (String, Style) {
    match &row.merge {
        MergeState::None => (String::new(), dim()),
        MergeState::Merged(n) => (
            format!("✓{n}"),
            Style::default().fg(crate::theme::theme().accent_success),
        ),
        MergeState::Held(_) => (
            "⇡held".to_string(),
            Style::default().fg(crate::theme::theme().warning),
        ),
        MergeState::VerifyFailed => (
            "⇡unverified".to_string(),
            Style::default().fg(crate::theme::theme().warning),
        ),
    }
}

/// A short human-readable summary of a workflow outcome, for the dashboard
/// flash message and log.
fn workflow_outcome_summary(outcome: &hi_workflow::WorkflowOutcome) -> String {
    match outcome {
        hi_workflow::WorkflowOutcome::Completed { .. } => "✓ completed".to_string(),
        hi_workflow::WorkflowOutcome::Paused { kind, message } => {
            format!("⏸ paused ({}): {message}", kind.as_str())
        }
        hi_workflow::WorkflowOutcome::BudgetExceeded { message } => {
            format!("⏸ budget exceeded: {message}")
        }
        hi_workflow::WorkflowOutcome::Cancelled => "◌ cancelled".to_string(),
        hi_workflow::WorkflowOutcome::Failed { error } => format!("✗ failed: {error}"),
    }
}

/// Render a phase trail as a compact `✓Scan · ●Analyze · ○Synthesize` string.
fn phase_trail(goal: &RowGoal) -> Option<String> {
    if goal.phases.is_empty() {
        return None;
    }
    Some(
        goal.phases
            .iter()
            .map(|(title, state)| {
                let mark = match state.as_str() {
                    "done" => "✓",
                    "active" => "●",
                    _ => "○",
                };
                format!("{title} {mark}")
            })
            .collect::<Vec<_>>()
            .join(" · "),
    )
}

/// Phase-header lines the workflow grouping adds to the fleet table: one per
/// contiguous run of rows sharing a `workflow_phase` (workflow agents spawn
/// in phase order, so contiguous runs are the phases).
fn workflow_phase_header_count(app: &App) -> usize {
    if app.workflow_run.is_none() {
        return 0;
    }
    phase_header_count(&app.fleet)
}

fn phase_header_count(rows: &[FleetRow]) -> usize {
    let mut count = 0;
    let mut last: Option<&str> = None;
    for row in rows {
        if let Some(phase) = row.workflow_phase.as_deref()
            && last != Some(phase)
        {
            count += 1;
            last = Some(phase);
        }
    }
    count
}

fn render_table(frame: &mut ratatui::Frame, app: &App, selected: usize, area: Rect) {
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(crate::theme::theme().accent_assistant))
        .title(" fleet — each row works in its own worktree; clean diffs merge back ");
    let inner_rows = area.height.saturating_sub(2) as usize;
    let start = selected.saturating_sub(inner_rows.saturating_sub(1));
    let mut lines: Vec<Line> = Vec::new();
    if app.fleet.is_empty() {
        lines.push(Line::styled(
            "no agents yet — type a prompt below and press Enter to dispatch one".to_string(),
            dim(),
        ));
    }
    // Workflow runs group rows under their phase: a header line opens each
    // contiguous run of rows sharing a `workflow_phase`.
    let group_phases = app.workflow_run.is_some();
    let mut last_phase: Option<&str> = None;
    for (i, row) in app.fleet.iter().enumerate().skip(start).take(inner_rows) {
        if group_phases
            && let Some(phase) = row.workflow_phase.as_deref()
            && last_phase != Some(phase)
        {
            lines.push(Line::styled(
                format!(" ▸ {phase}"),
                Style::default()
                    .fg(crate::theme::theme().accent_assistant)
                    .add_modifier(Modifier::BOLD),
            ));
            last_phase = Some(phase);
        }
        let (glyph, glyph_style) = match row.state {
            RowState::Working => (
                SPINNER[app.spinner % SPINNER.len()].to_string(),
                Style::default().fg(crate::theme::theme().accent_system),
            ),
            RowState::Idle => (
                "·".to_string(),
                Style::default().fg(crate::theme::theme().accent_success),
            ),
            RowState::Failed => (
                "✗".to_string(),
                Style::default().fg(crate::theme::theme().accent_error),
            ),
            RowState::Closed => ("—".to_string(), dim()),
        };
        let elapsed = row
            .started
            .map(|t| {
                let s = t.elapsed().as_secs();
                format!("{}m{:02}s", s / 60, s % 60)
            })
            .unwrap_or_else(|| format!("{} turn(s)", row.turns));
        let (badge, badge_style) = merge_badge(row);
        let lead = if row.state == RowState::Working && !row.activity.is_empty() {
            &row.activity
        } else {
            &row.title
        };
        let queued = if row.pending.is_empty() {
            String::new()
        } else {
            format!(" ⧗{}", row.pending.len())
        };
        let style = if i == selected {
            Style::default().add_modifier(Modifier::BOLD | Modifier::REVERSED)
        } else if row.state == RowState::Closed {
            dim()
        } else {
            Style::default()
        };
        // Attention (●), goal progress (◎d/t) or phase trail, stale (⟳),
        // tokens — the fleet vitals at a glance.
        let attention = if row.attention { "●" } else { " " };
        let goal_span = row
            .goal
            .as_ref()
            .and_then(|g| {
                phase_trail(g).map(|trail| {
                    Span::styled(
                        truncate(&trail, 38),
                        Style::default().fg(crate::theme::theme().accent_assistant),
                    )
                })
            })
            .or_else(|| {
                row.goal.as_ref().map(|g| {
                    Span::styled(
                        format!("◎{}/{}", g.done, g.total),
                        Style::default().fg(crate::theme::theme().accent_assistant),
                    )
                })
            })
            .or_else(|| {
                // Workflow rows have no goal; show the script-assigned stable
                // label so the row stays identifiable while activity streams.
                row.workflow_label.as_deref().map(|label| {
                    Span::styled(
                        truncate(label, 24).to_string(),
                        Style::default().fg(crate::theme::theme().accent_assistant),
                    )
                })
            })
            .unwrap_or_else(|| Span::raw(""));
        let stale = if row.stale { "⟳" } else { " " };
        lines.push(Line::from(vec![
            Span::styled(format!(" {glyph} "), glyph_style),
            Span::styled(
                attention.to_string(),
                Style::default().fg(crate::theme::theme().accent_system),
            ),
            Span::styled(format!("#{:<2} {:>9}{} ", row.id, elapsed, queued), style),
            Span::styled(format!("↓{:>6} ", crate::util::fmt_count(row.usage)), dim()),
            goal_span,
            Span::raw(" "),
            Span::styled(
                stale.to_string(),
                Style::default().fg(crate::theme::theme().warning),
            ),
            Span::styled(format!("{badge:>11} "), badge_style),
            Span::styled(truncate(lead, 46), style),
        ]));
    }
    frame.render_widget(Paragraph::new(lines).block(block), area);
}

fn render_peek(
    frame: &mut ratatui::Frame,
    app: &App,
    selected: usize,
    area: Rect,
    attach: bool,
    offset: usize,
) {
    let Some(row) = app.fleet.get(selected) else {
        let block = Block::bordered()
            .border_type(BorderType::Rounded)
            .border_style(dim())
            .title(" peek ");
        frame.render_widget(
            Paragraph::new(Line::styled(
                "select a row to peek at its output".to_string(),
                dim(),
            ))
            .block(block),
            area,
        );
        return;
    };
    let title = format!(
        " #{} · {} {} ",
        row.id,
        truncate(&row.title, 48),
        if attach { "(attached)" } else { "" },
    );
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(if attach {
            crate::theme::theme().accent_system
        } else {
            crate::theme::theme().gray_dim
        }))
        .title(title);
    let inner = area.height.saturating_sub(2) as usize;
    let mut lines: Vec<Line> = Vec::new();
    // Phase trail header — shows the goal's phase progress at the top of the
    // peek panel when the row has phases (a `/goal`-driven row with a plan).
    if let Some(g) = &row.goal
        && let Some(trail) = phase_trail(g)
    {
        lines.push(Line::styled(
            trail,
            Style::default().fg(crate::theme::theme().accent_assistant),
        ));
    }
    let follow = row.tail.len().saturating_sub(inner.saturating_sub(1));
    let offset = offset.min(follow);
    let shown = follow - offset;
    for line in row.tail.iter().skip(shown).take(inner.saturating_sub(1)) {
        let style = if line.starts_with('⚙') || line.starts_with('›') {
            dim()
        } else if line.starts_with('✗') || line.starts_with('⚠') {
            Style::default().fg(crate::theme::theme().accent_error)
        } else if line.starts_with('✓') || line.starts_with('⇡') {
            Style::default().fg(crate::theme::theme().accent_success)
        } else {
            Style::default()
        };
        lines.push(Line::styled(line.clone(), style));
    }
    if offset > 0 {
        // Scrolled back: show how far off the live tail we are instead of the
        // spinner (PgDn returns to follow).
        lines.push(Line::styled(format!("↓ {offset} newer (PgDn)"), dim()));
    } else if row.state == RowState::Working {
        lines.push(Line::styled(
            format!(
                "{} {}",
                SPINNER[app.spinner % SPINNER.len()],
                if row.activity.is_empty() {
                    "Working…"
                } else {
                    &row.activity
                }
            ),
            Style::default().fg(crate::theme::theme().accent_system),
        ));
    }
    frame.render_widget(Paragraph::new(lines).block(block), area);
}

fn render_input(
    frame: &mut ratatui::Frame,
    app: &App,
    selected: usize,
    focus: Focus,
    dispatch: &InputLine,
    area: Rect,
) {
    let (title, input, accent) = match focus {
        Focus::Dispatch => (
            " dispatch — Enter spawns a new agent · Ctrl+S spawns and attaches ".to_string(),
            dispatch,
            crate::theme::theme().accent_assistant,
        ),
        Focus::Reply | Focus::Attach => {
            let id = app.fleet.get(selected).map(|r| r.id).unwrap_or_default();
            let state = app
                .fleet
                .get(selected)
                .map(|r| {
                    if r.state == RowState::Working {
                        " (working — reply will queue)"
                    } else {
                        ""
                    }
                })
                .unwrap_or_default();
            (
                format!(" reply → #{id}{state} "),
                app.fleet
                    .get(selected)
                    .map(|r| &r.reply)
                    .unwrap_or(dispatch),
                crate::theme::theme().accent_system,
            )
        }
    };
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(accent))
        .title(title);
    let text = input.text();
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("› ", dim()),
            Span::raw(text.clone()),
        ]))
        .block(block),
        area,
    );
    let cursor_col = input.cursor().min(text.chars().count()) as u16;
    frame.set_cursor_position((area.x + 3 + cursor_col, area.y + 1));
}

/// Strip ANSI from one child-output line (shared with `/loop` firings).
pub(crate) fn strip_ansi_line(s: &str) -> String {
    strip_ansi(s)
}

/// Truncate for single-line display (shared with the /fleet status view).
pub(crate) fn truncate_title(s: &str, max: usize) -> String {
    truncate(s, max)
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let cut: String = s.chars().take(max.saturating_sub(1)).collect();
        format!("{cut}…")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row() -> FleetRow {
        FleetRow {
            id: 1,
            title: "test".into(),
            worktree: PathBuf::from("/tmp/x"),
            base: "abc".into(),
            session: PathBuf::from("/tmp/x.jsonl"),
            state: RowState::Working,
            merge: MergeState::None,
            changed: Vec::new(),
            activity: String::new(),
            tail: Vec::new(),
            pending: VecDeque::new(),
            reply: InputLine::default(),
            kill: None,
            started: None,
            turns: 0,
            usage: 0,
            goal: None,
            goal_objective: None,
            last_goal_json: None,
            driving: false,
            drive_stall: 0,
            stale: false,
            attention: false,
            workflow_reply: None,
            workflow_phase: None,
            workflow_label: None,
            workflow_status: None,
            workflow_expects_json: false,
        }
    }

    #[test]
    fn phase_headers_count_contiguous_phase_runs() {
        let phased = |phase: Option<&str>| {
            let mut r = row();
            r.workflow_phase = phase.map(str::to_string);
            r
        };
        assert_eq!(phase_header_count(&[]), 0);
        // Rows without phases contribute no headers.
        assert_eq!(phase_header_count(&[phased(None), phased(None)]), 0);
        // Contiguous runs share one header; phase changes open a new one.
        let rows = [
            phased(Some("Research")),
            phased(Some("Research")),
            phased(Some("Verify")),
            phased(None),
            phased(Some("Report")),
        ];
        assert_eq!(phase_header_count(&rows), 3);
    }

    #[test]
    fn goal_dispatch_prefix_is_stripped() {
        let (obj, prompt) = split_goal_dispatch("/goal port the parser to Rust".to_string());
        assert_eq!(obj.as_deref(), Some("port the parser to Rust"));
        assert_eq!(prompt, "port the parser to Rust");
        let (obj, prompt) = split_goal_dispatch("fix the failing test".to_string());
        assert!(obj.is_none());
        assert_eq!(prompt, "fix the failing test");
    }

    #[test]
    fn workflow_outcome_summary_formats_correctly() {
        use hi_workflow::{PauseKind, WorkflowOutcome};
        assert_eq!(
            workflow_outcome_summary(&WorkflowOutcome::Completed {
                result: serde_json::json!(null)
            }),
            "✓ completed"
        );
        assert_eq!(
            workflow_outcome_summary(&WorkflowOutcome::Cancelled),
            "◌ cancelled"
        );
        let paused = workflow_outcome_summary(&WorkflowOutcome::Paused {
            kind: PauseKind::User,
            message: "need input".into(),
        });
        assert!(paused.contains("paused") && paused.contains("need input"));
        let failed = workflow_outcome_summary(&WorkflowOutcome::Failed {
            error: "boom".into(),
        });
        assert!(failed.contains("failed") && failed.contains("boom"));
    }

    #[test]
    fn workflow_run_on_phase_tracks_progress() {
        let (_host_tx, host_rx) = mpsc::unbounded_channel::<hi_workflow::WorkflowHostRequest>();
        let mut run = WorkflowRun {
            run_id: "test-run".into(),
            name: "test".into(),
            objective: "test".into(),
            phases: vec![
                ("Scan".into(), "pending".into()),
                ("Analyze".into(), "pending".into()),
                ("Synthesize".into(), "pending".into()),
            ],
            current_phase: None,
            host_rx: Some(host_rx),
            join_handle: None,
            cancel: tokio_util::sync::CancellationToken::new(),
            outcome: None,
            log: Vec::new(),
            agent_budget: hi_workflow::DEFAULT_AGENT_BUDGET,
            agent_spent: 0,
            agent_reserved: 0,
        };
        // First phase becomes active.
        run.on_phase("Scan");
        assert_eq!(run.current_phase, Some(0));
        assert_eq!(run.phases[0].1, "active");
        // Second phase: first becomes done, second becomes active.
        run.on_phase("Analyze");
        assert_eq!(run.phases[0].1, "done");
        assert_eq!(run.phases[1].1, "active");
        assert_eq!(run.current_phase, Some(1));
        // Third phase.
        run.on_phase("Synthesize");
        assert_eq!(run.phases[1].1, "done");
        assert_eq!(run.phases[2].1, "active");
        assert_eq!(run.current_phase, Some(2));
        // Log was appended for each phase.
        assert_eq!(run.log.len(), 3);
    }

    #[test]
    fn workflow_run_on_phase_adds_unknown_phase() {
        let (_host_tx, host_rx) = mpsc::unbounded_channel::<hi_workflow::WorkflowHostRequest>();
        let mut run = WorkflowRun {
            run_id: "test-run".into(),
            name: "test".into(),
            objective: "test".into(),
            phases: vec![],
            current_phase: None,
            host_rx: Some(host_rx),
            join_handle: None,
            cancel: tokio_util::sync::CancellationToken::new(),
            outcome: None,
            log: Vec::new(),
            agent_budget: hi_workflow::DEFAULT_AGENT_BUDGET,
            agent_spent: 0,
            agent_reserved: 0,
        };
        run.on_phase("Adhoc");
        assert_eq!(run.phases.len(), 1);
        assert_eq!(run.phases[0], ("Adhoc".into(), "active".into()));
        assert_eq!(run.current_phase, Some(0));
    }

    #[test]
    fn output_lines_are_stripped_and_tailed() {
        let mut r = row();
        r.push_output("\u{1b}[1;35m↳ delegate subagent 1/4\u{1b}[0m");
        r.push_output("   ");
        r.push_output("plain line");
        assert_eq!(r.tail, vec!["↳ delegate subagent 1/4", "plain line"]);
        assert_eq!(r.activity, "plain line");
    }

    #[test]
    fn tail_is_capped() {
        let mut r = row();
        for i in 0..(TAIL_CAP + 50) {
            r.push_line(format!("line {i}"));
        }
        assert_eq!(r.tail.len(), TAIL_CAP);
        assert_eq!(r.tail.first().map(String::as_str), Some("line 50"));
    }

    #[test]
    fn strip_ansi_handles_csi_and_osc() {
        assert_eq!(strip_ansi("\u{1b}[32m✓ ok\u{1b}[0m"), "✓ ok");
        assert_eq!(strip_ansi("\u{1b}]0;title\u{7}body"), "body");
        assert_eq!(strip_ansi("no escapes"), "no escapes");
    }

    #[test]
    fn load_transcript_renders_conversation_lines() {
        use hi_ai::Message;
        let dir = std::env::temp_dir().join(format!("hi-fleet-lt-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("s.jsonl");
        let lines = [
            serde_json::to_string(&Message::system("sys prompt")).unwrap(),
            serde_json::to_string(&Message::user("fix the parser\nsecond line")).unwrap(),
            serde_json::to_string(&Message::assistant(vec![hi_ai::Content::Text(
                "done, it parses".into(),
            )]))
            .unwrap(),
            r#"{"type":"usage","input_tokens":1,"output_tokens":2}"#.to_string(),
        ];
        std::fs::write(&path, lines.join("\n") + "\n").unwrap();

        let out = load_transcript(&path, 50);
        assert!(
            out.iter().any(|l| l.starts_with("› fix the parser")),
            "{out:?}"
        );
        assert!(out.iter().any(|l| l == "done, it parses"), "{out:?}");
        // System prompt + meta lines are skipped.
        assert!(!out.iter().any(|l| l.contains("sys prompt")), "{out:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
