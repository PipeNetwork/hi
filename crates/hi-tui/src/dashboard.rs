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
use std::io::{BufRead, BufReader as StdBufReader};
use std::path::PathBuf;
use std::pin::Pin;
use std::process::Stdio;
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use crossterm::event::{Event, KeyCode, KeyEventKind, KeyModifiers};
use futures_util::stream::FuturesUnordered;
use futures_util::{FutureExt, StreamExt};
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
/// Per-row next-turn backlog while a child turn is running (same cap as the
/// main session queue).
const MAX_ROW_PENDING: usize = crate::MAX_PROMPT_QUEUE;

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
    /// Run that owns this workflow child, keeping concurrent runs isolated.
    pub(crate) workflow_run_id: Option<String>,
    /// The phase this row's agent belongs to (from `AgentOpts.phase`), used to
    /// group rows under phase headers in the workflow run view.
    pub(crate) workflow_phase: Option<String>,
    /// Stable label assigned by the workflow, used in the fleet/detail views.
    pub(crate) workflow_label: Option<String>,
    /// Typed workflow child state, independent of the generic row state.
    pub(crate) workflow_status: Option<WorkflowJobStatus>,
    /// The `output_schema` the workflow requested for this agent, when any:
    /// the reply's `assistant_response` is parsed back into JSON and validated
    /// against it before it reaches the engine (fleet children only produce
    /// text). One corrective retry is spent on a mismatch.
    pub(crate) workflow_schema: Option<serde_json::Value>,
    /// The single schema-mismatch corrective retry has been used.
    pub(crate) workflow_schema_retry_used: bool,
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
    /// Canonical presentation state shared with the transcript/event surface.
    pub(crate) snapshot: hi_workflow::WorkflowRunSnapshot,
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
    pub(crate) fn from_managed(
        managed: hi_workflow::ManagedWorkflowRun,
        objective: String,
        phases: Vec<(String, String)>,
    ) -> Self {
        let (manifest, host_rx, cancel, task) = managed.into_parts();
        let name = manifest.workflow_name.clone();
        let run_id = manifest.run_id.clone();
        let snapshot = hi_workflow::WorkflowRunSnapshot {
            run_id: run_id.clone(),
            revision: 1,
            workflow_name: name.clone(),
            objective: objective.clone(),
            status: manifest.status(),
            phases: phases
                .iter()
                .map(|(title, state)| hi_workflow::WorkflowPhaseSnapshot {
                    title: title.clone(),
                    state: state.clone(),
                })
                .collect(),
            current_phase: manifest.current_phase.clone(),
            agents: Vec::new(),
            agent_budget: manifest.agent_budget,
            agents_used: manifest.agent_spent,
            agents_reserved: 0,
            elapsed_ms: 0,
            pause_message: None,
            result_summary: None,
            history: Vec::new(),
        };
        Self {
            snapshot,
            run_id,
            name,
            objective,
            phases,
            current_phase: None,
            host_rx: Some(host_rx),
            join_handle: Some(task),
            cancel,
            outcome: None,
            log: Vec::new(),
            agent_budget: manifest.agent_budget,
            agent_spent: manifest.agent_spent,
            agent_reserved: 0,
        }
    }

    /// Update the phase trail when a `Phase` host request arrives.
    fn on_phase(&mut self, title: &str) {
        // Mark the previous active phase as done.
        if let Some(idx) = self.current_phase
            && idx < self.phases.len()
        {
            self.phases[idx].1 = "done".into();
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
        self.snapshot
            .record_event("phase_started", Some(title.to_string()), now_ms());
        self.snapshot.current_phase = Some(title.to_string());
        self.snapshot.phases = self
            .phases
            .iter()
            .map(|(title, state)| hi_workflow::WorkflowPhaseSnapshot {
                title: title.clone(),
                state: state.clone(),
            })
            .collect();
    }
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
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
    /// The verified worktree diff was applied to the real workspace.
    MergeApply {
        changed: Vec<String>,
        result: Result<(), String>,
    },
    /// A user-requested force merge completed off the render task.
    ForceMerge {
        changed: Vec<String>,
        result: Result<(), String>,
    },
    /// A user-requested worktree rebase completed off the render task.
    Rebase {
        base: String,
        result: Result<(), String>,
    },
    /// A user-requested row cleanup completed off the render task.
    Cleanup,
    /// Post-merge: combined-tree verify verdict (None = no verify configured)
    /// + the refreshed base the worktree was reset onto (None = refresh failed).
    PostVerify {
        verify_ok: Option<bool>,
        new_base: Option<String>,
    },
}

pub(crate) type RowFut = Pin<Box<dyn Future<Output = (usize, RowDone)>>>;

/// Persistent fleet execution state. It outlives the dashboard view so child
/// rows and workflow host traffic continue to make progress while chat is open.
pub(crate) struct FleetRuntime {
    line_tx: mpsc::UnboundedSender<(usize, String)>,
    line_rx: mpsc::UnboundedReceiver<(usize, String)>,
    in_flight: FuturesUnordered<RowFut>,
    wf_join_handles:
        std::collections::HashMap<String, tokio::task::JoinHandle<hi_workflow::WorkflowOutcome>>,
}

impl FleetRuntime {
    pub(crate) fn new() -> Self {
        let (line_tx, line_rx) = mpsc::unbounded_channel();
        Self {
            line_tx,
            line_rx,
            in_flight: FuturesUnordered::new(),
            wf_join_handles: std::collections::HashMap::new(),
        }
    }

    fn capture_workflow_handles(&mut self, app: &mut App) {
        for (run_id, run) in &mut app.workflow_runs {
            if !self.wf_join_handles.contains_key(run_id)
                && let Some(handle) = run.join_handle.take()
            {
                self.wf_join_handles.insert(run_id.clone(), handle);
            }
        }
    }
}

/// Service all fleet work that is immediately ready without waiting for it.
pub(crate) async fn pump_fleet(
    app: &mut App,
    launcher: &FleetLauncher,
    runtime: &mut FleetRuntime,
) {
    runtime.capture_workflow_handles(app);
    while let Ok((idx, line)) = runtime.line_rx.try_recv() {
        if let Some(row) = app.fleet.get_mut(idx) {
            row.push_output(&line);
        }
    }
    while let Some(Some((idx, done))) = runtime.in_flight.next().now_or_never() {
        match done {
            RowDone::Turn { ok, killed } => finish_turn(
                app,
                idx,
                ok,
                killed,
                launcher,
                &runtime.line_tx,
                &mut runtime.in_flight,
            ),
            RowDone::MergeCheck { changed, verified } => finish_merge_check(
                app,
                idx,
                changed,
                verified,
                launcher,
                &runtime.line_tx,
                &mut runtime.in_flight,
            ),
            RowDone::MergeApply { changed, result } => finish_merge_apply(
                app,
                idx,
                changed,
                result,
                launcher,
                &runtime.line_tx,
                &mut runtime.in_flight,
            ),
            RowDone::ForceMerge { changed, result } => {
                finish_force_merge(app, idx, changed, result, launcher, &mut runtime.in_flight)
            }
            RowDone::Rebase { base, result } => {
                finish_rebase(app, idx, base, result);
            }
            RowDone::Cleanup => finish_cleanup(app, idx),
            RowDone::PostVerify {
                verify_ok,
                new_base,
            } => finish_post_verify(
                app,
                idx,
                verify_ok,
                new_base,
                launcher,
                &runtime.line_tx,
                &mut runtime.in_flight,
            ),
        }
    }
    loop {
        let next = app.workflow_runs.iter_mut().find_map(|(run_id, run)| {
            run.host_rx
                .as_mut()
                .and_then(|rx| rx.try_recv().ok())
                .map(|req| (run_id.clone(), req))
        });
        let Some((run_id, req)) = next else { break };
        handle_workflow_host_request(
            app,
            &run_id,
            req,
            launcher,
            &runtime.line_tx,
            &mut runtime.in_flight,
        )
        .await;
    }
    let finished: Vec<String> = runtime
        .wf_join_handles
        .iter()
        .filter(|(_, handle)| handle.is_finished())
        .map(|(run_id, _)| run_id.clone())
        .collect();
    for run_id in finished {
        let outcome = match runtime.wf_join_handles.remove(&run_id).unwrap().await {
            Ok(outcome) => outcome,
            Err(_) => hi_workflow::WorkflowOutcome::Failed {
                error: "workflow engine thread panicked".into(),
            },
        };
        if let Some(run) = app.workflow_runs.get_mut(&run_id) {
            run.outcome = Some(outcome.clone());
            run.snapshot.status = (&outcome).into();
            run.snapshot.pause_message = match &outcome {
                hi_workflow::WorkflowOutcome::Paused { message, .. }
                | hi_workflow::WorkflowOutcome::BudgetExceeded { message } => Some(message.clone()),
                _ => None,
            };
            run.snapshot.result_summary = Some(workflow_outcome_summary(&outcome));
            run.snapshot.record_event(
                "workflow_stopped",
                run.snapshot.result_summary.clone(),
                now_ms(),
            );
            let snapshot = run.snapshot.clone();
            app.apply(crate::event::UiEvent::WorkflowUpdated { snapshot });
        }
    }
}

async fn pump_workflow_runs(
    app: &mut App,
    launcher: &FleetLauncher,
    line_tx: &mpsc::UnboundedSender<(usize, String)>,
    in_flight: &mut FuturesUnordered<RowFut>,
    wf_join_handles: &mut std::collections::HashMap<
        String,
        tokio::task::JoinHandle<hi_workflow::WorkflowOutcome>,
    >,
) {
    loop {
        let next = app.workflow_runs.iter_mut().find_map(|(run_id, run)| {
            run.host_rx
                .as_mut()?
                .try_recv()
                .ok()
                .map(|req| (run_id.clone(), req))
        });
        let Some((run_id, req)) = next else { break };
        handle_workflow_host_request(app, &run_id, req, launcher, line_tx, in_flight).await;
    }
    let finished: Vec<_> = wf_join_handles
        .iter()
        .filter(|(_, handle)| handle.is_finished())
        .map(|(id, _)| id.clone())
        .collect();
    for run_id in finished {
        let outcome = match wf_join_handles
            .remove(&run_id)
            .expect("finished handle")
            .await
        {
            Ok(outcome) => outcome,
            Err(_) => hi_workflow::WorkflowOutcome::Failed {
                error: "workflow engine thread panicked".into(),
            },
        };
        if let Some(run) = app.workflow_runs.get_mut(&run_id) {
            run.outcome = Some(outcome.clone());
            run.snapshot.status = (&outcome).into();
            run.snapshot.pause_message = match &outcome {
                hi_workflow::WorkflowOutcome::Paused { message, .. }
                | hi_workflow::WorkflowOutcome::BudgetExceeded { message } => Some(message.clone()),
                _ => None,
            };
            run.snapshot.result_summary = Some(workflow_outcome_summary(&outcome));
            run.snapshot.record_event(
                "workflow_stopped",
                run.snapshot.result_summary.clone(),
                now_ms(),
            );
            let snapshot = run.snapshot.clone();
            app.apply(crate::event::UiEvent::WorkflowUpdated { snapshot });
        }
    }
}

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
    runtime: &mut FleetRuntime,
    adopt: Option<crate::FleetResumeInfo>,
) -> Result<()> {
    let FleetRuntime {
        line_tx,
        line_rx,
        in_flight,
        wf_join_handles,
    } = runtime;
    let mut selected: usize = app.fleet.len().saturating_sub(1);
    // `/fleet resume [id]`: re-adopt a past session as a row before the loop
    // starts (needs the loop's channels for its first drive turn).
    let mut adopt_flash: Option<String> = None;
    if let Some(info) = adopt {
        match adopt_session(app, info, launcher, line_tx, in_flight).await {
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
    for (run_id, run) in &mut app.workflow_runs {
        if !wf_join_handles.contains_key(run_id)
            && let Some(handle) = run.join_handle.take()
        {
            wf_join_handles.insert(run_id.clone(), handle);
        }
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
                        finish_turn(app, idx, ok, killed, launcher, line_tx, in_flight);
                    }
                    RowDone::MergeCheck { changed, verified } => {
                        finish_merge_check(app, idx, changed, verified, launcher, line_tx, in_flight);
                    }
                    RowDone::MergeApply { changed, result } => {
                        finish_merge_apply(app, idx, changed, result, launcher, line_tx, in_flight);
                    }
                    RowDone::ForceMerge { changed, result } => {
                        finish_force_merge(app, idx, changed, result, launcher, in_flight);
                    }
                    RowDone::Rebase { base, result } => {
                        finish_rebase(app, idx, base, result);
                    }
                    RowDone::Cleanup => finish_cleanup(app, idx),
                    RowDone::PostVerify { verify_ok, new_base } => {
                        finish_post_verify(app, idx, verify_ok, new_base, launcher, line_tx, in_flight);
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
                pump_workflow_runs(app, launcher, line_tx, in_flight, wf_join_handles).await;
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
                                // Closing the view never cancels fleet work.
                                return Ok(());
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
                                let base = hi_tools::checkpoint::create(&app.workspace_root).await;
                                if let Some(error) =
                                    queue_rebase(app, selected, base, in_flight)
                                {
                                    flash = Some(error);
                                }
                            }
                            // Ctrl+S: dispatch AND attach (or attach the selected
                            // row when the dispatch box is empty).
                            KeyCode::Char('s') if ctrl => {
                                let text = dispatch.submit();
                                let text = text.trim().to_string();
                                if !text.is_empty() {
                                    match dispatch_new(app, text, launcher, line_tx, in_flight).await {
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
                                        match dispatch_new(app, text, launcher, line_tx, in_flight).await {
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
                                            send_reply(app, selected, text, launcher, line_tx, in_flight);
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
                                send_reply(app, selected, c.to_string(), launcher, line_tx, in_flight);
                            }
                            // m: force-merge the selected row's diff (held or
                            // verify-failed) into the real tree.
                            KeyCode::Char('m')
                                if focus == Focus::Dispatch
                                    && app.fleet.get(selected).is_some_and(|r| {
                                        r.state != RowState::Working && r.state != RowState::Closed
                                    }) =>
                            {
                                if let Some(error) =
                                    queue_force_merge(app, selected, in_flight)
                                {
                                    flash = Some(error);
                                }
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
                                    let cleanup_root = app.workspace_root.clone();
                                    let cleanup_path = row.worktree.clone();
                                    row.state = RowState::Working;
                                    row.activity = "closing…".to_string();
                                    in_flight.push(Box::pin(async move {
                                        let _ = tokio::task::spawn_blocking(move || {
                                            worktree::cleanup(
                                                &cleanup_root,
                                                std::slice::from_ref(&cleanup_path),
                                            );
                                        })
                                        .await;
                                        (selected, RowDone::Cleanup)
                                    }));
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
    for run in app.workflow_runs.values() {
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
    let workspace_root = app.workspace_root.clone();
    let in_git = tokio::task::spawn_blocking({
        let workspace_root = workspace_root.clone();
        move || worktree::in_git_repo(&workspace_root)
    })
    .await
    .context("git repository probe worker failed")?;
    if !in_git {
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
    let add_root = workspace_root;
    let add_path = path.clone();
    let add_base = base.clone();
    tokio::task::spawn_blocking(move || worktree::add_worktree(&add_root, &add_path, &add_base))
        .await
        .context("worktree setup worker failed")??;
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
        workflow_run_id: None,
        workflow_phase: None,
        workflow_label: None,
        workflow_status: None,
        workflow_schema: None,
        workflow_schema_retry_used: false,
    };
    app.fleet.push(row);
    let idx = app.fleet.len() - 1;
    start_turn(app, idx, first_prompt, launcher, line_tx, in_flight);
    Ok(idx)
}

fn collect_workflow_phases(steps: &[hi_workflow::DeclarativeStep], phases: &mut Vec<String>) {
    for step in steps {
        match step {
            hi_workflow::DeclarativeStep::Phase { title } if !phases.contains(title) => {
                phases.push(title.clone())
            }
            hi_workflow::DeclarativeStep::IfAgentSuccess {
                then_steps,
                else_steps,
                ..
            } => {
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
        definition
            .validate()
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
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
        .or_else(|| {
            std::env::var_os("HOME").map(|home| std::path::PathBuf::from(home).join(".local/state"))
        })
        .map(|base| base.join("hi/workflow-runs"));
    let run_id = format!(
        "run-{}-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis(),
        std::process::id(),
    );
    let journal = if let Some(root) = journal_path {
        let store = hi_workflow::WorkflowRunStore::new(root);
        let manifest = hi_workflow::WorkflowRunManifest::new(
            run_id.clone(),
            workflow_name.clone(),
            hi_workflow::DEFAULT_AGENT_BUDGET,
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
                    hi_workflow::DeclarativeOutcome::Completed { result, .. } => {
                        hi_workflow::WorkflowOutcome::Completed { result }
                    }
                    hi_workflow::DeclarativeOutcome::Paused { kind, message, .. } => {
                        hi_workflow::WorkflowOutcome::Paused { kind, message }
                    }
                    hi_workflow::DeclarativeOutcome::Cancelled { .. } => {
                        hi_workflow::WorkflowOutcome::Cancelled
                    }
                    hi_workflow::DeclarativeOutcome::BudgetExceeded { message, .. } => {
                        hi_workflow::WorkflowOutcome::BudgetExceeded { message }
                    }
                    hi_workflow::DeclarativeOutcome::Failed { error, .. } => {
                        hi_workflow::WorkflowOutcome::Failed {
                            error: error.to_string(),
                        }
                    }
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

    let snapshot = hi_workflow::WorkflowRunSnapshot {
        run_id: run_id.clone(),
        revision: 1,
        workflow_name: workflow_name.clone(),
        objective: workflow_description.clone(),
        status: hi_workflow::WorkflowRunStatus::Active,
        phases: phases
            .iter()
            .map(|(title, state)| hi_workflow::WorkflowPhaseSnapshot {
                title: title.clone(),
                state: state.clone(),
            })
            .collect(),
        current_phase: None,
        agents: Vec::new(),
        agent_budget: hi_workflow::DEFAULT_AGENT_BUDGET,
        agents_used: 0,
        agents_reserved: 0,
        elapsed_ms: 0,
        pause_message: None,
        result_summary: None,
        history: Vec::new(),
    };
    app.apply(crate::event::UiEvent::WorkflowUpdated {
        snapshot: snapshot.clone(),
    });
    let run = WorkflowRun {
        snapshot,
        run_id: run_id.clone(),
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
    };
    app.selected_workflow_run = Some(run_id.clone());
    app.workflow_runs.insert(run_id, run);

    Ok(())
}

/// Service a `WorkflowHostRequest` that arrived from the engine thread. Called
/// from the dashboard's `select!` loop. Returns `true` if the request was
/// handled, `false` if it was a `SpawnAgent` that needs a turn to complete
/// (the reply is stored on the row and sent in `finish_turn`).
pub(crate) async fn handle_workflow_host_request(
    app: &mut App,
    run_id: &str,
    req: hi_workflow::WorkflowHostRequest,
    launcher: &FleetLauncher,
    line_tx: &mpsc::UnboundedSender<(usize, String)>,
    in_flight: &mut FuturesUnordered<RowFut>,
) {
    use hi_workflow::WorkflowHostRequest as R;
    let mut publish_snapshot = false;
    match req {
        R::SpawnAgent { opts, reply } => {
            // Create a FleetRow for this agent, start a turn, and store the
            // reply sender. When the turn completes, finish_turn sends the
            // AgentResult back so the workflow can continue.
            spawn_workflow_agent(app, run_id, opts, reply, launcher, line_tx, in_flight).await;
        }
        R::Phase { title, replayed } => {
            if let Some(run) = app.workflow_runs.get_mut(run_id)
                && !replayed
            {
                run.on_phase(&title);
                publish_snapshot = true;
            }
        }
        R::Log { message, replayed } => {
            if let Some(run) = app.workflow_runs.get_mut(run_id)
                && !replayed
            {
                run.log.push(message.clone());
                run.snapshot
                    .record_event("workflow_log", Some(message), now_ms());
                publish_snapshot = true;
            }
        }
        R::Telemetry {
            name,
            fields,
            replayed,
        } => {
            if let Some(run) = app.workflow_runs.get_mut(run_id)
                && !replayed
            {
                run.log.push(format!("telemetry: {name} {fields}"));
                run.snapshot
                    .record_event("workflow_telemetry", Some(name), now_ms());
                publish_snapshot = true;
            }
        }
        R::BudgetQuery { reply } => {
            let state = app
                .workflow_runs
                .get(run_id)
                .map(|run| hi_workflow::BudgetState {
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
            let result = app
                .workflow_runs
                .get_mut(run_id)
                .ok_or_else(|| {
                    hi_workflow::HostError::Failed("workflow run is no longer active".into())
                })
                .and_then(|run| {
                    let requested = run
                        .agent_spent
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
            let result = app
                .workflow_runs
                .get_mut(run_id)
                .ok_or_else(|| {
                    hi_workflow::HostError::Failed("workflow run is no longer active".into())
                })
                .and_then(|run| {
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
        R::WriteScratchFile {
            name,
            content,
            reply,
        } => match workflow_scratch_path(app, run_id, &name) {
            Err(error) => {
                let _ = reply.send(Err(error));
            }
            Ok(_path) if content.len() > 1024 * 1024 => {
                let _ = reply.send(Err(hi_workflow::HostError::Failed(
                    "scratch file exceeds 1 MiB".into(),
                )));
            }
            Ok(path) => {
                tokio::spawn(async move {
                    let result = tokio::task::spawn_blocking(move || {
                        if let Some(parent) = path.parent() {
                            std::fs::create_dir_all(parent)
                                .map_err(|e| hi_workflow::HostError::Failed(e.to_string()))?;
                        }
                        std::fs::write(&path, &content)
                            .map_err(|e| hi_workflow::HostError::Failed(e.to_string()))?;
                        Ok(path.display().to_string())
                    })
                    .await
                    .map_err(|error| {
                        hi_workflow::HostError::Failed(format!("scratch worker failed: {error}"))
                    })
                    .and_then(|result| result);
                    let _ = reply.send(result);
                });
            }
        },
        R::ReadScratchFile { name, reply } => match workflow_scratch_path(app, run_id, &name) {
            Err(error) => {
                let _ = reply.send(Err(error));
            }
            Ok(path) => {
                tokio::spawn(async move {
                    let result = tokio::task::spawn_blocking(move || {
                        let meta = std::fs::metadata(&path)
                            .map_err(|e| hi_workflow::HostError::Failed(e.to_string()))?;
                        if meta.len() > 1024 * 1024 {
                            return Err(hi_workflow::HostError::Failed(
                                "scratch file exceeds 1 MiB".into(),
                            ));
                        }
                        std::fs::read_to_string(path)
                            .map_err(|e| hi_workflow::HostError::Failed(e.to_string()))
                    })
                    .await
                    .map_err(|error| {
                        hi_workflow::HostError::Failed(format!("scratch worker failed: {error}"))
                    })
                    .and_then(|result| result);
                    let _ = reply.send(result);
                });
            }
        },
        R::GitDiffSince { commit, reply } => {
            let valid = !commit.is_empty()
                && commit.len() <= 128
                && commit.bytes().all(|b| b.is_ascii_hexdigit());
            if !valid {
                let _ = reply.send(Err(hi_workflow::HostError::Failed(
                    "invalid commit id".into(),
                )));
            } else {
                let workspace_root = app.workspace_root.clone();
                tokio::spawn(async move {
                    let result = match hi_tools::ProcessRunner::new(&workspace_root) {
                        Ok(runner) => {
                            let args = vec![
                                std::ffi::OsString::from("diff"),
                                std::ffi::OsString::from("--no-ext-diff"),
                                std::ffi::OsString::from(commit),
                                std::ffi::OsString::from("--"),
                            ];
                            match runner
                                .run_program("git", args, std::time::Duration::from_secs(60))
                                .await
                            {
                                Ok(execution)
                                    if execution.status == hi_tools::ToolStatus::Succeeded =>
                                {
                                    let output = execution.outcome.stdout_summary;
                                    if output.len() > 256 * 1024 {
                                        Err(hi_workflow::HostError::Failed(
                                            "git diff exceeds 256 KiB".into(),
                                        ))
                                    } else {
                                        Ok(output)
                                    }
                                }
                                Ok(execution) => {
                                    Err(hi_workflow::HostError::Failed(execution.model_content()))
                                }
                                Err(error) => {
                                    Err(hi_workflow::HostError::Failed(error.to_string()))
                                }
                            }
                        }
                        Err(error) => Err(hi_workflow::HostError::Failed(error.to_string())),
                    };
                    let _ = reply.send(result);
                });
            }
        }
    }
    if publish_snapshot
        && let Some(snapshot) = app
            .workflow_runs
            .get(run_id)
            .map(|run| run.snapshot.clone())
    {
        app.apply(crate::event::UiEvent::WorkflowUpdated { snapshot });
    }
}

fn workflow_scratch_path(
    app: &App,
    run_id: &str,
    name: &str,
) -> Result<std::path::PathBuf, hi_workflow::HostError> {
    if name.is_empty()
        || name.len() > 255
        || name.contains('/')
        || name.contains('\\')
        || name == "."
        || name == ".."
    {
        return Err(hi_workflow::HostError::Failed(
            "invalid scratch file name".into(),
        ));
    }
    if !app.workflow_runs.contains_key(run_id) {
        return Err(hi_workflow::HostError::Cancelled);
    }
    Ok(std::env::temp_dir()
        .join("hi-workflows")
        .join(run_id)
        .join(name))
}

/// Create a `FleetRow` for a workflow `SpawnAgent` request, start the child
/// turn, and store the reply sender so `finish_turn` can send the result back.
async fn spawn_workflow_agent(
    app: &mut App,
    run_id: &str,
    opts: hi_workflow::AgentOpts,
    reply: oneshot::Sender<Result<hi_workflow::AgentResult, hi_workflow::HostError>>,
    launcher: &FleetLauncher,
    line_tx: &mpsc::UnboundedSender<(usize, String)>,
    in_flight: &mut FuturesUnordered<RowFut>,
) {
    let workspace_root = app.workspace_root.clone();
    let in_git = tokio::task::spawn_blocking({
        let workspace_root = workspace_root.clone();
        move || worktree::in_git_repo(&workspace_root)
    })
    .await
    .unwrap_or(false);
    if !in_git {
        let _ = reply.send(Err(hi_workflow::HostError::Failed(
            "not in a git repository (workflow agents need worktrees)".into(),
        )));
        return;
    }

    let mut prompt = opts.prompt.clone();
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
    let base = match hi_tools::checkpoint::create(&workspace_root).await {
        Some(commit) => commit,
        None => {
            let _ = reply.send(Err(hi_workflow::HostError::Failed(
                "couldn't snapshot the working tree".into(),
            )));
            return;
        }
    };
    app.fleet_next_id += 1;
    let id = app.fleet_next_id;
    let path = worktree::worktree_path("fleet", id as u32);
    let add_root = workspace_root.clone();
    let add_path = path.clone();
    let add_base = base.clone();
    let add_result = tokio::task::spawn_blocking(move || {
        worktree::add_worktree(&add_root, &add_path, &add_base)
    })
    .await
    .unwrap_or_else(|error| Err(anyhow!("worktree setup worker failed: {error}")));
    if let Err(err) = add_result {
        let _ = reply.send(Err(hi_workflow::HostError::Failed(format!(
            "couldn't create worktree: {err}"
        ))));
        return;
    }
    let session = match (launcher.session_path)() {
        Ok(s) => s,
        Err(err) => {
            let _ = tokio::task::spawn_blocking({
                let path = path.clone();
                move || std::fs::remove_dir_all(path)
            })
            .await;
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
        workflow_run_id: Some(run_id.to_string()),
        workflow_phase: phase,
        workflow_label: label,
        workflow_status: Some(WorkflowJobStatus::Running),
        workflow_schema: opts.output_schema.clone(),
        workflow_schema_retry_used: false,
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
    let workspace_root = app.workspace_root.clone();
    let in_git = tokio::task::spawn_blocking({
        let workspace_root = workspace_root.clone();
        move || worktree::in_git_repo(&workspace_root)
    })
    .await
    .context("git repository probe worker failed")?;
    if !in_git {
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
    let add_root = workspace_root;
    let add_path = path.clone();
    let add_base = base.clone();
    tokio::task::spawn_blocking(move || worktree::add_worktree(&add_root, &add_path, &add_base))
        .await
        .context("worktree setup worker failed")??;
    let goal = (info.goal_total > 0).then_some(RowGoal {
        done: info.goal_done,
        total: info.goal_total,
        active: info.goal_active,
        paused: false,
        phases: Vec::new(),
    });
    // Preload the peek tail with the session's conversation so attach shows
    // history immediately, before any new turn runs.
    let tail = load_transcript_async(info.path.clone(), TAIL_CAP).await;
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
        workflow_run_id: None,
        workflow_phase: None,
        workflow_label: None,
        workflow_status: None,
        workflow_schema: None,
        workflow_schema_retry_used: false,
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
    if cap == 0 {
        return Vec::new();
    }
    // Keep resume-time memory and parsing bounded by retaining only the tail
    // while streaming the JSONL session. A long-lived fleet session can be
    // hundreds of megabytes; reading the whole file just to show 200 lines
    // used to freeze the dashboard and temporarily double its memory use.
    const MAX_DISPLAY_LINE_CHARS: usize = 2_000;
    let Ok(file) = std::fs::File::open(path) else {
        return Vec::new();
    };
    let reader = StdBufReader::new(file);
    let mut lines = VecDeque::with_capacity(cap);
    let mut push = |line: String| {
        if lines.len() == cap {
            lines.pop_front();
        }
        lines.push_back(truncate(&line, MAX_DISPLAY_LINE_CHARS));
    };
    for line in reader.lines().map_while(std::result::Result::ok) {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&line) else {
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
                            push(format!("› {}", truncate(first, 100)));
                        }
                    }
                }
            }
            hi_ai::Role::Assistant => {
                for c in &msg.content {
                    match c {
                        hi_ai::Content::Text(t) => {
                            for line in t
                                .lines()
                                .map(str::trim_end)
                                .filter(|line| !line.trim().is_empty())
                            {
                                push(line.to_string());
                            }
                        }
                        hi_ai::Content::ToolCall {
                            name, arguments, ..
                        } => push(format!("⚙ {}", hi_agent::ui::tool_label(name, arguments))),
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
    lines.into_iter().collect()
}

async fn load_transcript_async(path: PathBuf, cap: usize) -> Vec<String> {
    tokio::task::spawn_blocking(move || load_transcript(&path, cap))
        .await
        .unwrap_or_default()
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
        if row.pending.len() >= MAX_ROW_PENDING {
            row.push_line(format!(
                "⚠ queue full ({}/{}) — finish or drop a pending prompt",
                row.pending.len(),
                MAX_ROW_PENDING
            ));
            return;
        }
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

    // A report belongs to exactly one child turn. Remove the previous report
    // before spawning so a child that crashes before writing cannot make the
    // parent consume stale goal progress and launch another drive.
    let _ = std::fs::remove_file(report_path(row));
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
    let report = std::fs::read_to_string(report_path(row))
        .ok()
        .and_then(|t| parse_report(&t));
    if let Some(report) = report {
        let goal_disappeared = (was_driving || row.goal.is_some()) && report.goal.is_none();
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
        if goal_disappeared {
            // A syntactically valid report with `goal: null` is still unsafe
            // during a goal-driven row: the child may have failed to restore
            // its session or accidentally cleared the durable goal. Stop the
            // autonomous chain and explain why instead of silently idling.
            row.last_goal_json = None;
            row.push_line(
                "⚠ goal progress disappeared — automatic drive paused; reply to resume".to_string(),
            );
        }
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
    } else if was_driving || row.goal.is_some() {
        // Do not trust the previous report after a crash, early init failure,
        // or malformed child output. Clearing the cached goal stops another
        // synthetic drive; the user can reply to resume once the child is
        // healthy and a fresh report is available.
        row.goal = None;
        row.last_goal_json = None;
        row.push_line(
            "⚠ goal progress report missing — automatic drive paused; reply to resume".to_string(),
        );
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
    if row.workflow_reply.is_none() {
        return false;
    }
    let mut output = std::fs::read_to_string(report_path(row))
        .ok()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        .and_then(|report| report.get("assistant_response").cloned())
        .unwrap_or_else(|| serde_json::json!({"summary": summary}));
    if let Some(schema) = &row.workflow_schema {
        output = coerce_structured_output(output);
        // A successful turn whose reply doesn't match the declared schema gets
        // exactly one corrective follow-up before the engine sees it. The
        // prompt rides the row's pending queue; returning false lets the
        // caller fall through to `continue_row`, which dispatches it.
        if success
            && !row.workflow_schema_retry_used
            && let Err(error) = hi_workflow::validate_output_schema(&output, schema)
        {
            row.workflow_schema_retry_used = true;
            row.workflow_status = Some(WorkflowJobStatus::Running);
            row.push_line(format!(
                "⚠ structured output rejected ({error}) — requesting corrected JSON"
            ));
            if row.pending.len() < MAX_ROW_PENDING {
                row.pending.push_back(format!(
                    "Your previous reply did not match the required output schema: {error}\n\n\
                     Respond again with ONLY a single JSON object matching the schema — \
                     no prose before or after it, no markdown code fences."
                ));
            } else {
                row.push_line(format!(
                    "⚠ schema retry not queued — row queue full ({MAX_ROW_PENDING})"
                ));
            }
            return false;
        }
    }
    let Some(reply) = row.workflow_reply.take() else {
        return false;
    };
    row.workflow_status = Some(if success {
        WorkflowJobStatus::Completed
    } else {
        WorkflowJobStatus::Failed
    });
    let _ = reply.send(Ok(hi_workflow::AgentResult {
        agent_id: format!("#{}", row.id),
        success,
        output,
        cancelled: false,
        tokens_used: row.usage,
        duration_ms: 0,
    }));
    true
}

/// A workflow agent was asked for schema-shaped output, but fleet children
/// reply with free text: recover the JSON object from the reply when possible
/// (the whole reply, or the outermost `{…}` span when prose or code fences
/// surround it). Scripts treat unrecoverable replies as failed structured
/// output, so the original string is returned unchanged on a parse failure.
fn coerce_structured_output(value: serde_json::Value) -> serde_json::Value {
    let serde_json::Value::String(text) = &value else {
        return value;
    };
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
        let workflow_run_id = row.workflow_run_id.clone();
        if finish_workflow_agent(row, true, "completed without workspace changes".into()) {
            if let Some(run) = workflow_run_id
                .as_deref()
                .and_then(|id| app.workflow_runs.get_mut(id))
            {
                run.agent_reserved = run.agent_reserved.saturating_sub(1);
                run.agent_spent = run.agent_spent.saturating_add(1);
            }
            return;
        }
    } else if !verified {
        row.merge = MergeState::VerifyFailed;
        row.push_line("⇡ verify failed in the worktree — not merged (m forces)".to_string());
        if finish_workflow_agent(
            row,
            false,
            "worktree verification failed; changes were not merged".into(),
        ) {
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
        if finish_workflow_agent(
            row,
            false,
            format!("merge held because files overlap rows {overlaps:?}"),
        ) {
            return;
        }
    } else {
        // Applying a verified diff still runs git and can be slow on a large
        // tree. Keep it off the render/input task; the completion handler below
        // owns the UI state transition and post-merge verification.
        let worktree_path = row.worktree.clone();
        let base = row.base.clone();
        let destination = app.workspace_root.clone();
        let changed_for_apply = row.changed.clone();
        row.state = RowState::Working;
        row.activity = "merging…".to_string();
        in_flight.push(Box::pin(async move {
            let result = tokio::task::spawn_blocking(move || {
                worktree::apply_changes_to(&worktree_path, &base, &destination)
                    .map(|_| ())
                    .map_err(|error| format!("{error:#}"))
            })
            .await
            .unwrap_or_else(|error| Err(format!("merge worker failed: {error}")));
            (
                idx,
                RowDone::MergeApply {
                    changed: changed_for_apply,
                    result,
                },
            )
        }));
        return;
    }
    continue_row(app, idx, launcher, line_tx, in_flight);
}

/// The verified diff was applied by a blocking worker. Record the merge and
/// queue the combined-tree verification/base refresh without touching the real
/// tree from the render task.
fn finish_merge_apply(
    app: &mut App,
    idx: usize,
    changed: Vec<String>,
    result: Result<(), String>,
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
    if let Err(error) = result {
        row.merge = MergeState::VerifyFailed;
        row.push_line(format!("✗ merge failed: {error} (m retries)"));
        if finish_workflow_agent(row, false, format!("merge failed: {error}")) {
            return;
        }
        continue_row(app, idx, launcher, line_tx, in_flight);
        return;
    }

    row.merge = MergeState::Merged(changed.len());
    row.changed = changed.clone();
    row.push_line(format!(
        "✓ merged {} file(s) into your tree: {}",
        changed.len(),
        changed.join(", ")
    ));
    record_fleet(
        launcher,
        row.id,
        &row.title,
        &format!("merged {} file(s): {}", changed.len(), changed.join(", ")),
    );
    mark_others_stale(app, idx);
    queue_post_merge_verify(app, idx, launcher, in_flight);
}

/// Verify the combined explicit workspace root and refresh the row's base.
/// Both the potentially slow verification and checkpoint/reset operations stay
/// off the render task; the explicit root avoids depending on process cwd.
fn queue_post_merge_verify(
    app: &mut App,
    idx: usize,
    launcher: &FleetLauncher,
    in_flight: &mut FuturesUnordered<RowFut>,
) {
    let verify = launcher.verify.clone();
    let workspace_root = app.workspace_root.clone();
    let worktree_path = app.fleet.get(idx).map(|row| row.worktree.clone());
    let Some(worktree_path) = worktree_path else {
        return;
    };
    let Some(row) = app.fleet.get_mut(idx) else {
        return;
    };
    row.state = RowState::Working;
    row.activity = "post-merge check…".to_string();
    in_flight.push(Box::pin(async move {
        let verify_ok = match &verify {
            Some(v) => {
                let root = workspace_root.clone();
                let v = v.clone();
                Some(
                    tokio::task::spawn_blocking(move || worktree::verify_passes(&root, &v))
                        .await
                        .unwrap_or(false),
                )
            }
            None => None,
        };
        let new_base = hi_tools::checkpoint::create(&workspace_root).await;
        let new_base = match new_base {
            Some(base) => {
                let wt = worktree_path.clone();
                let sha = base.clone();
                let reset_ok =
                    tokio::task::spawn_blocking(move || worktree::reset_to(&wt, &sha).is_ok())
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
    let workflow_run_id = row.workflow_run_id.clone();
    if finish_workflow_agent(row, success, summary) {
        if let Some(run) = workflow_run_id
            .as_deref()
            .and_then(|id| app.workflow_runs.get_mut(id))
        {
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
            row.push_line(format!(
                "⏸ drive parked — no progress for {} turns; reply to steer and resume",
                hi_agent::GOAL_DRIVE_STALL_LIMIT
            ));
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
    let Some(rest) = prompt.strip_prefix("/goal") else {
        return (None, prompt);
    };
    if !rest.chars().next().is_some_and(char::is_whitespace) {
        return (None, prompt);
    }
    let objective = rest.trim().to_string();
    if objective.is_empty() {
        (None, prompt)
    } else {
        (Some(objective.clone()), objective)
    }
}

/// `m`: apply the selected row's diff to the real tree regardless of holds.
/// The potentially slow git diff/apply work runs in the fleet future pool so a
/// force merge cannot freeze input or rendering.
fn queue_force_merge(
    app: &mut App,
    idx: usize,
    in_flight: &mut FuturesUnordered<RowFut>,
) -> Option<String> {
    let Some(row) = app.fleet.get_mut(idx) else {
        return Some("selected fleet row no longer exists".to_string());
    };
    let worktree_path = row.worktree.clone();
    let base = row.base.clone();
    let destination = app.workspace_root.clone();
    row.state = RowState::Working;
    row.activity = "force merging…".to_string();
    row.attention = false;
    in_flight.push(Box::pin(async move {
        let result = tokio::task::spawn_blocking(move || {
            let changed = worktree::changed_files(&worktree_path, &base);
            if changed.is_empty() {
                return (changed, Ok(()));
            }
            let result = worktree::apply_changes_to(&worktree_path, &base, &destination)
                .map(|_| ())
                .map_err(|error| format!("{error:#}"));
            (changed, result)
        })
        .await
        .unwrap_or_else(|error| (Vec::new(), Err(format!("merge worker failed: {error}"))));
        (
            idx,
            RowDone::ForceMerge {
                changed: result.0,
                result: result.1,
            },
        )
    }));
    None
}

fn finish_force_merge(
    app: &mut App,
    idx: usize,
    changed: Vec<String>,
    result: Result<(), String>,
    launcher: &FleetLauncher,
    in_flight: &mut FuturesUnordered<RowFut>,
) {
    let Some(row) = app.fleet.get_mut(idx) else {
        return;
    };
    row.state = RowState::Idle;
    row.started = None;
    row.activity.clear();
    if let Err(error) = result {
        row.merge = MergeState::VerifyFailed;
        row.push_line(format!("✗ force merge failed: {error}"));
        flag_attention(app, idx);
        return;
    }
    if changed.is_empty() {
        row.push_line("nothing to merge".to_string());
        flag_attention(app, idx);
        return;
    }
    row.changed = changed.clone();
    row.merge = MergeState::Merged(changed.len());
    row.push_line(format!(
        "✓ merged {} file(s) into your tree (forced)",
        changed.len()
    ));
    record_fleet(
        launcher,
        row.id,
        &row.title,
        &format!(
            "force-merged {} file(s): {}",
            changed.len(),
            changed.join(", ")
        ),
    );
    mark_others_stale(app, idx);
    queue_post_merge_verify(app, idx, launcher, in_flight);
}

/// `r`: rebase an idle row's worktree onto a fresh snapshot of the real tree.
/// Refused while the row has unmerged changes (merge or close first).
fn queue_rebase(
    app: &mut App,
    idx: usize,
    new_base: Option<String>,
    in_flight: &mut FuturesUnordered<RowFut>,
) -> Option<String> {
    let Some(row) = app.fleet.get_mut(idx) else {
        return Some("selected fleet row no longer exists".to_string());
    };
    let unmerged = !row.changed.is_empty() && !matches!(row.merge, MergeState::Merged(_));
    if unmerged {
        return Some(format!(
            "#{}: unmerged changes — merge (m) or close (x) first",
            row.id
        ));
    }
    let Some(base) = new_base else {
        return Some(format!("#{}: couldn't snapshot the tree", row.id));
    };
    let worktree_path = row.worktree.clone();
    let base_for_reset = base.clone();
    row.state = RowState::Working;
    row.activity = "rebasing…".to_string();
    row.attention = false;
    in_flight.push(Box::pin(async move {
        let result = tokio::task::spawn_blocking(move || {
            worktree::reset_to(&worktree_path, &base_for_reset)
                .map_err(|error| format!("{error:#}"))
        })
        .await
        .unwrap_or_else(|error| Err(format!("rebase worker failed: {error}")));
        (idx, RowDone::Rebase { base, result })
    }));
    None
}

fn finish_rebase(app: &mut App, idx: usize, base: String, result: Result<(), String>) {
    let Some(row) = app.fleet.get_mut(idx) else {
        return;
    };
    row.state = RowState::Idle;
    row.started = None;
    row.activity.clear();
    match result {
        Ok(()) => {
            row.base = base;
            row.changed.clear();
            row.stale = false;
            row.attention = false;
            row.push_line("⟳ rebased onto the current tree".to_string());
        }
        Err(error) => {
            row.push_line(format!("✗ rebase failed: {error}"));
            flag_attention(app, idx);
        }
    }
}

fn finish_cleanup(app: &mut App, idx: usize) {
    let Some(row) = app.fleet.get_mut(idx) else {
        return;
    };
    row.state = RowState::Closed;
    row.started = None;
    row.activity.clear();
    row.push_line("row closed — worktree removed; session remains resumable".to_string());
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

    let selected_run = app
        .selected_workflow_run
        .as_deref()
        .and_then(|id| app.workflow_runs.get(id));
    let title = if let Some(run) = selected_run {
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
    if app
        .selected_workflow_run
        .as_deref()
        .and_then(|id| app.workflow_runs.get(id))
        .is_none()
    {
        return 0;
    }
    let selected = app.selected_workflow_run.as_deref();
    phase_header_count(
        &app.fleet
            .iter()
            .filter(|row| row.workflow_run_id.as_deref() == selected)
            .collect::<Vec<_>>(),
    )
}

fn phase_header_count<R: std::borrow::Borrow<FleetRow>>(rows: &[R]) -> usize {
    let mut count = 0;
    let mut last: Option<&str> = None;
    for row in rows {
        let row = row.borrow();
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
    let selected_run = app.selected_workflow_run.as_deref();
    let group_phases = selected_run.is_some();
    let mut last_phase: Option<&str> = None;
    for (i, row) in app
        .fleet
        .iter()
        .enumerate()
        .filter(|(_, row)| !group_phases || row.workflow_run_id.as_deref() == selected_run)
        .skip(start)
        .take(inner_rows)
    {
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

    fn test_fleet_launcher() -> FleetLauncher {
        FleetLauncher {
            exe: PathBuf::from("/bin/false"),
            workspace_root: PathBuf::from("/tmp"),
            provider: "test".into(),
            model: "test".into(),
            base_url: String::new(),
            api_key: String::new(),
            verify: None,
            max_verify: 1,
            max_steps: 1,
            session_path: Box::new(|| Ok(PathBuf::from("/tmp/test-session.jsonl"))),
            sessions: Box::new(Vec::new),
            resume_info: Box::new(|_| None),
            loop_session_path: Box::new(|| Ok(PathBuf::from("/tmp/test-loop.jsonl"))),
            loops_file: None,
        }
    }

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
            workflow_run_id: None,
            workflow_phase: None,
            workflow_label: None,
            workflow_status: None,
            workflow_schema: None,
            workflow_schema_retry_used: false,
        }
    }

    #[test]
    fn phase_headers_count_contiguous_phase_runs() {
        let phased = |phase: Option<&str>| {
            let mut r = row();
            r.workflow_phase = phase.map(str::to_string);
            r
        };
        assert_eq!(phase_header_count(&[] as &[FleetRow]), 0);
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
        let (obj, prompt) = split_goal_dispatch("/goal\t  port the parser to Rust".to_string());
        assert_eq!(obj.as_deref(), Some("port the parser to Rust"));
        assert_eq!(prompt, "port the parser to Rust");
        let (obj, prompt) = split_goal_dispatch("/goalkeeper ship it".to_string());
        assert!(obj.is_none());
        assert_eq!(prompt, "/goalkeeper ship it");
        let (obj, prompt) = split_goal_dispatch("fix the failing test".to_string());
        assert!(obj.is_none());
        assert_eq!(prompt, "fix the failing test");
    }

    #[tokio::test]
    async fn fleet_pump_is_nonblocking_without_work() {
        let mut app = crate::tests::test_app("openai", "gpt-4o");
        let mut runtime = FleetRuntime::new();
        let launcher = test_fleet_launcher();
        tokio::time::timeout(
            std::time::Duration::from_millis(100),
            pump_fleet(&mut app, &launcher, &mut runtime),
        )
        .await
        .expect("idle fleet pump must not block");
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
            snapshot: hi_workflow::WorkflowRunSnapshot {
                run_id: "test-run".into(),
                revision: 0,
                workflow_name: "test".into(),
                objective: "test".into(),
                status: hi_workflow::WorkflowRunStatus::Active,
                phases: vec![],
                current_phase: None,
                agents: vec![],
                agent_budget: hi_workflow::DEFAULT_AGENT_BUDGET,
                agents_used: 0,
                agents_reserved: 0,
                elapsed_ms: 0,
                pause_message: None,
                result_summary: None,
                history: vec![],
            },
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
            snapshot: hi_workflow::WorkflowRunSnapshot {
                run_id: "test-run".into(),
                revision: 0,
                workflow_name: "test".into(),
                objective: "test".into(),
                status: hi_workflow::WorkflowRunStatus::Active,
                phases: vec![],
                current_phase: None,
                agents: vec![],
                agent_budget: hi_workflow::DEFAULT_AGENT_BUDGET,
                agents_used: 0,
                agents_reserved: 0,
                elapsed_ms: 0,
                pause_message: None,
                result_summary: None,
                history: vec![],
            },
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
