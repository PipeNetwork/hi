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
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Paragraph};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::{mpsc, oneshot};

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
}

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
enum RowDone {
    /// The child turn exited.
    Turn { ok: bool, killed: bool },
    /// The off-thread merge check finished (diff vs base + verify verdict).
    MergeCheck {
        changed: Vec<String>,
        verified: bool,
    },
}

type RowFut = Pin<Box<dyn Future<Output = (usize, RowDone)>>>;

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
) -> Result<()> {
    let (line_tx, mut line_rx) = mpsc::unbounded_channel::<(usize, String)>();
    let mut in_flight: FuturesUnordered<RowFut> = FuturesUnordered::new();
    let mut selected: usize = app.fleet.len().saturating_sub(1);
    let mut focus = Focus::Dispatch;
    let mut dispatch = InputLine::default();
    let mut exit_armed = false;
    let mut flash: Option<String> = None;

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
            )
        })?;

        tokio::select! {
            Some((idx, done)) = in_flight.next(), if !in_flight.is_empty() => {
                match done {
                    RowDone::Turn { ok, killed } => {
                        finish_turn(app, idx, ok, killed, launcher, &mut in_flight);
                    }
                    RowDone::MergeCheck { changed, verified } => {
                        finish_merge_check(app, idx, changed, verified, launcher, &line_tx, &mut in_flight);
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
            maybe = input_rx.recv() => {
                let Some(event) = maybe else { return Ok(()) };
                match event {
                    Event::Paste(text) => {
                        flash = None;
                        focused_input(app, selected, focus, &mut dispatch)
                            .map(|input| input.insert_str(&text));
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
                            }
                            KeyCode::Down => {
                                if !app.fleet.is_empty() {
                                    selected = (selected + 1).min(app.fleet.len() - 1);
                                }
                            }
                            KeyCode::Tab => {
                                focus = match focus {
                                    Focus::Dispatch if !app.fleet.is_empty() => Focus::Reply,
                                    _ => Focus::Dispatch,
                                };
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
                                    worktree::cleanup(&[row.worktree.clone()]);
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
                                focused_input(app, selected, focus, &mut dispatch)
                                    .map(|input| input.insert(c));
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
                    _ => {}
                }
            }
        }
    }
}

/// Remove every remaining fleet worktree (called at TUI shutdown).
pub(crate) fn cleanup_fleet(app: &mut App) {
    let paths: Vec<PathBuf> = app
        .fleet
        .iter()
        .filter(|r| r.state != RowState::Closed)
        .map(|r| r.worktree.clone())
        .collect();
    if !paths.is_empty() {
        worktree::cleanup(&paths);
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
    if !worktree::in_git_repo() {
        return Err(anyhow!(
            "not in a git repository (fleet rows need worktrees)"
        ));
    }
    // Snapshot the current tree (incl. uncommitted work) as the row's base.
    let base = hi_tools::checkpoint::create(std::path::Path::new("."))
        .await
        .context("couldn't snapshot the working tree")?;
    app.fleet_next_id += 1;
    let id = app.fleet_next_id;
    let path = worktree::worktree_path("fleet", id as u32);
    worktree::add_worktree(&path, &base)?;
    let session = (launcher.session_path)()?;
    let row = FleetRow {
        id,
        title: prompt.clone(),
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
    };
    app.fleet.push(row);
    let idx = app.fleet.len() - 1;
    start_turn(app, idx, prompt, launcher, line_tx, in_flight);
    Ok(idx)
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
    row.push_line(format!("› {prompt}"));

    let mut cmd = tokio::process::Command::new(&launcher.exe);
    cmd.current_dir(&row.worktree)
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
            "--max-steps",
            &launcher.max_steps.to_string(),
        ]);
    cmd.arg("--session-file").arg(&row.session);
    if let Some(v) = &launcher.verify {
        cmd.args([
            "--verify",
            v,
            "--max-verify",
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
    in_flight: &mut FuturesUnordered<RowFut>,
) {
    let Some(row) = app.fleet.get_mut(idx) else {
        return;
    };
    row.kill = None;
    row.turns += 1;
    if killed {
        row.state = RowState::Failed;
        row.started = None;
        row.activity.clear();
        row.push_line("⚠ turn killed".to_string());
        return;
    }
    if !ok {
        row.state = RowState::Failed;
        row.started = None;
        row.activity.clear();
        row.push_line("✗ agent run failed (see output above)".to_string());
        return;
    }
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
    } else if !verified {
        row.merge = MergeState::VerifyFailed;
        row.push_line("⇡ verify failed in the worktree — not merged (m forces)".to_string());
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
    } else {
        match worktree::apply_changes(&row.worktree, &row.base) {
            Ok(_) => {
                row.merge = MergeState::Merged(row.changed.len());
                row.push_line(format!(
                    "✓ merged {} file(s) into your tree: {}",
                    row.changed.len(),
                    row.changed.join(", ")
                ));
            }
            Err(err) => {
                row.merge = MergeState::VerifyFailed;
                row.push_line(format!("✗ merge failed: {err:#} (m retries)"));
            }
        }
    }
    if let Some(next) = row.pending.pop_front() {
        start_turn(app, idx, next, launcher, line_tx, in_flight);
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
    match worktree::apply_changes(&row.worktree, &row.base) {
        Ok(_) => {
            row.changed = changed;
            row.merge = MergeState::Merged(row.changed.len());
            row.push_line(format!(
                "✓ merged {} file(s) into your tree (forced)",
                row.changed.len()
            ));
        }
        Err(err) => {
            *flash = Some(format!("#{}: merge failed: {err:#}", row.id));
        }
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
) {
    let area = frame.area();
    let attach = focus == Focus::Attach;
    let table_height = if attach {
        0
    } else {
        app.fleet.len().clamp(1, TABLE_ROWS) as u16 + 2
    };
    let rows = Layout::vertical([
        Constraint::Length(1),            // header
        Constraint::Length(table_height), // fleet table (hidden in attach)
        Constraint::Min(3),               // peek / attach panel
        Constraint::Length(3),            // focused input
        Constraint::Length(1),            // footer hints
    ])
    .split(area);

    let title = format!(
        " hi fleet · {} agent(s) · {} working{} ",
        app.fleet.len(),
        working,
        if exit_armed {
            " — turns in flight! Esc again kills them (sessions stay resumable)"
        } else {
            ""
        },
    );
    let header_style = if exit_armed {
        Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default()
            .fg(Color::Magenta)
            .add_modifier(Modifier::BOLD)
    };
    frame.render_widget(Paragraph::new(Line::styled(title, header_style)), rows[0]);

    if !attach {
        render_table(frame, app, selected, rows[1]);
    }
    render_peek(frame, app, selected, rows[2], attach);
    render_input(frame, app, selected, focus, dispatch, rows[3]);

    let hint = match flash {
        Some(msg) => Line::styled(msg.to_string(), Style::default().fg(Color::Yellow)),
        None => Line::styled(
            match focus {
                Focus::Dispatch => {
                    "Enter dispatch · Ctrl+S dispatch+attach · ↑↓ select · Tab reply · m merge · x close · Ctrl+K kill · Esc back"
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
        MergeState::Merged(n) => (format!("✓{n}"), Style::default().fg(Color::Green)),
        MergeState::Held(_) => ("⇡held".to_string(), Style::default().fg(Color::Yellow)),
        MergeState::VerifyFailed => (
            "⇡unverified".to_string(),
            Style::default().fg(Color::Yellow),
        ),
    }
}

fn render_table(frame: &mut ratatui::Frame, app: &App, selected: usize, area: Rect) {
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(Color::Magenta))
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
    for (i, row) in app.fleet.iter().enumerate().skip(start).take(inner_rows) {
        let (glyph, glyph_style) = match row.state {
            RowState::Working => (
                SPINNER[app.spinner % SPINNER.len()].to_string(),
                Style::default().fg(Color::Cyan),
            ),
            RowState::Idle => ("·".to_string(), Style::default().fg(Color::Green)),
            RowState::Failed => ("✗".to_string(), Style::default().fg(Color::Red)),
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
        lines.push(Line::from(vec![
            Span::styled(format!(" {glyph} "), glyph_style),
            Span::styled(format!("#{:<2} {:>9}{} ", row.id, elapsed, queued), style),
            Span::styled(format!("{badge:>11} "), badge_style),
            Span::styled(truncate(lead, 60), style),
        ]));
    }
    frame.render_widget(Paragraph::new(lines).block(block), area);
}

fn render_peek(frame: &mut ratatui::Frame, app: &App, selected: usize, area: Rect, attach: bool) {
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
        .border_style(Style::default().fg(if attach { Color::Cyan } else { Color::DarkGray }))
        .title(title);
    let inner = area.height.saturating_sub(2) as usize;
    let mut lines: Vec<Line> = Vec::new();
    let shown = row.tail.len().saturating_sub(inner.saturating_sub(1));
    for line in row.tail.iter().skip(shown) {
        let style = if line.starts_with('⚙') || line.starts_with('›') {
            dim()
        } else if line.starts_with('✗') || line.starts_with('⚠') {
            Style::default().fg(Color::Red)
        } else if line.starts_with('✓') || line.starts_with('⇡') {
            Style::default().fg(Color::Green)
        } else {
            Style::default()
        };
        lines.push(Line::styled(line.clone(), style));
    }
    if row.state == RowState::Working {
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
            Style::default().fg(Color::Cyan),
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
            Color::Magenta,
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
                Color::Cyan,
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
        }
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
}
