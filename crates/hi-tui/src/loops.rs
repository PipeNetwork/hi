//! `/loop` — the same prompt, on a cadence.
//!
//! A loop fires a full agent turn every N seconds (60s–7d): a fleet-style
//! child `hi` run in the real working directory, resuming the loop's own
//! session file — so every firing *remembers* the previous checks and can
//! compare instead of re-describing. The wrapper prompt asks the child to
//! reply exactly `NOTHING NEW` when nothing meaningful changed; quiet firings
//! render as a dim one-liner while changes render loud (plus a terminal ping).
//!
//! Loops auto-expire 7 days after creation, are cancellable by id, persist to
//! a per-project `loops.json`, and re-arm when the TUI restarts (they only
//! fire while `hi` is running). The manager is one background task — it never
//! touches the `Agent`; results drain into the transcript on UI ticks.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::{mpsc, oneshot};

use crate::FleetLauncher;

/// Loops expire this long after creation.
pub(crate) const LOOP_TTL_SECS: u64 = 7 * 86_400;
/// A single firing is killed after this long (a watcher turn should be quick).
const FIRING_TIMEOUT_SECS: u64 = 600;
/// Max simultaneously-armed loops per project.
const MAX_LOOPS: usize = 8;
/// The marker a firing replies with when nothing changed since the last check.
pub(crate) const QUIET_MARKER: &str = "NOTHING NEW";

/// One recurring loop (persisted verbatim in `loops.json`).
#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct LoopSpec {
    pub(crate) id: u64,
    pub(crate) prompt: String,
    pub(crate) interval_secs: u64,
    /// Unix millis.
    pub(crate) created_ms: u64,
    pub(crate) expires_ms: u64,
    pub(crate) next_ms: u64,
    /// The loop's session file (each firing resumes it).
    pub(crate) session: PathBuf,
    #[serde(default)]
    pub(crate) firings: u64,
    /// Held: stops firing but stays resumable (manual, or hit its budget).
    #[serde(default)]
    pub(crate) paused: bool,
    /// Optional token spend cap; the loop auto-pauses once `spent_tokens`
    /// reaches it.
    #[serde(default)]
    pub(crate) token_budget: Option<u64>,
    /// Cumulative tokens spent across firings (session-cumulative, from the
    /// child's `--report`). Persisted so the cost survives a restart.
    #[serde(default)]
    pub(crate) spent_tokens: u64,
}

impl LoopSpec {
    /// Short display name: the first few words of the prompt.
    pub(crate) fn name(&self) -> String {
        let words: Vec<&str> = self.prompt.split_whitespace().take(4).collect();
        let mut name = words.join(" ");
        if self.prompt.split_whitespace().count() > 4 {
            name.push('…');
        }
        name
    }
}

#[derive(Default, Serialize, Deserialize)]
struct LoopsFile {
    loops: Vec<LoopSpec>,
    #[serde(default)]
    next_id: u64,
}

/// How many recent firings the manager retains per loop for the `/watch` peek.
const HISTORY_CAP: usize = 30;

/// One recorded firing result (for the `/watch` history panel).
#[derive(Clone)]
pub(crate) struct HistItem {
    pub(crate) at_ms: u64,
    pub(crate) quiet: bool,
    pub(crate) summary: String,
}

/// Live per-loop state the manager publishes for the `/watch` dashboard. Built
/// from the persisted `LoopSpec` plus the manager's in-memory runtime (whether a
/// firing is in flight, and the recent result history) — never persisted.
#[derive(Clone)]
pub(crate) struct LoopWatchRow {
    pub(crate) id: u64,
    pub(crate) name: String,
    pub(crate) prompt: String,
    pub(crate) interval_secs: u64,
    pub(crate) created_ms: u64,
    pub(crate) next_ms: u64,
    pub(crate) expires_ms: u64,
    pub(crate) firings: u64,
    /// A firing is currently in flight.
    pub(crate) firing: bool,
    pub(crate) paused: bool,
    pub(crate) token_budget: Option<u64>,
    pub(crate) spent_tokens: u64,
    pub(crate) last_summary: Option<String>,
    pub(crate) last_quiet: bool,
    pub(crate) last_fired_ms: u64,
    /// Recent firings, oldest first.
    pub(crate) history: Vec<HistItem>,
}

/// The manager's in-memory runtime for one loop (not persisted).
#[derive(Default)]
struct LoopRuntime {
    firing: bool,
    last_summary: Option<String>,
    last_quiet: bool,
    last_fired_ms: u64,
    history: VecDeque<HistItem>,
}

/// A line for the transcript: (text, loud). Loud lines also ping when the
/// terminal is unfocused.
pub(crate) type LoopLine = (String, bool);

/// Control messages from the UI to the manager task.
pub(crate) enum LoopCtl {
    Create {
        secs: u64,
        prompt: String,
        reply: oneshot::Sender<Result<LoopSpec, String>>,
    },
    Cancel {
        id: u64,
        reply: oneshot::Sender<bool>,
    },
    /// Fire a loop immediately (its cadence continues unchanged after).
    FireNow {
        id: u64,
        reply: oneshot::Sender<bool>,
    },
    /// Pause (`on: true`) or resume (`on: false`) a loop.
    Pause {
        id: u64,
        on: bool,
        reply: oneshot::Sender<bool>,
    },
    /// Set (`Some`) or clear (`None`) a loop's token budget.
    Budget {
        id: u64,
        tokens: Option<u64>,
        reply: oneshot::Sender<bool>,
    },
    List {
        reply: oneshot::Sender<Vec<LoopSpec>>,
    },
}

/// The UI's handle to the loop manager.
pub(crate) struct LoopsHandle {
    pub(crate) ctl: mpsc::UnboundedSender<LoopCtl>,
    /// Firing results awaiting display; the UI drains this on ticks.
    pub(crate) pending: Arc<Mutex<Vec<LoopLine>>>,
    /// Live per-loop state for the `/watch` dashboard; the manager keeps it
    /// current on every state change.
    pub(crate) snapshot: Arc<Mutex<Vec<LoopWatchRow>>>,
}

impl LoopsHandle {
    /// Take any queued transcript lines (called from UI tick arms).
    pub(crate) fn drain(&self) -> Vec<LoopLine> {
        std::mem::take(&mut *self.pending.lock().unwrap())
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// The standing instructions wrapped around the user's prompt on every firing.
/// The first firing establishes (and reports) the baseline; later firings
/// compare against the session's previous checks and stay quiet when nothing
/// changed.
fn wrapper_prompt(spec: &LoopSpec) -> String {
    let contract = if spec.firings <= 1 {
        "This is the FIRST check of this watch — establish the baseline and report it briefly \
         (never reply NOTHING NEW on the first check)."
            .to_string()
    } else {
        format!(
            "This conversation contains your previous checks — compare against them rather than \
             re-describing everything. If nothing meaningful changed since the last check, reply \
             with exactly: {QUIET_MARKER}. Otherwise summarize what changed and why it matters, \
             briefly."
        )
    };
    format!(
        "Recurring watch (loop \"{}\", every {}): {}\n\n{contract}",
        spec.name(),
        humanize_secs(spec.interval_secs),
        spec.prompt,
    )
}

/// "90s", "30m", "2h", "1d" style rendering.
pub(crate) fn humanize_secs(secs: u64) -> String {
    if secs.is_multiple_of(86_400) {
        format!("{}d", secs / 86_400)
    } else if secs.is_multiple_of(3600) {
        format!("{}h", secs / 3600)
    } else if secs.is_multiple_of(60) {
        format!("{}m", secs / 60)
    } else {
        format!("{secs}s")
    }
}

/// Compact token count for display: `0`, `999`, `1.2k`, `12k`, `1.5m`.
pub(crate) fn fmt_tokens(n: u64) -> String {
    if n < 1_000 {
        n.to_string()
    } else if n < 1_000_000 {
        let k = n as f64 / 1_000.0;
        if k < 10.0 {
            format!("{k:.1}k")
        } else {
            format!("{}k", k.round() as u64)
        }
    } else {
        format!("{:.1}m", n as f64 / 1_000_000.0)
    }
}

/// Whether a firing's final reply is the quiet marker (nothing to report).
pub(crate) fn is_quiet(summary: &str) -> bool {
    let s = summary.trim().trim_end_matches('.').trim();
    s.eq_ignore_ascii_case(QUIET_MARKER)
        || s.ends_with(QUIET_MARKER)
        || s.ends_with(&QUIET_MARKER.to_lowercase())
}

/// Spawn the loop manager: loads persisted loops (dropping expired ones),
/// re-arms the rest, and runs the timer wheel until the TUI exits.
pub(crate) fn start(launcher: Arc<FleetLauncher>, loops_file: Option<PathBuf>) -> LoopsHandle {
    let (ctl_tx, ctl_rx) = mpsc::unbounded_channel();
    let pending: Arc<Mutex<Vec<LoopLine>>> = Arc::new(Mutex::new(Vec::new()));
    let snapshot: Arc<Mutex<Vec<LoopWatchRow>>> = Arc::new(Mutex::new(Vec::new()));
    let handle = LoopsHandle {
        ctl: ctl_tx,
        pending: pending.clone(),
        snapshot: snapshot.clone(),
    };
    tokio::spawn(manager(launcher, loops_file, ctl_rx, pending, snapshot));
    handle
}

/// Rebuild the published `/watch` snapshot from persisted specs + live runtime,
/// pruning runtime for loops that no longer exist.
fn publish(
    state: &LoopsFile,
    runtime: &mut HashMap<u64, LoopRuntime>,
    snapshot: &Arc<Mutex<Vec<LoopWatchRow>>>,
) {
    let live: HashSet<u64> = state.loops.iter().map(|l| l.id).collect();
    runtime.retain(|id, _| live.contains(id));
    let rows = state
        .loops
        .iter()
        .map(|l| {
            let rt = runtime.get(&l.id);
            LoopWatchRow {
                id: l.id,
                name: l.name(),
                prompt: l.prompt.clone(),
                interval_secs: l.interval_secs,
                created_ms: l.created_ms,
                next_ms: l.next_ms,
                expires_ms: l.expires_ms,
                firings: l.firings,
                firing: rt.is_some_and(|r| r.firing),
                paused: l.paused,
                token_budget: l.token_budget,
                spent_tokens: l.spent_tokens,
                last_summary: rt.and_then(|r| r.last_summary.clone()),
                last_quiet: rt.is_some_and(|r| r.last_quiet),
                last_fired_ms: rt.map_or(0, |r| r.last_fired_ms),
                history: rt
                    .map(|r| r.history.iter().cloned().collect())
                    .unwrap_or_default(),
            }
        })
        .collect();
    *snapshot.lock().unwrap() = rows;
}

async fn manager(
    launcher: Arc<FleetLauncher>,
    loops_file: Option<PathBuf>,
    mut ctl: mpsc::UnboundedReceiver<LoopCtl>,
    pending: Arc<Mutex<Vec<LoopLine>>>,
    snapshot: Arc<Mutex<Vec<LoopWatchRow>>>,
) {
    let mut runtime: HashMap<u64, LoopRuntime> = HashMap::new();
    let mut state = load(loops_file.as_deref());
    let now = now_ms();
    let before = state.loops.len();
    state.loops.retain(|l| l.expires_ms > now);
    if before > state.loops.len() {
        pending.lock().unwrap().push((
            format!(
                "{} loop(s) expired while hi was closed",
                before - state.loops.len()
            ),
            false,
        ));
    }
    for spec in &mut state.loops {
        // Missed firings while hi was closed: schedule the next one soon.
        if spec.next_ms < now {
            spec.next_ms = now + 5_000;
        }
        pending.lock().unwrap().push((
            format!(
                "⟳ loop#{} re-armed ({} · every {})",
                spec.id,
                spec.name(),
                humanize_secs(spec.interval_secs)
            ),
            false,
        ));
    }
    save(loops_file.as_deref(), &state);
    publish(&state, &mut runtime, &snapshot);

    // Firings in flight: (loop id, outcome) results come back over a channel.
    let (done_tx, mut done_rx) = mpsc::unbounded_channel::<(u64, Result<FiringOutcome, String>)>();
    let mut in_flight: usize = 0;

    loop {
        let now = now_ms();
        // Expire + fire due loops (respecting a small concurrency cap).
        let mut fired = false;
        state.loops.retain(|l| {
            if l.expires_ms <= now {
                pending.lock().unwrap().push((
                    format!("⟳ loop#{} ({}) expired after 7 days", l.id, l.name()),
                    true,
                ));
                fired = true;
                false
            } else {
                true
            }
        });
        for spec in &mut state.loops {
            if !spec.paused && spec.next_ms <= now && in_flight < 2 {
                spec.next_ms = now + spec.interval_secs * 1000;
                spec.firings += 1;
                in_flight += 1;
                fired = true;
                runtime.entry(spec.id).or_default().firing = true;
                let launcher = launcher.clone();
                let spec_snapshot = spec.clone();
                let done = done_tx.clone();
                tokio::spawn(async move {
                    let result = run_firing(&launcher, &spec_snapshot).await;
                    let _ = done.send((spec_snapshot.id, result));
                });
            }
        }
        if fired {
            save(loops_file.as_deref(), &state);
        }
        publish(&state, &mut runtime, &snapshot);

        // Sleep until the next due time (capped so ctl/expiry stay responsive).
        // A paused loop never fires, so only its expiry can wake us — otherwise
        // its stale `next_ms` would pin the sleep to the 250ms floor and spin.
        let next_due = state
            .loops
            .iter()
            .map(|l| {
                if l.paused {
                    l.expires_ms
                } else {
                    l.next_ms.min(l.expires_ms)
                }
            })
            .min()
            .unwrap_or(now + 60_000);
        let sleep_ms = next_due.saturating_sub(now).clamp(250, 30_000);

        tokio::select! {
            maybe = ctl.recv() => {
                let Some(msg) = maybe else { return }; // UI gone — stop
                match msg {
                    LoopCtl::Create { secs, prompt, reply } => {
                        let result = if state.loops.len() >= MAX_LOOPS {
                            Err(format!("loop limit reached ({MAX_LOOPS}) — cancel one first"))
                        } else {
                            match (launcher.loop_session_path)() {
                                Ok(session) => {
                                    state.next_id += 1;
                                    let now = now_ms();
                                    let spec = LoopSpec {
                                        id: state.next_id,
                                        prompt,
                                        interval_secs: secs,
                                        created_ms: now,
                                        expires_ms: now + LOOP_TTL_SECS * 1000,
                                        // First firing right away.
                                        next_ms: now,
                                        session,
                                        firings: 0,
                                        paused: false,
                                        token_budget: None,
                                        spent_tokens: 0,
                                    };
                                    state.loops.push(spec.clone());
                                    save(loops_file.as_deref(), &state);
                                    Ok(spec)
                                }
                                Err(err) => Err(format!("couldn't allocate a session: {err:#}")),
                            }
                        };
                        let _ = reply.send(result);
                    }
                    LoopCtl::Cancel { id, reply } => {
                        let before = state.loops.len();
                        state.loops.retain(|l| l.id != id);
                        let removed = state.loops.len() < before;
                        if removed {
                            save(loops_file.as_deref(), &state);
                        }
                        let _ = reply.send(removed);
                    }
                    LoopCtl::FireNow { id, reply } => {
                        // Due it now; the top of the loop fires it this cycle
                        // (subject to the concurrency cap) and the cadence resumes.
                        let mut ok = false;
                        let now = now_ms();
                        for l in &mut state.loops {
                            if l.id == id {
                                l.next_ms = now;
                                ok = true;
                            }
                        }
                        if ok {
                            save(loops_file.as_deref(), &state);
                        }
                        let _ = reply.send(ok);
                    }
                    LoopCtl::Pause { id, on, reply } => {
                        let mut ok = false;
                        let now = now_ms();
                        for l in &mut state.loops {
                            if l.id == id {
                                l.paused = on;
                                // Resuming an overdue loop: fire it soon rather
                                // than immediately hammering (and never in the past).
                                if !on && l.next_ms < now {
                                    l.next_ms = now + 2_000;
                                }
                                ok = true;
                            }
                        }
                        if ok {
                            save(loops_file.as_deref(), &state);
                            publish(&state, &mut runtime, &snapshot);
                        }
                        let _ = reply.send(ok);
                    }
                    LoopCtl::Budget { id, tokens, reply } => {
                        let mut ok = false;
                        for l in &mut state.loops {
                            if l.id == id {
                                l.token_budget = tokens;
                                // Setting/raising a budget above current spend
                                // lifts an earlier budget auto-pause.
                                if l.paused
                                    && tokens.is_some_and(|b| l.spent_tokens < b)
                                {
                                    l.paused = false;
                                    if l.next_ms < now_ms() {
                                        l.next_ms = now_ms() + 2_000;
                                    }
                                }
                                ok = true;
                            }
                        }
                        if ok {
                            save(loops_file.as_deref(), &state);
                            publish(&state, &mut runtime, &snapshot);
                        }
                        let _ = reply.send(ok);
                    }
                    LoopCtl::List { reply } => {
                        let _ = reply.send(state.loops.clone());
                    }
                }
            }
            Some((id, result)) = done_rx.recv() => {
                in_flight = in_flight.saturating_sub(1);
                let name = state
                    .loops
                    .iter()
                    .find(|l| l.id == id)
                    .map(LoopSpec::name)
                    .unwrap_or_else(|| format!("#{id}"));
                let fired_ms = now_ms();
                let (line, summary, quiet, tokens) = match result {
                    Ok(outcome) => {
                        let quiet = is_quiet(&outcome.summary);
                        let text = if quiet {
                            format!("⟳ loop#{id} ({name}): nothing new")
                        } else {
                            format!("⟳ loop#{id} ({name}): {}", truncate(&outcome.summary, 160))
                        };
                        ((text, !quiet), outcome.summary, quiet, Some(outcome.total_tokens))
                    }
                    Err(err) => {
                        let text = format!("⟳ loop#{id} ({name}) firing failed: {err}");
                        ((text, true), format!("firing failed: {err}"), false, None)
                    }
                };
                // Fold in the cost and enforce the budget: `total_tokens` is
                // session-cumulative, so it *is* the loop's running spend.
                let mut budget_line: Option<LoopLine> = None;
                if let Some(spent) = tokens {
                    if let Some(l) = state.loops.iter_mut().find(|l| l.id == id) {
                        l.spent_tokens = spent;
                        if let Some(budget) = l.token_budget
                            && !l.paused
                            && spent >= budget
                        {
                            l.paused = true;
                            budget_line = Some((
                                format!(
                                    "⏸ loop#{id} ({name}) paused — hit token budget ({} / {})",
                                    fmt_tokens(spent),
                                    fmt_tokens(budget),
                                ),
                                true,
                            ));
                        }
                    }
                    save(loops_file.as_deref(), &state);
                }
                // Record the runtime result for the /watch dashboard + history.
                let rt = runtime.entry(id).or_default();
                rt.firing = false;
                rt.last_summary = Some(summary.clone());
                rt.last_quiet = quiet;
                rt.last_fired_ms = fired_ms;
                rt.history.push_back(HistItem { at_ms: fired_ms, quiet, summary });
                while rt.history.len() > HISTORY_CAP {
                    rt.history.pop_front();
                }
                pending.lock().unwrap().push(line);
                if let Some(bl) = budget_line {
                    pending.lock().unwrap().push(bl);
                }
                publish(&state, &mut runtime, &snapshot);
            }
            _ = tokio::time::sleep(Duration::from_millis(sleep_ms)) => {}
        }
    }
}

/// The result of one firing: the child's reply summary plus its
/// session-cumulative token spend (from the `--report`).
struct FiringOutcome {
    summary: String,
    total_tokens: u64,
}

/// One firing: a fleet-style child run in the real cwd, resuming the loop's
/// session. Returns the child's final reply line as the summary, plus the
/// session-cumulative token total read back from its `--report`.
async fn run_firing(launcher: &FleetLauncher, spec: &LoopSpec) -> Result<FiringOutcome, String> {
    // One report file per loop, alongside its session, overwritten each firing.
    let report_path = spec.session.with_extension("report.json");
    let mut cmd = tokio::process::Command::new(&launcher.exe);
    cmd.env("HI_API_KEY", &launcher.api_key)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .args([
            "--provider",
            &launcher.provider,
            "--model",
            &launcher.model,
            "--base-url",
            &launcher.base_url,
            "--max-steps",
            "30",
        ]);
    cmd.arg("--session-file").arg(&spec.session);
    cmd.arg("--report").arg(&report_path);
    cmd.arg(wrapper_prompt(spec));

    let mut child = cmd.spawn().map_err(|e| format!("couldn't launch: {e}"))?;
    let mut tail: Vec<String> = Vec::new();
    if let Some(stdout) = child.stdout.take() {
        let mut lines = BufReader::new(stdout).lines();
        let read = async {
            while let Ok(Some(line)) = lines.next_line().await {
                let line = crate::dashboard::strip_ansi_line(&line);
                let line = line.trim_end();
                if !line.trim().is_empty() {
                    tail.push(line.to_string());
                    if tail.len() > 50 {
                        tail.remove(0);
                    }
                }
            }
        };
        // Read output with the firing timeout; on timeout, kill the child.
        if tokio::time::timeout(Duration::from_secs(FIRING_TIMEOUT_SECS), read)
            .await
            .is_err()
        {
            let _ = child.start_kill();
            let _ = child.wait().await;
            return Err(format!("timed out after {FIRING_TIMEOUT_SECS}s"));
        }
    }
    let status = child
        .wait()
        .await
        .map_err(|e| format!("wait failed: {e}"))?;
    if !status.success() {
        return Err(format!(
            "agent run failed ({}): {}",
            status,
            tail.last().cloned().unwrap_or_default()
        ));
    }
    // The final non-decoration line is the reply's tail — the summary.
    let summary = tail
        .iter()
        .rev()
        .find(|l| !l.trim_start().starts_with(['⏺', '✓', '⚙', '›', '↳']))
        .cloned()
        .unwrap_or_else(|| tail.last().cloned().unwrap_or_default());
    let total_tokens = read_report_tokens(&report_path);
    Ok(FiringOutcome {
        summary,
        total_tokens,
    })
}

/// Read the session-cumulative `total_tokens` from a firing's `--report` JSON
/// (0 if the file is missing or malformed — cost tracking is best-effort).
fn read_report_tokens(path: &std::path::Path) -> u64 {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .and_then(|v| v.get("total_tokens").and_then(serde_json::Value::as_u64))
        .unwrap_or(0)
}

fn load(path: Option<&std::path::Path>) -> LoopsFile {
    path.and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default()
}

fn save(path: Option<&std::path::Path>, state: &LoopsFile) {
    let Some(path) = path else { return };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(json) = serde_json::to_string_pretty(state) {
        let _ = std::fs::write(path, json);
    }
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

    fn spec() -> LoopSpec {
        LoopSpec {
            id: 1,
            prompt: "check whether the CI pipeline on main is green".into(),
            interval_secs: 1800,
            created_ms: 0,
            expires_ms: LOOP_TTL_SECS * 1000,
            next_ms: 0,
            session: PathBuf::from("/tmp/loop.jsonl"),
            firings: 0,
            paused: false,
            token_budget: None,
            spent_tokens: 0,
        }
    }

    #[test]
    fn quiet_marker_detection() {
        assert!(is_quiet("NOTHING NEW"));
        assert!(is_quiet("  nothing new.  "));
        assert!(is_quiet("Checked the logs again — NOTHING NEW"));
        assert!(!is_quiet("CI is now red: 3 failures in parser tests"));
        assert!(!is_quiet(""));
    }

    #[test]
    fn wrapper_prompt_carries_contract() {
        // First firing: baseline instructions, never the quiet marker.
        let mut s = spec();
        s.firings = 1;
        let first = wrapper_prompt(&s);
        assert!(first.contains("FIRST check"), "{first}");
        assert!(first.contains("check whether the CI pipeline"));
        // Later firings: compare + quiet contract.
        s.firings = 2;
        let later = wrapper_prompt(&s);
        assert!(later.contains("every 30m"), "{later}");
        assert!(later.contains(QUIET_MARKER));
        assert!(!later.contains("FIRST check"));
    }

    #[test]
    fn fmt_tokens_units() {
        assert_eq!(fmt_tokens(0), "0");
        assert_eq!(fmt_tokens(999), "999");
        assert_eq!(fmt_tokens(1_200), "1.2k");
        assert_eq!(fmt_tokens(12_000), "12k");
        assert_eq!(fmt_tokens(1_500_000), "1.5m");
    }

    #[test]
    fn humanize_units() {
        assert_eq!(humanize_secs(90), "90s");
        assert_eq!(humanize_secs(1800), "30m");
        assert_eq!(humanize_secs(7200), "2h");
        assert_eq!(humanize_secs(86_400), "1d");
    }

    #[test]
    fn loops_file_round_trips() {
        let state = LoopsFile {
            loops: vec![spec()],
            next_id: 1,
        };
        let dir = std::env::temp_dir().join(format!("hi-loops-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("loops.json");
        save(Some(&path), &state);
        let loaded = load(Some(&path));
        assert_eq!(loaded.loops.len(), 1);
        assert_eq!(loaded.loops[0].prompt, spec().prompt);
        assert_eq!(loaded.next_id, 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn name_truncates_to_first_words() {
        assert_eq!(spec().name(), "check whether the CI…");
    }

    #[test]
    fn publish_builds_and_prunes_snapshot() {
        let mut state = LoopsFile {
            loops: vec![spec()],
            next_id: 2,
        };
        let mut s2 = spec();
        s2.id = 2;
        s2.prompt = "watch prod p99 latency".into();
        state.loops.push(s2);

        let mut runtime: HashMap<u64, LoopRuntime> = HashMap::new();
        let rt1 = runtime.entry(1).or_default();
        rt1.last_summary = Some("CI went red".into());
        rt1.last_quiet = false;
        rt1.last_fired_ms = 123;
        rt1.history.push_back(HistItem {
            at_ms: 123,
            quiet: false,
            summary: "CI went red".into(),
        });
        // An orphaned runtime entry (loop no longer exists) must be pruned.
        runtime.entry(99).or_default().firing = true;

        let snap = Arc::new(Mutex::new(Vec::new()));
        publish(&state, &mut runtime, &snap);

        assert!(!runtime.contains_key(&99), "orphan runtime pruned");
        let rows = snap.lock().unwrap();
        assert_eq!(rows.len(), 2);
        let r1 = rows.iter().find(|r| r.id == 1).unwrap();
        assert_eq!(r1.last_summary.as_deref(), Some("CI went red"));
        assert!(!r1.last_quiet);
        assert_eq!(r1.history.len(), 1);
        let r2 = rows.iter().find(|r| r.id == 2).unwrap();
        assert!(r2.last_summary.is_none(), "unfired loop has no summary");
        assert_eq!(r2.firings, 0);
        assert!(r2.history.is_empty());
    }

    /// Poll the published snapshot until `pred` holds (or time out).
    async fn wait_until(handle: &LoopsHandle, pred: impl Fn(&[LoopWatchRow]) -> bool) {
        for _ in 0..200 {
            if pred(&handle.snapshot.lock().unwrap()) {
                return;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        let n = handle.snapshot.lock().unwrap().len();
        panic!("condition not met within 5s; snapshot has {n} row(s)");
    }

    /// Drive the real manager end-to-end with `/bin/echo` standing in for `hi`:
    /// each firing is a genuine subprocess that exits 0 fast. Validates the whole
    /// spine `/watch` reads — start → fire → done → runtime → snapshot — plus the
    /// `FireNow` and `Cancel` controls its keys send.
    #[tokio::test]
    async fn manager_fires_records_refires_and_cancels() {
        let dir = std::env::temp_dir().join(format!("hi-watch-mgr-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let sess = dir.join("loop.jsonl");
        let launcher = FleetLauncher {
            exe: PathBuf::from("/bin/echo"),
            provider: "p".into(),
            model: "m".into(),
            base_url: "u".into(),
            api_key: "k".into(),
            verify: None,
            max_verify: 0,
            max_steps: 30,
            session_path: Box::new(|| Ok(PathBuf::from("/tmp/unused.jsonl"))),
            sessions: Box::new(Vec::new),
            resume_info: Box::new(|_| None),
            loop_session_path: Box::new(move || Ok(sess.clone())),
            loops_file: None,
        };
        let handle = start(Arc::new(launcher), None);

        // Arm a loop — it fires immediately (next_ms = now).
        let (tx, rx) = oneshot::channel();
        handle
            .ctl
            .send(LoopCtl::Create {
                secs: 3600,
                prompt: "watch the thing".into(),
                reply: tx,
            })
            .unwrap();
        let spec = rx.await.unwrap().unwrap();
        let id = spec.id;

        // First firing completes and is recorded in the snapshot.
        wait_until(&handle, |rows| {
            rows.iter()
                .find(|r| r.id == id)
                .is_some_and(|r| !r.firing && !r.history.is_empty())
        })
        .await;
        {
            let rows = handle.snapshot.lock().unwrap();
            let r = rows.iter().find(|r| r.id == id).unwrap();
            assert!(r.firings >= 1, "fired at least once");
            assert!(r.last_summary.is_some(), "recorded a summary");
            assert_eq!(r.history.len(), 1);
            assert_eq!(r.last_fired_ms, r.history[0].at_ms);
        }

        // FireNow → a second recorded firing without waiting out the cadence.
        let (tx, rx) = oneshot::channel();
        handle.ctl.send(LoopCtl::FireNow { id, reply: tx }).unwrap();
        assert!(rx.await.unwrap(), "FireNow accepted");
        wait_until(&handle, |rows| {
            rows.iter()
                .find(|r| r.id == id)
                .is_some_and(|r| r.history.len() >= 2)
        })
        .await;

        // Cancel → the loop leaves the snapshot.
        let (tx, rx) = oneshot::channel();
        handle.ctl.send(LoopCtl::Cancel { id, reply: tx }).unwrap();
        assert!(rx.await.unwrap(), "cancel removed the loop");
        wait_until(&handle, |rows| rows.is_empty()).await;

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A stub `hi` that writes a `--report` with a fixed token total, so firings
    /// exercise the cost-tracking + budget path. Returns the script path.
    fn report_stub(dir: &std::path::Path, tokens: u64) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let path = dir.join("stub.sh");
        let script = format!(
            "#!/bin/sh\nprev=\nfor a in \"$@\"; do\n  [ \"$prev\" = \"--report\" ] && \
             printf '{{\"total_tokens\": {tokens}}}' > \"$a\"\n  prev=\"$a\"\ndone\n\
             printf 'stub check reply\\n'\n"
        );
        std::fs::write(&path, script).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    /// Drive the manager with a report-writing stub to validate cost tracking,
    /// budget auto-pause, and manual pause blocking a due firing.
    #[tokio::test]
    async fn manager_pause_resume_and_budget_autopause() {
        let dir = std::env::temp_dir().join(format!("hi-watch-cost-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let sess = dir.join("loop.jsonl");
        let exe = report_stub(&dir, 1_000_000);
        let launcher = FleetLauncher {
            exe,
            provider: "p".into(),
            model: "m".into(),
            base_url: "u".into(),
            api_key: "k".into(),
            verify: None,
            max_verify: 0,
            max_steps: 30,
            session_path: Box::new(|| Ok(PathBuf::from("/tmp/unused.jsonl"))),
            sessions: Box::new(Vec::new),
            resume_info: Box::new(|_| None),
            loop_session_path: Box::new(move || Ok(sess.clone())),
            loops_file: None,
        };
        let handle = start(Arc::new(launcher), None);

        let (tx, rx) = oneshot::channel();
        handle
            .ctl
            .send(LoopCtl::Create {
                secs: 60,
                prompt: "watch prod".into(),
                reply: tx,
            })
            .unwrap();
        let id = rx.await.unwrap().unwrap().id;

        // First firing records the report's cumulative token spend.
        wait_until(&handle, |rows| {
            rows.iter()
                .find(|r| r.id == id)
                .is_some_and(|r| r.spent_tokens == 1_000_000)
        })
        .await;

        // Set a budget below current spend, then fire: the firing auto-pauses.
        let (tx, rx) = oneshot::channel();
        handle
            .ctl
            .send(LoopCtl::Budget {
                id,
                tokens: Some(500_000),
                reply: tx,
            })
            .unwrap();
        assert!(rx.await.unwrap());
        let (tx, rx) = oneshot::channel();
        handle.ctl.send(LoopCtl::FireNow { id, reply: tx }).unwrap();
        assert!(rx.await.unwrap());
        wait_until(&handle, |rows| {
            rows.iter().find(|r| r.id == id).is_some_and(|r| r.paused)
        })
        .await;

        // A paused loop does not fire even when forced due (FireNow).
        let firings_now = handle
            .snapshot
            .lock()
            .unwrap()
            .iter()
            .find(|r| r.id == id)
            .unwrap()
            .firings;
        let (tx, rx) = oneshot::channel();
        handle.ctl.send(LoopCtl::FireNow { id, reply: tx }).unwrap();
        assert!(rx.await.unwrap());
        // Give the manager time to (not) fire.
        tokio::time::sleep(Duration::from_millis(400)).await;
        {
            let rows = handle.snapshot.lock().unwrap();
            let r = rows.iter().find(|r| r.id == id).unwrap();
            assert!(r.paused, "still paused");
            assert_eq!(r.firings, firings_now, "paused loop did not fire");
        }

        // Resume → clears the pause; raising the budget above spend keeps it live.
        let (tx, rx) = oneshot::channel();
        handle
            .ctl
            .send(LoopCtl::Budget {
                id,
                tokens: Some(5_000_000),
                reply: tx,
            })
            .unwrap();
        assert!(rx.await.unwrap());
        wait_until(&handle, |rows| {
            rows.iter().find(|r| r.id == id).is_some_and(|r| !r.paused)
        })
        .await;

        let _ = std::fs::remove_dir_all(&dir);
    }
}
