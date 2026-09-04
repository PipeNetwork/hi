//! `/loop` — the same prompt, on a cadence.
//!
//! A loop fires a full agent turn every N seconds (60s–7d): a fleet-style
//! child `hi` run in the real working directory, resuming the loop's own
//! session file — so every firing *remembers* the previous checks and can
//! compare instead of re-describing. The wrapper prompt asks the child to
//! reply exactly `NOTHING NEW` when nothing meaningful changed; quiet firings
//! render as a dim one-liner while changes render loud (plus a terminal ping).
//!
//! Loops run until explicitly cancelled, persist to a per-project `loops.json`,
//! and re-arm when the TUI restarts (they only fire while `hi` is running).
//! Legacy automatic expiries are removed on load. The manager is one background
//! task — it never touches the `Agent`; results drain into the transcript on UI
//! ticks.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

use crate::FleetLauncher;

/// Leave a child enough time to verify and persist after an explicitly
/// configured soft turn deadline. Ordinary loop work has no wall-clock cap.
const CHILD_SETTLEMENT_GRACE_SECS: u64 = 60;
/// Once the direct child exits, a descendant that inherited stdout must not
/// strand the loop manager forever. This only bounds pipe cleanup, not work.
const CHILD_PIPE_DRAIN_GRACE: Duration = Duration::from_secs(5);
/// Bound retained one-shot child output while continually draining it.
const FIRING_OUTPUT_CAP: usize = 256 * 1024;
/// Once a trigger process group is stopped, allow already-buffered diagnostic
/// output a brief drain before dropping its pipes.
const TRIGGER_PIPE_DRAIN_GRACE: Duration = Duration::from_secs(2);
/// Bound retained trigger evidence while continuously draining both pipes.
const TRIGGER_EVIDENCE_CAP: usize = 16 * 1024;
/// Max simultaneously-armed loops per project.
const MAX_LOOPS: usize = 8;
/// Outside its fire window, a loop re-checks at least this often — rather than
/// deferring a full interval, which would strand a long-interval loop outside
/// its window forever (it would keep its out-of-window time-of-day phase).
const WINDOW_RECHECK_SECS: u64 = 900;
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
    /// Optional Unix-millis expiry. New loops and migrated legacy loops are
    /// unlimited. Kept optional for forwards-compatible persisted state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) expires_ms: Option<u64>,
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
    /// Optional shell command run (via `sh -c`) after a firing reports a real
    /// change — a watcher that also *responds*. Off unless explicitly set.
    #[serde(default)]
    pub(crate) trigger: Option<String>,
    /// When set, a loud change dispatches a worktree-isolated agent to *fix* it,
    /// verify-gated (only merged if the verify passes). Off by default.
    #[serde(default)]
    pub(crate) autofix: bool,
    /// With `autofix`, land the verified fix as a pushed branch + PR (review)
    /// instead of merging it into the working tree. Off by default.
    #[serde(default)]
    pub(crate) fix_pr: bool,
    /// Optional local-time fire window; the loop only fires inside it.
    #[serde(default)]
    pub(crate) schedule: Option<Schedule>,
}

/// A local-time window a loop is allowed to fire within (e.g. 9–17 weekdays).
#[derive(Clone, Copy, Serialize, Deserialize)]
pub(crate) struct Schedule {
    pub(crate) start_hour: u8,
    pub(crate) end_hour: u8,
    pub(crate) weekdays_only: bool,
}

impl Schedule {
    /// Whether `hour` (0–23) / `weekday` (1=Mon..7=Sun) is inside the window.
    fn active(&self, hour: u8, weekday: u8) -> bool {
        let in_hours = if self.start_hour < self.end_hour {
            hour >= self.start_hour && hour < self.end_hour
        } else {
            // A window that wraps past midnight (e.g. 22–6).
            hour >= self.start_hour || hour < self.end_hour
        };
        let in_days = !self.weekdays_only || (1..=5).contains(&weekday);
        in_hours && in_days
    }

    fn is_active_now(&self) -> bool {
        let (hour, weekday) = local_hour_weekday();
        self.active(hour, weekday)
    }

    pub(crate) fn label(&self) -> String {
        format!(
            "{:02}-{:02}{}",
            self.start_hour,
            self.end_hour,
            if self.weekdays_only { " weekdays" } else { "" }
        )
    }
}

/// Local hour (0–23) and ISO weekday (1=Mon..7=Sun) via `date` — respects the
/// system timezone with no time-crate dependency. Falls back to a midday
/// weekday (i.e. "fire") if `date` is unavailable, so a broken clock never
/// silently stops a loop.
fn local_hour_weekday() -> (u8, u8) {
    std::process::Command::new("date")
        .arg("+%H %u")
        .output()
        .ok()
        .and_then(|o| {
            let s = String::from_utf8_lossy(&o.stdout);
            let mut it = s.split_whitespace();
            Some((it.next()?.parse().ok()?, it.next()?.parse().ok()?))
        })
        .unwrap_or((12, 3))
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
    pub(crate) expires_ms: Option<u64>,
    pub(crate) firings: u64,
    /// A firing is currently in flight.
    pub(crate) firing: bool,
    pub(crate) paused: bool,
    pub(crate) token_budget: Option<u64>,
    pub(crate) spent_tokens: u64,
    /// The configured on-change command, if any.
    pub(crate) trigger: Option<String>,
    /// The outcome of the most recent trigger run (for the peek).
    pub(crate) last_trigger: Option<String>,
    /// Auto-fix is enabled for this loop.
    pub(crate) autofix: bool,
    /// Auto-fix lands as a PR (review) rather than a working-tree merge.
    pub(crate) fix_pr: bool,
    /// The fire-window label, if scheduled (e.g. "09-17 weekdays").
    pub(crate) window: Option<String>,
    /// A fix attempt is currently in flight.
    pub(crate) fixing: bool,
    /// The outcome of the most recent fix attempt (for the peek).
    pub(crate) last_fix: Option<String>,
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
    /// Outcome of the most recent on-change trigger run.
    last_trigger: Option<String>,
    /// A fix attempt is in flight (guards against dispatching a second).
    fixing: bool,
    /// Outcome of the most recent fix attempt.
    last_fix: Option<String>,
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
    /// Set (`Some`) or clear (`None`) a loop's on-change trigger command.
    Trigger {
        id: u64,
        cmd: Option<String>,
        reply: oneshot::Sender<bool>,
    },
    /// Enable/disable auto-fix for a loop (`pr`: land as a PR, not a merge).
    Fix {
        id: u64,
        on: bool,
        pr: bool,
        reply: oneshot::Sender<bool>,
    },
    /// Set (`Some`) or clear (`None`) a loop's fire window `(start, end, weekdays)`.
    Window {
        id: u64,
        window: Option<(u8, u8, bool)>,
        reply: oneshot::Sender<bool>,
    },
    List {
        reply: oneshot::Sender<Vec<LoopSpec>>,
    },
    /// Stop admitting work, cancel every manager-owned child, and acknowledge
    /// only after firing, trigger, and auto-fix tasks have reaped/cleaned up.
    Shutdown { reply: oneshot::Sender<()> },
}

/// The UI's handle to the loop manager.
pub(crate) struct LoopsHandle {
    pub(crate) ctl: mpsc::UnboundedSender<LoopCtl>,
    /// Firing results awaiting display; the UI drains this on ticks.
    pub(crate) pending: Arc<Mutex<Vec<LoopLine>>>,
    /// Live per-loop state for the `/watch` dashboard; the manager keeps it
    /// current on every state change.
    pub(crate) snapshot: Arc<Mutex<Vec<LoopWatchRow>>>,
    task: tokio::task::JoinHandle<()>,
}

struct ManagerContext {
    activity: Option<PathBuf>,
    event_sink: Option<Arc<dyn hi_events::EventSink>>,
    _fire_lock: Option<Arc<crate::lock::FireLock>>,
}

impl LoopsHandle {
    /// Take any queued transcript lines (called from UI tick arms).
    pub(crate) fn drain(&self) -> Vec<LoopLine> {
        std::mem::take(&mut *self.pending.lock().unwrap())
    }

    /// Cooperatively stop the manager and wait for all child process trees and
    /// auto-fix worktrees to settle. Consuming the handle prevents later
    /// controls from racing the acknowledged shutdown.
    pub(crate) async fn shutdown(self) -> Result<(), String> {
        let Self { ctl, task, .. } = self;
        let (reply, acknowledged) = oneshot::channel();
        let requested = ctl.send(LoopCtl::Shutdown { reply }).is_ok();
        let acknowledgement = if requested {
            acknowledged.await.map_err(|_| {
                "recurring-loop manager stopped before shutdown was acknowledged".to_string()
            })
        } else {
            Err("recurring-loop manager stopped before shutdown was requested".to_string())
        };
        drop(ctl);
        task.await
            .map_err(|error| format!("recurring-loop manager failed during shutdown: {error}"))?;
        acknowledgement
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn loop_expired(spec: &LoopSpec, now: u64) -> bool {
    spec.expires_ms.is_some_and(|expires| expires <= now)
}

/// The next fire time (unix ms) after a fire decision. Inside the window (or when
/// there is no window) it's a full interval away; *outside* the window we re-check
/// within `WINDOW_RECHECK_SECS` so a long-interval loop re-enters its window at
/// the next opening instead of keeping its out-of-window phase forever.
fn next_fire_ms(now: u64, interval_secs: u64, in_window: bool) -> u64 {
    let step = if in_window {
        interval_secs
    } else {
        interval_secs.min(WINDOW_RECHECK_SECS)
    };
    now + step * 1000
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
///
/// The child is asked to reply *exactly* the marker, but often prepends a short
/// lead-in ("checked again — NOTHING NEW"). We accept that, but only when the
/// marker is set off as its own final line or by a separator (dash/colon) — so a
/// genuinely *loud* summary that merely ends with the words "nothing new" (e.g.
/// "the banner now reads NOTHING NEW") isn't silently suppressed.
pub(crate) fn is_quiet(summary: &str) -> bool {
    let s = summary.trim().trim_end_matches('.').trim();
    if s.eq_ignore_ascii_case(QUIET_MARKER) {
        return true;
    }
    let last_line = s.lines().last().unwrap_or(s).trim();
    if last_line.eq_ignore_ascii_case(QUIET_MARKER) {
        return true;
    }
    let lower = last_line.to_ascii_lowercase();
    if let Some(prefix) = lower.strip_suffix(&QUIET_MARKER.to_ascii_lowercase()) {
        return prefix.trim_end().ends_with(['—', '–', '-', ':']);
    }
    false
}

/// Spawn the loop manager: loads persisted loops (dropping expired ones),
/// re-arms the rest, and runs the timer wheel until the TUI exits.
#[cfg(test)]
pub(crate) fn start(
    launcher: Arc<FleetLauncher>,
    loops_file: Option<PathBuf>,
    event_sink: Option<Arc<dyn hi_events::EventSink>>,
) -> LoopsHandle {
    start_with_fire_lock(launcher, loops_file, event_sink, None)
}

/// Start a TUI-owned manager while sharing ownership of its fire lock. If the
/// TUI unwinds before it can request acknowledged shutdown, the manager keeps
/// the lock until its channel-close cancellation has joined every child.
pub(crate) fn start_with_fire_lock(
    launcher: Arc<FleetLauncher>,
    loops_file: Option<PathBuf>,
    event_sink: Option<Arc<dyn hi_events::EventSink>>,
    fire_lock: Option<Arc<crate::lock::FireLock>>,
) -> LoopsHandle {
    let (ctl_tx, ctl_rx) = mpsc::unbounded_channel();
    let pending: Arc<Mutex<Vec<LoopLine>>> = Arc::new(Mutex::new(Vec::new()));
    let snapshot: Arc<Mutex<Vec<LoopWatchRow>>> = Arc::new(Mutex::new(Vec::new()));
    let context = ManagerContext {
        activity: loops_file
            .as_ref()
            .map(|path| crate::activity::activity_path(path)),
        event_sink,
        _fire_lock: fire_lock,
    };
    let task = tokio::spawn(manager(
        launcher,
        loops_file,
        context,
        ctl_rx,
        pending.clone(),
        snapshot.clone(),
    ));
    LoopsHandle {
        ctl: ctl_tx,
        pending: pending.clone(),
        snapshot: snapshot.clone(),
        task,
    }
}

/// Append one loud event to the project's activity feed (best-effort).
fn record(
    activity: Option<&std::path::Path>,
    event_sink: Option<&dyn hi_events::EventSink>,
    loop_id: u64,
    source: &str,
    text: &str,
) {
    if let Some(path) = activity {
        crate::activity::append(
            path,
            &crate::activity::ActivityEntry {
                at_ms: now_ms(),
                loop_id,
                source: source.to_string(),
                text: text.to_string(),
                event_id: None,
                group_key: None,
                state: None,
                detail: None,
            },
        );
    }
    if let Some(sink) = event_sink {
        let _ = sink.publish(hi_events::RunEvent::new(
            hi_events::EventKind::LoopFired,
            hi_events::EventContext::default(),
            hi_events::SemanticActivity {
                verb: hi_events::ActivityVerb::Change,
                object: hi_events::ActivityObject::Loop,
                state: hi_events::ActivityState::Succeeded,
                group_key: format!("loop:{loop_id}"),
                title: source.to_string(),
                detail: Some(text.to_string()),
                refs: vec![],
                progress: None,
            },
        ));
    }
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
                trigger: l.trigger.clone(),
                last_trigger: rt.and_then(|r| r.last_trigger.clone()),
                autofix: l.autofix,
                fix_pr: l.fix_pr,
                window: l.schedule.map(|s| s.label()),
                fixing: rt.is_some_and(|r| r.fixing),
                last_fix: rt.and_then(|r| r.last_fix.clone()),
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
    context: ManagerContext,
    mut ctl: mpsc::UnboundedReceiver<LoopCtl>,
    pending: Arc<Mutex<Vec<LoopLine>>>,
    snapshot: Arc<Mutex<Vec<LoopWatchRow>>>,
) {
    let activity = context.activity.as_deref();
    let event_sink = context.event_sink.as_deref();
    // Reach-you notifications for loud events (opt-in via env; no-op otherwise).
    let notify = crate::notify::NotifyConfig::from_env();
    // Every trigger gets a child token. The drop guard also cancels them when
    // the manager future is dropped/aborted, not just when ctl closes cleanly.
    let manager_cancel = CancellationToken::new();
    let _manager_cancel_guard = manager_cancel.clone().drop_guard();
    let mut firing_cancellations: HashMap<u64, CancellationToken> = HashMap::new();
    let mut trigger_cancellations: HashMap<u64, CancellationToken> = HashMap::new();
    let mut fix_cancellations: HashMap<u64, CancellationToken> = HashMap::new();
    // Every manager-owned task is joined during acknowledged shutdown. The
    // cancellation tokens stop its native child/process group; the JoinSet
    // proves cleanup and reaping finished before a workspace can be rebound.
    let mut children = JoinSet::new();
    let mut runtime: HashMap<u64, LoopRuntime> = HashMap::new();
    let mut state = load(loops_file.as_deref());
    let now = now_ms();
    let before = state.loops.len();
    state.loops.retain(|loop_| !loop_expired(loop_, now));
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
    // On-change trigger runs report their outcome line over this channel.
    let (trig_tx, mut trig_rx) = mpsc::unbounded_channel::<(u64, String)>();
    // Auto-fix attempts report (id, outcome-line, loud) over this channel.
    let (fix_tx, mut fix_rx) = mpsc::unbounded_channel::<(u64, String, bool)>();
    let mut in_flight: usize = 0;

    loop {
        let now = now_ms();
        // Fire due loops while respecting a small concurrency cap. `load`
        // removes the automatic expiry written by older versions.
        let mut fired = false;
        state.loops.retain(|l| {
            if loop_expired(l, now) {
                if let Some(cancel) = firing_cancellations.remove(&l.id) {
                    cancel.cancel();
                }
                if let Some(cancel) = trigger_cancellations.remove(&l.id) {
                    cancel.cancel();
                }
                if let Some(cancel) = fix_cancellations.remove(&l.id) {
                    cancel.cancel();
                }
                record(
                    activity,
                    event_sink,
                    l.id,
                    &format!("loop#{} {}", l.id, l.name()),
                    "expired",
                );
                pending
                    .lock()
                    .unwrap()
                    .push((format!("⟳ loop#{} ({}) expired", l.id, l.name()), true));
                fired = true;
                false
            } else {
                true
            }
        });
        for spec in &mut state.loops {
            // Also require this loop isn't already firing. `next_ms` is bumped at
            // spawn, but a firing can outlive its interval (a 60s loop whose turn
            // takes 90s), or a `FireNow` can land mid-flight — and re-firing would
            // resume the *same* session file in a second child, corrupting the
            // session/report and double-counting spend. One firing per loop.
            if !spec.paused
                && spec.next_ms <= now
                && in_flight < 2
                && !runtime.get(&spec.id).is_some_and(|r| r.firing)
            {
                // Outside its fire window? Re-check soon rather than deferring a
                // whole interval — a day-aligned loop armed outside its window
                // would otherwise never re-enter it. It fires shortly after open.
                if spec.schedule.is_some_and(|s| !s.is_active_now()) {
                    spec.next_ms = next_fire_ms(now, spec.interval_secs, false);
                    fired = true; // next_ms changed → persist
                    continue;
                }
                spec.next_ms = next_fire_ms(now, spec.interval_secs, true);
                spec.firings += 1;
                in_flight += 1;
                fired = true;
                runtime.entry(spec.id).or_default().firing = true;
                let launcher = launcher.clone();
                let spec_snapshot = spec.clone();
                let done = done_tx.clone();
                let cancel = manager_cancel.child_token();
                firing_cancellations.insert(spec.id, cancel.clone());
                children.spawn(async move {
                    let result = run_firing(&launcher, &spec_snapshot, cancel).await;
                    let _ = done.send((spec_snapshot.id, result));
                });
            }
        }
        if fired {
            save(loops_file.as_deref(), &state);
        }
        publish(&state, &mut runtime, &snapshot);

        // Sleep until the next due time (capped so controls stay responsive).
        // A paused, unlimited loop contributes no due time; otherwise its stale
        // `next_ms` would pin the sleep to the 250ms floor and spin.
        let next_due = state
            .loops
            .iter()
            .filter_map(|l| {
                if l.paused {
                    l.expires_ms
                } else {
                    Some(
                        l.expires_ms
                            .map_or(l.next_ms, |expires| l.next_ms.min(expires)),
                    )
                }
            })
            .min()
            .unwrap_or(now + 60_000);
        let sleep_ms = next_due.saturating_sub(now).clamp(250, 30_000);

        tokio::select! {
            maybe = ctl.recv() => {
                let Some(msg) = maybe else {
                    manager_cancel.cancel();
                    while children.join_next().await.is_some() {}
                    settle_firings_for_shutdown(&mut state, &mut runtime, &mut done_rx);
                    save(loops_file.as_deref(), &state);
                    return;
                }; // UI gone — stop
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
                                        expires_ms: None,
                                        // First firing right away.
                                        next_ms: now,
                                        session,
                                        firings: 0,
                                        paused: false,
                                        token_budget: None,
                                        spent_tokens: 0,
                                        trigger: None,
                                        autofix: false,
                                        fix_pr: false,
                                        schedule: None,
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
                            if let Some(cancel) = firing_cancellations.remove(&id) {
                                cancel.cancel();
                            }
                            if let Some(cancel) = trigger_cancellations.remove(&id) {
                                cancel.cancel();
                            }
                            if let Some(cancel) = fix_cancellations.remove(&id) {
                                cancel.cancel();
                            }
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
                    LoopCtl::Trigger { id, cmd, reply } => {
                        let mut ok = false;
                        for l in &mut state.loops {
                            if l.id == id {
                                l.trigger = cmd.clone();
                                // A changed/cleared command supersedes any
                                // currently-running invocation for this loop.
                                if let Some(cancel) = trigger_cancellations.remove(&id) {
                                    cancel.cancel();
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
                    LoopCtl::Fix { id, on, pr, reply } => {
                        let mut ok = false;
                        for l in &mut state.loops {
                            if l.id == id {
                                l.autofix = on;
                                l.fix_pr = on && pr;
                                ok = true;
                            }
                        }
                        if ok {
                            save(loops_file.as_deref(), &state);
                            publish(&state, &mut runtime, &snapshot);
                        }
                        let _ = reply.send(ok);
                    }
                    LoopCtl::Window { id, window, reply } => {
                        let mut ok = false;
                        for l in &mut state.loops {
                            if l.id == id {
                                l.schedule = window.map(|(start_hour, end_hour, weekdays_only)| {
                                    Schedule {
                                        start_hour,
                                        end_hour,
                                        weekdays_only,
                                    }
                                });
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
                    LoopCtl::Shutdown { reply } => {
                        manager_cancel.cancel();
                        while children.join_next().await.is_some() {}
                        settle_firings_for_shutdown(&mut state, &mut runtime, &mut done_rx);
                        save(loops_file.as_deref(), &state);
                        let _ = reply.send(());
                        return;
                    }
                }
            }
            _ = children.join_next(), if !children.is_empty() => {}
            Some((id, result)) = done_rx.recv() => {
                in_flight = in_flight.saturating_sub(1);
                firing_cancellations.remove(&id);
                if !state.loops.iter().any(|loop_| loop_.id == id) {
                    continue;
                }
                let name = state
                    .loops
                    .iter()
                    .find(|l| l.id == id)
                    .map(LoopSpec::name)
                    .unwrap_or_else(|| format!("#{id}"));
                let fired_ms = now_ms();
                let errored = result.is_err();
                let (line, summary, quiet, tokens) = match result {
                    Ok(outcome) => {
                        let parked = outcome
                            .summary
                            .to_ascii_lowercase()
                            .contains("parked for approval");
                        let quiet = !parked && is_quiet(&outcome.summary);
                        let text = if parked {
                            format!(
                                "⏸ loop#{id} ({name}): parked for approval — /inbox"
                            )
                        } else if quiet {
                            format!("⟳ loop#{id} ({name}): nothing new")
                        } else {
                            format!("⟳ loop#{id} ({name}): {}", truncate(&outcome.summary, 160))
                        };
                        ((text, !quiet || parked), outcome.summary, quiet, Some(outcome.total_tokens))
                    }
                    Err(err) => {
                        let text = format!("⟳ loop#{id} ({name}) firing failed: {err}");
                        ((text, true), format!("firing failed: {err}"), false, None)
                    }
                };
                // A firing that failed to launch/run never established or advanced
                // the baseline, so roll back the at-spawn `firings += 1`. Otherwise
                // a failed FIRST firing leaves firings == 1, and the next firing
                // (firings == 2) is told to "compare against previous checks, reply
                // NOTHING NEW" against a session that has no baseline — silently
                // suppressing the first genuine report.
                //
                // Also re-arm soon: `next_ms` was advanced at spawn to a full
                // interval out. Leaving it there strands a flaky first launch
                // (cold stub / busy machine) until the whole cadence elapses.
                if errored
                    && let Some(l) = state.loops.iter_mut().find(|l| l.id == id)
                {
                    l.firings = l.firings.saturating_sub(1);
                    let now = now_ms();
                    if !l.paused {
                        l.next_ms = now + 2_000;
                    }
                    save(loops_file.as_deref(), &state);
                }
                // Fold in the cost and enforce the budget: `total_tokens` is
                // session-cumulative, so it *is* the loop's running spend.
                let mut budget_line: Option<LoopLine> = None;
                if let Some(spent) = tokens {
                    if let Some(l) = state.loops.iter_mut().find(|l| l.id == id) {
                        // Never let a missing/torn report (spent == 0) clobber the
                        // running total — cost history only ever grows.
                        l.spent_tokens = l.spent_tokens.max(spent);
                        if let Some(budget) = l.token_budget
                            && !l.paused
                            && spent >= budget
                        {
                            l.paused = true;
                            let msg = format!(
                                "paused — hit token budget ({} / {})",
                                fmt_tokens(spent),
                                fmt_tokens(budget),
                            );
                            record(activity, event_sink, id, &format!("loop#{id} {name}"), &msg);
                            budget_line = Some((format!("⏸ loop#{id} ({name}) {msg}"), true));
                        }
                    }
                    save(loops_file.as_deref(), &state);
                }
                let parked = summary.to_ascii_lowercase().contains("parked for approval");
                if parked
                    && let Some(l) = state.loops.iter_mut().find(|l| l.id == id)
                {
                    l.paused = true;
                    save(loops_file.as_deref(), &state);
                    record(
                        activity,
                        event_sink,
                        id,
                        &format!("loop#{id} {name}"),
                        "parked for approval — /inbox",
                    );
                }
                // On a genuine loud change (not quiet, not a firing error, not a
                // parked confirm), record it to the activity feed and run the
                // loop's on-change trigger.
                let loud_change = tokens.is_some() && !quiet && !parked;
                if loud_change {
                    record(activity, event_sink, id, &format!("loop#{id} {name}"), &summary);
                }
                if loud_change
                    && let Some(cmd) = state
                        .loops
                        .iter()
                        .find(|l| l.id == id)
                        .and_then(|l| l.trigger.clone())
                {
                    let trig = trig_tx.clone();
                    let (name, summary) = (name.clone(), summary.clone());
                    let cancel = trigger_cancellations
                        .entry(id)
                        .or_insert_with(|| manager_cancel.child_token())
                        .clone();
                    children.spawn(async move {
                        let outcome = run_trigger(&cmd, id, &name, &summary, cancel).await;
                        let _ = trig.send((id, outcome));
                    });
                }
                // Auto-fix: on a loud change, if enabled and no fix is already in
                // flight for this loop, dispatch a worktree-isolated agent to fix
                // it. The merge is verify-gated inside run_fix.
                let autofix_spec = state
                    .loops
                    .iter()
                    .find(|l| l.id == id)
                    .filter(|l| l.autofix)
                    .cloned();
                if loud_change
                    && let Some(spec) = autofix_spec
                    && !runtime.get(&id).is_some_and(|r| r.fixing)
                {
                    runtime.entry(id).or_default().fixing = true;
                    let launcher = launcher.clone();
                    let fix = fix_tx.clone();
                    let summary = summary.clone();
                    let cancel = manager_cancel.child_token();
                    fix_cancellations.insert(id, cancel.clone());
                    children.spawn(async move {
                        let (line, loud) = run_fix(&launcher, &spec, &summary, cancel).await;
                        let _ = fix.send((spec.id, line, loud));
                    });
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
                if line.1 {
                    crate::notify::maybe_notify(&notify, &format!("loop#{id} {name}"), &line.0);
                }
                pending.lock().unwrap().push(line);
                if let Some(bl) = budget_line {
                    crate::notify::maybe_notify(&notify, &format!("loop#{id} {name}"), &bl.0);
                    pending.lock().unwrap().push(bl);
                }
                publish(&state, &mut runtime, &snapshot);
            }
            Some((id, outcome)) = trig_rx.recv() => {
                // Cancellation/expiry can race the trigger result. Do not
                // resurrect dashboard runtime for a loop that no longer exists.
                if !state.loops.iter().any(|loop_| loop_.id == id) {
                    continue;
                }
                let name = state
                    .loops
                    .iter()
                    .find(|l| l.id == id)
                    .map(LoopSpec::name)
                    .unwrap_or_else(|| format!("#{id}"));
                let failed = !outcome.starts_with("ok");
                pending
                    .lock()
                    .unwrap()
                    .push((format!("⚡ loop#{id} ({name}) trigger: {outcome}"), failed));
                runtime.entry(id).or_default().last_trigger = Some(outcome);
                publish(&state, &mut runtime, &snapshot);
            }
            Some((id, outcome, loud)) = fix_rx.recv() => {
                fix_cancellations.remove(&id);
                if !state.loops.iter().any(|loop_| loop_.id == id) {
                    continue;
                }
                let name = state
                    .loops
                    .iter()
                    .find(|l| l.id == id)
                    .map(LoopSpec::name)
                    .unwrap_or_else(|| format!("#{id}"));
                record(
                    activity,
                    event_sink,
                    id,
                    &format!("loop#{id} {name}"),
                    &format!("auto-fix: {outcome}"),
                );
                if loud {
                    crate::notify::maybe_notify(
                        &notify,
                        &format!("loop#{id} {name} auto-fix"),
                        &outcome,
                    );
                }
                pending
                    .lock()
                    .unwrap()
                    .push((format!("⚒ loop#{id} ({name}) auto-fix: {outcome}"), loud));
                let rt = runtime.entry(id).or_default();
                rt.fixing = false;
                rt.last_fix = Some(outcome);
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

/// Settle firing results after shutdown cancellation without dispatching new
/// trigger/autofix work. A child that completed before observing cancellation
/// keeps its cadence and spend; cancelled, failed, or panicked children have
/// their at-spawn reservation rolled back and are due again shortly.
fn settle_firings_for_shutdown(
    state: &mut LoopsFile,
    runtime: &mut HashMap<u64, LoopRuntime>,
    done: &mut mpsc::UnboundedReceiver<(u64, Result<FiringOutcome, String>)>,
) {
    while let Ok((id, result)) = done.try_recv() {
        let Some(loop_runtime) = runtime.get_mut(&id).filter(|runtime| runtime.firing) else {
            continue;
        };
        let Ok(outcome) = result else {
            continue;
        };
        loop_runtime.firing = false;
        loop_runtime.last_quiet = is_quiet(&outcome.summary);
        loop_runtime.last_summary = Some(outcome.summary.clone());
        loop_runtime.last_fired_ms = now_ms();
        if let Some(spec) = state.loops.iter_mut().find(|spec| spec.id == id) {
            spec.spent_tokens = spec.spent_tokens.max(outcome.total_tokens);
            if spec
                .token_budget
                .is_some_and(|budget| spec.spent_tokens >= budget)
                || outcome
                    .summary
                    .to_ascii_lowercase()
                    .contains("parked for approval")
            {
                spec.paused = true;
            }
        }
    }

    let now = now_ms();
    for spec in &mut state.loops {
        if let Some(loop_runtime) = runtime.get_mut(&spec.id)
            && loop_runtime.firing
        {
            spec.firings = spec.firings.saturating_sub(1);
            if !spec.paused {
                spec.next_ms = now.saturating_add(2_000);
            }
            loop_runtime.firing = false;
        }
    }
}

/// Whether a child-output line is decoration rather than reply text: a tool-call
/// glyph line, or the trailing usage footer (`[↑… ↓… · ctx …]`) the one-shot
/// child prints after its reply. Excluding the footer keeps it from being picked
/// as the firing summary (it isn't a decoration glyph, so a naive filter missed
/// it — caught in a live daemon run).
fn is_decoration_line(line: &str) -> bool {
    let l = line.trim_start();
    l.starts_with(['⏺', '✓', '⚙', '›', '↳'])
        || (l.starts_with('[') && l.contains('↑') && l.contains("ctx"))
}

/// One firing: a fleet-style child run in the real cwd, resuming the loop's
/// session. Returns the child's final reply line as the summary, plus the
/// session-cumulative token total read back from its `--report`.
fn loop_turn_timeout_from_value(value: Option<&str>) -> Option<Duration> {
    value
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|seconds| *seconds > 0)
        .map(Duration::from_secs)
        .filter(|timeout| std::time::Instant::now().checked_add(*timeout).is_some())
}

fn loop_turn_timeout() -> Option<Duration> {
    let configured = std::env::var("HI_LOOP_TURN_TIMEOUT_SECS").ok();
    loop_turn_timeout_from_value(configured.as_deref())
}

fn child_turn_deadline_secs(outer_timeout: Option<Duration>) -> Option<u64> {
    outer_timeout.map(|timeout| {
        timeout
            .as_secs()
            .saturating_sub(CHILD_SETTLEMENT_GRACE_SECS)
            .max(1)
    })
}

fn loop_child_command(
    launcher: &FleetLauncher,
    outer_timeout: Option<Duration>,
) -> tokio::process::Command {
    let mut cmd = tokio::process::Command::new(&launcher.exe);
    cmd.env("HI_API_KEY", &launcher.api_key).args([
        "--provider",
        &launcher.provider,
        "--model",
        &launcher.model,
        "--base-url",
        &launcher.base_url,
    ]);
    if let Some(turn_deadline_secs) = child_turn_deadline_secs(outer_timeout) {
        cmd.args(["--turn-deadline", &turn_deadline_secs.to_string()]);
    }
    // Do not install private loop-specific caps: ordinary children inherit
    // hi's unlimited defaults, while the user's explicit caps are preserved.
    cmd.args(launcher.child_execution_cap_args());
    #[cfg(unix)]
    cmd.process_group(0);
    cmd
}

fn append_fix_verification_args(cmd: &mut tokio::process::Command, launcher: &FleetLauncher) {
    if let Some(verify) = &launcher.verify {
        cmd.args(["--verify", verify]);
        if let Some(max_verify_repairs) = launcher.model_verify_repair_limit() {
            cmd.args(["--max-verify-repairs", &max_verify_repairs.to_string()]);
        }
    }
}

async fn drain_firing_tail(mut stdout: impl AsyncRead + Unpin) -> Vec<String> {
    let mut evidence = VecDeque::with_capacity(FIRING_OUTPUT_CAP);
    let mut chunk = [0_u8; 8192];
    loop {
        match stdout.read(&mut chunk).await {
            Ok(0) | Err(_) => break,
            Ok(read) => {
                evidence.extend(&chunk[..read]);
                while evidence.len() > FIRING_OUTPUT_CAP {
                    evidence.pop_front();
                }
            }
        }
    }
    let evidence: Vec<u8> = evidence.into_iter().collect();
    let text = String::from_utf8_lossy(&evidence);
    let mut tail = VecDeque::with_capacity(50);
    for line in text.lines() {
        let line = crate::dashboard::strip_ansi_line(line);
        let line = line.trim_end();
        if !line.trim().is_empty() {
            tail.push_back(line.to_string());
            if tail.len() > 50 {
                tail.pop_front();
            }
        }
    }
    tail.into_iter().collect()
}

async fn run_firing(
    launcher: &FleetLauncher,
    spec: &LoopSpec,
    cancellation: CancellationToken,
) -> Result<FiringOutcome, String> {
    // One report file per loop, alongside its session, overwritten each firing.
    let report_path = spec.session.with_extension("report.json");
    let turn_timeout = loop_turn_timeout();
    let mut cmd = loop_child_command(launcher, turn_timeout);
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true);
    cmd.arg("--session-file").arg(&spec.session);
    cmd.env("HI_LOOP_ID", spec.id.to_string());
    cmd.arg("--report").arg(&report_path);
    cmd.arg(wrapper_prompt(spec));

    let mut child = cmd.spawn().map_err(|e| format!("couldn't launch: {e}"))?;
    let mut process_group = ChildProcessGroup::for_child(&child);
    let mut stdout_read = tokio::spawn(drain_firing_tail(
        child.stdout.take().expect("piped loop child stdout"),
    ));
    let waited = wait_loop_child(&mut child, turn_timeout, &cancellation).await;
    process_group.kill_now();
    if !matches!(waited, ChildWait::Exited(_)) {
        let _ = child.start_kill();
        let _ = child.wait().await;
    }
    let tail = match tokio::time::timeout(CHILD_PIPE_DRAIN_GRACE, &mut stdout_read).await {
        Ok(Ok(tail)) => tail,
        Ok(Err(_)) => Vec::new(),
        Err(_) => {
            stdout_read.abort();
            let _ = stdout_read.await;
            Vec::new()
        }
    };
    let status = match waited {
        ChildWait::Exited(status) => status.map_err(|e| format!("wait failed: {e}"))?,
        ChildWait::TimedOut(timeout) => {
            return Err(format!("timed out after {}s", timeout.as_secs()));
        }
        ChildWait::Cancelled => return Err("cancelled".to_string()),
    };
    if !status.success() {
        return Err(format!(
            "agent run failed ({}): {}",
            status,
            tail.last().cloned().unwrap_or_default()
        ));
    }
    // The final non-decoration line is the reply's tail — the summary. Skip the
    // trailing usage line (`[↑… ↓… · ctx …]`) the child prints after the reply,
    // which is otherwise picked as the summary (it isn't a decoration glyph).
    let summary = tail
        .iter()
        .rev()
        .find(|l| !is_decoration_line(l))
        .cloned()
        .unwrap_or_else(|| tail.last().cloned().unwrap_or_default());
    let total_tokens = read_report_tokens(&report_path);
    Ok(FiringOutcome {
        summary,
        total_tokens,
    })
}

/// Read the session-cumulative token total from a firing's `--report` JSON.
///
/// Schema v2 nests this value under `usage.session`; the top-level field is a
/// read-only migration fallback for reports written by 0.1 children. Missing
/// or malformed reports yield zero because loop cost tracking is best-effort.
fn read_report_tokens(path: &std::path::Path) -> u64 {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .and_then(|v| {
            v.pointer("/usage/session/total_tokens")
                .and_then(serde_json::Value::as_u64)
                .or_else(|| v.get("total_tokens").and_then(serde_json::Value::as_u64))
        })
        .unwrap_or(0)
}

/// Run a loop's on-change trigger via `sh -c`, passing the loop id/name and the
/// firing's summary in the environment. Returns a compact outcome line — one of
/// `ok`, `ok: <stdout>`, `exit N: <stderr>`, `cancelled`, `timed out …`, or
/// `failed …` — so the transcript and `/watch` can show whether the response
/// actually ran.
fn trigger_timeout_from_value(value: Option<&str>) -> Option<Duration> {
    value
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|seconds| *seconds > 0)
        .map(Duration::from_secs)
        .filter(|timeout| std::time::Instant::now().checked_add(*timeout).is_some())
}

fn trigger_timeout() -> Option<Duration> {
    let configured = std::env::var("HI_LOOP_TRIGGER_TIMEOUT_SECS").ok();
    trigger_timeout_from_value(configured.as_deref())
}

/// Drain an entire pipe while retaining only a small diagnostic prefix. Using
/// chunked reads avoids `lines()` allocating an unbounded buffer for a hostile
/// or accidental megabyte-long line.
async fn drain_trigger_pipe(mut pipe: impl AsyncRead + Unpin) -> Vec<u8> {
    let mut evidence = Vec::with_capacity(TRIGGER_EVIDENCE_CAP);
    let mut chunk = [0_u8; 8192];
    loop {
        match pipe.read(&mut chunk).await {
            Ok(0) | Err(_) => return evidence,
            Ok(read) => {
                let keep = read.min(TRIGGER_EVIDENCE_CAP.saturating_sub(evidence.len()));
                evidence.extend_from_slice(&chunk[..keep]);
            }
        }
    }
}

#[cfg(unix)]
struct ChildProcessGroup {
    pgid: Option<i32>,
}

#[cfg(unix)]
impl ChildProcessGroup {
    fn for_child(child: &tokio::process::Child) -> Self {
        Self {
            pgid: child.id().map(|pid| pid as i32),
        }
    }

    fn kill_now(&mut self) {
        let Some(pgid) = self.pgid.take() else {
            return;
        };
        // SAFETY: a negative pid addresses the private process group created
        // below. A process that already exited simply produces an OS error.
        unsafe {
            libc::kill(-pgid, libc::SIGKILL);
        }
    }
}

#[cfg(unix)]
impl Drop for ChildProcessGroup {
    fn drop(&mut self) {
        self.kill_now();
    }
}

#[cfg(not(unix))]
struct ChildProcessGroup;

#[cfg(not(unix))]
impl ChildProcessGroup {
    fn for_child(_child: &tokio::process::Child) -> Self {
        Self
    }

    fn kill_now(&mut self) {}
}

enum ChildWait {
    Exited(std::io::Result<std::process::ExitStatus>),
    TimedOut(Duration),
    Cancelled,
}

async fn wait_loop_child(
    child: &mut tokio::process::Child,
    timeout: Option<Duration>,
    cancellation: &CancellationToken,
) -> ChildWait {
    let wait = child.wait();
    tokio::pin!(wait);
    match timeout {
        Some(limit) => {
            let deadline = tokio::time::sleep(limit);
            tokio::pin!(deadline);
            tokio::select! {
                biased;
                _ = cancellation.cancelled() => ChildWait::Cancelled,
                status = &mut wait => ChildWait::Exited(status),
                _ = &mut deadline => ChildWait::TimedOut(limit),
            }
        }
        None => {
            tokio::select! {
                biased;
                _ = cancellation.cancelled() => ChildWait::Cancelled,
                status = &mut wait => ChildWait::Exited(status),
            }
        }
    }
}

async fn run_trigger(
    cmd: &str,
    id: u64,
    name: &str,
    summary: &str,
    cancellation: CancellationToken,
) -> String {
    run_trigger_with_timeout(cmd, id, name, summary, cancellation, trigger_timeout()).await
}

async fn run_trigger_with_timeout(
    cmd: &str,
    id: u64,
    name: &str,
    summary: &str,
    cancellation: CancellationToken,
    timeout: Option<Duration>,
) -> String {
    let mut c = tokio::process::Command::new("sh");
    c.arg("-c")
        .arg(cmd)
        .env("HI_LOOP_ID", id.to_string())
        .env("HI_LOOP_NAME", name)
        .env("HI_LOOP_SUMMARY", summary)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    #[cfg(unix)]
    {
        c.process_group(0);
    }
    let mut child = match c.spawn() {
        Ok(child) => child,
        Err(error) => return format!("failed to run: {error}"),
    };
    // The guard is deliberately never defused: even a successful shell can
    // leave a background descendant holding a pipe. It also protects task
    // abort/runtime shutdown paths where cooperative cancellation cannot run.
    let mut process_group = ChildProcessGroup::for_child(&child);
    let mut stdout_read = tokio::spawn(drain_trigger_pipe(
        child.stdout.take().expect("piped trigger stdout"),
    ));
    let mut stderr_read = tokio::spawn(drain_trigger_pipe(
        child.stderr.take().expect("piped trigger stderr"),
    ));

    let waited = wait_loop_child(&mut child, timeout, &cancellation).await;

    // Stop the entire tree for timeout/cancel and also clean up any descendant
    // leaked by a shell that exited successfully. Then bound only pipe/reap
    // cleanup; this is not an execution deadline.
    process_group.kill_now();
    if !matches!(waited, ChildWait::Exited(_)) {
        let _ = child.start_kill();
        let _ = child.wait().await;
    }
    let drained = tokio::time::timeout(TRIGGER_PIPE_DRAIN_GRACE, async {
        tokio::join!(&mut stdout_read, &mut stderr_read)
    })
    .await;
    let (stdout, stderr) = match drained {
        Ok((stdout, stderr)) => (stdout.unwrap_or_default(), stderr.unwrap_or_default()),
        Err(_) => {
            stdout_read.abort();
            stderr_read.abort();
            let _ = stdout_read.await;
            let _ = stderr_read.await;
            (Vec::new(), Vec::new())
        }
    };
    let first_line = |bytes: &[u8]| {
        String::from_utf8_lossy(bytes)
            .lines()
            .find(|l| !l.trim().is_empty())
            .map(|l| truncate(l.trim(), 100))
            .unwrap_or_default()
    };
    match waited {
        ChildWait::Exited(Ok(status)) if status.success() => {
            let head = first_line(&stdout);
            if head.is_empty() {
                "ok".to_string()
            } else {
                format!("ok: {head}")
            }
        }
        ChildWait::Exited(Ok(status)) => {
            let code = status.code().unwrap_or(-1);
            let err = first_line(&stderr);
            if err.is_empty() {
                format!("exit {code}")
            } else {
                format!("exit {code}: {err}")
            }
        }
        ChildWait::Exited(Err(error)) => format!("failed to wait: {error}"),
        ChildWait::TimedOut(limit) => format!("timed out after {}s", limit.as_secs()),
        ChildWait::Cancelled => "cancelled".to_string(),
    }
}

/// The verify-gated verdict for a completed fix attempt. Kept as a pure function
/// so the safety rule — *never merge an unverified change* — is unit-testable in
/// isolation from all the git I/O.
#[derive(Debug, PartialEq, Eq)]
enum FixDecision {
    NotGitRepo,
    NoChanges,
    /// Safe to apply the worktree's diff to the real tree.
    Merge,
    /// Changes exist but must not be merged; carries why.
    Reject(&'static str),
}

fn decide_fix(
    in_repo: bool,
    completed: bool,
    changed_count: usize,
    has_verify: bool,
    verified: bool,
) -> FixDecision {
    if !in_repo {
        FixDecision::NotGitRepo
    } else if changed_count == 0 {
        FixDecision::NoChanges
    } else if !completed {
        FixDecision::Reject("the fixer did not finish cleanly")
    } else if !has_verify {
        FixDecision::Reject("no verify command — set /verify to enable auto-merge")
    } else if !verified {
        FixDecision::Reject("the fix did not pass verify")
    } else {
        FixDecision::Merge
    }
}

/// The task handed to the fix agent, built from the loud change it must resolve.
///
/// Phrased as an **implementation task** on purpose: `hi`'s steering runs a
/// read-only preflight for review-shaped prompts (and any "make no changes"
/// wording), which made an earlier version inspect-but-never-edit. Matching
/// `classify_implementation_intent` ("implementation task" + an edit affordance,
/// no no-edit clause) keeps the fixer in write mode. The verify gate — not the
/// prompt — is the real safety boundary, so no defensive "make no changes"
/// clause is needed here.
fn fix_prompt(spec: &LoopSpec, summary: &str) -> String {
    format!(
        "Implementation task: fix a problem the recurring watch \"{}\" just detected.\n\n\
         Problem:\n{summary}\n\n\
         You are expected to edit files and apply patches in this working copy to make the \
         minimal change that resolves it, then run the verification command to confirm the \
         project builds and its tests pass. Prefer the smallest correct change; if the fix is \
         genuinely unclear, stop and explain rather than guess.",
        spec.name()
    )
}

/// One auto-fix attempt: snapshot the tree, run a write-capable child agent in an
/// isolated worktree to fix the loud change, and — only if the diff passes the
/// verify command — merge it into the real tree. Returns a `(line, loud)` outcome
/// for the transcript. The verify gate ([`decide_fix`]) is the safety boundary:
/// an unverified change is never applied.
/// After a verified fix merges into the working tree, re-verify the *real* tree
/// — which may have drifted while the fix ran (a user edit, another loop's
/// merge) — as ground truth. The worktree verify only proved the fix good against
/// `base`; this closes the gap where the merged *combination* was never checked.
/// On failure we surface it loudly (no auto-revert — the user decides whether to
/// keep or `/undo`).
async fn merged_outcome(
    root: &std::path::Path,
    verify: Option<&str>,
    changed: &[String],
    cancellation: Option<&CancellationToken>,
) -> (String, bool) {
    let combined_ok = match verify {
        Some(verify) => hi_tools::worktree::verify_passes_async(root, verify, cancellation).await,
        None => true,
    };
    if combined_ok {
        (
            format!(
                "fixed & merged {} file(s): {}",
                changed.len(),
                changed.join(", ")
            ),
            true,
        )
    } else {
        (
            format!(
                "⚠ merged {} file(s) but the combined tree fails verify — inspect: {}",
                changed.len(),
                changed.join(", ")
            ),
            true,
        )
    }
}

async fn cleanup_loop_fix(root: &std::path::Path, worktree: &std::path::Path) {
    let cleanup_root = root.to_path_buf();
    let cleanup_worktree = worktree.to_path_buf();
    let _ = tokio::task::spawn_blocking(move || {
        hi_tools::worktree::cleanup(&cleanup_root, std::slice::from_ref(&cleanup_worktree));
    })
    .await;
}

async fn run_fix(
    launcher: &FleetLauncher,
    spec: &LoopSpec,
    summary: &str,
    cancellation: CancellationToken,
) -> (String, bool) {
    use hi_tools::worktree;

    if cancellation.is_cancelled() {
        return ("cancelled".into(), false);
    }
    let root = launcher.workspace_root.clone();
    let in_git = {
        let root = root.clone();
        tokio::task::spawn_blocking(move || worktree::in_git_repo(&root))
            .await
            .unwrap_or(false)
    };
    if cancellation.is_cancelled() {
        return ("cancelled".into(), false);
    }
    if !in_git {
        return ("skipped — not a git repository".into(), false);
    }
    let base = match hi_tools::checkpoint::create(&root).await {
        Some(b) => b,
        None => return ("skipped — couldn't snapshot the working tree".into(), true),
    };
    if cancellation.is_cancelled() {
        return ("cancelled".into(), false);
    }
    let wt = std::env::temp_dir().join(format!(
        "hi-loopfix-{}-{}-{}",
        std::process::id(),
        spec.id,
        base.chars().take(12).collect::<String>()
    ));
    let setup_root = root.clone();
    let setup_wt = wt.clone();
    let setup_base = base.clone();
    let setup = tokio::task::spawn_blocking(move || {
        worktree::cleanup(&setup_root, std::slice::from_ref(&setup_wt));
        // `git worktree remove` cannot remove an unregistered directory left by a
        // killed process (or by a previous failed add). Clear that stale path too.
        let _ = std::fs::remove_dir_all(&setup_wt);
        worktree::add_worktree(&setup_root, &setup_wt, &setup_base)
    })
    .await;
    if let Err(e) = match setup {
        Ok(result) => result,
        Err(error) => Err(anyhow::anyhow!("worktree setup worker failed: {error}")),
    } {
        return (format!("skipped — worktree setup failed: {e}"), true);
    }
    if cancellation.is_cancelled() {
        cleanup_loop_fix(&root, &wt).await;
        return ("cancelled".into(), false);
    }

    // A write-capable child agent runs the fix in the worktree, self-verifying
    // via `--verify` if the session has one.
    let turn_timeout = loop_turn_timeout();
    let mut cmd = loop_child_command(launcher, turn_timeout);
    cmd.current_dir(&wt)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true);
    append_fix_verification_args(&mut cmd, launcher);
    cmd.arg(fix_prompt(spec, summary));

    let completed = match cmd.spawn() {
        Ok(mut child) => {
            let mut process_group = ChildProcessGroup::for_child(&child);
            let waited = wait_loop_child(&mut child, turn_timeout, &cancellation).await;
            process_group.kill_now();
            if !matches!(waited, ChildWait::Exited(_)) {
                let _ = child.start_kill();
                let _ = child.wait().await;
            }
            match waited {
                ChildWait::Exited(Ok(status)) => status.success(),
                ChildWait::Exited(Err(_)) | ChildWait::TimedOut(_) | ChildWait::Cancelled => false,
            }
        }
        Err(e) => {
            cleanup_loop_fix(&root, &wt).await;
            return (format!("skipped — couldn't launch the fixer: {e}"), true);
        }
    };
    if cancellation.is_cancelled() {
        cleanup_loop_fix(&root, &wt).await;
        return ("cancelled".into(), false);
    }

    let has_verify = launcher.verify.is_some();
    // Ground-truth re-verify of the final worktree state before any merge.
    let wt_for_check = wt.clone();
    let base_for_check = base.clone();
    let changed = tokio::task::spawn_blocking(move || {
        worktree::changed_files(&wt_for_check, &base_for_check)
    })
    .await
    .unwrap_or_default();
    let verified = if completed && !changed.is_empty() {
        match launcher.verify.as_deref() {
            Some(verify) => worktree::verify_passes_async(&wt, verify, Some(&cancellation)).await,
            None => false,
        }
    } else {
        false
    };
    if cancellation.is_cancelled() {
        cleanup_loop_fix(&root, &wt).await;
        return ("cancelled".into(), false);
    }

    let result = match decide_fix(true, completed, changed.len(), has_verify, verified) {
        // PR mode: land the verified fix as a reviewable branch + PR.
        FixDecision::Merge if spec.fix_pr => {
            let wt_for_pr = wt.clone();
            let spec_for_pr = spec.clone();
            let summary_for_pr = summary.to_string();
            let changed_for_pr = changed.clone();
            let cancellation_for_pr = cancellation.clone();
            tokio::task::spawn_blocking(move || {
                open_fix_pr(
                    &wt_for_pr,
                    &spec_for_pr,
                    &summary_for_pr,
                    &changed_for_pr,
                    &cancellation_for_pr,
                )
            })
            .await
            .unwrap_or_else(|error| (format!("verified, but PR worker failed: {error}"), true))
        }
        // Merge mode: apply the verified diff into the working tree, then
        // re-verify the merged real tree (see merged_outcome — the base may have
        // drifted during the fix).
        FixDecision::Merge => {
            let applied =
                worktree::apply_changes_to_async(&wt, &base, &root, Some(&cancellation)).await;
            match applied {
                Ok(_) => {
                    merged_outcome(
                        &root,
                        launcher.verify.as_deref(),
                        &changed,
                        Some(&cancellation),
                    )
                    .await
                }
                Err(error) => (format!("verified but merge failed: {error}"), true),
            }
        }
        FixDecision::NoChanges => ("made no changes".into(), false),
        FixDecision::Reject(why) => (
            format!("{} file(s) changed but NOT merged — {why}", changed.len()),
            true,
        ),
        FixDecision::NotGitRepo => ("skipped — not a git repository".into(), false),
    };
    cleanup_loop_fix(&root, &wt).await;
    result
}

/// Land a verified fix as a reviewable branch + PR instead of a working-tree
/// merge. Commits the worktree's diff on a fresh branch, pushes it, and opens a
/// PR with `gh`. Degrades gracefully: no remote → left on a local branch; no
/// `gh` → a pushed branch to open a PR from. The branch persists after the
/// worktree is cleaned up (it lives in the shared repo).
fn open_fix_pr(
    worktree: &std::path::Path,
    spec: &LoopSpec,
    summary: &str,
    changed: &[String],
    cancellation: &CancellationToken,
) -> (String, bool) {
    use hi_tools::worktree;
    if cancellation.is_cancelled() {
        return ("cancelled".to_string(), false);
    }
    let name = spec.name();
    let branch = format!("hi-autofix/loop{}-{}", spec.id, now_ms());
    let commit_msg = format!("hi auto-fix: {name}\n\n{}", truncate(summary, 500));
    if let Err(e) = worktree::commit_to_branch(worktree, &branch, &commit_msg) {
        return (
            format!("verified, but couldn't prepare the PR branch: {e}"),
            true,
        );
    }
    if cancellation.is_cancelled() {
        return (
            format!("cancelled after committing fix to local branch {branch}"),
            false,
        );
    }
    if let Err(e) = worktree::push_branch(worktree, &branch) {
        return (
            format!("fix committed to branch {branch} (couldn't push: {e}) — review it locally"),
            true,
        );
    }
    if cancellation.is_cancelled() {
        return (
            format!("cancelled after pushing fix branch {branch}; no PR was opened"),
            false,
        );
    }
    // Open the PR (best-effort; the pushed branch stands alone if `gh` is absent).
    let title = format!("hi auto-fix: {name}");
    let body = format!(
        "A recurring `hi` watch (\"{name}\") detected a problem and an agent produced a \
         verify-passing fix.\n\n**Problem**\n\n{summary}\n\n**Changed files**\n\n{}\n",
        changed
            .iter()
            .map(|f| format!("- `{f}`"))
            .collect::<Vec<_>>()
            .join("\n"),
    );
    match std::process::Command::new("gh")
        .current_dir(worktree)
        .args([
            "pr", "create", "--head", &branch, "--title", &title, "--body", &body,
        ])
        .output()
    {
        Ok(o) if o.status.success() => {
            let url = String::from_utf8_lossy(&o.stdout).trim().to_string();
            (format!("opened PR: {url}"), true)
        }
        _ => (
            format!("fix pushed to branch {branch} — open a PR to land it"),
            true,
        ),
    }
}

/// How many loops are persisted for this project (for the daemon startup line).
pub(crate) fn persisted_count(loops_file: &std::path::Path) -> usize {
    load(Some(loops_file)).loops.len()
}

fn load(path: Option<&std::path::Path>) -> LoopsFile {
    let Some(path) = path else {
        return LoopsFile::default();
    };
    let Ok(text) = std::fs::read_to_string(path) else {
        return LoopsFile::default(); // no file yet — a fresh project
    };
    match serde_json::from_str::<LoopsFile>(&text) {
        Ok(mut state) => {
            // Older releases assigned every loop an automatic seven-day
            // `expires_ms`; there was no user-facing way to request a finite
            // lifetime. Migrate before the manager can prune an already-expired
            // record, and persist via the same atomic sibling-rename as normal
            // loop updates so the deadline cannot return after restart.
            let migrated = state.loops.iter_mut().fold(false, |migrated, loop_| {
                let had_legacy_expiry = loop_.expires_ms.is_some();
                loop_.expires_ms = None;
                migrated || had_legacy_expiry
            });
            if migrated {
                save(Some(path), &state);
            }
            state
        }
        Err(_) => {
            // A corrupt/truncated loops.json would otherwise be silently replaced
            // by an empty set — losing every persisted loop. Preserve it aside so
            // it's recoverable rather than clobbered by the next save.
            let _ = std::fs::rename(path, path.with_extension("json.corrupt"));
            LoopsFile::default()
        }
    }
}

fn save(path: Option<&std::path::Path>, state: &LoopsFile) {
    let Some(path) = path else { return };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let Ok(json) = serde_json::to_string_pretty(state) else {
        return;
    };
    // Write a temp sibling then atomically rename into place, so a crash mid-write
    // can't leave a truncated loops.json (which load() would parse-fail and drop).
    // rename within a directory is atomic on POSIX and Windows.
    let tmp = path.with_extension("json.tmp");
    if std::fs::write(&tmp, &json).is_ok() && std::fs::rename(&tmp, path).is_err() {
        // Keep the prior state intact rather than risking a truncating write.
        let _ = std::fs::remove_file(&tmp);
    }
}

/// Pause or resume a loop in `loops.json` without a live manager (CLI `/inbox`).
pub fn set_loop_paused(path: &std::path::Path, id: u64, paused: bool) -> bool {
    let mut state = load(Some(path));
    let mut ok = false;
    for l in &mut state.loops {
        if l.id == id {
            l.paused = paused;
            if !paused {
                let now = now_ms();
                if l.next_ms < now {
                    l.next_ms = now + 2_000;
                }
            }
            ok = true;
        }
    }
    if ok {
        save(Some(path), &state);
    }
    ok
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

    /// Init a throwaway git repo with one commit.
    fn init_git_repo(dir: &std::path::Path) {
        let git = |args: &[&str]| {
            std::process::Command::new("git")
                .args(args)
                .current_dir(dir)
                .output()
                .unwrap();
        };
        git(&["init", "-q", "-b", "main"]);
        git(&["config", "user.email", "t@t"]);
        git(&["config", "user.name", "t"]);
        std::fs::write(dir.join("README"), "hi\n").unwrap();
        git(&["add", "-A"]);
        git(&["commit", "-qm", "init"]);
    }

    fn spec() -> LoopSpec {
        LoopSpec {
            id: 1,
            prompt: "check whether the CI pipeline on main is green".into(),
            interval_secs: 1800,
            created_ms: 0,
            expires_ms: None,
            next_ms: 0,
            session: PathBuf::from("/tmp/loop.jsonl"),
            firings: 0,
            paused: false,
            token_budget: None,
            spent_tokens: 0,
            trigger: None,
            autofix: false,
            fix_pr: false,
            schedule: None,
        }
    }

    fn command_launcher(max_steps: u32, max_tool_calls: Option<u32>) -> FleetLauncher {
        FleetLauncher {
            exe: PathBuf::from("hi"),
            workspace_root: PathBuf::from("/tmp"),
            provider: "pipenetwork".into(),
            model: "pipe/test".into(),
            base_url: "https://example.test/v1".into(),
            api_key: "test-key".into(),
            verify: None,
            max_verify: 0,
            max_steps: std::sync::atomic::AtomicU32::new(max_steps),
            max_tool_calls: std::sync::atomic::AtomicU64::new(
                max_tool_calls.map(u64::from).unwrap_or(u64::MAX),
            ),
            session_path: Box::new(|| Ok(PathBuf::from("/tmp/unused.jsonl"))),
            sessions: Box::new(Vec::new),
            resume_info: Box::new(|_| None),
            loop_session_path: Box::new(|| Ok(PathBuf::from("/tmp/unused-loop.jsonl"))),
            loops_file: None,
        }
    }

    #[test]
    fn loop_children_have_no_hidden_work_cap_but_preserve_explicit_caps() {
        let uncapped = loop_child_command(&command_launcher(0, None), None);
        let uncapped_args: Vec<_> = uncapped
            .as_std()
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect();
        assert!(
            !uncapped_args.iter().any(|arg| arg == "--max-steps"),
            "an omitted top-level cap must stay omitted: {uncapped_args:?}"
        );
        assert!(
            !uncapped_args.iter().any(|arg| arg == "30" || arg == "40"),
            "legacy private loop caps must not be injected: {uncapped_args:?}"
        );
        assert!(
            !uncapped_args.iter().any(|arg| arg == "--turn-deadline"),
            "ordinary loop work must not receive a hidden wall-clock deadline: {uncapped_args:?}"
        );

        let capped_launcher = command_launcher(7, Some(0));
        let timeout = Some(Duration::from_secs(600));
        let capped = loop_child_command(&capped_launcher, timeout);
        let capped_args: Vec<_> = capped
            .as_std()
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect();
        assert!(
            capped_args
                .windows(2)
                .any(|pair| pair == ["--max-steps", "7"]),
            "the user's explicit cap must propagate: {capped_args:?}"
        );
        assert!(
            capped_args
                .windows(2)
                .any(|pair| pair == ["--max-tool-calls", "0"]),
            "an explicit zero tool cap must propagate losslessly: {capped_args:?}"
        );
        let fix_deadline = child_turn_deadline_secs(timeout).unwrap().to_string();
        assert!(
            capped_args
                .windows(2)
                .any(|pair| pair[0] == "--turn-deadline" && pair[1] == fix_deadline),
            "the Agent must settle before the fix's outer kill: {capped_args:?}"
        );

        capped_launcher.set_model_step_limit(None);
        capped_launcher.set_model_tool_call_limit(None);
        let cleared_args = loop_child_command(&capped_launcher, None)
            .as_std()
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        assert!(
            !cleared_args.iter().any(|arg| arg == "--max-steps"),
            "turning the runtime cap off must uncap later children: {cleared_args:?}"
        );
        assert!(
            !cleared_args.iter().any(|arg| arg == "--max-tool-calls"),
            "turning the runtime tool cap off must uncap later children: {cleared_args:?}"
        );

        capped_launcher.set_model_step_limit(Some(9));
        capped_launcher.set_model_tool_call_limit(Some(13));
        let reset_args = loop_child_command(&capped_launcher, None)
            .as_std()
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        assert!(
            reset_args
                .windows(2)
                .any(|pair| pair == ["--max-steps", "9"]),
            "a runtime opt-in cap must govern later children: {reset_args:?}"
        );
        assert!(
            reset_args
                .windows(2)
                .any(|pair| pair == ["--max-tool-calls", "13"]),
            "a runtime opt-in tool cap must govern later children: {reset_args:?}"
        );
    }

    #[test]
    fn loop_turn_timeout_is_explicit_and_child_deadline_precedes_it() {
        assert_eq!(loop_turn_timeout_from_value(None), None);
        assert_eq!(loop_turn_timeout_from_value(Some("0")), None);
        assert_eq!(loop_turn_timeout_from_value(Some("invalid")), None);
        assert_eq!(
            loop_turn_timeout_from_value(Some("600")),
            Some(Duration::from_secs(600))
        );
        assert_eq!(
            child_turn_deadline_secs(Some(Duration::from_secs(300))),
            Some(240)
        );
        assert_eq!(
            child_turn_deadline_secs(Some(Duration::from_secs(30))),
            Some(1)
        );
        assert_eq!(child_turn_deadline_secs(None), None);
    }

    #[test]
    fn trigger_timeout_is_explicit_and_zero_means_unlimited() {
        assert_eq!(trigger_timeout_from_value(None), None);
        assert_eq!(trigger_timeout_from_value(Some("")), None);
        assert_eq!(trigger_timeout_from_value(Some("0")), None);
        assert_eq!(trigger_timeout_from_value(Some("invalid")), None);
        assert_eq!(
            trigger_timeout_from_value(Some("60")),
            Some(Duration::from_secs(60))
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn trigger_without_timeout_can_run_past_a_short_deadline() {
        let result = tokio::time::timeout(
            Duration::from_secs(3),
            run_trigger_with_timeout(
                "sleep 1; printf 'finished\\n'",
                1,
                "long trigger",
                "change",
                CancellationToken::new(),
                None,
            ),
        )
        .await
        .expect("unlimited trigger should complete normally");
        assert_eq!(result, "ok: finished");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn explicit_trigger_timeout_is_enforced() {
        let started = std::time::Instant::now();
        let result = run_trigger_with_timeout(
            "sleep 30",
            1,
            "timed trigger",
            "change",
            CancellationToken::new(),
            Some(Duration::from_secs(1)),
        )
        .await;
        assert_eq!(result, "timed out after 1s");
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "explicit timeout did not settle promptly"
        );
    }

    #[cfg(unix)]
    fn process_is_live(pid: i32) -> bool {
        // SAFETY: signal 0 performs a liveness check without delivering a
        // signal. The pid came from the test's own child process.
        unsafe { libc::kill(pid, 0) == 0 }
    }

    #[cfg(unix)]
    async fn assert_process_exits(pid: i32) {
        for _ in 0..100 {
            if !process_is_live(pid) {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("trigger descendant {pid} is still live");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn cancelled_trigger_kills_its_descendant_group() {
        let dir = test_dir("trigger-cancel");
        let pid_file = dir.join("descendant.pid");
        let command = format!(
            "sleep 30 & child=$!; printf '%s\\n' \"$child\" > {}; wait \"$child\"",
            pid_file.display()
        );
        let cancellation = CancellationToken::new();
        let task_cancellation = cancellation.clone();
        let task = tokio::spawn(async move {
            run_trigger_with_timeout(
                &command,
                1,
                "cancel trigger",
                "change",
                task_cancellation,
                None,
            )
            .await
        });
        for _ in 0..100 {
            if pid_file.exists() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        let pid: i32 = std::fs::read_to_string(&pid_file)
            .expect("trigger recorded descendant pid")
            .trim()
            .parse()
            .unwrap();
        assert!(process_is_live(pid), "descendant started");
        cancellation.cancel();
        let result = tokio::time::timeout(Duration::from_secs(3), task)
            .await
            .expect("cancelled trigger settled")
            .expect("trigger task did not panic");
        assert_eq!(result, "cancelled");
        assert_process_exits(pid).await;
        let _ = std::fs::remove_dir_all(dir);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn trigger_drains_noisy_output_and_cleans_leaked_descendant() {
        let dir = test_dir("trigger-noisy");
        let pid_file = dir.join("descendant.pid");
        let command = format!(
            "printf 'headline\\n'; yes x | head -c 1048576; sleep 30 & child=$!; \
             printf '%s\\n' \"$child\" > {}",
            pid_file.display()
        );
        let result = tokio::time::timeout(
            Duration::from_secs(5),
            run_trigger_with_timeout(
                &command,
                1,
                "noisy trigger",
                "change",
                CancellationToken::new(),
                None,
            ),
        )
        .await
        .expect("large output must be continuously drained");
        assert_eq!(result, "ok: headline");
        assert!(
            result.len() <= 104,
            "trigger evidence escaped its display cap: {} bytes",
            result.len()
        );
        let pid: i32 = std::fs::read_to_string(&pid_file)
            .expect("trigger recorded leaked descendant pid")
            .trim()
            .parse()
            .unwrap();
        assert_process_exits(pid).await;
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn loop_fix_preserves_only_an_explicit_verification_repair_cap() {
        let mut launcher = command_launcher(0, None);
        launcher.verify = Some("cargo test".into());
        launcher.max_verify = hi_agent::UNLIMITED_REPAIR_CYCLES;
        let mut unlimited = tokio::process::Command::new("hi");
        append_fix_verification_args(&mut unlimited, &launcher);
        let unlimited = unlimited
            .as_std()
            .get_args()
            .map(|argument| argument.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        assert!(
            unlimited
                .windows(2)
                .any(|pair| pair == ["--verify", "cargo test"])
        );
        assert!(
            !unlimited
                .iter()
                .any(|argument| argument == "--max-verify-repairs")
        );

        launcher.max_verify = 0;
        let mut capped = tokio::process::Command::new("hi");
        append_fix_verification_args(&mut capped, &launcher);
        let capped = capped
            .as_std()
            .get_args()
            .map(|argument| argument.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        assert!(
            capped
                .windows(2)
                .any(|pair| pair == ["--max-verify-repairs", "0"])
        );
    }

    #[test]
    fn save_is_atomic_and_load_preserves_corrupt() {
        let dir = std::env::temp_dir().join(format!("hi-loops-persist-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("loops.json");

        // save → load round-trips and leaves no temp file behind.
        let state = LoopsFile {
            loops: vec![spec()],
            next_id: 5,
        };
        save(Some(&path), &state);
        assert!(!dir.join("loops.json.tmp").exists(), "temp file cleaned up");
        let loaded = load(Some(&path));
        assert_eq!(loaded.loops.len(), 1);
        assert_eq!(loaded.next_id, 5);

        // A corrupt/truncated file is preserved aside, not silently emptied.
        std::fs::write(&path, "{ this is not json").unwrap();
        let recovered = load(Some(&path));
        assert!(recovered.loops.is_empty(), "corrupt file loads as empty");
        assert!(
            dir.join("loops.json.corrupt").exists(),
            "corrupt file preserved for recovery"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn quiet_marker_detection() {
        assert!(is_quiet("NOTHING NEW"));
        assert!(is_quiet("  nothing new.  "));
        assert!(is_quiet("Checked the logs again — NOTHING NEW"));
        assert!(is_quiet("Status: NOTHING NEW"));
        // Marker on its own final line is quiet.
        assert!(is_quiet("summary of the check:\nNOTHING NEW"));
        assert!(!is_quiet("CI is now red: 3 failures in parser tests"));
        // A *loud* summary that merely ends with the words "nothing new"
        // mid-sentence (not set off by a separator) must NOT be suppressed.
        assert!(!is_quiet("the banner now reads NOTHING NEW"));
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
    fn decide_fix_never_merges_unverified() {
        // The one rule that matters: Merge requires in-repo + completed +
        // changes + a verify command + a passing verify. Anything missing → not
        // a merge.
        assert_eq!(
            decide_fix(true, true, 2, true, true),
            FixDecision::Merge,
            "all conditions met → merge"
        );
        assert_eq!(
            decide_fix(false, true, 2, true, true),
            FixDecision::NotGitRepo
        );
        assert_eq!(
            decide_fix(true, true, 0, true, true),
            FixDecision::NoChanges
        );
        // Every unsafe combination must NOT be a merge.
        for &completed in &[true, false] {
            for &has_verify in &[true, false] {
                for &verified in &[true, false] {
                    // Skip the one all-true, changes-present, completed case.
                    if completed && has_verify && verified {
                        continue;
                    }
                    assert_ne!(
                        decide_fix(true, completed, 3, has_verify, verified),
                        FixDecision::Merge,
                        "completed={completed} has_verify={has_verify} verified={verified} must not merge"
                    );
                }
            }
        }
        // A change with no verify command is rejected, not merged.
        assert!(matches!(
            decide_fix(true, true, 1, false, false),
            FixDecision::Reject(_)
        ));
    }

    #[test]
    fn decoration_excludes_usage_footer_and_glyphs() {
        // The reply text is kept…
        assert!(!is_decoration_line(
            "tests fail: add(2,3) returned -1, expected 5"
        ));
        // …the trailing usage footer and tool-glyph lines are not.
        assert!(is_decoration_line("[↑3.9k ↓133 · ctx 0% (1.4k/1.0M)]"));
        assert!(is_decoration_line("⏺ ran python3 test_calc.py"));
        assert!(is_decoration_line("✓ done"));
        // A bracketed sentence that isn't the usage footer stays reply text.
        assert!(!is_decoration_line("[note] the parser is fine"));
    }

    #[test]
    fn report_tokens_prefers_schema_v2_and_reads_legacy_fallback() {
        let dir = std::env::temp_dir().join(format!(
            "hi-loop-report-tokens-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let report = dir.join("report.json");

        std::fs::write(
            &report,
            r#"{"schema_version":2,"usage":{"session":{"total_tokens":41}},"total_tokens":7}"#,
        )
        .unwrap();
        assert_eq!(read_report_tokens(&report), 41);

        std::fs::write(&report, r#"{"total_tokens":7}"#).unwrap();
        assert_eq!(read_report_tokens(&report), 7);

        std::fs::write(&report, "not json").unwrap();
        assert_eq!(read_report_tokens(&report), 0);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn fix_prompt_is_an_implementation_task() {
        let mut s = spec();
        s.prompt = "watch prod p99 latency".into();
        let p = fix_prompt(&s, "p99 jumped to 4200ms").to_lowercase();
        assert!(
            p.contains("p99 jumped to 4200ms"),
            "carries the change\n{p}"
        );
        // Must read as an implementation task (edits), not a review — otherwise
        // hi's read-only preflight makes the fixer inspect-but-never-edit.
        assert!(p.contains("implementation task"), "{p}");
        assert!(p.contains("edit files"), "{p}");
        // Must NOT contain a no-edit clause that trips the read-only guard.
        for bad in [
            "make no changes",
            "do not edit",
            "without modifying",
            "no changes",
        ] {
            assert!(
                !p.contains(bad),
                "must not contain the no-edit phrase {bad:?}\n{p}"
            );
        }
    }

    #[test]
    fn schedule_active_windows() {
        let day = Schedule {
            start_hour: 9,
            end_hour: 17,
            weekdays_only: false,
        };
        assert!(day.active(9, 3), "start inclusive");
        assert!(day.active(16, 3));
        assert!(!day.active(17, 3), "end exclusive");
        assert!(!day.active(8, 3));
        // Weekdays-only excludes Sat(6)/Sun(7).
        let wk = Schedule {
            start_hour: 9,
            end_hour: 17,
            weekdays_only: true,
        };
        assert!(wk.active(10, 5), "Friday ok");
        assert!(!wk.active(10, 6), "Saturday excluded");
        // A window that wraps past midnight (22–6).
        let night = Schedule {
            start_hour: 22,
            end_hour: 6,
            weekdays_only: false,
        };
        assert!(night.active(23, 3));
        assert!(night.active(2, 3));
        assert!(!night.active(12, 3));
        assert_eq!(day.label(), "09-17");
        assert_eq!(wk.label(), "09-17 weekdays");
    }

    #[test]
    fn next_fire_respects_window_recheck() {
        // Inside the window (or no window): a full interval away.
        assert_eq!(next_fire_ms(1_000, 3600, true), 1_000 + 3600 * 1000);
        // A day-interval loop OUTSIDE its window re-checks within the cap, so it
        // re-enters the window instead of stranding a whole day away.
        assert_eq!(
            next_fire_ms(1_000, 86_400, false),
            1_000 + WINDOW_RECHECK_SECS * 1000
        );
        // A short-interval loop keeps its own (shorter) cadence either way.
        assert_eq!(next_fire_ms(1_000, 300, false), 1_000 + 300 * 1000);
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
        assert_eq!(loaded.loops[0].expires_ms, None);
        assert_eq!(loaded.next_id, 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn expired_legacy_loop_is_migrated_before_pruning_and_stays_unlimited() {
        let unlimited = serde_json::to_string(&spec()).unwrap();
        assert!(
            !unlimited.contains("expires_ms"),
            "new unlimited loops should not persist a synthetic deadline: {unlimited}"
        );

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("loops.json");
        let mut value = serde_json::to_value(LoopsFile {
            loops: vec![spec()],
            next_id: 2,
        })
        .unwrap();
        value["loops"][0]["expires_ms"] = serde_json::json!(1_u64);
        std::fs::write(&path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();

        let migrated = load(Some(&path));
        assert_eq!(migrated.loops.len(), 1, "expired loop must survive load");
        assert_eq!(migrated.loops[0].expires_ms, None);
        assert!(!loop_expired(&migrated.loops[0], u64::MAX));

        let persisted: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert!(
            persisted["loops"][0].get("expires_ms").is_none(),
            "migration must atomically persist the unlimited lifetime: {persisted}"
        );
        let reloaded = load(Some(&path));
        assert_eq!(reloaded.loops.len(), 1);
        assert_eq!(reloaded.loops[0].expires_ms, None);
        assert!(!loop_expired(&reloaded.loops[0], u64::MAX));
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
        wait_until_for(handle, Duration::from_secs(5), pred).await;
    }

    async fn wait_until_for(
        handle: &LoopsHandle,
        budget: Duration,
        pred: impl Fn(&[LoopWatchRow]) -> bool,
    ) {
        let steps = (budget.as_millis() / 25).max(1) as usize;
        for _ in 0..steps {
            if pred(&handle.snapshot.lock().unwrap()) {
                return;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        let n = handle.snapshot.lock().unwrap().len();
        panic!(
            "condition not met within {}s; snapshot has {n} row(s)",
            budget.as_secs()
        );
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
            workspace_root: dir.clone(),
            provider: "p".into(),
            model: "m".into(),
            base_url: "u".into(),
            api_key: "k".into(),
            verify: None,
            max_verify: 0,
            max_steps: std::sync::atomic::AtomicU32::new(30),
            max_tool_calls: std::sync::atomic::AtomicU64::new(u64::MAX),
            session_path: Box::new(|| Ok(PathBuf::from("/tmp/unused.jsonl"))),
            sessions: Box::new(Vec::new),
            resume_info: Box::new(|_| None),
            loop_session_path: Box::new(move || Ok(sess.clone())),
            loops_file: None,
        };
        let handle = start(Arc::new(launcher), None, None);

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
        assert_eq!(spec.expires_ms, None, "new loops run until cancelled");
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

    /// A stub `hi` that sleeps before replying, so a firing is still in flight
    /// when the next one comes due — the case the per-loop fire guard protects.
    fn slow_stub(dir: &std::path::Path, secs: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let path = dir.join("slow.sh");
        // Write the report *before* sleeping so a killed/raced child still leaves
        // a parseable spend artifact; sleep is only to hold `firing == true`.
        let script = format!(
            "#!/bin/sh\nprev=\nfor a in \"$@\"; do\n  \
             [ \"$prev\" = \"--report\" ] && printf '{{\"total_tokens\": 10}}' > \"$a\"\n  \
             prev=\"$a\"\ndone\nsleep {secs}\nprintf 'slow reply\\n'\n"
        );
        std::fs::write(&path, script).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    #[cfg(unix)]
    fn descendant_stub(dir: &std::path::Path, name: &str, pid_file: &std::path::Path) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let path = dir.join(name);
        let script = format!(
            "#!/bin/sh\nsleep 30 & child=$!\nprintf '%s\\n' \"$child\" > {}\nwait \"$child\"\n",
            pid_file.display()
        );
        std::fs::write(&path, script).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    /// Unique temp dir per test invocation (pid alone collides under parallel cargo).
    fn test_dir(label: &str) -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "hi-watch-{label}-{}-{}-{}",
            std::process::id(),
            n,
            now_ms()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn acknowledged_manager_shutdown_reaps_an_in_flight_firing_tree() {
        let dir = test_dir("manager-shutdown");
        let descendant_pid = dir.join("firing-descendant.pid");
        let launcher = fix_launcher(
            &dir,
            descendant_stub(&dir, "firing-wait.sh", &descendant_pid),
            None,
        );
        let handle = start(Arc::new(launcher), None, None);
        let (reply, created) = oneshot::channel();
        handle
            .ctl
            .send(LoopCtl::Create {
                secs: 3600,
                prompt: "wait until cancelled".to_string(),
                reply,
            })
            .unwrap();
        created.await.unwrap().unwrap();
        for _ in 0..250 {
            if descendant_pid.exists() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        let pid: i32 = std::fs::read_to_string(&descendant_pid)
            .expect("loop firing recorded its descendant pid")
            .trim()
            .parse()
            .unwrap();

        let stale_control = handle.ctl.clone();
        tokio::time::timeout(Duration::from_secs(5), handle.shutdown())
            .await
            .expect("manager shutdown settled")
            .expect("manager acknowledged shutdown");
        assert_process_exits(pid).await;
        let (reply, _) = oneshot::channel();
        assert!(
            stale_control.send(LoopCtl::List { reply }).is_err(),
            "an acknowledged shutdown must close every copied control sender"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn shutdown_settlement_keeps_success_and_rearms_cancelled_firings() {
        let mut succeeded = spec();
        succeeded.id = 1;
        succeeded.firings = 1;
        succeeded.next_ms = now_ms().saturating_add(60_000);
        let mut cancelled = spec();
        cancelled.id = 2;
        cancelled.firings = 1;
        cancelled.next_ms = now_ms().saturating_add(60_000);
        let mut state = LoopsFile {
            loops: vec![succeeded, cancelled],
            next_id: 2,
        };
        let mut runtime: HashMap<u64, LoopRuntime> = HashMap::new();
        runtime.entry(1).or_default().firing = true;
        runtime.entry(2).or_default().firing = true;
        let (done_tx, mut done_rx) = mpsc::unbounded_channel();
        done_tx
            .send((
                1,
                Ok(FiringOutcome {
                    summary: "baseline established".to_string(),
                    total_tokens: 42,
                }),
            ))
            .unwrap();
        done_tx.send((2, Err("cancelled".to_string()))).unwrap();

        settle_firings_for_shutdown(&mut state, &mut runtime, &mut done_rx);

        assert_eq!(state.loops[0].firings, 1);
        assert_eq!(state.loops[0].spent_tokens, 42);
        assert_eq!(state.loops[1].firings, 0);
        assert!(state.loops[1].next_ms <= now_ms().saturating_add(3_000));
        assert!(!runtime.values().any(|runtime| runtime.firing));
    }

    /// A firing that outlives its interval (or a `FireNow` mid-flight) must NOT
    /// spawn a second child on the *same* session. Without the per-loop guard,
    /// `FireNow` while a firing is in flight double-fires (firings jumps to 2 with
    /// two children racing one session/report); with it, the second attempt is
    /// deferred until the first completes.
    #[tokio::test]
    async fn manager_does_not_double_fire_a_loop_in_flight() {
        let dir = test_dir("nodouble");
        let sess = dir.join("loop.jsonl");
        // Long enough that FireNow lands mid-flight even on a loaded CI host.
        let exe = slow_stub(&dir, "2.5");
        let launcher = FleetLauncher {
            exe,
            workspace_root: dir.clone(),
            provider: "p".into(),
            model: "m".into(),
            base_url: "u".into(),
            api_key: "k".into(),
            verify: None,
            max_verify: 0,
            max_steps: std::sync::atomic::AtomicU32::new(30),
            max_tool_calls: std::sync::atomic::AtomicU64::new(u64::MAX),
            session_path: Box::new(|| Ok(PathBuf::from("/tmp/unused.jsonl"))),
            sessions: Box::new(Vec::new),
            resume_info: Box::new(|_| None),
            loop_session_path: Box::new(move || Ok(sess.clone())),
            loops_file: None,
        };
        let handle = start(Arc::new(launcher), None, None);

        let (tx, rx) = oneshot::channel();
        handle
            .ctl
            .send(LoopCtl::Create {
                secs: 3600,
                prompt: "slow watch".into(),
                reply: tx,
            })
            .unwrap();
        let id = rx.await.unwrap().unwrap().id;

        // Wait until the (slow) first firing is in flight. Accept either the live
        // `firing` flag or firings already advanced (spawn bumped the counter) so a
        // brief publish gap can't strand the wait on a never-seen true.
        wait_until(&handle, |rows| {
            rows.iter()
                .find(|r| r.id == id)
                .is_some_and(|r| r.firing || r.firings >= 1)
        })
        .await;

        // If the first child already finished (very fast host), re-arm a slow fire
        // so the in-flight guard still has something to protect.
        let done = {
            let rows = handle.snapshot.lock().unwrap();
            rows.iter()
                .find(|r| r.id == id)
                .is_some_and(|r| !r.firing && r.firings >= 1)
        };
        if done {
            let (tx, rx) = oneshot::channel();
            handle.ctl.send(LoopCtl::FireNow { id, reply: tx }).unwrap();
            assert!(rx.await.unwrap(), "re-arm FireNow accepted");
            wait_until(&handle, |rows| {
                rows.iter().find(|r| r.id == id).is_some_and(|r| r.firing)
            })
            .await;
        }

        // Force a second fire attempt while a child is still sleeping.
        let firings_before = handle
            .snapshot
            .lock()
            .unwrap()
            .iter()
            .find(|r| r.id == id)
            .map(|r| r.firings)
            .unwrap_or(0);
        let (tx, rx) = oneshot::channel();
        handle.ctl.send(LoopCtl::FireNow { id, reply: tx }).unwrap();
        assert!(rx.await.unwrap(), "FireNow accepted");

        // The guard defers it: for as long as the *original* firing is still the
        // one in flight (`firings` still at the pre-FireNow count), we must not
        // observe a concurrent bump. Without the guard, FireNow would spawn
        // immediately and `firings` would jump while the first child still runs.
        let mut saw_deferred = false;
        for _ in 0..20 {
            tokio::time::sleep(Duration::from_millis(40)).await;
            let rows = handle.snapshot.lock().unwrap();
            let r = rows.iter().find(|r| r.id == id).unwrap();
            if r.firing && r.firings == firings_before {
                saw_deferred = true;
            }
            // A jump of 2+ without an intervening idle would mean two children.
            assert!(
                r.firings <= firings_before + 1,
                "firings jumped too far (concurrent double-fire?): {}",
                r.firings
            );
        }
        assert!(
            saw_deferred
                || handle
                    .snapshot
                    .lock()
                    .unwrap()
                    .iter()
                    .find(|r| r.id == id)
                    .is_some_and(|r| r.firings > firings_before),
            "expected either a deferred in-flight window or a completed follow-up fire"
        );

        // Once the in-flight firing completes, the deferred FireNow fires exactly
        // once more — proving it was queued, not dropped. Budget covers the rest of
        // the slow child plus a second full slow firing.
        wait_until_for(&handle, Duration::from_secs(12), |rows| {
            rows.iter()
                .find(|r| r.id == id)
                .is_some_and(|r| !r.firing && r.firings > firings_before)
        })
        .await;

        let (tx, rx) = oneshot::channel();
        handle.ctl.send(LoopCtl::Cancel { id, reply: tx }).unwrap();
        let _ = rx.await;
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn cancelling_loop_stops_firing_descendants_without_resurrection() {
        let dir = test_dir("cancel-firing");
        let pid_file = dir.join("firing-descendant.pid");
        let exe = descendant_stub(&dir, "firing.sh", &pid_file);
        let sess = dir.join("loop.jsonl");
        let launcher = FleetLauncher {
            exe,
            workspace_root: dir.clone(),
            provider: "p".into(),
            model: "m".into(),
            base_url: "u".into(),
            api_key: "k".into(),
            verify: None,
            max_verify: 0,
            max_steps: std::sync::atomic::AtomicU32::new(0),
            max_tool_calls: std::sync::atomic::AtomicU64::new(u64::MAX),
            session_path: Box::new(|| Ok(PathBuf::from("/tmp/unused.jsonl"))),
            sessions: Box::new(Vec::new),
            resume_info: Box::new(|_| None),
            loop_session_path: Box::new(move || Ok(sess.clone())),
            loops_file: None,
        };
        let handle = start(Arc::new(launcher), None, None);
        let (tx, rx) = oneshot::channel();
        handle
            .ctl
            .send(LoopCtl::Create {
                secs: 3600,
                prompt: "long firing".into(),
                reply: tx,
            })
            .unwrap();
        let id = rx.await.unwrap().unwrap().id;
        // Child startup can exceed two seconds under the parallel suite. Wait
        // for a complete PID record before exercising cancellation.
        wait_until(&handle, |_| {
            std::fs::read_to_string(&pid_file).is_ok_and(|pid| pid.trim().parse::<i32>().is_ok())
        })
        .await;
        let pid: i32 = std::fs::read_to_string(&pid_file)
            .expect("firing recorded descendant pid")
            .trim()
            .parse()
            .unwrap();
        let (tx, rx) = oneshot::channel();
        handle.ctl.send(LoopCtl::Cancel { id, reply: tx }).unwrap();
        assert!(rx.await.unwrap());
        assert_process_exits(pid).await;
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(
            handle.snapshot.lock().unwrap().is_empty(),
            "late cancellation result must not recreate a removed loop"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn dropping_manager_stops_firing_descendants() {
        let dir = test_dir("drop-firing");
        let pid_file = dir.join("firing-descendant.pid");
        let exe = descendant_stub(&dir, "firing.sh", &pid_file);
        let sess = dir.join("loop.jsonl");
        let launcher = FleetLauncher {
            exe,
            workspace_root: dir.clone(),
            provider: "p".into(),
            model: "m".into(),
            base_url: "u".into(),
            api_key: "k".into(),
            verify: None,
            max_verify: 0,
            max_steps: std::sync::atomic::AtomicU32::new(0),
            max_tool_calls: std::sync::atomic::AtomicU64::new(u64::MAX),
            session_path: Box::new(|| Ok(PathBuf::from("/tmp/unused.jsonl"))),
            sessions: Box::new(Vec::new),
            resume_info: Box::new(|_| None),
            loop_session_path: Box::new(move || Ok(sess.clone())),
            loops_file: None,
        };
        let handle = start(Arc::new(launcher), None, None);
        let (tx, rx) = oneshot::channel();
        handle
            .ctl
            .send(LoopCtl::Create {
                secs: 3600,
                prompt: "long firing".into(),
                reply: tx,
            })
            .unwrap();
        rx.await.unwrap().unwrap();
        wait_until(&handle, |_| {
            std::fs::read_to_string(&pid_file).is_ok_and(|pid| pid.trim().parse::<i32>().is_ok())
        })
        .await;
        let pid: i32 = std::fs::read_to_string(&pid_file)
            .expect("firing recorded descendant pid")
            .trim()
            .parse()
            .unwrap();
        drop(handle);
        assert_process_exits(pid).await;
        let _ = std::fs::remove_dir_all(dir);
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
        let dir = test_dir("cost");
        let sess = dir.join("loop.jsonl");
        let exe = report_stub(&dir, 1_000_000);
        let launcher = FleetLauncher {
            exe,
            workspace_root: dir.clone(),
            provider: "p".into(),
            model: "m".into(),
            base_url: "u".into(),
            api_key: "k".into(),
            verify: None,
            max_verify: 0,
            max_steps: std::sync::atomic::AtomicU32::new(30),
            max_tool_calls: std::sync::atomic::AtomicU64::new(u64::MAX),
            session_path: Box::new(|| Ok(PathBuf::from("/tmp/unused.jsonl"))),
            sessions: Box::new(Vec::new),
            resume_info: Box::new(|_| None),
            loop_session_path: Box::new(move || Ok(sess.clone())),
            loops_file: None,
        };
        let handle = start(Arc::new(launcher), None, None);

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
        // Allow a retry window: a failed first spawn re-arms ~2s later.
        wait_until_for(&handle, Duration::from_secs(10), |rows| {
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

    /// A scheduled loop doesn't fire outside its window and does inside it.
    /// Windows are computed from the real current hour, so the check is
    /// deterministic regardless of when the test runs.
    #[tokio::test]
    async fn manager_respects_the_fire_window() {
        let dir = std::env::temp_dir().join(format!("hi-watch-win-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let sess = dir.join("loop.jsonl");
        let launcher = FleetLauncher {
            exe: PathBuf::from("/bin/echo"),
            workspace_root: dir.clone(),
            provider: "p".into(),
            model: "m".into(),
            base_url: "u".into(),
            api_key: "k".into(),
            verify: None,
            max_verify: 0,
            max_steps: std::sync::atomic::AtomicU32::new(30),
            max_tool_calls: std::sync::atomic::AtomicU64::new(u64::MAX),
            session_path: Box::new(|| Ok(PathBuf::from("/tmp/unused.jsonl"))),
            sessions: Box::new(Vec::new),
            resume_info: Box::new(|_| None),
            loop_session_path: Box::new(move || Ok(sess.clone())),
            loops_file: None,
        };
        let handle = start(Arc::new(launcher), None, None);

        let (tx, rx) = oneshot::channel();
        handle
            .ctl
            .send(LoopCtl::Create {
                secs: 60,
                prompt: "watch".into(),
                reply: tx,
            })
            .unwrap();
        let id = rx.await.unwrap().unwrap().id;
        wait_until(&handle, |rows| {
            rows.iter()
                .find(|r| r.id == id)
                .is_some_and(|r| !r.history.is_empty())
        })
        .await;
        let base_firings = handle
            .snapshot
            .lock()
            .unwrap()
            .iter()
            .find(|r| r.id == id)
            .unwrap()
            .firings;

        // Current local hour → build a window that excludes "now".
        let hour: u8 = String::from_utf8_lossy(
            &std::process::Command::new("date")
                .arg("+%H")
                .output()
                .unwrap()
                .stdout,
        )
        .trim()
        .parse()
        .unwrap();
        let exclude = ((hour + 2) % 24, (hour + 3) % 24, false);
        let (tx, rx) = oneshot::channel();
        handle
            .ctl
            .send(LoopCtl::Window {
                id,
                window: Some(exclude),
                reply: tx,
            })
            .unwrap();
        assert!(rx.await.unwrap());
        let (tx, rx) = oneshot::channel();
        handle.ctl.send(LoopCtl::FireNow { id, reply: tx }).unwrap();
        assert!(rx.await.unwrap());
        tokio::time::sleep(Duration::from_millis(400)).await;
        assert_eq!(
            handle
                .snapshot
                .lock()
                .unwrap()
                .iter()
                .find(|r| r.id == id)
                .unwrap()
                .firings,
            base_firings,
            "a loop outside its window must not fire"
        );

        // A window that includes "now" → firing resumes.
        let include = (hour, (hour + 1) % 24, false);
        let (tx, rx) = oneshot::channel();
        handle
            .ctl
            .send(LoopCtl::Window {
                id,
                window: Some(include),
                reply: tx,
            })
            .unwrap();
        assert!(rx.await.unwrap());
        let (tx, rx) = oneshot::channel();
        handle.ctl.send(LoopCtl::FireNow { id, reply: tx }).unwrap();
        assert!(rx.await.unwrap());
        wait_until(&handle, |rows| {
            rows.iter()
                .find(|r| r.id == id)
                .is_some_and(|r| r.firings > base_firings)
        })
        .await;

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A loud firing runs the loop's on-change trigger, with the firing summary
    /// in `$HI_LOOP_SUMMARY`. (`/bin/echo` firings are loud — they never reply
    /// the quiet marker.)
    #[tokio::test]
    async fn manager_runs_trigger_on_loud_change() {
        let dir = std::env::temp_dir().join(format!("hi-watch-trig-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let sess = dir.join("loop.jsonl");
        let sentinel = dir.join("fired.txt");
        let launcher = FleetLauncher {
            exe: PathBuf::from("/bin/echo"),
            workspace_root: dir.clone(),
            provider: "p".into(),
            model: "m".into(),
            base_url: "u".into(),
            api_key: "k".into(),
            verify: None,
            max_verify: 0,
            max_steps: std::sync::atomic::AtomicU32::new(30),
            max_tool_calls: std::sync::atomic::AtomicU64::new(u64::MAX),
            session_path: Box::new(|| Ok(PathBuf::from("/tmp/unused.jsonl"))),
            sessions: Box::new(Vec::new),
            resume_info: Box::new(|_| None),
            loop_session_path: Box::new(move || Ok(sess.clone())),
            loops_file: None,
        };
        let handle = start(Arc::new(launcher), None, None);

        let (tx, rx) = oneshot::channel();
        handle
            .ctl
            .send(LoopCtl::Create {
                secs: 3600,
                prompt: "watch the thing".into(),
                reply: tx,
            })
            .unwrap();
        let id = rx.await.unwrap().unwrap().id;
        wait_until(&handle, |rows| {
            rows.iter()
                .find(|r| r.id == id)
                .is_some_and(|r| !r.history.is_empty())
        })
        .await;

        // Attach a trigger that records $HI_LOOP_SUMMARY, then fire.
        let cmd = format!(
            "printf '%s' \"$HI_LOOP_SUMMARY\" > '{}'",
            sentinel.display()
        );
        let (tx, rx) = oneshot::channel();
        handle
            .ctl
            .send(LoopCtl::Trigger {
                id,
                cmd: Some(cmd),
                reply: tx,
            })
            .unwrap();
        assert!(rx.await.unwrap());
        let (tx, rx) = oneshot::channel();
        handle.ctl.send(LoopCtl::FireNow { id, reply: tx }).unwrap();
        assert!(rx.await.unwrap());

        // The trigger runs and reports an ok outcome into the runtime.
        wait_until(&handle, |rows| {
            rows.iter()
                .find(|r| r.id == id)
                .and_then(|r| r.last_trigger.as_deref())
                .is_some_and(|t| t.starts_with("ok"))
        })
        .await;
        // …and it actually executed, receiving the summary via the environment.
        let written = std::fs::read_to_string(&sentinel).unwrap_or_default();
        assert!(!written.trim().is_empty(), "trigger wrote the summary");

        // Clearing the trigger removes it from the snapshot.
        let (tx, rx) = oneshot::channel();
        handle
            .ctl
            .send(LoopCtl::Trigger {
                id,
                cmd: None,
                reply: tx,
            })
            .unwrap();
        assert!(rx.await.unwrap());
        wait_until(&handle, |rows| {
            rows.iter()
                .find(|r| r.id == id)
                .is_some_and(|r| r.trigger.is_none())
        })
        .await;

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The pipeline works *together*, not just each piece in isolation: one loud
    /// firing on a loop with BOTH a trigger and auto-fix dispatches both, and
    /// both result channels (trigger + fix) resolve into the snapshot without
    /// starving each other or the manager. Runs in a throwaway git repo.
    #[tokio::test]
    async fn manager_pipeline_trigger_and_autofix_together() {
        let dir = std::env::temp_dir().join(format!("hi-watch-pipe-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        init_git_repo(&dir);
        let sess = dir.join("loop.jsonl");
        let sentinel = dir.join("trig.txt");
        let launcher = FleetLauncher {
            exe: PathBuf::from("/bin/echo"),
            workspace_root: dir.clone(),
            provider: "p".into(),
            model: "m".into(),
            base_url: "u".into(),
            api_key: "k".into(),
            verify: None,
            max_verify: 0,
            max_steps: std::sync::atomic::AtomicU32::new(30),
            max_tool_calls: std::sync::atomic::AtomicU64::new(u64::MAX),
            session_path: Box::new(|| Ok(PathBuf::from("/tmp/unused.jsonl"))),
            sessions: Box::new(Vec::new),
            resume_info: Box::new(|_| None),
            loop_session_path: Box::new(move || Ok(sess.clone())),
            loops_file: None,
        };
        let handle = start(Arc::new(launcher), None, None);

        let (tx, rx) = oneshot::channel();
        handle
            .ctl
            .send(LoopCtl::Create {
                secs: 3600,
                prompt: "watch".into(),
                reply: tx,
            })
            .unwrap();
        let id = rx.await.unwrap().unwrap().id;
        wait_until(&handle, |rows| {
            rows.iter()
                .find(|r| r.id == id)
                .is_some_and(|r| !r.history.is_empty())
        })
        .await;

        // Attach BOTH a trigger and auto-fix, then fire once.
        let cmd = format!("touch '{}'", sentinel.display());
        for ctl in [
            LoopCtl::Trigger {
                id,
                cmd: Some(cmd),
                reply: oneshot::channel().0,
            },
            LoopCtl::Fix {
                id,
                on: true,
                pr: false,
                reply: oneshot::channel().0,
            },
        ] {
            handle.ctl.send(ctl).unwrap();
        }
        // Small settle so both ctl messages land before the firing.
        tokio::time::sleep(Duration::from_millis(50)).await;
        let (tx, rx) = oneshot::channel();
        handle.ctl.send(LoopCtl::FireNow { id, reply: tx }).unwrap();
        assert!(rx.await.unwrap());

        // Both the trigger AND the fix resolve from the one firing.
        wait_until(&handle, |rows| {
            rows.iter().find(|r| r.id == id).is_some_and(|r| {
                r.last_trigger
                    .as_deref()
                    .is_some_and(|t| t.starts_with("ok"))
                    && r.last_fix.is_some()
            })
        })
        .await;
        assert!(sentinel.exists(), "trigger ran");
        let last_fix = handle
            .snapshot
            .lock()
            .unwrap()
            .iter()
            .find(|r| r.id == id)
            .unwrap()
            .last_fix
            .clone()
            .unwrap();
        // echo isn't a real fixer → no changes in the clean repo → nothing merged.
        assert!(
            last_fix.contains("made no changes"),
            "fix dispatched + resolved from the same firing: {last_fix}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A loud firing is persisted to the project's activity feed (for /digest).
    #[tokio::test]
    async fn manager_records_loud_firing_to_activity() {
        let dir = std::env::temp_dir().join(format!("hi-watch-act-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let sess = dir.join("loop.jsonl");
        let loops_file = dir.join("loops.json");
        let launcher = FleetLauncher {
            exe: PathBuf::from("/bin/echo"),
            workspace_root: dir.clone(),
            provider: "p".into(),
            model: "m".into(),
            base_url: "u".into(),
            api_key: "k".into(),
            verify: None,
            max_verify: 0,
            max_steps: std::sync::atomic::AtomicU32::new(30),
            max_tool_calls: std::sync::atomic::AtomicU64::new(u64::MAX),
            session_path: Box::new(|| Ok(PathBuf::from("/tmp/unused.jsonl"))),
            sessions: Box::new(Vec::new),
            resume_info: Box::new(|_| None),
            loop_session_path: Box::new(move || Ok(sess.clone())),
            loops_file: Some(loops_file.clone()),
        };
        let handle = start(Arc::new(launcher), Some(loops_file.clone()), None);

        let (tx, rx) = oneshot::channel();
        handle
            .ctl
            .send(LoopCtl::Create {
                secs: 3600,
                prompt: "watch the thing".into(),
                reply: tx,
            })
            .unwrap();
        let id = rx.await.unwrap().unwrap().id;
        wait_until(&handle, |rows| {
            rows.iter()
                .find(|r| r.id == id)
                .is_some_and(|r| !r.history.is_empty())
        })
        .await;

        // The loud firing landed in activity.jsonl next to loops.json.
        let entries = crate::activity::load(&crate::activity::activity_path(&loops_file));
        assert!(
            entries.iter().any(|e| e.loop_id == id),
            "loud firing recorded to the activity feed"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    fn git(args: &[&str], cwd: &std::path::Path) {
        let out = std::process::Command::new("git")
            .args(args)
            .current_dir(cwd)
            .output()
            .unwrap();
        assert!(out.status.success(), "git {args:?}: {out:?}");
    }

    /// A stub "fixer" `hi` that writes `file` in its cwd (the worktree),
    /// simulating an agent that made a change. LLM-free.
    fn fixer_stub(_dir: &std::path::Path, name: &str, file: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let bin_dir =
            std::env::temp_dir().join(format!("hi-fixer-bin-{}-{}", std::process::id(), name));
        let _ = std::fs::remove_dir_all(&bin_dir);
        std::fs::create_dir_all(&bin_dir).unwrap();
        let path = bin_dir.join(name);
        std::fs::write(&path, format!("#!/bin/sh\nprintf 'patched' > '{file}'\n")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    fn fix_launcher(root: &std::path::Path, exe: PathBuf, verify: Option<&str>) -> FleetLauncher {
        FleetLauncher {
            exe,
            workspace_root: root.to_path_buf(),
            provider: "p".into(),
            model: "m".into(),
            base_url: "u".into(),
            api_key: "k".into(),
            verify: verify.map(str::to_string),
            max_verify: 0,
            max_steps: std::sync::atomic::AtomicU32::new(40),
            max_tool_calls: std::sync::atomic::AtomicU64::new(u64::MAX),
            session_path: Box::new(|| Ok(PathBuf::from("/tmp/unused.jsonl"))),
            sessions: Box::new(Vec::new),
            resume_info: Box::new(|_| None),
            loop_session_path: Box::new(|| Ok(PathBuf::from("/tmp/unused.jsonl"))),
            loops_file: None,
        }
    }

    /// End-to-end auto-fix over *real git*, with a stub fixer standing in for the
    /// LLM: a verified fix is merged into the working tree; a fix that fails
    /// verify is NOT merged (the safety gate, proven for real — not just in
    /// `decide_fix`).
    #[tokio::test]
    async fn run_fix_merges_verified_and_rejects_unverified() {
        let dir = std::env::temp_dir().join(format!("hi-runfix-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        git(&["init", "-q", "-b", "main"], &dir);
        git(&["config", "user.email", "t@t"], &dir);
        git(&["config", "user.name", "t"], &dir);
        std::fs::write(dir.join("README"), "hi\n").unwrap();
        git(&["add", "-A"], &dir);
        git(&["commit", "-qm", "init"], &dir);

        let mut s = spec();
        s.id = 1;

        // Passing verify → the fix merges into the real tree.
        let pass = fix_launcher(
            &dir,
            fixer_stub(&dir, "pass.sh", "fixed.txt"),
            Some("test -f fixed.txt"),
        );
        // Failing verify → the fix is rejected, never applied.
        let mut s2 = spec();
        s2.id = 2;
        let fail = fix_launcher(&dir, fixer_stub(&dir, "fail.sh", "bad.txt"), Some("false"));

        let merged = run_fix(&pass, &s, "something broke", CancellationToken::new()).await;
        let rejected = run_fix(&fail, &s2, "something else broke", CancellationToken::new()).await;

        assert!(
            merged.0.contains("merged"),
            "verified fix merged: {}",
            merged.0
        );
        assert!(
            dir.join("fixed.txt").exists(),
            "the verified fix landed in the real tree"
        );
        assert!(
            rejected.0.contains("NOT merged"),
            "unverified fix rejected: {}",
            rejected.0
        );
        assert!(
            !dir.join("bad.txt").exists(),
            "the unverified change must NOT reach the real tree"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn cancelling_fix_stops_child_descendants_and_never_merges() {
        let dir = test_dir("cancel-fix");
        init_git_repo(&dir);
        let pid_file = dir.join("fix-descendant.pid");
        let exe = descendant_stub(&dir, "fixer-wait.sh", &pid_file);
        let launcher = fix_launcher(&dir, exe, Some("true"));
        let mut loop_spec = spec();
        loop_spec.id = 91;
        let cancellation = CancellationToken::new();
        let task_cancellation = cancellation.clone();
        let task = tokio::spawn(async move {
            run_fix(&launcher, &loop_spec, "wait forever", task_cancellation).await
        });
        for _ in 0..250 {
            if pid_file.exists() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        let pid: i32 = std::fs::read_to_string(&pid_file)
            .expect("fixer recorded descendant pid")
            .trim()
            .parse()
            .unwrap();
        cancellation.cancel();
        let result = tokio::time::timeout(Duration::from_secs(5), task)
            .await
            .expect("cancelled fixer settled")
            .expect("fixer task did not panic");
        assert_eq!(result, ("cancelled".to_string(), false));
        assert_process_exits(pid).await;
        assert!(
            !dir.join("fixed.txt").exists(),
            "cancelled fix must not merge a change"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn merged_outcome_reflects_the_real_tree_verify() {
        let dir = std::env::temp_dir().join(format!("hi-merged-outcome-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // Combined tree passes verify → normal success line.
        let ok = merged_outcome(&dir, Some("true"), &["a.rs".to_string()], None).await;
        // Combined tree FAILS verify (the base drifted under the fix) → a loud
        // warning, not a false "merged" success.
        let bad = merged_outcome(&dir, Some("false"), &["a.rs".to_string()], None).await;
        // No verify command → nothing to re-check; trust the merge.
        let none = merged_outcome(&dir, None, &["a.rs".to_string()], None).await;

        assert!(ok.0.contains("merged") && !ok.0.contains('⚠'), "{}", ok.0);
        assert!(
            bad.0.contains('⚠') && bad.0.contains("fails verify"),
            "{}",
            bad.0
        );
        assert!(
            none.0.contains("merged") && !none.0.contains('⚠'),
            "{}",
            none.0
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
