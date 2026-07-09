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
/// An on-change trigger command is killed after this long.
const TRIGGER_TIMEOUT_SECS: u64 = 60;
/// An auto-fix attempt (a full write-capable agent run) is killed after this.
const FIX_TIMEOUT_SECS: u64 = 900;
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
    pub(crate) expires_ms: u64,
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
pub(crate) fn start(launcher: Arc<FleetLauncher>, loops_file: Option<PathBuf>) -> LoopsHandle {
    let (ctl_tx, ctl_rx) = mpsc::unbounded_channel();
    let pending: Arc<Mutex<Vec<LoopLine>>> = Arc::new(Mutex::new(Vec::new()));
    let snapshot: Arc<Mutex<Vec<LoopWatchRow>>> = Arc::new(Mutex::new(Vec::new()));
    let handle = LoopsHandle {
        ctl: ctl_tx,
        pending: pending.clone(),
        snapshot: snapshot.clone(),
    };
    let activity = loops_file
        .as_ref()
        .map(|p| crate::activity::activity_path(p));
    tokio::spawn(manager(
        launcher, loops_file, activity, ctl_rx, pending, snapshot,
    ));
    handle
}

/// Append one loud event to the project's activity feed (best-effort).
fn record(activity: Option<&std::path::Path>, loop_id: u64, source: &str, text: &str) {
    if let Some(path) = activity {
        crate::activity::append(
            path,
            &crate::activity::ActivityEntry {
                at_ms: now_ms(),
                loop_id,
                source: source.to_string(),
                text: text.to_string(),
            },
        );
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
    activity: Option<PathBuf>,
    mut ctl: mpsc::UnboundedReceiver<LoopCtl>,
    pending: Arc<Mutex<Vec<LoopLine>>>,
    snapshot: Arc<Mutex<Vec<LoopWatchRow>>>,
) {
    let activity = activity.as_deref();
    // Reach-you notifications for loud events (opt-in via env; no-op otherwise).
    let notify = crate::notify::NotifyConfig::from_env();
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
    // On-change trigger runs report their outcome line over this channel.
    let (trig_tx, mut trig_rx) = mpsc::unbounded_channel::<(u64, String)>();
    // Auto-fix attempts report (id, outcome-line, loud) over this channel.
    let (fix_tx, mut fix_rx) = mpsc::unbounded_channel::<(u64, String, bool)>();
    let mut in_flight: usize = 0;

    loop {
        let now = now_ms();
        // Expire + fire due loops (respecting a small concurrency cap).
        let mut fired = false;
        state.loops.retain(|l| {
            if l.expires_ms <= now {
                record(
                    activity,
                    l.id,
                    &format!("loop#{} {}", l.id, l.name()),
                    "expired after 7 days",
                );
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
                            record(activity, id, &format!("loop#{id} {name}"), &msg);
                            budget_line = Some((format!("⏸ loop#{id} ({name}) {msg}"), true));
                        }
                    }
                    save(loops_file.as_deref(), &state);
                }
                // On a genuine loud change (not quiet, not a firing error), record
                // it to the activity feed and run the loop's on-change trigger.
                let loud_change = tokens.is_some() && !quiet;
                if loud_change {
                    record(activity, id, &format!("loop#{id} {name}"), &summary);
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
                    tokio::spawn(async move {
                        let outcome = run_trigger(&cmd, id, &name, &summary).await;
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
                    tokio::spawn(async move {
                        let (line, loud) = run_fix(&launcher, &spec, &summary).await;
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
                let name = state
                    .loops
                    .iter()
                    .find(|l| l.id == id)
                    .map(LoopSpec::name)
                    .unwrap_or_else(|| format!("#{id}"));
                record(activity, id, &format!("loop#{id} {name}"), &format!("auto-fix: {outcome}"));
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

/// Read the session-cumulative `total_tokens` from a firing's `--report` JSON
/// (0 if the file is missing or malformed — cost tracking is best-effort).
fn read_report_tokens(path: &std::path::Path) -> u64 {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .and_then(|v| v.get("total_tokens").and_then(serde_json::Value::as_u64))
        .unwrap_or(0)
}

/// Run a loop's on-change trigger via `sh -c`, passing the loop id/name and the
/// firing's summary in the environment. Returns a compact outcome line — one of
/// `ok`, `ok: <stdout>`, `exit N: <stderr>`, `timed out …`, or `failed …` — so
/// the transcript and `/watch` can show whether the response actually ran.
async fn run_trigger(cmd: &str, id: u64, name: &str, summary: &str) -> String {
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
    let first_line = |bytes: &[u8]| {
        String::from_utf8_lossy(bytes)
            .lines()
            .find(|l| !l.trim().is_empty())
            .map(|l| truncate(l.trim(), 100))
            .unwrap_or_default()
    };
    match tokio::time::timeout(Duration::from_secs(TRIGGER_TIMEOUT_SECS), c.output()).await {
        Ok(Ok(out)) if out.status.success() => {
            let head = first_line(&out.stdout);
            if head.is_empty() {
                "ok".to_string()
            } else {
                format!("ok: {head}")
            }
        }
        Ok(Ok(out)) => {
            let code = out.status.code().unwrap_or(-1);
            let err = first_line(&out.stderr);
            if err.is_empty() {
                format!("exit {code}")
            } else {
                format!("exit {code}: {err}")
            }
        }
        Ok(Err(e)) => format!("failed to run: {e}"),
        Err(_) => format!("timed out after {TRIGGER_TIMEOUT_SECS}s"),
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
async fn run_fix(launcher: &FleetLauncher, spec: &LoopSpec, summary: &str) -> (String, bool) {
    use hi_tools::worktree;

    if !worktree::in_git_repo() {
        return ("skipped — not a git repository".into(), false);
    }
    let base = match hi_tools::checkpoint::create(std::path::Path::new(".")).await {
        Some(b) => b,
        None => return ("skipped — couldn't snapshot the working tree".into(), true),
    };
    let wt = worktree::worktree_path("loopfix", spec.id as u32);
    worktree::cleanup(std::slice::from_ref(&wt)); // clear any stale worktree
    if let Err(e) = worktree::add_worktree(&wt, &base) {
        return (format!("skipped — worktree setup failed: {e}"), true);
    }

    // A write-capable child agent runs the fix in the worktree, self-verifying
    // via `--verify` if the session has one.
    let mut cmd = tokio::process::Command::new(&launcher.exe);
    cmd.current_dir(&wt)
        .env("HI_API_KEY", &launcher.api_key)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
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
            "40",
        ]);
    if let Some(v) = &launcher.verify {
        cmd.arg("--verify").arg(v);
    }
    cmd.arg(fix_prompt(spec, summary));

    let completed = match cmd.spawn() {
        Ok(mut child) => {
            match tokio::time::timeout(Duration::from_secs(FIX_TIMEOUT_SECS), child.wait()).await {
                Ok(Ok(status)) => status.success(),
                _ => {
                    let _ = child.start_kill();
                    let _ = child.wait().await;
                    false
                }
            }
        }
        Err(e) => {
            worktree::cleanup(std::slice::from_ref(&wt));
            return (format!("skipped — couldn't launch the fixer: {e}"), true);
        }
    };

    let changed = worktree::changed_files(&wt, &base);
    let has_verify = launcher.verify.is_some();
    // Ground-truth re-verify of the final worktree state before any merge.
    let verified = completed
        && !changed.is_empty()
        && launcher
            .verify
            .as_deref()
            .is_some_and(|v| worktree::verify_passes(&wt, v));

    let result = match decide_fix(true, completed, changed.len(), has_verify, verified) {
        // PR mode: land the verified fix as a reviewable branch + PR.
        FixDecision::Merge if spec.fix_pr => open_fix_pr(&wt, spec, summary, &changed),
        // Merge mode: apply the verified diff into the working tree.
        FixDecision::Merge => match worktree::apply_changes(&wt, &base) {
            Ok(_) => (
                format!(
                    "fixed & merged {} file(s): {}",
                    changed.len(),
                    changed.join(", ")
                ),
                true,
            ),
            Err(e) => (format!("verified but merge failed: {e}"), true),
        },
        FixDecision::NoChanges => ("made no changes".into(), false),
        FixDecision::Reject(why) => (
            format!("{} file(s) changed but NOT merged — {why}", changed.len()),
            true,
        ),
        FixDecision::NotGitRepo => ("skipped — not a git repository".into(), false),
    };
    worktree::cleanup(std::slice::from_ref(&wt));
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
) -> (String, bool) {
    use hi_tools::worktree;
    let name = spec.name();
    let branch = format!("hi-autofix/loop{}-{}", spec.id, now_ms());
    let commit_msg = format!("hi auto-fix: {name}\n\n{}", truncate(summary, 500));
    if let Err(e) = worktree::commit_to_branch(worktree, &branch, &commit_msg) {
        return (
            format!("verified, but couldn't prepare the PR branch: {e}"),
            true,
        );
    }
    if let Err(e) = worktree::push_branch(worktree, &branch) {
        return (
            format!("fix committed to branch {branch} (couldn't push: {e}) — review it locally"),
            true,
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
    match serde_json::from_str(&text) {
        Ok(state) => state,
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
        // Rename failed (e.g. cross-device) — fall back to a direct write.
        let _ = std::fs::write(path, &json);
        let _ = std::fs::remove_file(&tmp);
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

    /// Serializes tests that mutate the process cwd (`run_fix` and any manager
    /// firing that reaches it operate on the cwd), so they don't race.
    static CWD_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Init a throwaway git repo with one commit; returns nothing (caller uses it
    /// as cwd). Kept tiny so cwd-controlled tests read cleanly.
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
            expires_ms: LOOP_TTL_SECS * 1000,
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

    /// A stub `hi` that sleeps before replying, so a firing is still in flight
    /// when the next one comes due — the case the per-loop fire guard protects.
    fn slow_stub(dir: &std::path::Path, secs: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let path = dir.join("slow.sh");
        let script = format!(
            "#!/bin/sh\nsleep {secs}\nprev=\nfor a in \"$@\"; do\n  \
             [ \"$prev\" = \"--report\" ] && printf '{{\"total_tokens\": 10}}' > \"$a\"\n  \
             prev=\"$a\"\ndone\nprintf 'slow reply\\n'\n"
        );
        std::fs::write(&path, script).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    /// A firing that outlives its interval (or a `FireNow` mid-flight) must NOT
    /// spawn a second child on the *same* session. Without the per-loop guard,
    /// `FireNow` while a firing is in flight double-fires (firings jumps to 2 with
    /// two children racing one session/report); with it, the second attempt is
    /// deferred until the first completes.
    #[tokio::test]
    async fn manager_does_not_double_fire_a_loop_in_flight() {
        let dir = std::env::temp_dir().join(format!("hi-watch-nodouble-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let sess = dir.join("loop.jsonl");
        let exe = slow_stub(&dir, "1.2");
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
                secs: 3600,
                prompt: "slow watch".into(),
                reply: tx,
            })
            .unwrap();
        let id = rx.await.unwrap().unwrap().id;

        // Wait until the (slow) first firing is in flight.
        wait_until(&handle, |rows| {
            rows.iter().find(|r| r.id == id).is_some_and(|r| r.firing)
        })
        .await;

        // Force a second fire attempt while the first child is still sleeping.
        let (tx, rx) = oneshot::channel();
        handle.ctl.send(LoopCtl::FireNow { id, reply: tx }).unwrap();
        assert!(rx.await.unwrap(), "FireNow accepted");

        // The guard defers it: while the firing is in flight, firings stays 1
        // (no concurrent second child on the same session). Without the guard the
        // FireNow spawns immediately and this sees firings == 2.
        for _ in 0..8 {
            tokio::time::sleep(Duration::from_millis(30)).await;
            let rows = handle.snapshot.lock().unwrap();
            let r = rows.iter().find(|r| r.id == id).unwrap();
            if r.firing {
                assert_eq!(r.firings, 1, "no second concurrent firing while in flight");
            }
        }

        // Once the first firing completes, the deferred FireNow fires exactly once
        // more — proving it was queued, not dropped.
        wait_until(&handle, |rows| {
            rows.iter()
                .find(|r| r.id == id)
                .is_some_and(|r| !r.firing && r.firings >= 2)
        })
        .await;

        let (tx, rx) = oneshot::channel();
        handle.ctl.send(LoopCtl::Cancel { id, reply: tx }).unwrap();
        let _ = rx.await;
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
    /// starving each other or the manager. Runs cwd-controlled in a throwaway git
    /// repo (the fix operates on the cwd), serialized via CWD_LOCK.
    #[tokio::test]
    async fn manager_pipeline_trigger_and_autofix_together() {
        let _guard = CWD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev_cwd = std::env::current_dir().unwrap();
        let dir = std::env::temp_dir().join(format!("hi-watch-pipe-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        init_git_repo(&dir);
        std::env::set_current_dir(&dir).unwrap();
        let sess = dir.join("loop.jsonl");
        let sentinel = dir.join("trig.txt");
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

        let _ = std::env::set_current_dir(prev_cwd);
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
            loops_file: Some(loops_file.clone()),
        };
        let handle = start(Arc::new(launcher), Some(loops_file.clone()));

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
    fn fixer_stub(dir: &std::path::Path, name: &str, file: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let path = dir.join(name);
        std::fs::write(&path, format!("#!/bin/sh\nprintf 'patched' > {file}\n")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    fn fix_launcher(exe: PathBuf, verify: Option<&str>) -> FleetLauncher {
        FleetLauncher {
            exe,
            provider: "p".into(),
            model: "m".into(),
            base_url: "u".into(),
            api_key: "k".into(),
            verify: verify.map(str::to_string),
            max_verify: 0,
            max_steps: 40,
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
    /// `decide_fix`). Serialized + cwd-restored since `run_fix` operates on the
    /// process cwd (like the worktree helpers it reuses).
    #[tokio::test]
    async fn run_fix_merges_verified_and_rejects_unverified() {
        let _guard = CWD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::current_dir().unwrap();

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
            fixer_stub(&dir, "pass.sh", "fixed.txt"),
            Some("test -f fixed.txt"),
        );
        // Failing verify → the fix is rejected, never applied.
        let mut s2 = spec();
        s2.id = 2;
        let fail = fix_launcher(fixer_stub(&dir, "fail.sh", "bad.txt"), Some("false"));

        std::env::set_current_dir(&dir).unwrap();
        let merged = run_fix(&pass, &s, "something broke").await;
        let rejected = run_fix(&fail, &s2, "something else broke").await;
        std::env::set_current_dir(&prev).unwrap();

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
}
