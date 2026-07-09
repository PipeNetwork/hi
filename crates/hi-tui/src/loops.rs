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
    List {
        reply: oneshot::Sender<Vec<LoopSpec>>,
    },
}

/// The UI's handle to the loop manager.
pub(crate) struct LoopsHandle {
    pub(crate) ctl: mpsc::UnboundedSender<LoopCtl>,
    /// Firing results awaiting display; the UI drains this on ticks.
    pub(crate) pending: Arc<Mutex<Vec<LoopLine>>>,
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
fn wrapper_prompt(spec: &LoopSpec) -> String {
    format!(
        "Recurring watch (loop \"{}\", every {}): {}\n\nThis conversation contains your previous \
         checks — compare against them rather than re-describing everything. If nothing \
         meaningful changed since the last check, reply with exactly: {QUIET_MARKER}. Otherwise \
         summarize what changed and why it matters, briefly.",
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
    let handle = LoopsHandle {
        ctl: ctl_tx,
        pending: pending.clone(),
    };
    tokio::spawn(manager(launcher, loops_file, ctl_rx, pending));
    handle
}

async fn manager(
    launcher: Arc<FleetLauncher>,
    loops_file: Option<PathBuf>,
    mut ctl: mpsc::UnboundedReceiver<LoopCtl>,
    pending: Arc<Mutex<Vec<LoopLine>>>,
) {
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

    // Firings in flight: (loop id, summary) results come back over a channel.
    let (done_tx, mut done_rx) = mpsc::unbounded_channel::<(u64, Result<String, String>)>();
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
            if spec.next_ms <= now && in_flight < 2 {
                spec.next_ms = now + spec.interval_secs * 1000;
                spec.firings += 1;
                in_flight += 1;
                fired = true;
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

        // Sleep until the next due time (capped so ctl/expiry stay responsive).
        let next_due = state
            .loops
            .iter()
            .map(|l| l.next_ms.min(l.expires_ms))
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
                let line = match result {
                    Ok(summary) => {
                        let quiet = is_quiet(&summary);
                        let text = if quiet {
                            format!("⟳ loop#{id} ({name}): nothing new")
                        } else {
                            format!("⟳ loop#{id} ({name}): {}", truncate(&summary, 160))
                        };
                        (text, !quiet)
                    }
                    Err(err) => (format!("⟳ loop#{id} ({name}) firing failed: {err}"), true),
                };
                pending.lock().unwrap().push(line);
            }
            _ = tokio::time::sleep(Duration::from_millis(sleep_ms)) => {}
        }
    }
}

/// One firing: a fleet-style child run in the real cwd, resuming the loop's
/// session. Returns the child's final reply line(s) as the summary.
async fn run_firing(launcher: &FleetLauncher, spec: &LoopSpec) -> Result<String, String> {
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
    Ok(summary)
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
        let w = wrapper_prompt(&spec());
        assert!(w.contains("every 30m"), "{w}");
        assert!(w.contains(QUIET_MARKER));
        assert!(w.contains("check whether the CI pipeline"));
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
}
