//! Full-screen terminal UI for `hi`.
//!
//! A ratatui application on the alternate screen in grok-build's session
//! chrome: a flat status bar, unboxed scrollback, a quiet rounded prompt, and
//! a shortcuts row. The agent runs behind an mpsc channel ([`ChannelUi`]) so
//! the event loop can keep redrawing — spinner, streaming output, scrolling —
//! while a turn is in flight, and can cancel it with Ctrl-C.

mod action;
mod activity;
mod activity_feed;
mod app;
#[doc(hidden)]
pub mod benchmark;
mod daemon;
mod dashboard;
mod dashboard_goal;
mod diff_lab;
mod dispatch;
mod domain;
mod file_mentions;
mod keys;
mod lock;
mod loops;
mod memory_browser;
mod mode;
mod notify;
mod palette;
mod plan_approval;
mod profiling;
mod race;
pub use app::run;
pub use daemon::run_loops_daemon;
mod block_viewer;
mod btw;
mod chrome;
mod completion;
mod confirm_overlay;
pub mod event;
mod input;
mod layout;
mod local_picker;
mod model_picker;
mod provider_form;
mod provider_picker;
mod render;
mod session_face;
mod session_pickers;
mod subagent_overlay;
mod sync_tui;
mod theme;
mod timeline;
mod turn_status;
mod tutorial;
mod util;
mod view_cache;
mod watch;
mod workflow_tui;

use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use anyhow::Result;
use hi_agent::{Agent, AgentStateSnapshot};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};

#[cfg(test)]
use {
    crate::event::UiEvent,
    crossterm::event::{KeyCode, KeyEvent, KeyModifiers},
    hi_agent::PlanStatus,
    ratatui::Terminal,
};

/// Info about a configured profile, for the `/provider` list and picker.
#[derive(Clone, Debug)]
pub struct ProfileInfo {
    pub name: String,
    /// Display label for the provider (e.g. "anthropic", "ollama").
    pub provider: String,
    /// The model configured on this profile, if any.
    pub model: Option<String>,
    /// The base URL configured on this profile, if any (non-default only).
    pub base_url: Option<String>,
    /// Repository identity for a hi-managed local runtime, when this profile
    /// can be recreated instead of using its stale persisted endpoint.
    pub managed_local_repo: Option<String>,
    /// Filesystem source for a managed local runtime, when it serves an
    /// existing MLX directory rather than a Hub repository.
    pub managed_local_path: Option<std::path::PathBuf>,
}

/// Runtime identity shown in the TUI independently of the OpenAI-compatible
/// wire provider label.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalRuntimeIdentity {
    pub backend: String,
    pub model_id: String,
    pub quantization: Option<String>,
    pub source: String,
    pub endpoint: Option<String>,
    pub ready: bool,
}

/// The result of resolving a profile name at runtime: a built provider, the
/// model id to use, and the provider's display label. The caller swaps these
/// into the agent via [`Agent::set_provider`].
pub struct SwitchedProvider {
    pub provider: Box<dyn hi_ai::Provider>,
    pub model: String,
    pub label: String,
    pub max_tokens: u32,
    pub max_tokens_explicit: bool,
    pub tool_mode: hi_ai::ToolMode,
    pub local_runtime: Option<LocalRuntimeIdentity>,
}

/// Result of saving/selecting a managed local MLX profile.
pub struct MlxProfileSwitch {
    pub switched: SwitchedProvider,
    pub profiles: Vec<ProfileInfo>,
}

/// A callback that persists a managed local runtime profile and builds the
/// OpenAI-compatible provider once the runtime has passed verification.
pub type LocalRuntimeSwitcher = Box<
    dyn Fn(&hi_agent::local_skeptic::ManagedLocalRuntime) -> Result<MlxProfileSwitch> + Send + Sync,
>;

/// A callback that resolves a named profile into a built provider + model +
/// label, for `/provider` mid-session. `hi-cli` supplies this; the TUI calls
/// it without needing to know about `Config`/`Settings` (which live in
/// `hi-cli`).
pub type ProfileResolver = Box<dyn Fn(&str) -> Result<SwitchedProvider> + Send + Sync>;

/// A single API target selected in Diff Lab. The profile is resolved by
/// `hi-cli`; only the non-secret profile name and model id cross the TUI seam.
#[derive(Clone, Debug)]
pub struct DiffApiTarget {
    pub name: String,
    pub profile: String,
    pub model: String,
}

/// An explicitly selected canonical request for a Diff Lab API run.
#[derive(Clone, Debug)]
pub struct DiffApiRunRequest {
    pub prompt: String,
    pub targets: Vec<DiffApiTarget>,
    pub seed: u64,
    pub cases: u64,
    pub max_concurrency: usize,
    pub max_requests: u64,
    pub max_tokens: u32,
}

/// Runtime callback supplied by `hi-cli` so the TUI can launch real provider
/// comparisons without depending on CLI config types or handling credentials.
pub type DiffApiRunner = Arc<
    dyn Fn(
            DiffApiRunRequest,
        ) -> Pin<Box<dyn Future<Output = Result<hi_diff::DiffRunSnapshot>> + Send>>
        + Send
        + Sync,
>;

/// An explicitly selected coding-race request. Credentials are resolved by
/// the CLI callback; only project-safe target metadata crosses this seam.
#[derive(Clone, Debug)]
pub struct RaceRunRequest {
    pub task: String,
    pub targets: Vec<hi_race::RaceTarget>,
    pub max_candidates: u32,
    pub max_concurrency: usize,
    pub verify_commands: Vec<String>,
    pub fuzz: Option<hi_race::FuzzConfig>,
    pub apply: bool,
    pub source_run_id: Option<String>,
    pub artifact_root: Option<std::path::PathBuf>,
    pub selected_candidate: Option<String>,
    pub expected_workspace_digest: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct RaceDefaults {
    pub targets: Vec<hi_race::RaceTarget>,
    pub max_candidates: u32,
    pub max_concurrency: usize,
    pub verify_commands: Vec<String>,
    pub fuzz: Option<hi_race::FuzzConfig>,
}

/// Runtime callback supplied by `hi-cli` for provider-backed coding races.
pub type RaceRunner = Arc<
    dyn Fn(RaceRunRequest) -> Pin<Box<dyn Future<Output = Result<hi_race::RaceSnapshot>> + Send>>
        + Send
        + Sync,
>;

pub type RaceSetupSaver = Arc<dyn Fn(Vec<hi_race::RaceTarget>) -> Result<String> + Send + Sync>;

/// Persist the active profile (if any), provider label, and model so the next
/// bare `hi` in this workspace restores the same routing. Best-effort: errors
/// are logged by the callback or ignored.
pub type SessionRemember = std::sync::Arc<dyn Fn(Option<&str>, &str, &str) + Send + Sync>;

/// Everything the `/dashboard` fleet needs to launch worktree-isolated child
/// `hi` runs: the binary + provider wiring for the child command line, the
/// verify pipeline for the merge gate, and a session-path allocator. `hi-cli`
/// supplies this so the TUI never touches `Settings`/session paths directly.
pub struct FleetLauncher {
    /// The `hi` binary to spawn for each row turn.
    pub exe: std::path::PathBuf,
    /// Explicit workspace root for trigger, worktree, merge, and verification operations.
    pub workspace_root: std::path::PathBuf,
    pub provider: String,
    pub model: String,
    pub base_url: String,
    pub api_key: String,
    /// Combined verify pipeline command, when the session has one: passed to
    /// the child (`--verify`) and re-run as the ground-truth merge gate.
    pub verify: Option<String>,
    pub max_verify: u32,
    pub max_steps: u32,
    /// Allocates a unique session file for a new fleet row (collision-safe).
    pub session_path: Box<dyn Fn() -> Result<std::path::PathBuf> + Send + Sync>,
    /// Lists this project's resumable fleet sessions (`/fleet status`).
    pub sessions: Box<dyn Fn() -> Vec<FleetSessionInfo> + Send + Sync>,
    /// Resolves a fleet session id (or "" = most recent) into everything needed
    /// to re-adopt it as a dashboard row (`/fleet resume [id]`).
    pub resume_info: FleetResumeResolver,
    /// Allocates a session file for a `/loop` (each firing resumes it).
    pub loop_session_path: Box<dyn Fn() -> Result<std::path::PathBuf> + Send + Sync>,
    /// Where `/loop` definitions persist across restarts (per project).
    pub loops_file: Option<std::path::PathBuf>,
}

/// Resolves a fleet session id into re-adoption info (`/fleet resume`).
pub type FleetResumeResolver = Box<dyn Fn(&str) -> Option<FleetResumeInfo> + Send + Sync>;

/// Lists sessions cached on this machine. The TUI merges these with synced
/// sessions before presenting the single `/sessions` view.
pub type SessionLister = Box<dyn Fn() -> Vec<LocalSessionInfo> + Send + Sync>;

/// Loads a session into the live agent and replaces its persistence sink,
/// restoring it from sync first when it is not cached on this machine.
pub type SessionSwitcher = Box<
    dyn for<'a> Fn(
            &'a str,
            &'a mut hi_agent::Agent,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = anyhow::Result<SessionSwitchInfo>> + Send + 'a>,
        > + Send
        + Sync,
>;

/// Persists a display name for a session cached on this machine.
pub type SessionRenamer = Box<dyn Fn(&str, &str) -> anyhow::Result<String> + Send + Sync>;
pub type SyncModeSetter = std::sync::Arc<dyn Fn(&str) -> anyhow::Result<()> + Send + Sync>;
pub type SyncStatusReader =
    std::sync::Arc<dyn Fn(Option<&str>) -> anyhow::Result<String> + Send + Sync>;
pub type SyncPurger = std::sync::Arc<dyn Fn() -> anyhow::Result<()> + Send + Sync>;

#[derive(Clone)]
pub struct SyncControl {
    pub set_mode: SyncModeSetter,
    pub status: SyncStatusReader,
    pub purge: SyncPurger,
}

#[derive(Clone, Debug)]
pub struct SessionSwitchInfo {
    pub id: String,
    pub summary: String,
}

/// Receives a copy of each agent UI event for live portal streaming.
pub type RemoteEventTap = std::sync::Arc<dyn Fn(&crate::event::UiEvent) + Send + Sync>;

/// Starts a non-blocking flush of portal records and live events.
pub type RemoteFlushCallback = std::sync::Arc<dyn Fn() + Send + Sync>;

/// Result of enabling host mode: a prompt receiver plus an abort handle for
/// the background poller. `None` means host mode is off.
pub type SessionHostEnable = (
    tokio::sync::mpsc::UnboundedReceiver<String>,
    tokio::task::AbortHandle,
);

/// Flip whether the active session advertises remote-input acceptance and
/// return a channel that yields prompts posted by attach clients. Returning
/// `None` means host mode was turned off (or failed).
pub type SessionHostController = Box<
    dyn Fn(
            bool,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = anyhow::Result<Option<SessionHostEnable>>> + Send>,
        > + Send
        + Sync,
>;

/// An in-flight `/team` local-model provisioning task.
pub(crate) struct PendingTeamProvision {
    pub(crate) role: String,
    pub(crate) display: String,
    /// Set when the user changes this role before setup finishes. The task is
    /// allowed to reach a safe completion so a spawned server can be stopped;
    /// its result must not overwrite the newer route choice.
    pub(crate) cancelled: bool,
    pub(crate) task: tokio::task::JoinHandle<anyhow::Result<(String, String, String)>>,
    /// Live phase reported by the provisioning task (download → build →
    /// load), so the transcript narrates what is actually happening.
    pub(crate) phase_rx: tokio::sync::watch::Receiver<hi_agent::local_skeptic::ProvisionPhase>,
    /// The last phase already announced in the transcript.
    pub(crate) announced_phase: hi_agent::local_skeptic::ProvisionPhase,
    /// When the current phase began (drives "Ns elapsed" heartbeats).
    pub(crate) phase_started: std::time::Instant,
    /// Where the weights land — polled for size so download heartbeats can
    /// say how much is on disk (the downloader itself is fully quiet; raw
    /// aria2c output once painted over the alternate screen).
    pub(crate) model_dir: std::path::PathBuf,
    /// Ticker calls since the last heartbeat line.
    pub(crate) ticks_since_report: u32,
    /// Bytes on disk at the last heartbeat.
    pub(crate) last_reported_bytes: u64,
    /// Transcript index of the in-place progress line for the current phase.
    pub(crate) progress_entry_index: Option<usize>,
}

/// In-flight driver-provider local runtime provisioning. The old provider is
/// left untouched until the task returns a verified runtime and the profile
/// callback succeeds.
pub(crate) struct PendingLocalProviderProvision {
    pub(crate) display: String,
    pub(crate) task:
        tokio::task::JoinHandle<anyhow::Result<hi_agent::local_skeptic::ManagedLocalRuntime>>,
    pub(crate) phase_rx: tokio::sync::watch::Receiver<hi_agent::local_skeptic::LocalRuntimePhase>,
    pub(crate) announced_phase: hi_agent::local_skeptic::LocalRuntimePhase,
    /// When the current phase began; used for honest elapsed-time heartbeats.
    pub(crate) phase_started: std::time::Instant,
    /// Ticker calls since the last in-place progress update.
    pub(crate) ticks_since_report: u32,
    /// Model directory polled for download progress.
    pub(crate) model_dir: std::path::PathBuf,
    /// Bytes on disk at the last progress update.
    pub(crate) last_reported_bytes: u64,
    /// Transcript index of the current in-place progress line.
    pub(crate) progress_entry_index: Option<usize>,
}

/// A session cached on this machine, merged into the `/sessions` list view.
#[derive(Clone, Debug)]
pub struct LocalSessionInfo {
    pub id: String,
    pub title: String,
    pub age: String,
    pub lines: usize,
}

/// A fleet session resolved for re-adoption as a dashboard row.
pub struct FleetResumeInfo {
    pub id: String,
    /// The session file (the row's child turns keep appending to it).
    pub path: std::path::PathBuf,
    /// The original dispatch prompt (row title).
    pub title: String,
    /// Whether the session's goal should keep auto-driving.
    pub goal_active: bool,
    pub goal_done: usize,
    pub goal_total: usize,
}

/// A resumable fleet session, as shown by `/fleet status`.
pub struct FleetSessionInfo {
    /// The `--resume` id.
    pub id: String,
    /// The row's dispatch prompt (cleaned first user message).
    pub title: String,
    /// Humanized age ("3m ago").
    pub age: String,
    /// Session length in lines.
    pub lines: usize,
}

/// A callback that persists the `/hf run --mlx` profile and returns a built
/// provider for immediate use.
pub type MlxProfileSwitcher =
    Box<dyn Fn(&hi_tools::HfMlxRun) -> Result<MlxProfileSwitch> + Send + Sync>;

/// Form data for creating or editing a profile, exchanged between the TUI
/// (which collects it via a form) and `hi-cli` (which writes it to the config
/// file). Mirrors `hi_cli::config::ProfileForm` but without the dependency.
#[derive(Clone, Debug)]
pub struct ProfileFormData {
    pub name: String,
    /// "ollama", "pipenetwork", "anthropic", or "openai".
    pub provider: String,
    pub api_key: String,
    /// If true, `api_key` is an env var name (stored as `api_key_env`).
    pub store_as_env: bool,
    pub model: String,
    pub base_url: String,
}

/// A callback that saves a profile (add or edit) to the config file and
/// returns the updated profile list. `hi-cli` supplies this; the TUI calls it
/// when the user submits the provider form.
pub type ProfileSaver = Box<dyn Fn(&ProfileFormData) -> Result<Vec<ProfileInfo>> + Send + Sync>;

/// A callback that loads an existing profile's form data for editing.
pub type ProfileLoader = Box<dyn Fn(&str) -> Result<ProfileFormData> + Send + Sync>;

/// A callback that removes a profile from the config file and returns the
/// updated profile list. `hi-cli` supplies this; the TUI calls it for
/// `/provider remove <name>`.
pub type ProfileRemover = Box<dyn Fn(&str) -> Result<Vec<ProfileInfo>> + Send + Sync>;

/// A callback that persists `reasoning_effort` machine-wide and, when `name` is
/// a real profile, onto that profile too. `hi-cli` supplies this; the TUI calls
/// it from `/config reasoning` so the choice sticks on this computer. Pass an
/// empty `name` (or a non-profile preset) for machine-only save — returns
/// `Ok(false)` then, `Ok(true)` when a profile field was also written.
pub type ReasoningEffortSaver =
    Box<dyn Fn(&str, Option<hi_ai::ReasoningEffort>) -> Result<bool> + Send + Sync>;

/// Everything needed to start the interactive TUI besides the live [`Agent`].
///
/// Prefer this over a long argument list at the `hi-cli` → `hi-tui` seam so new
/// callbacks/options don't grow another positional parameter.
pub struct RunOptions {
    pub provider: String,
    pub base_url: String,
    pub model: String,
    pub history_path: Option<std::path::PathBuf>,
    pub auto_memory: bool,
    pub profiles: Vec<ProfileInfo>,
    pub active_profile: Option<String>,
    pub resolver: ProfileResolver,
    pub saver: ProfileSaver,
    pub loader: ProfileLoader,
    pub remover: ProfileRemover,
    pub reasoning_effort_saver: Option<ReasoningEffortSaver>,
    pub mlx_switcher: MlxProfileSwitcher,
    pub local_runtime_switcher: LocalRuntimeSwitcher,
    /// A managed local profile selected on the previous launch. The TUI starts
    /// this runtime after first paint instead of blocking CLI startup.
    pub startup_local_runtime: Option<hi_agent::local_skeptic::LocalRuntimeSpec>,
    /// A prior non-local route to offer when persisted local startup fails.
    pub startup_fallback_profile: Option<String>,
    pub session_remember: Option<SessionRemember>,
    pub resume_summary: Option<String>,
    pub mcp_url: Option<String>,
    pub api_key: String,
    pub diff_api_runner: Option<DiffApiRunner>,
    pub race_runner: Option<RaceRunner>,
    pub race_defaults: RaceDefaults,
    pub race_setup_saver: Option<RaceSetupSaver>,
    /// Optional canonical lifecycle sink. UI transport remains separate from
    /// durable semantic events.
    pub event_sink: Option<Arc<dyn hi_events::EventSink>>,
    /// Optional durable approval broker. When present, confirmations are
    /// persisted and consumed before the side effect is allowed to run.
    pub approval_store: Option<Arc<dyn hi_policy::ApprovalStore>>,
    pub fleet_launcher: FleetLauncher,
    pub remote_event_tap: Option<RemoteEventTap>,
    pub remote_flush_callback: Option<RemoteFlushCallback>,
    pub sync_config: Option<SyncConfig>,
    pub sync_session_id: Option<String>,
    pub session_lister: Option<SessionLister>,
    pub session_switcher: Option<SessionSwitcher>,
    pub session_renamer: Option<SessionRenamer>,
    pub session_host: Option<SessionHostController>,
    pub sync_control: Option<SyncControl>,
}

use completion::CompletionState;
use input::InputLine;
use model_picker::ModelPicker;
pub(crate) use render::dim;
use render::line_text;

pub(crate) const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
/// How many model rows the `/model` picker shows at once.
pub(crate) const PICKER_ROWS: usize = 12;

/// Column width for the `/provider add|edit` form's field labels, so values
/// line up instead of starting at a ragged edge. Sized for "Base URL".
pub(crate) const FORM_LABEL_WIDTH: usize = 9;

/// A synchronous, plain (uncolored) `git diff` of the working tree, for the
/// full-screen review overlay (Ctrl-G / Ctrl-D). The TUI applies its own
/// highlighting via `diff_lines`, so we want the raw diff without ANSI codes.
/// Returns empty when not a git repo or there are no changes. Synchronous
/// because the key handler isn't async and `git diff` is fast/user-initiated.
pub(crate) fn working_tree_diff_sync(root: &std::path::Path) -> String {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["--no-pager", "diff", "--no-color", "HEAD"])
        .output();
    match out {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).into_owned(),
        // Not a git repo / no HEAD: fall back to an untracked+unstaged diff.
        Ok(_) => {
            let untracked = std::process::Command::new("git")
                .arg("-C")
                .arg(root)
                .args(["--no-pager", "diff", "--no-color"])
                .output();
            untracked
                .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
                .unwrap_or_default()
        }
        Err(_) => String::new(),
    }
}

/// Cheap session-start ghost text from `git status --porcelain` (no model call).
/// Mirrors Claude Code's "example from recent work" landing suggestion.
pub(crate) fn startup_prompt_suggestion(root: &std::path::Path) -> Option<String> {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["--no-pager", "status", "--porcelain", "-uall"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut paths: Vec<String> = Vec::new();
    for line in text.lines() {
        // porcelain v1: XY PATH or XY ORIG -> PATH for renames
        let rest = line.get(3..).unwrap_or("").trim();
        if rest.is_empty() {
            continue;
        }
        let path = rest
            .rsplit_once(" -> ")
            .map(|(_, to)| to)
            .unwrap_or(rest)
            .trim()
            .trim_matches('"');
        if path.is_empty() || path.starts_with(".hi/") {
            continue;
        }
        if !paths.iter().any(|p| p == path) {
            paths.push(path.to_string());
        }
        if paths.len() >= 3 {
            break;
        }
    }
    match paths.as_slice() {
        [] => None,
        [one] => Some(format!("Continue working on {one}")),
        [a, b] => Some(format!("Review uncommitted changes in {a} and {b}")),
        [a, b, ..] => Some(format!("Review uncommitted changes in {a}, {b}, and more")),
    }
}

/// Working-tree diff filtered to `files` (paths relative to `root`), via
/// `git diff HEAD -- <files>`. Used by the deep-link from a `✎ files changed`
/// transcript line to the full-screen diff review — opens the review showing
/// only the files the agent edited in that turn. Empty on failure or when no
/// paths match.
pub(crate) fn diff_for_files_sync(root: &std::path::Path, files: &[String]) -> String {
    if files.is_empty() {
        return String::new();
    }
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["--no-pager", "diff", "--no-color", "HEAD", "--"])
        .args(files)
        .output();
    match out {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).into_owned(),
        Ok(_) => String::new(),
        Err(_) => String::new(),
    }
}

pub(crate) const TICK: Duration = Duration::from_millis(120);
/// Only show an informational notice after a long, genuinely silent wait. This
/// is deliberately not a model-health signal: hosted APIs may buffer and retry
/// on the backend without streaming visible tokens to the TUI.
const DEFAULT_WATCHDOG_STUCK_SECS: u64 = 180;
const MIN_WATCHDOG_STUCK_SECS: u64 = 30;
const MAX_WATCHDOG_STUCK_SECS: u64 = 1_800;
/// On terminals that don't report focus, notify after a turn at least this long
/// (a proxy for "you probably stepped away").
pub(crate) const NOTIFY_THRESHOLD: Duration = Duration::from_secs(30);

pub(crate) fn watchdog_stuck_timeout() -> Duration {
    let configured = std::env::var("HI_TUI_WATCHDOG_SECS").ok();
    watchdog_stuck_timeout_from_value(configured.as_deref())
}

fn watchdog_stuck_timeout_from_value(value: Option<&str>) -> Duration {
    let seconds = value
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|seconds| *seconds > 0)
        .unwrap_or(DEFAULT_WATCHDOG_STUCK_SECS)
        .clamp(MIN_WATCHDOG_STUCK_SECS, MAX_WATCHDOG_STUCK_SECS);
    Duration::from_secs(seconds)
}

/// Apply a freshly fetched `/models` result: update the served-metadata map,
/// re-apply the current model (so its window/price refresh), and persist the
/// result to the on-disk cache for next startup. A failure or empty list sets a
/// startup notice instead of panicking.
pub(crate) fn apply_metadata(
    app: &mut App,
    agent: &mut Agent,
    result: &Result<Vec<hi_ai::ServedModel>>,
    cache_key: &str,
) {
    match result {
        Ok(served) if !served.is_empty() => {
            app.served = served.iter().cloned().map(|m| (m.id.clone(), m)).collect();
            app.model_ids = served.iter().map(|m| m.id.clone()).collect();
            app.model_ids.sort();
            let model_id = app.model.clone();
            app.apply_model(agent, &model_id);
            // Persist for next startup (best-effort, fire-and-forget).
            let models = served.clone();
            let key = cache_key.to_string();
            tokio::spawn(async move {
                hi_ai::save_cache(&key, &models).await;
            });
        }
        Ok(_) => {
            app.startup_notice = Some("model metadata not loaded".into());
        }
        Err(err) => {
            app.startup_notice = Some(format!("model metadata not loaded: {err:#}"));
        }
    }
}

/// One entry in the display transcript. Most content is a plain styled line;
/// reasoning (CoT) is stored specially so it can be collapsed by default and
/// expanded on demand via Ctrl-T, rather than flooding the transcript inline.
#[derive(Clone)]
pub(crate) enum TranscriptEntry {
    Line(Line<'static>),
    /// A user prompt echo (`❯ …`). Structurally distinct from a plain `Line` so
    /// the render pass can find prompt boundaries for sticky headers — when the
    /// transcript is scrolled past a prompt, that prompt pins to the top so the
    /// visible output always shows which request it belongs to. `at` is the
    /// wall-clock stamp grok-build right-aligns on the first prompt row.
    UserPrompt {
        line: Line<'static>,
        at: SystemTime,
    },
    /// A line of assistant prose. It stays separate from status/tool lines so
    /// the renderer can add the assistant gutter without changing copied text.
    Assistant(Line<'static>),
    /// One grok-build assistant reply (markdown source), flattened as a block.
    AssistantMessage {
        text: String,
    },
    /// Assistant reasoning/thinking, buffered until the reasoning phase ends.
    /// Shown collapsed ("thought for Ns") unless `show_reasoning` is on.
    Reasoning {
        text: String,
        elapsed: Duration,
    },
    /// A `✎ N files changed: …` line. Carries the file list so a click (or
    /// block-nav Enter) can open the full-screen diff review (Ctrl-G) filtered
    /// to just those files — deep-linking the transcript to the review overlay.
    ChangedFiles {
        line: Line<'static>,
        files: Vec<String>,
    },
    /// Persisted `/btw` aside after the overlay is dismissed. Collapsed shows
    /// the golden `/btw <question>` header (grok-build); expand to read the
    /// markdown answer.
    Btw {
        question: String,
        answer: String,
        expanded: bool,
    },
    /// Durable workflow lifecycle block, replaced in place as newer revisions
    /// arrive rather than appended once per update.
    Workflow {
        snapshot: hi_workflow::WorkflowRunSnapshot,
    },
    /// Typed activity row (Read / Edit / Run / verb group). Collapsed by
    /// default; Edit/Run expand into hunks or stdout.
    Activity(crate::activity_feed::ActivityBlock),
    /// A tool's (non-explore) output as a foldable block: the full body is
    /// retained, but only a preview shows by default when it's long, with the
    /// remainder revealed by `Ctrl-O` (or per the global `show_tool_output`).
    /// Keeps a burst of shell output from burying the conversation while never
    /// discarding it (the old path hard-truncated at 16 lines).
    ToolOutput {
        /// The already-styled body lines (gutter + diff/ANSI coloring applied).
        body: Vec<Line<'static>>,
        /// Per-block expand override set by block-nav (Ctrl-B → Enter). When
        /// `true` this block shows in full even with the global fold on; the
        /// global `show_tool_output` still force-expands every block over it.
        expanded: bool,
    },
}

/// How many lines of a long generic tool-output block show before it folds.
/// Activity rows (Read/Edit/Run) are one line regardless; this budget only
/// applies to leftover `ToolOutput` dumps. Comfortable density uses 0 so the
/// transcript stays a verb list, matching grok-build.
pub(crate) const TOOL_OUTPUT_PREVIEW_LINES: usize = 0;

/// Grok-build user prompts are a quiet `❯ ` prefix, not a `┃` accent gutter.
fn style_user_prompt(line: &Line<'static>) -> Line<'static> {
    let th = crate::theme::theme();
    let text = line_text(line);
    let trimmed = text.trim_start();
    if trimmed.starts_with('❯') || trimmed.starts_with('>') {
        return line.clone();
    }
    let mut spans = Vec::with_capacity(line.spans.len() + 1);
    spans.push(Span::styled(
        "❯ ",
        Style::default()
            .fg(th.tone_color(crate::theme::UiTone::User))
            .add_modifier(Modifier::BOLD),
    ));
    spans.extend(line.spans.iter().cloned());
    Line::from(spans)
}

impl TranscriptEntry {
    /// Flatten this entry into display lines under the current fold settings.
    /// A collapsed reasoning block is one dim summary line; a long tool-output
    /// block shows a preview plus a fold footer unless `show_tool_output` is on
    /// (or density is verbose). Compact density keeps headers only.
    pub(crate) fn flatten(
        &self,
        show_reasoning: bool,
        show_tool_output: bool,
        density: Density,
    ) -> Vec<Line<'static>> {
        let th = crate::theme::theme();
        let show_tool = density.show_tool_output(show_tool_output);
        let preview_n = density.tool_preview_lines();
        match self {
            TranscriptEntry::Line(line) => vec![line.clone()],
            TranscriptEntry::UserPrompt { line, .. } => {
                let mut prompt = style_user_prompt(line);
                if th.paints_backgrounds() {
                    prompt.style = prompt.style.bg(th.bg_highlight);
                }
                vec![prompt]
            }
            TranscriptEntry::Assistant(line) => crate::render::wrap_line_to_width(line, 120),
            TranscriptEntry::AssistantMessage { text } => {
                let mut lines = Vec::new();
                for line in crate::render::markdown_body_lines(text) {
                    lines.extend(crate::render::wrap_line_to_width(&line, 120));
                }
                lines
            }
            TranscriptEntry::ChangedFiles { line, .. } => vec![line.clone()],
            TranscriptEntry::Btw {
                question,
                answer,
                expanded,
            } => {
                let header = Line::from(Span::styled(
                    format!("/btw {question}"),
                    Style::default()
                        .fg(th.accent_plan)
                        .add_modifier(Modifier::BOLD),
                ));
                if !*expanded {
                    vec![header]
                } else {
                    let mut lines = vec![header, Line::raw("")];
                    lines.extend(crate::render::markdown_body_lines(answer));
                    lines
                }
            }
            TranscriptEntry::Workflow { snapshot } => workflow_snapshot_lines(snapshot),
            TranscriptEntry::Activity(block) => block.flatten(show_tool, show_reasoning, density),
            TranscriptEntry::Reasoning { text, elapsed } => {
                let secs = elapsed.as_secs();
                // Instant CoT is noise in the collapsed feed — grok-build
                // folds it into the tool row. Keep it for Ctrl-T.
                if !show_reasoning && secs == 0 {
                    return Vec::new();
                }
                crate::activity_feed::thinking_block_lines(text, *elapsed, show_reasoning)
            }
            TranscriptEntry::ToolOutput { body, expanded } => {
                // The visible body lines sit in a sunken panel (a `panel` base
                // background) on truecolor themes, tagging them so the render
                // pass can pad them to full width. The fold footer stays plain
                // so the fold boundary reads as the panel's edge.
                let panel = th.panel;
                let tag = |line: &Line<'static>| -> Line<'static> {
                    // Event handlers normally add this gutter before storing
                    // the body. Normalize restored/test/future bodies here as
                    // well so every tool block has the same visual grammar.
                    let mut l =
                        crate::render::with_gutter(line, th.tone_color(crate::theme::UiTone::Tool));
                    if th.paints_backgrounds() {
                        l.style = l.style.bg(panel);
                    }
                    l
                };
                // Compact: a single fold line naming the hidden body.
                if density == Density::Compact && !*expanded && !show_tool {
                    let hidden = body.len();
                    return vec![Line::from(vec![
                        Span::styled("┃ ", Style::default().fg(th.gray_dim)),
                        Span::styled(
                            format!("… {hidden} lines folded · Ctrl-O / /density"),
                            Style::default()
                                .fg(th.gray_dim)
                                .add_modifier(Modifier::ITALIC),
                        ),
                    ])];
                }
                // Short output, the global expand toggle, or this block's own
                // expand override shows in full; otherwise a preview + a fold
                // footer naming what's hidden.
                if show_tool || *expanded || body.len() <= preview_n {
                    body.iter().map(tag).collect()
                } else {
                    let hidden = body.len() - preview_n;
                    let mut lines: Vec<Line<'static>> = body[..preview_n].iter().map(tag).collect();
                    lines.push(Line::from(vec![
                        Span::styled("┃ ", Style::default().fg(th.gray_dim)),
                        Span::styled(
                            format!("… +{hidden} more lines · Ctrl-O to expand"),
                            Style::default()
                                .fg(th.gray_dim)
                                .add_modifier(Modifier::ITALIC),
                        ),
                    ]));
                    lines
                }
            }
        }
    }

    /// The plain text of this entry, for /copy and /export (always the full
    /// content regardless of collapse state).
    pub(crate) fn text(&self) -> String {
        match self {
            TranscriptEntry::Line(line)
            | TranscriptEntry::Assistant(line)
            | TranscriptEntry::ChangedFiles { line, .. } => crate::render::copy_line_text(line),
            TranscriptEntry::AssistantMessage { text } => text.clone(),
            // User prompts are stored without a semantic gutter. Preserve
            // literal leading glyphs the user typed instead of normalizing
            // their content as renderer decoration.
            TranscriptEntry::UserPrompt { line, .. } => line_text(line),
            TranscriptEntry::Btw {
                question, answer, ..
            } => format!("/btw {question}\n{answer}"),
            TranscriptEntry::Reasoning { text, .. } => text.clone(),
            TranscriptEntry::Workflow { snapshot } => workflow_snapshot_text(snapshot),
            TranscriptEntry::Activity(block) => block.text(),
            TranscriptEntry::ToolOutput { body, .. } => {
                body.iter().map(line_text).collect::<Vec<_>>().join("\n")
            }
        }
    }

    /// Foldable blocks that block-nav / click-to-expand step over.
    pub(crate) fn is_foldable(&self) -> bool {
        match self {
            Self::ToolOutput { .. } | Self::Btw { .. } => true,
            Self::Activity(block) => block.is_foldable() || block.subagent_id().is_some(),
            _ => false,
        }
    }

    /// Per-block expand flag, when this entry is foldable.
    pub(crate) fn expanded_mut(&mut self) -> Option<&mut bool> {
        match self {
            Self::ToolOutput { expanded, .. } | Self::Btw { expanded, .. } => Some(expanded),
            Self::Activity(block) if block.is_foldable() => Some(&mut block.expanded),
            _ => None,
        }
    }
}

fn workflow_status_label(status: hi_workflow::WorkflowRunStatus) -> &'static str {
    use hi_workflow::WorkflowRunStatus::*;
    match status {
        Active => "running",
        UserPaused => "paused by user",
        BackOffPaused => "paused for backoff",
        NoProgressPaused => "paused: no progress",
        InfraPaused => "paused: infrastructure",
        Blocked => "blocked",
        BudgetLimited => "paused: budget limited",
        Interrupted => "interrupted",
        Complete => "completed",
        Failed => "failed",
        Cancelled => "cancelled",
    }
}

fn workflow_snapshot_lines(snapshot: &hi_workflow::WorkflowRunSnapshot) -> Vec<Line<'static>> {
    let th = crate::theme::theme();
    let color = match snapshot.status {
        hi_workflow::WorkflowRunStatus::Complete => th.accent_success,
        hi_workflow::WorkflowRunStatus::Failed => th.accent_error,
        _ => th.accent_assistant,
    };
    let mut lines = vec![Line::styled(
        format!(
            "◆ workflow · {} · {}",
            snapshot.workflow_name,
            workflow_status_label(snapshot.status)
        ),
        Style::default().fg(color).add_modifier(Modifier::BOLD),
    )];
    if let Some(phase) = &snapshot.current_phase {
        lines.push(Line::styled(
            format!("  phase · {phase}"),
            Style::default().fg(th.text_secondary),
        ));
    }
    if let Some(message) = snapshot
        .pause_message
        .as_deref()
        .or(snapshot.result_summary.as_deref())
    {
        lines.push(Line::styled(
            format!("  {message}"),
            Style::default().fg(th.text_secondary),
        ));
    }
    lines.push(Line::styled(
        format!(
            "  agents · {}/{} · {} ms",
            snapshot.agents_used, snapshot.agent_budget, snapshot.elapsed_ms
        ),
        Style::default().fg(th.gray_dim),
    ));
    lines
}

fn workflow_snapshot_text(snapshot: &hi_workflow::WorkflowRunSnapshot) -> String {
    workflow_snapshot_lines(snapshot)
        .iter()
        .map(line_text)
        .collect::<Vec<_>>()
        .join("\n")
}

/// One line (or multi-line block) in the live `/btw` overlay thread.
#[derive(Clone, Debug)]
pub(crate) enum BtwEntry {
    /// User side question.
    Question(String),
    /// In-flight marker while the side loop thinks / inspects.
    Thinking(String),
    /// Compact tool crumb (`· read path`).
    Tool { name: String, detail: String },
    /// Streamed / final side answer.
    Answer(String),
}

impl BtwEntry {
    #[cfg(test)]
    pub(crate) fn as_lines(&self) -> Vec<String> {
        match self {
            BtwEntry::Question(q) => vec![format!("❓ {q}")],
            BtwEntry::Thinking(msg) => vec![format!("  … {msg}")],
            BtwEntry::Tool { name, detail } => {
                let d = detail.trim();
                if d.is_empty() {
                    vec![format!("  · {name}")]
                } else {
                    // Keep crumbs short so the overlay stays scannable.
                    vec![format!(
                        "  · {name} {}",
                        crate::layout::truncate_display(d, 56)
                    )]
                }
            }
            BtwEntry::Answer(a) => a
                .lines()
                .enumerate()
                .map(|(i, line)| {
                    if i == 0 {
                        format!("↳ {line}")
                    } else {
                        format!("  {line}")
                    }
                })
                .collect(),
        }
    }
}

pub(crate) struct App {
    pub(crate) provider: String,
    pub(crate) model: String,
    /// Live execution mode shown in the title bar and controlled by
    /// `/durable`. Mirrored from the agent because rendering does not borrow
    /// the agent.
    pub(crate) execution: hi_agent::ExecutionMode,
    /// The current reasoning effort level (`None` = off / endpoint default),
    /// mirrored from the agent for the title bar.
    pub(crate) reasoning_effort: Option<hi_ai::ReasoningEffort>,
    /// Explicit workspace root copied from the agent runtime for synchronous
    /// frontend-only operations such as the full-screen diff review overlay.
    pub(crate) workspace_root: std::path::PathBuf,
    /// A shared interrupt handle for the running turn. When the user presses
    /// Esc during a tool call, this is set so the agent skips the current tool
    /// and feeds "interrupted by user" back to the model.
    pub(crate) interrupt: Option<Arc<std::sync::atomic::AtomicBool>>,
    /// The name of the currently-active profile, if any (for marking it in the
    /// `/provider` list). Updated when the user uses `/provider <name>`.
    pub(crate) active_profile: Option<String>,
    /// Configured profiles (for `/provider` with no arg).
    pub(crate) profiles: Vec<ProfileInfo>,
    /// Resolves a profile name to a built provider at runtime (for `/provider`).
    pub(crate) resolver: ProfileResolver,
    /// Saves a profile form to the config file (for `/provider add/edit`).
    pub(crate) saver: ProfileSaver,
    /// Loads an existing profile's form data (for `/provider edit`).
    pub(crate) loader: ProfileLoader,
    /// Removes a profile from the config file (for `/provider remove`).
    pub(crate) remover: ProfileRemover,
    /// Persists `reasoning_effort` to a profile (for `/config reasoning`).
    pub(crate) reasoning_effort_saver: Option<ReasoningEffortSaver>,
    /// Saves/selects a managed local MLX profile after `/hf run --mlx`.
    pub(crate) mlx_switcher: MlxProfileSwitcher,
    /// Provisions and saves a managed local model selected in the provider
    /// picker. The callback runs only after the runtime is verified.
    pub(crate) local_runtime_switcher: LocalRuntimeSwitcher,
    /// Best-effort persist of active profile/provider/model for next launch.
    pub(crate) session_remember: Option<crate::SessionRemember>,
    pub(crate) transcript: Vec<TranscriptEntry>,
    /// Highest accepted workflow revision by run. Terminal entries remain here
    /// as tombstones so delayed active updates cannot resurrect a completed run.
    pub(crate) workflow_revisions: HashMap<String, (u64, bool)>,
    /// Completion-reportable workflow revisions already handed back to the
    /// primary agent, preventing duplicate summary turns after redraw/replay.
    pub(crate) workflow_completion_handoffs: HashMap<String, u64>,
    /// The in-progress streamed line: (style, markdown?, text). Committed on
    /// newline/end. `markdown` is set for assistant prose so it's rendered with
    /// light markdown styling; reasoning and other streams stay literal.
    pub(crate) pending: Option<(Style, bool, String)>,
    /// Buffer for assistant reasoning (CoT) chunks: accumulated until the
    /// reasoning phase ends, then committed as a single collapsible
    /// `TranscriptEntry::Reasoning` so it doesn't flood the transcript inline.
    pub(crate) reasoning_buffer: String,
    /// When the current reasoning phase started (for the "thought for Ns" label).
    pub(crate) reasoning_started: Option<Instant>,
    /// Whether reasoning (CoT) blocks are expanded inline. Off by default —
    /// reasoning is collapsed to a one-line "thought for Ns" summary; Ctrl-T
    /// toggles this to show/hide the full thinking text.
    pub(crate) show_reasoning: bool,
    /// Whether long tool-output blocks are expanded in full. Off by default —
    /// output beyond [`TOOL_OUTPUT_PREVIEW_LINES`] folds to a preview; Ctrl-O
    /// toggles this to reveal every block's full body. Verbose [`Density`] also
    /// forces expansion.
    pub(crate) show_tool_output: bool,
    /// Transcript density (compact / comfortable / verbose). Independent of the
    /// Ctrl-O expand toggle; verbose forces full tool bodies.
    pub(crate) density: Density,
    /// Exclusive keyboard-owning interaction mode (insert / normal / block-nav /
    /// history-search / review). Panels (help, debug, diff) stay separate flags.
    pub(crate) mode: crate::mode::UiMode,
    /// The last transcript search query, reused by `n`/`N` to jump to the
    /// next/previous match.
    pub(crate) last_search: Option<String>,
    /// The selected block's ordinal among tool-output blocks (0-based, oldest
    /// first). Clamped to the current block count wherever it's used.
    pub(crate) block_cursor: usize,
    /// Bumped on every structural transcript change so the view cache can skip
    /// rebuilds on spinner-only redraws.
    pub(crate) transcript_gen: u64,
    /// Cached flatten + wrap measurements for the transcript viewport.
    pub(crate) view_cache: crate::view_cache::TranscriptViewCache,
    /// Identity of the cache data copied into the interaction geometry fields.
    /// Scroll-only redraws leave those maps untouched.
    pub(crate) view_geometry_key: Option<crate::view_cache::ViewCacheKey>,
    /// The language of the ``` fence the streamed assistant text is currently
    /// inside (empty string if the fence gave none); `None` when not in a fence.
    /// Carries across streamed lines so code interiors highlight consistently.
    pub(crate) code_lang: Option<String>,
    /// The most recent fenced code block the assistant streamed, captured as
    /// plain text so Ctrl-Y can copy it with one keystroke (no mouse drag).
    /// Rebuilt line-by-line as code streams in (`commit_md_line`), and cleared
    /// when a fence closes so it holds the just-finished block.
    pub(crate) last_code_block: Option<String>,
    /// Source lines of a pipe table being accumulated during streaming, so it can
    /// be rendered with columns aligned across all rows once the table ends (a
    /// non-table line, or the message ends). Empty when not inside a table.
    pub(crate) table_buf: Vec<String>,
    pub(crate) input: InputLine,
    /// Voice dictation state (Ctrl+Space). Idle unless the user is recording
    /// or a transcription is still running.
    pub(crate) voice: crate::app::voice::VoiceState,
    /// Lazily-loaded Whisper model, kept across recordings — loading it costs
    /// seconds and ~1.6 GB, so it must not be repeated per dictation.
    pub(crate) voice_model: crate::app::voice::VoiceModelCache,
    /// Language / model settings for dictation.
    pub(crate) voice_config: hi_voice::VoiceConfig,
    /// Transcript scroll state. `following` pins the view to the latest output
    /// (the default); scrolling up unpins it and `scroll` holds the absolute
    /// offset (wrapped lines hidden above the viewport). It re-pins once scrolled
    /// back to the bottom, so streaming output never yanks a reader downward.
    pub(crate) following: bool,
    pub(crate) scroll: u16,
    /// Cached each render so scroll events (which fire outside render and don't
    /// know the wrapped height) can clamp and detect the bottom.
    pub(crate) view_max_scroll: u16,
    pub(crate) view_total: u16,
    /// Cached each render so a mouse click can be mapped back to a transcript
    /// block: the transcript's inner rect, the scroll offset applied, and each
    /// tool-output block's absolute wrapped-row span with its ordinal.
    pub(crate) view_inner: ratatui::layout::Rect,
    pub(crate) view_scroll: u16,
    pub(crate) block_row_spans: Vec<(u32, u32, usize)>,
    /// Cached each render for mouse text selection: the prefix-sum of wrapped
    /// rows per flattened line (`view_prefix[i]` = rows above line `i`; length is
    /// `lines + 1`) and each flattened line's plain text, so a drag can be mapped
    /// to a line range and that range copied.
    pub(crate) view_prefix: Vec<u32>,
    pub(crate) view_line_texts: Vec<String>,
    /// Active mouse text selection, as `(flattened line index, char column)`
    /// points (anchor = where the drag began, cursor = where it is now). The
    /// column drives character-precise selection when both points are on the same
    /// non-wrapped line; otherwise selection falls back to whole lines. `dragged`
    /// marks that motion occurred, so a plain click still folds a block.
    pub(crate) select_anchor: Option<(usize, usize)>,
    pub(crate) select_cursor: Option<(usize, usize)>,
    pub(crate) select_dragged: bool,
    /// A transient "copied N chars" confirmation (char count + when it was set),
    /// shown briefly above the input after a drag-copy so the copy is visible.
    pub(crate) copy_toast: Option<(usize, Instant)>,
    /// Whether the app is capturing the mouse (scroll wheel, click-to-fold,
    /// drag-to-copy). `/mouse off` releases it so the terminal's native text
    /// selection works; `/mouse on` re-enables. On by default.
    pub(crate) mouse_capture: bool,
    /// Scrollback-oriented minimal transcript rendering preference.
    pub(crate) minimal_screen: bool,
    /// Whether Esc/normal-mode vim navigation is enabled.
    pub(crate) vim_mode: bool,
    /// Explicit multiline composer preference (Alt-Enter always remains available).
    pub(crate) multiline_mode: bool,
    /// Show a turn rail in the transcript.
    pub(crate) timeline_enabled: bool,
    /// Prefix newly displayed command reports with local timestamps.
    pub(crate) timestamps_enabled: bool,
    /// Wrapped-line total at the moment the view last left the bottom — drives
    /// the "↓ N new" indicator while scrolled up.
    pub(crate) total_when_unpinned: u16,
    pub(crate) working: bool,
    pub(crate) spinner: usize,
    /// When the current turn started, for the elapsed-time readout.
    pub(crate) started: Option<Instant>,
    /// Wall-clock latency of the most recently completed turn.
    pub(crate) last_turn_latency: Option<Duration>,
    /// When the last turn finished (working true→false), for the brief accent
    /// "finish flash" on the status line. Cleared implicitly once its window
    /// elapses (the flash weight decays to zero).
    pub(crate) finished_at: Option<Instant>,
    /// The tool currently executing (its display label) and when it started, so
    /// Esc can interrupt just that call. `None` while the model is the active party.
    pub(crate) current_tool: Option<String>,
    pub(crate) current_tool_started: Option<Instant>,
    /// Streamed stdout already landed on the live `Run` row for this tool call.
    pub(crate) run_streamed_this_call: bool,
    /// Lines typed while a turn was running, to run once it finishes (FIFO).
    pub(crate) queue: VecDeque<String>,
    /// Plain-text lines offered to the in-flight turn as mid-turn steering
    /// (also present in [`Self::queue`] until applied or the turn ends). Used to
    /// drop queue entries that the agent already consumed so they don't re-run.
    pub(crate) mid_turn_offered: VecDeque<String>,
    /// Index into `queue` for Alt-Up/Down selection (reorder / delete). `None`
    /// when nothing is highlighted; clamped whenever the queue shrinks.
    pub(crate) queue_selected: Option<usize>,
    /// After submit, pin the new `❯` at the top of the transcript (grok-build
    /// `page_flip_on_send`) once, then leave scroll alone.
    pub(crate) page_flip_on_send: bool,
    /// The last message actually sent to the model, for `/retry`.
    pub(crate) last_prompt: Option<String>,
    /// Message-history length just before the last turn started, so `/retry`
    /// can drop that turn before re-running.
    pub(crate) last_turn_start: usize,
    /// Prompt-injected state just before the last turn started, so `/retry` and
    /// interrupt cleanup do not leak decisions/goals/plans from the discarded
    /// attempt.
    pub(crate) last_turn_snapshot: Option<AgentStateSnapshot>,
    /// Active model picker (`/model` with no argument), if any.
    pub(crate) picker: Option<ModelPicker>,
    /// The shared picker is browsing sessions rather than models.
    pub(crate) session_picker: bool,
    pub(crate) session_picker_searching: bool,
    pub(crate) session_catalog_flags: HashMap<String, (bool, bool)>,
    pub(crate) session_delete_pending: Option<String>,
    /// Active provider form (`/provider add` or `/provider edit`), if any.
    pub(crate) provider_form: Option<provider_form::ProviderForm>,
    /// Dedicated local-model picker opened by `/local`.
    pub(crate) local_picker: Option<local_picker::LocalModelPicker>,
    /// Path entry mode for the local-model directory action.
    pub(crate) local_directory_prompt: Option<String>,
    pub(crate) local_download_confirmation: Option<hi_agent::local_skeptic::LocalModelOption>,
    /// True while the startup-managed local runtime is being restored.
    pub(crate) local_startup_blocked: bool,
    pub(crate) local_startup_error: Option<String>,
    pub(crate) local_startup_spec: Option<hi_agent::local_skeptic::LocalRuntimeSpec>,
    pub(crate) local_startup_fallback_profile: Option<String>,
    /// Runtime identity shown in the header/status bar.
    pub(crate) local_runtime: Option<LocalRuntimeIdentity>,
    /// Active `/provider` selector (no arg), if any. Selecting a row queues
    /// `/provider <name>`, so it shares the typed-command switch path.
    pub(crate) provider_picker: Option<provider_picker::ProviderPicker>,
    /// Background refresh of the Pipe Network local-model catalog. The picker
    /// opens immediately with built-in rows and updates when this completes.
    pub(crate) pending_local_catalog: Option<
        tokio::task::JoinHandle<anyhow::Result<Vec<hi_agent::local_skeptic::LocalCatalogModel>>>,
    >,
    /// Callback for launching configured multi-provider Diff Lab API runs.
    pub(crate) diff_api_runner: Option<DiffApiRunner>,
    pub(crate) race_runner: Option<RaceRunner>,
    pub(crate) race_defaults: RaceDefaults,
    pub(crate) race_setup_saver: Option<RaceSetupSaver>,
    /// Canonical semantic lifecycle sink; transport events remain separate.
    pub(crate) event_sink: Option<Arc<dyn hi_events::EventSink>>,
    /// Local-only approval broker for recoverable workflow approval resumes.
    pub(crate) approval_store: Option<Arc<dyn hi_policy::ApprovalStore>>,
    /// When set, a model-list fetch is in flight (start time, for the spinner).
    pub(crate) fetching: Option<Instant>,
    /// When set, a `/goal` decomposition (planner call) is in flight (start time,
    /// for the spinner).
    pub(crate) planning: Option<Instant>,
    pub(crate) status: String,
    /// The latest task plan from the `update_plan` tool, pinned above the input
    /// as a live checklist. Empty until the model posts a plan; replaced wholesale
    /// on each update so it never drifts.
    pub(crate) plan: Vec<hi_agent::PlanStep>,
    /// Local interactive mutation confirmation currently shown by the turn driver.
    pub(crate) confirmation: Option<hi_agent::ConfirmationRequest>,
    pub(crate) confirmation_scroll: usize,
    /// Highlighted option on the permission / ask-user overlay.
    pub(crate) confirmation_selected: usize,
    pub(crate) confirm_focus: crate::confirm_overlay::ConfirmFocus,
    /// Parked leftover-plan card (Approve / Request changes / Quit).
    pub(crate) plan_approval: Option<crate::plan_approval::PlanApproval>,
    /// `/memory` split browser over project/global markdown files.
    pub(crate) memory_browser: Option<crate::memory_browser::MemoryBrowser>,
    /// Confirmations waiting behind the active overlay (`N waiting`).
    pub(crate) confirmation_waiting: usize,
    /// Last mouse cell, for hover chrome (context bar).
    pub(crate) mouse_col: u16,
    pub(crate) mouse_row: u16,
    pub(crate) ctx_chip_rect: ratatui::layout::Rect,
    pub(crate) turn_status_rect: ratatui::layout::Rect,
    /// Cached `git rev-parse --abbrev-ref HEAD` for the welcome home line.
    pub(crate) git_branch: Option<String>,
    /// When false, the plan/todo list above the composer collapses to a header.
    pub(crate) plan_pane_expanded: bool,
    /// The active long-horizon goal, mirrored from the agent so the pinned plan
    /// block and header can show sub-goal progress. Refreshed when `/goal` sets it
    /// and after every turn (the driver may advance it). `None` when no goal is set.
    pub(crate) goal: Option<hi_agent::Goal>,
    /// Mirrored from the agent so empty Enter can respect `/plan` draft mode.
    pub(crate) plan_mode: bool,
    /// Mirrored permission ladder (`ask` / `auto` / `always`).
    pub(crate) permission_mode: hi_agent::PermissionMode,
    /// Composer flags changed while the agent was borrowed (mid-turn Shift-Tab).
    pub(crate) session_face_dirty: bool,
    /// Mirrored from the agent so chrome can show paused/parked.
    pub(crate) plan_drive_paused: bool,
    /// Cached leftover-work gate, refreshed after each turn and `/plan`/`/goal`.
    pub(crate) last_drive: hi_agent::DriveAction,
    /// Last turn's stop reason, used to keep Cancelled / infrastructure idle.
    pub(crate) last_stop_reason: Option<hi_agent::TurnStopReason>,
    /// In-progress custom answer while an `ask_user` overlay is open.
    pub(crate) ask_user_draft: String,
    /// The `/dashboard` fleet: dispatched agents (one session each), persisted
    /// across dashboard open/close so rows aren't lost when you drop back to
    /// the chat. In-flight turns live only inside the dashboard loop.
    pub(crate) fleet: Vec<crate::dashboard::FleetRow>,
    /// Monotonic display id for fleet rows (never reused within a session).
    pub(crate) fleet_next_id: usize,
    /// Script workflow runs launched via `/workflow <name>`, keyed by their
    /// durable run ID. The selected ID controls dashboard presentation.
    pub(crate) workflow_runs: HashMap<String, crate::dashboard::WorkflowRun>,
    pub(crate) selected_workflow_run: Option<String>,
    /// Modal multi-run workflow browser opened by `/workflow` with no args.
    pub(crate) workflow_overlay: Option<crate::workflow_tui::WorkflowOverlay>,
    /// Live child explore/delegate/task rows, keyed by subagent id.
    pub(crate) subagents: HashMap<String, crate::subagent_overlay::SubagentInfo>,
    pub(crate) inspect_subagent: Option<crate::subagent_overlay::InspectOverlay>,
    pub(crate) tasks_overlay: Option<crate::subagent_overlay::TasksOverlay>,
    pub(crate) block_viewer: Option<crate::block_viewer::BlockViewer>,
    pub(crate) jump_picker: Option<crate::session_pickers::JumpPicker>,
    pub(crate) rewind_picker: Option<crate::session_pickers::RewindPicker>,
    /// Last painted timeline rail hit targets (screen row → tick).
    pub(crate) timeline_hits: Vec<(u16, crate::timeline::TimelineHit)>,
    pub(crate) timeline_rect: ratatui::layout::Rect,
    /// Interactive differential runner overlay. Large run data lives in
    /// `hi-diff` artifacts; this field only retains the bounded UI snapshot.
    pub(crate) diff_lab: Option<crate::diff_lab::DiffLabOverlay>,
    /// Active coding-race review overlay.
    pub(crate) race: Option<crate::race::RaceOverlay>,
    /// Detached `hi workflow run <plan>` child launched via `/workflow plan`:
    /// (pid, log path, plan label). Session-local tracking for status/stop.
    pub(crate) plan_workflow_child: Option<(u32, std::path::PathBuf, String)>,
    /// Handle to the `/loop` manager (timers + firings run in a background
    /// task; results drain into the transcript on UI ticks).
    pub(crate) loops: Option<crate::loops::LoopsHandle>,
    /// Current-turn token display: raw user prompt estimate and output across
    /// all model calls, shown in the observability panel.
    pub(crate) usage: (u64, u64),
    pub(crate) usage_estimated: bool,
    /// Current context occupancy (tokens of the last request) and the model's
    /// window, for the live context-fill gauge.
    pub(crate) context_used: u64,
    pub(crate) context_window: Option<u32>,
    /// Latest provider rate-limit buckets observed on a model response.
    pub(crate) rate_limits: Option<hi_ai::RateLimitState>,
    /// Live per-model metadata (window/price/limits) learned from the endpoint's
    /// `/models`, keyed by id — used to apply a model's settings.
    pub(crate) served: HashMap<String, hi_ai::ServedModel>,
    /// The model catalog (ids), for inline `/model <id>` type-ahead completion.
    pub(crate) model_ids: Vec<String>,
    /// MCP endpoint URL (for `/mcp`), if configured for this provider.
    pub(crate) mcp_url: Option<String>,
    /// API key used both for chat and for MCP `/mcp` inspection.
    pub(crate) api_key: String,
    /// How many transcript lines have been trimmed from the top by
    /// [`cap_transcript`]. When > 0, a "↑ N lines compacted" marker shows at
    /// the top of the transcript so it's obvious older content scrolled off.
    pub(crate) trimmed: u64,
    /// Assistant prose currently streaming. Tool output is intentionally not
    /// included; `/copy` copies the assistant's answer, not command logs.
    pub(crate) current_assistant: String,
    /// Inline `/btw` overlay is visible (auto-opens on first side activity;
    /// Esc dismisses it).
    pub(crate) show_btw: bool,
    /// Wrapped-row offset inside a long Done overlay (from the top).
    pub(crate) btw_scroll: usize,
    /// Last painted overlay rect, for mouse-wheel scroll and click-to-dismiss.
    pub(crate) last_btw_area: ratatui::layout::Rect,
    /// `[Esc]` hit target on the overlay's top border.
    pub(crate) last_btw_close: ratatui::layout::Rect,
    /// Live overlay thread: question / thinking / tools / answer. Not part of
    /// the main transcript until the overlay is dismissed.
    pub(crate) btw_thread: Vec<BtwEntry>,
    /// Last completed assistant prose, copied by `/copy`.
    pub(crate) last_assistant: String,
    /// Last event type applied during the active turn, for better fallback
    /// diagnostics when the provider stops without a final turn-end event.
    pub(crate) last_turn_event: Option<TurnEventKind>,
    /// Whether the current/last turn invoked file-editing tools.
    pub(crate) last_turn_had_file_edits: bool,
    /// Files the last turn changed (from `agent.last_changed_files()`), shown
    /// as a compact "changed: …" line above the input so the user always sees
    /// what a turn touched without scrolling the transcript.
    pub(crate) last_changed_files: Vec<String>,
    /// All files touched across the entire session (accumulated from
    /// `last_changed_files` after each turn). Shown by `/files` so a coder can
    /// see at a glance what the session has modified, even while a turn is
    /// running (when the per-turn line is hidden).
    pub(crate) session_changed_files: Vec<String>,
    /// Ghost-text suggestion for the composer. Set from
    /// [`UiEvent::SuggestedPrompt`] after a turn, or from a cheap git heuristic
    /// at session start. Typing a matching prefix shrinks it; Esc on empty
    /// dismisses it until the next suggestion.
    pub(crate) suggested_prompt: Option<String>,
    pub(crate) suggested_prompt_dismissed: bool,
    /// Cached working-tree diff text for the full-screen review overlay.
    pub(crate) diff_text: Option<String>,
    /// Scroll position (line index) within the full-screen diff review overlay.
    pub(crate) review_scroll: usize,
    /// When true, all confirmation requests are auto-approved for the rest of
    /// the session without showing the modal. Set by pressing `a` on an
    /// approval prompt ("always allow this session"). Cleared only by quitting
    /// — it's intentionally session-scoped, not per-turn.
    pub(crate) auto_approve_session: bool,
    /// Path prefixes (normalized) that are auto-approved for file edits this
    /// session. Set by pressing `p` on a file-edit confirmation. Shell mutations
    /// are never covered by path prefixes.
    pub(crate) auto_approve_paths: Vec<String>,
    /// Whether the `Ctrl-?` agent-observability panel is open: telemetry
    /// counters, per-turn tool-call count, and context composition.
    pub(crate) show_debug: bool,
    /// Whether the keybindings help overlay is open (toggled by `?`).
    pub(crate) show_help: bool,
    /// Ctrl-K command palette (fuzzy slash-command launcher).
    pub(crate) palette: Option<crate::palette::CommandPalette>,
    /// Opt-in `/tutorial` modal. Session-local and created fresh on every open.
    pub(crate) tutorial: Option<crate::tutorial::TutorialOverlay>,
    /// Telemetry from the last turn (verify rounds, recovery retries, nudges,
    /// stalls), captured post-turn from `agent.last_turn_telemetry()` for the
    /// observability panel.
    pub(crate) last_telemetry: Option<hi_agent::TurnTelemetry>,
    /// Last-seen [`hi_agent::TurnPhase`] label for the debug panel (updated when
    /// a turn ends, and optionally mid-turn when the agent handle is available).
    pub(crate) last_turn_phase: Option<&'static str>,
    /// Tool calls seen this turn (incremented on each `UiEvent::ToolCall`),
    /// for the observability panel's "tool calls this turn" line.
    pub(crate) turn_tool_calls: u32,
    /// Model rounds seen this turn (incremented on each `UiEvent::AssistantEnd`),
    /// so the activity line can show "round 3 · 5 tool calls" for multi-step turns.
    pub(crate) turn_rounds: u32,
    pub(crate) waiting_for: Option<Duration>,
    pub(crate) last_turn_state: TurnState,
    pub(crate) last_error: Option<String>,
    pub(crate) event_log: Vec<String>,
    pub(crate) model_issues: HashMap<String, u32>,
    pub(crate) startup_notice: Option<String>,
    pub(crate) checkpoint_warning: Option<String>,
    /// A transient "Press Ctrl-C again to exit" notice, shown after the first
    /// Ctrl-C when idle. Cleared after ~1.8s (see the deadline race in the idle
    /// input loop) or when any other key is pressed. A second Ctrl-C while this
    /// is active quits the session.
    pub(crate) quit_notice: Option<Instant>,
    /// Active `/`-command completion menu: the query it's synced to and the
    /// highlighted row. `None` when the input isn't a slash-command prefix.
    pub(crate) completion: Option<CompletionState>,
    /// Cached `git ls-files` output for `@file` path completion, so the menu
    /// doesn't shell out on every keystroke. Refreshed when the path menu opens
    /// (context changes to `Path`); reused while the prefix narrows.
    pub(crate) path_completion_cache: Vec<String>,
    /// Whether the terminal currently has focus (best-effort, via focus-change
    /// reporting). Stays `true` on terminals that don't report it.
    pub(crate) focused: bool,
    /// Set once we've seen any focus event — i.e. the terminal reports focus, so
    /// `focused` is trustworthy.
    pub(crate) focus_known: bool,
    /// Sync configuration for cross-machine session resume. `None` when sync
    /// is not configured (no base_url/api_key). Set from the `--sync` CLI flag
    /// or the `[sync]` config section.
    pub(crate) sync_config: Option<crate::SyncConfig>,
    /// Whether sync is currently active (pushing records + events to ipop).
    pub(crate) sync_active: bool,
    /// The session id used for sync (derived from the local session file stem).
    pub(crate) sync_session_id: Option<String>,
    /// An HTTP client for sync API calls (session list, attach, etc.).
    /// Reused across calls for connection pooling.
    pub(crate) sync_http: Option<reqwest::Client>,
    /// Lists sessions cached on this machine. Provided by hi-cli.
    pub(crate) session_lister: Option<crate::SessionLister>,
    /// Snapshot used while session-id completion is open. Avoids rescanning
    /// and rereading every JSONL file on each render tick.
    pub(crate) session_completion_cache: Vec<crate::LocalSessionInfo>,
    /// Switches the live agent and persistence sink for `/sessions switch <id>`.
    pub(crate) session_switcher: Option<crate::SessionSwitcher>,
    /// Persists names for `/sessions rename <id> <name>`.
    pub(crate) session_renamer: Option<crate::SessionRenamer>,
    /// Enables/disables remote-input host mode for the active session.
    pub(crate) session_host: Option<crate::SessionHostController>,
    /// When set, the open model picker assigns its selection to this team
    /// role (`/team delegate` with no argument) instead of switching the
    /// driver model.
    pub(crate) team_picker_role: Option<String>,
    /// When true, the open picker is the `/team` ROLE menu: selecting a row
    /// opens that role's model picker (or runs auto-setup) instead of
    /// switching the driver model.
    pub(crate) team_role_menu: bool,
    /// Roles waiting behind the single in-flight provisioning slot
    /// (auto-setup wires delegate → editor → explore in sequence; later
    /// entries usually reuse the server the first one started).
    pub(crate) queued_team_assignments: Vec<(String, hi_agent::local_skeptic::ResolvedLocalModel)>,
    /// After auto-setup's queue drains, also point the skeptic gate at the
    /// running team server (free local review).
    pub(crate) auto_setup_skeptic: bool,
    /// In-flight `/team` local-model provisioning (download + server spawn on
    /// a background task). The event loop applies the outcome when it lands;
    /// a 15 GB model fetch must never block the UI.
    pub(crate) pending_team_provision: Option<PendingTeamProvision>,
    /// In-flight managed local driver-provider setup.
    pub(crate) pending_local_provider: Option<PendingLocalProviderProvision>,
    /// In-flight background host-enable (startup auto-host). The controller's
    /// network work (portal registration) runs off the UI path; the event
    /// loop applies the outcome when it completes. A dead portal must never
    /// delay first paint.
    pub(crate) pending_host_enable:
        Option<tokio::task::JoinHandle<anyhow::Result<Option<crate::SessionHostEnable>>>>,
    pub(crate) sync_control: Option<crate::SyncControl>,
    /// The remote event tap for live streaming. When set, the `drive` function
    /// calls this after each `UiEvent` is applied to `App`, forwarding events
    /// to the `RemoteUi` for ipop sync. Set at startup or by `/sync on`.
    pub(crate) remote_event_tap: Option<crate::RemoteEventTap>,
    /// The startup tap exactly as main.rs installed it (it publishes to the
    /// local runtime and the swappable startup RemoteUi slot). `/sync` and
    /// session-switch commands COMPOSE their TUI-local streamer onto this
    /// instead of chaining onto `remote_event_tap`, so cycles can't grow the
    /// chain or orphan RemoteUis, and restoring it is what `/sync off` does.
    pub(crate) base_event_tap: Option<crate::RemoteEventTap>,
    /// A `RemoteUi` created by `/sync on` for mid-session live streaming.
    /// Flushed after each turn and on `/sync off`.
    pub(crate) sync_remote_ui: Option<std::sync::Arc<crate::sync_tui::RemoteUi>>,
    /// A flush callback for the startup `RemoteUi` (created in main.rs). Called
    /// after each turn so live events are actually streamed during the session,
    /// not just buffered until exit. This is a `Box<dyn Fn + Send + Sync>` that
    /// spawns an async flush task internally (since the TUI can't hold a
    /// `hi-cli` type directly).
    pub(crate) remote_flush_callback: Option<crate::RemoteFlushCallback>,
    /// Live receiver of remote attach prompts while host mode is on.
    pub(crate) remote_input_rx: Option<tokio::sync::mpsc::UnboundedReceiver<String>>,
    /// Abort handle for the background remote-input poller (if any).
    pub(crate) remote_input_poller: Option<tokio::task::AbortHandle>,
    /// True while this TUI is advertising `accepts_input` for the active session.
    pub(crate) hosting_remote_input: bool,
    /// When set, typed lines are POSTed to this remote host's input queue
    /// (hosted/steer mode) instead of running on the local agent.
    pub(crate) steering_remote_session: Option<crate::app::SteeringRemote>,
}

impl Drop for App {
    fn drop(&mut self) {
        // Do not detach a potentially multi-gigabyte model download on every
        // exit path. If setup already spawned a server, the run-level local
        // server guard stops it after App is dropped; aborting the task here
        // prevents a late spawn after that guard has run.
        if let Some(pending) = self.pending_team_provision.take() {
            pending.task.abort();
        }
        if let Some(pending) = self.pending_local_provider.take() {
            pending.task.abort();
        }
        if let Some(pending) = self.pending_local_catalog.take() {
            pending.abort();
        }
    }
}

/// Sync configuration passed into the TUI for `/sync`, `/sessions`, `/attach`.
/// Mirrors `hi_cli::sync::SyncConfig` but lives in `hi-tui` so the TUI can
/// make sync API calls without depending on `hi-cli`.
#[derive(Clone, Debug)]
pub struct SyncConfig {
    pub base_url: String,
    pub api_key: String,
    pub machine_id: Option<String>,
    pub cwd_digest: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TurnEventKind {
    Assistant,
    Reasoning,
    AssistantEnd,
    ToolCall,
    ToolResult,
    Status,
    Usage,
    TurnEnd,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum TurnState {
    Idle,
    Running,
    Done(String),
    Warning(String),
    Failed(String),
    Cancelled,
}

/// Hard cap on next-turn prompts in [`App::queue`]. Bounds memory when many
/// lines are submitted mid-turn or remote attach floods the host. Further
/// enqueues are rejected (with a status note) once full — oldest work is kept.
pub(crate) const MAX_PROMPT_QUEUE: usize = 64;

/// Max transcript lines kept for display and scrolling. Older lines scroll off
/// the top (the full session is still in the JSONL log). Bounds the u16 scroll
/// range, the per-frame render clone, and memory on very long sessions.
pub(crate) const MAX_TRANSCRIPT_LINES: usize = 10_000;

/// Max debug-event log entries kept (one per streamed chunk / tool call /
/// status). Read only by `/log`; without a cap it grows unbounded for the life
/// of a long session (hours of streaming push millions of small entries) even
/// though the visible transcript stays bounded. Trimmed oldest-first.
pub(crate) const MAX_EVENT_LOG: usize = 20_000;

/// How densely the transcript renders tool output and explore runs.
/// Cycled by `/density` (and shown in the `?` help under Review & Tools).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Density {
    /// Headers only for tool output; explore runs stay collapsed.
    Compact,
    /// Default: one-line activity rows (Read / Edit / Run); expand on Ctrl-O.
    #[default]
    Comfortable,
    /// Force-expand tool output (same as Ctrl-O on).
    Verbose,
}

impl Density {
    pub(crate) fn next(self) -> Self {
        match self {
            Self::Compact => Self::Comfortable,
            Self::Comfortable => Self::Verbose,
            Self::Verbose => Self::Compact,
        }
    }

    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Compact => "compact",
            Self::Comfortable => "comfortable",
            Self::Verbose => "verbose",
        }
    }

    pub(crate) fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "compact" | "c" => Some(Self::Compact),
            "comfortable" | "default" | "normal" => Some(Self::Comfortable),
            "verbose" | "full" | "v" => Some(Self::Verbose),
            _ => None,
        }
    }

    /// Effective "show full tool output" for flatten — verbose density forces it.
    pub(crate) fn show_tool_output(self, global_toggle: bool) -> bool {
        match self {
            Self::Verbose => true,
            Self::Compact | Self::Comfortable => global_toggle,
        }
    }

    /// Preview line budget before folding a tool-output block.
    pub(crate) fn tool_preview_lines(self) -> usize {
        match self {
            Self::Compact => 0,
            Self::Comfortable => TOOL_OUTPUT_PREVIEW_LINES,
            Self::Verbose => usize::MAX / 4,
        }
    }
}

#[cfg(test)]
mod tests;
