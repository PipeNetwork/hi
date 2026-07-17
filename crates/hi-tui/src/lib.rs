//! Full-screen terminal UI for `hi`.
//!
//! A ratatui application on the alternate screen: a bordered, scrollable
//! conversation transcript with a title/status bar, and an input box with a
//! "working" spinner. The agent runs behind an mpsc channel ([`ChannelUi`]) so
//! the event loop can keep redrawing — spinner, streaming output, scrolling —
//! while a turn is in flight, and can cancel it with Ctrl-C.

mod activity;
mod app;
mod daemon;
mod dashboard;
mod dashboard_goal;
mod lock;
mod loops;
mod notify;
pub use app::run;
pub use daemon::run_loops_daemon;
mod completion;
pub mod event;
mod input;
mod model_picker;
mod provider_form;
mod render;
mod sync_tui;
mod theme;
mod util;
mod watch;

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use hi_agent::{Agent, AgentStateSnapshot};
use ratatui::style::{Color, Modifier, Style};
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
}

/// Result of saving/selecting a managed local MLX profile.
pub struct MlxProfileSwitch {
    pub switched: SwitchedProvider,
    pub profiles: Vec<ProfileInfo>,
}

/// A callback that resolves a named profile into a built provider + model +
/// label, for `/provider` mid-session. `hi-cli` supplies this; the TUI calls
/// it without needing to know about `Config`/`Settings` (which live in
/// `hi-cli`).
pub type ProfileResolver = Box<dyn Fn(&str) -> Result<SwitchedProvider> + Send + Sync>;

/// Everything the `/dashboard` fleet needs to launch worktree-isolated child
/// `hi` runs: the binary + provider wiring for the child command line, the
/// verify pipeline for the merge gate, and a session-path allocator. `hi-cli`
/// supplies this so the TUI never touches `Settings`/session paths directly.
pub struct FleetLauncher {
    /// The `hi` binary to spawn for each row turn.
    pub exe: std::path::PathBuf,
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

use completion::CompletionState;
use input::{HistorySearch, InputLine};
use model_picker::ModelPicker;
use render::{dim, line_text};

pub(crate) const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
/// How many model rows the `/model` picker shows at once.
pub(crate) const PICKER_ROWS: usize = 12;

/// A synchronous, plain (uncolored) `git diff` of the working tree, for the
/// `Ctrl-D` diff panel. The TUI applies its own highlighting via `diff_lines`,
/// so we want the raw diff without ANSI codes. Returns empty when not a git
/// repo or there are no changes. Synchronous because the key handler isn't
/// async and `git diff` is fast/user-initiated.
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

/// The "PipeNetwork.AI" wordmark rendered as figlet-style 5-row block letters.
/// All orange, ~2x the height of a normal line — the splash centerpiece.
/// Generated from `figlet -f small`, then hand-trimmed of trailing whitespace.
const BANNER: [&str; 5] = [
    " ___ _           _  _     _                  _       _   ___ ",
    "| _ (_)_ __  ___| \\| |___| |___ __ _____ _ _| |__   /_\\ |_ _|",
    "|  _/ | '_ \\/ -_) .` / -_)  _\\ V  V / _ \\ '_| / /_ / _ \\ | | ",
    "|_| |_| .__/\\___|_|\\_\\___|\\__|\\_/\\_/\\___/_| |_\\_(_)_/ \\_\\___|",
    "      |_|                                                    ",
];

/// Build the PipeNetwork.AI landing banner as styled lines, pushed as the
/// first transcript entries on startup. The full "PipeNetwork.AI" wordmark is
/// rendered ~2x size as a 5-row block-letter banner (all orange), followed by
/// the dim model line and the working directory. A blank line of breathing
/// room follows before the usage hint. Sits at the top of the transcript and
/// scrolls up naturally as you work.
pub(crate) fn splash_lines(
    provider: &str,
    model: &str,
    context_window: Option<u32>,
) -> Vec<Line<'static>> {
    let orange = Style::default().fg(Color::Rgb(255, 140, 0));
    let bold_orange = orange.add_modifier(Modifier::BOLD);
    let dim = dim();

    let ctx = context_window
        .map(|w| format!(" ({}K context)", w / 1000))
        .unwrap_or_default();
    let cwd = std::env::current_dir()
        .map(|d| d.display().to_string())
        .unwrap_or_else(|_| "?".into());

    // The 5-row block-letter banner, all orange + bold.
    let mut lines: Vec<Line<'static>> = BANNER
        .iter()
        .map(|row| Line::from(vec![Span::styled((*row).to_string(), bold_orange)]))
        .collect();

    // Dim model line + working directory below the banner.
    lines.push(Line::from(vec![Span::styled(
        format!("{model}{ctx} · {provider}"),
        dim,
    )]));
    lines.push(Line::from(vec![Span::styled(cwd, dim)]));

    // A blank line for breathing room before the usage hint.
    lines.push(Line::raw(""));
    lines
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
    /// visible output always shows which request it belongs to.
    UserPrompt(Line<'static>),
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

/// How many lines of a long tool-output block show before it folds to a preview.
pub(crate) const TOOL_OUTPUT_PREVIEW_LINES: usize = 16;

/// A run of consecutive same-tool exploration results (read/list/grep) being
/// collapsed into one transcript line, so a burst of reads renders as
/// `⏺ read 6 files · 743 lines` instead of six separate lines.
#[derive(Clone, Debug)]
pub(crate) struct ExploreRun {
    /// The tool name (`read`/`list`/`grep`).
    pub tool: String,
    /// How many results have been folded into this run.
    pub count: u32,
    /// Total lines across all folded results.
    pub lines: u32,
    /// Whether every result so far was empty (`(no output)`).
    pub all_empty: bool,
    /// Absolute transcript position (`trimmed` + local index) of this run's
    /// summary line. Merging is only valid while that line is still the *last*
    /// transcript entry — otherwise the in-place update would overwrite
    /// whatever landed after it (e.g. committed assistant prose).
    pub line_pos: u64,
}

impl TranscriptEntry {
    /// Flatten this entry into display lines under the current fold settings.
    /// A collapsed reasoning block is one dim summary line; a long tool-output
    /// block shows a preview plus a fold footer unless `show_tool_output` is on.
    pub(crate) fn flatten(
        &self,
        show_reasoning: bool,
        show_tool_output: bool,
    ) -> Vec<Line<'static>> {
        let th = crate::theme::theme();
        match self {
            TranscriptEntry::Line(line) | TranscriptEntry::UserPrompt(line) => vec![line.clone()],
            TranscriptEntry::ChangedFiles { line, .. } => vec![line.clone()],
            TranscriptEntry::Reasoning { text, elapsed } => {
                let secs = elapsed.as_secs();
                let label = if secs >= 60 {
                    format!("{}m {:02}s", secs / 60, secs % 60)
                } else {
                    format!("{secs}s")
                };
                if show_reasoning {
                    let mut lines = vec![Line::styled(
                        format!("⏺ thought for {label} (Ctrl-T to collapse)"),
                        Style::default().fg(th.accent_thinking),
                    )];
                    for line in text.lines() {
                        lines.push(Line::styled(
                            format!("  {line}"),
                            Style::default().fg(th.gray_dim),
                        ));
                    }
                    lines
                } else {
                    vec![Line::styled(
                        format!("⏺ thought for {label}  (Ctrl-T to expand)",),
                        Style::default().fg(th.accent_thinking),
                    )]
                }
            }
            TranscriptEntry::ToolOutput { body, expanded } => {
                // The visible body lines sit in a sunken panel (a `panel` base
                // background) on truecolor themes, tagging them so the render
                // pass can pad them to full width. The fold footer stays plain
                // so the fold boundary reads as the panel's edge.
                let panel = th.panel;
                let tag = |line: &Line<'static>| -> Line<'static> {
                    let mut l = line.clone();
                    if th.paints_backgrounds() {
                        l.style = l.style.bg(panel);
                    }
                    l
                };
                // Short output, the global expand toggle, or this block's own
                // expand override shows in full; otherwise a preview + a fold
                // footer naming what's hidden.
                if show_tool_output || *expanded || body.len() <= TOOL_OUTPUT_PREVIEW_LINES {
                    body.iter().map(tag).collect()
                } else {
                    let hidden = body.len() - TOOL_OUTPUT_PREVIEW_LINES;
                    let mut lines: Vec<Line<'static>> =
                        body[..TOOL_OUTPUT_PREVIEW_LINES].iter().map(tag).collect();
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
            | TranscriptEntry::UserPrompt(line)
            | TranscriptEntry::ChangedFiles { line, .. } => line_text(line),
            TranscriptEntry::Reasoning { text, .. } => text.clone(),
            TranscriptEntry::ToolOutput { body, .. } => {
                body.iter().map(line_text).collect::<Vec<_>>().join("\n")
            }
        }
    }
}

pub(crate) struct App {
    pub(crate) provider: String,
    pub(crate) model: String,
    /// Explicit workspace root copied from the agent runtime for synchronous
    /// frontend-only operations such as the Ctrl-D diff panel.
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
    /// Saves/selects a managed local MLX profile after `/hf run --mlx`.
    pub(crate) mlx_switcher: MlxProfileSwitcher,
    pub(crate) transcript: Vec<TranscriptEntry>,
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
    /// toggles this to reveal every block's full body.
    pub(crate) show_tool_output: bool,
    /// Block-navigation mode (Ctrl-B): a cursor moves over tool-output blocks so
    /// individual ones can be folded/unfolded, independent of the global Ctrl-O.
    /// While on, ↑/↓ (or j/k) move the cursor, Enter/Space fold the selected
    /// block, and the view follows the cursor.
    pub(crate) nav_mode: bool,
    /// The selected block's ordinal among tool-output blocks (0-based, oldest
    /// first). Clamped to the current block count wherever it's used.
    pub(crate) block_cursor: usize,
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
    /// Wrapped-line total at the moment the view last left the bottom — drives
    /// the "↓ N new" indicator while scrolled up.
    pub(crate) total_when_unpinned: u16,
    pub(crate) working: bool,
    pub(crate) spinner: usize,
    /// When the current turn started, for the elapsed-time readout.
    pub(crate) started: Option<Instant>,
    /// When the last turn finished (working true→false), for the brief accent
    /// "finish flash" on the status line. Cleared implicitly once its window
    /// elapses (the flash weight decays to zero).
    pub(crate) finished_at: Option<Instant>,
    /// The tool currently executing (its display label) and when it started, so
    /// the working line can name the in-flight action with its own timer. `None`
    /// while the model — not a tool — is the active party.
    pub(crate) current_tool: Option<String>,
    pub(crate) current_tool_started: Option<Instant>,
    /// For read/list/grep we defer the `⏺` header until the result lands, so the
    /// file name and line count collapse into one transcript line instead of two.
    pub(crate) pending_explore_label: Option<String>,
    /// A run of consecutive same-tool explore results being collapsed into one
    /// transcript line. `None` when the last transcript line isn't an explore
    /// result (or a new run hasn't started). Reset by any non-explore event.
    pub(crate) explore_run: Option<ExploreRun>,
    /// Lines typed while a turn was running, to run once it finishes (FIFO).
    pub(crate) queue: VecDeque<String>,
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
    /// Ctrl-R reverse-search over input history. When active, keystrokes go to
    /// the search filter instead of the input line.
    pub(crate) history_search: Option<HistorySearch>,
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
    /// The active long-horizon goal, mirrored from the agent so the pinned plan
    /// block and header can show sub-goal progress. Refreshed when `/goal` sets it
    /// and after every turn (the driver may advance it). `None` when no goal is set.
    pub(crate) goal: Option<hi_agent::Goal>,
    /// Consecutive auto-drive turns that left the goal state unchanged. At
    /// [`hi_agent::GOAL_DRIVE_STALL_LIMIT`] the drive stops queuing itself (the
    /// goal stays active); any user turn resets it.
    pub(crate) goal_drive_stall: u32,
    /// The `/dashboard` fleet: dispatched agents (one session each), persisted
    /// across dashboard open/close so rows aren't lost when you drop back to
    /// the chat. In-flight turns live only inside the dashboard loop.
    pub(crate) fleet: Vec<crate::dashboard::FleetRow>,
    /// Monotonic display id for fleet rows (never reused within a session).
    pub(crate) fleet_next_id: usize,
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
    /// Whether the `Ctrl-D` diff panel is open (a full working-tree diff pinned
    /// above the input, rendered with the same highlighting as tool-output diffs).
    pub(crate) show_diff: bool,
    /// Cached working-tree diff text for the open diff panel, refreshed when the
    /// panel is toggled on so it reflects the current tree, not a stale snapshot.
    pub(crate) diff_text: Option<String>,
    /// Whether the full-screen diff review overlay is open (Ctrl-G). A
    /// scrollable, syntax-colored view of the entire working-tree diff with
    /// hunk-to-hunk navigation (n/p). Takes over the whole screen until closed.
    pub(crate) show_review: bool,
    /// Scroll position (line index) within the full-screen diff review overlay.
    pub(crate) review_scroll: usize,
    /// When true, all confirmation requests are auto-approved for the rest of
    /// the session without showing the modal. Set by pressing `a` on an
    /// approval prompt ("always allow this session"). Cleared only by quitting
    /// — it's intentionally session-scoped, not per-turn.
    pub(crate) auto_approve_session: bool,
    /// Whether the `Ctrl-?` agent-observability panel is open: telemetry
    /// counters, per-turn tool-call count, and context composition.
    pub(crate) show_debug: bool,
    /// Whether the keybindings help overlay is open (toggled by `?`).
    pub(crate) show_help: bool,
    /// Telemetry from the last turn (verify rounds, recovery retries, nudges,
    /// stalls), captured post-turn from `agent.last_turn_telemetry()` for the
    /// observability panel.
    pub(crate) last_telemetry: Option<hi_agent::TurnTelemetry>,
    /// Tool calls seen this turn (incremented on each `UiEvent::ToolCall`),
    /// for the observability panel's "tool calls this turn" line.
    pub(crate) turn_tool_calls: u32,
    /// Model rounds seen this turn (incremented on each `UiEvent::AssistantEnd`),
    /// so the activity line can show "round 3 · 5 tool calls" for multi-step turns.
    pub(crate) turn_rounds: u32,
    /// A tail of recent streamed tool output lines (e.g. bash stdout), shown
    /// live in the working area while a tool runs. Cleared when the tool
    /// finishes and its final result is pushed to the transcript.
    pub(crate) tool_stream_tail: Vec<String>,
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
    pub(crate) sync_control: Option<crate::SyncControl>,
    /// The remote event tap for live streaming. When set, the `drive` function
    /// calls this after each `UiEvent` is applied to `App`, forwarding events
    /// to the `RemoteUi` for ipop sync. Set at startup or by `/sync on`.
    pub(crate) remote_event_tap: Option<crate::RemoteEventTap>,
    /// A `RemoteUi` created by `/sync on` for mid-session live streaming.
    /// Flushed after each turn and on `/sync off`.
    pub(crate) sync_remote_ui: Option<std::sync::Arc<crate::sync_tui::RemoteUi>>,
    /// A flush callback for the startup `RemoteUi` (created in main.rs). Called
    /// after each turn so live events are actually streamed during the session,
    /// not just buffered until exit. This is a `Box<dyn Fn + Send + Sync>` that
    /// spawns an async flush task internally (since the TUI can't hold a
    /// `hi-cli` type directly).
    pub(crate) remote_flush_callback: Option<crate::RemoteFlushCallback>,
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

/// Max transcript lines kept for display and scrolling. Older lines scroll off
/// the top (the full session is still in the JSONL log). Bounds the u16 scroll
/// range, the per-frame render clone, and memory on very long sessions.
pub(crate) const MAX_TRANSCRIPT_LINES: usize = 10_000;

/// Max debug-event log entries kept (one per streamed chunk / tool call /
/// status). Read only by `/log`; without a cap it grows unbounded for the life
/// of a long session (hours of streaming push millions of small entries) even
/// though the visible transcript stays bounded. Trimmed oldest-first.
pub(crate) const MAX_EVENT_LOG: usize = 20_000;

#[cfg(test)]
mod tests;
