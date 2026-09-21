//! Pipe Network coding-agent harness.
//!
//! One loop: stream from `api.pipenetwork.ai`, execute local tools, repeat.

mod command;
mod compact;
mod completion;
mod dashboard;
mod jev_auto;
mod jev_compact;
mod live;
mod managed;
pub use managed::{ManagedInspection, ManagedSettings, inspect_managed_journal};
mod liveness;
mod pipe;
mod prompt;
mod review;
mod review_citations;
mod review_drive;
mod review_guard;
mod review_prompts;
mod review_report;
mod review_scope;
mod review_stream;
mod session;
mod session_index;
mod session_lease;
mod tools;
mod turn;
mod typesafe;
mod ui;
mod usage;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use hi_ai::{Message, ServedModel, Usage, effective_coding_agent_max_tokens};
use hi_tools::{PlanStep, checkpoint};
use tokio::sync::Notify;

pub use command::{
    COMMANDS, CORE_COMMANDS, Command, CommandSpec, EffortArg, ModelArgs, parse as parse_command,
    parse_effort_arg, parse_model_args, resolve_model_query, trust_command,
};
pub use dashboard::{
    Dashboard, DashboardKnobs, DispatchOpts, PIPE_GPT6, RowState, RowView, is_openai_model,
    normalize_row_model, split_model_prefix,
};
pub use hi_ai::ReasoningEffort;
pub use live::LiveSettings;
pub use liveness::AwaitingUserGuard;
pub use pipe::{
    DEFAULT_BASE_URL, DEFAULT_MAX_TOKENS, DEFAULT_MODEL, PipeClient, PipeError, default_base_url,
};
pub use prompt::SYSTEM_PROMPT;
pub use review::{
    ChecklistItem, CoverageRow, CoverageState, DEFAULT_REVIEW_PASSES, Finding, GitScope, GitSource,
    InputKind, REVIEW_DEFECTS_FORMAT_BLOCK, REVIEW_FORMAT_BLOCK, REVIEW_FORMAT_HINT, REVIEW_PREFIX,
    ReviewAction, ReviewArgs, ReviewInput, ReviewInputs, ReviewVerdict, Severity, check_citations,
    discover_inputs, fingerprint as review_fingerprint, parse_checklist,
    take_all as review_take_all, transcript_label as review_transcript_label,
};
pub use review_drive::{ReviewCommand, ReviewDrive, ReviewPhase, ReviewStep, ReviewTurnFacts};
pub const CONTINUE_PLAN_PROMPT: &str = "Continue the open plan.";
pub use completion::plan_is_open;
pub use session::{JsonlSession, LoadedSession, PendingTurn, PlanDrive, UserTurn, list_user_turns};
pub use session_index::{
    SessionIndex, index_path, is_harness_injection, load_or_refresh_index, session_id_from_path,
};
pub use session_lease::{
    SessionBusy, SessionLockStatus, inspect_session_lock, lock_path as session_lock_path,
};
pub use tools::{ToolHost, ToolInterrupt, advertised_tools};
pub use typesafe::{NextAction, TypesafeSettings};
pub use usage::{
    BILLING_URL, UsageCategory, UsageSnapshot, UsageTab, fmt_tokens, occupancy_bar,
    open_billing_url,
};

/// Default Pipe context window used for occupancy display and auto-compact.
pub const DEFAULT_CONTEXT_WINDOW: u32 = 128_000;
pub use ui::{
    AutoHint, ConfirmationFuture, ConfirmationRequest, ConfirmationResult, PermissionMode, TestUi,
    Ui,
};

use pipe::PipeClient as Client;
use session::JsonlSession as SessionFile;

/// Cloneable cancellation flag for an in-flight turn.
#[derive(Clone)]
pub struct TurnCancellation {
    flag: Arc<AtomicBool>,
    notify: Arc<Notify>,
}

impl Default for TurnCancellation {
    fn default() -> Self {
        Self::new()
    }
}

impl TurnCancellation {
    pub fn new() -> Self {
        Self {
            flag: Arc::new(AtomicBool::new(false)),
            notify: Arc::new(Notify::new()),
        }
    }

    pub fn cancel(&self) {
        self.flag.store(true, Ordering::Release);
        self.notify.notify_waiters();
    }

    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Acquire)
    }

    pub async fn cancelled(&self) {
        loop {
            let notified = self.notify.notified();
            if self.is_cancelled() {
                return;
            }
            notified.await;
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TurnStopReason {
    Completed,
    Cancelled,
    Error,
}

#[derive(Clone, Debug)]
pub struct TurnOutcome {
    pub stop_reason: TurnStopReason,
    pub usage: Usage,
    pub changed_files: Vec<String>,
    pub error: Option<String>,
    /// `passed` / `failed` after `/verify`; `None` when no check ran.
    pub verification: Option<String>,
}

#[derive(Clone, Debug)]
pub struct SessionSnapshot {
    pub messages: Vec<Message>,
    pub plan: Vec<PlanStep>,
    pub usage: Usage,
    pub last_changed_files: Vec<String>,
}

pub struct HarnessConfig {
    pub managed: ManagedSettings,
    pub workspace_root: PathBuf,
    pub state_root: PathBuf,
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub max_tokens: u32,
    /// True when `max_tokens` came from `--max-tokens` or a profile override.
    /// Implicit Pipe coding turns then use the advertised `/models` output cap.
    pub max_tokens_explicit: bool,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub session_path: Option<PathBuf>,
    pub liveness: Option<hi_liveness::Publisher>,
    /// When false, the turn loop skips cheap shrink and model auto-compact.
    /// `/compact` still runs.
    pub auto_compact: bool,
    /// Session-only Jev tool prune. Requires a TypeSafe key to take effect.
    pub jev_compact: bool,
    /// Optional TypeSafe (Jev) next-action gate. Off when no API key is set.
    pub typesafe: TypesafeSettings,
}

impl HarnessConfig {
    pub fn pipe(workspace_root: PathBuf, api_key: impl Into<String>) -> Self {
        let state_root = workspace_root.join(".hi");
        Self {
            managed: ManagedSettings::default(),
            workspace_root,
            state_root,
            api_key: api_key.into(),
            base_url: default_base_url(),
            model: DEFAULT_MODEL.to_string(),
            max_tokens: DEFAULT_MAX_TOKENS,
            max_tokens_explicit: false,
            reasoning_effort: None,
            session_path: None,
            liveness: None,
            auto_compact: true,
            jev_compact: false,
            typesafe: TypesafeSettings::disabled(),
        }
    }
}

pub struct Harness {
    client: Client,
    tools: ToolHost,
    workspace_root: PathBuf,
    state_root: PathBuf,
    live: LiveSettings,
    configured_max_tokens: u32,
    max_tokens_explicit: bool,
    messages: Vec<Message>,
    plan: Vec<PlanStep>,
    plan_drive: PlanDrive,
    session_usage: Usage,
    checkpoints: Vec<String>,
    last_changed_files: Vec<String>,
    last_context_occupancy: u64,
    context_window: u32,
    model_windows: HashMap<String, u32>,
    model_output_caps: HashMap<String, u32>,
    auto_compact: bool,
    jev_compact: bool,
    compact_suppressed: bool,
    verify_command: Option<String>,
    session: Option<SessionFile>,
    interrupt: Arc<AtomicBool>,
    turn_cancel: Option<TurnCancellation>,
    steer: SteerQueue,
    pending_turn: Option<PendingTurn>,
    turn_index: u32,
    turn_open: bool,
    turn_oneshot: bool,
    turn_plain: bool,
    liveness: hi_liveness::Publisher,
    typesafe: TypesafeSettings,
    next_action: typesafe::NextActionSource,
    /// Turn-scoped Jev effort. Never persisted.
    turn_effort: Option<ReasoningEffort>,
    /// Intent for the next turn, armed by the review drive and consumed once
    /// at turn start. Never persisted.
    forced_intent: Option<completion::Intent>,
    review_drive: ReviewDrive,
}

impl Harness {
    pub fn new(config: HarnessConfig) -> Result<Self> {
        let mut tools = ToolHost::new(config.workspace_root.clone(), config.state_root.clone())?;
        let session = open_session(config.session_path.as_deref(), &config.model)?;
        let liveness = resolve_liveness(&config, &tools);
        tools.set_liveness(liveness.clone());
        let workspace = config.workspace_root.display().to_string();
        liveness.set_workspace(workspace);
        if let Some(session) = &session {
            liveness.set_session_path(Some(session.path().display().to_string()));
        }
        liveness.set_state(hi_liveness::HarnessState::Idle);
        Ok(Self {
            client: Client::new(config.base_url, config.api_key).with_managed(
                config
                    .session_path
                    .as_ref()
                    .map(|p| p.with_extension("managed.json"))
                    .unwrap_or_else(|| config.state_root.join("managed-turn.json")),
                config.managed,
            ),
            tools,
            workspace_root: config.workspace_root,
            state_root: config.state_root,
            live: LiveSettings::new(config.model, PermissionMode::Ask, config.reasoning_effort),
            configured_max_tokens: config.max_tokens.max(1),
            max_tokens_explicit: config.max_tokens_explicit,
            messages: Vec::new(),
            plan: Vec::new(),
            plan_drive: PlanDrive::default(),
            session_usage: Usage::default(),
            checkpoints: Vec::new(),
            last_changed_files: Vec::new(),
            last_context_occupancy: 0,
            context_window: DEFAULT_CONTEXT_WINDOW,
            model_windows: HashMap::new(),
            model_output_caps: HashMap::new(),
            auto_compact: config.auto_compact,
            jev_compact: config.jev_compact && config.typesafe.is_enabled(),
            compact_suppressed: false,
            verify_command: None,
            session,
            interrupt: Arc::new(AtomicBool::new(false)),
            turn_cancel: None,
            steer: SteerQueue::default(),
            pending_turn: None,
            turn_index: 0,
            turn_open: false,
            turn_oneshot: false,
            turn_plain: false,
            liveness,
            typesafe: config.typesafe.clone(),
            next_action: config.typesafe.source(),
            turn_effort: None,
            forced_intent: None,
            review_drive: ReviewDrive::default(),
        })
    }

    /// Test/embedded constructor with a caller-owned process runner (sandbox off).
    pub fn new_with_tools(config: HarnessConfig, mut tools: ToolHost) -> Result<Self> {
        let session = open_session(config.session_path.as_deref(), &config.model)?;
        let liveness = resolve_liveness(&config, &tools);
        tools.set_liveness(liveness.clone());
        liveness.set_workspace(config.workspace_root.display().to_string());
        if let Some(session) = &session {
            liveness.set_session_path(Some(session.path().display().to_string()));
        }
        liveness.set_state(hi_liveness::HarnessState::Idle);
        Ok(Self {
            client: Client::new(config.base_url, config.api_key).with_managed(
                config
                    .session_path
                    .as_ref()
                    .map(|p| p.with_extension("managed.json"))
                    .unwrap_or_else(|| config.state_root.join("managed-turn.json")),
                config.managed,
            ),
            tools,
            workspace_root: config.workspace_root,
            state_root: config.state_root,
            live: LiveSettings::new(
                config.model,
                PermissionMode::Always,
                config.reasoning_effort,
            ),
            configured_max_tokens: config.max_tokens.max(1),
            max_tokens_explicit: config.max_tokens_explicit,
            messages: Vec::new(),
            plan: Vec::new(),
            plan_drive: PlanDrive::default(),
            session_usage: Usage::default(),
            checkpoints: Vec::new(),
            last_changed_files: Vec::new(),
            last_context_occupancy: 0,
            context_window: DEFAULT_CONTEXT_WINDOW,
            model_windows: HashMap::new(),
            model_output_caps: HashMap::new(),
            auto_compact: config.auto_compact,
            jev_compact: config.jev_compact && config.typesafe.is_enabled(),
            compact_suppressed: false,
            verify_command: None,
            session,
            interrupt: Arc::new(AtomicBool::new(false)),
            turn_cancel: None,
            steer: SteerQueue::default(),
            pending_turn: None,
            turn_index: 0,
            turn_open: false,
            turn_oneshot: false,
            turn_plain: false,
            liveness,
            typesafe: config.typesafe.clone(),
            next_action: config.typesafe.source(),
            turn_effort: None,
            forced_intent: None,
            review_drive: ReviewDrive::default(),
        })
    }

    pub fn apply_loaded_session(&mut self, loaded: LoadedSession) {
        self.messages = loaded.messages;
        self.session_usage = loaded.usage;
        self.checkpoints = loaded.checkpoints;
        self.verify_command = loaded.verify_command;
        if let Some(model) = loaded.model {
            self.live.set_model(model);
            self.apply_cached_window();
        }
        if let Some(permission) = loaded.permission {
            self.live
                .set_permission_mode(PermissionMode::from_u8(permission));
        }
        if let Some(effort) = loaded.effort.as_deref() {
            self.pin_user_effort();
            match crate::parse_effort_arg(effort) {
                Ok(crate::EffortArg::Off) => self.live.set_reasoning_effort(None),
                Ok(crate::EffortArg::Level(level)) => self.live.set_reasoning_effort(Some(level)),
                Err(_) => {}
            }
        }
        self.pending_turn = loaded.pending_turn;
        self.plan_drive = loaded.plan_drive;
        self.review_drive = loaded.review_drive;
        self.plan = if loaded.plan.is_empty() {
            crate::completion::plan_from_messages(&self.messages)
        } else {
            loaded.plan
        };
        if let Some(pending) = &self.pending_turn {
            self.turn_index = pending.turn_index;
            self.liveness.set_turn_index(pending.turn_index);
            self.liveness
                .set_pre_checkpoint(pending.pre_checkpoint.clone());
        }
    }

    fn snapshot_state(&self) -> LoadedSession {
        LoadedSession {
            messages: self.messages.clone(),
            usage: self.session_usage,
            checkpoints: self.checkpoints.clone(),
            verify_command: self.verify_command.clone(),
            model: Some(self.model()),
            name: None,
            permission: Some(self.permission_mode().as_u8()),
            effort: self
                .reasoning_effort()
                .map(|effort| effort.as_str().to_string()),
            pending_turn: self.pending_turn.clone(),
            plan: self.plan.clone(),
            plan_drive: self.plan_drive.clone(),
            review_drive: self.review_drive.clone(),
        }
    }

    /// Rewrite the session file to match in-memory state.
    ///
    /// Returns true when there is no session file, or when the rewrite
    /// succeeded. False means the file is unchanged: callers that already
    /// replaced `messages` must restore the previous history so the persist
    /// cursor cannot run ahead of the file.
    ///
    /// Do not report `SessionAppendFailed` here. That invariant auto-repairs
    /// and would kill a turn that just rolled back to a consistent snapshot.
    /// Close still reports if the final append cannot be written.
    #[must_use]
    pub(crate) fn persist_snapshot(&mut self) -> bool {
        let state = self.snapshot_state();
        let Some(session) = &mut self.session else {
            return true;
        };
        if session.rewrite(&state).is_err() {
            return false;
        }
        self.liveness.note_progress();
        self.liveness
            .emit(hi_liveness::EventCode::SessionRewrite, None, None, None);
        true
    }

    /// Replace live history and rewrite the session. On rewrite failure,
    /// restore the previous messages and occupancy so later appends still
    /// line up with the file.
    #[must_use]
    pub(crate) fn replace_messages_persisting(&mut self, next: Vec<Message>) -> bool {
        let previous = std::mem::replace(&mut self.messages, next);
        let prev_occ = self.last_context_occupancy;
        self.last_context_occupancy = crate::compact::estimate_message_tokens(&self.messages);
        self.session_usage.context_occupancy = self.last_context_occupancy;
        if self.persist_snapshot() {
            true
        } else {
            self.messages = previous;
            self.last_context_occupancy = prev_occ;
            self.session_usage.context_occupancy = prev_occ;
            false
        }
    }

    /// Drop messages after `len` and rewrite the session file.
    pub fn truncate_messages(&mut self, len: usize) {
        if len < self.messages.len() {
            let previous = self.messages.clone();
            let prev_pending = self.pending_turn.clone();
            self.messages.truncate(len);
            self.pending_turn = None;
            if !self.persist_snapshot() {
                self.messages = previous;
                self.pending_turn = prev_pending;
            }
        }
    }

    pub fn liveness(&self) -> hi_liveness::Publisher {
        self.liveness.clone()
    }

    pub fn pending_turn(&self) -> Option<&PendingTurn> {
        self.pending_turn.as_ref()
    }

    /// Fail closed unless the in-flight user line is still in the session.
    /// Trailing assistant/tool (or steered user) messages after that line are
    /// mid-turn progress and must not block resume. Unmatched `tool_call`s
    /// (crash mid-round) are not a valid Pipe transcript, so those fail closed.
    pub fn can_resume_incomplete(&self, expected_prompt: Option<&str>) -> bool {
        if self.pending_turn.is_none() {
            return false;
        }
        let Some(idx) = self.messages.iter().rposition(|message| {
            message.role == hi_ai::Role::User
                && expected_prompt.is_none_or(|expected| message.text() == expected)
        }) else {
            return false;
        };
        tool_results_match_calls(&self.messages[idx..])
    }

    pub fn set_turn_intent_mode(&mut self, oneshot: bool, plain: bool) {
        self.turn_oneshot = oneshot;
        self.turn_plain = plain;
    }

    #[cfg(test)]
    pub(crate) fn turn_intent_mode(&self) -> (bool, bool) {
        (self.turn_oneshot, self.turn_plain)
    }

    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    pub fn workspace_root(&self) -> &Path {
        &self.workspace_root
    }

    pub fn interrupt_handle(&self) -> Arc<AtomicBool> {
        self.interrupt.clone()
    }

    pub fn live(&self) -> LiveSettings {
        self.live.clone()
    }

    pub fn permission_mode(&self) -> PermissionMode {
        self.live.permission_mode()
    }

    pub fn set_permission_mode(&mut self, mode: PermissionMode) {
        self.live.set_permission_mode(mode);
        self.persist_knobs();
    }

    pub fn current_plan(&self) -> &[PlanStep] {
        &self.plan
    }

    pub fn plan_drive(&self) -> &PlanDrive {
        &self.plan_drive
    }

    /// Drop a fully finished checklist so the next prompt does not keep
    /// showing last turn's todos. In-progress plans stay put.
    pub(crate) fn dismiss_completed_plan(&mut self) -> bool {
        if !PlanStep::all_complete(&self.plan) {
            return false;
        }
        self.plan.clear();
        true
    }

    pub fn last_changed_files(&self) -> &[String] {
        &self.last_changed_files
    }

    pub fn model(&self) -> String {
        self.live.model()
    }

    pub fn set_model(&mut self, model: String) {
        if let Some(session) = &mut self.session {
            let _ = session.record_model(&model);
        }
        self.live.set_model(model);
        self.apply_cached_window();
    }

    pub fn reasoning_effort(&self) -> Option<ReasoningEffort> {
        self.live.reasoning_effort()
    }

    pub fn set_reasoning_effort(&mut self, effort: Option<ReasoningEffort>) {
        self.pin_user_effort();
        self.live.set_reasoning_effort(effort);
        self.persist_knobs();
    }

    pub fn apply_effort_arg(&mut self, effort: EffortArg) {
        self.pin_user_effort();
        self.live.apply_effort_arg(effort);
        self.persist_knobs();
    }

    fn persist_knobs(&mut self) {
        if let Some(session) = &mut self.session {
            let _ = session.record_knobs(
                self.live.permission_mode().as_u8(),
                self.live.reasoning_effort().map(|effort| effort.as_str()),
            );
        }
    }

    pub fn persist_live_knobs(&mut self) {
        self.persist_knobs();
    }

    pub fn set_auto_compact(&mut self, enabled: bool) {
        self.auto_compact = enabled;
    }

    pub fn auto_compact(&self) -> bool {
        self.auto_compact
    }

    pub fn typesafe_key_present(&self) -> bool {
        self.typesafe.is_enabled()
    }

    pub fn jev_compact(&self) -> bool {
        self.jev_compact
    }

    pub fn jev_compact_ready(&self) -> bool {
        self.jev_compact && self.typesafe.is_enabled()
    }

    pub fn jev_compact_status_line(&self) -> String {
        let on = if self.jev_compact { "on" } else { "off" };
        let key = if self.typesafe.is_enabled() {
            "TypeSafe key present"
        } else {
            "no TypeSafe key"
        };
        format!("jev-compact: {on} ({key})")
    }

    /// Enable or disable Jev prune for this process. Enabling without a
    /// TypeSafe key is refused so the toggle cannot silently no-op.
    pub fn set_jev_compact(&mut self, enabled: bool) -> Result<(), String> {
        if enabled && !self.typesafe.is_enabled() {
            return Err(
                "jev-compact needs TYPESAFE_API_KEY or [typesafe] in ~/.config/hi/config.toml"
                    .into(),
            );
        }
        self.jev_compact = enabled;
        Ok(())
    }

    pub fn apply_jev_compact_arg(&mut self, arg: &str) -> String {
        match arg.trim().to_ascii_lowercase().as_str() {
            "" | "status" => self.jev_compact_status_line(),
            "on" | "enable" => match self.set_jev_compact(true) {
                Ok(()) => self.jev_compact_status_line(),
                Err(err) => err,
            },
            "off" | "disable" => {
                let _ = self.set_jev_compact(false);
                self.jev_compact_status_line()
            }
            "toggle" => {
                if self.jev_compact {
                    let _ = self.set_jev_compact(false);
                    self.jev_compact_status_line()
                } else {
                    match self.set_jev_compact(true) {
                        Ok(()) => self.jev_compact_status_line(),
                        Err(err) => err,
                    }
                }
            }
            _ => "use /jev-compact on|off".into(),
        }
    }

    pub fn steer(&self) -> SteerQueue {
        self.steer.clone()
    }

    pub fn sandbox_enforced(&self) -> bool {
        self.tools.sandbox_enforced()
    }

    pub fn sandbox_backend_name(&self) -> &'static str {
        self.tools.sandbox_backend_name()
    }

    pub fn context_window(&self) -> u32 {
        if self.model() == "pipe/auto" {
            return self
                .model_windows
                .get("pipe/auto")
                .copied()
                .unwrap_or(64_000)
                .min(64_000);
        }
        self.model_windows
            .get(&self.model())
            .copied()
            .unwrap_or(self.context_window)
    }

    /// Occupancy for cheap-shrink / auto-compact. A 1–2M advertised Pipe
    /// window must not disable shrinking; the model still falls apart well
    /// before that.
    pub(crate) fn working_context_window(&self) -> u32 {
        self.context_window().clamp(1, DEFAULT_CONTEXT_WINDOW)
    }

    pub fn context_window_source(&self) -> &'static str {
        if self.model_windows.contains_key(&self.model()) {
            "models"
        } else {
            "default"
        }
    }

    /// Remember `/models` context windows and advertised output caps. Unknown
    /// ids keep the previous window so a missing metadata row cannot snap
    /// occupancy back to 128k and spuriously auto-compact.
    pub fn remember_model_windows(&mut self, models: &[ServedModel]) {
        for model in models {
            if let Some(window) = model.context_window.filter(|window| *window > 0) {
                self.model_windows.insert(model.id.clone(), window);
            }
            if let Some(output) = model.max_output_tokens.filter(|output| *output > 0) {
                self.model_output_caps.insert(model.id.clone(), output);
            }
        }
        self.apply_cached_window();
    }

    fn apply_cached_window(&mut self) {
        if let Some(window) = self.model_windows.get(&self.model()).copied() {
            self.context_window = window;
        }
    }

    pub fn version() -> &'static str {
        env!("CARGO_PKG_VERSION")
    }

    pub fn set_api_key(&mut self, api_key: impl Into<String>) {
        self.client.set_api_key(api_key);
    }

    pub fn api_key(&self) -> &str {
        self.client.api_key()
    }

    pub fn max_tokens(&self) -> u32 {
        if self.model() == "pipe/auto" {
            return self.configured_max_tokens.min(8192).min(
                self.model_output_caps
                    .get("pipe/auto")
                    .copied()
                    .unwrap_or(8192),
            );
        }
        let model = self.model();
        effective_coding_agent_max_tokens(
            &model,
            self.configured_max_tokens,
            self.max_tokens_explicit,
            self.model_output_caps.get(&model).copied(),
        )
    }

    pub fn set_max_tokens(&mut self, max_tokens: u32) {
        self.configured_max_tokens = max_tokens.max(1);
        self.max_tokens_explicit = true;
    }

    /// Fetch `/models` and apply advertised context windows and output caps.
    /// Failures leave the previous limits in place.
    pub async fn refresh_provider_limits(&mut self) -> Result<Vec<ServedModel>> {
        let models = self.client.list_models().await?;
        self.remember_model_windows(&models);
        Ok(models)
    }

    pub fn session_usage(&self) -> Usage {
        self.session_usage
    }

    pub fn verify_command(&self) -> Option<&str> {
        self.verify_command.as_deref()
    }

    pub fn set_verify_command(&mut self, command: Option<String>) {
        self.verify_command = command.filter(|c| !c.is_empty() && c != "off" && c != "none");
        if let Some(session) = &mut self.session {
            let _ = session.record_verify(self.verify_command.as_deref());
        }
    }

    pub async fn list_models(&self) -> Result<Vec<ServedModel>> {
        self.client.list_models().await
    }

    pub fn checkpoint_count(&self) -> usize {
        self.checkpoints.len()
    }

    pub async fn undo(&mut self) -> Result<Option<usize>> {
        let Some(reference) = self.checkpoints.last().cloned() else {
            return Ok(None);
        };
        let (target, expected) = checkpoint::parse_reference(&reference)?;
        let changed = if let Some(expected) = expected {
            checkpoint::restore_sealed_with_state(
                &self.workspace_root,
                target,
                expected,
                &self.state_root,
            )
            .await
            .context("restoring sealed checkpoint")?
        } else {
            anyhow::bail!("checkpoint is not sealed; cannot undo safely");
        };
        self.checkpoints.pop();
        if let Some(session) = &mut self.session {
            let _ = session.record_checkpoints(&self.checkpoints);
        }
        Ok(Some(changed))
    }

    pub fn clear_history(&mut self) {
        let previous = std::mem::take(&mut self.messages);
        let prev_plan = std::mem::take(&mut self.plan);
        let prev_files = std::mem::take(&mut self.last_changed_files);
        let prev_pending = self.pending_turn.take();
        self.review_drive = ReviewDrive::default();
        self.forced_intent = None;
        if !self.persist_snapshot() {
            self.messages = previous;
            self.plan = prev_plan;
            self.last_changed_files = prev_files;
            self.pending_turn = prev_pending;
        }
    }

    pub fn snapshot(&self) -> SessionSnapshot {
        SessionSnapshot {
            messages: self.messages.clone(),
            plan: self.plan.clone(),
            usage: self.session_usage,
            last_changed_files: self.last_changed_files.clone(),
        }
    }

    pub fn restore_snapshot(&mut self, snapshot: SessionSnapshot) {
        self.messages = snapshot.messages;
        self.plan = snapshot.plan;
        self.session_usage = snapshot.usage;
        self.last_changed_files = snapshot.last_changed_files;
    }

    pub fn rewind_to_user_turn(&mut self, n: usize) -> Result<()> {
        let turns = list_user_turns(&self.messages);
        let Some(turn) = turns.iter().find(|t| t.n == n) else {
            anyhow::bail!("no user turn {n}");
        };
        self.truncate_messages(turn.message_index);
        Ok(())
    }

    pub(crate) fn record_context_occupancy(&mut self, usage: Usage) {
        self.last_context_occupancy = usage.context_occupancy.max(usage.input_tokens);
        self.session_usage.context_occupancy = self.last_context_occupancy;
    }

    pub(crate) fn current_occupancy(&self) -> u64 {
        if self.model() == "pipe/auto" {
            let messages = self.request_messages(None);
            let tools = tools::advertised_tools();
            return serde_json::to_vec(&pipe::build_body(
                "pipe/auto",
                &messages,
                &tools,
                self.max_tokens(),
                None,
            ))
            .map_or(u64::MAX, |v| {
                v.len() as u64 + self.max_tokens() as u64 + 1024
            });
        }
        let estimated = compact::estimate_message_tokens(&self.messages);
        self.last_context_occupancy.max(estimated)
    }

    pub async fn doctor_report(&mut self) -> String {
        let mut lines = vec![format!("hi {version}", version = Self::version())];
        let key = if self.client.has_api_key() {
            "ok"
        } else {
            "missing — /login pipenetwork"
        };
        lines.push(format!("credential: {key}"));
        match self.refresh_provider_limits().await {
            Ok(models) => {
                lines.push(format!(
                    "pipe /models: ok ({} model{})",
                    models.len(),
                    if models.len() == 1 { "" } else { "s" }
                ));
            }
            Err(err) => lines.push(format!("pipe /models: {err:#}")),
        }
        let git = std::process::Command::new("git")
            .args(["rev-parse", "--is-inside-work-tree"])
            .current_dir(&self.workspace_root)
            .output();
        match git {
            Ok(out) if out.status.success() => {
                lines.push("git: ok (undo checkpoints available)".into())
            }
            Ok(_) => lines.push("git: not a repository — /undo needs git".into()),
            Err(err) => lines.push(format!("git: {err}")),
        }
        if self.sandbox_enforced() {
            lines.push(format!(
                "sandbox: {} (HI_SANDBOX=off disables)",
                self.sandbox_backend_name()
            ));
        } else {
            lines.push(format!("sandbox: off ({})", self.sandbox_backend_name()));
        }
        lines.push(sentinel_doctor_line());
        if let Some(label) = self.next_action.doctor_label() {
            lines.push(format!("typesafe next-action: {label}"));
        }
        if self.typesafe.is_enabled() {
            lines.push(format!(
                "typesafe auto: {}",
                if self.typesafe.auto { "on" } else { "off" }
            ));
            lines.push(format!(
                "typesafe effort: {}",
                if self.typesafe.effort { "on" } else { "off" }
            ));
            if self.turn_open
                && let Some(effort) = self.turn_effort
            {
                lines.push(format!("typesafe effort override: {}", effort.as_str()));
            }
        }
        lines.push(self.jev_compact_status_line());
        lines.join("\n")
    }

    pub fn session_path(&self) -> Option<&Path> {
        self.session.as_ref().map(|session| session.path())
    }

    pub fn cancel_turn(&self) {
        if let Some(cancel) = &self.turn_cancel {
            cancel.cancel();
        }
        self.interrupt.store(true, Ordering::Release);
        self.tools.interrupt();
    }

    pub fn base_url(&self) -> &str {
        self.client.base_url()
    }
}

/// Cloneable queue of mid-turn user steering. The turn loop drains this before
/// each model round after the first.
#[derive(Clone, Default)]
pub struct SteerQueue {
    inner: Arc<Mutex<Vec<String>>>,
}

impl SteerQueue {
    pub fn push(&self, text: String) {
        let text = text.trim().to_string();
        if text.is_empty() {
            return;
        }
        lock_mut_vec(&self.inner).push(text);
    }

    pub fn drain(&self) -> Vec<String> {
        std::mem::take(&mut *lock_mut_vec(&self.inner))
    }
}

fn lock_mut_vec<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn resolve_liveness(config: &HarnessConfig, tools: &ToolHost) -> hi_liveness::Publisher {
    config
        .liveness
        .clone()
        .or_else(hi_liveness::installed_publisher)
        .unwrap_or_else(|| tools.liveness())
}

impl Drop for Harness {
    fn drop(&mut self) {
        if self.turn_open {
            hi_liveness::report_invariant(&self.liveness, hi_liveness::InvariantCode::TurnUnclosed);
        }
    }
}

fn sentinel_doctor_line() -> String {
    sentinel_doctor_line_from(
        std::env::var(hi_liveness::ENV_SUPERVISED)
            .ok()
            .as_deref()
            .is_some_and(hi_liveness::env_flag_on),
        std::env::var(hi_liveness::ENV_GENERATION)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(0),
    )
}

/// Every `ToolCall` after `messages[0]` (the in-flight user line) must have a
/// matching `ToolResult` before the next assistant or user message. Otherwise
/// resume would replay a truncated round that Pipe rejects.
fn tool_results_match_calls(messages: &[Message]) -> bool {
    let mut pending: Vec<&str> = Vec::new();
    for message in messages {
        match message.role {
            hi_ai::Role::Assistant => {
                if !pending.is_empty() {
                    return false;
                }
                for block in &message.content {
                    if let hi_ai::Content::ToolCall { id, .. } = block {
                        pending.push(id);
                    }
                }
            }
            hi_ai::Role::Tool => {
                for block in &message.content {
                    if let hi_ai::Content::ToolResult { call_id, .. } = block {
                        if let Some(i) = pending.iter().position(|id| *id == call_id) {
                            pending.remove(i);
                        } else {
                            return false;
                        }
                    }
                }
            }
            hi_ai::Role::User => {
                if !pending.is_empty() {
                    return false;
                }
            }
            hi_ai::Role::System => return false,
        }
    }
    pending.is_empty()
}

fn sentinel_doctor_line_from(supervised: bool, generation: u32) -> String {
    if supervised {
        format!("sentinel: on (generation {generation})")
    } else {
        "sentinel: off".into()
    }
}

fn open_session(path: Option<&Path>, model: &str) -> Result<Option<SessionFile>> {
    let Some(path) = path else {
        return Ok(None);
    };
    let existed = path.is_file()
        && std::fs::metadata(path)
            .map(|meta| meta.len() > 0)
            .unwrap_or(false);
    let mut session = SessionFile::create(path)?;
    if !existed {
        let _ = session.record_model(model);
    }
    Ok(Some(session))
}

#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "review_harness_tests.rs"]
mod review_harness_tests;

#[cfg(test)]
#[path = "review_guard_tests.rs"]
mod review_guard_tests;

#[cfg(test)]
#[path = "review_harness_scope_tests.rs"]
mod review_harness_scope_tests;
