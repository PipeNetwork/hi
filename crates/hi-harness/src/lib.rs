//! Pipe Network coding-agent harness.
//!
//! One loop: stream from `api.pipenetwork.ai`, execute local tools, repeat.

mod command;
mod compact;
mod dashboard;
mod live;
mod liveness;
mod pipe;
mod prompt;
mod session;
mod tools;
mod turn;
mod ui;
mod usage;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use hi_ai::{Message, ServedModel, Usage};
use hi_tools::{PlanStep, checkpoint};
use tokio::sync::Notify;

pub use command::{
    COMMANDS, CORE_COMMANDS, Command, CommandSpec, EffortArg, ModelArgs, parse as parse_command,
    parse_effort_arg, parse_model_args, resolve_model_query,
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
pub use session::{JsonlSession, LoadedSession, PendingTurn, UserTurn, list_user_turns};
pub use tools::{ToolHost, ToolInterrupt, advertised_tools};
pub use usage::{
    BILLING_URL, UsageCategory, UsageSnapshot, UsageTab, fmt_tokens, occupancy_bar,
    open_billing_url,
};

/// Default Pipe context window used for occupancy display and auto-compact.
pub const DEFAULT_CONTEXT_WINDOW: u32 = 128_000;
pub use ui::{
    ConfirmationFuture, ConfirmationRequest, ConfirmationResult, PermissionMode, TestUi, Ui,
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
    pub workspace_root: PathBuf,
    pub state_root: PathBuf,
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub max_tokens: u32,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub session_path: Option<PathBuf>,
    pub liveness: Option<hi_liveness::Publisher>,
}

impl HarnessConfig {
    pub fn pipe(workspace_root: PathBuf, api_key: impl Into<String>) -> Self {
        let state_root = workspace_root.join(".hi");
        Self {
            workspace_root,
            state_root,
            api_key: api_key.into(),
            base_url: default_base_url(),
            model: DEFAULT_MODEL.to_string(),
            max_tokens: DEFAULT_MAX_TOKENS,
            reasoning_effort: None,
            session_path: None,
            liveness: None,
        }
    }
}

pub struct Harness {
    client: Client,
    tools: ToolHost,
    workspace_root: PathBuf,
    state_root: PathBuf,
    live: LiveSettings,
    max_tokens: u32,
    messages: Vec<Message>,
    plan: Vec<PlanStep>,
    session_usage: Usage,
    checkpoints: Vec<String>,
    last_changed_files: Vec<String>,
    last_context_occupancy: u64,
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
            client: Client::new(config.base_url, config.api_key),
            tools,
            workspace_root: config.workspace_root,
            state_root: config.state_root,
            live: LiveSettings::new(config.model, PermissionMode::Ask, config.reasoning_effort),
            max_tokens: config.max_tokens,
            messages: Vec::new(),
            plan: Vec::new(),
            session_usage: Usage::default(),
            checkpoints: Vec::new(),
            last_changed_files: Vec::new(),
            last_context_occupancy: 0,
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
            client: Client::new(config.base_url, config.api_key),
            tools,
            workspace_root: config.workspace_root,
            state_root: config.state_root,
            live: LiveSettings::new(
                config.model,
                PermissionMode::Always,
                config.reasoning_effort,
            ),
            max_tokens: config.max_tokens,
            messages: Vec::new(),
            plan: Vec::new(),
            session_usage: Usage::default(),
            checkpoints: Vec::new(),
            last_changed_files: Vec::new(),
            last_context_occupancy: 0,
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
        })
    }

    pub fn apply_loaded_session(&mut self, loaded: LoadedSession) {
        self.messages = loaded.messages;
        self.session_usage = loaded.usage;
        self.checkpoints = loaded.checkpoints;
        self.verify_command = loaded.verify_command;
        if let Some(model) = loaded.model {
            self.live.set_model(model);
        }
        if let Some(permission) = loaded.permission {
            self.live
                .set_permission_mode(PermissionMode::from_u8(permission));
        }
        if let Some(effort) = loaded.effort.as_deref() {
            match crate::parse_effort_arg(effort) {
                Ok(crate::EffortArg::Off) => self.live.set_reasoning_effort(None),
                Ok(crate::EffortArg::Level(level)) => self.live.set_reasoning_effort(Some(level)),
                Err(_) => {}
            }
        }
        self.pending_turn = loaded.pending_turn;
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
        }
    }

    pub(crate) fn persist_snapshot(&mut self) {
        let state = self.snapshot_state();
        if let Some(session) = &mut self.session {
            if session.rewrite(&state).is_err() {
                hi_liveness::report_invariant(
                    &self.liveness,
                    hi_liveness::InvariantCode::SessionAppendFailed,
                );
            } else {
                self.liveness.note_progress();
                self.liveness
                    .emit(hi_liveness::EventCode::SessionRewrite, None, None, None);
            }
        }
    }

    /// Drop messages after `len` and rewrite the session file.
    pub fn truncate_messages(&mut self, len: usize) {
        if len < self.messages.len() {
            self.messages.truncate(len);
            self.pending_turn = None;
            self.persist_snapshot();
        }
    }

    pub fn liveness(&self) -> hi_liveness::Publisher {
        self.liveness.clone()
    }

    pub fn pending_turn(&self) -> Option<&PendingTurn> {
        self.pending_turn.as_ref()
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
    }

    pub fn reasoning_effort(&self) -> Option<ReasoningEffort> {
        self.live.reasoning_effort()
    }

    pub fn set_reasoning_effort(&mut self, effort: Option<ReasoningEffort>) {
        self.live.set_reasoning_effort(effort);
        self.persist_knobs();
    }

    pub fn apply_effort_arg(&mut self, effort: EffortArg) {
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
        DEFAULT_CONTEXT_WINDOW
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
        self.max_tokens
    }

    pub fn set_max_tokens(&mut self, max_tokens: u32) {
        self.max_tokens = max_tokens.max(1);
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
        self.messages.clear();
        self.plan.clear();
        self.last_changed_files.clear();
        self.pending_turn = None;
        self.persist_snapshot();
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
        if self.last_context_occupancy > 0 {
            self.last_context_occupancy
        } else {
            compact::estimate_message_tokens(&self.messages)
        }
    }

    pub async fn doctor_report(&self) -> String {
        let mut lines = vec![format!("hi {version}", version = Self::version())];
        let key = if self.client.has_api_key() {
            "ok"
        } else {
            "missing — /login pipenetwork"
        };
        lines.push(format!("credential: {key}"));
        match self.client.list_models().await {
            Ok(models) => lines.push(format!(
                "pipe /models: ok ({} model{})",
                models.len(),
                if models.len() == 1 { "" } else { "s" }
            )),
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
