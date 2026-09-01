//! Canonical interactive-session events and an owned session driver.
//!
//! The existing [`crate::Ui`] trait remains the compatibility surface for
//! terminals and ACP.  This module adds a provider/frontend-neutral stream on
//! top of it: every consumer can subscribe without taking ownership of the
//! agent or sharing a mutable [`crate::Agent`] reference.

use std::io::{BufRead, BufReader, Write};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, mpsc, oneshot};
use uuid::Uuid;

use crate::{Agent, AgentSessionSnapshot, TurnCancellation, TurnOutcome, Ui};

/// A typed lifecycle event emitted by an interactive agent session.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEventKind {
    SessionStarted,
    TurnStarted {
        input: String,
    },
    AssistantText {
        text: String,
    },
    Reasoning {
        text: String,
    },
    ToolStarted {
        id: String,
        name: String,
        arguments: String,
    },
    ToolStream {
        id: String,
        name: String,
        line: String,
    },
    ToolCall {
        id: String,
        name: String,
        arguments: String,
    },
    ToolResult {
        id: String,
        name: String,
        result: String,
        status: String,
    },
    Status {
        text: String,
    },
    PlanChanged {
        steps: Vec<hi_tools::PlanStep>,
    },
    Usage {
        usage: hi_ai::Usage,
        context_used: u64,
        context_window: Option<u32>,
        usage_estimated: bool,
    },
    ChangedFiles {
        files: Vec<String>,
    },
    TurnCompleted {
        outcome: TurnOutcome,
        summary: String,
    },
    Error {
        kind: String,
        message: String,
        guidance: String,
    },
    Nudge {
        text: String,
    },
    Sandbox {
        backend: String,
        status: String,
        detail: String,
    },
    Forked {
        child_session_id: String,
        worktree: Option<String>,
    },
}

impl AgentEventKind {
    /// Full live streams remain ephemeral; these lifecycle/tool/telemetry
    /// events are safe durable milestones for reports and replay.
    pub fn is_durable_milestone(&self) -> bool {
        matches!(
            self,
            Self::SessionStarted
                | Self::TurnStarted { .. }
                | Self::ToolStarted { .. }
                | Self::ToolCall { .. }
                | Self::ToolResult { .. }
                | Self::Usage { .. }
                | Self::ChangedFiles { .. }
                | Self::TurnCompleted { .. }
                | Self::Error { .. }
                | Self::Sandbox { .. }
                | Self::Forked { .. }
        )
    }
}

/// Envelope shared by all live and durable event consumers.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentEvent {
    pub session_id: String,
    pub turn_id: Option<String>,
    pub sequence: u64,
    pub timestamp_ms: u128,
    pub kind: AgentEventKind,
}

/// Append-only JSONL milestone store. It is deliberately a separate sink from
/// `SessionSink`: event history can be replayed without changing transcript
/// resume semantics.
pub struct EventJournal {
    file: Mutex<std::fs::File>,
}

impl EventJournal {
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        Ok(Self {
            file: Mutex::new(
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)?,
            ),
        })
    }

    pub fn record(&self, event: &AgentEvent) -> Result<()> {
        if !event.kind.is_durable_milestone() {
            return Ok(());
        }
        let mut file = self
            .file
            .lock()
            .map_err(|_| anyhow!("event journal lock poisoned"))?;
        serde_json::to_writer(&mut *file, event)?;
        file.write_all(b"\n")?;
        file.flush()?;
        Ok(())
    }

    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Vec<AgentEvent>> {
        let file = std::fs::File::open(path)?;
        BufReader::new(file)
            .lines()
            .filter(|line| line.as_ref().is_ok_and(|line| !line.trim().is_empty()))
            .map(|line| Ok(serde_json::from_str(&line?)?))
            .collect()
    }
}

/// Result of a turn submitted through [`SessionHandle`].
pub struct TurnResult {
    pub turn_id: String,
    pub outcome: TurnOutcome,
    pub snapshot: AgentSessionSnapshot,
}

#[derive(Clone, Debug, Default)]
pub struct ForkOptions {
    pub worktree: bool,
    pub label: String,
}

/// A snapshot-based branch. The caller can create a new Agent with
/// `Agent::resume_snapshot`; the optional worktree is created only when
/// requested, so transcript-only forks remain cheap.
pub struct SessionFork {
    pub parent_session_id: String,
    pub child_session_id: String,
    pub snapshot: AgentSessionSnapshot,
    pub worktree: Option<std::path::PathBuf>,
}

/// A cloneable command handle for an owned session driver.
#[derive(Clone)]
pub struct SessionHandle {
    commands: mpsc::Sender<SessionCommand>,
    events: broadcast::Sender<AgentEvent>,
}

impl SessionHandle {
    /// Subscribe to the live event stream. A slow subscriber receives a
    /// `RecvError::Lagged` and can reconnect from its durable milestone sink.
    pub fn subscribe(&self) -> EventStream {
        EventStream(self.events.subscribe())
    }

    /// Submit one turn. The driver serializes turns for the owned Agent.
    pub async fn run_turn(&self, input: impl Into<String>) -> Result<TurnResult> {
        let (reply, result) = oneshot::channel();
        self.commands
            .send(SessionCommand::Turn {
                input: input.into(),
                reply,
            })
            .await
            .map_err(|_| anyhow!("session driver stopped"))?;
        result
            .await
            .map_err(|_| anyhow!("session driver dropped turn result"))?
            .map_err(|message| anyhow!("{message}"))
    }

    /// Cooperatively cancel the active turn.
    pub async fn cancel(&self) -> Result<()> {
        self.commands
            .send(SessionCommand::Cancel)
            .await
            .map_err(|_| anyhow!("session driver stopped"))
    }

    /// Inject steering text at the next safe model-round boundary.
    pub async fn interject(&self, input: impl Into<String>) -> Result<()> {
        self.commands
            .send(SessionCommand::Interject(input.into()))
            .await
            .map_err(|_| anyhow!("session driver stopped"))
    }

    /// Capture a snapshot between turns.
    pub async fn snapshot(&self) -> Result<AgentSessionSnapshot> {
        let (reply, result) = oneshot::channel();
        self.commands
            .send(SessionCommand::Snapshot { reply })
            .await
            .map_err(|_| anyhow!("session driver stopped"))?;
        result
            .await
            .map_err(|_| anyhow!("session driver dropped snapshot"))?
            .map_err(|message| anyhow!("{message}"))
    }

    pub async fn fork(&self, options: ForkOptions) -> Result<SessionFork> {
        let (reply, result) = oneshot::channel();
        self.commands
            .send(SessionCommand::Fork { options, reply })
            .await
            .map_err(|_| anyhow!("session driver stopped"))?;
        result
            .await
            .map_err(|_| anyhow!("session driver dropped fork"))?
            .map_err(|message| anyhow!("{message}"))
    }
}

/// A subscription to a session's event stream.
pub struct EventStream(broadcast::Receiver<AgentEvent>);

impl EventStream {
    pub async fn recv(&mut self) -> Result<AgentEvent, broadcast::error::RecvError> {
        self.0.recv().await
    }

    pub fn try_recv(&mut self) -> Result<AgentEvent, broadcast::error::TryRecvError> {
        self.0.try_recv()
    }
}

/// Owns one mutable Agent and one frontend UI, allowing callers to interact
/// through a cheap cloneable [`SessionHandle`].
pub struct SessionDriver {
    handle: SessionHandle,
    join: tokio::task::JoinHandle<()>,
}

impl SessionDriver {
    pub fn spawn(agent: Agent, ui: Box<dyn Ui>) -> Self {
        Self::spawn_inner(agent, ui, None)
    }

    /// Spawn a session and persist only durable milestone events to a JSONL
    /// journal. The live stream remains independently available to UI clients.
    pub fn spawn_with_journal(agent: Agent, ui: Box<dyn Ui>, journal: Arc<EventJournal>) -> Self {
        Self::spawn_inner(agent, ui, Some(journal))
    }

    fn spawn_inner(agent: Agent, ui: Box<dyn Ui>, journal: Option<Arc<EventJournal>>) -> Self {
        let (commands, receiver) = mpsc::channel(32);
        let (events, _) = broadcast::channel(256);
        let handle = SessionHandle {
            commands,
            events: events.clone(),
        };
        if let Some(journal) = journal {
            let mut stream = EventStream(events.subscribe());
            tokio::spawn(async move {
                while let Ok(event) = stream.recv().await {
                    let _ = journal.record(&event);
                }
            });
        }
        // Agent is Send but intentionally not Sync: a SessionSink may contain
        // a mutable file handle. Run the owner on a dedicated Tokio worker so
        // the public handle remains Send without weakening SessionSink's
        // single-owner invariant.
        let join = tokio::task::spawn_blocking(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("session driver runtime");
            runtime.block_on(run_driver(agent, ui, receiver, events));
        });
        Self { handle, join }
    }

    pub fn handle(&self) -> SessionHandle {
        self.handle.clone()
    }

    pub async fn join(self) -> Result<()> {
        self.join
            .await
            .map_err(|error| anyhow!("session driver panicked: {error}"))
    }
}

enum SessionCommand {
    Turn {
        input: String,
        reply: oneshot::Sender<std::result::Result<TurnResult, String>>,
    },
    Cancel,
    Interject(String),
    Snapshot {
        reply: oneshot::Sender<std::result::Result<AgentSessionSnapshot, String>>,
    },
    Fork {
        options: ForkOptions,
        reply: oneshot::Sender<std::result::Result<SessionFork, String>>,
    },
}

async fn run_driver(
    mut agent: Agent,
    ui: Box<dyn Ui>,
    mut commands: mpsc::Receiver<SessionCommand>,
    events: broadcast::Sender<AgentEvent>,
) {
    let session_id = Uuid::new_v4().to_string();
    let sequence = Arc::new(AtomicU64::new(0));
    emit(
        &events,
        &session_id,
        &sequence,
        None,
        AgentEventKind::SessionStarted,
    );
    if let Ok(runner) = hi_tools::ProcessRunner::new(agent.workspace_root()) {
        emit(
            &events,
            &session_id,
            &sequence,
            None,
            AgentEventKind::Sandbox {
                backend: runner.sandbox_backend_name().into(),
                status: format!("{:?}", runner.sandbox_backend_status()).to_ascii_lowercase(),
                detail: if runner.sandbox_enforced() {
                    "sandbox backend active".into()
                } else {
                    hi_tools::sandbox::SandboxProfile::unenforced_warning().into()
                },
            },
        );
    }
    let mut event_ui = EventUi::new(ui, events.clone(), session_id.clone(), sequence.clone());

    while let Some(command) = commands.recv().await {
        match command {
            SessionCommand::Turn { input, reply } => {
                let turn_id = Uuid::new_v4().to_string();
                event_ui.turn_id = Some(turn_id.clone());
                emit(
                    &events,
                    &session_id,
                    &sequence,
                    Some(&turn_id),
                    AgentEventKind::TurnStarted {
                        input: redact(&input),
                    },
                );
                let cancellation = TurnCancellation::new();
                let inbox = agent.interjection_inbox();
                let mut turn = Box::pin(agent.run_turn_cancellable(
                    &input,
                    &mut event_ui,
                    cancellation.clone(),
                ));
                let result = loop {
                    tokio::select! {
                        result = &mut turn => break result,
                        command = commands.recv() => match command {
                            Some(SessionCommand::Cancel) => cancellation.cancel(),
                            Some(SessionCommand::Interject(message)) => inbox.push(message),
                            Some(SessionCommand::Snapshot { reply }) => {
                                let _ = reply.send(Err("cannot snapshot while a turn is running".into()));
                            }
                            Some(SessionCommand::Fork { reply, .. }) => {
                                let _ = reply.send(Err("cannot fork while a turn is running".into()));
                            }
                            Some(SessionCommand::Turn { reply, .. }) => {
                                let _ = reply.send(Err("a turn is already running".into()));
                            }
                            None => {
                                // The client disappeared with a live turn. Do not
                                // drop the future here: that bypasses transcript,
                                // workspace-ledger, and background-job cleanup.
                                // `run_turn_cancellable` bounds cooperative
                                // settlement before applying its hard cleanup
                                // backstop, so it is safe to await here.
                                cancellation.disconnect();
                                break match (&mut turn).await {
                                    Ok(_) => Err(anyhow!("session command channel closed")),
                                    Err(error) => Err(anyhow!(
                                        "session command channel closed; cancellation cleanup failed: {error:#}"
                                    )),
                                };
                            }
                        }
                    }
                };
                // The future retains the mutable borrows of Agent and EventUi
                // until it is explicitly dropped, even after select! returns.
                drop(turn);
                let response = match result {
                    Ok(outcome) => {
                        emit(
                            &events,
                            &session_id,
                            &sequence,
                            Some(&turn_id),
                            AgentEventKind::TurnCompleted {
                                summary: format!("{outcome:?}"),
                                outcome: outcome.clone(),
                            },
                        );
                        Ok(TurnResult {
                            turn_id,
                            outcome,
                            snapshot: agent.session_snapshot(),
                        })
                    }
                    Err(error) => {
                        emit(
                            &events,
                            &session_id,
                            &sequence,
                            Some(&turn_id),
                            AgentEventKind::Error {
                                kind: "turn".into(),
                                message: redact(&format!("{error:#}")),
                                guidance: String::new(),
                            },
                        );
                        Err(format!("{error:#}"))
                    }
                };
                event_ui.turn_id = None;
                let _ = reply.send(response);
            }
            SessionCommand::Cancel | SessionCommand::Interject(_) => {
                // There is no active turn to cancel or steer.
            }
            SessionCommand::Snapshot { reply } => {
                let _ = reply.send(Ok(agent.session_snapshot()));
            }
            SessionCommand::Fork { options, reply } => {
                let child_session_id = Uuid::new_v4().to_string();
                let worktree = if options.worktree {
                    let label = if options.label.trim().is_empty() {
                        "branch"
                    } else {
                        options.label.trim()
                    };
                    match crate::fork_worktree(agent.workspace_root(), label) {
                        Ok(path) => Some(path),
                        Err(error) => {
                            let _ = reply.send(Err(format!("fork worktree failed: {error:#}")));
                            continue;
                        }
                    }
                } else {
                    None
                };
                emit(
                    &events,
                    &session_id,
                    &sequence,
                    None,
                    AgentEventKind::Forked {
                        child_session_id: child_session_id.clone(),
                        worktree: worktree.as_ref().map(|path| path.display().to_string()),
                    },
                );
                let _ = reply.send(Ok(SessionFork {
                    parent_session_id: session_id.clone(),
                    child_session_id,
                    snapshot: agent.session_snapshot(),
                    worktree,
                }));
            }
        }
    }
}

fn emit(
    events: &broadcast::Sender<AgentEvent>,
    session_id: &str,
    sequence: &AtomicU64,
    turn_id: Option<&str>,
    kind: AgentEventKind,
) {
    let _ = events.send(AgentEvent {
        session_id: session_id.to_string(),
        turn_id: turn_id.map(str::to_string),
        sequence: sequence.fetch_add(1, Ordering::Relaxed),
        timestamp_ms: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or_default(),
        kind,
    });
}

fn redact(text: &str) -> String {
    crate::ui::redact_debug_text(text, &[])
}

struct EventUi {
    inner: Box<dyn Ui>,
    events: broadcast::Sender<AgentEvent>,
    session_id: String,
    sequence: Arc<AtomicU64>,
    turn_id: Option<String>,
}

impl EventUi {
    fn new(
        inner: Box<dyn Ui>,
        events: broadcast::Sender<AgentEvent>,
        session_id: String,
        sequence: Arc<AtomicU64>,
    ) -> Self {
        Self {
            inner,
            events,
            session_id,
            sequence,
            turn_id: None,
        }
    }

    fn emit(&self, kind: AgentEventKind) {
        emit(
            &self.events,
            &self.session_id,
            &self.sequence,
            self.turn_id.as_deref(),
            kind,
        );
    }
}

impl Ui for EventUi {
    fn assistant_text(&mut self, text: &str) {
        self.emit(AgentEventKind::AssistantText { text: redact(text) });
        self.inner.assistant_text(text);
    }
    fn btw_answer(&mut self, text: &str) {
        self.inner.btw_answer(text);
    }
    fn btw_question(&mut self, question: &str) {
        self.inner.btw_question(question);
    }
    fn btw_tool_started(&mut self, name: &str, arguments: &str) {
        self.inner.btw_tool_started(name, arguments);
    }
    fn btw_tool_result(&mut self, name: &str, result: &str) {
        self.inner.btw_tool_result(name, result);
    }
    fn btw_end(&mut self) {
        self.inner.btw_end();
    }
    fn assistant_reasoning(&mut self, text: &str) {
        self.emit(AgentEventKind::Reasoning { text: redact(text) });
        self.inner.assistant_reasoning(text);
    }
    fn assistant_end(&mut self) {
        self.inner.assistant_end();
    }
    fn tool_started_id(&mut self, id: &str, name: &str, arguments: &str) {
        self.emit(AgentEventKind::ToolStarted {
            id: id.into(),
            name: name.into(),
            arguments: redact(arguments),
        });
        self.inner.tool_started_id(id, name, arguments);
    }
    fn tool_stream_id(&mut self, id: &str, name: &str, line: &str) {
        self.emit(AgentEventKind::ToolStream {
            id: id.into(),
            name: name.into(),
            line: redact(line),
        });
        self.inner.tool_stream_id(id, name, line);
    }
    fn confirm(&mut self, request: crate::ConfirmationRequest) -> crate::ConfirmationFuture<'_> {
        self.inner.confirm(request)
    }
    fn tool_call(&mut self, name: &str, arguments: &str) {
        self.inner.tool_call(name, arguments);
    }
    fn tool_call_id(&mut self, id: &str, name: &str, arguments: &str) {
        self.emit(AgentEventKind::ToolCall {
            id: id.into(),
            name: name.into(),
            arguments: redact(arguments),
        });
        self.inner.tool_call_id(id, name, arguments);
    }
    fn tool_result(&mut self, name: &str, result: &str) {
        self.inner.tool_result(name, result);
    }
    fn tool_result_id(&mut self, id: &str, name: &str, result: &str, status: hi_tools::ToolStatus) {
        self.emit(AgentEventKind::ToolResult {
            id: id.into(),
            name: name.into(),
            result: redact(result),
            status: format!("{status:?}"),
        });
        self.inner.tool_result_id(id, name, result, status);
    }
    fn status(&mut self, text: &str) {
        self.emit(AgentEventKind::Status { text: redact(text) });
        self.inner.status(text);
    }
    fn checkpoint_warning(&mut self, text: &str) {
        self.inner.checkpoint_warning(text);
    }
    fn subagent_note(&mut self, text: &str) {
        self.inner.subagent_note(text);
    }
    fn subagent_sink(&self) -> Option<std::sync::Arc<dyn crate::SubagentSink>> {
        self.inner.subagent_sink()
    }
    fn subagent_spawned(&mut self, id: &str, kind: &str, description: &str, background: bool) {
        self.inner
            .subagent_spawned(id, kind, description, background);
    }
    fn subagent_progress(&mut self, id: &str, activity: &str) {
        self.inner.subagent_progress(id, activity);
    }
    fn subagent_finished(&mut self, id: &str, status: &str, elapsed_ms: u64, summary: &str) {
        self.inner
            .subagent_finished(id, status, elapsed_ms, summary);
    }
    fn plan(&mut self, steps: &[hi_tools::PlanStep]) {
        self.emit(AgentEventKind::PlanChanged {
            steps: steps.to_vec(),
        });
        self.inner.plan(steps);
    }
    fn usage(
        &mut self,
        prompt_tokens: u64,
        generated_tokens: u64,
        context_used: u64,
        context_window: Option<u32>,
        usage_estimated: bool,
    ) {
        self.emit(AgentEventKind::Usage {
            usage: hi_ai::Usage {
                input_tokens: prompt_tokens,
                output_tokens: generated_tokens,
                estimated: usage_estimated,
                ..hi_ai::Usage::default()
            },
            context_used,
            context_window,
            usage_estimated,
        });
        self.inner.usage(
            prompt_tokens,
            generated_tokens,
            context_used,
            context_window,
            usage_estimated,
        );
    }
    fn session_usage(&mut self, usage: &hi_ai::Usage) {
        self.inner.session_usage(usage);
    }
    fn rate_limits(&mut self, rate_limits: Option<hi_ai::RateLimitState>) {
        self.inner.rate_limits(rate_limits);
    }
    fn turn_end(&mut self, summary: &str) {
        self.inner.turn_end(summary);
    }
    fn changed_files(&mut self, files: &[String]) {
        self.emit(AgentEventKind::ChangedFiles {
            files: files.to_vec(),
        });
        self.inner.changed_files(files);
    }
    fn suggested_prompt(&mut self, text: &str) {
        // Ghost-text prediction is UI chrome, not a durable agent event.
        self.inner.suggested_prompt(text);
    }
    fn turn_error(&mut self, kind: &str, message: &str, guidance: &str) {
        self.emit(AgentEventKind::Error {
            kind: kind.into(),
            message: redact(message),
            guidance: redact(guidance),
        });
        self.inner.turn_error(kind, message, guidance);
    }
    fn nudge(&mut self, text: &str) {
        self.emit(AgentEventKind::Nudge { text: redact(text) });
        self.inner.nudge(text);
    }
}

/// Compatibility alias for code that wants the generic name from the session
/// API without depending on the historical struct name.
pub type SessionSnapshot = AgentSessionSnapshot;

#[cfg(test)]
mod tests {
    use super::*;

    struct DisconnectProvider {
        started: Arc<std::sync::atomic::AtomicBool>,
        completed: Arc<std::sync::atomic::AtomicBool>,
    }

    #[async_trait::async_trait]
    impl hi_ai::Provider for DisconnectProvider {
        async fn stream(
            &self,
            _request: hi_ai::ChatRequest,
            _sink: &mut (dyn FnMut(hi_ai::StreamEvent) + Send),
        ) -> Result<hi_ai::Completion> {
            self.started
                .store(true, std::sync::atomic::Ordering::SeqCst);
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            self.completed
                .store(true, std::sync::atomic::Ordering::SeqCst);
            Ok(hi_ai::Completion {
                content: vec![hi_ai::Content::ToolCall {
                    id: "read-after-disconnect".into(),
                    name: "read".into(),
                    arguments: r#"{"path":"missing.txt"}"#.into(),
                }],
                usage: hi_ai::Usage {
                    input_tokens: 1,
                    output_tokens: 1,
                    ..hi_ai::Usage::default()
                },
                ..hi_ai::Completion::default()
            })
        }
    }

    #[derive(Default)]
    struct DriverLifecycleProbe {
        done: std::sync::atomic::AtomicUsize,
        aborts: Mutex<Vec<hi_agent_lifecycle::TurnAbortReason>>,
    }

    #[async_trait::async_trait]
    impl hi_agent_lifecycle::TurnLifecycleContributor for DriverLifecycleProbe {
        async fn on_turn_done(&self, _: &hi_agent_lifecycle::TurnDoneInput) {
            self.done.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }

        async fn on_turn_abort(&self, input: &hi_agent_lifecycle::TurnAbortInput) {
            self.aborts.lock().unwrap().push(input.reason);
        }
    }

    struct DriverNullUi;

    impl Ui for DriverNullUi {
        fn assistant_text(&mut self, _: &str) {}
        fn assistant_reasoning(&mut self, _: &str) {}
        fn assistant_end(&mut self) {}
        fn tool_call(&mut self, _: &str, _: &str) {}
        fn tool_result(&mut self, _: &str, _: &str) {}
        fn status(&mut self, _: &str) {}
        fn turn_end(&mut self, _: &str) {}
    }

    #[test]
    fn event_envelope_is_json_stable_and_redaction_is_applied_by_helper() {
        let event = AgentEvent {
            session_id: "s".into(),
            turn_id: None,
            sequence: 0,
            timestamp_ms: 1,
            kind: AgentEventKind::Status {
                text: redact("authorization: Bearer sk-secret"),
            },
        };
        let json = serde_json::to_value(event).unwrap();
        assert_eq!(json["kind"]["type"], "status");
        assert!(
            json["kind"]["text"]
                .as_str()
                .unwrap()
                .contains("[REDACTED]")
        );
    }

    #[tokio::test]
    async fn command_disconnect_awaits_turn_cancellation_cleanup() {
        let root =
            std::env::temp_dir().join(format!("hi-agent-events-disconnect-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let mut config = crate::AgentConfig::default();
        config.paths.workspace_root = root.clone();
        config.paths.state_root = root.join(".hi/state");
        config.routing.model = "test-model".into();
        config.routing.max_tokens = 100;
        config.routing.requested_max_tokens = 100;
        config.gates.verification = crate::VerificationMode::Disabled;
        config.gates.read_only_preflight = false;
        config.memory.auto_compact = false;
        config.memory.finalize = false;
        config.memory.suggest_next_prompt = false;
        config.memory.inject_stack_skill = false;
        config.memory.inject_review_skill = false;
        config.loop_limits.max_silent_continues = 0;
        config.loop_limits.max_keep_working = 0;

        let started = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let completed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let lifecycle = Arc::new(DriverLifecycleProbe::default());
        let provider = Arc::new(DisconnectProvider {
            started: started.clone(),
            completed: completed.clone(),
        });
        let mut extensions = hi_agent_lifecycle::ExtensionRegistryBuilder::new();
        extensions.turn_lifecycle_contributor(lifecycle.clone());
        let agent = Agent::new(provider, config)
            .unwrap()
            .with_extension_registry(extensions.build());

        let (command_tx, command_rx) = mpsc::channel(1);
        let (event_tx, _) = broadcast::channel(16);
        let (reply_tx, reply_rx) = oneshot::channel();
        let client = async move {
            command_tx
                .send(SessionCommand::Turn {
                    input: "read the missing file".into(),
                    reply: reply_tx,
                })
                .await
                .unwrap();
            tokio::time::timeout(std::time::Duration::from_secs(2), async {
                while !started.load(std::sync::atomic::Ordering::SeqCst) {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("provider should start");
            drop(command_tx);
            tokio::time::timeout(std::time::Duration::from_secs(2), reply_rx)
                .await
                .expect("driver should settle and reply after disconnect")
                .expect("driver should retain the pending reply sender")
        };

        let ((), response) = tokio::join!(
            run_driver(agent, Box::new(DriverNullUi), command_rx, event_tx),
            client
        );

        let Err(error) = response else {
            panic!("a disconnected command channel must fail the pending request");
        };
        assert!(error.contains("session command channel closed"), "{error}");
        assert!(
            completed.load(std::sync::atomic::Ordering::SeqCst),
            "the driver dropped the live provider future instead of awaiting cancellation"
        );
        assert_eq!(lifecycle.done.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert_eq!(
            *lifecycle.aborts.lock().unwrap(),
            vec![hi_agent_lifecycle::TurnAbortReason::Disconnected],
            "disconnect should dispatch exactly one specifically classified abort"
        );
        let _ = std::fs::remove_dir_all(root);
    }
}
