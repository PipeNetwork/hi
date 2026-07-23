//! ACP server and event adapter for the `hi-agent` coding harness.

use std::{
    collections::HashMap,
    path::Path,
    sync::{
        Arc, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
};

use agent_client_protocol as acp;
use anyhow::{Context, Result};
use hi_agent::{Agent, AgentConfig, TurnCancellation, Ui};
use hi_ai::{Message, Provider, Role, ToolMode};
use tokio::sync::{Mutex, mpsc, oneshot};
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

#[derive(Clone)]
pub struct ShellConfig {
    pub provider: Arc<dyn Provider>,
    pub template: AgentConfig,
    /// Models clients may select. The template model is always available.
    pub models: Vec<String>,
}

pub struct HiShell {
    config: ShellConfig,
    client: OnceLock<Arc<acp::AgentSideConnection>>,
    sessions: Mutex<HashMap<acp::SessionId, Arc<Session>>>,
    snapshots: Mutex<HashMap<acp::SessionId, StoredSession>>,
}

#[derive(Clone)]
struct StoredSession {
    cwd: std::path::PathBuf,
    snapshot: hi_agent::AgentSessionSnapshot,
    model: String,
    mode: ToolMode,
}

struct Session {
    agent: Mutex<Agent>,
    active_turn: Mutex<Option<TurnCancellation>>,
    closed: AtomicBool,
}

impl HiShell {
    pub fn new(config: ShellConfig) -> Self {
        Self {
            config,
            client: OnceLock::new(),
            sessions: Mutex::new(HashMap::new()),
            snapshots: Mutex::new(HashMap::new()),
        }
    }

    pub fn connect(&self, client: Arc<acp::AgentSideConnection>) {
        self.client
            .set(client)
            .expect("ACP connection set more than once");
    }

    fn config_for(&self, cwd: &Path, session_id: &acp::SessionId) -> AgentConfig {
        let mut config = self.config.template.clone();
        config.paths.workspace_root = cwd.to_path_buf();
        config.paths.state_root = self
            .config
            .template
            .paths
            .state_root
            .join("acp")
            .join(session_id.0.as_ref());
        config
    }

    fn model_state(&self, current: &str) -> acp::SessionModelState {
        let mut models = vec![current.to_string()];
        for model in &self.config.models {
            if !models.iter().any(|existing| existing == model) {
                models.push(model.clone());
            }
        }
        acp::SessionModelState::new(
            current.to_string(),
            models
                .into_iter()
                .map(|model| acp::ModelInfo::new(model.clone(), model))
                .collect(),
        )
    }

    fn model_allowed(&self, model: &str) -> bool {
        model == self.config.template.routing.model
            || self
                .config
                .models
                .iter()
                .any(|candidate| candidate == model)
    }

    async fn replay_messages(
        &self,
        session_id: &acp::SessionId,
        messages: &[Message],
    ) -> acp::Result<()> {
        let client = self
            .client
            .get()
            .ok_or_else(|| acp::Error::internal_error().data("ACP connection not ready"))?;
        for message in messages {
            let text = message.text();
            if text.is_empty() || message.role == Role::System {
                continue;
            }
            let update = match message.role {
                Role::User => acp::SessionUpdate::UserMessageChunk(acp::ContentChunk::new(
                    acp::ContentBlock::Text(acp::TextContent::new(text)),
                )),
                Role::Assistant => acp::SessionUpdate::AgentMessageChunk(acp::ContentChunk::new(
                    acp::ContentBlock::Text(acp::TextContent::new(text)),
                )),
                Role::System | Role::Tool => continue,
            };
            acp::Client::session_notification(
                client.as_ref(),
                acp::SessionNotification::new(session_id.clone(), update),
            )
            .await
            .map_err(|error| {
                eprintln!("hi-shell: replaying loaded session failed: {error}");
                acp::Error::internal_error().data("replaying session failed")
            })?;
        }
        Ok(())
    }

    async fn session(&self, id: &acp::SessionId) -> acp::Result<Arc<Session>> {
        self.sessions
            .lock()
            .await
            .get(id)
            .cloned()
            .filter(|session| !session.closed.load(Ordering::Acquire))
            .ok_or_else(|| acp::Error::invalid_params().data("unknown session id"))
    }
}

#[async_trait::async_trait(?Send)]
impl acp::Agent for HiShell {
    async fn initialize(
        &self,
        args: acp::InitializeRequest,
    ) -> acp::Result<acp::InitializeResponse> {
        let _ = args;
        Ok(acp::InitializeResponse::new(acp::ProtocolVersion::LATEST)
            .agent_capabilities(
                acp::AgentCapabilities::new()
                    .load_session(true)
                    .session_capabilities(
                        acp::SessionCapabilities::new().close(acp::SessionCloseCapabilities::new()),
                    )
                    .prompt_capabilities(
                        acp::PromptCapabilities::new()
                            .image(false)
                            .audio(false)
                            .embedded_context(false),
                    ),
            )
            .agent_info(acp::Implementation::new("hi", env!("CARGO_PKG_VERSION"))))
    }

    async fn authenticate(
        &self,
        _args: acp::AuthenticateRequest,
    ) -> acp::Result<acp::AuthenticateResponse> {
        Ok(acp::AuthenticateResponse::new())
    }

    async fn new_session(
        &self,
        args: acp::NewSessionRequest,
    ) -> acp::Result<acp::NewSessionResponse> {
        if !args.mcp_servers.is_empty() {
            return Err(acp::Error::invalid_params().data("MCP servers are not supported"));
        }
        let cwd = args
            .cwd
            .canonicalize()
            .map_err(|error| acp::Error::invalid_params().data(error.to_string()))?;
        if !cwd.is_dir() {
            return Err(acp::Error::invalid_params().data("session cwd must be a directory"));
        }
        let session_id = acp::SessionId::new(uuid::Uuid::new_v4().to_string());
        let agent = Agent::new(
            self.config.provider.clone(),
            self.config_for(&cwd, &session_id),
        )
        .map_err(|error| acp::Error::internal_error().data(format!("{error:#}")))?;
        self.sessions.lock().await.insert(
            session_id.clone(),
            Arc::new(Session {
                agent: Mutex::new(agent),
                active_turn: Mutex::new(None),
                closed: AtomicBool::new(false),
            }),
        );
        Ok(acp::NewSessionResponse::new(session_id)
            .modes(mode_state(ToolMode::Auto))
            .models(self.model_state(&self.config.template.routing.model)))
    }

    async fn prompt(&self, args: acp::PromptRequest) -> acp::Result<acp::PromptResponse> {
        let input = prompt_text(&args.prompt)?;
        let session = self.session(&args.session_id).await?;
        let mut agent = session.agent.try_lock().map_err(|_| {
            acp::Error::invalid_params().data("session already has an active prompt")
        })?;
        let cancellation = TurnCancellation::new();
        let mut active_turn = session.active_turn.lock().await;
        if active_turn.is_some() {
            return Err(acp::Error::invalid_params().data("session already has an active prompt"));
        }
        *active_turn = Some(cancellation.clone());
        drop(active_turn);
        let client = match self.client.get().cloned() {
            Some(client) => client,
            None => {
                session.active_turn.lock().await.take();
                return Err(acp::Error::internal_error().data("ACP connection not ready"));
            }
        };
        let mut ui = AcpUi::new(client, args.session_id.clone(), cancellation.clone());
        let result = agent
            .run_turn_cancellable(&input, &mut ui, cancellation)
            .await;
        session.active_turn.lock().await.take();
        self.snapshots.lock().await.insert(
            args.session_id.clone(),
            StoredSession {
                cwd: agent.workspace_root().to_path_buf(),
                snapshot: agent.session_snapshot(),
                model: agent.model().to_string(),
                mode: agent.tool_mode(),
            },
        );
        match result {
            Ok(outcome) => {
                if let Some(status) = stop_status(outcome.stop_reason) {
                    ui.status(status);
                }
                ui.flush().await?;
                Ok(acp::PromptResponse::new(stop_reason(outcome.stop_reason)))
            }
            Err(error) => {
                let delivery = ui.flush().await;
                eprintln!("hi-shell: agent turn failed: {error:#}");
                delivery?;
                Err(acp::Error::internal_error().data("agent turn failed"))
            }
        }
    }

    async fn cancel(&self, args: acp::CancelNotification) -> acp::Result<()> {
        let session = self.sessions.lock().await.get(&args.session_id).cloned();
        if let Some(session) = session {
            if let Some(cancellation) = session.active_turn.lock().await.as_ref() {
                cancellation.cancel();
            }
        }
        Ok(())
    }

    async fn load_session(
        &self,
        args: acp::LoadSessionRequest,
    ) -> acp::Result<acp::LoadSessionResponse> {
        if !args.mcp_servers.is_empty() {
            return Err(acp::Error::invalid_params().data("MCP servers are not supported"));
        }
        if self.sessions.lock().await.contains_key(&args.session_id) {
            return Err(acp::Error::invalid_params().data("session is already loaded"));
        }
        let cwd = args
            .cwd
            .canonicalize()
            .map_err(|error| acp::Error::invalid_params().data(error.to_string()))?;
        let stored = self
            .snapshots
            .lock()
            .await
            .get(&args.session_id)
            .cloned()
            .ok_or_else(|| acp::Error::invalid_params().data("unknown session id"))?;
        if cwd != stored.cwd {
            return Err(acp::Error::invalid_params().data("session cwd does not match"));
        }
        let agent = Agent::resume_snapshot(
            self.config.provider.clone(),
            self.config_for(&cwd, &args.session_id),
            stored.snapshot,
        )
        .map_err(|error| acp::Error::internal_error().data(format!("{error:#}")))?;
        let messages = agent.session_snapshot().messages;
        self.sessions.lock().await.insert(
            args.session_id.clone(),
            Arc::new(Session {
                agent: Mutex::new(agent),
                active_turn: Mutex::new(None),
                closed: AtomicBool::new(false),
            }),
        );
        self.replay_messages(&args.session_id, &messages).await?;
        Ok(acp::LoadSessionResponse::new()
            .modes(mode_state(stored.mode))
            .models(self.model_state(&stored.model)))
    }

    async fn close_session(
        &self,
        args: acp::CloseSessionRequest,
    ) -> acp::Result<acp::CloseSessionResponse> {
        let session = self
            .sessions
            .lock()
            .await
            .remove(&args.session_id)
            .ok_or_else(|| acp::Error::invalid_params().data("unknown session id"))?;
        session.closed.store(true, Ordering::Release);
        if let Some(cancellation) = session.active_turn.lock().await.as_ref() {
            cancellation.cancel();
        }
        let agent = session.agent.lock().await;
        self.snapshots.lock().await.insert(
            args.session_id,
            StoredSession {
                cwd: agent.workspace_root().to_path_buf(),
                snapshot: agent.session_snapshot(),
                model: agent.model().to_string(),
                mode: agent.tool_mode(),
            },
        );
        agent.kill_background_processes();
        Ok(acp::CloseSessionResponse::new())
    }

    async fn set_session_mode(
        &self,
        args: acp::SetSessionModeRequest,
    ) -> acp::Result<acp::SetSessionModeResponse> {
        let mode = mode_from_id(args.mode_id.0.as_ref())?;
        let session = self.session(&args.session_id).await?;
        let mut agent = session.agent.try_lock().map_err(|_| {
            acp::Error::invalid_params().data("cannot change mode during an active prompt")
        })?;
        agent.set_tool_mode(mode);
        Ok(acp::SetSessionModeResponse::new())
    }

    async fn set_session_model(
        &self,
        args: acp::SetSessionModelRequest,
    ) -> acp::Result<acp::SetSessionModelResponse> {
        if !self.model_allowed(args.model_id.0.as_ref()) {
            return Err(acp::Error::invalid_params().data("unknown session model"));
        }
        let session = self.session(&args.session_id).await?;
        let mut agent = session.agent.try_lock().map_err(|_| {
            acp::Error::invalid_params().data("cannot change model during an active prompt")
        })?;
        agent.set_model(args.model_id.0.to_string(), None, None);
        Ok(acp::SetSessionModelResponse::new())
    }
}

fn mode_state(mode: ToolMode) -> acp::SessionModeState {
    acp::SessionModeState::new(
        mode_id(mode),
        vec![
            acp::SessionMode::new("auto", "Code"),
            acp::SessionMode::new("read-only", "Ask"),
            acp::SessionMode::new("chat-only", "Chat"),
        ],
    )
}

fn mode_id(mode: ToolMode) -> &'static str {
    match mode {
        ToolMode::Auto | ToolMode::Required => "auto",
        ToolMode::ReadOnly => "read-only",
        ToolMode::ChatOnly => "chat-only",
    }
}

fn mode_from_id(id: &str) -> acp::Result<ToolMode> {
    match id {
        "auto" => Ok(ToolMode::Auto),
        "read-only" => Ok(ToolMode::ReadOnly),
        "chat-only" => Ok(ToolMode::ChatOnly),
        _ => Err(acp::Error::invalid_params().data("unknown session mode")),
    }
}

async fn wait_for_cancellation(cancellation: TurnCancellation) {
    while !cancellation.is_cancelled() {
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
}

fn prompt_text(blocks: &[acp::ContentBlock]) -> acp::Result<String> {
    let mut text = Vec::new();
    for block in blocks {
        match block {
            acp::ContentBlock::Text(value) => text.push(value.text.clone()),
            acp::ContentBlock::ResourceLink(_) => {
                return Err(acp::Error::invalid_params().data("resource links are not supported"));
            }
            acp::ContentBlock::Resource(_) => {
                return Err(
                    acp::Error::invalid_params().data("embedded resources are not supported")
                );
            }
            _ => return Err(acp::Error::invalid_params().data("only text prompts are supported")),
        }
    }
    if text.is_empty() || text.iter().all(|value| value.trim().is_empty()) {
        return Err(acp::Error::invalid_params().data("prompt must contain non-whitespace text"));
    }
    Ok(text.join("\n"))
}

fn stop_reason(reason: hi_agent::TurnStopReason) -> acp::StopReason {
    match reason {
        hi_agent::TurnStopReason::Cancelled => acp::StopReason::Cancelled,
        hi_agent::TurnStopReason::StepLimit => acp::StopReason::MaxTokens,
        _ => acp::StopReason::EndTurn,
    }
}

fn stop_status(reason: hi_agent::TurnStopReason) -> Option<&'static str> {
    match reason {
        hi_agent::TurnStopReason::VerificationUnavailable => Some("verification unavailable"),
        hi_agent::TurnStopReason::VerificationFailed => Some("verification failed"),
        hi_agent::TurnStopReason::VerificationUnstable => Some("verification was unstable"),
        hi_agent::TurnStopReason::ReviewObjected => Some("independent review objected"),
        hi_agent::TurnStopReason::ToolModeDenied => Some("required tool use was denied"),
        hi_agent::TurnStopReason::Stalled => Some("the turn stalled"),
        hi_agent::TurnStopReason::InfrastructureFailure => Some("the turn failed internally"),
        _ => None,
    }
}

struct AcpUi {
    client: Arc<acp::AgentSideConnection>,
    session_id: acp::SessionId,
    events: mpsc::Sender<AcpUiEvent>,
    delivery_failed: Arc<AtomicBool>,
    next_permission_id: u64,
    cancellation: TurnCancellation,
}

enum AcpUiEvent {
    Update(acp::SessionUpdate),
    Flush(oneshot::Sender<Result<(), String>>),
}

impl AcpUi {
    fn new(
        client: Arc<acp::AgentSideConnection>,
        session_id: acp::SessionId,
        cancellation: TurnCancellation,
    ) -> Self {
        let (events, mut receiver) = mpsc::channel(256);
        let event_client = client.clone();
        let event_session_id = session_id.clone();
        let delivery_failed = Arc::new(AtomicBool::new(false));
        let worker_failed = delivery_failed.clone();
        tokio::task::spawn_local(async move {
            let mut delivery_error = None;
            while let Some(event) = receiver.recv().await {
                match event {
                    AcpUiEvent::Update(update) if delivery_error.is_none() => {
                        let notification =
                            acp::SessionNotification::new(event_session_id.clone(), update);
                        if let Err(error) =
                            acp::Client::session_notification(event_client.as_ref(), notification)
                                .await
                        {
                            worker_failed.store(true, Ordering::Release);
                            delivery_error = Some(error.to_string());
                        }
                    }
                    AcpUiEvent::Update(_) => {}
                    AcpUiEvent::Flush(response) => {
                        let result = delivery_error.clone().map_or(Ok(()), Err);
                        let _ = response.send(result);
                    }
                }
            }
        });
        Self {
            client,
            session_id,
            events,
            delivery_failed,
            next_permission_id: 0,
            cancellation,
        }
    }

    fn send(&self, update: acp::SessionUpdate) {
        if self.delivery_failed.load(Ordering::Acquire) {
            return;
        }
        match self.events.try_send(AcpUiEvent::Update(update)) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(_)) => {
                self.delivery_failed.store(true, Ordering::Release);
                eprintln!("hi-shell: ACP event buffer exhausted");
            }
            Err(_) => {
                self.delivery_failed.store(true, Ordering::Release);
            }
        }
    }

    async fn flush(&self) -> acp::Result<()> {
        if self.delivery_failed.load(Ordering::Acquire) {
            return Err(acp::Error::internal_error().data("delivering session update failed"));
        }
        let (sender, receiver) = oneshot::channel();
        self.events
            .send(AcpUiEvent::Flush(sender))
            .await
            .map_err(|_| acp::Error::internal_error().data("ACP event stream closed"))?;
        receiver
            .await
            .map_err(|_| acp::Error::internal_error().data("ACP event stream closed"))?
            .map_err(|error| {
                eprintln!("hi-shell: delivering ACP session update failed: {error}");
                acp::Error::internal_error().data("delivering session update failed")
            })
    }

    fn text(&self, text: &str, thought: bool) {
        let chunk = acp::ContentChunk::new(acp::ContentBlock::Text(acp::TextContent::new(text)));
        self.send(if thought {
            acp::SessionUpdate::AgentThoughtChunk(chunk)
        } else {
            acp::SessionUpdate::AgentMessageChunk(chunk)
        });
    }
}

impl Ui for AcpUi {
    fn assistant_text(&mut self, text: &str) {
        self.text(text, false);
    }
    fn assistant_reasoning(&mut self, text: &str) {
        self.text(text, true);
    }
    fn assistant_end(&mut self) {}
    fn confirm(
        &mut self,
        request: hi_agent::ConfirmationRequest,
    ) -> hi_agent::ConfirmationFuture<'_> {
        let client = self.client.clone();
        let session_id = self.session_id.clone();
        let cancellation = self.cancellation.clone();
        self.next_permission_id += 1;
        let id = acp::ToolCallId::new(format!("permission-{}", self.next_permission_id));
        let tool_call = acp::ToolCallUpdate::new(
            id,
            acp::ToolCallUpdateFields::new()
                .title(request.title().to_string())
                .raw_input(serde_json::Value::String(request.details())),
        );
        let options = vec![
            acp::PermissionOption::new(
                "allow-once",
                "Allow once",
                acp::PermissionOptionKind::AllowOnce,
            ),
            acp::PermissionOption::new(
                "reject-once",
                "Reject",
                acp::PermissionOptionKind::RejectOnce,
            ),
        ];
        let (sender, receiver) = oneshot::channel();
        tokio::task::spawn_local(async move {
            let response = tokio::select! {
                response = acp::Client::request_permission(
                    client.as_ref(),
                    acp::RequestPermissionRequest::new(session_id, tool_call, options),
                ) => Some(response),
                _ = wait_for_cancellation(cancellation) => None,
            };
            let result = match response {
                None => hi_agent::ConfirmationResult::Cancelled,
                Some(response) => match response.map(|response| response.outcome) {
                    Ok(acp::RequestPermissionOutcome::Selected(selected))
                        if selected.option_id.0.as_ref() == "allow-once" =>
                    {
                        hi_agent::ConfirmationResult::Approved
                    }
                    Ok(acp::RequestPermissionOutcome::Selected(_)) => {
                        hi_agent::ConfirmationResult::Rejected
                    }
                    Ok(acp::RequestPermissionOutcome::Cancelled) => {
                        hi_agent::ConfirmationResult::Cancelled
                    }
                    _ => hi_agent::ConfirmationResult::Unavailable,
                },
            };
            let _ = sender.send(result);
        });
        Box::pin(async move {
            receiver
                .await
                .unwrap_or(hi_agent::ConfirmationResult::Unavailable)
        })
    }
    fn plan(&mut self, steps: &[hi_agent::PlanStep]) {
        let entries = steps
            .iter()
            .map(|step| {
                let status = match step.status {
                    hi_agent::PlanStatus::Pending => acp::PlanEntryStatus::Pending,
                    hi_agent::PlanStatus::Active => acp::PlanEntryStatus::InProgress,
                    hi_agent::PlanStatus::Done => acp::PlanEntryStatus::Completed,
                };
                acp::PlanEntry::new(step.title.clone(), acp::PlanEntryPriority::Medium, status)
            })
            .collect();
        self.send(acp::SessionUpdate::Plan(acp::Plan::new(entries)));
    }
    fn tool_call(&mut self, name: &str, arguments: &str) {
        self.tool_started_id(name, name, arguments);
    }
    fn tool_started_id(&mut self, id: &str, name: &str, arguments: &str) {
        self.send(acp::SessionUpdate::ToolCall(
            acp::ToolCall::new(acp::ToolCallId::new(id), name)
                .status(acp::ToolCallStatus::InProgress)
                .raw_input(
                    serde_json::from_str(arguments)
                        .unwrap_or_else(|_| serde_json::Value::String(arguments.into())),
                ),
        ));
    }
    fn tool_stream_id(&mut self, id: &str, _name: &str, line: &str) {
        self.send(acp::SessionUpdate::ToolCallUpdate(
            acp::ToolCallUpdate::new(
                acp::ToolCallId::new(id),
                acp::ToolCallUpdateFields::new().raw_output(serde_json::Value::String(line.into())),
            ),
        ));
    }
    fn tool_result(&mut self, name: &str, result: &str) {
        self.tool_result_id(name, name, result, hi_tools::ToolStatus::Succeeded);
    }
    fn tool_result_id(
        &mut self,
        id: &str,
        _name: &str,
        result: &str,
        status: hi_tools::ToolStatus,
    ) {
        let status = if status == hi_tools::ToolStatus::Succeeded {
            acp::ToolCallStatus::Completed
        } else {
            acp::ToolCallStatus::Failed
        };
        self.send(acp::SessionUpdate::ToolCallUpdate(
            acp::ToolCallUpdate::new(
                acp::ToolCallId::new(id),
                acp::ToolCallUpdateFields::new()
                    .status(status)
                    .raw_output(serde_json::Value::String(result.into())),
            ),
        ));
    }
    fn status(&mut self, text: &str) {
        self.text(text, true);
    }
    fn turn_end(&mut self, _summary: &str) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_text_rejects_empty_and_unsupported_content() {
        assert!(prompt_text(&[]).is_err());
        assert!(prompt_text(&[acp::ContentBlock::Text(acp::TextContent::new("  \n",))]).is_err());
        assert!(
            prompt_text(&[acp::ContentBlock::ResourceLink(acp::ResourceLink::new(
                "file:///tmp/context",
                "context"
            ),)])
            .is_err()
        );
    }

    #[test]
    fn prompt_text_preserves_text_order() {
        let prompt = prompt_text(&[
            acp::ContentBlock::Text(acp::TextContent::new("first")),
            acp::ContentBlock::Text(acp::TextContent::new("second")),
        ])
        .unwrap();
        assert_eq!(prompt, "first\nsecond");
    }

    #[test]
    fn stop_reasons_do_not_report_turn_limit_as_user_cancellation() {
        assert_eq!(
            stop_reason(hi_agent::TurnStopReason::Cancelled),
            acp::StopReason::Cancelled
        );
        assert_eq!(
            stop_reason(hi_agent::TurnStopReason::StepLimit),
            acp::StopReason::MaxTokens
        );
        assert_eq!(
            stop_reason(hi_agent::TurnStopReason::TurnLimit),
            acp::StopReason::EndTurn
        );
    }

    #[test]
    fn modes_are_strictly_validated() {
        assert_eq!(mode_from_id("auto").unwrap(), ToolMode::Auto);
        assert_eq!(mode_from_id("read-only").unwrap(), ToolMode::ReadOnly);
        assert_eq!(mode_from_id("chat-only").unwrap(), ToolMode::ChatOnly);
        assert!(mode_from_id("dangerous").is_err());
    }

    #[test]
    fn non_success_tool_statuses_fail_in_acp() {
        fn mapped(status: hi_tools::ToolStatus) -> acp::ToolCallStatus {
            if status == hi_tools::ToolStatus::Succeeded {
                acp::ToolCallStatus::Completed
            } else {
                acp::ToolCallStatus::Failed
            }
        }
        assert_eq!(
            mapped(hi_tools::ToolStatus::Succeeded),
            acp::ToolCallStatus::Completed
        );
        for status in [
            hi_tools::ToolStatus::Failed,
            hi_tools::ToolStatus::Denied,
            hi_tools::ToolStatus::Cancelled,
        ] {
            assert_eq!(mapped(status), acp::ToolCallStatus::Failed);
        }
    }

    #[test]
    fn turn_cancellation_is_generation_local() {
        let first = TurnCancellation::new();
        let second = TurnCancellation::new();
        first.cancel();
        assert!(first.is_cancelled());
        assert!(!second.is_cancelled());
    }

    #[test]
    fn model_allowlist_includes_template_and_explicit_models() {
        struct NeverProvider;
        #[async_trait::async_trait]
        impl Provider for NeverProvider {
            async fn stream(
                &self,
                _request: hi_ai::ChatRequest,
                _on_event: &mut (dyn FnMut(hi_ai::StreamEvent) + Send),
            ) -> anyhow::Result<hi_ai::Completion> {
                unreachable!()
            }
        }
        let mut template = AgentConfig::default();
        template.routing.model = "default-model".into();
        let shell = HiShell::new(ShellConfig {
            provider: Arc::new(NeverProvider),
            template,
            models: vec!["other-model".into()],
        });
        assert!(shell.model_allowed("default-model"));
        assert!(shell.model_allowed("other-model"));
        assert!(!shell.model_allowed("unadvertised"));
    }
}

pub async fn serve_stdio(config: ShellConfig) -> Result<()> {
    let stdin = tokio::io::stdin().compat();
    let stdout = tokio::io::stdout().compat_write();
    let shell = Arc::new(HiShell::new(config));
    let (connection, io) = acp::AgentSideConnection::new(shell.clone(), stdout, stdin, |future| {
        tokio::task::spawn_local(future);
    });
    shell.connect(Arc::new(connection));
    io.await.context("serving ACP over stdio")
}
