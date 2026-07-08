use super::*;
use async_trait::async_trait;
use hi_ai::{
    ChatRequest, Completion, Content, Provider, ProviderError, ProviderErrorKind, StreamEvent,
    Usage,
};
use std::sync::{LazyLock, Mutex};

/// A provider that returns canned completions in order.
pub(crate) struct Canned(pub(crate) Mutex<Vec<Completion>>);

#[async_trait]
impl Provider for Canned {
    async fn stream(
        &self,
        _request: ChatRequest,
        _sink: &mut (dyn FnMut(StreamEvent) + Send),
    ) -> Result<Completion> {
        Ok(self.0.lock().unwrap().remove(0))
    }
}

/// Like [`Canned`], but records each request's sampling tuple
/// `(temperature, top_p, frequency_penalty)` (shared via an `Arc` so the test
/// can inspect it after the provider is moved in).
pub(crate) type Sample = (Option<f32>, Option<f32>, Option<f32>);
pub(crate) struct RecordTemps {
    pub(crate) responses: Mutex<Vec<Completion>>,
    pub(crate) samples: std::sync::Arc<Mutex<Vec<Sample>>>,
}

#[async_trait]
impl Provider for RecordTemps {
    async fn stream(
        &self,
        request: ChatRequest,
        _sink: &mut (dyn FnMut(StreamEvent) + Send),
    ) -> Result<Completion> {
        self.samples.lock().unwrap().push((
            request.temperature,
            request.top_p,
            request.frequency_penalty,
        ));
        Ok(self.responses.lock().unwrap().remove(0))
    }
}

/// Like [`Canned`], but records each request's `tool_mode` so a test can
/// assert when the agent forces `tool_choice` (e.g. after a continue-nudge).
pub(crate) struct RecordToolModes {
    pub(crate) responses: Mutex<Vec<Completion>>,
    pub(crate) modes: std::sync::Arc<Mutex<Vec<ToolMode>>>,
}

#[async_trait]
impl Provider for RecordToolModes {
    async fn stream(
        &self,
        request: ChatRequest,
        _sink: &mut (dyn FnMut(StreamEvent) + Send),
    ) -> Result<Completion> {
        self.modes.lock().unwrap().push(request.profile.tool_mode);
        Ok(self.responses.lock().unwrap().remove(0))
    }
}

pub(crate) struct RecordRequests {
    pub(crate) responses: Mutex<Vec<Completion>>,
    pub(crate) tool_names: std::sync::Arc<Mutex<Vec<Vec<String>>>>,
    pub(crate) modes: std::sync::Arc<Mutex<Vec<ToolMode>>>,
}

#[async_trait]
impl Provider for RecordRequests {
    async fn stream(
        &self,
        request: ChatRequest,
        _sink: &mut (dyn FnMut(StreamEvent) + Send),
    ) -> Result<Completion> {
        self.tool_names
            .lock()
            .unwrap()
            .push(request.tools.iter().map(|tool| tool.name.clone()).collect());
        self.modes.lock().unwrap().push(request.profile.tool_mode);
        Ok(self.responses.lock().unwrap().remove(0))
    }
}

pub(crate) enum ProviderStep {
    Completion(Completion),
    RequestTooLarge,
    /// Fail this round with a provider error of the given kind.
    Error(ProviderErrorKind),
    ErrorMessage(ProviderErrorKind, String),
    ErrorWithUsage(ProviderErrorKind, Usage),
}

pub(crate) struct ScriptedProvider {
    pub(crate) steps: Mutex<Vec<ProviderStep>>,
    pub(crate) requests: std::sync::Arc<Mutex<Vec<Vec<Message>>>>,
    pub(crate) max_tokens: Option<std::sync::Arc<Mutex<Vec<u32>>>>,
}

#[async_trait]
impl Provider for ScriptedProvider {
    async fn stream(
        &self,
        request: ChatRequest,
        _sink: &mut (dyn FnMut(StreamEvent) + Send),
    ) -> Result<Completion> {
        self.requests
            .lock()
            .unwrap()
            .push(request.messages.to_vec());
        if let Some(max_tokens) = &self.max_tokens {
            max_tokens.lock().unwrap().push(request.max_tokens);
        }
        match self.steps.lock().unwrap().remove(0) {
            ProviderStep::Completion(completion) => Ok(completion),
            ProviderStep::RequestTooLarge => Err(ProviderError::new(
                ProviderErrorKind::RequestTooLarge,
                "API error 400 Bad Request: chat input exceeds the maximum allowed size",
            )
            .into()),
            ProviderStep::Error(kind) => {
                Err(ProviderError::new(kind, "scripted provider error").into())
            }
            ProviderStep::ErrorMessage(kind, message) => {
                Err(ProviderError::new(kind, message).into())
            }
            ProviderStep::ErrorWithUsage(kind, usage) => {
                Err(ProviderError::new(kind, "scripted provider error")
                    .with_usage(usage)
                    .into())
            }
        }
    }
}

pub(crate) struct NullUi;
impl Ui for NullUi {
    fn assistant_text(&mut self, _: &str) {}
    fn assistant_reasoning(&mut self, _: &str) {}
    fn assistant_end(&mut self) {}
    fn tool_call(&mut self, _: &str, _: &str) {}
    fn tool_result(&mut self, _: &str, _: &str) {}
    fn status(&mut self, _: &str) {}
    fn turn_end(&mut self, _: &str) {}
}

pub(crate) type UsageRecords = std::sync::Arc<Mutex<Vec<Usage>>>;

pub(crate) struct RecordingSession {
    pub(crate) records: UsageRecords,
}

impl SessionSink for RecordingSession {
    fn record(&mut self, _messages: &[Message], usage: Usage) -> Result<()> {
        self.records.lock().unwrap().push(usage);
        Ok(())
    }

    fn record_compaction(&mut self, _messages: &[Message]) -> Result<()> {
        Ok(())
    }
}

#[derive(Default)]
pub(crate) struct RecordingUi {
    pub(crate) statuses: Vec<String>,
    pub(crate) turn_ends: Vec<String>,
}

impl Ui for RecordingUi {
    fn assistant_text(&mut self, _: &str) {}
    fn assistant_reasoning(&mut self, _: &str) {}
    fn assistant_end(&mut self) {}
    fn tool_call(&mut self, _: &str, _: &str) {}
    fn tool_result(&mut self, _: &str, _: &str) {}
    fn status(&mut self, s: &str) {
        self.statuses.push(s.to_string());
    }
    fn nudge(&mut self, s: &str) {
        self.statuses.push(s.to_string());
    }
    fn turn_end(&mut self, s: &str) {
        self.turn_ends.push(s.to_string());
    }
}

pub(crate) fn config() -> AgentConfig {
    AgentConfig {
        model: "m".into(),
        requested_max_tokens: 100,
        max_tokens: 100,
        max_tokens_explicit: true,
        max_verify_iterations: 2,
        auto_compact: false,
        // Default to summarize so the existing summarize/auto tests are
        // unaffected; hybrid/elide get dedicated tests.
        compaction: CompactionKind::Summarize,
        // Off by default so the canned-provider tests don't need an extra
        // completion for the recap; the finalization tests opt in.
        finalize: false,
        // Off so canned-provider tests don't need extra completions for the
        // silent auto-continue; tests that exercise it opt in.
        max_silent_continues: 0,
        // Most canned-provider tests assert specific nudge behavior before
        // any deterministic context is added. Preflight has dedicated tests.
        read_only_preflight: false,
        ..AgentConfig::default()
    }
}

pub(crate) fn completion(content: Vec<Content>, input: u64, output: u64) -> Completion {
    Completion {
        content,
        usage: Usage {
            input_tokens: input,
            output_tokens: output,
            ..Default::default()
        },
        stop_reason: None,
    }
}

pub(crate) fn agent(responses: Vec<Completion>, cfg: AgentConfig) -> Agent {
    Agent::new(std::sync::Arc::new(Canned(Mutex::new(responses))), cfg)
}

pub(crate) fn resumed_agent(
    history: Vec<Message>,
    usage: Usage,
    structured_goal: Option<Goal>,
    cfg: AgentConfig,
) -> Agent {
    Agent::resume(
        std::sync::Arc::new(Canned(Mutex::new(Vec::new()))),
        cfg,
        history,
        usage,
        Vec::new(),
        structured_goal,
        DecisionLog::default(),
    )
}

pub(crate) fn scripted_agent(
    steps: Vec<ProviderStep>,
    cfg: AgentConfig,
) -> (Agent, std::sync::Arc<Mutex<Vec<Vec<Message>>>>) {
    let requests = std::sync::Arc::new(Mutex::new(Vec::new()));
    let provider = ScriptedProvider {
        steps: Mutex::new(steps),
        requests: requests.clone(),
        max_tokens: None,
    };
    (Agent::new(std::sync::Arc::new(provider), cfg), requests)
}

#[allow(clippy::type_complexity)]
pub(crate) fn scripted_agent_recording_max_tokens(
    steps: Vec<ProviderStep>,
    cfg: AgentConfig,
) -> (
    Agent,
    std::sync::Arc<Mutex<Vec<Vec<Message>>>>,
    std::sync::Arc<Mutex<Vec<u32>>>,
) {
    let requests = std::sync::Arc::new(Mutex::new(Vec::new()));
    let max_tokens = std::sync::Arc::new(Mutex::new(Vec::new()));
    let provider = ScriptedProvider {
        steps: Mutex::new(steps),
        requests: requests.clone(),
        max_tokens: Some(max_tokens.clone()),
    };
    (
        Agent::new(std::sync::Arc::new(provider), cfg),
        requests,
        max_tokens,
    )
}

pub(crate) static VERIFY_TEST_LOCK: LazyLock<tokio::sync::Mutex<()>> =
    LazyLock::new(|| tokio::sync::Mutex::new(()));

/// A completion that writes a throwaway file — marks the turn as having
/// edited, so the (edit-gated) verification pipeline runs.
pub(crate) fn write_completion(path: &str) -> Completion {
    completion(
        vec![Content::ToolCall {
            id: "w".into(),
            name: "write".into(),
            arguments: format!("{{\"path\":{path:?},\"content\":\"x\"}}"),
        }],
        1,
        1,
    )
}

pub(crate) fn bash_completion(command: &str) -> Completion {
    completion(
        vec![Content::ToolCall {
            id: "b".into(),
            name: "bash".into(),
            arguments: serde_json::json!({ "command": command }).to_string(),
        }],
        1,
        1,
    )
}

/// A unique throwaway file path under the current workspace. The name is
/// unique per *call* (not just per process), so concurrent test runs and
/// repeated calls within one process never collide — and a file left
/// behind by a test that panicked before cleanup doesn't get clobbered
/// or mistaken for another test's artifact. The file lives in the
/// workspace (cwd) on purpose: the verify snapshot walks `.` to detect
/// changes, so the temp file must be inside it for verify to notice.
pub(crate) fn temp_file(tag: &str) -> std::path::PathBuf {
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::current_dir()
        .unwrap()
        .join(format!("hi-test-{tag}-{}-{n}", std::process::id()))
}

#[derive(Default)]
pub(crate) struct RecUi {
    pub(crate) statuses: Vec<String>,
    pub(crate) usages: Vec<(u64, u64)>,
    pub(crate) rate_limits: Vec<Option<hi_ai::RateLimitState>>,
    pub(crate) turn_end: Option<String>,
    pub(crate) assistant: String,
    pub(crate) tool_results: Vec<(String, String)>,
}
impl Ui for RecUi {
    fn assistant_text(&mut self, t: &str) {
        self.assistant.push_str(t);
    }
    fn assistant_reasoning(&mut self, _: &str) {}
    fn assistant_end(&mut self) {}
    fn tool_call(&mut self, _: &str, _: &str) {}
    fn tool_result(&mut self, name: &str, result: &str) {
        self.tool_results
            .push((name.to_string(), result.to_string()));
    }
    fn status(&mut self, t: &str) {
        self.statuses.push(t.to_string());
    }
    fn nudge(&mut self, t: &str) {
        // Steering diagnostics share the status capture so tests can assert on
        // them, even though real frontends ignore `nudge`.
        self.statuses.push(t.to_string());
    }
    fn usage(
        &mut self,
        input_tokens: u64,
        output_tokens: u64,
        _ctx_used: u64,
        _ctx_win: Option<u32>,
    ) {
        self.usages.push((input_tokens, output_tokens));
    }
    fn rate_limits(&mut self, rate_limits: Option<hi_ai::RateLimitState>) {
        self.rate_limits.push(rate_limits);
    }
    fn turn_end(&mut self, summary: &str) {
        self.turn_end = Some(summary.to_string());
    }
    fn turn_error(&mut self, kind: &str, message: &str, guidance: &str) {
        let suffix = if guidance.is_empty() {
            String::new()
        } else {
            format!(" — {guidance}")
        };
        self.statuses.push(format!("{kind}: {message}{suffix}"));
    }
}

#[derive(Default)]
pub(crate) struct SplitUi {
    pub(crate) statuses: Vec<String>,
    pub(crate) nudges: Vec<String>,
    pub(crate) turn_end: Option<String>,
}

impl Ui for SplitUi {
    fn assistant_text(&mut self, _: &str) {}
    fn assistant_reasoning(&mut self, _: &str) {}
    fn assistant_end(&mut self) {}
    fn tool_call(&mut self, _: &str, _: &str) {}
    fn tool_result(&mut self, _: &str, _: &str) {}
    fn status(&mut self, t: &str) {
        self.statuses.push(t.to_string());
    }
    fn nudge(&mut self, t: &str) {
        self.nudges.push(t.to_string());
    }
    fn turn_end(&mut self, s: &str) {
        self.turn_end = Some(s.to_string());
    }
}

/// A harmless tool-call round (runs `echo`), marking the turn as actively
/// working so a later text-only stop is nudge-eligible.
pub(crate) fn echo_call() -> Completion {
    completion(
        vec![Content::ToolCall {
            id: "t".into(),
            name: "bash".into(),
            arguments: "{\"command\":\"echo hi\"}".into(),
        }],
        1,
        1,
    )
}
