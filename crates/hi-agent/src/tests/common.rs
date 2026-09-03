use super::*;
use async_trait::async_trait;
use hi_ai::{
    ChatRequest, Completion, Content, Provider, ProviderError, ProviderErrorKind, StreamEvent,
    Usage,
};
use std::cell::RefCell;
use std::sync::{Arc, Mutex, Weak};

fn test_workspace() -> Arc<tempfile::TempDir> {
    // libtest reuses worker threads, and a background task can briefly retain
    // an older fixture after its test returns. Keying the weak lease by the
    // current test name prevents that late owner from making a later test
    // observe the old workspace.
    let test_name = std::thread::current()
        .name()
        .unwrap_or("unnamed-test")
        .to_string();
    TEST_WORKSPACE.with(|slot| {
        let mut slot = slot.borrow_mut();
        if let Some((owner, workspace)) = slot.as_ref()
            && owner == &test_name
            && let Some(existing) = workspace.upgrade()
        {
            return existing;
        }
        let workspace = Arc::new(
            tempfile::Builder::new()
                .prefix("hi-agent-workspace-")
                .tempdir()
                .expect("create temporary test workspace"),
        );
        *slot = Some((test_name, Arc::downgrade(&workspace)));
        workspace
    })
}

// Keep path fixtures and the agent created later in the same test on one
// temporary workspace, while the weak reference lets the whole fixture drop
// when the test's last owner disappears. Each libtest worker has its own slot,
// so parallel tests remain isolated.
thread_local! {
    static TEST_WORKSPACE: RefCell<Option<(String, Weak<tempfile::TempDir>)>> = const { RefCell::new(None) };
}

/// A provider that returns canned completions in order.
pub(crate) struct Canned(pub(crate) Mutex<Vec<Completion>>);

pub(crate) fn pop_canned_completion(
    responses: &Mutex<Vec<Completion>>,
    provider: &str,
) -> Result<Completion> {
    let mut responses = responses.lock().unwrap();
    if responses.is_empty() {
        anyhow::bail!(
            "{provider} exhausted: test scripted fewer completions than the agent requested \
(often an extra repair/nudge round from a failed bash/validation step under parallel load)"
        );
    }
    Ok(responses.remove(0))
}

#[async_trait]
impl Provider for Canned {
    async fn stream(
        &self,
        _request: ChatRequest,
        _sink: &mut (dyn FnMut(StreamEvent) + Send),
    ) -> Result<Completion> {
        pop_canned_completion(&self.0, "Canned")
    }
}

/// Canned provider that also emits text through the streaming sink. Most unit
/// providers return only the final `Completion`, which cannot verify that a
/// rejected draft stayed hidden from the live UI.
pub(crate) struct StreamingCanned(pub(crate) Mutex<Vec<Completion>>);

#[async_trait]
impl Provider for StreamingCanned {
    async fn stream(
        &self,
        _request: ChatRequest,
        sink: &mut (dyn FnMut(StreamEvent) + Send),
    ) -> Result<Completion> {
        let completion = pop_canned_completion(&self.0, "StreamingCanned")?;
        for content in &completion.content {
            if let Content::Text(text) = content {
                sink(StreamEvent::Text(text.clone()));
            }
        }
        Ok(completion)
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
        pop_canned_completion(&self.responses, "RecordTemps")
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
        pop_canned_completion(&self.responses, "RecordToolModes")
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
        pop_canned_completion(&self.responses, "RecordRequests")
    }
}

pub(crate) enum ProviderStep {
    Completion(Completion),
    /// Complete successfully after a deterministic delay. Used to prove that
    /// agent-side policy does not impose an implicit short deadline.
    DelayedCompletion(std::time::Duration, Completion),
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
        let step = {
            let mut steps = self.steps.lock().unwrap();
            if steps.is_empty() {
                anyhow::bail!(
                    "ScriptedProvider exhausted: test scripted fewer steps than the agent requested \
(often an extra repair/nudge round from a failed bash/validation step under parallel load)"
                );
            }
            steps.remove(0)
        };
        match step {
            ProviderStep::Completion(completion) => Ok(completion),
            ProviderStep::DelayedCompletion(delay, completion) => {
                tokio::time::sleep(delay).await;
                Ok(completion)
            }
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
    // ProcessRunner shells may reject macOS sandbox-exec in this test host
    // (exit 71). Pass the policy directly to the runtime; mutating the
    // process-wide environment here races with parallel tests and can abort.
    let state_guard = tempfile::Builder::new()
        .prefix("hi-agent-state-")
        .tempdir()
        .expect("create temporary agent state root");
    let state_root = state_guard.path().to_path_buf();
    let test_workspace_root = Some(test_workspace());
    let workspace_root = test_workspace_root
        .as_ref()
        .expect("test workspace should exist")
        .path()
        .to_path_buf();
    AgentConfig {
        paths: crate::AgentPaths {
            workspace_root,
            state_root,
        },
        routing: crate::AgentRouting {
            model: "m".into(),
            requested_max_tokens: 100,
            max_tokens: 100,
            max_tokens_explicit: true,
            ..crate::AgentRouting::default()
        },
        gates: crate::AgentGates {
            max_verify_repairs: 1,
            verification: crate::VerificationMode::Disabled,
            // The common canned-agent fixture does not exercise language
            // servers. Keep them off so a long single-process test run cannot
            // accumulate native LSP children; focused LSP tests opt in.
            lsp_mode: crate::LspMode::Off,
            // Most canned-provider tests assert specific nudge behavior before
            // any deterministic context is added. Preflight has dedicated tests.
            read_only_preflight: false,
            // Missing checkpoints follow the production YOLO default. Tests that
            // exercise strict checkpoint requirements opt out explicitly.
            allow_no_checkpoint: true,
            ..crate::AgentGates::default()
        },
        loop_limits: crate::AgentLoopLimits {
            // Off so canned-provider tests don't need extra completions for the
            // silent auto-continue; tests that exercise it opt in.
            max_silent_continues: 0,
            max_keep_working: 0,
            ..crate::AgentLoopLimits::default()
        },
        memory: crate::AgentMemory {
            auto_compact: false,
            // Default to summarize so the existing summarize/auto tests are
            // unaffected; hybrid/elide get dedicated tests.
            compaction: CompactionKind::Summarize,
            // Off by default so the canned-provider tests don't need an extra
            // completion for the recap; the finalization tests opt in.
            finalize: false,
            // Off so canned turns don't need an extra completion for ghost text.
            suggest_next_prompt: false,
            // Off so token-budget and message-shape tests aren't perturbed by the
            // rust-workspace stack pack body (this repo has a Cargo.toml). Tests
            // that assert the injection opt in.
            inject_stack_skill: false,
            // Off so review-shaped canned turns don't grow by the code-review pack.
            inject_review_skill: false,
            ..crate::AgentMemory::default()
        },
        test_state_root: Some(std::sync::Arc::new(state_guard)),
        _test_workspace_root: test_workspace_root,
        sandbox_policy: Some(hi_tools::sandbox::SandboxPolicy::Off),
        subagents: crate::AgentSubagents {
            // Tests that need explore/delegate opt in; keep the base config quiet so
            // canned tool lists stay predictable.
            explore_subagents: false,
            write_subagents: crate::WriteSubagentPolicy::Off,
            ..crate::AgentSubagents::default()
        },
        ..AgentConfig::default()
    }
}

#[test]
fn test_agent_state_is_owned_by_a_temporary_directory() {
    let cfg = config();
    let state_root = cfg.paths.state_root.clone();
    let guard = cfg
        .test_state_root
        .clone()
        .expect("test config owns its state directory");

    assert!(state_root.starts_with(std::env::temp_dir()));
    assert!(!state_root.starts_with(std::env::current_dir().unwrap()));
    std::fs::write(state_root.join("state.json"), "temporary").unwrap();

    drop(cfg);
    drop(guard);
    assert!(!state_root.exists(), "temporary state root was not removed");
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
        ..Completion::default()
    }
}

pub(crate) fn agent(responses: Vec<Completion>, cfg: AgentConfig) -> Agent {
    Agent::new(std::sync::Arc::new(Canned(Mutex::new(responses))), cfg).unwrap()
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
    .unwrap()
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
    (
        Agent::new(std::sync::Arc::new(provider), cfg).unwrap(),
        requests,
    )
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
        Agent::new(std::sync::Arc::new(provider), cfg).unwrap(),
        requests,
        max_tokens,
    )
}

/// A completion that writes a throwaway file — marks the turn as having
/// edited, so the (edit-gated) verification pipeline runs.
pub(crate) fn write_completion(path: &str) -> Completion {
    write_content_completion(path, "x")
}

/// Like [`write_completion`] but with a caller-sized body. Tests that exercise
/// the `/goal team` skeptic gate need the write to clear the gate's
/// trivial-diff exemption ([`crate::goal::SKEPTIC_TRIVIAL_DIFF_BYTES`]) — a
/// one-byte write skips the review they're scripting.
pub(crate) fn write_content_completion(path: &str, content: &str) -> Completion {
    completion(
        vec![Content::ToolCall {
            id: "w".into(),
            name: "write".into(),
            arguments: format!("{{\"path\":{path:?},\"content\":{content:?}}}"),
        }],
        1,
        1,
    )
}

/// Whether `python3 -m py_compile` can actually run here — not just whether a
/// `python3` binary is on PATH. Sandboxed environments (macOS seatbelt,
/// containers without a writable bytecode cache) make py_compile exit non-zero
/// trying to write `__pycache__`, which would fail proactive-verify tests for
/// reasons unrelated to the agent. Preflight in `dir` with `sys.dont_write_bytecode`
/// so the probe itself leaves no cache behind.
pub(crate) fn python_fast_check_works(dir: &std::path::Path) -> bool {
    let probe = dir.join("hi_pycompile_probe.py");
    if std::fs::write(&probe, "ok = True\n").is_err() {
        return false;
    }
    let works = std::process::Command::new("python3")
        .args(["-m", "py_compile"])
        .arg(&probe)
        .current_dir(dir)
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    let _ = std::fs::remove_file(&probe);
    let _ = std::fs::remove_dir_all(dir.join("__pycache__"));
    works
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

/// A unique throwaway path in an RAII-owned temporary directory.
///
/// The path is intentionally not created: callers use it for both files and
/// directories. The owning [`tempfile::TempDir`] removes the whole fixture,
/// including anything a test creates beneath it, even when the test panics.
pub(crate) struct TempTestPath {
    _workspace: Arc<tempfile::TempDir>,
    path: std::path::PathBuf,
}

impl AsRef<std::path::Path> for TempTestPath {
    fn as_ref(&self) -> &std::path::Path {
        &self.path
    }
}

impl std::ops::Deref for TempTestPath {
    type Target = std::path::Path;

    fn deref(&self) -> &Self::Target {
        &self.path
    }
}

pub(crate) fn temp_file(tag: &str) -> TempTestPath {
    let workspace = test_workspace();
    let path = workspace.path().join(unique_name(tag));
    TempTestPath {
        _workspace: workspace,
        path,
    }
}

/// A deterministic path beneath the same RAII-owned workspace used by
/// [`config`]. This is useful when a test exercises repository conventions
/// such as `Cargo.toml` or `README.md` discovery.
pub(crate) fn temp_workspace_path(name: &str) -> TempTestPath {
    let workspace = test_workspace();
    TempTestPath {
        _workspace: workspace.clone(),
        path: workspace.path().join(name),
    }
}

#[test]
fn test_file_fixture_is_outside_repository_and_removed_with_its_owner() {
    let repository = std::env::current_dir().unwrap();
    let fixture = temp_file("fixture-cleanup");
    let path = fixture.as_ref().to_path_buf();

    assert!(path.starts_with(std::env::temp_dir()));
    assert!(!path.starts_with(&repository));
    std::fs::write(&path, "temporary").unwrap();
    assert!(path.exists());

    drop(fixture);
    assert!(!path.exists(), "temporary test fixture was not removed");
    assert!(
        !repository.join("hi-test-scratch").exists(),
        "tests must not recreate repository-relative scratch state"
    );
}

/// Disposable, per-test workspace for tests that exercise workspace change
/// detection. Keeping these roots outside the package checkout lets such tests
/// run in parallel without one agent observing another test's mutations.
pub(crate) struct IsolatedWorkspace {
    root: std::path::PathBuf,
}

impl IsolatedWorkspace {
    pub(crate) fn new(tag: &str) -> Self {
        loop {
            let root = std::env::temp_dir().join(unique_name(tag));
            match std::fs::create_dir(&root) {
                Ok(()) => return Self { root },
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("create isolated test workspace {}: {error}", root.display()),
            }
        }
    }

    pub(crate) fn config(&self) -> AgentConfig {
        let mut cfg = config();
        cfg.paths.workspace_root = self.root.clone();
        cfg.paths.state_root = self.root.join(".hi/state");
        cfg
    }

    pub(crate) fn path(&self, relative: impl AsRef<std::path::Path>) -> std::path::PathBuf {
        self.root.join(relative)
    }
}

impl Drop for IsolatedWorkspace {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn unique_name(tag: &str) -> String {
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    format!("hi-test-{tag}-{}-{n}", std::process::id())
}

#[derive(Default)]
pub(crate) struct RecUi {
    pub(crate) statuses: Vec<String>,
    pub(crate) usages: Vec<(u64, u64)>,
    pub(crate) rate_limits: Vec<Option<hi_ai::RateLimitState>>,
    pub(crate) turn_end: Option<String>,
    pub(crate) assistant: String,
    pub(crate) tool_results: Vec<(String, String)>,
    pub(crate) plans: Vec<Vec<PlanStep>>,
    pub(crate) ask_user_questions: Vec<String>,
    pub(crate) pending_ask_user: bool,
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
    fn plan(&mut self, steps: &[PlanStep]) {
        self.plans.push(steps.to_vec());
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
        _usage_estimated: bool,
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
    fn ask_user(&mut self, question: &str, _options: &[String]) -> crate::AskUserFuture<'_> {
        self.ask_user_questions.push(question.to_string());
        if self.pending_ask_user {
            Box::pin(std::future::pending())
        } else {
            Box::pin(async { crate::AskUserResult::Unavailable })
        }
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
