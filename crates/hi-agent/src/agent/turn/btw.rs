//! `/btw` side-channel answers.
//!
//! A mid-turn `/btw` question is answered **immediately** via
//! [`BtwDispatcher::ask`] — its own model call(s), not waiting for the main
//! turn's next model-round boundary. Fast questions that the session snapshot
//! already answers (branch, first commit, plan, jobs) skip the model entirely.
//! Everything else may inspect the workspace for a few read-only rounds.
//!
//! The inbox-drain path remains as a fallback for tests / headless frontends
//! that only have [`crate::InterjectionInbox`].

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use hi_ai::{
    ChatRequest, CompatMode, Content, Message, Provider, RequestProfile, Role, StreamEvent,
    ToolMode, ToolSpec, Usage,
};
use hi_tools::execute_in_runtime_shared;
use tokio::sync::mpsc;

use crate::Ui;
use crate::heuristics::mode_blocks_tool;

/// Soft cap on recent transcript characters attached to a side question.
const BTW_CONTEXT_CHARS: usize = 2_400;
/// Keep side answers short — this is an aside, not a task deliverable.
const BTW_MAX_TOKENS: u32 = 768;
/// Hard cap on model↔tool rounds so a side question cannot starve the main task.
const BTW_MAX_ROUNDS: u32 = 4;
/// Cap concurrent inspection tools inside one side round.
const BTW_MAX_PARALLEL_TOOLS: usize = 4;
/// Wall-clock budget for one side question (including tool rounds).
const BTW_DEADLINE_SECS: u64 = 45;
/// Capacity for the bounded report channel. `Usage` (telemetry) is dropped if
/// full; `Done` falls back to disconnection signaling when the task exits.
const BTW_REPORT_CHANNEL_CAPACITY: usize = 64;

const BTW_SYSTEM: &str = "\
You are answering a brief side question the user asked while a coding task runs.
You may use the advertised read-only inspection tools when the session snapshot \
is not enough (e.g. project age, file contents, symbol lookup). Do not continue \
the main task, do not propose a plan, do not mutate files, and do not run shell. \
When the snapshot already has the fact (branch, HEAD, first/latest commit, jobs, \
plan), use it directly. Reply in one short paragraph once you know the answer.";

/// Tools a `/btw` side loop may advertise. Excludes session-mutating
/// "read-only" tools (`update_plan`, `record_decision`) and nested agents.
const BTW_TOOL_ALLOWLIST: &[&str] = &[
    "read",
    "list",
    "grep",
    "glob",
    "diff",
    "repo_map",
    "find_symbol",
    "bash_output",
    "diagnostics",
    "definition",
    "references",
    "hover",
    "web_search",
    "web_fetch",
    "memory_search",
    "memory_get",
    "search_tool",
];

/// Detached `/btw` work item — everything needed to answer without `&mut Agent`.
struct BtwJob {
    provider: Arc<dyn Provider>,
    model: String,
    temperature: Option<f32>,
    compat: CompatMode,
    deepseek_compat: hi_ai::DeepSeekCompat,
    system: Message,
    snapshot: String,
    recent: String,
    tools: Arc<[ToolSpec]>,
    questions: Vec<String>,
    root: PathBuf,
    state_root: PathBuf,
    lsp: Arc<hi_lsp::LspManager>,
    background: Arc<hi_tools::BackgroundRegistry>,
    read_cache: Arc<Mutex<hi_tools::ReadCache>>,
    repo_map: Arc<Mutex<hi_tools::RepoMapCache>>,
}

/// Usage / error bookkeeping folded back onto the agent after a side job.
pub(crate) enum BtwJobReport {
    Usage(Usage),
    Done,
}

/// Public side-channel events for frontends that fire `/btw` immediately
/// (outside the main turn's `Ui` trait). Mirrors the internal channel.
#[derive(Clone, Debug)]
pub enum BtwSideEvent {
    Question(String),
    Answer(String),
    ToolStarted { name: String, arguments: String },
    ToolResult { name: String, result: String },
    Status(String),
    End,
}

/// Cloneable sink that forwards BTW UI events onto a shared `Ui` via a channel.
/// Used so a spawned side loop can emit while the main turn also holds `&mut dyn Ui`.
struct BtwEventUi {
    tx: mpsc::UnboundedSender<BtwUiEvent>,
}

enum BtwUiEvent {
    Question(String),
    Answer(String),
    ToolStarted { name: String, arguments: String },
    ToolResult { name: String, result: String },
    Status(String),
    End,
}

impl From<BtwUiEvent> for BtwSideEvent {
    fn from(ev: BtwUiEvent) -> Self {
        match ev {
            BtwUiEvent::Question(q) => Self::Question(q),
            BtwUiEvent::Answer(t) => Self::Answer(t),
            BtwUiEvent::ToolStarted { name, arguments } => Self::ToolStarted { name, arguments },
            BtwUiEvent::ToolResult { name, result } => Self::ToolResult { name, result },
            BtwUiEvent::Status(s) => Self::Status(s),
            BtwUiEvent::End => Self::End,
        }
    }
}

impl Ui for BtwEventUi {
    fn assistant_text(&mut self, text: &str) {
        let _ = self.tx.send(BtwUiEvent::Answer(text.to_string()));
    }
    fn btw_answer(&mut self, text: &str) {
        let _ = self.tx.send(BtwUiEvent::Answer(text.to_string()));
    }
    fn btw_question(&mut self, question: &str) {
        let _ = self.tx.send(BtwUiEvent::Question(question.to_string()));
    }
    fn btw_tool_started(&mut self, name: &str, arguments: &str) {
        let _ = self.tx.send(BtwUiEvent::ToolStarted {
            name: name.to_string(),
            arguments: arguments.to_string(),
        });
    }
    fn btw_tool_result(&mut self, name: &str, result: &str) {
        let _ = self.tx.send(BtwUiEvent::ToolResult {
            name: name.to_string(),
            result: result.to_string(),
        });
    }
    fn btw_end(&mut self) {
        let _ = self.tx.send(BtwUiEvent::End);
    }
    fn assistant_reasoning(&mut self, _: &str) {}
    fn assistant_end(&mut self) {
        let _ = self.tx.send(BtwUiEvent::End);
    }
    fn tool_call(&mut self, _: &str, _: &str) {}
    fn tool_result(&mut self, _: &str, _: &str) {}
    fn status(&mut self, text: &str) {
        let _ = self.tx.send(BtwUiEvent::Status(text.to_string()));
    }
    fn turn_end(&mut self, _: &str) {}
}

/// Live context refreshed as the main turn progresses (plan/jobs/transcript).
#[derive(Clone)]
pub(crate) struct BtwLiveContext {
    snapshot: String,
    recent: String,
    system: Message,
    tools: Arc<[ToolSpec]>,
    model: String,
    temperature: Option<f32>,
    compat: CompatMode,
    deepseek_compat: hi_ai::DeepSeekCompat,
}

/// Cloneable handle the TUI uses to fire `/btw` **immediately** — own model
/// call(s), no wait for the main turn's next round.
///
/// Always holds a shared [`Arc`] created with the agent so clones taken before
/// turn start still see arming (live context + enabled flag).
#[derive(Clone)]
pub struct BtwDispatcher {
    inner: Arc<BtwDispatcherInner>,
}

struct BtwDispatcherInner {
    /// When false, `ask` refuses (idle / turn ended).
    armed: std::sync::atomic::AtomicBool,
    provider: Mutex<Option<Arc<dyn Provider>>>,
    root: Mutex<PathBuf>,
    state_root: Mutex<PathBuf>,
    lsp: Mutex<Option<Arc<hi_lsp::LspManager>>>,
    background: Mutex<Option<Arc<hi_tools::BackgroundRegistry>>>,
    read_cache: Mutex<Option<Arc<Mutex<hi_tools::ReadCache>>>>,
    repo_map: Mutex<Option<Arc<Mutex<hi_tools::RepoMapCache>>>>,
    live: Mutex<Option<BtwLiveContext>>,
    jobs: Arc<Mutex<Vec<BtwJobHandle>>>,
}

impl BtwDispatcher {
    /// Empty shell shared for the agent lifetime; `arm` fills it each turn.
    pub(crate) fn new(jobs: Arc<Mutex<Vec<BtwJobHandle>>>) -> Self {
        Self {
            inner: Arc::new(BtwDispatcherInner {
                armed: std::sync::atomic::AtomicBool::new(false),
                provider: Mutex::new(None),
                root: Mutex::new(PathBuf::new()),
                state_root: Mutex::new(PathBuf::new()),
                lsp: Mutex::new(None),
                background: Mutex::new(None),
                read_cache: Mutex::new(None),
                repo_map: Mutex::new(None),
                live: Mutex::new(None),
                jobs,
            }),
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.inner.armed.load(std::sync::atomic::Ordering::Acquire)
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "the dispatcher is armed once with the complete runtime boundary"
    )]
    pub(crate) fn arm(
        &self,
        provider: Arc<dyn Provider>,
        root: PathBuf,
        state_root: PathBuf,
        lsp: Arc<hi_lsp::LspManager>,
        background: Arc<hi_tools::BackgroundRegistry>,
        read_cache: Arc<Mutex<hi_tools::ReadCache>>,
        repo_map: Arc<Mutex<hi_tools::RepoMapCache>>,
        live: BtwLiveContext,
    ) {
        *self
            .inner
            .provider
            .lock()
            .unwrap_or_else(|p| p.into_inner()) = Some(provider);
        *self.inner.root.lock().unwrap_or_else(|p| p.into_inner()) = root;
        *self
            .inner
            .state_root
            .lock()
            .unwrap_or_else(|p| p.into_inner()) = state_root;
        *self.inner.lsp.lock().unwrap_or_else(|p| p.into_inner()) = Some(lsp);
        *self
            .inner
            .background
            .lock()
            .unwrap_or_else(|p| p.into_inner()) = Some(background);
        *self
            .inner
            .read_cache
            .lock()
            .unwrap_or_else(|p| p.into_inner()) = Some(read_cache);
        *self
            .inner
            .repo_map
            .lock()
            .unwrap_or_else(|p| p.into_inner()) = Some(repo_map);
        *self.inner.live.lock().unwrap_or_else(|p| p.into_inner()) = Some(live);
        self.inner
            .armed
            .store(true, std::sync::atomic::Ordering::Release);
    }

    pub(crate) fn disarm(&self) {
        self.inner
            .armed
            .store(false, std::sync::atomic::Ordering::Release);
    }

    /// Fire a side question **now** (own model call(s)). Events stream on `events`.
    /// Returns `false` if the dispatcher isn't armed (no active turn).
    pub fn ask(&self, question: &str, events: mpsc::UnboundedSender<BtwSideEvent>) -> bool {
        if !self.is_enabled() {
            return false;
        }
        let question = question.trim();
        if question.is_empty() {
            return false;
        }
        let question = question.to_string();
        let inner = self.inner.clone();

        let (ui_tx, mut ui_rx) = mpsc::unbounded_channel::<BtwUiEvent>();
        let (report_tx, report_rx) = mpsc::channel::<BtwJobReport>(BTW_REPORT_CHANNEL_CAPACITY);

        let bridge = events;
        tokio::spawn(async move {
            while let Some(ev) = ui_rx.recv().await {
                if bridge.send(BtwSideEvent::from(ev)).is_err() {
                    break;
                }
            }
        });

        let job = {
            let provider = inner
                .provider
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .clone();
            let live = inner.live.lock().unwrap_or_else(|p| p.into_inner()).clone();
            let root = inner.root.lock().unwrap_or_else(|p| p.into_inner()).clone();
            let state_root = inner
                .state_root
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .clone();
            let lsp = inner.lsp.lock().unwrap_or_else(|p| p.into_inner()).clone();
            let background = inner
                .background
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .clone();
            let read_cache = inner
                .read_cache
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .clone();
            let repo_map = inner
                .repo_map
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .clone();
            let (
                Some(provider),
                Some(live),
                Some(lsp),
                Some(background),
                Some(read_cache),
                Some(repo_map),
            ) = (provider, live, lsp, background, read_cache, repo_map)
            else {
                return false;
            };
            BtwJob {
                provider,
                model: live.model.clone(),
                temperature: live.temperature,
                compat: live.compat,
                deepseek_compat: live.deepseek_compat,
                system: live.system.clone(),
                snapshot: live.snapshot.clone(),
                recent: live.recent.clone(),
                tools: live.tools.clone(),
                questions: vec![question],
                root,
                state_root,
                lsp,
                background,
                read_cache,
                repo_map,
            }
        };

        // UI is already bridged to the TUI; agent poll/join only needs usage reports.
        // Recover from poison so a panicking peer doesn't orphan this job's handle.
        let join = tokio::spawn(async move {
            let mut side_ui = BtwEventUi { tx: ui_tx };
            run_btw_job(job, &mut side_ui, report_tx).await;
        });
        let mut jobs = inner.jobs.lock().unwrap_or_else(|p| p.into_inner());
        jobs.push(BtwJobHandle {
            ui_rx: None,
            report_rx,
            join: Some(join),
        });
        true
    }
}

impl crate::Agent {
    /// Cloneable immediate `/btw` launcher for the frontend. Armed at turn start.
    pub fn btw_dispatcher(&self) -> BtwDispatcher {
        self.btw_dispatch.clone()
    }

    /// Arm (or refresh) the immediate `/btw` dispatcher with live session context.
    /// Call at turn start and each model-round boundary so snapshot/transcript stay fresh.
    /// Updates the shared Arc in place so TUI clones taken earlier still work.
    pub(crate) fn arm_btw_dispatcher(&mut self) {
        let live = BtwLiveContext {
            snapshot: self.btw_session_snapshot(),
            recent: recent_transcript_excerpt(self.messages.as_slice(), BTW_CONTEXT_CHARS),
            system: {
                let base = self.minimal_system_message().text();
                Message::system(format!("{base}\n\n{BTW_SYSTEM}"))
            },
            tools: btw_tool_specs(self.request_tools_for(ToolMode::ReadOnly).as_ref()),
            model: self.config.routing.model.clone(),
            temperature: self.config.routing.temperature,
            compat: self.config.routing.compat,
            deepseek_compat: self.config.routing.deepseek_compat,
        };
        self.btw_dispatch.arm(
            self.provider.clone(),
            self.runtime.root().to_path_buf(),
            self.runtime.state_root().to_path_buf(),
            self.runtime.lsp(),
            self.runtime.background_arc(),
            self.runtime.read_cache_arc(),
            self.runtime.repo_map_arc(),
            live,
        );
    }

    /// Disarm so idle `/btw` can't fire against a finished turn.
    pub(crate) fn disarm_btw_dispatcher(&mut self) {
        self.btw_dispatch.disarm();
    }

    fn push_btw_job(&self, handle: BtwJobHandle) {
        // Recover from poison so a panicking peer doesn't drop this job's handle.
        self.btw_jobs
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push(handle);
    }

    fn take_btw_jobs(&self) -> Vec<BtwJobHandle> {
        // Recover from poison so a panicking peer doesn't lose in-flight job
        // state (which would orphan handles and leave reports undrained).
        let mut guard = self.btw_jobs.lock().unwrap_or_else(|p| p.into_inner());
        std::mem::take(&mut *guard)
    }

    fn store_btw_jobs(&self, jobs: Vec<BtwJobHandle>) {
        // Recover from poison so a panicking peer doesn't silently drop the
        // still-pending job list.
        *self.btw_jobs.lock().unwrap_or_else(|p| p.into_inner()) = jobs;
    }

    fn btw_jobs_pending(&self) -> bool {
        // Recover from poison: treat the recovered (possibly stale) state as
        // authoritative rather than reporting an empty queue that hides jobs.
        self.btw_jobs
            .lock()
            .map(|j| !j.is_empty())
            .unwrap_or_else(|p| !p.into_inner().is_empty())
    }

    /// Fallback: answer `/btw` tags drained from the interjection inbox (tests /
    /// frontends without [`BtwDispatcher`]). Preferred path is immediate `ask`.
    pub(super) async fn answer_btw_side_questions(
        &mut self,
        interjected: Vec<String>,
        ui: &mut dyn Ui,
    ) -> Vec<String> {
        self.poll_btw_jobs(ui).await;
        self.arm_btw_dispatcher();

        if interjected.is_empty() {
            return interjected;
        }

        let mut steering = Vec::new();
        let mut questions = Vec::new();
        for message in interjected {
            if let Some(question) = message.strip_prefix(crate::BTW_INTERJECTION_PREFIX) {
                let question = question.trim();
                if !question.is_empty() {
                    questions.push(question.to_string());
                }
            } else {
                steering.push(message);
            }
        }

        if questions.is_empty() {
            return steering;
        }

        // No main-transcript status spam — the BTW pane owns side-channel chrome.

        let job = BtwJob {
            provider: self.provider.clone(),
            model: self.config.routing.model.clone(),
            temperature: self.config.routing.temperature,
            compat: self.config.routing.compat,
            deepseek_compat: self.config.routing.deepseek_compat,
            system: {
                let base = self.minimal_system_message().text();
                Message::system(format!("{base}\n\n{BTW_SYSTEM}"))
            },
            snapshot: self.btw_session_snapshot(),
            recent: recent_transcript_excerpt(self.messages.as_slice(), BTW_CONTEXT_CHARS),
            tools: btw_tool_specs(self.request_tools_for(ToolMode::ReadOnly).as_ref()),
            questions,
            root: self.runtime.root().to_path_buf(),
            state_root: self.runtime.state_root().to_path_buf(),
            lsp: self.runtime.lsp(),
            background: self.runtime.background_arc(),
            read_cache: self.runtime.read_cache_arc(),
            repo_map: self.runtime.repo_map_arc(),
        };

        let (ui_tx, ui_rx) = mpsc::unbounded_channel::<BtwUiEvent>();
        let (report_tx, report_rx) = mpsc::channel::<BtwJobReport>(BTW_REPORT_CHANNEL_CAPACITY);

        // Fallback path joins so scripted tests see the answer before the next
        // main-model step (shared canned provider). Live TUI uses BtwDispatcher::ask.
        // Do NOT disarm here — mid-turn join must leave the dispatcher armed.
        let join = tokio::spawn(async move {
            let mut side_ui = BtwEventUi { tx: ui_tx };
            run_btw_job(job, &mut side_ui, report_tx).await;
        });
        self.push_btw_job(BtwJobHandle {
            ui_rx: Some(ui_rx),
            report_rx,
            // The fallback path joins inline below; no detached handle to abort.
            join: None,
        });
        self.join_btw_jobs(ui).await;
        let _ = join.await;

        steering
    }

    /// Drain UI + usage from any finished or in-flight `/btw` side jobs.
    pub(crate) async fn poll_btw_jobs(&mut self, ui: &mut dyn Ui) {
        if !self.btw_jobs_pending() {
            return;
        }
        let mut still = Vec::new();
        for mut handle in self.take_btw_jobs() {
            if let Some(rx) = handle.ui_rx.as_mut() {
                drain_btw_ui(rx, ui);
            }
            let mut done = false;
            loop {
                match handle.report_rx.try_recv() {
                    Ok(BtwJobReport::Usage(u)) => self.add_side_usage(u),
                    Ok(BtwJobReport::Done) => done = true,
                    Err(mpsc::error::TryRecvError::Empty) => break,
                    Err(mpsc::error::TryRecvError::Disconnected) => {
                        done = true;
                        break;
                    }
                }
            }
            if let Some(rx) = handle.ui_rx.as_mut() {
                drain_btw_ui(rx, ui);
            }
            // Completion is report-driven. UI may be absent (TUI-bridged jobs) or
            // still draining; once Done/disconnected and UI empty/absent, drop.
            let ui_drained = handle
                .ui_rx
                .as_ref()
                .map(|rx| rx.is_empty())
                .unwrap_or(true);
            if done && ui_drained {
                // Abort the spawned task if it's still running (defensive: the
                // Done/disconnected signal usually means it already exited, but
                // a slow stream teardown could linger). Dropping the handle
                // without abort would let it run until the runtime shuts down.
                if let Some(join) = handle.join.as_ref() {
                    join.abort();
                }
                continue;
            }
            still.push(handle);
        }
        self.store_btw_jobs(still);
    }

    /// Block until all `/btw` side jobs finish. Streams late UI events when present.
    /// Does **not** disarm the dispatcher — callers that end the turn must
    /// [`Self::disarm_btw_dispatcher`] separately so mid-turn inbox fallbacks
    /// don't kill immediate TUI `ask` for the rest of the round.
    pub(crate) async fn join_btw_jobs(&mut self, ui: &mut dyn Ui) {
        while self.btw_jobs_pending() {
            let mut still = Vec::new();
            for mut handle in self.take_btw_jobs() {
                if let Some(rx) = handle.ui_rx.as_mut() {
                    drain_btw_ui(rx, ui);
                }
                // Report channel is the source of truth for job lifetime.
                // UI is optional (dispatcher path bridges elsewhere).
                match handle.report_rx.recv().await {
                    Some(BtwJobReport::Usage(u)) => {
                        self.add_side_usage(u);
                        // Keep waiting for Done — more reports may follow.
                        still.push(handle);
                    }
                    Some(BtwJobReport::Done) => {
                        // Flush any trailing usage already buffered, then UI.
                        while let Ok(r) = handle.report_rx.try_recv() {
                            if let BtwJobReport::Usage(u) = r {
                                self.add_side_usage(u);
                            }
                        }
                        if let Some(rx) = handle.ui_rx.as_mut() {
                            // Drain whatever UI frames already arrived. The
                            // channel is unbounded, so late frames are buffered
                            // and picked up on the next poll — no sleep needed.
                            drain_btw_ui(rx, ui);
                        }
                        // Job complete — abort any lingering task and drop.
                        if let Some(join) = handle.join.as_ref() {
                            join.abort();
                        }
                    }
                    None => {
                        // Worker dropped without Done — still fold leftover usage.
                        while let Ok(r) = handle.report_rx.try_recv() {
                            if let BtwJobReport::Usage(u) = r {
                                self.add_side_usage(u);
                            }
                        }
                        if let Some(rx) = handle.ui_rx.as_mut() {
                            drain_btw_ui(rx, ui);
                        }
                        // Abort any lingering task and drop.
                        if let Some(join) = handle.join.as_ref() {
                            join.abort();
                        }
                    }
                }
            }
            self.store_btw_jobs(still);
            if self.btw_jobs_pending() {
                tokio::task::yield_now().await;
            }
        }
    }
}

pub(crate) struct BtwJobHandle {
    /// Present when the agent owns UI delivery (inbox fallback). `None` when
    /// the TUI already bridges side events (dispatcher `ask`).
    ui_rx: Option<mpsc::UnboundedReceiver<BtwUiEvent>>,
    report_rx: mpsc::Receiver<BtwJobReport>,
    /// The spawned job task. `None` for the fallback path where the caller
    /// joins inline. Aborted when the handle is dropped after completion so
    /// a finished-but-not-yet-polled task can't outlive the agent.
    join: Option<tokio::task::JoinHandle<()>>,
}

fn drain_btw_ui(rx: &mut mpsc::UnboundedReceiver<BtwUiEvent>, ui: &mut dyn Ui) {
    while let Ok(ev) = rx.try_recv() {
        apply_btw_ui_event(ev, ui);
    }
}

fn apply_btw_ui_event(ev: BtwUiEvent, ui: &mut dyn Ui) {
    match ev {
        BtwUiEvent::Question(q) => ui.btw_question(&q),
        BtwUiEvent::Answer(t) => ui.btw_answer(&t),
        BtwUiEvent::ToolStarted { name, arguments } => ui.btw_tool_started(&name, &arguments),
        BtwUiEvent::ToolResult { name, result } => ui.btw_tool_result(&name, &result),
        BtwUiEvent::Status(s) => ui.status(&s),
        BtwUiEvent::End => ui.btw_end(),
    }
}

async fn run_btw_job(job: BtwJob, ui: &mut dyn Ui, report_tx: mpsc::Sender<BtwJobReport>) {
    for question in &job.questions {
        ui.btw_question(question);
        // Phase D: snapshot-only fast path — no model, no tools.
        if let Some(fast) = route_snapshot_answer(question, &job.snapshot) {
            ui.btw_answer(&fast);
            ui.btw_end();
            continue;
        }
        answer_one_btw_question(&job, question, ui, &report_tx).await;
        ui.btw_end();
    }
    // `Done` is the lifetime signal. If the channel is full (usage backlog),
    // the receiver will see Disconnected when this task exits — same outcome.
    let _ = report_tx.try_send(BtwJobReport::Done);
}

async fn answer_one_btw_question(
    job: &BtwJob,
    question: &str,
    ui: &mut dyn Ui,
    report_tx: &mpsc::Sender<BtwJobReport>,
) {
    let user = format!(
        "Side question (inspect if needed, then answer briefly and stop):\n{question}\n\n\
         Current session snapshot:\n{}\n\n\
         Recent task transcript (context only — do not continue it):\n{}",
        job.snapshot, job.recent
    );
    let mut messages: Vec<Message> = vec![job.system.clone(), Message::user(user)];
    let tools_empty = job.tools.is_empty();
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(BTW_DEADLINE_SECS);

    for round in 0..BTW_MAX_ROUNDS {
        if tokio::time::Instant::now() >= deadline {
            ui.btw_answer("(side question timed out)");
            return;
        }
        let last_round = round + 1 >= BTW_MAX_ROUNDS || tools_empty;
        let request_tools = if last_round {
            Arc::new([])
        } else {
            job.tools.clone()
        };
        let tool_mode = if last_round {
            ToolMode::ChatOnly
        } else {
            ToolMode::ReadOnly
        };

        let request = ChatRequest {
            model: job.model.clone(),
            request_id: None,
            retry_attempt: 0,
            user_turn: false,
            canonical_objective: None,
            messages: Arc::from(messages.clone()),
            tools: request_tools,
            max_tokens: BTW_MAX_TOKENS,
            temperature: job.temperature,
            top_p: None,
            frequency_penalty: None,
            thinking_budget: None,
            reasoning_effort: None,
            profile: RequestProfile {
                compat: job.compat,
                tool_mode,
                stream_usage: None,
                deepseek_compat: job.deepseek_compat,
                deepseek_strict: None,
                deepseek_thinking: None,
                output_token_parameter: hi_ai::OutputTokenParameter::Auto,
            },
        };

        let mut streamed = String::new();
        let mut sink = |event: StreamEvent| match event {
            StreamEvent::Text(text) => {
                streamed.push_str(&text);
            }
            StreamEvent::Status(text) => ui.status(&text),
            StreamEvent::Reasoning(_) => {}
            StreamEvent::WireAudit(_) => {}
        };

        let completion = match job.provider.stream(request, &mut sink).await {
            Ok(c) => {
                // Usage is telemetry — drop if the bounded channel is full
                // rather than blocking the side job's model stream.
                let _ = report_tx.try_send(BtwJobReport::Usage(c.usage));
                c
            }
            Err(err) => {
                let msg = format!("(could not answer side question: {err:#})");
                ui.btw_answer(&msg);
                return;
            }
        };

        let calls = completion.tool_calls();
        if calls.is_empty() || last_round {
            if !streamed.trim().is_empty() {
                ui.btw_answer(&streamed);
            } else {
                let mut fallback = String::new();
                for c in &completion.content {
                    if let Content::Text(t) = c {
                        fallback.push_str(t);
                    }
                }
                if fallback.trim().is_empty() {
                    ui.btw_answer("(no answer)");
                } else {
                    ui.btw_answer(&fallback);
                }
            }
            return;
        }

        messages.push(Message::assistant(completion.content.clone()));

        let mut batch: Vec<(String, String, String)> = Vec::new();
        for call in &calls {
            if let Some(reason) = mode_blocks_tool(ToolMode::ReadOnly, call.name) {
                messages.push(Message::tool_result(call.id, reason));
                continue;
            }
            if !BTW_TOOL_ALLOWLIST.contains(&call.name) {
                messages.push(Message::tool_result(
                    call.id,
                    format!(
                        "tool `{}` is not available on /btw side questions \
                         (read-only inspection only)",
                        call.name
                    ),
                ));
                continue;
            }
            batch.push((
                call.id.to_string(),
                call.name.to_string(),
                call.arguments.to_string(),
            ));
        }

        if batch.is_empty() {
            continue;
        }

        // Tools go to the BTW pane via btw_tool_* — never main-transcript status.
        for (_, name, args) in &batch {
            ui.btw_tool_started(name, args);
        }

        use futures_util::StreamExt;
        let root = job.root.clone();
        let state_root = job.state_root.clone();
        let lsp = job.lsp.clone();
        let background = job.background.clone();
        let read_cache = job.read_cache.clone();
        let repo_map = job.repo_map.clone();
        // Enforce the wall-clock deadline inside the tool batch too: a slow
        // tool call (long bash/inspection) would otherwise block past the
        // deadline with no cancellation. On timeout, abandon the side
        // question — the partial transcript is discarded (BTW is a
        // side-channel, not the main transcript) and `run_btw_job` still
        // sends `Done` after this returns.
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        let tool_stream = futures_util::stream::iter(batch.into_iter().map(|(id, name, args)| {
            let root = root.clone();
            let state_root = state_root.clone();
            let lsp = lsp.clone();
            let background = background.clone();
            let read_cache = read_cache.clone();
            let repo_map = repo_map.clone();
            async move {
                let outcome = execute_in_runtime_shared(
                    &root,
                    &state_root,
                    &lsp,
                    background.as_ref(),
                    read_cache.as_ref(),
                    &repo_map,
                    &name,
                    &args,
                )
                .await;
                (id, name, outcome.content)
            }
        }))
        .buffer_unordered(BTW_MAX_PARALLEL_TOOLS.max(1));

        let results = match tokio::time::timeout(remaining, tool_stream.collect::<Vec<_>>()).await {
            Ok(results) => results,
            Err(_) => {
                ui.btw_answer("(side question timed out during tool execution)");
                return;
            }
        };

        for (id, name, content) in results {
            ui.btw_tool_result(&name, &content);
            messages.push(Message::tool_result(id, content));
        }
    }
}

/// Snapshot-only router: answer common asides without a model call when the
/// session snapshot already carries the fact.
pub(crate) fn route_snapshot_answer(question: &str, snapshot: &str) -> Option<String> {
    let q = question.to_ascii_lowercase();
    let q = q.trim();

    // Calendar date (UTC, from the snapshot — not training cutoff).
    if matches_any(
        q,
        &[
            "what day is it",
            "what's the date",
            "whats the date",
            "what is the date",
            "what date is it",
            "today's date",
            "todays date",
            "what day is today",
            "current date",
        ],
    ) && let Some(v) = snapshot_value(snapshot, "- utc_date:")
    {
        return Some(format!("Today (UTC): {v}."));
    }

    // Git / project age.
    if matches_any(
        q,
        &[
            "how old",
            "how long",
            "when was",
            "when did",
            "first commit",
            "project age",
            "repo age",
            "created",
            "started",
        ],
    ) && (q.contains("project")
        || q.contains("repo")
        || q.contains("codebase")
        || q.contains("this")
        || q.contains("commit")
        || q.contains("old")
        || q.contains("age"))
    {
        if let Some(line) = snapshot_line(snapshot, "- git first commit:") {
            let detail = line.trim();
            return Some(format!(
                "First commit on record: {detail}. (From the live git log — no tools needed.)"
            ));
        }
        if snapshot.contains("- git: not a repository") {
            return Some(
                "This workspace is not a git repository, so I can't date it from history.".into(),
            );
        }
    }

    // Branch.
    if (q.contains("branch") || q.contains("what branch"))
        && (q.contains("what") || q.contains("which") || q.contains("current") || q.ends_with('?'))
        && let Some(b) = snapshot_value(snapshot, "- git branch:")
    {
        return Some(format!("Current branch: `{b}`."));
    }

    // HEAD.
    if q.contains("head")
        && (q.contains("commit") || q.contains("sha") || q.contains("revision"))
        && let Some(h) = snapshot_value(snapshot, "- git HEAD:")
    {
        return Some(format!("HEAD is `{h}`."));
    }

    // Dirty / uncommitted.
    if (q.contains("dirty") || q.contains("uncommitted") || q.contains("working tree"))
        && (q.contains("is") || q.contains("any") || q.contains("clean"))
        && let Some(line) = snapshot_line(snapshot, "- git dirty:")
    {
        return Some(format!("Working tree: {}.", line.trim()));
    }

    // Background jobs.
    if (q.contains("job") || q.contains("background") || q.contains("still running"))
        && (q.contains("running")
            || q.contains("status")
            || q.contains("job")
            || q.contains("background"))
    {
        if let Some(block) = snapshot_section(snapshot, "- background jobs:") {
            return Some(format!("Background jobs:\n{block}"));
        }
        return Some("No background jobs are registered right now.".into());
    }

    // Plan / step.
    if (q.contains("plan") || q.contains("step") || q.contains("working on"))
        && (q.contains("what")
            || q.contains("which")
            || q.contains("status")
            || q.contains("progress")
            || q.contains("doing"))
    {
        if let Some(block) = snapshot_section(snapshot, "- plan:") {
            return Some(format!("Current plan:\n{block}"));
        }
        if let Some(goal) = snapshot_value(snapshot, "- goal:") {
            return Some(format!("No structured plan steps; goal is: {goal}"));
        }
        return Some("No plan is set for this turn.".into());
    }

    // Model / route.
    if q.contains("model")
        && (q.contains("what") || q.contains("which") || q.contains("using"))
        && let Some(m) = snapshot_value(snapshot, "- model:")
    {
        let route = snapshot_value(snapshot, "- provider route:")
            .map(|r| format!(" (route: {r})"))
            .unwrap_or_default();
        return Some(format!("This session is on `{m}`{route}."));
    }

    None
}

fn matches_any(q: &str, needles: &[&str]) -> bool {
    needles.iter().any(|n| q.contains(n))
}

fn snapshot_line<'a>(snapshot: &'a str, prefix: &str) -> Option<&'a str> {
    snapshot
        .lines()
        .find_map(|l| l.trim_start().strip_prefix(prefix))
}

fn snapshot_value(snapshot: &str, prefix: &str) -> Option<String> {
    snapshot_line(snapshot, prefix).map(|s| s.trim().to_string())
}

/// Header line plus following indented detail lines.
fn snapshot_section(snapshot: &str, prefix: &str) -> Option<String> {
    let mut lines = snapshot.lines().peekable();
    while let Some(line) = lines.next() {
        if line.trim_start().strip_prefix(prefix).is_some() {
            let mut block = vec![line.to_string()];
            while let Some(next) = lines.peek() {
                if next.starts_with("    ") || next.starts_with('\t') {
                    block.push(lines.next().unwrap().to_string());
                } else {
                    break;
                }
            }
            return Some(block.join("\n"));
        }
    }
    None
}

/// Read-only inspection specs allowed on the `/btw` side channel.
fn btw_tool_specs(available: &[ToolSpec]) -> Arc<[ToolSpec]> {
    available
        .iter()
        .filter(|t| BTW_TOOL_ALLOWLIST.contains(&t.name.as_str()))
        .cloned()
        .collect::<Vec<_>>()
        .into()
}

/// Compact, newest-last excerpt of recent non-system messages for side context.
fn recent_transcript_excerpt(messages: &[Message], budget: usize) -> String {
    let mut chunks: Vec<String> = Vec::new();
    let mut used = 0usize;
    for message in messages.iter().rev() {
        if message.role == Role::System {
            continue;
        }
        let role = match message.role {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
            Role::System => continue,
        };
        let mut text = message.text();
        text = text.split_whitespace().collect::<Vec<_>>().join(" ");
        if text.is_empty() {
            continue;
        }
        if text.chars().count() > 400 {
            let truncated: String = text.chars().take(397).collect();
            text = format!("{truncated}…");
        }
        let line = format!("{role}: {text}");
        let len = line.chars().count() + 1;
        if used + len > budget && !chunks.is_empty() {
            break;
        }
        used += len;
        chunks.push(line);
    }
    chunks.reverse();
    if chunks.is_empty() {
        "(no recent task messages)".to_string()
    } else {
        chunks.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hi_ai::Message;

    #[test]
    fn recent_transcript_excerpt_skips_system_and_caps() {
        let messages = vec![
            Message::system("sys"),
            Message::user("hello there"),
            Message::assistant(vec![Content::Text("working on it".into())]),
            Message::user("status?"),
        ];
        let excerpt = recent_transcript_excerpt(&messages, 2_400);
        assert!(excerpt.contains("user: hello there"));
        assert!(excerpt.contains("assistant: working on it"));
        assert!(excerpt.contains("user: status?"));
        assert!(!excerpt.contains("sys"));
    }

    #[test]
    fn btw_tool_specs_keeps_only_allowlisted_inspection() {
        let specs = vec![
            ToolSpec {
                name: "read".into(),
                description: "r".into(),
                parameters: serde_json::json!({}),
            },
            ToolSpec {
                name: "update_plan".into(),
                description: "p".into(),
                parameters: serde_json::json!({}),
            },
            ToolSpec {
                name: "explore".into(),
                description: "e".into(),
                parameters: serde_json::json!({}),
            },
            ToolSpec {
                name: "bash".into(),
                description: "b".into(),
                parameters: serde_json::json!({}),
            },
            ToolSpec {
                name: "repo_map".into(),
                description: "m".into(),
                parameters: serde_json::json!({}),
            },
        ];
        let filtered = btw_tool_specs(&specs);
        let names: Vec<&str> = filtered.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, vec!["read", "repo_map"]);
    }

    #[test]
    fn router_answers_project_age_from_first_commit() {
        let snap = "\
- model: test
- workspace: /tmp/x
- git branch: main
- git HEAD: abc
- git first commit: a1b2c3d 2019-03-04 project born
- git latest commit: deadbeef 2024-01-01 later
- git dirty: clean
";
        let ans = route_snapshot_answer("how old is this project?", snap).unwrap();
        assert!(ans.contains("2019-03-04"), "{ans}");
        assert!(ans.contains("project born"), "{ans}");
    }

    #[test]
    fn router_answers_branch() {
        let snap = "- git branch: feature/btw\n- model: m\n";
        let ans = route_snapshot_answer("what branch are we on?", snap).unwrap();
        assert!(ans.contains("feature/btw"), "{ans}");
    }

    #[test]
    fn router_answers_utc_date() {
        let snap = "- utc_date: 2026-08-20 (Thursday)\n- git branch: main\n";
        let ans = route_snapshot_answer("what day is it?", snap).unwrap();
        assert!(ans.contains("2026-08-20"), "{ans}");
        assert!(ans.contains("Thursday"), "{ans}");
    }

    #[test]
    fn router_skips_file_content_questions() {
        let snap = "- git first commit: a 2019-01-01 x\n";
        assert!(route_snapshot_answer("what does AGE.txt say?", snap).is_none());
        assert!(route_snapshot_answer("read the readme", snap).is_none());
    }

    #[test]
    fn router_answers_no_jobs() {
        let snap = "- model: m\n- workspace: /x\n";
        let ans = route_snapshot_answer("is my background job still running?", snap).unwrap();
        assert!(ans.to_ascii_lowercase().contains("no background"), "{ans}");
    }
}
