use super::common::{
    Canned, IsolatedWorkspace, NullUi, ProviderStep, RecUi, RecordingUi, ScriptedProvider, agent,
    bash_completion, completion, config, scripted_agent, write_completion,
};
use super::*;
use hi_ai::{ChatRequest, ProviderErrorKind, StreamEvent};
use std::sync::Mutex;

struct PendingProvider;

struct CancelThenCompleteProvider {
    cancellation: TurnCancellation,
}

struct CancelDuringSuggestionProvider {
    cancellation: TurnCancellation,
    calls: std::sync::atomic::AtomicUsize,
}

struct HangAfterMainProvider {
    calls: std::sync::atomic::AtomicUsize,
}

struct DecisionReplacementSession {
    decisions: std::sync::Arc<Mutex<DecisionLog>>,
}

struct OutcomeRecordingSession {
    outcomes: std::sync::Arc<Mutex<Vec<TurnOutcome>>>,
}

struct PlanPauseOrderSession {
    records: std::sync::Arc<Mutex<Vec<String>>>,
}

impl SessionSink for DecisionReplacementSession {
    fn record(&mut self, _: &[Message], _: Usage) -> anyhow::Result<()> {
        Ok(())
    }

    fn record_compaction(&mut self, _: &[Message]) -> anyhow::Result<()> {
        Ok(())
    }

    fn record_decisions(&mut self, decisions: &DecisionLog) -> anyhow::Result<()> {
        *self.decisions.lock().unwrap() = decisions.clone();
        Ok(())
    }
}

impl SessionSink for OutcomeRecordingSession {
    fn record(&mut self, _: &[Message], _: Usage) -> anyhow::Result<()> {
        Ok(())
    }

    fn record_compaction(&mut self, _: &[Message]) -> anyhow::Result<()> {
        Ok(())
    }

    fn record_turn_outcome(
        &mut self,
        outcome: &TurnOutcome,
        _: Option<&str>,
    ) -> anyhow::Result<()> {
        self.outcomes.lock().unwrap().push(outcome.clone());
        Ok(())
    }
}

impl SessionSink for PlanPauseOrderSession {
    fn record(&mut self, _: &[Message], _: Usage) -> anyhow::Result<()> {
        Ok(())
    }

    fn record_compaction(&mut self, _: &[Message]) -> anyhow::Result<()> {
        Ok(())
    }

    fn record_state_replacement(
        &mut self,
        _: &[Message],
        _: Option<&Goal>,
        _: &DecisionLog,
        _: &[PlanStep],
    ) -> anyhow::Result<()> {
        self.records.lock().unwrap().push("rewind".into());
        Ok(())
    }

    fn record_plan_drive_state_with_policy(
        &mut self,
        paused: bool,
        _: u32,
        resume_on_user_input: bool,
        _: bool,
        _: &[String],
    ) -> anyhow::Result<()> {
        self.records
            .lock()
            .unwrap()
            .push(format!("pause:{paused}:{resume_on_user_input}"));
        Ok(())
    }
}

#[derive(Default)]
struct TurnLifecycleProbe {
    starts: std::sync::atomic::AtomicUsize,
    done: std::sync::atomic::AtomicUsize,
    aborts: Mutex<Vec<hi_agent_lifecycle::TurnAbortReason>>,
}

#[async_trait::async_trait]
impl hi_agent_lifecycle::TurnLifecycleContributor for TurnLifecycleProbe {
    async fn on_turn_start(&self, _: &hi_agent_lifecycle::TurnStartInput) {
        self.starts
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }

    async fn on_turn_done(&self, _: &hi_agent_lifecycle::TurnDoneInput) {
        self.done.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }

    async fn on_turn_abort(&self, input: &hi_agent_lifecycle::TurnAbortInput) {
        self.aborts.lock().unwrap().push(input.reason);
    }
}

#[derive(Default)]
struct StuckAbortLifecycleProbe {
    aborts: std::sync::atomic::AtomicUsize,
}

#[async_trait::async_trait]
impl hi_agent_lifecycle::TurnLifecycleContributor for StuckAbortLifecycleProbe {
    async fn on_turn_abort(&self, _: &hi_agent_lifecycle::TurnAbortInput) {
        self.aborts
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        std::future::pending().await
    }
}

struct CancelOnRunStartUi {
    cancellation: TurnCancellation,
    fired: bool,
}

impl Ui for CancelOnRunStartUi {
    fn semantic_event(&mut self, _: hi_events::RunEvent) {
        if !self.fired {
            self.fired = true;
            self.cancellation.cancel();
        }
    }

    fn assistant_text(&mut self, _: &str) {}
    fn assistant_reasoning(&mut self, _: &str) {}
    fn assistant_end(&mut self) {}
    fn tool_call(&mut self, _: &str, _: &str) {}
    fn tool_result(&mut self, _: &str, _: &str) {}
    fn status(&mut self, _: &str) {}
    fn turn_end(&mut self, _: &str) {}
}

struct CancelAfterToolUi {
    cancellation: TurnCancellation,
    fired: bool,
}

struct CancelAfterToolCountUi {
    cancellation: TurnCancellation,
    remaining: usize,
}

impl Ui for CancelAfterToolUi {
    fn assistant_text(&mut self, _: &str) {}
    fn assistant_reasoning(&mut self, _: &str) {}
    fn assistant_end(&mut self) {}
    fn tool_call(&mut self, _: &str, _: &str) {}
    fn tool_result(&mut self, _: &str, _: &str) {
        if !self.fired {
            self.fired = true;
            self.cancellation.cancel();
        }
    }
    fn status(&mut self, _: &str) {}
    fn turn_end(&mut self, _: &str) {}
}

impl Ui for CancelAfterToolCountUi {
    fn assistant_text(&mut self, _: &str) {}
    fn assistant_reasoning(&mut self, _: &str) {}
    fn assistant_end(&mut self) {}
    fn tool_call(&mut self, _: &str, _: &str) {}
    fn tool_result(&mut self, _: &str, _: &str) {
        self.remaining = self.remaining.saturating_sub(1);
        if self.remaining == 0 {
            self.cancellation.cancel();
        }
    }
    fn plan_result_id(
        &mut self,
        _: &str,
        _: &str,
        _: &str,
        _: hi_tools::ToolStatus,
        _: &[PlanStep],
    ) {
        self.remaining = self.remaining.saturating_sub(1);
        if self.remaining == 0 {
            self.cancellation.cancel();
        }
    }
    fn status(&mut self, _: &str) {}
    fn turn_end(&mut self, _: &str) {}
}

#[async_trait::async_trait]
impl Provider for PendingProvider {
    async fn stream(
        &self,
        _request: ChatRequest,
        _sink: &mut (dyn FnMut(StreamEvent) + Send),
    ) -> anyhow::Result<Completion> {
        std::future::pending().await
    }
}

#[async_trait::async_trait]
impl Provider for CancelThenCompleteProvider {
    async fn stream(
        &self,
        _request: ChatRequest,
        _sink: &mut (dyn FnMut(StreamEvent) + Send),
    ) -> anyhow::Result<Completion> {
        // Yield after signalling so the outer cancellation selector wins,
        // then settle normally during its cooperative grace window.
        self.cancellation.cancel();
        tokio::task::yield_now().await;
        Ok(completion(vec![Content::Text("too late".into())], 1, 1))
    }
}

#[async_trait::async_trait]
impl Provider for CancelDuringSuggestionProvider {
    async fn stream(
        &self,
        _request: ChatRequest,
        _sink: &mut (dyn FnMut(StreamEvent) + Send),
    ) -> anyhow::Result<Completion> {
        match self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) {
            0 => Ok(bash_completion("echo working")),
            1 => Ok(completion(
                vec![Content::Text(
                    "The configured step cap left follow-up work.".into(),
                )],
                1,
                1,
            )),
            _ => {
                self.cancellation.cancel();
                std::future::pending().await
            }
        }
    }
}

#[async_trait::async_trait]
impl Provider for HangAfterMainProvider {
    async fn stream(
        &self,
        _request: ChatRequest,
        _sink: &mut (dyn FnMut(StreamEvent) + Send),
    ) -> anyhow::Result<Completion> {
        if self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
            Ok(completion(vec![Content::Text("done".into())], 1, 1))
        } else {
            // Simulate a provider whose primary response completed but whose
            // optional next-prompt request never produces a response.
            std::future::pending().await
        }
    }
}

struct ReviewMutationProvider {
    responses: Mutex<Vec<Completion>>,
    calls: std::sync::atomic::AtomicUsize,
    root: std::path::PathBuf,
}

struct TurnEndMutationUi {
    root: std::path::PathBuf,
}

#[derive(Default)]
struct RejectAllConfirmUi {
    confirm_calls: usize,
    checkpoint_warnings: Vec<String>,
}

impl Ui for RejectAllConfirmUi {
    fn assistant_text(&mut self, _: &str) {}
    fn assistant_reasoning(&mut self, _: &str) {}
    fn assistant_end(&mut self) {}
    fn confirm(&mut self, _: ConfirmationRequest) -> ConfirmationFuture<'_> {
        self.confirm_calls += 1;
        Box::pin(async { ConfirmationResult::Rejected })
    }
    fn tool_call(&mut self, _: &str, _: &str) {}
    fn tool_result(&mut self, _: &str, _: &str) {}
    fn status(&mut self, _: &str) {}
    fn checkpoint_warning(&mut self, text: &str) {
        self.checkpoint_warnings.push(text.to_string());
    }
    fn turn_end(&mut self, _: &str) {}
}

struct FailingRecordSession;

impl SessionSink for FailingRecordSession {
    fn record(&mut self, _: &[Message], _: Usage) -> anyhow::Result<()> {
        anyhow::bail!("session persistence failed")
    }

    fn record_compaction(&mut self, _: &[Message]) -> anyhow::Result<()> {
        anyhow::bail!("session persistence failed")
    }
}

impl Ui for TurnEndMutationUi {
    fn assistant_text(&mut self, _: &str) {}
    fn assistant_reasoning(&mut self, _: &str) {}
    fn assistant_end(&mut self) {}
    fn tool_call(&mut self, _: &str, _: &str) {}
    fn tool_result(&mut self, _: &str, _: &str) {}
    fn status(&mut self, _: &str) {}
    fn turn_end(&mut self, _: &str) {
        std::fs::write(self.root.join("late.rs"), "late mutation\n").unwrap();
    }
}

#[async_trait::async_trait]
impl Provider for ReviewMutationProvider {
    async fn stream(
        &self,
        _request: ChatRequest,
        _sink: &mut (dyn FnMut(StreamEvent) + Send),
    ) -> anyhow::Result<Completion> {
        let call = self
            .calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if call == 2 {
            std::fs::write(self.root.join("late.rs"), "late mutation\n")?;
        }
        Ok(self.responses.lock().unwrap().remove(0))
    }
}

#[test]
fn agent_construction_reports_runtime_and_verification_configuration_errors() {
    let provider = || std::sync::Arc::new(Canned(Mutex::new(Vec::new())));

    let mut invalid_verify = config();
    invalid_verify.gates.verification =
        VerificationMode::Explicit(vec![VerifyStage::new("verify", "   ")]);
    let error = match Agent::new(provider(), invalid_verify) {
        Ok(_) => panic!("blank verification command was accepted"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("non-empty command"));

    let root = std::env::temp_dir().join(format!("hi-agent-runtime-error-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let state_file = root.join("state-is-a-file");
    std::fs::write(&state_file, "not a directory").unwrap();
    let mut invalid_runtime = config();
    invalid_runtime.paths.workspace_root = root.clone();
    invalid_runtime.paths.state_root = state_file;
    let error = match Agent::new(provider(), invalid_runtime) {
        Ok(_) => panic!("invalid state root was accepted"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("workspace state root"));
    let _ = std::fs::remove_dir_all(root);
}

#[tokio::test]
async fn cancelled_turn_reconciles_surviving_workspace_changes() {
    let root = std::env::temp_dir().join(format!("hi-agent-cancel-outcome-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let mut cfg = config();
    cfg.paths.workspace_root = root.clone();
    cfg.paths.state_root = root.join(".hi/state");
    let mut agent = agent(Vec::new(), cfg);
    agent.workspace.active_turn_ledger_revision = Some(agent.runtime.ledger().revision());
    agent.workspace.active_turn_message_start = Some(agent.messages().len());
    std::fs::write(root.join("survived.txt"), "kept\n").unwrap();

    let outcome = agent
        .cleanup_turn(crate::TurnCleanupKind::Cancel {
            session: crate::SessionRollback::AlreadyApplied,
        })
        .await
        .unwrap()
        .outcome;

    assert_eq!(outcome.status, TurnStatus::Cancelled);
    assert_eq!(outcome.verification, VerificationStatus::Unverified);
    assert_eq!(outcome.changed_files, vec!["survived.txt"]);
    let change = agent.last_file_changes().first().unwrap();
    assert_eq!(change.kind, hi_tools::FileChangeKind::Create);
    assert!(change.before_digest.is_none());
    assert!(change.after_digest.is_some());
    let _ = std::fs::remove_dir_all(root);
}

#[test]
#[allow(deprecated)]
fn legacy_sync_finalizers_keep_their_full_reconcile_contract() {
    let cancelled_workspace = IsolatedWorkspace::new("legacy-cancelled-finalizer-reconcile");
    let mut cancelled = agent(Vec::new(), cancelled_workspace.config());
    cancelled.workspace.active_turn_ledger_revision = Some(cancelled.runtime.ledger().revision());
    std::fs::write(
        cancelled.workspace_root().join("cancelled-survivor.txt"),
        "kept\n",
    )
    .unwrap();

    let cancelled_outcome = cancelled.finalize_cancelled_turn().unwrap();
    assert_eq!(cancelled_outcome.status, TurnStatus::Cancelled);
    assert_eq!(
        cancelled_outcome.changed_files,
        vec!["cancelled-survivor.txt"]
    );

    let failed_workspace = IsolatedWorkspace::new("legacy-failed-finalizer-reconcile");
    let mut failed = agent(Vec::new(), failed_workspace.config());
    failed.workspace.active_turn_ledger_revision = Some(failed.runtime.ledger().revision());
    std::fs::write(
        failed.workspace_root().join("failed-survivor.txt"),
        "kept\n",
    )
    .unwrap();

    let failed_outcome = failed.finalize_failed_turn();
    assert_eq!(failed_outcome.status, TurnStatus::Failed);
    assert_eq!(failed_outcome.changed_files, vec!["failed-survivor.txt"]);

    let implicit_workspace = IsolatedWorkspace::new("legacy-failed-implicit-baseline");
    let mut implicit = agent(Vec::new(), implicit_workspace.config());
    assert!(implicit.workspace.active_turn_ledger_revision.is_none());
    std::fs::write(
        implicit.workspace_root().join("implicit-survivor.txt"),
        "kept\n",
    )
    .unwrap();

    let implicit_outcome = implicit.finalize_failed_turn();
    assert_eq!(implicit_outcome.status, TurnStatus::Failed);
    assert_eq!(
        implicit_outcome.changed_files,
        vec!["implicit-survivor.txt"]
    );
}

#[tokio::test]
async fn cancelled_turn_bounds_its_final_ledger_reconcile() {
    let workspace = IsolatedWorkspace::new("cancel-bounded-final-reconcile");
    let mut agent = agent(Vec::new(), workspace.config());
    agent.workspace.active_turn_ledger_revision = Some(agent.runtime.ledger().revision());
    agent.workspace.active_turn_message_start = Some(agent.messages().len());
    let gate = crate::change_ledger::install_scan_test_gate(agent.workspace_root());

    let cleanup = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        agent.cleanup_turn(crate::TurnCleanupKind::Cancel {
            session: crate::SessionRollback::AlreadyApplied,
        }),
    )
    .await
    .expect("a stalled final reconcile must not block cancellation")
    .unwrap();

    assert_eq!(cleanup.outcome.status, TurnStatus::Cancelled);
    assert_eq!(
        gate.exited(),
        gate.entered(),
        "every cleanup scan that started must observe cancellation"
    );
    assert!(agent.runtime.try_ledger().is_some());
}

#[test]
fn busy_ledger_finalizer_does_not_reuse_previous_turn_changes() {
    let workspace = IsolatedWorkspace::new("busy-ledger-stale-turn-files");
    let mut agent = agent(Vec::new(), workspace.config());
    let baseline = agent.runtime.ledger().revision();
    agent.workspace.active_turn_ledger_revision = Some(baseline);
    agent.workspace.last_changed_files = vec!["previous-turn.txt".into()];
    agent.workspace.last_file_changes = vec![hi_tools::FileChange {
        path: "previous-turn.txt".into(),
        kind: hi_tools::FileChangeKind::Modify,
        before_digest: Some("before".into()),
        after_digest: Some("after".into()),
        before_len: Some(6),
        after_len: Some(5),
        before_mode: Some(0o644),
        after_mode: Some(0o644),
    }];
    let ledger = agent.runtime.ledger_arc();
    let _held = ledger.lock().unwrap();

    let outcome = agent.finalize_failed_turn_snapshot_only();

    assert!(outcome.changed_files.is_empty());
    assert!(agent.last_changed_files().is_empty());
    assert!(agent.last_file_changes().is_empty());
}

#[tokio::test]
async fn plain_answer_returns_completed_not_applicable_outcome() {
    let workspace = IsolatedWorkspace::new("outcome-plain-answer");
    let mut cfg = workspace.config();
    cfg.routing.provider_route = Some("test-provider".into());
    let mut agent = agent(
        vec![completion(vec![Content::Text("42".into())], 1, 1)],
        cfg,
    );

    let outcome = agent
        .run_turn("what is six times seven?", &mut NullUi)
        .await
        .unwrap();

    assert_eq!(outcome.status, TurnStatus::Completed);
    assert_eq!(outcome.verification, VerificationStatus::NotApplicable);
    assert_eq!(outcome.review, ReviewStatus::NotRequired);
    assert_eq!(
        outcome.effective_route.provider.as_deref(),
        Some("test-provider")
    );
    assert_eq!(agent.last_turn_outcome(), Some(&outcome));
}

#[tokio::test]
async fn ambiguous_question_answered_in_text_completes() {
    let workspace = IsolatedWorkspace::new("outcome-question-answer");
    let mut agent = agent(
        vec![completion(
            vec![Content::Text(
                "It turns on automatically when the model does not fit in VRAM.".into(),
            )],
            1,
            1,
        )],
        workspace.config(),
    );

    // "how …" is not a recognized read-only opener, so the contract intent
    // defaults to mutation-capable. Answering it with text and no file
    // changes must still complete after the bounded implementation challenge.
    let outcome = agent
        .run_turn(
            "how do users use it? does that build hi-mlx or turn on automatically?",
            &mut NullUi,
        )
        .await
        .unwrap();

    assert_eq!(outcome.status, TurnStatus::Completed);
    assert_eq!(
        outcome.stop_reason,
        TurnStopReason::NoApplicableVerification
    );
    assert_eq!(outcome.verification, VerificationStatus::NotApplicable);
}

#[tokio::test]
async fn explicit_mutation_request_without_changes_settles_with_the_available_answer() {
    // Explicit mutation turns now share the implementation no-change cascade
    // (two edit nudges) before accepting the available user-visible answer.
    let workspace = IsolatedWorkspace::new("outcome-explicit-no-changes");
    let mut agent = agent(
        vec![
            completion(
                vec![Content::Text(
                    "The bug is in parser.rs line 42; an edit there would resolve it.".into(),
                )],
                1,
                1,
            ),
            completion(
                vec![Content::Text(
                    "Still diagnosing; the edit belongs in parser.rs.".into(),
                )],
                1,
                1,
            ),
            completion(
                vec![Content::Text(
                    "I would edit parser.rs but am not calling tools.".into(),
                )],
                1,
                1,
            ),
        ],
        workspace.config(),
    );
    let mut ui = RecordingUi::default();

    let outcome = agent.run_turn("fix the parser bug", &mut ui).await.unwrap();

    assert_eq!(outcome.status, TurnStatus::Completed);
    assert_eq!(outcome.verification, VerificationStatus::NotApplicable);
    assert_eq!(
        outcome.stop_reason,
        TurnStopReason::NoApplicableVerification
    );
    assert_eq!(
        agent.last_turn_telemetry().continue_nudges,
        2,
        "two no-change repair continues before bounded settlement"
    );
    assert!(
        ui.statuses.iter().any(|s| s.contains("no file changes")),
        "expected no-change repair status, got: {:?}",
        ui.statuses
    );
    assert!(
        !ui.statuses.iter().any(|status| {
            status.contains("incomplete") || status.to_ascii_lowercase().contains("stalled")
        }),
        "no-change repair must not manufacture a legacy terminal state: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn conversational_greenfield_request_cannot_complete_without_work() {
    let workspace = IsolatedWorkspace::new("outcome-conversational-greenfield-noop");
    let generic = || {
        completion(
            vec![Content::Text("Completed the requested action.".into())],
            1,
            1,
        )
    };
    let mut agent = agent(
        vec![generic(), generic(), generic(), generic()],
        workspace.config(),
    );
    let mut ui = RecordingUi::default();

    let outcome = agent
        .run_turn(
            "we want to build a twitter style app. we also want to seed it with agents. \
             we have each agent post between 6-8 times a day. each agent should have its own \
             name, hobby and topics. we want to spin up 30 new agents per hour. we also want \
             the agents to reply to other agents messages and like them.",
            &mut ui,
        )
        .await
        .unwrap();

    assert_eq!(outcome.status, TurnStatus::Completed);
    assert_eq!(
        outcome.stop_reason,
        TurnStopReason::NoApplicableVerification
    );
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("no file changes")),
        "the no-op implementation answer must be challenged: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn explicit_validation_request_requires_an_observed_successful_command() {
    let workspace = IsolatedWorkspace::new("outcome-explicit-validation");

    let mut agent = agent(
        vec![
            completion(
                vec![Content::Text("Completed the requested action.".into())],
                1,
                1,
            ),
            bash_completion("true # validate"),
            completion(
                vec![Content::Text("The requested validation passed.".into())],
                1,
                1,
            ),
        ],
        workspace.config(),
    );
    let mut ui = RecordingUi::default();

    let outcome = agent
        .run_turn(
            "Run cargo test --quiet before reporting that the work passes.",
            &mut ui,
        )
        .await
        .unwrap();

    assert_eq!(outcome.status, TurnStatus::Completed);
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("requested validation did not run")),
        "the generic completion must be challenged before the command runs: {:?}",
        ui.statuses
    );
    assert_eq!(agent.last_turn_telemetry().tool_calls, 1);
}

#[tokio::test]
async fn tool_using_turn_with_explicit_noop_answer_completes_after_challenge() {
    // A mutation request may legitimately conclude that no edit is needed,
    // but only after the implementation guard gives the model one explicit
    // edit-or-explain challenge. This also covers the live failure where a
    // fetch/read round was incorrectly allowed to settle as a generic
    // completion without that challenge.
    let workspace = IsolatedWorkspace::new("outcome-informed-no-changes");
    std::fs::create_dir_all(workspace.path("src")).unwrap();
    std::fs::write(workspace.path("src/parser.rs"), "fn parse() {}\n").unwrap();
    let mut agent = agent(
        vec![
            completion(
                vec![Content::ToolCall {
                    id: "r".into(),
                    name: "read".into(),
                    arguments: "{\"path\":\"src/parser.rs\"}".into(),
                }],
                1,
                1,
            ),
            completion(
                vec![Content::Text(
                    "The parser is already correct; the reported bug lives in the caller. \
                     No file changes are needed."
                        .into(),
                )],
                1,
                1,
            ),
            completion(
                vec![Content::Text(
                    "The parser is already correct; the reported bug lives in the caller. \
                     No file changes are needed."
                        .into(),
                )],
                1,
                1,
            ),
        ],
        workspace.config(),
    );
    let mut ui = RecordingUi::default();

    let outcome = agent.run_turn("fix the parser bug", &mut ui).await.unwrap();

    assert_eq!(outcome.status, TurnStatus::Completed);
    assert_eq!(
        outcome.stop_reason,
        TurnStopReason::NoApplicableVerification
    );
    assert!(
        agent.last_turn_telemetry().no_progress_streak == 0,
        "an unchallenged informed answer must not accumulate no-progress state"
    );
}

#[tokio::test]
async fn no_change_challenge_accepts_explicit_decline() {
    // The no-change nudge offers an escape hatch: edit now, or state plainly
    // that no file changes are needed. A challenged model that explicitly
    // declines completes the turn; the stall brand is reserved for a model
    // that agrees work is owed and never does it.
    let workspace = IsolatedWorkspace::new("outcome-no-change-decline");
    let mut agent = agent(
        vec![
            completion(
                vec![Content::Text(
                    "The reported bug does not reproduce; the parser handles this case.".into(),
                )],
                1,
                1,
            ),
            completion(
                vec![Content::Text(
                    "No file changes are needed — the parser already rejects empty input; \
                     the report was against an older build."
                        .into(),
                )],
                1,
                1,
            ),
        ],
        workspace.config(),
    );
    let mut ui = RecordingUi::default();

    let outcome = agent.run_turn("fix the parser bug", &mut ui).await.unwrap();

    assert_eq!(outcome.status, TurnStatus::Completed);
    assert_eq!(
        outcome.stop_reason,
        TurnStopReason::NoApplicableVerification
    );
    assert!(
        agent.last_turn_telemetry().no_progress_streak == 0,
        "an accepted decline must not accumulate no-progress state"
    );
    assert_eq!(
        agent.last_turn_telemetry().continue_nudges,
        1,
        "one no-change challenge, then the decline is accepted"
    );
    assert!(
        ui.statuses
            .iter()
            .any(|s| s.contains("no file changes are needed")),
        "decline acceptance should be visible in status, got: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn leftover_goal_rejects_explicit_mutation_decline() {
    // Live Flash stall: after a productive cap, the model text-declines
    // mutation ("already done") while the structured goal still has leftover
    // drive work. That must not take the no-change hatch — keep nudging the
    // goal instead of parking 9/9 remaining.
    let workspace = IsolatedWorkspace::new("outcome-goal-decline");
    let mut cfg = workspace.config();
    cfg.subagents.long_horizon = true;
    cfg.loop_limits.max_silent_continues = 1;
    let mut agent = agent(
        vec![
            completion(
                vec![Content::Text(
                    "The reported bug does not reproduce; the parser handles this case.".into(),
                )],
                1,
                1,
            ),
            completion(
                vec![Content::Text(
                    "No file changes are needed — Phase 1 is already complete.".into(),
                )],
                1,
                1,
            ),
        ],
        cfg,
    );
    assert!(
        agent
            .set_structured_goal(Some(crate::Goal::new(
                "ship phase 1 wallet auth",
                vec![
                    "implement the domain crate".into(),
                    "implement the api crate".into(),
                ],
            )))
            .unwrap()
    );
    let mut ui = RecordingUi::default();

    let _outcome = agent.run_turn("fix the parser bug", &mut ui).await.unwrap();

    assert!(
        agent.leftover_work().is_some(),
        "leftover goal work must still be queued"
    );
    assert!(
        !ui.statuses
            .iter()
            .any(|s| s.contains("no file changes are needed")),
        "leftover goal work must not take the mutation-decline hatch, got: {:?}",
        ui.statuses
    );
    assert!(
        agent
            .messages()
            .iter()
            .any(|m| m.text().contains(crate::GOAL_CONTINUE_NUDGE)),
        "leftover goal work should get the goal-continue nudge, last: {:?}",
        agent.messages().last().map(|m| m.text())
    );
}

#[tokio::test]
async fn explicit_mutation_text_only_gets_edit_repair_then_lands() {
    // Live fingerprint: "fix …" + text diagnosis used to settle without ever
    // forcing an edit. The cascade must nudge, then accept a write and complete.
    let workspace = IsolatedWorkspace::new("outcome-explicit-repair-lands");
    std::fs::create_dir_all(workspace.path("src")).unwrap();
    std::fs::write(workspace.path("src/parser.rs"), "fn parse() {}\n").unwrap();
    let mut cfg = workspace.config();
    cfg.gates.verification = crate::VerificationMode::Disabled;
    cfg.gates.allow_unverified = true;
    let mut agent = agent(
        vec![
            completion(
                vec![Content::Text(
                    "The bug is in parser.rs line 1; an edit there would resolve it.".into(),
                )],
                1,
                1,
            ),
            // After the no-change repair nudge the model edits and finishes.
            write_completion("src/parser.rs"),
            completion(vec![Content::Text("Fixed the parser bug.".into())], 1, 1),
        ],
        cfg,
    );
    let mut ui = RecordingUi::default();

    let outcome = agent.run_turn("fix the parser bug", &mut ui).await.unwrap();

    assert_eq!(outcome.status, TurnStatus::Completed);
    assert_eq!(outcome.verification, VerificationStatus::Unverified);
    assert_eq!(outcome.stop_reason, TurnStopReason::VerificationUnavailable);
    assert!(
        agent.last_turn_telemetry().no_progress_streak == 0,
        "successful write must not leave no-progress state behind"
    );
    assert!(
        ui.statuses.iter().any(|s| s.contains("no file changes")),
        "expected no-change repair nudge before the write, got: {:?}",
        ui.statuses
    );
    assert!(!ui.statuses.iter().any(|status| {
        status.contains("incomplete") || status.to_ascii_lowercase().contains("stalled")
    }));
}

#[tokio::test]
async fn managed_read_only_inspection_completes_despite_prior_mutation_context() {
    let workspace = IsolatedWorkspace::new("outcome-managed-read-only");
    std::fs::create_dir(workspace.path("src")).unwrap();
    std::fs::write(
        workspace.path("Cargo.toml"),
        "[package]\nname = \"sample\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    std::fs::write(workspace.path("README.md"), "# Sample\n").unwrap();
    std::fs::write(
        workspace.path("src/lib.rs"),
        "pub fn answer() -> u8 { 42 }\n",
    )
    .unwrap();
    let mut cfg = workspace.config();
    cfg.gates.read_only_preflight = true;
    let mut agent = agent(
        vec![
            completion(
                vec![Content::ToolCall {
                    id: "read-lib".into(),
                    name: "read".into(),
                    arguments: serde_json::json!({"path": "src/lib.rs"}).to_string(),
                }],
                10,
                5,
            ),
            completion(
                vec![Content::Text(
                    "Findings: Cargo.toml defines the sample crate; README.md identifies it as Sample; src/lib.rs exports answer().\n\nInspected Evidence: Cargo.toml, README.md, and src/lib.rs were read directly.\n\nFollow-up: none requested.\n\nLimits: this summary is limited to those three files. No changes were made."
                        .into(),
                )],
                10,
                30,
            ),
        ],
        cfg,
    );
    agent.set_managed_rsi_context(Some(
        r#"{"schema_version":1,"messages":[{"role":"user","content":"fix the parser bug and edit files"}]}"#
            .into(),
    ));

    let outcome = agent
        .run_turn(
            "Inspect Cargo.toml, README.md, and src/lib.rs, make no changes, and summarize what they contain.",
            &mut NullUi,
        )
        .await
        .unwrap();

    assert_eq!(outcome.status, TurnStatus::Completed);
    assert_eq!(outcome.verification, VerificationStatus::NotApplicable);
    assert_eq!(
        outcome.stop_reason,
        TurnStopReason::NoApplicableVerification
    );
    assert!(outcome.changed_files.is_empty());
}

#[tokio::test]
async fn public_rsi_skips_local_read_only_preflight() {
    let workspace = IsolatedWorkspace::new("outcome-public-rsi-no-local-preflight");
    std::fs::write(workspace.path("Cargo.toml"), "[workspace]\n").unwrap();
    let mut cfg = workspace.config();
    cfg.gates.read_only_preflight = true;
    cfg.rsi.remote_switch = Some(std::sync::Arc::new(std::sync::atomic::AtomicBool::new(
        true,
    )));
    let mut remote = completion(
        vec![Content::Text(
            "Remote inspection completed with evidence.".into(),
        )],
        10,
        10,
    );
    remote.stop_reason = Some("rsi_remote_completed".into());
    let (mut agent, requests) = scripted_agent(vec![ProviderStep::Completion(remote)], cfg);

    agent
        .run_turn("Inspect Cargo.toml and make no changes.", &mut NullUi)
        .await
        .unwrap();

    assert_eq!(agent.last_turn_telemetry().file_reads, 0);
    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert!(!requests[0].iter().any(|message| {
        message.content.iter().any(|content| {
            matches!(content, Content::ToolCall { id, .. } if id.starts_with("hi_preflight_"))
        })
    }));
}

#[tokio::test]
async fn mutation_without_verify_pipeline_is_not_applicable_not_unverified() {
    // Empty auto-detect workspace: no Cargo.toml/package.json/etc. After the
    // model directly validates the mutation, there is no remaining applicable
    // pipeline stage. That is `NotApplicable` / completed.
    let workspace = IsolatedWorkspace::new("outcome-no-pipeline-mutation");
    let path = "created.rs";
    let write = completion(
        vec![Content::ToolCall {
            id: "write-1".into(),
            name: "write".into(),
            arguments: serde_json::json!({ "path": path, "content": "changed\n" }).to_string(),
        }],
        1,
        1,
    );
    let smoke = bash_completion("true # validate");
    let done = completion(vec![Content::Text("done".into())], 1, 1);
    let mut cfg = workspace.config();
    cfg.gates.verification = VerificationMode::Auto;
    let mut agent = agent(vec![write, smoke, done], cfg);

    let outcome = agent
        .run_turn("create the file", &mut NullUi)
        .await
        .unwrap();
    assert_eq!(outcome.status, TurnStatus::Completed, "{outcome:#?}");
    assert_eq!(outcome.verification, VerificationStatus::NotApplicable);
    assert_eq!(
        outcome.stop_reason,
        TurnStopReason::NoApplicableVerification
    );
    assert!(outcome.changed_files.iter().any(|changed| changed == path));
}

#[tokio::test]
async fn disabled_verification_mutation_is_unverified_and_failed_by_default() {
    let workspace = IsolatedWorkspace::new("outcome-disabled-verify-mutation");
    let path = "created.rs";
    let mut cfg = workspace.config();
    cfg.gates.verification = crate::VerificationMode::Disabled;
    let write = completion(
        vec![Content::ToolCall {
            id: "write-1".into(),
            name: "write".into(),
            arguments: serde_json::json!({ "path": path, "content": "changed\n" }).to_string(),
        }],
        1,
        1,
    );
    let done = completion(vec![Content::Text("done".into())], 1, 1);
    let mut agent = agent(vec![write, done], cfg);

    let outcome = agent
        .run_turn("create the file", &mut NullUi)
        .await
        .unwrap();
    assert_eq!(outcome.status, TurnStatus::Failed);
    assert_eq!(outcome.verification, VerificationStatus::Unverified);
    assert_eq!(outcome.stop_reason, TurnStopReason::VerificationUnavailable);
}

#[tokio::test]
async fn disabled_verification_stays_unverified_after_model_run_smoke_check() {
    let workspace = IsolatedWorkspace::new("outcome-disabled-verify-with-smoke");
    let path = "created.rs";
    let mut cfg = workspace.config();
    cfg.gates.verification = crate::VerificationMode::Disabled;
    let write = completion(
        vec![Content::ToolCall {
            id: "write-1".into(),
            name: "write".into(),
            arguments: serde_json::json!({ "path": path, "content": "changed\n" }).to_string(),
        }],
        1,
        1,
    );
    let mut agent = agent(
        vec![
            write,
            bash_completion("true # validate"),
            completion(vec![Content::Text("done".into())], 1, 1),
        ],
        cfg,
    );

    let outcome = agent
        .run_turn("create the file", &mut NullUi)
        .await
        .unwrap();
    assert_eq!(outcome.status, TurnStatus::Failed);
    assert_eq!(outcome.verification, VerificationStatus::Unverified);
    assert_eq!(outcome.stop_reason, TurnStopReason::VerificationUnavailable);
}

#[tokio::test]
async fn failed_verify_budget_exhausted_stays_failed_not_not_applicable() {
    // A project with a verify pipeline that fails must still report Failed
    // after the repair budget — not quietly collapse to NotApplicable just
    // because the next outer-loop check returns NotRun (budget spent).
    let workspace = IsolatedWorkspace::new("outcome-failed-verify-budget");
    let mut cfg = workspace.config();
    cfg.gates.verification =
        crate::VerificationMode::Explicit(vec![crate::VerifyStage::new("fail", "false")]);
    cfg.gates.max_verify_repairs = 0;
    let path = workspace.path("changed.rs");
    let p = path.to_string_lossy().to_string();
    // write → text finish after fail-nudge → spare finish (same shape as verify.rs)
    let responses = vec![
        write_completion(&p),
        completion(vec![Content::Text("attempt 1".into())], 1, 1),
        completion(vec![Content::Text("attempt 2".into())], 1, 1),
    ];
    let mut agent = agent(responses, cfg);

    let outcome = agent
        .run_turn("change the file", &mut NullUi)
        .await
        .unwrap();
    assert_eq!(outcome.status, TurnStatus::Failed);
    assert_eq!(outcome.verification, VerificationStatus::Failed);
    assert_eq!(outcome.stop_reason, TurnStopReason::VerificationFailed);
}

#[tokio::test]
async fn independent_review_retries_once_after_transient_provider_error() {
    // A single rate-limit blip must not downgrade the review to
    // "unavailable" — one bounded retry runs first. Persistent failures
    // (or non-transient kinds) still report unavailable.
    let workspace = IsolatedWorkspace::new("outcome-review-retry");
    let provider = std::sync::Arc::new(ScriptedProvider {
        steps: Mutex::new(vec![
            ProviderStep::Error(hi_ai::ProviderErrorKind::RateLimit),
            ProviderStep::Completion(completion(vec![Content::Text("APPROVE".into())], 1, 1)),
        ]),
        requests: std::sync::Arc::new(Mutex::new(Vec::new())),
        max_tokens: None,
    });
    let mut agent = Agent::new(provider, workspace.config()).unwrap();

    let verdict = agent.independent_review("review context").await;

    assert_eq!(verdict, crate::agent::skeptic::SkepticVerdict::Approve);
}

#[tokio::test]
async fn independent_review_reports_unavailable_after_persistent_errors() {
    let workspace = IsolatedWorkspace::new("outcome-review-unavailable");
    let provider = std::sync::Arc::new(ScriptedProvider {
        steps: Mutex::new(vec![
            ProviderStep::Error(hi_ai::ProviderErrorKind::RateLimit),
            ProviderStep::Error(hi_ai::ProviderErrorKind::RateLimit),
        ]),
        requests: std::sync::Arc::new(Mutex::new(Vec::new())),
        max_tokens: None,
    });
    let mut agent = Agent::new(provider, workspace.config()).unwrap();

    let verdict = agent.independent_review("review context").await;

    assert!(matches!(
        verdict,
        crate::agent::skeptic::SkepticVerdict::Unavailable(_)
    ));
}

/// IR integration helpers: prose paths avoid mid-turn Rust cargo/LSP feedback
/// (which can consume ~30s and extra model rounds in empty workspaces).
fn independent_review_cfg(workspace: &IsolatedWorkspace) -> AgentConfig {
    // `config()` initializes the test process's sandbox policy once, before
    // the runner is constructed. Avoid another process-global env write here.
    let mut cfg = workspace.config();
    cfg.gates.verification = VerificationMode::Explicit(vec![VerifyStage::new("test", "true")]);
    cfg.gates.review = ReviewPolicy::Always;
    cfg.gates.lsp_mode = LspMode::Off;
    // Keep YOLO checkpoints so empty isolated workspaces can still mutate via
    // the internal snapshot backend; ReviewPolicy::Always still forces IR.
    cfg.gates.allow_no_checkpoint = true;
    cfg
}

fn write_file_completion(id: &str, path: &str, content: &str) -> Completion {
    completion(
        vec![Content::ToolCall {
            id: id.into(),
            name: "write".into(),
            arguments: serde_json::json!({ "path": path, "content": content }).to_string(),
        }],
        1,
        1,
    )
}

fn edit_file_completion(id: &str, path: &str, old: &str, new: &str) -> Completion {
    completion(
        vec![Content::ToolCall {
            id: id.into(),
            name: "edit".into(),
            arguments: serde_json::json!({
                "path": path,
                "old_string": old,
                "new_string": new,
            })
            .to_string(),
        }],
        1,
        1,
    )
}

#[tokio::test]
async fn repeated_model_authored_validation_failure_is_bounded_across_edits() {
    let workspace = IsolatedWorkspace::new("outcome-repeated-validation");
    let failure = "printf 'running 1 test\\ntest moves::checkmate_detected ... FAILED\\n\\nfailures:\\n    moves::checkmate_detected\\n\\ntest result: FAILED. 0 passed; 1 failed\\n' >&2; false # cargo test";
    let steps = vec![
        ProviderStep::Completion(write_file_completion("write-state", "state.rs", "one\n")),
        ProviderStep::Completion(bash_completion(failure)),
        ProviderStep::Completion(edit_file_completion(
            "edit-state-1",
            "state.rs",
            "one\n",
            "two\n",
        )),
        ProviderStep::Completion(bash_completion(failure)),
        ProviderStep::Completion(edit_file_completion(
            "edit-state-2",
            "state.rs",
            "two\n",
            "three\n",
        )),
        ProviderStep::Completion(bash_completion(failure)),
    ];
    let mut cfg = workspace.config();
    cfg.gates.verification = VerificationMode::Disabled;
    cfg.memory.finalize = false;
    cfg.memory.suggest_next_prompt = false;
    let (mut agent, requests) = scripted_agent(steps, cfg);
    let mut ui = RecUi::default();

    let outcome = agent
        .run_turn("build a small game project", &mut ui)
        .await
        .unwrap();

    assert_eq!(outcome.status, TurnStatus::Failed);
    assert_eq!(requests.lock().unwrap().len(), 6);
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("focused root-cause diagnosis")),
        "second unchanged repair should trigger diagnosis: {:?}",
        ui.statuses
    );
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("persisted after focused repair")),
        "third unchanged repair should stop the bounded run: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn independent_review_status_is_emitted_in_turn_outcome() {
    let workspace = IsolatedWorkspace::new("outcome-review");
    let path = "reviewed.txt";
    // write → validate (satisfy implementation completeness) → done → IR APPROVE
    let responses = vec![
        write_file_completion("write-review", path, "reviewed\n"),
        bash_completion("true # validate"),
        completion(vec![Content::Text("done".into())], 1, 1),
        completion(vec![Content::Text("APPROVE".into())], 1, 1),
    ];
    let mut agent = agent(responses, independent_review_cfg(&workspace));

    let outcome = agent
        .run_turn("create the reviewed file", &mut NullUi)
        .await
        .unwrap();
    assert_eq!(outcome.status, TurnStatus::Completed);
    assert_eq!(outcome.verification, VerificationStatus::Passed);
    assert_eq!(outcome.review, ReviewStatus::Passed);
    // Default cfg has no distinct skeptic_model → same-model observability.
    assert!(
        outcome.review_same_model,
        "unconfigured skeptic_model should flag same-model review"
    );
}

#[tokio::test]
async fn completion_review_receives_the_canonical_greenfield_objective() {
    let workspace = IsolatedWorkspace::new("outcome-review-objective");
    let prompt = "lets build a chess TUI game and also fully enable mouse usage";
    let steps = vec![
        ProviderStep::Completion(write_file_completion(
            "write-review-objective",
            "reviewed.txt",
            "reviewed\n",
        )),
        ProviderStep::Completion(bash_completion("true # validate")),
        ProviderStep::Completion(completion(vec![Content::Text("done".into())], 1, 1)),
        ProviderStep::Completion(completion(vec![Content::Text("APPROVE".into())], 1, 1)),
    ];
    let (mut agent, requests) = scripted_agent(steps, independent_review_cfg(&workspace));

    agent.run_turn(prompt, &mut NullUi).await.unwrap();

    let requests = requests.lock().unwrap();
    assert!(
        requests.iter().any(|request| {
            request.iter().any(|message| {
                message.text().contains("Canonical user objective:")
                    && message.text().contains(prompt)
            })
        }),
        "completion reviewer never received the original objective: {requests:#?}"
    );
}

#[tokio::test]
async fn hygiene_gate_reenters_model_on_unreferenced_creates() {
    let workspace = IsolatedWorkspace::new("outcome-hygiene-sprawl");
    let mut cfg = independent_review_cfg(&workspace);
    cfg.gates.review = ReviewPolicy::Off;
    cfg.gates.max_independent_review_repairs = 1;
    let responses = vec![
        completion(
            vec![
                Content::ToolCall {
                    id: "w0".into(),
                    name: "write".into(),
                    arguments: serde_json::json!({
                        "path": "src/parser.rs",
                        "content": "fn parse() {}\n"
                    })
                    .to_string(),
                },
                Content::ToolCall {
                    id: "w1".into(),
                    name: "write".into(),
                    arguments: serde_json::json!({
                        "path": "extra_a.rs",
                        "content": "a\n"
                    })
                    .to_string(),
                },
                Content::ToolCall {
                    id: "w2".into(),
                    name: "write".into(),
                    arguments: serde_json::json!({
                        "path": "extra_b.rs",
                        "content": "b\n"
                    })
                    .to_string(),
                },
                Content::ToolCall {
                    id: "w3".into(),
                    name: "write".into(),
                    arguments: serde_json::json!({
                        "path": "extra_c.rs",
                        "content": "c\n"
                    })
                    .to_string(),
                },
            ],
            1,
            1,
        ),
        bash_completion("true # validate"),
        completion(vec![Content::Text("done".into())], 1, 1),
        bash_completion("true # hygiene repair"),
        completion(vec![Content::Text("repaired".into())], 1, 1),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    let outcome = agent.run_turn("fix src/parser.rs", &mut ui).await.unwrap();
    assert!(
        ui.statuses.iter().any(|s| s.contains("diff hygiene")),
        "hygiene should re-enter the model: {:?}",
        ui.statuses
    );
    assert_eq!(outcome.status, TurnStatus::Completed);
}

#[tokio::test]
async fn independent_review_unavailable_completes_with_visible_status() {
    // Soft transport failure: an IR outage must not fail a green turn.
    let workspace = IsolatedWorkspace::new("outcome-review-unavailable");
    let path = "reviewed.txt";
    let steps = vec![
        ProviderStep::Completion(write_file_completion("write-review", path, "reviewed\n")),
        ProviderStep::Completion(bash_completion("true # validate")),
        ProviderStep::Completion(completion(vec![Content::Text("done".into())], 1, 1)),
        // IR retries once on transient error before Unavailable.
        ProviderStep::Error(ProviderErrorKind::Outage),
        ProviderStep::Error(ProviderErrorKind::Outage),
    ];
    let (mut agent, _requests) = scripted_agent(steps, independent_review_cfg(&workspace));

    let outcome = agent
        .run_turn("create the reviewed file", &mut NullUi)
        .await
        .unwrap();
    assert_eq!(outcome.status, TurnStatus::Completed);
    assert_eq!(outcome.verification, VerificationStatus::Passed);
    assert_eq!(outcome.review, ReviewStatus::Unavailable);
    assert!(outcome.review_same_model);
}

#[tokio::test]
async fn independent_review_distinct_skeptic_model_clears_same_model_flag() {
    let workspace = IsolatedWorkspace::new("outcome-review-distinct-model");
    let path = "reviewed.txt";
    let responses = vec![
        write_file_completion("write-review", path, "reviewed\n"),
        bash_completion("true # validate"),
        completion(vec![Content::Text("done".into())], 1, 1),
        completion(vec![Content::Text("APPROVE".into())], 1, 1),
    ];
    let mut cfg = independent_review_cfg(&workspace);
    cfg.subagents.skeptic_model = Some("skeptic-other".into());
    let mut agent = agent(responses, cfg);
    assert!(
        !agent.skeptic_shares_session_model(),
        "configured distinct skeptic_model must not share session model"
    );

    let outcome = agent
        .run_turn("create the reviewed file", &mut NullUi)
        .await
        .unwrap();
    assert_eq!(outcome.review, ReviewStatus::Passed);
    assert!(
        !outcome.review_same_model,
        "distinct skeptic_model must clear the same-model flag"
    );
}

#[tokio::test]
async fn independent_review_object_allows_one_repair_then_pass() {
    // Object once → re-enter Model for repair → re-verify → second review APPROVE.
    let workspace = IsolatedWorkspace::new("outcome-review-object-pass");
    let path = "fixed.txt";
    let responses = vec![
        write_file_completion("write-review", path, "v1\n"),
        bash_completion("true # validate"),
        completion(vec![Content::Text("done".into())], 1, 1),
        completion(
            vec![Content::Text(
                "OBJECT\n- missing error handling on the happy path".into(),
            )],
            1,
            1,
        ),
        write_file_completion("repair-write", path, "v2 fixed\n"),
        bash_completion("true # validate"),
        completion(vec![Content::Text("repaired".into())], 1, 1),
        completion(vec![Content::Text("APPROVE".into())], 1, 1),
    ];
    let mut cfg = independent_review_cfg(&workspace);
    cfg.gates.max_independent_review_repairs = 1;
    let mut agent = agent(responses, cfg);

    let outcome = agent
        .run_turn("implement the reviewed file", &mut NullUi)
        .await
        .unwrap();

    assert_eq!(outcome.status, TurnStatus::Completed);
    assert_eq!(outcome.verification, VerificationStatus::Passed);
    assert_eq!(outcome.review, ReviewStatus::Passed);
}

#[tokio::test]
async fn default_independent_review_repairs_continue_past_one_productive_cycle() {
    let workspace = IsolatedWorkspace::new("outcome-review-unlimited-default");
    let path = "reviewed-twice.txt";
    let responses = vec![
        write_file_completion("write-review", path, "v1\n"),
        bash_completion("true # validate"),
        completion(vec![Content::Text("initial implementation".into())], 1, 1),
        completion(
            vec![Content::Text("OBJECT\n- first concrete defect".into())],
            1,
            1,
        ),
        write_file_completion("repair-one", path, "v2\n"),
        bash_completion("true # validate"),
        completion(vec![Content::Text("first repair".into())], 1, 1),
        completion(
            vec![Content::Text("OBJECT\n- second concrete defect".into())],
            1,
            1,
        ),
        write_file_completion("repair-two", path, "v3 fixed\n"),
        bash_completion("true # validate"),
        completion(vec![Content::Text("second repair".into())], 1, 1),
        completion(vec![Content::Text("APPROVE".into())], 1, 1),
    ];
    let cfg = independent_review_cfg(&workspace);
    assert_eq!(
        cfg.gates.max_independent_review_repairs,
        crate::UNLIMITED_REPAIR_CYCLES
    );
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();

    let outcome = agent
        .run_turn(
            "implement the reviewed file through every productive repair",
            &mut ui,
        )
        .await
        .unwrap();

    assert_eq!(outcome.status, TurnStatus::Completed);
    assert_eq!(outcome.review, ReviewStatus::Passed);
    assert_eq!(
        std::fs::read_to_string(workspace.path(path)).unwrap(),
        "v3 fixed\n"
    );
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("repair cycle 2/unlimited")),
        "the second productive review repair must be allowed: {:?}",
        ui.statuses
    );
    assert!(
        ui.statuses
            .iter()
            .all(|status| !status.contains("4294967295")),
        "the unlimited sentinel must not leak into UI text: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn independent_review_escalate_allows_one_repair_then_pass() {
    // Stray ESCALATE (goal-skeptic vocabulary) maps to Object so completion
    // review still gets a repair cycle when budget remains.
    let workspace = IsolatedWorkspace::new("outcome-review-escalate-pass");
    let path = "escalated.txt";
    let responses = vec![
        write_file_completion("write-review", path, "v1\n"),
        bash_completion("true # validate"),
        completion(vec![Content::Text("done".into())], 1, 1),
        completion(
            vec![Content::Text(
                "ESCALATE\n- needs a clearer error path before merge".into(),
            )],
            1,
            1,
        ),
        write_file_completion("repair-write", path, "v2 fixed\n"),
        bash_completion("true # validate"),
        completion(vec![Content::Text("repaired".into())], 1, 1),
        completion(vec![Content::Text("APPROVE".into())], 1, 1),
    ];
    let mut cfg = independent_review_cfg(&workspace);
    cfg.gates.max_independent_review_repairs = 1;
    let mut agent = agent(responses, cfg);

    let outcome = agent
        .run_turn("implement the escalated file", &mut NullUi)
        .await
        .unwrap();

    assert_eq!(outcome.status, TurnStatus::Completed);
    assert_eq!(outcome.verification, VerificationStatus::Passed);
    assert_eq!(outcome.review, ReviewStatus::Passed);
}

#[tokio::test]
async fn independent_review_object_again_after_repair_completes_with_scar() {
    // Object → one repair cycle → Object again → no second cycle. Deterministic
    // verification passed, so the exhausted objection rides as a scar on a
    // Completed turn instead of stalling verified work on a reviewer opinion.
    let workspace = IsolatedWorkspace::new("outcome-review-object-again");
    let path = "stuck.txt";
    let responses = vec![
        write_file_completion("write-review", path, "v1\n"),
        bash_completion("true # validate"),
        completion(vec![Content::Text("done".into())], 1, 1),
        completion(
            vec![Content::Text("OBJECT\n- incomplete implementation".into())],
            1,
            1,
        ),
        write_file_completion("repair-write", path, "v2 still broken\n"),
        bash_completion("true # validate"),
        completion(vec![Content::Text("tried".into())], 1, 1),
        completion(
            vec![Content::Text(
                "OBJECT\n- still incomplete after repair".into(),
            )],
            1,
            1,
        ),
    ];
    let mut cfg = independent_review_cfg(&workspace);
    cfg.gates.max_independent_review_repairs = 1;
    let mut agent = agent(responses, cfg);

    let outcome = agent
        .run_turn("implement the stuck file", &mut NullUi)
        .await
        .unwrap();

    assert_eq!(outcome.status, TurnStatus::Completed);
    assert_eq!(outcome.review, ReviewStatus::Objected);
    assert_eq!(outcome.stop_reason, TurnStopReason::ReviewObjected);
}

#[tokio::test]
async fn independent_review_zero_repair_budget_records_scar_immediately() {
    // With no repair budget the objection is final on the first pass; green
    // verification still outranks it (Completed + scar).
    let workspace = IsolatedWorkspace::new("outcome-review-zero-repair");
    let path = "no-repair.txt";
    let responses = vec![
        write_file_completion("write-review", path, "v1\n"),
        bash_completion("true # validate"),
        completion(vec![Content::Text("done".into())], 1, 1),
        completion(vec![Content::Text("OBJECT\n- defect".into())], 1, 1),
    ];
    let mut cfg = independent_review_cfg(&workspace);
    cfg.gates.max_independent_review_repairs = 0;
    let mut agent = agent(responses, cfg);

    let outcome = agent
        .run_turn("implement without review repair", &mut NullUi)
        .await
        .unwrap();

    assert_eq!(outcome.status, TurnStatus::Completed);
    assert_eq!(outcome.review, ReviewStatus::Objected);
    assert_eq!(outcome.stop_reason, TurnStopReason::ReviewObjected);
}

#[tokio::test]
async fn mutation_after_verification_invalidates_pass_and_verified_revision() {
    let root = std::env::temp_dir().join(format!(
        "hi-agent-late-review-mutation-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let write = completion(
        vec![Content::ToolCall {
            id: "write-review".into(),
            name: "write".into(),
            arguments: serde_json::json!({ "path": "work.rs", "content": "checked\n" }).to_string(),
        }],
        1,
        1,
    );
    let provider = std::sync::Arc::new(ReviewMutationProvider {
        responses: Mutex::new(vec![
            write,
            completion(vec![Content::Text("done".into())], 1, 1),
            completion(vec![Content::Text("APPROVE".into())], 1, 1),
        ]),
        calls: std::sync::atomic::AtomicUsize::new(0),
        root: root.clone(),
    });
    let mut cfg = config();
    cfg.paths.workspace_root = root.clone();
    cfg.paths.state_root = root.join(".hi/state");
    cfg.gates.verification = VerificationMode::Explicit(vec![VerifyStage::new("test", "true")]);
    cfg.gates.review = ReviewPolicy::Always;
    cfg.gates.allow_no_checkpoint = false;
    let mut agent = Agent::new(provider, cfg).unwrap();

    let outcome = agent
        .run_turn("implement the reviewed file", &mut NullUi)
        .await
        .unwrap();

    assert_eq!(outcome.status, TurnStatus::Failed);
    assert_eq!(outcome.verification, VerificationStatus::Unverified);
    assert_eq!(outcome.review, ReviewStatus::Unavailable);
    assert!(outcome.verified_workspace_revision.is_none());
    assert!(outcome.changed_files.contains(&"work.rs".to_string()));
    assert!(outcome.changed_files.contains(&"late.rs".to_string()));
    let _ = std::fs::remove_dir_all(root);
}

#[tokio::test]
async fn ui_turn_end_mutation_cannot_create_a_false_current_revision_pass() {
    let root =
        std::env::temp_dir().join(format!("hi-agent-turn-end-mutation-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let write = completion(
        vec![Content::ToolCall {
            id: "write".into(),
            name: "write".into(),
            arguments: serde_json::json!({ "path": "work.rs", "content": "checked\n" }).to_string(),
        }],
        1,
        1,
    );
    let mut cfg = config();
    cfg.paths.workspace_root = root.clone();
    cfg.paths.state_root = root.join(".hi/state");
    cfg.gates.verification = VerificationMode::Explicit(vec![VerifyStage::new("test", "true")]);
    let mut agent = agent(
        vec![write, completion(vec![Content::Text("done".into())], 1, 1)],
        cfg,
    );
    let mut ui = TurnEndMutationUi { root: root.clone() };

    let outcome = agent.run_turn("implement work.rs", &mut ui).await.unwrap();

    assert_eq!(outcome.status, TurnStatus::Failed);
    assert_eq!(outcome.verification, VerificationStatus::Unverified);
    assert!(outcome.verified_workspace_revision.is_none());
    assert!(outcome.changed_files.contains(&"work.rs".to_string()));
    assert!(outcome.changed_files.contains(&"late.rs".to_string()));
    let _ = std::fs::remove_dir_all(root);
}

#[tokio::test]
async fn yolo_default_continues_without_undo_and_never_prompts() {
    let root =
        std::env::temp_dir().join(format!("hi-agent-checkpoint-yolo-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join("data")).unwrap();
    // Oversized NON-artifact data: artifact-named trees no longer count
    // toward checkpoint limits, so genuine failure needs unskippable bytes.
    let huge = std::fs::File::create(root.join("data/blob.bin")).unwrap();
    huge.set_len(512 * 1024 * 1024 + 1).unwrap();
    let write = completion(
        vec![Content::ToolCall {
            id: "write-target-yolo".into(),
            name: "write".into(),
            arguments: serde_json::json!({
                "path": "src_new.rs",
                "content": "fn generated() {}\n"
            })
            .to_string(),
        }],
        1,
        1,
    );
    let mut cfg = config();
    cfg.paths.workspace_root = root.clone();
    cfg.paths.state_root = root.join(".hi/state");
    assert!(cfg.gates.allow_no_checkpoint, "YOLO must be the default");
    let mut agent = agent(
        vec![
            write,
            completion(vec![Content::Text("edited".into())], 1, 1),
        ],
        cfg,
    );
    let mut ui = RejectAllConfirmUi::default();

    agent
        .run_turn("write target/new.rs", &mut ui)
        .await
        .unwrap();

    assert_eq!(ui.confirm_calls, 0, "missing /undo must never prompt");
    assert!(
        ui.checkpoint_warnings.is_empty(),
        "default YOLO checkpoint failures must stay silent: {:?}",
        ui.checkpoint_warnings
    );
    assert_eq!(
        std::fs::read_to_string(root.join("src_new.rs")).unwrap(),
        "fn generated() {}\n"
    );
    assert_eq!(
        agent.last_turn_telemetry().checkpoint_available,
        Some(false)
    );
    let _ = std::fs::remove_dir_all(root);
}

#[tokio::test]
async fn missing_checkpoint_does_not_bypass_large_diff_risk_review() {
    let root = std::env::temp_dir().join(format!(
        "hi-agent-risk-review-no-checkpoint-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join("target")).unwrap();
    // Force both checkpoint backends past the per-checkpoint ceiling. YOLO
    // mode should continue, while review still sees the complete live diff.
    let huge = std::fs::File::create(root.join("target/cache.bin")).unwrap();
    huge.set_len(512 * 1024 * 1024 + 1).unwrap();
    let content = (0..301)
        .map(|line| format!("line {line}\n"))
        .collect::<String>();
    let write = completion(
        vec![Content::ToolCall {
            id: "write-large".into(),
            name: "write".into(),
            arguments: serde_json::json!({ "path": "large.rs", "content": content }).to_string(),
        }],
        1,
        1,
    );
    let mut cfg = config();
    cfg.paths.workspace_root = root.clone();
    cfg.paths.state_root = root.join(".hi/state");
    cfg.gates.verification = VerificationMode::Explicit(vec![VerifyStage::new("test", "true")]);
    cfg.gates.review = ReviewPolicy::Risk;
    // Keep the default YOLO fallback on: this is specifically the path where
    // no complete checkpoint-backed diff exists.
    assert!(cfg.gates.allow_no_checkpoint);
    let mut agent = agent(
        vec![write, completion(vec![Content::Text("done".into())], 1, 1)],
        cfg,
    );

    let outcome = agent
        .run_turn("implement the large source file", &mut NullUi)
        .await
        .unwrap();

    assert_eq!(outcome.verification, VerificationStatus::Passed);
    assert_eq!(outcome.review, ReviewStatus::Unavailable);
    let _ = std::fs::remove_dir_all(root);
}

#[tokio::test]
async fn infrastructure_finalizer_reconciles_ui_effects_after_session_failure() {
    let root =
        std::env::temp_dir().join(format!("hi-agent-failed-finalizer-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let write = completion(
        vec![Content::ToolCall {
            id: "write".into(),
            name: "write".into(),
            arguments: serde_json::json!({ "path": "work.rs", "content": "checked\n" }).to_string(),
        }],
        1,
        1,
    );
    let mut cfg = config();
    cfg.paths.workspace_root = root.clone();
    cfg.paths.state_root = root.join(".hi/state");
    cfg.gates.verification = VerificationMode::Explicit(vec![VerifyStage::new("test", "true")]);
    let mut agent = agent(
        vec![write, completion(vec![Content::Text("done".into())], 1, 1)],
        cfg,
    );
    agent.set_session(Box::new(FailingRecordSession));
    let mut ui = TurnEndMutationUi { root: root.clone() };

    let error = agent
        .run_turn("implement work.rs", &mut ui)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("session persistence failed"));
    let outcome = agent
        .cleanup_turn(crate::TurnCleanupKind::Fail)
        .await
        .unwrap()
        .outcome;

    assert_eq!(outcome.status, TurnStatus::Failed);
    // The late UI mutation invalidates the earlier pass, but a session write
    // failure is not a verifier infrastructure failure.
    assert_eq!(outcome.verification, VerificationStatus::Unverified);
    assert!(outcome.changed_files.contains(&"work.rs".to_string()));
    assert!(outcome.changed_files.contains(&"late.rs".to_string()));
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn turn_outcome_exit_codes_match_one_shot_table() {
    use crate::{EffectiveModelRoute, ReviewStatus, TurnStopReason};

    let route = EffectiveModelRoute {
        provider: None,
        model: "m".into(),
    };
    let base = |status, verification, stop_reason| TurnOutcome {
        status,
        verification,
        review: ReviewStatus::NotRequired,
        stop_reason,
        changed_files: Vec::new(),
        verified_workspace_revision: None,
        effective_route: route.clone(),
        review_same_model: false,
        leftover: None,
        plan_leftover: None,
    };
    assert_eq!(
        base(
            TurnStatus::Cancelled,
            VerificationStatus::Unverified,
            TurnStopReason::Cancelled
        )
        .exit_code(false),
        130
    );
    assert_eq!(
        base(
            TurnStatus::Failed,
            VerificationStatus::InfrastructureError,
            TurnStopReason::InfrastructureFailure
        )
        .exit_code(false),
        3
    );
    assert_eq!(
        base(
            TurnStatus::Failed,
            VerificationStatus::Unverified,
            TurnStopReason::VerificationUnavailable
        )
        .exit_code(false),
        1
    );
    assert_eq!(
        base(
            TurnStatus::Completed,
            VerificationStatus::Passed,
            TurnStopReason::Completed
        )
        .exit_code(false),
        0
    );
    assert_eq!(
        base(
            TurnStatus::Completed,
            VerificationStatus::Unverified,
            TurnStopReason::NoApplicableVerification
        )
        .exit_code(false),
        1
    );
    assert_eq!(
        base(
            TurnStatus::Completed,
            VerificationStatus::Unverified,
            TurnStopReason::NoApplicableVerification
        )
        .exit_code(true),
        0
    );
}

#[tokio::test]
async fn expired_soft_deadline_settles_the_turn_instead_of_aborting() {
    // An already-expired budget must end the turn through the normal Settle
    // path — a real outcome with TimeLimit — not an Err like the hard
    // `turn_timeout` produces, and without consuming a model call.
    let mut cfg = config();
    // Zero, not a tiny non-zero budget: the deadline is anchored inside the
    // turn, so `now + 1ns` is only reliably expired if the clock ticks between
    // two adjacent reads — which it need not, and under a loaded test suite it
    // did not, letting the check fire one loop deeper than intended.
    cfg.loop_limits.turn_soft_deadline = Some(std::time::Duration::ZERO);
    // No canned completions: reaching the provider at all would panic, which
    // is exactly the assertion that no new work was started.
    let mut agent = agent(Vec::new(), cfg);

    let outcome = agent
        .run_turn("do something expensive", &mut NullUi)
        .await
        .expect("an expired budget settles the turn rather than erroring");

    assert_eq!(outcome.stop_reason, TurnStopReason::TimeLimit);
    // Nothing was mutated, so the turn is honestly complete, not a failure.
    assert_eq!(outcome.status, TurnStatus::Completed);
}

#[tokio::test]
async fn hard_turn_timeout_runs_cancellation_cleanup_before_returning() {
    let workspace = IsolatedWorkspace::new("hard-timeout-cleanup");
    let mut cfg = workspace.config();
    cfg.routing.tool_mode = ToolMode::ChatOnly;
    cfg.loop_limits.turn_timeout = Some(std::time::Duration::from_millis(20));
    let probe = std::sync::Arc::new(TurnLifecycleProbe::default());
    let stuck_probe = std::sync::Arc::new(StuckAbortLifecycleProbe::default());
    let mut builder = hi_agent_lifecycle::ExtensionRegistryBuilder::new();
    builder.turn_lifecycle_contributor(stuck_probe.clone());
    builder.turn_lifecycle_contributor(probe.clone());
    let mut agent = Agent::new(std::sync::Arc::new(PendingProvider), cfg)
        .unwrap()
        .with_extension_registry(builder.build());
    let history_len = agent.messages().len();

    let error = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        agent.run_turn("wait forever", &mut NullUi),
    )
    .await
    .expect("a stuck abort contributor must not defeat the hard backstop")
    .unwrap_err();

    assert!(error.to_string().contains("turn deadline exceeded"));
    assert!(agent.turn_cancellation.is_none());
    assert!(!agent.interrupt.load(std::sync::atomic::Ordering::Acquire));
    assert!(agent.workspace.active_turn_ledger_revision.is_none());
    assert!(agent.workspace.active_turn_message_start.is_none());
    assert!(agent.workspace.active_turn_background_baseline.is_none());
    assert_eq!(agent.turn_phase(), TurnPhase::Done);
    assert_eq!(
        probe.starts.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "the dropped body had started exactly one lifecycle turn"
    );
    assert_eq!(
        probe.done.load(std::sync::atomic::Ordering::SeqCst),
        0,
        "the hard cancellation backstop must not dispatch done"
    );
    assert_eq!(
        *probe.aborts.lock().unwrap(),
        vec![hi_agent_lifecycle::TurnAbortReason::Interrupted],
        "the hard backstop must dispatch exactly one abort"
    );
    assert_eq!(
        stuck_probe.aborts.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "even a wedged abort callback is attempted exactly once"
    );
    assert_eq!(
        agent.messages().len(),
        history_len,
        "the cancelled prompt must be rolled back"
    );
}

#[tokio::test]
async fn hard_turn_timeout_cancels_an_inflight_ledger_reconcile() {
    let workspace = IsolatedWorkspace::new("hard-timeout-ledger-reconcile");
    let mut cfg = workspace.config();
    cfg.routing.tool_mode = ToolMode::ChatOnly;
    cfg.loop_limits.turn_timeout = Some(std::time::Duration::from_millis(20));
    let mut agent = Agent::new(std::sync::Arc::new(PendingProvider), cfg).unwrap();
    let gate = crate::change_ledger::install_scan_test_gate(agent.workspace_root());

    let error = tokio::time::timeout(
        std::time::Duration::from_secs(3),
        agent.run_turn("wait in ledger setup", &mut NullUi),
    )
    .await
    .expect("a ledger worker must not defeat the hard turn deadline")
    .unwrap_err();

    assert!(error.to_string().contains("turn deadline exceeded"));
    assert_eq!(
        gate.exited(),
        gate.entered(),
        "every deadline scan that started must observe cancellation"
    );
    assert!(agent.runtime.try_ledger().is_some());
}

#[tokio::test]
async fn slow_cancellation_rollback_is_owned_and_awaited_exactly_once() {
    let workspace = IsolatedWorkspace::new("slow-cancel-rollback");
    let mut cfg = workspace.config();
    cfg.gates.verification = VerificationMode::Disabled;
    cfg.memory.finalize = false;
    cfg.memory.suggest_next_prompt = false;
    let mut agent = agent(vec![write_completion("cancelled.txt")], cfg);
    let rollback_calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    // Longer than COOPERATIVE_CANCEL_GRACE. With rollback inside the droppable
    // body this first attempt was detached at 750ms and cleanup was re-entered.
    agent.undo_test_probe = Some((
        std::time::Duration::from_millis(900),
        rollback_calls.clone(),
    ));
    let checkpoint_count_before = agent.checkpoint_count();
    let cancellation = TurnCancellation::new();
    let mut ui = CancelAfterToolUi {
        cancellation: cancellation.clone(),
        fired: false,
    };

    let outcome = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        agent.run_turn_cancellable("write cancelled.txt", &mut ui, cancellation),
    )
    .await
    .expect("a slow rollback must be awaited, not abandoned")
    .expect("cancellation cleanup should produce a typed outcome");

    assert_eq!(outcome.status, TurnStatus::Cancelled);
    assert_eq!(
        rollback_calls.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "rollback was dropped and re-entered"
    );
    assert_eq!(agent.checkpoint_count(), checkpoint_count_before);
    assert!(
        !workspace.path("cancelled.txt").exists(),
        "the single rollback did not restore the pre-turn workspace"
    );
}

#[tokio::test]
async fn cancelled_plan_drive_persists_interruption_before_transcript_rewind() {
    let workspace = IsolatedWorkspace::new("cancel-plan-pause-order");
    let mut cfg = workspace.config();
    cfg.gates.verification = VerificationMode::Disabled;
    cfg.memory.finalize = false;
    cfg.memory.suggest_next_prompt = false;
    let cancellation = TurnCancellation::new();
    let provider = std::sync::Arc::new(CancelThenCompleteProvider {
        cancellation: cancellation.clone(),
    });
    let mut agent = Agent::new(provider, cfg).unwrap();
    agent.restore_plan(vec![PlanStep {
        title: "finish safely".into(),
        status: PlanStatus::Active,
    }]);
    let records = std::sync::Arc::new(Mutex::new(Vec::new()));
    agent.set_session(Box::new(PlanPauseOrderSession {
        records: records.clone(),
    }));

    let outcome = agent
        .run_turn_cancellable(PLAN_DRIVE_PROMPT, &mut NullUi, cancellation)
        .await
        .expect("cancellation should settle to a typed outcome");

    assert_eq!(outcome.status, TurnStatus::Cancelled);
    assert!(agent.plan_drive_paused());
    let records = records.lock().unwrap();
    let pause = records
        .iter()
        .position(|record| record == "pause:true:true")
        .expect("interruption pause record");
    let rewind = records
        .iter()
        .position(|record| record == "rewind")
        .expect("transcript rewind record");
    assert!(pause < rewind, "records={records:?}");
}

#[tokio::test]
async fn cancelled_user_steering_keeps_interruption_pause_durable() {
    let workspace = IsolatedWorkspace::new("cancel-user-interruption-resume");
    let mut cfg = workspace.config();
    cfg.gates.verification = VerificationMode::Disabled;
    cfg.memory.finalize = false;
    cfg.memory.suggest_next_prompt = false;
    let cancellation = TurnCancellation::new();
    let provider = std::sync::Arc::new(CancelThenCompleteProvider {
        cancellation: cancellation.clone(),
    });
    let mut agent = Agent::new(provider, cfg).unwrap();
    agent.restore_plan(vec![PlanStep {
        title: "finish safely".into(),
        status: PlanStatus::Active,
    }]);
    agent.restore_plan_drive_with_policy(true, true, 0, Vec::new());

    let outcome = agent
        .run_turn_cancellable(
            "steer the interrupted implementation",
            &mut NullUi,
            cancellation,
        )
        .await
        .expect("cancellation should settle to a typed outcome");

    assert_eq!(outcome.status, TurnStatus::Cancelled);
    assert!(
        agent.plan_drive_paused(),
        "failed steering must not unlock autonomous plan drive"
    );
    assert!(
        agent.prepare_plan_drive_for_turn(DriveKind::User).unwrap(),
        "the retained pause must remain interruption-resumable"
    );
}

#[tokio::test]
async fn cancellation_reopens_plan_step_whose_workspace_effect_was_rolled_back() {
    let workspace = IsolatedWorkspace::new("cancel-plan-completion-rollback");
    let changed = workspace.path("cancelled-plan-work.txt");
    let mut cfg = workspace.config();
    cfg.gates.verification = VerificationMode::Disabled;
    cfg.memory.finalize = false;
    cfg.memory.suggest_next_prompt = false;
    let update_plan = completion(
        vec![Content::ToolCall {
            id: "plan-done".into(),
            name: "update_plan".into(),
            arguments: serde_json::json!({
                "steps": [
                    {
                        "title": "Implement the parser fix",
                        "status": "done"
                    },
                    {
                        "title": "Document the parser behavior",
                        "status": "pending"
                    }
                ]
            })
            .to_string(),
        }],
        1,
        1,
    );
    let (mut agent, _) = scripted_agent(
        vec![
            ProviderStep::Completion(write_completion(&changed.to_string_lossy())),
            ProviderStep::Completion(update_plan),
            ProviderStep::DelayedCompletion(
                std::time::Duration::from_secs(5),
                completion(vec![Content::Text("too late".into())], 1, 1),
            ),
        ],
        cfg,
    );
    agent.restore_plan(vec![
        PlanStep {
            title: "Implement the parser fix".into(),
            status: PlanStatus::Active,
        },
        PlanStep {
            title: "Document the parser behavior".into(),
            status: PlanStatus::Pending,
        },
    ]);
    let cancellation = TurnCancellation::new();
    let mut ui = CancelAfterToolCountUi {
        cancellation: cancellation.clone(),
        remaining: 2,
    };

    let outcome = agent
        .run_turn_cancellable("continue the implementation", &mut ui, cancellation)
        .await
        .expect("cancellation cleanup should produce a typed outcome");

    assert_eq!(outcome.status, TurnStatus::Cancelled);
    assert!(
        !changed.exists(),
        "the cancelled turn's workspace effect was not rolled back"
    );
    assert_eq!(
        agent.current_plan()[0].status,
        PlanStatus::Active,
        "the rolled-back implementation must not remain durably complete"
    );
}

#[tokio::test]
async fn cancellation_rolls_back_after_the_bounded_checkpoint_stack_is_full() {
    let workspace = IsolatedWorkspace::new("cancel-full-checkpoint-stack");
    let mut cfg = workspace.config();
    cfg.gates.verification = VerificationMode::Disabled;
    cfg.memory.finalize = false;
    cfg.memory.suggest_next_prompt = false;
    let mut agent = agent(vec![write_completion("cancelled-at-cap.txt")], cfg);
    agent.workspace.checkpoints = (0..MAX_CHECKPOINTS)
        .map(|index| format!("{index:040x}"))
        .collect();
    let checkpoints_before = agent.checkpoint_refs().to_vec();
    let cancellation = TurnCancellation::new();
    let mut ui = CancelAfterToolUi {
        cancellation: cancellation.clone(),
        fired: false,
    };

    let outcome = agent
        .run_turn_cancellable("write cancelled-at-cap.txt", &mut ui, cancellation)
        .await
        .expect("cancellation cleanup should produce a typed outcome");

    assert_eq!(outcome.status, TurnStatus::Cancelled);
    assert!(
        !workspace.path("cancelled-at-cap.txt").exists(),
        "the new checkpoint must be identified and restored even when retention keeps the count flat"
    );
    assert_eq!(
        agent.checkpoint_refs(),
        checkpoints_before,
        "cancellation must restore the exact bounded pre-turn undo stack"
    );
}

#[tokio::test]
async fn cancellation_that_wins_outer_race_overrides_normal_body_settlement() {
    let workspace = IsolatedWorkspace::new("cancel-normal-settlement");
    let mut cfg = workspace.config();
    cfg.routing.tool_mode = ToolMode::ChatOnly;
    cfg.gates.verification = VerificationMode::Disabled;
    cfg.memory.finalize = false;
    cfg.memory.suggest_next_prompt = false;
    let cancellation = TurnCancellation::new();
    let probe = std::sync::Arc::new(TurnLifecycleProbe::default());
    let mut builder = hi_agent_lifecycle::ExtensionRegistryBuilder::new();
    builder.turn_lifecycle_contributor(probe.clone());
    let mut agent = Agent::new(
        std::sync::Arc::new(CancelThenCompleteProvider {
            cancellation: cancellation.clone(),
        }),
        cfg,
    )
    .unwrap()
    .with_extension_registry(builder.build());
    let history_len = agent.messages().len();

    let outcome = agent
        .run_turn_cancellable("cancel while completing", &mut NullUi, cancellation)
        .await
        .expect("cancellation cleanup should produce a typed outcome");

    assert_eq!(outcome.status, TurnStatus::Cancelled);
    assert_eq!(agent.messages().len(), history_len);
    assert_eq!(probe.done.load(std::sync::atomic::Ordering::SeqCst), 0);
    assert_eq!(
        *probe.aborts.lock().unwrap(),
        vec![hi_agent_lifecycle::TurnAbortReason::Interrupted]
    );
}

#[tokio::test]
async fn cancellable_turn_rewinds_prompt_injected_state_and_durable_decisions() {
    let workspace = IsolatedWorkspace::new("cancel-side-state");
    let mut cfg = workspace.config();
    cfg.gates.verification = VerificationMode::Disabled;
    cfg.execution = crate::ExecutionMode::Durable;
    cfg.memory.finalize = false;
    cfg.memory.suggest_next_prompt = false;
    let record = completion(
        vec![Content::ToolCall {
            id: "decision".into(),
            name: "record_decision".into(),
            arguments: serde_json::json!({
                "summary": "abandoned choice",
                "rationale": "this turn will be cancelled",
                "files": []
            })
            .to_string(),
        }],
        1,
        1,
    );
    let mut agent = agent(
        vec![record, completion(vec![Content::Text("done".into())], 1, 1)],
        cfg,
    );
    let persisted = std::sync::Arc::new(Mutex::new(DecisionLog::default()));
    agent.set_session(Box::new(DecisionReplacementSession {
        decisions: persisted.clone(),
    }));
    let cancellation = TurnCancellation::new();
    let mut ui = CancelAfterToolUi {
        cancellation: cancellation.clone(),
        fired: false,
    };

    let outcome = agent
        .run_turn_cancellable("record then cancel", &mut ui, cancellation)
        .await
        .expect("cancellation should settle into a typed outcome");

    assert_eq!(outcome.status, TurnStatus::Cancelled);
    assert!(agent.decisions().is_empty());
    assert!(
        persisted.lock().unwrap().is_empty(),
        "the abandoned decision remained in durable session state"
    );
}

#[tokio::test]
async fn cancellation_during_late_suggestion_does_not_publish_a_normal_outcome_or_finding() {
    let workspace = IsolatedWorkspace::new("cancel-late-suggestion-diagnostics");
    let mut cfg = workspace.config();
    cfg.gates.verification = VerificationMode::Disabled;
    cfg.memory.finalize = false;
    cfg.memory.suggest_next_prompt = true;
    cfg.loop_limits.max_steps = 1;
    let cancellation = TurnCancellation::new();
    let provider = CancelDuringSuggestionProvider {
        cancellation: cancellation.clone(),
        calls: std::sync::atomic::AtomicUsize::new(0),
    };
    let mut agent = Agent::new(std::sync::Arc::new(provider), cfg).unwrap();
    let recorded = std::sync::Arc::new(Mutex::new(Vec::new()));
    agent.set_session(Box::new(OutcomeRecordingSession {
        outcomes: recorded.clone(),
    }));

    let outcome = tokio::time::timeout(
        std::time::Duration::from_secs(3),
        agent.run_turn_cancellable("run the first check", &mut NullUi, cancellation),
    )
    .await
    .expect("late suggestion cancellation must stay bounded")
    .expect("cancellation cleanup should produce a typed outcome");

    assert_eq!(outcome.status, TurnStatus::Cancelled);
    assert!(
        recorded.lock().unwrap().is_empty(),
        "the pre-cancellation outcome leaked into durable diagnostics"
    );
    assert!(
        !workspace.path(".hi/state/learning/findings.jsonl").exists(),
        "the cancelled attempt contaminated the findings ledger"
    );
}

#[tokio::test]
async fn hanging_late_suggestion_cannot_keep_a_completed_turn_working() {
    let workspace = IsolatedWorkspace::new("hanging-late-suggestion");
    let mut cfg = workspace.config();
    cfg.routing.tool_mode = ToolMode::ChatOnly;
    cfg.gates.verification = VerificationMode::Disabled;
    cfg.memory.finalize = false;
    cfg.memory.suggest_next_prompt = true;
    let provider = std::sync::Arc::new(HangAfterMainProvider {
        calls: std::sync::atomic::AtomicUsize::new(0),
    });
    let mut agent = Agent::new(provider.clone(), cfg).unwrap();
    agent.side_call_timeout = Some(std::time::Duration::from_millis(25));

    let outcome = tokio::time::timeout(
        std::time::Duration::from_secs(1),
        agent.run_turn("answer briefly", &mut NullUi),
    )
    .await
    .expect("a hanging optional call must be bounded")
    .expect("the completed primary turn should still settle normally");

    assert_eq!(outcome.status, TurnStatus::Completed);
    assert_eq!(
        provider.calls.load(std::sync::atomic::Ordering::SeqCst),
        2,
        "the primary call and exactly one bounded suggestion call should run"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn timeout_during_post_turn_hook_returns_the_committed_outcome() {
    use std::os::unix::fs::PermissionsExt as _;

    let workspace = IsolatedWorkspace::new("timeout-post-turn-committed");
    let hook_dir = workspace.path(".hi/hooks");
    std::fs::create_dir_all(&hook_dir).unwrap();
    let hook = hook_dir.join("post-turn");
    std::fs::write(&hook, "#!/bin/sh\nsleep 5\n").unwrap();
    std::fs::set_permissions(&hook, std::fs::Permissions::from_mode(0o755)).unwrap();
    let mut cfg = workspace.config();
    cfg.routing.tool_mode = ToolMode::ChatOnly;
    cfg.gates.verification = VerificationMode::Disabled;
    cfg.memory.finalize = false;
    cfg.memory.suggest_next_prompt = false;
    cfg.loop_limits.turn_timeout = None;
    let mut agent = agent(
        vec![completion(vec![Content::Text("done".into())], 1, 1)],
        cfg,
    );
    // Keep the deadline focused on terminal finalization, not the one-time
    // workspace scan that can be slow under a concurrently loaded test suite.
    agent
        .runtime
        .ensure_ledger_scan_complete_async()
        .await
        .unwrap();
    agent.config.loop_limits.turn_timeout = Some(std::time::Duration::from_secs(1));

    let outcome = tokio::time::timeout(
        std::time::Duration::from_secs(3),
        agent.run_turn("answer briefly", &mut NullUi),
    )
    .await
    .expect("deadline must cancel the slow post-turn hook")
    .expect("the body committed before its best-effort hook timed out");

    assert_eq!(outcome.status, TurnStatus::Completed);
}

#[cfg(unix)]
#[tokio::test]
async fn timeout_during_error_post_turn_hook_preserves_the_body_error() {
    use std::os::unix::fs::PermissionsExt as _;

    let workspace = IsolatedWorkspace::new("timeout-error-post-turn");
    let hook_dir = workspace.path(".hi/hooks");
    std::fs::create_dir_all(&hook_dir).unwrap();
    let hook = hook_dir.join("post-turn");
    std::fs::write(&hook, "#!/bin/sh\nsleep 5\n").unwrap();
    std::fs::set_permissions(&hook, std::fs::Permissions::from_mode(0o755)).unwrap();
    let mut cfg = workspace.config();
    cfg.routing.tool_mode = ToolMode::ChatOnly;
    cfg.gates.verification = VerificationMode::Disabled;
    cfg.memory.finalize = false;
    cfg.memory.suggest_next_prompt = false;
    cfg.loop_limits.turn_timeout = None;
    let (mut agent, _) = scripted_agent(vec![ProviderStep::Error(ProviderErrorKind::Auth)], cfg);
    agent
        .runtime
        .ensure_ledger_scan_complete_async()
        .await
        .unwrap();
    agent.config.loop_limits.turn_timeout = Some(std::time::Duration::from_secs(1));

    let error = tokio::time::timeout(
        std::time::Duration::from_secs(3),
        agent.run_turn("fail once", &mut NullUi),
    )
    .await
    .expect("deadline must cancel the slow error hook")
    .unwrap_err();

    assert!(
        !error.to_string().contains("turn deadline exceeded"),
        "the already-settled provider error was misreported as rollback: {error:#}"
    );
}

#[tokio::test]
async fn cancelled_turn_dispatches_abort_instead_of_done_lifecycle_hook() {
    let workspace = IsolatedWorkspace::new("cancelled-turn-lifecycle");
    let mut cfg = workspace.config();
    cfg.routing.tool_mode = ToolMode::ChatOnly;
    let probe = std::sync::Arc::new(TurnLifecycleProbe::default());
    let mut builder = hi_agent_lifecycle::ExtensionRegistryBuilder::new();
    builder.turn_lifecycle_contributor(probe.clone());
    let mut agent = agent(Vec::new(), cfg).with_extension_registry(builder.build());
    let cancellation = TurnCancellation::new();
    let mut ui = CancelOnRunStartUi {
        cancellation: cancellation.clone(),
        fired: false,
    };

    let outcome = agent
        .run_turn_cancellable("cancel this turn", &mut ui, cancellation)
        .await
        .expect("cancellation should settle into a typed outcome");

    assert_eq!(outcome.status, TurnStatus::Cancelled);
    assert_eq!(probe.starts.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(
        probe.done.load(std::sync::atomic::Ordering::SeqCst),
        0,
        "cancelled turns are not completed turns"
    );
    assert_eq!(
        *probe.aborts.lock().unwrap(),
        vec![hi_agent_lifecycle::TurnAbortReason::Interrupted]
    );
}

#[tokio::test]
async fn turn_limit_settles_as_completed_without_reporting_a_user_interrupt() {
    let workspace = IsolatedWorkspace::new("turn-limit-lifecycle");
    let mut cfg = workspace.config();
    cfg.max_turns = Some(0);
    let probe = std::sync::Arc::new(TurnLifecycleProbe::default());
    let mut builder = hi_agent_lifecycle::ExtensionRegistryBuilder::new();
    builder.turn_lifecycle_contributor(probe.clone());
    let mut agent = agent(Vec::new(), cfg).with_extension_registry(builder.build());

    let outcome = agent
        .run_turn("must not start", &mut NullUi)
        .await
        .expect("turn limit is a typed bounded-settlement outcome");

    assert_eq!(outcome.status, TurnStatus::Completed);
    assert_eq!(outcome.stop_reason, TurnStopReason::TurnLimit);
    assert_eq!(agent.turn_phase(), TurnPhase::Done);
    assert_eq!(probe.starts.load(std::sync::atomic::Ordering::SeqCst), 0);
    assert_eq!(probe.done.load(std::sync::atomic::Ordering::SeqCst), 0);
    assert!(probe.aborts.lock().unwrap().is_empty());
}

#[tokio::test]
async fn generous_soft_deadline_does_not_disturb_a_normal_turn() {
    let mut cfg = config();
    cfg.loop_limits.turn_soft_deadline = Some(std::time::Duration::from_secs(600));
    let mut agent = agent(
        vec![completion(vec![Content::Text("all done".into())], 1, 1)],
        cfg,
    );

    let outcome = agent
        .run_turn("say hello", &mut NullUi)
        .await
        .expect("turn runs normally");

    assert_ne!(outcome.stop_reason, TurnStopReason::TimeLimit);
    assert_eq!(outcome.status, TurnStatus::Completed);
}

#[test]
fn top_level_error_kind_classifies_usage_vs_infra() {
    use crate::TopLevelErrorKind;
    assert_eq!(
        TopLevelErrorKind::from_anyhow(&anyhow::anyhow!("usage: missing --model")),
        TopLevelErrorKind::Usage
    );
    assert_eq!(
        TopLevelErrorKind::from_anyhow(&anyhow::anyhow!("invalid configuration: bad tool mode")),
        TopLevelErrorKind::Usage
    );
    assert_eq!(
        TopLevelErrorKind::from_anyhow(&anyhow::anyhow!("connection reset by peer")),
        TopLevelErrorKind::Infra
    );
    assert_eq!(TopLevelErrorKind::Usage.exit_code(), 2);
    assert_eq!(TopLevelErrorKind::Infra.exit_code(), 3);
}
