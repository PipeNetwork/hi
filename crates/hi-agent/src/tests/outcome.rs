use super::common::{
    Canned, IsolatedWorkspace, NullUi, ProviderStep, ScriptedProvider, agent, completion, config,
    scripted_agent, write_completion,
};
use super::*;
use hi_ai::{ChatRequest, StreamEvent};
use std::sync::Mutex;

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

#[test]
fn cancelled_turn_reconciles_surviving_workspace_changes() {
    let root = std::env::temp_dir().join(format!("hi-agent-cancel-outcome-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let mut cfg = config();
    cfg.paths.workspace_root = root.clone();
    cfg.paths.state_root = root.join(".hi/state");
    let mut agent = agent(Vec::new(), cfg);
    agent.active_turn_ledger_revision = Some(agent.runtime.ledger().revision());
    agent.active_turn_message_start = Some(agent.messages().len());
    std::fs::write(root.join("survived.txt"), "kept\n").unwrap();

    let outcome = agent.finalize_cancelled_turn().unwrap();

    assert_eq!(outcome.status, TurnStatus::Cancelled);
    assert_eq!(outcome.verification, VerificationStatus::Unverified);
    assert_eq!(outcome.changed_files, vec!["survived.txt"]);
    let change = agent.last_file_changes().first().unwrap();
    assert_eq!(change.kind, hi_tools::FileChangeKind::Create);
    assert!(change.before_digest.is_none());
    assert!(change.after_digest.is_some());
    let _ = std::fs::remove_dir_all(root);
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
    // changes must still complete rather than report "incomplete · stalled".
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
async fn explicit_mutation_request_without_changes_is_stalled() {
    let workspace = IsolatedWorkspace::new("outcome-explicit-no-changes");
    let mut agent = agent(
        vec![completion(
            vec![Content::Text(
                "The bug is in parser.rs line 42; an edit there would resolve it.".into(),
            )],
            1,
            1,
        )],
        workspace.config(),
    );

    let outcome = agent
        .run_turn("fix the parser bug", &mut NullUi)
        .await
        .unwrap();

    assert_eq!(outcome.status, TurnStatus::Incomplete);
    assert_eq!(outcome.stop_reason, TurnStopReason::Stalled);
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
    // Empty auto-detect workspace: no Cargo.toml/package.json/etc. A successful
    // mutation must not scream "incomplete · unverified changes" — there were
    // never any checks to run. That is `NotApplicable` / completed.
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
    let done = completion(vec![Content::Text("done".into())], 1, 1);
    let mut agent = agent(vec![write, done], workspace.config());

    let outcome = agent
        .run_turn("create the file", &mut NullUi)
        .await
        .unwrap();
    assert_eq!(outcome.status, TurnStatus::Completed);
    assert_eq!(outcome.verification, VerificationStatus::NotApplicable);
    assert_eq!(
        outcome.stop_reason,
        TurnStopReason::NoApplicableVerification
    );
    assert!(outcome.changed_files.iter().any(|changed| changed == path));
}

#[tokio::test]
async fn disabled_verification_mutation_is_not_applicable() {
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
    assert_eq!(outcome.status, TurnStatus::Completed);
    assert_eq!(outcome.verification, VerificationStatus::NotApplicable);
    assert_eq!(
        outcome.stop_reason,
        TurnStopReason::NoApplicableVerification
    );
}

#[tokio::test]
async fn failed_verify_budget_exhausted_stays_failed_not_not_applicable() {
    // A project with a verify pipeline that fails must still report Failed /
    // Incomplete after the repair budget — not quietly collapse to NotApplicable
    // just because the next outer-loop check returns NotRun (budget spent).
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
    assert_eq!(outcome.status, TurnStatus::Incomplete);
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

#[tokio::test]
async fn independent_review_status_is_emitted_in_turn_outcome() {
    let workspace = IsolatedWorkspace::new("outcome-review");
    let path = "reviewed.rs";
    let write = completion(
        vec![Content::ToolCall {
            id: "write-review".into(),
            name: "write".into(),
            arguments: serde_json::json!({ "path": path, "content": "reviewed\n" }).to_string(),
        }],
        1,
        1,
    );
    let done = completion(vec![Content::Text("done".into())], 1, 1);
    let reviewer = completion(vec![Content::Text("APPROVE".into())], 1, 1);
    let mut cfg = workspace.config();
    cfg.gates.verification = VerificationMode::Explicit(vec![VerifyStage::new("test", "true")]);
    cfg.gates.review = ReviewPolicy::Always;
    // Independent review is only meaningful with the complete turn diff. The
    // shared test default deliberately bypasses checkpoints for older canned
    // tests, so opt back into the production safety contract here.
    cfg.gates.allow_no_checkpoint = false;
    let mut agent = agent(vec![write, done, reviewer], cfg);

    let outcome = agent
        .run_turn("create the reviewed file", &mut NullUi)
        .await
        .unwrap();
    assert_eq!(outcome.status, TurnStatus::Completed);
    assert_eq!(outcome.verification, VerificationStatus::Passed);
    assert_eq!(outcome.review, ReviewStatus::Passed);
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

    assert_eq!(outcome.status, TurnStatus::Incomplete);
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

    assert_eq!(outcome.status, TurnStatus::Incomplete);
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
    std::fs::create_dir_all(root.join("target")).unwrap();
    let huge = std::fs::File::create(root.join("target/cache.bin")).unwrap();
    huge.set_len(512 * 1024 * 1024 + 1).unwrap();
    let write = completion(
        vec![Content::ToolCall {
            id: "write-target-yolo".into(),
            name: "write".into(),
            arguments: serde_json::json!({
                "path": "target/new.rs",
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
        std::fs::read_to_string(root.join("target/new.rs")).unwrap(),
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
    let outcome = agent.finalize_failed_turn();

    assert_eq!(outcome.status, TurnStatus::Failed);
    assert_eq!(
        outcome.verification,
        VerificationStatus::InfrastructureError
    );
    assert!(outcome.changed_files.contains(&"work.rs".to_string()));
    assert!(outcome.changed_files.contains(&"late.rs".to_string()));
    let _ = std::fs::remove_dir_all(root);
}
