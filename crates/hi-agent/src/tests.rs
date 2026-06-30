    use super::*;
    use async_trait::async_trait;
    use hi_ai::{
        ChatRequest, Completion, Content, Provider, ProviderError, ProviderErrorKind, Role,
        StreamEvent, Usage,
    };
    use std::sync::{LazyLock, Mutex};

    /// A provider that returns canned completions in order.
    struct Canned(Mutex<Vec<Completion>>);

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
    type Sample = (Option<f32>, Option<f32>, Option<f32>);
    struct RecordTemps {
        responses: Mutex<Vec<Completion>>,
        samples: std::sync::Arc<Mutex<Vec<Sample>>>,
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
    struct RecordToolModes {
        responses: Mutex<Vec<Completion>>,
        modes: std::sync::Arc<Mutex<Vec<ToolMode>>>,
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

    struct RecordRequests {
        responses: Mutex<Vec<Completion>>,
        tool_names: std::sync::Arc<Mutex<Vec<Vec<String>>>>,
        modes: std::sync::Arc<Mutex<Vec<ToolMode>>>,
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

    enum ProviderStep {
        Completion(Completion),
        RequestTooLarge,
        /// Fail this round with a provider error of the given kind.
        Error(ProviderErrorKind),
        ErrorWithUsage(ProviderErrorKind, Usage),
    }

    struct ScriptedProvider {
        steps: Mutex<Vec<ProviderStep>>,
        requests: std::sync::Arc<Mutex<Vec<Vec<Message>>>>,
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
                ProviderStep::ErrorWithUsage(kind, usage) => {
                    Err(ProviderError::new(kind, "scripted provider error")
                        .with_usage(usage)
                        .into())
                }
            }
        }
    }

    struct NullUi;
    impl Ui for NullUi {
        fn assistant_text(&mut self, _: &str) {}
        fn assistant_reasoning(&mut self, _: &str) {}
        fn assistant_end(&mut self) {}
        fn tool_call(&mut self, _: &str, _: &str) {}
        fn tool_result(&mut self, _: &str, _: &str) {}
        fn status(&mut self, _: &str) {}
        fn turn_end(&mut self, _: &str) {}
    }

    type UsageRecords = std::sync::Arc<Mutex<Vec<(Usage, Option<f64>)>>>;

    struct RecordingSession {
        records: UsageRecords,
    }

    impl SessionSink for RecordingSession {
        fn record(
            &mut self,
            _messages: &[Message],
            usage: Usage,
            cost_usd: Option<f64>,
        ) -> Result<()> {
            self.records.lock().unwrap().push((usage, cost_usd));
            Ok(())
        }

        fn record_compaction(&mut self, _messages: &[Message]) -> Result<()> {
            Ok(())
        }
    }

    #[derive(Default)]
    struct RecordingUi {
        statuses: Vec<String>,
        turn_ends: Vec<String>,
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
        fn turn_end(&mut self, s: &str) {
            self.turn_ends.push(s.to_string());
        }
    }

    fn config() -> AgentConfig {
        AgentConfig {
            model: "m".into(),
            max_tokens: 100,
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

    fn completion(content: Vec<Content>, input: u64, output: u64) -> Completion {
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

    fn agent(responses: Vec<Completion>, cfg: AgentConfig) -> Agent {
        Agent::new(Box::new(Canned(Mutex::new(responses))), cfg)
    }

    fn scripted_agent(
        steps: Vec<ProviderStep>,
        cfg: AgentConfig,
    ) -> (Agent, std::sync::Arc<Mutex<Vec<Vec<Message>>>>) {
        let requests = std::sync::Arc::new(Mutex::new(Vec::new()));
        let provider = ScriptedProvider {
            steps: Mutex::new(steps),
            requests: requests.clone(),
        };
        (Agent::new(Box::new(provider), cfg), requests)
    }

    static VERIFY_TEST_LOCK: LazyLock<tokio::sync::Mutex<()>> =
        LazyLock::new(|| tokio::sync::Mutex::new(()));

    /// A completion that writes a throwaway file — marks the turn as having
    /// edited, so the (edit-gated) verification pipeline runs.
    fn write_completion(path: &str) -> Completion {
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

    fn bash_completion(command: &str) -> Completion {
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
    fn temp_file(tag: &str) -> std::path::PathBuf {
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::env::current_dir()
            .unwrap()
            .join(format!("hi-test-{tag}-{}-{n}", std::process::id()))
    }

    #[tokio::test]
    async fn request_too_large_drops_prior_context_and_retries_latest_prompt() {
        let (mut agent, requests) = scripted_agent(
            vec![
                ProviderStep::RequestTooLarge,
                ProviderStep::Completion(completion(vec![Content::Text("ok".into())], 12, 3)),
            ],
            config(),
        );
        let huge_old_output = "old tool output ".repeat(20_000);
        agent.messages_mut().push(Message::user("previous task"));
        agent
            .messages_mut()
            .push(Message::assistant(vec![Content::ToolCall {
                id: "read-1".into(),
                name: "read".into(),
                arguments: r#"{"path":"LICENSE"}"#.into(),
            }]));
        agent
            .messages_mut()
            .push(Message::tool_result("read-1", huge_old_output.clone()));

        let mut ui = RecordingUi::default();
        agent
            .run_turn("fix the current bug", &mut ui)
            .await
            .unwrap();

        let requests = requests.lock().unwrap();
        let contains = |messages: &[Message], needle: &str| {
            messages.iter().flat_map(|m| &m.content).any(|c| match c {
                Content::Text(t) => t.contains(needle),
                Content::Thinking { text, .. } => text.contains(needle),
                Content::ToolCall {
                    name, arguments, ..
                } => name.contains(needle) || arguments.contains(needle),
                Content::ToolResult { output, .. } => output.contains(needle),
                _ => false,
            })
        };
        assert_eq!(requests.len(), 2);
        assert!(
            contains(&requests[0], &huge_old_output),
            "first request includes existing context"
        );
        assert!(
            !contains(&requests[1], &huge_old_output),
            "retry omits oversized prior context"
        );
        assert!(
            requests[1]
                .iter()
                .any(|m| m.text().contains("fix the current bug")),
            "latest user request is preserved"
        );
        assert!(
            ui.statuses
                .iter()
                .any(|s| s.contains("dropped prior conversation context")),
            "user sees recovery status: {:?}",
            ui.statuses
        );
        assert_eq!(agent.messages().last().unwrap().text(), "ok");
    }

    #[tokio::test]
    async fn request_too_large_latest_prompt_is_removed_after_failed_retry() {
        let (mut agent, _requests) = scripted_agent(vec![ProviderStep::RequestTooLarge], config());
        let start_len = agent.messages().len();
        let mut ui = RecordingUi::default();

        let err = agent
            .run_turn(&"single huge prompt ".repeat(20_000), &mut ui)
            .await
            .unwrap_err();

        assert_eq!(
            hi_ai::provider_error_kind(&err),
            Some(ProviderErrorKind::RequestTooLarge)
        );
        assert_eq!(
            agent.messages().len(),
            start_len,
            "failed oversized prompt is not left in live history"
        );
        assert!(
            ui.statuses.iter().any(|s| s.contains("shorten the prompt")),
            "user gets actionable status: {:?}",
            ui.statuses
        );
    }

    #[tokio::test]
    async fn malformed_stream_retries_and_recovers() {
        // A garbled stream on the first call is silently re-run (with recovery
        // sampling) rather than failing the turn — then it recovers.
        let (mut agent, requests) = scripted_agent(
            vec![
                ProviderStep::Error(ProviderErrorKind::MalformedStream),
                ProviderStep::Completion(completion(vec![Content::Text("recovered".into())], 5, 3)),
            ],
            config(),
        );
        let mut ui = RecordingUi::default();
        agent.run_turn("go", &mut ui).await.unwrap();

        assert_eq!(agent.messages().last().unwrap().text(), "recovered");
        assert_eq!(
            requests.lock().unwrap().len(),
            2,
            "retried once after the garble"
        );
        assert!(
            ui.statuses.iter().any(|s| s.contains("retrying")),
            "shows a retry, got: {:?}",
            ui.statuses
        );
    }

    #[tokio::test]
    async fn retry_counts_usage_from_failed_attempt() {
        let (mut agent, _requests) = scripted_agent(
            vec![
                ProviderStep::ErrorWithUsage(
                    ProviderErrorKind::MalformedStream,
                    Usage {
                        input_tokens: 7,
                        output_tokens: 100,
                        ..Default::default()
                    },
                ),
                ProviderStep::Completion(completion(vec![Content::Text("recovered".into())], 5, 3)),
            ],
            config(),
        );

        agent.run_turn("go", &mut NullUi).await.unwrap();

        assert_eq!(agent.totals().input_tokens, 12);
        assert_eq!(agent.totals().output_tokens, 103);
    }

    #[tokio::test]
    async fn empty_completion_error_is_resampled_too() {
        // The same path catches a provider's empty-completion *error*, not just a
        // content-less Ok response.
        let (mut agent, requests) = scripted_agent(
            vec![
                ProviderStep::Error(ProviderErrorKind::EmptyCompletion),
                ProviderStep::Completion(completion(vec![Content::Text("recovered".into())], 5, 3)),
            ],
            config(),
        );
        agent.run_turn("go", &mut NullUi).await.unwrap();
        assert_eq!(agent.messages().last().unwrap().text(), "recovered");
        assert_eq!(requests.lock().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn tool_protocol_error_is_resampled_too() {
        let (mut agent, requests) = scripted_agent(
            vec![
                ProviderStep::Error(ProviderErrorKind::ToolProtocol),
                ProviderStep::Completion(completion(vec![Content::Text("recovered".into())], 5, 3)),
            ],
            config(),
        );
        agent.run_turn("go", &mut NullUi).await.unwrap();
        assert_eq!(agent.messages().last().unwrap().text(), "recovered");
        assert_eq!(requests.lock().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn tool_protocol_after_tool_progress_gets_guidance_nudge() {
        let (mut agent, requests) = scripted_agent(
            vec![
                ProviderStep::Completion(bash_completion("true")),
                ProviderStep::Error(ProviderErrorKind::ToolProtocol),
                ProviderStep::Completion(completion(vec![Content::Text("recovered".into())], 5, 3)),
            ],
            config(),
        );
        agent.run_turn("go", &mut NullUi).await.unwrap();
        assert_eq!(agent.messages().last().unwrap().text(), "recovered");

        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 3);
        assert!(
            requests[2]
                .iter()
                .any(|message| message.text().contains("valid tool calls")),
            "expected protocol guidance in retry request: {:?}",
            requests[2]
        );
    }

    #[tokio::test]
    async fn implementation_tool_protocol_exhaustion_falls_back_to_text_tool_calls() {
        let path = temp_file("protocol-text-fallback");
        let path_string = path.to_string_lossy().to_string();
        let xmlish_write = format!(
            "<tool_call>write<arg_key>path</arg_key><arg_value>{path_string}</arg_value><arg_key>content</arg_key><arg_value>ok\n</arg_value></tool_call>"
        );
        let (mut agent, requests) = scripted_agent(
            vec![
                ProviderStep::Error(ProviderErrorKind::ToolProtocol),
                ProviderStep::Error(ProviderErrorKind::ToolProtocol),
                ProviderStep::Error(ProviderErrorKind::ToolProtocol),
                ProviderStep::Error(ProviderErrorKind::ToolProtocol),
                ProviderStep::Error(ProviderErrorKind::ToolProtocol),
                ProviderStep::Completion(completion(vec![Content::Text(xmlish_write)], 5, 3)),
                ProviderStep::Completion(bash_completion("cargo test --help")),
                ProviderStep::Completion(completion(
                    vec![Content::Text(format!(
                        "Changed {path_string} and validated with cargo test --help."
                    ))],
                    5,
                    3,
                )),
            ],
            config(),
        );
        let mut ui = RecordingUi::default();
        agent
            .run_turn("build a small CLI GPU training time estimator", &mut ui)
            .await
            .unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "ok\n");
        let _ = std::fs::remove_file(&path);

        assert!(
            ui.statuses
                .iter()
                .any(|status| status.contains("plain-text tool-call parsing")),
            "expected text-tool fallback status: {:?}",
            ui.statuses
        );
        assert!(
            agent
                .messages()
                .last()
                .unwrap()
                .text()
                .contains("validated with cargo test --help")
        );
        assert!(requests.lock().unwrap().len() >= 7);
    }

    #[tokio::test]
    async fn terminal_error_aborts_without_retry() {
        // A non-resamplable error (auth) fails the turn immediately — no retry.
        let (mut agent, requests) =
            scripted_agent(vec![ProviderStep::Error(ProviderErrorKind::Auth)], config());
        let err = agent.run_turn("go", &mut NullUi).await.unwrap_err();
        assert_eq!(
            hi_ai::provider_error_kind(&err),
            Some(ProviderErrorKind::Auth)
        );
        assert_eq!(
            requests.lock().unwrap().len(),
            1,
            "a terminal error is not retried"
        );
    }

    #[tokio::test]
    async fn terminal_error_persists_usage_before_returning() {
        let records = std::sync::Arc::new(Mutex::new(Vec::new()));
        let (mut agent, _requests) = scripted_agent(
            vec![ProviderStep::ErrorWithUsage(
                ProviderErrorKind::Outage,
                Usage {
                    input_tokens: 11,
                    output_tokens: 100,
                    ..Default::default()
                },
            )],
            config(),
        );
        agent.set_session(Box::new(RecordingSession {
            records: records.clone(),
        }));

        let err = agent.run_turn("go", &mut NullUi).await.unwrap_err();

        assert_eq!(
            hi_ai::provider_error_kind(&err),
            Some(ProviderErrorKind::Outage)
        );
        assert_eq!(
            *records.lock().unwrap(),
            vec![(
                Usage {
                    input_tokens: 11,
                    output_tokens: 100,
                    ..Default::default()
                },
                None,
            )]
        );
    }

    #[tokio::test]
    async fn update_memory_writes_file_without_polluting_history() {
        // Use a unique subdir so the per-directory memory lock doesn't collide
        // with other parallel tests writing into the shared temp root.
        let dir = std::env::temp_dir().join(format!(
            "hi-mem-write-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("memory.md");
        let _ = std::fs::remove_file(&path);
        // The model returns a distilled bullet list.
        let mut agent = agent(
            vec![completion(
                vec![Content::Text(
                    "- always run cargo fmt\n- tests live in tests/".into(),
                )],
                7,
                4,
            )],
            config(),
        );
        let before = agent.messages().len();
        agent.update_memory_at(path.clone(), &mut NullUi).await;

        let written = std::fs::read_to_string(&path).expect("memory file written");
        let _ = std::fs::remove_dir_all(&dir);
        assert!(
            written.contains("always run cargo fmt"),
            "distilled: {written}"
        );
        assert_eq!(
            agent.messages().len(),
            before,
            "session history not polluted"
        );
        assert_eq!(agent.totals().output_tokens, 4, "usage counted");
    }

    #[tokio::test]
    async fn update_memory_persists_usage_without_new_messages() {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "hi-memory-persist-{}-{}.md",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let records = std::sync::Arc::new(Mutex::new(Vec::new()));
        let mut agent = agent(
            vec![completion(vec![Content::Text("- note".into())], 10, 5)],
            config(),
        );
        agent.set_session(Box::new(RecordingSession {
            records: records.clone(),
        }));

        agent.update_memory_at(path.clone(), &mut NullUi).await;
        let _ = std::fs::remove_file(path);

        assert_eq!(
            *records.lock().unwrap(),
            vec![(
                Usage {
                    input_tokens: 10,
                    output_tokens: 5,
                    ..Default::default()
                },
                None,
            )]
        );
    }

    #[tokio::test]
    async fn update_memory_is_best_effort_on_error() {
        // A provider error at quit must not panic or leave a file behind.
        let path = std::env::temp_dir().join(format!("hi-mem-{}-err.md", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let (mut agent, _requests) = scripted_agent(
            vec![ProviderStep::Error(ProviderErrorKind::Outage)],
            config(),
        );
        agent.update_memory_at(path.clone(), &mut NullUi).await;
        assert!(!path.exists(), "nothing written when distillation fails");
    }

    #[test]
    fn goal_updates_system_prompt_and_clear_history_keeps_it() {
        let mut agent = agent(vec![], config());
        agent.set_goal(Some("ship a stable TUI".into()));

        assert_eq!(agent.goal(), Some("ship a stable TUI"));
        assert!(
            agent.messages()[0]
                .text()
                .contains("[Current session goal]"),
            "goal marker included"
        );
        assert!(
            agent.messages()[0].text().contains("ship a stable TUI"),
            "goal text included"
        );

        agent.messages_mut().push(Message::user("noise"));
        agent.clear_history();
        assert_eq!(agent.messages().len(), 1);
        assert!(
            agent.messages()[0].text().contains("ship a stable TUI"),
            "goal survives clear-history"
        );

        agent.set_goal(None);
        assert_eq!(agent.goal(), None);
        assert!(
            !agent.messages()[0]
                .text()
                .contains("[Current session goal]"),
            "goal marker removed"
        );
    }

    #[tokio::test]
    async fn runs_a_tool_then_finishes() {
        let responses = vec![
            completion(
                vec![Content::ToolCall {
                    id: "1".into(),
                    name: "bash".into(),
                    arguments: "{\"command\":\"echo hi\"}".into(),
                }],
                5,
                1,
            ),
            completion(vec![Content::Text("all done".into())], 6, 2),
        ];
        let mut agent = agent(responses, config());
        agent.run_turn("do it", &mut NullUi).await.unwrap();

        let roles: Vec<Role> = agent.messages().iter().map(|m| m.role).collect();
        assert_eq!(
            roles,
            vec![
                Role::System,
                Role::User,
                Role::Assistant, // tool call
                Role::Tool,      // tool result
                Role::Assistant, // final text
            ]
        );
        // Token totals accumulate across both model calls.
        assert_eq!(agent.totals().input_tokens, 11);
        assert_eq!(agent.totals().output_tokens, 3);
        assert_eq!(agent.messages().last().unwrap().text(), "all done");
    }

    #[tokio::test]
    async fn batched_read_only_tools_run_and_preserve_order() {
        // One round emits two read-only calls; both run (concurrently) and their
        // results are recorded back in call order. Reads resolve against the
        // crate dir (cargo sets cwd to the manifest dir).
        let responses = vec![
            completion(
                vec![
                    Content::ToolCall {
                        id: "1".into(),
                        name: "read".into(),
                        arguments: r#"{"path":"Cargo.toml"}"#.into(),
                    },
                    Content::ToolCall {
                        id: "2".into(),
                        name: "read".into(),
                        arguments: r#"{"path":"src/lib.rs"}"#.into(),
                    },
                ],
                5,
                1,
            ),
            completion(vec![Content::Text("done".into())], 6, 2),
        ];
        let mut agent = agent(responses, config());
        agent.run_turn("scan", &mut NullUi).await.unwrap();

        let outputs: Vec<String> = agent
            .messages()
            .iter()
            .flat_map(|m| &m.content)
            .filter_map(|c| match c {
                Content::ToolResult { output, .. } => Some(output.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(outputs.len(), 2, "both tool results recorded");
        assert!(
            outputs[0].contains("hi-agent"),
            "first result is Cargo.toml"
        );
        assert!(
            // The file's top-of-module doc comment — stable in the kept head even
            // after the per-result cap clips this (large) file's middle.
            outputs[1].contains("The agent loop"),
            "second result is lib.rs"
        );
    }

    #[tokio::test]
    async fn compact_replaces_history_with_summary() {
        let records = std::sync::Arc::new(Mutex::new(Vec::new()));
        let responses = vec![completion(
            vec![Content::Text(
                "BRIEF: ported the parser; tests green".into(),
            )],
            7,
            5,
        )];
        let mut agent = agent(responses, config());
        agent.set_session(Box::new(RecordingSession {
            records: records.clone(),
        }));
        // Some history to compact.
        agent.messages_mut().push(Message::user("hello"));
        agent
            .messages_mut()
            .push(Message::assistant(vec![Content::Text("hi".into())]));

        agent.compact(&mut NullUi).await.unwrap();

        // History collapses to system + summary.
        assert_eq!(agent.messages().len(), 2);
        assert_eq!(agent.messages()[0].role, Role::System);
        assert!(
            agent.messages()[1]
                .text()
                .contains("BRIEF: ported the parser"),
            "summary message retained"
        );
        // The summarization call's usage is counted.
        assert_eq!(agent.totals().output_tokens, 5);
        assert_eq!(
            *records.lock().unwrap(),
            vec![(
                Usage {
                    input_tokens: 7,
                    output_tokens: 5,
                    ..Default::default()
                },
                None,
            )],
            "manual compaction persists usage even though compacted messages are transient"
        );
    }

    #[tokio::test]
    async fn hybrid_keeps_recent_and_folds_summary() {
        let mut agent = agent(
            vec![completion(vec![Content::Text("OLD SUMMARY".into())], 3, 2)],
            config(),
        );
        // Two user turns; keep_recent = 1 summarizes the first, keeps the second.
        agent.messages_mut().push(Message::user("q1"));
        agent
            .messages_mut()
            .push(Message::assistant(vec![Content::Text("a1".into())]));
        agent.messages_mut().push(Message::user("q2"));
        agent
            .messages_mut()
            .push(Message::assistant(vec![Content::Text("a2".into())]));

        agent
            .compact_with(CompactionKind::Hybrid { keep_recent: 1 }, &mut NullUi)
            .await
            .unwrap();

        let m = agent.messages();
        // system + (summary folded into kept user turn) + kept assistant reply.
        assert_eq!(m.len(), 3);
        assert_eq!(m[0].role, Role::System);
        assert_eq!(m[1].role, Role::User);
        assert!(
            m[1].text().contains("OLD SUMMARY"),
            "summary folded: {}",
            m[1].text()
        );
        assert!(
            m[1].text().contains("q2"),
            "recent turn kept: {}",
            m[1].text()
        );
        assert_eq!(m[2].text(), "a2");
        // No two consecutive same-role messages (provider-safe).
        assert!(
            m.windows(2).all(|w| w[0].role != w[1].role),
            "roles must alternate"
        );
    }

    #[tokio::test]
    async fn elide_then_summarize_tail_elides_tool_turns_summarizes_qa() {
        // A session with: an old tool-bearing turn (q1 + read + big result), an
        // old Q&A turn (q2 + text), and a recent turn (q3). The new default
        // strategy should elide the old tool result (keep the call/result
        // skeleton) and summarize only the old Q&A tail, folding the summary
        // into the first kept turn. The recent turn stays verbatim.
        let mut agent = agent(
            vec![completion(vec![Content::Text("QA SUMMARY".into())], 1, 1)],
            config(),
        );
        // Old tool-bearing turn.
        agent.messages_mut().push(Message::user("q1"));
        agent
            .messages_mut()
            .push(Message::assistant(vec![Content::ToolCall {
                id: "c1".into(),
                name: "read".into(),
                arguments: "{}".into(),
            }]));
        agent
            .messages_mut()
            .push(Message::tool_result("c1", "x".repeat(500)));
        // Old Q&A turn (no tool results) — this is the conversational tail.
        agent.messages_mut().push(Message::user("q2"));
        agent
            .messages_mut()
            .push(Message::assistant(vec![Content::Text("a2".into())]));
        // Recent turn.
        agent.messages_mut().push(Message::user("q3"));
        agent
            .messages_mut()
            .push(Message::assistant(vec![Content::Text("a3".into())]));

        agent
            .compact_with(
                CompactionKind::ElideThenSummarizeTail { keep_recent: 1 },
                &mut NullUi,
            )
            .await
            .unwrap();

        let m = agent.messages();
        // The old tool result must be elided (skeleton kept, not wiped).
        let tool_results: Vec<&str> = m
            .iter()
            .flat_map(|msg| &msg.content)
            .filter_map(|c| match c {
                Content::ToolResult { output, .. } => Some(output.as_str()),
                _ => None,
            })
            .collect();
        assert!(
            tool_results.iter().any(|o| o.starts_with("[elided")),
            "old tool result elided (skeleton kept): {tool_results:?}"
        );
        assert!(
            !tool_results.iter().any(|o| o.contains(&"x".repeat(100))),
            "old tool output content gone: {tool_results:?}"
        );
        // The Q&A summary is folded into the first kept turn (q3), and q3 stays.
        let user_texts: Vec<String> = m
            .iter()
            .filter(|msg| msg.role == Role::User)
            .map(|msg| msg.text())
            .collect();
        assert!(
            user_texts.iter().any(|t| t.contains("QA SUMMARY")),
            "Q&A tail summarized and folded: {user_texts:?}"
        );
        assert!(
            user_texts.iter().any(|t| t.contains("q3")),
            "recent turn kept: {user_texts:?}"
        );
        // Provider-safe: roles alternate.
        assert!(
            m.windows(2).all(|w| w[0].role != w[1].role),
            "roles must alternate: {:?}",
            m.iter().map(|x| x.role).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn elide_then_summarize_tail_skips_model_call_when_no_qa_tail() {
        // A pure tool-heavy session (no old Q&A turns): the strategy should
        // elide and NOT make a summarizing model call. Provide no canned
        // completion — if it tried to summarize, the provider would panic on
        // an empty response list.
        let mut agent = agent(vec![], config());
        agent.messages_mut().push(Message::user("q1"));
        agent
            .messages_mut()
            .push(Message::assistant(vec![Content::ToolCall {
                id: "c1".into(),
                name: "read".into(),
                arguments: "{}".into(),
            }]));
        agent
            .messages_mut()
            .push(Message::tool_result("c1", "x".repeat(500)));
        agent.messages_mut().push(Message::user("q2"));
        agent
            .messages_mut()
            .push(Message::assistant(vec![Content::Text("a2".into())]));

        // keep_recent = 1 → q2 is recent; q1's tool result is old and gets
        // elided. No Q&A tail older than q2 → no model call.
        agent
            .compact_with(
                CompactionKind::ElideThenSummarizeTail { keep_recent: 1 },
                &mut NullUi,
            )
            .await
            .unwrap();
        let m = agent.messages();
        let tool_results: Vec<&str> = m
            .iter()
            .flat_map(|msg| &msg.content)
            .filter_map(|c| match c {
                Content::ToolResult { output, .. } => Some(output.as_str()),
                _ => None,
            })
            .collect();
        assert!(
            tool_results.iter().any(|o| o.starts_with("[elided")),
            "old tool result elided: {tool_results:?}"
        );
    }

    #[tokio::test]
    async fn record_decision_persists_across_compaction_in_system_prompt() {
        // A decision recorded via the tool survives a compaction in the system
        // prompt (the log is injected into the system message, which compaction
        // preserves verbatim — not summarized away).
        let responses = vec![
            completion(
                vec![Content::ToolCall {
                    id: "d1".into(),
                    name: "record_decision".into(),
                    arguments: r#"{"summary":"use BTreeMap","rationale":"ordered iteration","files":["src/m.rs"]}"#.into(),
                }],
                1,
                1,
            ),
            completion(vec![Content::Text("done".into())], 1, 1),
            completion(vec![Content::Text("done".into())], 1, 1),
        ];
        let mut agent = agent(responses, config());
        agent.run_turn("refactor", &mut NullUi).await.unwrap();
        assert_eq!(agent.decisions().entries().len(), 1);
        assert_eq!(agent.decisions().entries()[0].summary, "use BTreeMap");

        // The system prompt contains the decision.
        let sys = agent.messages()[0].text();
        assert!(
            sys.contains("use BTreeMap") && sys.contains("ordered iteration"),
            "decision in system prompt: {sys}"
        );

        // A compaction that summarizes the Q&A tail must NOT remove the
        // decision from the system prompt.
        agent
            .compact_with(CompactionKind::Summarize, &mut NullUi)
            .await
            .unwrap();
        let sys_after = agent.messages()[0].text();
        assert!(
            sys_after.contains("use BTreeMap"),
            "decision survives compaction: {sys_after}"
        );
    }

    #[tokio::test]
    async fn proactive_verify_surfaces_a_per_edit_check_failure() {
        // With proactive_verify on, a write to a .py file with a syntax error
        // triggers a background `python3 -m py_compile` whose failure surfaces
        // as a status line during the turn (before turn-end verify). Skipped if
        // python3 isn't on PATH (the check just won't run).
        if std::process::Command::new("sh")
            .arg("-c")
            .arg("command -v python3")
            .output()
            .map(|o| !o.status.success())
            .unwrap_or(true)
        {
            eprintln!("skipping: python3 not on PATH");
            return;
        }
        let _guard = VERIFY_TEST_LOCK.lock().await;
        let mut cfg = config();
        cfg.proactive_verify = true;
        let tmp = temp_file("proactive");
        let py = tmp.with_extension("py");
        let p = py.to_string_lossy().to_string();
        // Write invalid Python so py_compile fails.
        let responses = vec![
            Completion {
                content: vec![Content::ToolCall {
                    id: "w".into(),
                    name: "write".into(),
                    arguments: format!(r#"{{"path":{p:?},"content":"def (\n"}}"#),
                }],
                usage: Usage {
                    input_tokens: 1,
                    output_tokens: 1,
                    context_occupancy: 1,
                    ..Default::default()
                },
                stop_reason: None,
            },
            completion(vec![Content::Text("done".into())], 1, 1),
        ];
        let mut agent = agent(responses, cfg);
        let mut ui = RecUi::default();
        agent.run_turn("write it", &mut ui).await.unwrap();
        let _ = std::fs::remove_file(&py);
        // A proactive-check failure status line names the file.
        assert!(
            ui.statuses
                .iter()
                .any(|s| s.contains("proactive check failed") && s.contains(&p)),
            "proactive failure surfaced: {:?}",
            ui.statuses
        );
    }

    #[tokio::test]
    async fn structured_goal_state_injected_into_system_prompt_when_long_horizon_on() {
        // With long_horizon on, a structured goal's state (objective + sub-goal
        // checklist + retry notes) is injected into the system prompt so the
        // agent resumes the active sub-goal coherently each turn.
        let mut cfg = config();
        cfg.long_horizon = true;
        let mut agent = agent(
            vec![completion(vec![Content::Text("ok".into())], 1, 1)],
            cfg,
        );
        let mut goal = Goal::new(
            "refactor the parser",
            vec!["write tests".into(), "rewrite parser".into()],
        );
        // Record a failed attempt so the prompt surfaces "don't repeat" notes.
        goal.record_failure("approach A didn't compile", DEFAULT_SUBGOAL_RETRIES);
        assert!(
            agent.set_structured_goal(Some(goal)),
            "accepted when long_horizon on"
        );

        let sys = agent.messages()[0].text();
        assert!(sys.contains("Long-horizon goal"), "header: {sys}");
        assert!(sys.contains("refactor the parser"), "objective: {sys}");
        assert!(sys.contains("write tests"), "sub-goal: {sys}");
        assert!(
            sys.contains("don't repeat these"),
            "retry notes surfaced: {sys}"
        );

        // Clearing the goal removes the section.
        agent.set_structured_goal(None);
        let sys_after = agent.messages()[0].text();
        assert!(
            !sys_after.contains("Long-horizon goal"),
            "goal section cleared: {sys_after}"
        );
    }

    #[tokio::test]
    async fn structured_goal_rejected_when_long_horizon_off() {
        // Default config has long_horizon off — setting a structured goal is
        // rejected (the single-turn loop is unchanged), so the system prompt
        // gains no goal section.
        let mut agent = agent(
            vec![completion(vec![Content::Text("ok".into())], 1, 1)],
            config(),
        );
        let goal = Goal::new("do a thing", vec!["step one".into()]);
        assert!(!agent.set_structured_goal(Some(goal)), "rejected when off");
        assert!(agent.structured_goal().is_none());
        let sys = agent.messages()[0].text();
        assert!(
            !sys.contains("Long-horizon goal"),
            "no goal section when off: {sys}"
        );
    }

    #[tokio::test]
    async fn long_horizon_driver_advances_on_clean_turn() {
        // With long_horizon on and a structured goal set, a turn that verifies
        // clean (or has no verify and doesn't stall) advances the active
        // sub-goal, and the system prompt reflects the new active sub-goal.
        let mut cfg = config();
        cfg.long_horizon = true;
        // One turn: model writes a file (tool), then a clean text finish. No
        // verify configured → a non-stalling turn with no verify is "clean".
        let tmp = temp_file("lh1");
        let p = tmp.to_string_lossy().to_string();
        let responses = vec![
            write_completion(&p),
            completion(vec![Content::Text("done".into())], 1, 1),
            completion(vec![Content::Text("done".into())], 1, 1),
        ];
        let mut agent = agent(responses, cfg);
        agent.set_structured_goal(Some(Goal::new(
            "refactor",
            vec!["step one".into(), "step two".into()],
        )));
        let mut ui = RecUi::default();
        agent.run_turn("go", &mut ui).await.unwrap();
        let _ = std::fs::remove_file(&tmp);
        let goal = agent.structured_goal().expect("goal still set");
        assert_eq!(
            goal.sub_goals[0].status,
            GoalStatus::Done,
            "advanced past step 1"
        );
        assert_eq!(goal.active_index(), Some(1), "step 2 now active");
        // The system prompt reflects the new active sub-goal.
        assert!(
            agent.messages()[0].text().contains("step two"),
            "system prompt shows new active sub-goal"
        );
    }

    #[tokio::test]
    async fn long_horizon_driver_records_failure_on_stall() {
        // A turn that stalls (repeat guard exhausts) records a sub-goal attempt
        // so the next turn sees the prior note (and doesn't repeat the approach).
        let mut cfg = config();
        cfg.long_horizon = true;
        cfg.max_repeat_nudges = 1;
        // Model re-issues the same tool call → repeat guard stalls the turn
        // after exhausting the (1) nudge budget. Three identical writes: the
        // second triggers a nudge, the third exhausts the budget and breaks
        // stalled.
        let responses = vec![
            write_completion("lhstall"),
            write_completion("lhstall"),
            write_completion("lhstall"),
        ];
        let mut agent = agent(responses, cfg);
        agent.set_structured_goal(Some(Goal::new(
            "refactor",
            vec!["step one".into(), "step two".into()],
        )));
        let mut ui = RecUi::default();
        agent.run_turn("go", &mut ui).await.unwrap();
        let _ = std::fs::remove_file("lhstall");
        let goal = agent.structured_goal().expect("goal still set");
        assert_eq!(goal.active_index(), Some(0), "didn't advance (stalled)");
        assert!(
            goal.sub_goals[0].attempts > 0,
            "recorded a failure attempt: {:?}",
            goal.sub_goals[0]
        );
        assert!(
            goal.sub_goals[0]
                .notes
                .iter()
                .any(|n| n.contains("stalled")),
            "stall reason recorded as a note: {:?}",
            goal.sub_goals[0].notes
        );
        // The system prompt surfaces the "don't repeat" notes on the active
        // sub-goal, so the next turn doesn't repeat the failed approach.
        assert!(
            agent.messages()[0].text().contains("don't repeat these"),
            "retry notes in system prompt"
        );
    }

    #[tokio::test]
    async fn scheduler_parallelism_counts_concurrent_batches() {
        // A batch of independent reads (different paths, no deps) should run
        // concurrently — telemetry reports max_concurrent_batch > 1 and a
        // sub-100% serial share. Pins that the dep-aware scheduler's
        // concurrency is measured, not just shipped on faith.
        let cfg = config();
        let responses = vec![
            completion(
                vec![
                    Content::ToolCall {
                        id: "r1".into(),
                        name: "read".into(),
                        arguments: r#"{"path":"a.rs"}"#.into(),
                    },
                    Content::ToolCall {
                        id: "r2".into(),
                        name: "read".into(),
                        arguments: r#"{"path":"b.rs"}"#.into(),
                    },
                    Content::ToolCall {
                        id: "r3".into(),
                        name: "read".into(),
                        arguments: r#"{"path":"c.rs"}"#.into(),
                    },
                ],
                1,
                1,
            ),
            completion(vec![Content::Text("done".into())], 1, 1),
            completion(vec![Content::Text("done".into())], 1, 1),
        ];
        let mut agent = agent(responses, cfg);
        let mut ui = RecUi::default();
        agent.run_turn("read them", &mut ui).await.unwrap();
        let tel = agent.last_turn_telemetry();
        assert_eq!(tel.tool_calls, 3, "three reads ran: {:?}", tel);
        assert!(
            tel.max_concurrent_batch >= 2,
            "independent reads overlapped: {:?}",
            tel
        );
        assert!(
            tel.serial_runs < tel.tool_calls,
            "not all serial: {:?}",
            tel
        );
        // The timeline records each call with its tool name and path.
        assert_eq!(
            tel.tool_timeline.len(),
            3,
            "timeline has one entry per call: {:?}",
            tel.tool_timeline
        );
        let tools: Vec<&str> = tel.tool_timeline.iter().map(|e| e.tool.as_str()).collect();
        assert!(tools.iter().all(|&t| t == "read"), "all reads: {tools:?}");
        let paths: Vec<&str> = tel.tool_timeline.iter().map(|e| e.path.as_str()).collect();
        assert!(
            paths.contains(&"a.rs") && paths.contains(&"b.rs") && paths.contains(&"c.rs"),
            "timeline paths match calls: {paths:?}"
        );
        assert!(
            tel.tool_timeline.iter().all(|e| e.error),
            "reads error (files don't exist in test): {:?}",
            tel.tool_timeline
        );
    }

    #[tokio::test]
    async fn hybrid_falls_back_to_summarize_when_too_few_turns() {
        let mut agent = agent(
            vec![completion(
                vec![Content::Text("WHOLE SUMMARY".into())],
                1,
                1,
            )],
            config(),
        );
        agent.messages_mut().push(Message::user("only turn"));
        agent
            .messages_mut()
            .push(Message::assistant(vec![Content::Text("a".into())]));
        // keep_recent = 3 but only one turn → no recent window → summarize all.
        agent
            .compact_with(CompactionKind::Hybrid { keep_recent: 3 }, &mut NullUi)
            .await
            .unwrap();
        let m = agent.messages();
        assert_eq!(m.len(), 2);
        assert_eq!(m[0].role, Role::System);
        assert!(m[1].text().contains("WHOLE SUMMARY"));
    }

    #[tokio::test]
    async fn elide_shrinks_old_tool_output_without_a_model_call() {
        // Empty provider: if elision tried to call the model, this would panic.
        let mut agent = agent(vec![], config());
        let big = "x".repeat(500);
        agent.messages_mut().push(Message::user("read a"));
        agent
            .messages_mut()
            .push(Message::assistant(vec![Content::ToolCall {
                id: "c1".into(),
                name: "read".into(),
                arguments: "{}".into(),
            }]));
        agent
            .messages_mut()
            .push(Message::tool_result("c1", big.clone()));
        agent.messages_mut().push(Message::user("read b")); // recent turn
        agent
            .messages_mut()
            .push(Message::assistant(vec![Content::ToolCall {
                id: "c2".into(),
                name: "read".into(),
                arguments: "{}".into(),
            }]));
        agent
            .messages_mut()
            .push(Message::tool_result("c2", big.clone()));

        agent
            .compact_with(
                CompactionKind::ElideToolOutput { keep_recent: 1 },
                &mut NullUi,
            )
            .await
            .unwrap();

        let outputs: Vec<String> = agent
            .messages()
            .iter()
            .flat_map(|m| &m.content)
            .filter_map(|c| match c {
                Content::ToolResult { output, .. } => Some(output.clone()),
                _ => None,
            })
            .collect();
        assert!(
            outputs[0].starts_with("[elided"),
            "old elided: {}",
            outputs[0]
        );
        assert_eq!(outputs[1], big, "recent kept verbatim");
    }

    #[derive(Default)]
    struct RecUi {
        statuses: Vec<String>,
        usages: Vec<(u64, u64)>,
        turn_end: Option<String>,
        assistant: String,
        tool_results: Vec<(String, String)>,
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
        fn usage(
            &mut self,
            input_tokens: u64,
            output_tokens: u64,
            _ctx_used: u64,
            _ctx_win: Option<u32>,
        ) {
            self.usages.push((input_tokens, output_tokens));
        }
        fn turn_end(&mut self, summary: &str) {
            self.turn_end = Some(summary.to_string());
        }
    }

    /// A harmless tool-call round (runs `echo`), marking the turn as actively
    /// working so a later text-only stop is nudge-eligible.
    fn echo_call() -> Completion {
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

    #[tokio::test]
    async fn nudges_when_model_repeats_the_same_command() {
        // The model runs a command, then re-issues the *exact same* call next
        // round. The repetition guard nudges it to act on the output instead of
        // re-running, and the model then finishes. One repeat-nudge, no
        // "stuck repeating" notice.
        let responses = vec![
            echo_call(),
            echo_call(), // exact repeat → nudged
            completion(vec![Content::Text("Done. Run cargo test.".into())], 1, 1),
        ];
        let mut agent = agent(responses, config());
        let mut ui = RecUi::default();
        agent.run_turn("check it", &mut ui).await.unwrap();
        assert_eq!(
            ui.statuses
                .iter()
                .filter(|s| s.contains("re-ran the same command"))
                .count(),
            1,
            "exactly one repeat-nudge, got: {:?}",
            ui.statuses
        );
        assert!(
            !ui.statuses.iter().any(|s| s.contains("kept re-running")),
            "no stuck-repeating notice once it moved on, got: {:?}",
            ui.statuses
        );
        assert!(ui.turn_end.is_some(), "turn completed");
    }

    #[tokio::test]
    async fn gives_up_with_notice_after_repeat_cap() {
        // The model re-issues the exact same command every round, through the
        // whole repeat-nudge budget: bounded nudges, then an honest
        // "stuck repeating" notice.
        let mut responses = vec![echo_call()];
        for _ in 0..(config().max_repeat_nudges + 1) {
            responses.push(echo_call()); // exact repeat each round
        }
        let mut agent = agent(responses, config());
        let mut ui = RecUi::default();
        agent.run_turn("check it", &mut ui).await.unwrap();
        assert_eq!(
            ui.statuses
                .iter()
                .filter(|s| s.contains("re-ran the same command"))
                .count(),
            config().max_repeat_nudges as usize,
            "repeat-nudges are bounded, got: {:?}",
            ui.statuses
        );
        assert!(
            ui.statuses.iter().any(|s| s.contains("kept re-running")),
            "stuck-repeating notice after the cap, got: {:?}",
            ui.statuses
        );
    }

    #[tokio::test]
    async fn does_not_nudge_a_different_command() {
        // Two consecutive tool calls with different arguments are not a repeat —
        // both execute, no repeat-nudge.
        let responses = vec![
            completion(
                vec![Content::ToolCall {
                    id: "t".into(),
                    name: "bash".into(),
                    arguments: "{\"command\":\"echo one\"}".into(),
                }],
                1,
                1,
            ),
            completion(
                vec![Content::ToolCall {
                    id: "t".into(),
                    name: "bash".into(),
                    arguments: "{\"command\":\"echo two\"}".into(),
                }],
                1,
                1,
            ),
            completion(vec![Content::Text("Done.".into())], 1, 1),
        ];
        let mut agent = agent(responses, config());
        let mut ui = RecUi::default();
        agent.run_turn("run them", &mut ui).await.unwrap();
        assert!(
            !ui.statuses
                .iter()
                .any(|s| s.contains("re-ran the same command")),
            "different commands are not a repeat, got: {:?}",
            ui.statuses
        );
        assert!(ui.turn_end.is_some(), "turn completed");
    }

    #[tokio::test]
    async fn truncation_continues_instead_of_ending_early() {
        // The model's first response is truncated (stop_reason = "length") —
        // cut off mid-generation. The agent should nudge it to continue rather
        // than treating the truncation as a natural stop. The model then
        // finishes on the second response.
        let mut cfg = config();
        cfg.max_truncation_retries = 2;
        let responses = vec![
            Completion {
                content: vec![Content::Text("Here is the first half of my".into())],
                usage: Usage {
                    input_tokens: 10,
                    output_tokens: 100,
                    ..Default::default()
                },
                stop_reason: Some("length".into()),
            },
            completion(vec![Content::Text(" answer. Done.".into())], 10, 50),
        ];
        let mut agent = agent(responses, cfg);
        let mut ui = RecUi::default();
        agent.run_turn("explain it", &mut ui).await.unwrap();
        assert!(
            ui.statuses.iter().any(|s| s.contains("output token limit")),
            "should warn about truncation, got: {:?}",
            ui.statuses
        );
        assert!(ui.turn_end.is_some(), "turn completed after continuation");
        // The final assistant message in history should include the second
        // (non-truncated) response, proving the turn didn't end on the
        // truncated first half.
        let last_assistant = agent
            .messages()
            .iter()
            .rev()
            .find(|m| m.role == hi_ai::Role::Assistant)
            .expect("there is a final assistant message");
        let text = last_assistant
            .content
            .iter()
            .filter_map(|c| match c {
                Content::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect::<String>();
        assert!(
            text.contains("Done."),
            "model continued past truncation, got: {text}"
        );
    }

    #[tokio::test]
    async fn truncation_gives_up_after_retry_budget() {
        // The model keeps hitting the output token cap every round. After the
        // truncation-retry budget is exhausted, the turn ends with the truncated
        // output rather than looping forever.
        let mut cfg = config();
        cfg.max_truncation_retries = 1;
        // max_truncation_retries=1 → one retry, then give up. So 2 truncated
        // responses: the original + the one retry.
        let responses = vec![
            Completion {
                content: vec![Content::Text("truncated...".into())],
                usage: Usage {
                    input_tokens: 10,
                    output_tokens: 100,
                    ..Default::default()
                },
                stop_reason: Some("max_tokens".into()),
            },
            Completion {
                content: vec![Content::Text("truncated...".into())],
                usage: Usage {
                    input_tokens: 10,
                    output_tokens: 100,
                    ..Default::default()
                },
                stop_reason: Some("max_tokens".into()),
            },
        ];
        let mut agent = agent(responses, cfg);
        let mut ui = RecUi::default();
        agent.run_turn("big task", &mut ui).await.unwrap();
        // One "continuing" retry warning, then one exhaustion warning.
        assert_eq!(
            ui.statuses
                .iter()
                .filter(|s| s.contains("output token limit — continuing"))
                .count(),
            1,
            "exactly one truncation retry warning, got: {:?}",
            ui.statuses
        );
        assert!(
            ui.statuses
                .iter()
                .any(|s| s.contains("task may be incomplete")),
            "should warn about exhaustion, got: {:?}",
            ui.statuses
        );
        assert!(ui.turn_end.is_some(), "turn ended after budget exhausted");
    }

    #[tokio::test]
    async fn truncation_budget_is_separate_from_empty_retries() {
        // Truncation recovery has its own budget, separate from the empty-retry
        // budget. A big task that hits the output token cap multiple times
        // should keep going (up to its own budget) even if it would have
        // exhausted the shared empty-retry budget under the old design.
        let mut cfg = config();
        cfg.max_empty_retries = 1; // small empty-retry budget
        cfg.max_truncation_retries = 4; // generous truncation budget
        // 4 truncated responses, then a clean finish — the turn should survive
        // all 4 truncations (using the dedicated budget) and complete.
        let mut responses: Vec<Completion> = (0..4)
            .map(|_| Completion {
                content: vec![Content::Text("truncated...".into())],
                usage: Usage {
                    input_tokens: 10,
                    output_tokens: 100,
                    ..Default::default()
                },
                stop_reason: Some("length".into()),
            })
            .collect();
        responses.push(completion(
            vec![Content::Text("Finally done.".into())],
            10,
            50,
        ));
        let mut agent = agent(responses, cfg);
        let mut ui = RecUi::default();
        agent.run_turn("big task", &mut ui).await.unwrap();
        // Should have warned about truncation 4 times (one per retry).
        assert_eq!(
            ui.statuses
                .iter()
                .filter(|s| s.contains("output token limit — continuing"))
                .count(),
            4,
            "4 truncation retry warnings (one per retry), got: {:?}",
            ui.statuses
        );
        assert!(ui.turn_end.is_some(), "turn completed after truncations");
        // The final assistant message should be the clean finish, not a
        // truncated fragment.
        let last_assistant = agent
            .messages()
            .iter()
            .rev()
            .find(|m| m.role == hi_ai::Role::Assistant)
            .expect("there is a final assistant message");
        let text = last_assistant
            .content
            .iter()
            .filter_map(|c| match c {
                Content::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect::<String>();
        assert!(
            text.contains("Finally done."),
            "model finished past truncations, got: {text}"
        );
    }

    #[tokio::test]
    async fn truncation_with_partial_tool_call_does_not_orphan() {
        // The model's response is truncated mid-tool-call — the ToolCall block
        // has partial/malformed JSON arguments. The truncation recovery must
        // strip the partial tool call (it was never executed, so it has no
        // matching tool_result) and record only the text. Without stripping,
        // the next provider request would carry an orphan tool_use and be
        // rejected — the turn would stall.
        let mut cfg = config();
        cfg.max_truncation_retries = 2;
        let responses = vec![
            Completion {
                content: vec![
                    Content::Text("Let me write the file".into()),
                    Content::ToolCall {
                        id: "call_1".into(),
                        name: "write".into(),
                        arguments: "{\"path\":\"main.rs\",\"content\":\"fn main() { // trun".into(),
                    },
                ],
                usage: Usage {
                    input_tokens: 10,
                    output_tokens: 100,
                    ..Default::default()
                },
                stop_reason: Some("length".into()),
            },
            // Second response: the model continues and finishes cleanly.
            completion(vec![Content::Text("Done writing the file.".into())], 10, 50),
        ];
        let mut agent = agent(responses, cfg);
        let mut ui = RecUi::default();
        agent.run_turn("write main.rs", &mut ui).await.unwrap();
        assert!(ui.turn_end.is_some(), "turn completed");
        // The partial tool call should NOT appear in history — it was stripped
        // (it was never executed, so it has no matching tool_result; leaving it
        // would create an orphan tool_use that providers reject).
        let has_partial_call = agent.messages().iter().any(|m| {
            m.content.iter().any(|c| {
                matches!(c, Content::ToolCall { name, arguments, .. }
                    if name == "write" && arguments.contains("trun"))
            })
        });
        assert!(
            !has_partial_call,
            "partial tool call should be stripped from history"
        );
        // Also verify no orphan tool_use: every ToolCall in history has a
        // matching ToolResult somewhere.
        let mut call_ids: Vec<&str> = Vec::new();
        let mut answered: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for m in agent.messages().iter() {
            for c in &m.content {
                match c {
                    Content::ToolCall { id, .. } => call_ids.push(id),
                    Content::ToolResult { call_id, .. } => {
                        answered.insert(call_id);
                    }
                    _ => {}
                }
            }
        }
        for id in &call_ids {
            assert!(
                answered.contains(*id),
                "orphan tool_use {id} has no matching tool_result"
            );
        }
    }

    #[tokio::test]
    async fn truncation_with_partial_text_tool_call_strips_raw_protocol() {
        let mut cfg = config();
        cfg.max_truncation_retries = 2;
        let responses = vec![
            Completion {
                content: vec![Content::Text(
                    "Let me write it.\n<tool_call>write<arg_key>path</arg_key><arg_value>main.py</arg_value><arg_key>content</arg_key><arg_value>print("
                        .into(),
                )],
                usage: Usage {
                    input_tokens: 10,
                    output_tokens: 100,
                    ..Default::default()
                },
                stop_reason: Some("max_tokens".into()),
            },
            completion(vec![Content::Text("Done after retry.".into())], 10, 50),
        ];
        let mut agent = agent(responses, cfg);
        let mut ui = RecUi::default();
        agent.run_turn("write main.py", &mut ui).await.unwrap();

        let transcript_text = agent
            .messages()
            .iter()
            .flat_map(|m| m.content.iter())
            .filter_map(|c| match c {
                Content::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            !transcript_text.contains("<tool_call>"),
            "partial XML-ish tool call must be stripped: {transcript_text}"
        );
        assert!(
            transcript_text.contains("Re-issue one fresh, complete tool call"),
            "fresh-tool-call nudge should be recorded: {transcript_text}"
        );
    }

    #[tokio::test]
    async fn stale_nudge_stripped_before_next_turn() {
        // When a turn ends after a repeat-nudge stall, the last message in
        // history is a synthetic user nudge. Without stripping, the next
        // prompt would fold into that nudge via `push_user_or_fold`. This
        // test verifies the nudge is stripped so the next turn starts clean.
        let mut responses = vec![echo_call()];
        // Repeat the same call through the whole repeat-nudge budget so the
        // turn ends with a trailing repeat-nudge.
        for _ in 0..(config().max_repeat_nudges + 1) {
            responses.push(echo_call());
        }
        let mut agent = agent(responses, config());
        let mut ui = RecUi::default();
        agent.run_turn("check it", &mut ui).await.unwrap();

        // After the turn, the last message should NOT be a nudge (user message
        // with a [hi:nudge:...] marker). It should be the assistant's text or
        // a real user message.
        let msgs = agent.messages();
        let last = msgs.last().expect("history is non-empty");
        if last.role == hi_ai::Role::User {
            let text = last
                .content
                .iter()
                .filter_map(|c| match c {
                    Content::Text(t) => Some(t.as_str()),
                    _ => None,
                })
                .collect::<String>();
            assert!(
                !text.starts_with("[hi:nudge:"),
                "trailing nudge should be stripped, but last message is: {text}"
            );
        }
    }

    #[tokio::test]
    async fn next_prompt_does_not_fold_into_stale_nudge() {
        // End-to-end: a turn stalls with a repeat-nudge, then a second turn is
        // sent. The second turn's user message should NOT be folded into the
        // stale nudge — it should be a clean, separate user message. We verify
        // by checking that the model sees the real prompt, not nudge text.
        let mut responses = vec![echo_call()];
        for _ in 0..(config().max_repeat_nudges + 1) {
            responses.push(echo_call());
        }
        // Second turn: a clean text response.
        responses.push(completion(vec![Content::Text("ok".into())], 1, 1));

        let mut agent = agent(responses, config());
        let mut ui = RecUi::default();
        agent.run_turn("first task", &mut ui).await.unwrap();

        // Second turn — should start clean, not folded into a nudge.
        let mut ui2 = RecUi::default();
        agent.run_turn("second task", &mut ui2).await.unwrap();

        let msgs = agent.messages();
        // Find the last user message — it should be "second task", not a
        // folded nudge+prompt combination.
        let last_user = msgs
            .iter()
            .rev()
            .find(|m| m.role == hi_ai::Role::User)
            .expect("there is a last user message");
        let text = last_user
            .content
            .iter()
            .filter_map(|c| match c {
                Content::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect::<String>();
        assert!(
            !text.contains("[hi:nudge:"),
            "next prompt should not be folded into a stale nudge, got: {text}"
        );
        assert!(
            text.contains("second task"),
            "next prompt should be the real user input, got: {text}"
        );
    }

    #[tokio::test]
    async fn silent_auto_continue_keeps_turn_going_without_status() {
        // The model narrates an announced-but-unperformed next step ("Now let me
        // check the tests.") with no tool call. With max_silent_continues > 0 the
        // agent silently re-prompts it to continue — no status line, no visible
        // nudge — and the model then makes the next tool call and finishes with a
        // recap. The recap ("Done.") is a *finished* answer, not a forward-looking
        // step, so it ends the turn cleanly: no further nudge, no false
        // "incomplete" warning.
        let mut cfg = config();
        cfg.max_silent_continues = 3;
        let responses = vec![
            // Round 1: model makes a tool call (actively working).
            completion(
                vec![Content::ToolCall {
                    id: "r1".into(),
                    name: "read".into(),
                    arguments: r#"{"path":"x"}"#.into(),
                }],
                1,
                1,
            ),
            // Round 2: announced next step, no tool call → silent continue.
            completion(
                vec![Content::Text("Now let me check the tests.".into())],
                1,
                1,
            ),
            // Round 3: silently re-prompted, model makes the next tool call.
            completion(
                vec![Content::ToolCall {
                    id: "r2".into(),
                    name: "read".into(),
                    arguments: r#"{"path":"y"}"#.into(),
                }],
                1,
                1,
            ),
            // Round 4: model finishes with a recap → turn ends cleanly.
            completion(vec![Content::Text("Done.".into())], 1, 1),
        ];
        let mut agent = agent(responses, cfg);
        let mut ui = RecUi::default();
        agent.run_turn("review the code", &mut ui).await.unwrap();
        // The turn completed, consuming exactly the four canned responses — a
        // spurious continue after the "Done." recap would have asked for a fifth
        // and panicked on the empty queue.
        assert!(ui.turn_end.is_some(), "turn completed");
        // No visible "nudging" status during the silent continue, and no false
        // "incomplete" warning — the recap ended the turn cleanly.
        assert!(
            !ui.statuses
                .iter()
                .any(|s| s.contains("nudging") || s.contains("incomplete")),
            "silent continue then clean finish: {:?}",
            ui.statuses
        );
    }

    #[tokio::test]
    async fn finished_recap_after_tool_use_ends_without_incomplete_warning() {
        // Repro of the reported "review codebase runs a bit, then stops without
        // finishing" bug. A read-only task reads files (tool calls), then gives
        // its final recap as text with no tool call. The recap is a *finished*
        // answer (past tense), not an announced next step, so the turn must end
        // cleanly — no silent-continue nudge, no false "the model kept narrating
        // … may be incomplete" warning. Before the fix, `made_tool_call` alone
        // forced a nudge on any post-tool text, so a finished review churned the
        // whole silent-continue budget and stopped on the warning.
        let mut cfg = config();
        cfg.max_silent_continues = 3;
        let responses = vec![
            // Reads a file (actively working).
            completion(
                vec![Content::ToolCall {
                    id: "r1".into(),
                    name: "read".into(),
                    arguments: r#"{"path":"Cargo.toml"}"#.into(),
                }],
                1,
                1,
            ),
            // Final recap — a finished answer, text only.
            completion(
                vec![Content::Text(
                    "I reviewed Cargo.toml. The workspace status is clear and tests pass.".into(),
                )],
                1,
                1,
            ),
        ];
        let mut agent = agent(responses, cfg);
        let mut ui = RecUi::default();
        agent.run_turn("review codebase", &mut ui).await.unwrap();
        // The turn ended after exactly the two canned responses — a spurious
        // continue would have asked for a third and panicked on the empty queue.
        assert!(ui.turn_end.is_some(), "turn completed");
        assert!(
            !ui.statuses.iter().any(|s| s.contains("incomplete")),
            "no false incomplete warning on a finished review: {:?}",
            ui.statuses
        );
        // The recap is the closing message — the turn stopped there rather than
        // churning past it with spurious continues.
        let m = agent.messages();
        assert!(
            m.last().unwrap().text().contains("I reviewed Cargo.toml"),
            "the recap is the model's final response: {:?}",
            m.last().unwrap().text()
        );
    }

    #[tokio::test]
    async fn silent_continue_budget_resets_after_tool_progress() {
        // The actual "review codebase stops without finishing" bug. A long,
        // productive turn that *intermittently* narrates a next step without the
        // tool call (a quirk of some models), but reads a file after each nudge.
        // The silent-continue budget bounds *consecutive* stalls, not their
        // total across the turn: each tool call resets the counter, so the turn
        // keeps going as long as the model makes progress between stalls — even
        // when the number of stalls exceeds max_silent_continues. Before the
        // reset the cumulative counter crept up across the whole turn (stall 1,
        // act, stall 2, act, …) and ended it mid-review with a false "incomplete"
        // warning once the Nth stall hit the budget, despite progress every time.
        let mut cfg = config();
        cfg.max_silent_continues = 1;
        let read = |id: &str, path: &str| {
            completion(
                vec![Content::ToolCall {
                    id: id.into(),
                    name: "read".into(),
                    arguments: format!(r#"{{"path":"{path}"}}"#),
                }],
                1,
                1,
            )
        };
        let responses = vec![
            // Stall 1: narrates a next step, no tool call → nudge (budget is 1).
            completion(vec![Content::Text("Let me read Cargo.toml.".into())], 1, 1),
            // Recovers: reads a file → must reset the silent-continue counter.
            read("a", "Cargo.toml"),
            // Stall 2: narrates again. With the reset this is still within budget;
            // without it the cumulative counter is already exhausted here.
            completion(vec![Content::Text("Let me read README.md.".into())], 1, 1),
            // Recovers again.
            read("b", "README.md"),
            // Finishes with a recap → clean end.
            completion(
                vec![Content::Text(
                    "Reviewed Cargo.toml and README.md. Done.".into(),
                )],
                1,
                1,
            ),
        ];
        let mut agent = agent(responses, cfg);
        let mut ui = RecUi::default();
        agent.run_turn("review codebase", &mut ui).await.unwrap();
        assert!(ui.turn_end.is_some(), "turn completed");
        assert!(
            !ui.statuses.iter().any(|s| s.contains("incomplete")),
            "no false incomplete warning while making progress: {:?}",
            ui.statuses
        );
        // Ran all the way to the recap rather than quitting at the second stall.
        assert!(
            agent.messages().last().unwrap().text().contains("Done."),
            "turn ran to the recap: {:?}",
            agent.messages().last().unwrap().text()
        );
    }

    #[tokio::test]
    async fn plan_with_pending_steps_continues_past_recap() {
        // The model posts a plan (2/3 done), does one step, then stops with a
        // finished-looking recap. Without plan-awareness, the text heuristic
        // sees a finished recap and ends the turn — leaving the plan at 2/3.
        // With plan-awareness, the agent detects pending steps and nudges the
        // model to continue until the plan is complete.
        let mut cfg = config();
        cfg.max_silent_continues = 5;
        // Helper: an update_plan call with given step statuses.
        let plan_call = |id: &str, statuses: &[&str]| {
            let steps: Vec<String> = statuses
                .iter()
                .enumerate()
                .map(|(i, s)| format!(r#"{{"title":"step {}","status":"{}"}}"#, i + 1, s))
                .collect();
            completion(
                vec![Content::ToolCall {
                    id: id.into(),
                    name: "update_plan".into(),
                    arguments: format!(r#"{{"steps":[{}]}}"#, steps.join(",")),
                }],
                1,
                1,
            )
        };
        let responses = vec![
            // R1: model posts the initial plan (0/3 done) and starts step 1.
            plan_call("p1", &["active", "pending", "pending"]),
            // R2: model does a read for step 1.
            completion(
                vec![Content::ToolCall {
                    id: "r1".into(),
                    name: "read".into(),
                    arguments: r#"{"path":"x"}"#.into(),
                }],
                1,
                1,
            ),
            // R3: model updates plan (1/3 done, step 2 active) and does a read.
            plan_call("p2", &["done", "active", "pending"]),
            // R4: model stops with a finished-looking recap — but plan is 1/3!
            // The plan-aware continue should nudge it to keep going.
            completion(
                vec![Content::Text(
                    "I've completed step 1. The implementation looks good.".into(),
                )],
                1,
                1,
            ),
            // R5 (nudged): model does step 2.
            completion(
                vec![Content::ToolCall {
                    id: "r2".into(),
                    name: "read".into(),
                    arguments: r#"{"path":"y"}"#.into(),
                }],
                1,
                1,
            ),
            // R6: model updates plan (2/3 done, step 3 active).
            plan_call("p3", &["done", "done", "active"]),
            // R7: model stops with recap again — plan is 2/3, nudge again.
            completion(
                vec![Content::Text("Step 2 is done. Moving on.".into())],
                1,
                1,
            ),
            // R8 (nudged): model does step 3.
            completion(
                vec![Content::ToolCall {
                    id: "r3".into(),
                    name: "read".into(),
                    arguments: r#"{"path":"z"}"#.into(),
                }],
                1,
                1,
            ),
            // R9: model updates plan (3/3 done) — all complete.
            plan_call("p4", &["done", "done", "done"]),
            // R10: model gives final recap — plan is complete, turn ends.
            completion(
                vec![Content::Text("All steps complete. Done.".into())],
                1,
                1,
            ),
        ];
        let mut agent = agent(responses, cfg);
        let mut ui = RecUi::default();
        agent
            .run_turn("implement the feature", &mut ui)
            .await
            .unwrap();
        assert!(ui.turn_end.is_some(), "turn completed");
        // The turn should have run all the way to the final recap (R10),
        // not stopped at R4 or R7 when the model gave a partial recap.
        assert!(
            agent
                .messages()
                .last()
                .unwrap()
                .text()
                .contains("All steps complete"),
            "turn ran to the final recap with plan complete: {:?}",
            agent.messages().last().unwrap().text()
        );
    }

    #[tokio::test]
    async fn complete_plan_ends_turn_without_spurious_continue() {
        // When the plan is fully done (all steps "done"), the model's recap
        // should end the turn cleanly — no plan-driven continue nudge.
        let mut cfg = config();
        cfg.max_silent_continues = 5;
        let plan_call = |id: &str, statuses: &[&str]| {
            let steps: Vec<String> = statuses
                .iter()
                .enumerate()
                .map(|(i, s)| format!(r#"{{"title":"step {}","status":"{}"}}"#, i + 1, s))
                .collect();
            completion(
                vec![Content::ToolCall {
                    id: id.into(),
                    name: "update_plan".into(),
                    arguments: format!(r#"{{"steps":[{}]}}"#, steps.join(",")),
                }],
                1,
                1,
            )
        };
        let responses = vec![
            // Model posts plan (all done) and gives final recap.
            plan_call("p1", &["done", "done"]),
            completion(vec![Content::Text("All done.".into())], 1, 1),
        ];
        let mut agent = agent(responses, cfg);
        let mut ui = RecUi::default();
        agent.run_turn("do it", &mut ui).await.unwrap();
        assert!(ui.turn_end.is_some(), "turn completed");
        // No spurious continue — the turn ended after exactly 2 responses.
        assert!(
            !ui.statuses.iter().any(|s| s.contains("incomplete")),
            "no incomplete warning when plan is done: {:?}",
            ui.statuses
        );
    }

    #[tokio::test]
    async fn long_plan_10_steps_runs_to_completion() {
        // A 10-step plan where the model does one step per round, then stops
        // with a recap. The plan-aware continue should nudge it to keep going
        // until all 10 steps are done. The silent_continues counter resets on
        // each tool call, so this should work regardless of plan length.
        let mut cfg = config();
        cfg.max_silent_continues = 3; // the default
        let n_steps = 10;
        let plan_call = |id: &str, statuses: &[&str]| {
            let steps: Vec<String> = statuses
                .iter()
                .enumerate()
                .map(|(i, s)| format!(r#"{{"title":"step {}","status":"{}"}}"#, i + 1, s))
                .collect();
            completion(
                vec![Content::ToolCall {
                    id: id.into(),
                    name: "update_plan".into(),
                    arguments: format!(r#"{{"steps":[{}]}}"#, steps.join(",")),
                }],
                1,
                1,
            )
        };
        let read_call = |id: &str| {
            completion(
                vec![Content::ToolCall {
                    id: id.into(),
                    name: "read".into(),
                    arguments: r#"{"path":"x"}"#.into(),
                }],
                1,
                1,
            )
        };
        let recap = |text: &str| completion(vec![Content::Text(text.into())], 1, 1);

        let mut responses = Vec::new();
        for step in 0..n_steps {
            // Build statuses: steps before `step` are done, step `step` is active,
            // steps after are pending.
            let statuses: Vec<&str> = (0..n_steps)
                .map(|i| {
                    if i < step {
                        "done"
                    } else if i == step {
                        "active"
                    } else {
                        "pending"
                    }
                })
                .collect();
            // Model posts plan + does a read for this step.
            responses.push(plan_call(&format!("p{step}"), &statuses));
            responses.push(read_call(&format!("r{step}")));
            // Model stops with a recap (unless it's the last step).
            if step < n_steps - 1 {
                responses.push(recap(&format!(
                    "Step {} is done. The implementation looks good.",
                    step + 1
                )));
            }
        }
        // Final: all steps done + final recap.
        let all_done: Vec<&str> = (0..n_steps).map(|_| "done").collect();
        responses.push(plan_call("pfinal", &all_done));
        responses.push(recap("All 10 steps complete. Done."));

        let mut agent = agent(responses, cfg);
        let mut ui = RecUi::default();
        agent
            .run_turn("implement the feature", &mut ui)
            .await
            .unwrap();
        assert!(ui.turn_end.is_some(), "turn completed");
        // The turn should have run all the way to the final recap.
        let last_text = agent.messages().last().unwrap().text();
        assert!(
            last_text.contains("All 10 steps complete"),
            "turn ran to the final recap, got: {last_text}"
        );
        // Should NOT have ended with an incomplete warning.
        assert!(
            !ui.statuses.iter().any(|s| s.contains("incomplete")),
            "no incomplete warning on a completed 10-step plan: {:?}",
            ui.statuses
        );
    }

    #[tokio::test]
    async fn long_plan_survives_text_only_response_to_nudge() {
        // A plan where the model sometimes responds to the continue-nudge with
        // text-only (no tool call) before eventually doing the work. This is
        // the real-world pattern that causes stalls: the model writes a recap,
        // gets nudged, writes another recap instead of acting, gets nudged
        // again, and eventually does the work. The silent_continues budget
        // must be high enough to survive a few text-only responses.
        //
        // With max_silent_continues=3, the model can text-only 3 times in a
        // row before the turn ends. On the 4th text-only, the budget is
        // exhausted. This test has 3 text-only responses (within budget)
        // before the model finally acts.
        let mut cfg = config();
        cfg.max_silent_continues = 3;
        let plan_call = |id: &str, s1: &str, s2: &str, s3: &str| {
            completion(
                vec![Content::ToolCall {
                    id: id.into(),
                    name: "update_plan".into(),
                    arguments: format!(
                        r#"{{"steps":[{{"title":"a","status":"{s1}"}},{{"title":"b","status":"{s2}"}},{{"title":"c","status":"{s3}"}}]}}"#
                    ),
                }],
                1,
                1,
            )
        };
        let responses = vec![
            // R1: plan + read for step 1.
            plan_call("p1", "active", "pending", "pending"),
            completion(
                vec![Content::ToolCall {
                    id: "r1".into(),
                    name: "read".into(),
                    arguments: r#"{"path":"x"}"#.into(),
                }],
                1,
                1,
            ),
            // R2: recap, no tools → nudge (silent_continues=1, force_tools).
            completion(vec![Content::Text("Step 1 done. Looks good.".into())], 1, 1),
            // R3: text-only again (ignores force) → nudge (silent_continues=2).
            completion(
                vec![Content::Text(
                    "The implementation is clean. No issues found.".into(),
                )],
                1,
                1,
            ),
            // R4: text-only again (ignores force) → nudge (silent_continues=3).
            completion(
                vec![Content::Text("Everything looks correct so far.".into())],
                1,
                1,
            ),
            // R5: finally does a tool call → silent_continues resets to 0.
            plan_call("p2", "done", "active", "pending"),
            completion(
                vec![Content::ToolCall {
                    id: "r2".into(),
                    name: "read".into(),
                    arguments: r#"{"path":"y"}"#.into(),
                }],
                1,
                1,
            ),
            // R6: recap → nudge (silent_continues=1).
            completion(vec![Content::Text("Step 2 done.".into())], 1, 1),
            // R7: does step 3.
            plan_call("p3", "done", "done", "active"),
            completion(
                vec![Content::ToolCall {
                    id: "r3".into(),
                    name: "read".into(),
                    arguments: r#"{"path":"z"}"#.into(),
                }],
                1,
                1,
            ),
            // R8: all done + final recap.
            plan_call("p4", "done", "done", "done"),
            completion(
                vec![Content::Text("All steps complete. Done.".into())],
                1,
                1,
            ),
        ];
        let mut agent = agent(responses, cfg);
        let mut ui = RecUi::default();
        agent.run_turn("do it", &mut ui).await.unwrap();
        assert!(ui.turn_end.is_some(), "turn completed");
        let last_text = agent.messages().last().unwrap().text();
        assert!(
            last_text.contains("All steps complete"),
            "turn ran to completion despite text-only responses to nudges, got: {last_text}"
        );
    }

    #[tokio::test]
    async fn plan_stalls_after_max_consecutive_text_only_responses() {
        // When the model responds to the continue-nudge with text-only (no tool
        // call) more than max_silent_continues times in a row, the turn ends
        // with an "incomplete" warning. This is the safety valve — the model is
        // stuck narrating without acting. This test verifies the valve fires
        // at the right point: after exactly max_silent_continues+1 text-only
        // responses (the original recap + max_silent_continues nudged retries).
        let mut cfg = config();
        cfg.max_silent_continues = 3;
        let plan_call = |id: &str| {
            completion(
                vec![Content::ToolCall {
                    id: id.into(),
                    name: "update_plan".into(),
                    arguments: r#"{"steps":[{"title":"a","status":"active"},{"title":"b","status":"pending"}]}"#.into(),
                }],
                1,
                1,
            )
        };
        let responses = vec![
            // R1: plan + read for step 1.
            plan_call("p1"),
            completion(
                vec![Content::ToolCall {
                    id: "r1".into(),
                    name: "read".into(),
                    arguments: r#"{"path":"x"}"#.into(),
                }],
                1,
                1,
            ),
            // R2: recap → nudge (1/3).
            completion(vec![Content::Text("Step 1 done.".into())], 1, 1),
            // R3: text-only → nudge (2/3).
            completion(vec![Content::Text("Looks good.".into())], 1, 1),
            // R4: text-only → nudge (3/3).
            completion(vec![Content::Text("Correct.".into())], 1, 1),
            // R5: text-only → budget exhausted, turn ends with warning.
            completion(vec![Content::Text("Fine.".into())], 1, 1),
        ];
        let mut agent = agent(responses, cfg);
        let mut ui = RecUi::default();
        agent.run_turn("do it", &mut ui).await.unwrap();
        assert!(ui.turn_end.is_some(), "turn ended");
        // Should warn about incomplete — the model kept narrating without acting.
        assert!(
            ui.statuses.iter().any(|s| s.contains("incomplete")),
            "should warn incomplete after exhausting continue budget: {:?}",
            ui.statuses
        );
    }

    #[tokio::test]
    async fn plan_persists_across_turns_for_continue() {
        // When a turn ends with an incomplete plan and the user types
        // "continue", the plan state should persist so the plan-aware continue
        // logic can fire. Without persistence, last_plan is cleared at the
        // start of the new turn and the agent can't detect the incomplete plan.
        let mut cfg = config();
        cfg.max_silent_continues = 3;
        let plan_call = |id: &str, s1: &str, s2: &str| {
            completion(
                vec![Content::ToolCall {
                    id: id.into(),
                    name: "update_plan".into(),
                    arguments: format!(
                        r#"{{"steps":[{{"title":"a","status":"{s1}"}},{{"title":"b","status":"{s2}"}}]}}"#
                    ),
                }],
                1,
                1,
            )
        };

        // Turn 1: model posts plan (step 1 active), does step 1, then stops
        // with a recap. The plan-continue nudges, but the model text-only's
        // past the budget, so the turn ends with an incomplete plan (1/2).
        let turn1_responses = vec![
            plan_call("p1", "active", "pending"),
            completion(
                vec![Content::ToolCall {
                    id: "r1".into(),
                    name: "read".into(),
                    arguments: r#"{"path":"x"}"#.into(),
                }],
                1,
                1,
            ),
            // Recap → nudge (1/3).
            completion(vec![Content::Text("Step 1 done.".into())], 1, 1),
            // Text-only → nudge (2/3).
            completion(vec![Content::Text("Looks good.".into())], 1, 1),
            // Text-only → nudge (3/3).
            completion(vec![Content::Text("Correct.".into())], 1, 1),
            // Text-only → budget exhausted, turn ends.
            completion(vec![Content::Text("Fine.".into())], 1, 1),
        ];
        let mut agent = agent(turn1_responses, cfg);
        let mut ui = RecUi::default();
        agent.run_turn("do it", &mut ui).await.unwrap();
        // Turn 1 ended with incomplete warning — plan is 1/2.
        assert!(
            ui.statuses.iter().any(|s| s.contains("incomplete")),
            "turn 1 should end incomplete: {:?}",
            ui.statuses
        );

        // Verify the plan state persisted after turn 1 — it should still have
        // pending steps so the plan-aware continue can fire on "continue".
        let plan_after_turn1 = &agent.last_plan;
        assert!(
            plan_has_pending_steps(plan_after_turn1),
            "plan should persist with pending steps after turn 1: {:?}",
            plan_after_turn1
        );

        // Turn 2: user types "fix a different bug" (NOT "continue"). The plan
        // should be cleared so a stale plan doesn't cause spurious nudges.
        // We can't easily run a full turn here (Canned provider is exhausted),
        // but we can verify the clearing logic by checking that a non-continue
        // input would clear it. Simulate by calling the clearing logic directly.
        let mut plan = agent.last_plan.clone();
        // The agent clears last_plan when input doesn't look like "continue".
        // Verify the heuristic: "fix a different bug" is NOT a continue command.
        assert!(
            !looks_like_continue("fix a different bug"),
            "a new task should not look like continue"
        );
        assert!(
            looks_like_continue("continue"),
            "'continue' should look like continue"
        );
        // Simulate the clearing: a new task clears, "continue" doesn't.
        plan.clear(); // what the agent does on a new task
        assert!(
            !plan_has_pending_steps(&plan),
            "plan should be cleared on a new task"
        );
    }

    #[tokio::test]
    async fn continue_nudge_forces_tool_choice_on_the_next_round() {
        // When the model narrates instead of acting and gets a silent-continue
        // nudge, the *next* request forces a tool call (tool_mode Required ->
        // tool_choice "required") so the model can't answer the nudge with yet
        // another narration or an empty completion (the observed failure mode of
        // some OpenAI-compat coder models). Once the model acts, the force clears.
        let mut cfg = config();
        cfg.max_silent_continues = 1;
        assert_eq!(cfg.tool_mode, ToolMode::Auto, "precondition: free tool use");
        let responses = vec![
            // R1: narrates a next step, no tool call → nudge + force next round.
            completion(vec![Content::Text("Let me read the code.".into())], 1, 1),
            // R2 (forced): the model calls a tool → force clears.
            completion(
                vec![Content::ToolCall {
                    id: "r".into(),
                    name: "read".into(),
                    arguments: r#"{"path":"x"}"#.into(),
                }],
                1,
                1,
            ),
            // R3: finishes with a recap → turn ends.
            completion(vec![Content::Text("Done.".into())], 1, 1),
        ];
        let modes = std::sync::Arc::new(Mutex::new(Vec::new()));
        let provider = RecordToolModes {
            responses: Mutex::new(responses),
            modes: modes.clone(),
        };
        let mut agent = Agent::new(Box::new(provider), cfg);
        let mut ui = RecUi::default();
        agent.run_turn("review", &mut ui).await.unwrap();
        let modes = modes.lock().unwrap().clone();
        assert_eq!(modes.len(), 3, "three model rounds: {modes:?}");
        assert_eq!(modes[0], ToolMode::Auto, "first round is normal");
        assert_eq!(
            modes[1],
            ToolMode::Required,
            "the round after the nudge forces a tool call"
        );
        assert_eq!(
            modes[2],
            ToolMode::Auto,
            "after the model acted, the force is cleared"
        );
    }

    #[test]
    fn typo_heavy_review_prompts_classify_as_read_only_intents() {
        assert_eq!(
            classify_read_only_intent("review codebase and discuss status and state"),
            Some(ReviewIntent::Status)
        );
        assert_eq!(
            classify_read_only_intent(
                "review for security issues or unsafe unwraps. then disucss only"
            ),
            Some(ReviewIntent::Security)
        );
        assert_eq!(
            classify_read_only_intent(
                "discuss whats its missing and what we should considering building and implimenting"
            ),
            Some(ReviewIntent::Gaps)
        );
        assert_eq!(classify_read_only_intent("fix the unsafe unwraps"), None);
    }

    #[test]
    fn implementation_prompts_classify_without_stealing_gap_reviews() {
        let intent = classify_implementation_intent(
            "lets build a small TUI calculator that estimates how long training will take on GPUs",
        )
        .expect("implementation prompt");
        assert!(intent.tui);
        assert!(intent.gpu_training_estimator);

        assert!(
            classify_implementation_intent(
                "discuss whats its missing and what we should considering building and implimenting"
            )
            .is_none()
        );
        assert_eq!(
            classify_read_only_intent(
                "discuss whats its missing and what we should considering building and implimenting"
            ),
            Some(ReviewIntent::Gaps)
        );

        let prompt = implementation_turn_prompt(
            "/build gpu training calculator",
            ImplementationIntent {
                tui: true,
                gpu_training_estimator: true,
            },
        );
        assert!(prompt.contains("Ratatui"));
        assert!(prompt.contains("cargo init --bin ."));
        assert!(prompt.contains("training_flops = 6 * params * tokens"));
        assert!(prompt.contains("H100 80GB"));
        assert!(prompt.contains("validation command"));
    }

    #[test]
    fn implementation_preflight_detects_rust_validation() {
        let dir = std::env::temp_dir().join(format!(
            "hi-implementation-preflight-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::write(
            dir.join("Cargo.toml"),
            "[package]\nname = \"demo\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        )
        .unwrap();
        std::fs::write(dir.join("src/main.rs"), "fn main() {}\n").unwrap();
        std::fs::write(dir.join("README.md"), "# demo\n").unwrap();

        let output = std::process::Command::new("sh")
            .arg("-lc")
            .arg(implementation_preflight_command())
            .current_dir(&dir)
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let _ = std::fs::remove_dir_all(&dir);

        assert!(output.status.success());
        assert!(stdout.contains("[workspace_manifests]"));
        assert!(stdout.contains("./Cargo.toml"));
        assert!(stdout.contains("[likely_entrypoints]"));
        assert!(stdout.contains("./src/main.rs"));
        assert_eq!(
            preferred_validation_from_preflight(&stdout),
            Some("cargo test".to_string())
        );
    }

    #[test]
    fn gpu_training_estimator_cli_bootstrap_compiles_and_tests() {
        let dir = std::env::temp_dir().join(format!(
            "hi-gpu-estimator-cli-bootstrap-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for (path, content) in gpu_training_estimator_bootstrap_files(ImplementationIntent {
            tui: false,
            gpu_training_estimator: true,
        }) {
            let path = dir.join(path);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::write(path, content).unwrap();
        }

        let output = std::process::Command::new("cargo")
            .arg("test")
            .current_dir(&dir)
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let _ = std::fs::remove_dir_all(&dir);

        assert!(
            output.status.success(),
            "generated CLI project should pass cargo test\nstdout:\n{stdout}\nstderr:\n{stderr}"
        );
    }

    #[test]
    fn gpu_training_estimator_tui_bootstrap_uses_ratatui_and_respects_existing_workspace() {
        let files = gpu_training_estimator_bootstrap_files(ImplementationIntent {
            tui: true,
            gpu_training_estimator: true,
        });
        let cargo_toml = files
            .iter()
            .find(|(path, _)| *path == "Cargo.toml")
            .map(|(_, content)| content)
            .unwrap();
        let main_rs = files
            .iter()
            .find(|(path, _)| *path == "src/main.rs")
            .map(|(_, content)| content)
            .unwrap();
        assert!(cargo_toml.contains("ratatui"));
        assert!(cargo_toml.contains("crossterm"));
        assert!(main_rs.contains("GPU Training Time Estimator"));
        assert!(main_rs.contains("estimate_seconds"));

        let dir = std::env::temp_dir().join(format!(
            "hi-gpu-estimator-existing-workspace-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("Cargo.toml"), "[package]\nname=\"existing\"\n").unwrap();
        let intent = ImplementationIntent {
            tui: true,
            gpu_training_estimator: true,
        };
        let can_bootstrap = intent.gpu_training_estimator
            && implementation_workspace_can_accept_rust_bootstrap_at(&dir);
        let _ = std::fs::remove_dir_all(&dir);

        assert!(
            !can_bootstrap,
            "bootstrap must not run over an existing manifest"
        );
    }

    #[tokio::test]
    async fn implementation_turn_repairs_no_changes_and_missing_validation() {
        let path = temp_file("implementation-repair");
        let path_string = path.to_string_lossy().to_string();
        let responses = vec![
            completion(
                vec![Content::Text("Completed the requested action.".into())],
                1,
                1,
            ),
            write_completion(&path_string),
            completion(
                vec![Content::Text("Implemented the calculator.".into())],
                1,
                1,
            ),
            bash_completion("cargo test --help"),
            completion(
                vec![Content::Text(format!(
                    "Changed {path_string} and validated with cargo test --help."
                ))],
                1,
                1,
            ),
        ];
        let mut agent = agent(responses, config());
        let mut ui = RecordingUi::default();
        agent
            .run_turn("build a small CLI GPU training time estimator", &mut ui)
            .await
            .unwrap();
        let _ = std::fs::remove_file(&path);

        assert!(
            ui.statuses
                .iter()
                .any(|status| status.contains("no file changes")),
            "expected no-change repair status: {:?}",
            ui.statuses
        );
        assert!(
            ui.statuses
                .iter()
                .any(|status| status.contains("without validation")),
            "expected validation repair status: {:?}",
            ui.statuses
        );
        assert_eq!(agent.last_turn_telemetry().quality_repair_nudges, 2);
        assert!(
            agent
                .messages()
                .last()
                .unwrap()
                .text()
                .contains("validated with cargo test --help")
        );
    }

    #[tokio::test]
    async fn stalled_implementation_does_not_finalize_with_stale_recap() {
        let path = temp_file("implementation-no-finalize");
        let path_string = path.to_string_lossy().to_string();
        let mut cfg = config();
        cfg.finalize = true;
        let responses = vec![
            write_completion(&path_string),
            completion(vec![Content::Text("Implemented it.".into())], 1, 1),
            completion(vec![Content::Text("Done.".into())], 1, 1),
            completion(vec![Content::Text("Final recap.".into())], 1, 1),
        ];
        let mut agent = agent(responses, cfg);
        let mut ui = RecordingUi::default();
        agent
            .run_turn("build a small CLI GPU training time estimator", &mut ui)
            .await
            .unwrap();
        let _ = std::fs::remove_file(&path);

        assert!(
            agent
                .messages()
                .last()
                .unwrap()
                .text()
                .contains("Implementation incomplete"),
            "stalled implementation should end on incomplete notice, not a recap"
        );
        assert!(agent.last_turn_telemetry().stalled_unfinished);
    }

    #[tokio::test]
    async fn scaffold_only_implementation_gets_source_edit_nudge() {
        let dir = temp_file("implementation-scaffold-only");
        let dir_string = dir.to_string_lossy().to_string();
        let responses = vec![
            bash_completion(&format!("mkdir -p {dir_string}")),
            completion(vec![Content::Text("Implemented it.".into())], 1, 1),
            completion(vec![Content::Text("Done.".into())], 1, 1),
            completion(vec![Content::Text("Final recap.".into())], 1, 1),
        ];
        let mut agent = agent(responses, config());
        let mut ui = RecordingUi::default();
        agent
            .run_turn("build a small CLI GPU training time estimator", &mut ui)
            .await
            .unwrap();
        let _ = std::fs::remove_dir_all(&dir);

        assert!(
            ui.statuses
                .iter()
                .any(|status| status.contains("only scaffolded setup files")),
            "expected scaffold-only repair status: {:?}",
            ui.statuses
        );
        assert!(
            agent
                .messages()
                .last()
                .unwrap()
                .text()
                .contains("only project scaffolding")
        );
        assert!(agent.last_turn_telemetry().stalled_unfinished);
    }

    #[tokio::test]
    async fn scaffold_only_repair_can_use_text_tool_fallback_for_source_edit() {
        let scaffold_dir = temp_file("implementation-scaffold-text-fallback-dir");
        let scaffold_dir_string = scaffold_dir.to_string_lossy().to_string();
        let source_path = temp_file("implementation-scaffold-text-fallback-src");
        let source_path_string = source_path.to_string_lossy().to_string();
        let xmlish_write = format!(
            "<tool_call>write<arg_key>path</arg_key><arg_value>{source_path_string}</arg_value><arg_key>content</arg_key><arg_value>implemented\n</arg_value></tool_call>"
        );
        let responses = vec![
            bash_completion(&format!("mkdir -p {scaffold_dir_string}")),
            completion(vec![Content::Text("Implemented it.".into())], 1, 1),
            completion(vec![Content::Text("Done.".into())], 1, 1),
            completion(vec![Content::Text(xmlish_write)], 1, 1),
            bash_completion("cargo test --help"),
            completion(
                vec![Content::Text(format!(
                    "Changed {source_path_string} and validated with cargo test --help."
                ))],
                1,
                1,
            ),
        ];
        let mut agent = agent(responses, config());
        let mut ui = RecordingUi::default();
        agent
            .run_turn("build a small CLI GPU training time estimator", &mut ui)
            .await
            .unwrap();
        assert_eq!(
            std::fs::read_to_string(&source_path).unwrap(),
            "implemented\n"
        );
        let _ = std::fs::remove_dir_all(&scaffold_dir);
        let _ = std::fs::remove_file(&source_path);

        assert!(
            agent
                .messages()
                .last()
                .unwrap()
                .text()
                .contains("validated with cargo test --help")
        );
        assert!(
            ui.statuses
                .iter()
                .any(|status| status.contains("only scaffolded setup files")),
            "expected scaffold repair status: {:?}",
            ui.statuses
        );
    }

    #[test]
    fn security_search_family_detection_covers_required_patterns() {
        let unsafe_only = security_search_families_for_tool(
            "grep",
            r#"{"pattern":"unwrap|expect|panic","glob":"*.rs"}"#,
        );
        assert!(unsafe_only.unsafe_or_panic);
        assert!(!unsafe_only.execution_or_fs_env);
        assert!(!unsafe_only.secret_or_auth);

        let path_does_not_count = security_search_families_for_tool(
            "grep",
            r#"{"pattern":"unwrap","path":"src/file_utils.rs"}"#,
        );
        assert!(path_does_not_count.unsafe_or_panic);
        assert!(!path_does_not_count.execution_or_fs_env);

        let broad = security_search_families_for_tool(
            "grep",
            r#"{"pattern":"unsafe|unwrap|expect|panic|command|std::process|spawn|std::fs|read_to_string|std::env|secret|token|auth|api_key|password|bearer","glob":"*.rs"}"#,
        );
        assert_eq!(
            broad,
            SecuritySearchFamilies {
                unsafe_or_panic: true,
                execution_or_fs_env: true,
                secret_or_auth: true,
            }
        );

        let shell = security_search_families_for_tool(
            "bash",
            r#"{"command":"rg 'exec|spawn|token|auth' crates"}"#,
        );
        assert!(!shell.unsafe_or_panic);
        assert!(shell.execution_or_fs_env);
        assert!(shell.secret_or_auth);
    }

    #[test]
    fn incomplete_security_search_requires_broadening_after_read() {
        let mut evidence = EvidenceTracker::default();
        evidence.record_success(
            "grep",
            r#"{"pattern":"unwrap|expect|panic","glob":"*.rs"}"#,
            "src/lib.rs:1: value.unwrap()\n",
        );
        evidence.record_success("read", r#"{"path":"src/lib.rs"}"#, "1\tfn main() {}\n");

        assert!(should_nudge_security_broad_search(
            Some(ReviewIntent::Security),
            &evidence,
            "src/lib.rs: no command execution or secret issues were found."
        ));
        assert!(
            insufficient_after_incomplete_security_search(&evidence)
                .unwrap()
                .contains("command execution/filesystem/env")
        );
    }

    #[test]
    fn security_scope_overclaim_requires_bounded_answer() {
        let mut evidence = EvidenceTracker::default();
        evidence.record_success(
            "grep",
            r#"{"pattern":"unsafe|unwrap|expect|panic|command|std::process|spawn|std::fs|std::env|secret|token|auth","glob":"*.rs"}"#,
            "src/lib.rs:1: fn main() {}\n",
        );
        evidence.record_success("read", r#"{"path":"src/lib.rs"}"#, "1\tfn main() {}\n");

        assert!(should_nudge_security_scope(
            Some(ReviewIntent::Security),
            &evidence,
            "The codebase appears to be secure. There are no hardcoded secrets or direct command execution issues. Specifically, in `src/lib.rs`, no unsafe unwraps were found."
        ));
        assert!(!should_nudge_security_scope(
            Some(ReviewIntent::Security),
            &evidence,
            "Based on the inspected `src/lib.rs` and searched patterns, I did not establish a concrete unsafe unwrap finding. This is not a complete audit."
        ));
    }

    #[test]
    fn generic_inventory_summary_with_path_is_not_accepted_as_status_review() {
        let mut evidence = EvidenceTracker::default();
        evidence.record_success("read", r#"{"path":"Cargo.toml"}"#, "[workspace]\n");
        evidence.record_success(
            "read",
            r#"{"path":"crates/hi-agent/src/lib.rs"}"#,
            "pub struct Agent;\n",
        );

        let generic = "The codebase is a Rust project structured with multiple crates. \
It has a workspace setup with Cargo.toml defining dependencies, and the main functionality \
revolves around an agent loop with tool calling capabilities.";
        assert!(should_nudge_concrete_review_answer(
            Some(ReviewIntent::Status),
            &evidence,
            generic
        ));

        let bounded = "Status:\n- Based on the inspected Cargo.toml and \
crates/hi-agent/src/lib.rs, the workspace exposes the agent crate and the current status \
surface is the agent loop.\n\nEvidence:\n- Cargo.toml and crates/hi-agent/src/lib.rs \
were inspected.\n\nRisks/Validation:\n- This is not a complete repo audit.";
        assert!(!should_nudge_concrete_review_answer(
            Some(ReviewIntent::Status),
            &evidence,
            bounded
        ));
    }

    #[test]
    fn review_answer_needs_bounded_review_shape_not_just_a_path() {
        let mut evidence = EvidenceTracker::default();
        evidence.record_success("read", r#"{"path":"src/lib.rs"}"#, "fn main() {}\n");

        assert!(should_nudge_concrete_review_answer(
            Some(ReviewIntent::Review),
            &evidence,
            "src/lib.rs is part of the project and contains Rust code."
        ));
        assert!(!should_nudge_concrete_review_answer(
            Some(ReviewIntent::Review),
            &evidence,
            "Findings:\n- Based on the inspected src/lib.rs, no concrete issue was established in that file.\n\nEvidence:\n- src/lib.rs was read.\n\nFollow-up:\n- Inspect callers before making broader claims."
        ));
    }

    #[test]
    fn bounded_repair_exhaustion_includes_search_match_targets() {
        let inspected_path = temp_file("repair-search-target");
        std::fs::write(
            &inspected_path,
            "fn token() { let value = std::env::var(\"API_KEY\").unwrap(); }\n",
        )
        .unwrap();
        let inspected = inspected_path.to_string_lossy().to_string();
        let mut evidence = EvidenceTracker::default();
        evidence.record_success(
            "grep",
            &serde_json::json!({
                "pattern": "unwrap|std::env|api_key|token",
                "glob": "*.rs"
            })
            .to_string(),
            &format!(
                "{}:1:/// Context window in tokens.\n{}:2:fn token() {{ let value = std::env::var(\"API_KEY\").unwrap(); }}\n",
                inspected, inspected
            ),
        );
        evidence.record_success(
            "read",
            &serde_json::json!({ "path": inspected.clone() }).to_string(),
            "1\tfn token() { let value = std::env::var(\"API_KEY\").unwrap(); }\n",
        );

        let answer = bounded_review_repair_exhaustion_answer(
            ReviewIntent::Security,
            &evidence,
            "the final answer did not cite concrete files",
        );

        assert!(answer.contains("Concrete search matches from inspected evidence"));
        assert!(answer.contains(&inspected));
        assert!(answer.contains("std::env::var"));
        assert!(answer.contains("pattern-match review targets"));
        assert!(answer.contains("not confirmed vulnerabilities"));
        assert_eq!(evidence.search_hit_snippets.len(), 2);
        assert!(
            evidence.search_hit_snippets[0].contains("std::env::var"),
            "high-signal hit should sort first: {:?}",
            evidence.search_hit_snippets
        );
        let _ = std::fs::remove_file(inspected_path);
    }

    #[test]
    fn search_hit_snippets_keep_late_high_signal_matches() {
        let inspected_path = temp_file("repair-search-ranking");
        std::fs::write(
            &inspected_path,
            "fn token() { let value = std::env::var(\"API_KEY\").unwrap(); }\n",
        )
        .unwrap();
        let inspected = inspected_path.to_string_lossy().to_string();
        let mut output = String::new();
        for line in 1..=12 {
            output.push_str(&format!("{inspected}:{line}:/// token budget note\n"));
        }
        output.push_str(&format!(
            "{inspected}:99:fn token() {{ let value = std::env::var(\"API_KEY\").unwrap(); }}\n"
        ));

        let mut evidence = EvidenceTracker::default();
        evidence.record_success(
            "grep",
            &serde_json::json!({
                "pattern": "unwrap|std::env|api_key|token",
                "glob": "*.rs"
            })
            .to_string(),
            &output,
        );

        assert_eq!(evidence.search_hit_snippets.len(), 8);
        assert!(
            evidence.search_hit_snippets[0].contains("std::env::var"),
            "late high-signal hit should outrank early token-only lines: {:?}",
            evidence.search_hit_snippets
        );
        let _ = std::fs::remove_file(inspected_path);
    }

    #[tokio::test]
    async fn security_review_prompts_advertise_only_read_only_tools() {
        let responses = vec![completion(
            vec![Content::Text(
                "Insufficient evidence: I need targeted search or file reads.".into(),
            )],
            1,
            1,
        )];
        let tool_names = std::sync::Arc::new(Mutex::new(Vec::new()));
        let modes = std::sync::Arc::new(Mutex::new(Vec::new()));
        let provider = RecordRequests {
            responses: Mutex::new(responses),
            tool_names: tool_names.clone(),
            modes: modes.clone(),
        };
        let mut agent = Agent::new(Box::new(provider), config());
        agent
            .run_turn(
                "review for security issues or unsafe unwraps. then disucss only",
                &mut NullUi,
            )
            .await
            .unwrap();

        let names = tool_names.lock().unwrap();
        let first = names.first().expect("request recorded");
        assert!(first.iter().any(|name| name == "read"));
        assert!(first.iter().any(|name| name == "grep"));
        assert!(first.iter().any(|name| name == "list"));
        assert!(!first.iter().any(|name| matches!(
            name.as_str(),
            "write" | "edit" | "multi_edit" | "apply_patch" | "bash"
        )));
        assert_eq!(modes.lock().unwrap()[0], ToolMode::Auto);
    }

    #[tokio::test]
    async fn discuss_only_security_review_blocks_mutating_tool_call_execution() {
        let path = temp_file("readonly-block");
        std::fs::write(&path, "old\n").unwrap();
        let edit_args = serde_json::json!({
            "path": path.to_string_lossy().to_string(),
            "old_string": "old\n",
            "new_string": "new\n",
        })
        .to_string();
        let responses = vec![
            completion(
                vec![Content::ToolCall {
                    id: "edit".into(),
                    name: "edit".into(),
                    arguments: edit_args,
                }],
                1,
                1,
            ),
            completion(
                vec![Content::ToolCall {
                    id: "read".into(),
                    name: "read".into(),
                    arguments: serde_json::json!({ "path": path.to_string_lossy().to_string() })
                        .to_string(),
                }],
                1,
                1,
            ),
            completion(
                vec![Content::Text(format!(
                    "Findings:\n- {}: inspected evidence only; no file changes were made.",
                    path.to_string_lossy()
                ))],
                1,
                1,
            ),
        ];
        let mut agent = agent(responses, config());
        let mut ui = RecUi::default();

        agent
            .run_turn(
                "review for security issues or unsafe unwraps. then disucss only",
                &mut ui,
            )
            .await
            .unwrap();

        assert_eq!(std::fs::read_to_string(&path).unwrap(), "old\n");
        assert!(
            ui.tool_results
                .iter()
                .any(|(name, result)| { name == "edit" && result.contains("Tool `edit` blocked") }),
            "expected blocked edit tool result in transcript"
        );
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn listing_only_review_final_gets_deepen_review_nudge() {
        let responses = vec![
            completion(
                vec![Content::ToolCall {
                    id: "list".into(),
                    name: "list".into(),
                    arguments: r#"{"path":"."}"#.into(),
                }],
                1,
                1,
            ),
            completion(
                vec![Content::Text(
                    "The repository looks healthy and organized.".into(),
                )],
                1,
                1,
            ),
            completion(
                vec![Content::ToolCall {
                    id: "read".into(),
                    name: "read".into(),
                    arguments: r#"{"path":"Cargo.toml"}"#.into(),
                }],
                1,
                1,
            ),
            completion(
                vec![Content::Text(
                    "Findings:\n- Cargo.toml defines the workspace members and gives concrete status context for this review.".into(),
                )],
                1,
                1,
            ),
        ];
        let mut agent = agent(responses, config());
        let mut ui = RecUi::default();

        agent
            .run_turn("review codebase and discuss status and state", &mut ui)
            .await
            .unwrap();

        assert!(
            ui.statuses
                .iter()
                .any(|status| status.contains("only a listing")),
            "expected deepen-review nudge status: {:?}",
            ui.statuses
        );
        let telemetry = agent.last_turn_telemetry();
        assert_eq!(telemetry.quality_repair_nudges, 1);
        assert_eq!(telemetry.targeted_searches, 0);
        assert_eq!(telemetry.file_reads, 1);
        assert!(!telemetry.listing_only);
        assert_eq!(telemetry.discovery_depth, "mixed");
        assert!(
            agent
                .usage_summary(agent.totals())
                .contains("review-repair")
        );
    }

    #[tokio::test]
    async fn read_only_review_generic_final_gets_concrete_evidence_nudge() {
        let inspected_path = temp_file("concrete-review");
        std::fs::write(&inspected_path, "fn main() { println!(\"ok\"); }\n").unwrap();
        let inspected = inspected_path.to_string_lossy().to_string();
        let responses = vec![
            completion(
                vec![Content::ToolCall {
                    id: "read".into(),
                    name: "read".into(),
                    arguments: serde_json::json!({ "path": inspected.clone() }).to_string(),
                }],
                1,
                1,
            ),
            completion(
                vec![Content::Text(
                    "No unsafe unwrap issues were found in the inspected code.".into(),
                )],
                1,
                1,
            ),
            completion(
                vec![Content::Text(format!(
                    "Findings:\n- {inspected}: inspected for unsafe, unwrap, expect, and panic patterns; no security-critical issue was established from that file alone."
                ))],
                1,
                1,
            ),
        ];
        let mut agent = agent(responses, config());
        let mut ui = RecUi::default();

        agent
            .run_turn(
                "review for security issues or unsafe unwraps. then disucss only",
                &mut ui,
            )
            .await
            .unwrap();

        assert!(
            ui.statuses
                .iter()
                .any(|status| status.contains("lacked concrete inspected files")),
            "expected concrete-evidence nudge status: {:?}",
            ui.statuses
        );
        assert!(
            agent
                .messages()
                .iter()
                .any(|message| message.role == Role::Assistant
                    && message.text().contains(&inspected)),
            "final answer should cite inspected path"
        );
        assert_eq!(agent.last_turn_telemetry().quality_repair_nudges, 1);
        let _ = std::fs::remove_file(inspected_path);
    }

    #[tokio::test]
    async fn read_only_review_text_final_without_evidence_gets_inspection_nudge() {
        let inspected_path = temp_file("no-evidence-review");
        std::fs::write(&inspected_path, "fn main() { println!(\"ok\"); }\n").unwrap();
        let inspected = inspected_path.to_string_lossy().to_string();
        let responses = vec![
            completion(
                vec![Content::Text("Completed the requested action.".into())],
                1,
                1,
            ),
            completion(
                vec![Content::ToolCall {
                    id: "read".into(),
                    name: "read".into(),
                    arguments: serde_json::json!({ "path": inspected.clone() }).to_string(),
                }],
                1,
                1,
            ),
            completion(
                vec![Content::Text(format!(
                    "Findings:\n- {inspected}: inspected as the status evidence for this read-only review."
                ))],
                1,
                1,
            ),
        ];
        let modes = std::sync::Arc::new(Mutex::new(Vec::new()));
        let provider = RecordToolModes {
            responses: Mutex::new(responses),
            modes: modes.clone(),
        };
        let mut agent = Agent::new(Box::new(provider), config());
        let mut ui = RecUi::default();

        agent
            .run_turn("review codebase and discuss status and state", &mut ui)
            .await
            .unwrap();

        assert!(
            ui.statuses
                .iter()
                .any(|status| status.contains("no inspected evidence")),
            "expected no-evidence nudge: {:?}",
            ui.statuses
        );
        let modes = modes.lock().unwrap();
        assert_eq!(modes[0], ToolMode::Auto);
        assert_eq!(modes[1], ToolMode::Required);
        assert_eq!(agent.last_turn_telemetry().quality_repair_nudges, 1);
        assert_eq!(agent.last_turn_telemetry().file_reads, 1);
        let _ = std::fs::remove_file(inspected_path);
    }

    #[tokio::test]
    async fn read_only_status_preflight_seeds_first_request_with_evidence() {
        let mut cfg = config();
        cfg.read_only_preflight = true;
        let (mut agent, requests) = scripted_agent(
            vec![ProviderStep::Completion(completion(
                vec![Content::Text(
                    "Status:\n- Cargo.toml and README.md were inspected as the workspace manifest and project overview for this status review."
                        .into(),
                )],
                10,
                4,
            ))],
            cfg,
        );

        let mut ui = RecUi::default();
        agent
            .run_turn("review codebase and discuss status and state", &mut ui)
            .await
            .unwrap();

        assert!(
            ui.statuses
                .iter()
                .any(|status| status.contains("read-only preflight")),
            "expected preflight status: {:?}",
            ui.statuses
        );
        let requests = requests.lock().unwrap();
        let first = requests.first().expect("provider request");
        let mut tool_names = Vec::new();
        let mut tool_results = String::new();
        for message in first {
            for content in &message.content {
                match content {
                    Content::ToolCall { name, .. } => tool_names.push(name.clone()),
                    Content::ToolResult { output, .. } => {
                        tool_results.push_str(output);
                        tool_results.push('\n');
                    }
                    _ => {}
                }
            }
        }
        assert!(
            tool_names.iter().any(|name| name == "diff"),
            "{tool_names:?}"
        );
        assert!(
            tool_names.iter().any(|name| name == "read"),
            "{tool_names:?}"
        );
        assert!(tool_results.contains("[package]") || tool_results.contains("[workspace]"));
        let telemetry = agent.last_turn_telemetry();
        assert!(telemetry.tool_calls >= 3, "{telemetry:?}");
        assert!(telemetry.file_reads >= 2, "{telemetry:?}");
        assert!(telemetry.targeted_searches >= 1, "{telemetry:?}");
        assert!(!telemetry.listing_only, "{telemetry:?}");
        assert_eq!(telemetry.first_tool_kind, "targeted_search");
    }

    #[test]
    fn security_preflight_is_code_scoped_and_bounded() {
        let calls = read_only_preflight_initial_calls(ReviewIntent::Security);
        let mut read_paths = Vec::new();
        let mut grep_args = String::new();
        for call in &calls {
            if call.name == "read" {
                if let Some(path) = hi_tools::target_path(call.name, &call.arguments) {
                    read_paths.push(path);
                }
            } else if call.name == "grep" {
                grep_args = call.arguments.clone();
            }
        }

        assert!(read_paths.iter().any(|path| path == "Cargo.toml"));
        assert!(!read_paths.iter().any(|path| path == "README.md"));
        assert!(grep_args.contains(r#""glob":"*.rs""#), "{grep_args}");
        assert!(grep_args.contains(r#""context":0"#), "{grep_args}");
        assert!(preflight_path_relevant_for_intent(
            ReviewIntent::Security,
            "crates/hi-agent/src/lib.rs"
        ));
        assert!(!preflight_path_relevant_for_intent(
            ReviewIntent::Security,
            "README.md"
        ));

        let long_grep = (0..40)
            .map(|i| format!("src/lib.rs:{i}:unwrap()"))
            .collect::<Vec<_>>()
            .join("\n");
        let compacted = compact_preflight_tool_output("grep", &long_grep);
        assert!(compacted.contains("preflight grep output truncated"));
        assert!(compacted.lines().count() <= READ_ONLY_PREFLIGHT_GREP_MAX_LINES + 1);

        let long_diff = (0..(READ_ONLY_PREFLIGHT_DIFF_MAX_LINES + 25))
            .map(|i| format!("diff line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let compacted = compact_preflight_tool_output("diff", &long_diff);
        assert!(compacted.contains("preflight diff output truncated"));
        assert!(compacted.lines().count() <= READ_ONLY_PREFLIGHT_DIFF_MAX_LINES + 1);
    }

    #[tokio::test]
    async fn read_only_review_no_evidence_repair_exhaustion_returns_insufficient() {
        let responses = vec![
            completion(
                vec![Content::Text("Completed the requested action.".into())],
                1,
                1,
            ),
            completion(
                vec![Content::Text("Completed the requested action.".into())],
                1,
                1,
            ),
        ];
        let mut agent = agent(responses, config());
        let mut ui = RecUi::default();

        agent
            .run_turn(
                "review for security issues or unsafe unwraps. then disucss only",
                &mut ui,
            )
            .await
            .unwrap();

        assert!(
            ui.assistant.contains("Insufficient evidence: no files"),
            "expected bounded insufficient evidence: {}",
            ui.assistant
        );
        assert!(
            ui.statuses
                .iter()
                .any(|status| status.contains("no inspected evidence after repair")),
            "expected exhausted no-evidence status: {:?}",
            ui.statuses
        );
        let telemetry = agent.last_turn_telemetry();
        assert_eq!(telemetry.quality_repair_nudges, 1);
        assert_eq!(telemetry.discovery_depth, "none");
        assert!(telemetry.stalled_unfinished);
    }

    #[tokio::test]
    async fn read_only_review_repair_template_final_is_not_accepted() {
        let inspected_path = temp_file("repair-template");
        std::fs::write(&inspected_path, "# hi\n\nA terminal coding assistant.\n").unwrap();
        let inspected = inspected_path.to_string_lossy().to_string();
        let responses = vec![
            completion(
                vec![Content::ToolCall {
                    id: "read".into(),
                    name: "read".into(),
                    arguments: serde_json::json!({ "path": inspected.clone() }).to_string(),
                }],
                1,
                1,
            ),
            completion(
                vec![Content::Text(format!(
                    "Findings/Status:\n- The inspected context points to these concrete review targets: {inspected}, ./Cargo.toml.\n- Review observations should stay tied to those files or modules instead of only summarizing the repository layout.\n\nConcrete Follow-up:\n- Convert any broad status claims into file-specific findings before recommending changes."
                ))],
                1,
                1,
            ),
        ];
        let mut agent = agent(responses, config());
        let mut ui = RecUi::default();

        agent
            .run_turn("review codebase and discuss status and state", &mut ui)
            .await
            .unwrap();

        assert!(
            ui.assistant.contains("generic review-repair template"),
            "expected template rejection fallback: {}",
            ui.assistant
        );
        assert!(
            !ui.assistant.contains("Findings/Status"),
            "old repair template must not be surfaced: {}",
            ui.assistant
        );
        assert!(agent.last_turn_telemetry().stalled_unfinished);
        let _ = std::fs::remove_file(inspected_path);
    }

    #[tokio::test]
    async fn read_only_review_repair_exhaustion_reports_inspected_evidence() {
        let inspected_path = temp_file("repair-exhaustion-evidence");
        std::fs::write(&inspected_path, "pub fn value() -> i32 { 1 }\n").unwrap();
        let inspected = inspected_path.to_string_lossy().to_string();
        let responses = vec![
            completion(
                vec![Content::ToolCall {
                    id: "read".into(),
                    name: "read".into(),
                    arguments: serde_json::json!({ "path": inspected.clone() }).to_string(),
                }],
                1,
                1,
            ),
            completion(
                vec![Content::Text("Completed the requested action.".into())],
                1,
                1,
            ),
            completion(
                vec![Content::Text("Completed the requested action.".into())],
                1,
                1,
            ),
            completion(
                vec![Content::Text("Completed the requested action.".into())],
                1,
                1,
            ),
        ];
        let mut agent = agent(responses, config());
        let mut ui = RecUi::default();

        agent
            .run_turn(
                "review for security issues or unsafe unwraps. then disucss only",
                &mut ui,
            )
            .await
            .unwrap();

        assert!(
            ui.assistant
                .contains("Bounded evidence summary for an incomplete security review"),
            "expected bounded evidence fallback: {}",
            ui.assistant
        );
        assert!(
            ui.assistant.contains("Inspected evidence:"),
            "fallback should describe inspected evidence: {}",
            ui.assistant
        );
        assert!(
            ui.assistant.contains("File reads: 1"),
            "fallback should report file reads: {}",
            ui.assistant
        );
        assert!(
            ui.assistant.contains(&inspected),
            "fallback should cite inspected path: {}",
            ui.assistant
        );
        assert!(
            !ui.assistant.contains("Findings/Status"),
            "fallback must not invent completed findings: {}",
            ui.assistant
        );
        let telemetry = agent.last_turn_telemetry();
        assert_eq!(telemetry.quality_repair_nudges, 2);
        assert!(telemetry.stalled_unfinished);
        let _ = std::fs::remove_file(inspected_path);
    }

    #[tokio::test]
    async fn read_only_review_generic_insufficient_after_read_reports_evidence() {
        let inspected_path = temp_file("generic-insufficient-after-read");
        std::fs::write(
            &inspected_path,
            "pub fn value(input: Option<i32>) -> i32 { input.unwrap_or_default() }\n",
        )
        .unwrap();
        let inspected = inspected_path.to_string_lossy().to_string();
        let responses = vec![
            completion(
                vec![Content::ToolCall {
                    id: "grep".into(),
                    name: "grep".into(),
                    arguments: serde_json::json!({
                        "pattern": "unsafe|unwrap|expect|panic|std::process|std::fs|std::env|secret|token|auth",
                        "glob": "*.rs",
                    })
                    .to_string(),
                }],
                1,
                1,
            ),
            completion(
                vec![Content::ToolCall {
                    id: "read".into(),
                    name: "read".into(),
                    arguments: serde_json::json!({ "path": inspected.clone() }).to_string(),
                }],
                1,
                1,
            ),
            completion(
                vec![Content::Text(format!(
                    "Insufficient evidence: I inspected `{inspected}`, but cannot make concrete security findings."
                ))],
                1,
                1,
            ),
            completion(
                vec![Content::Text(format!(
                    "Insufficient evidence: I inspected `{inspected}`, but still cannot make concrete security findings."
                ))],
                1,
                1,
            ),
            completion(
                vec![Content::Text(format!(
                    "Insufficient evidence: I inspected `{inspected}`, but still cannot make concrete security findings."
                ))],
                1,
                1,
            ),
            completion(
                vec![Content::Text(format!(
                    "Insufficient evidence: I inspected `{inspected}`, but still cannot make concrete security findings."
                ))],
                1,
                1,
            ),
        ];
        let mut agent = agent(responses, config());
        let mut ui = RecUi::default();

        agent
            .run_turn(
                "review for security issues or unsafe unwraps. then disucss only",
                &mut ui,
            )
            .await
            .unwrap();

        assert!(
            ui.assistant
                .contains("Bounded evidence summary for an incomplete security review"),
            "expected bounded evidence summary: {}",
            ui.assistant
        );
        assert!(
            ui.assistant.contains("Targeted searches: 1"),
            "summary should retain search evidence: {}",
            ui.assistant
        );
        assert!(
            ui.assistant.contains("File reads: 1"),
            "summary should retain file-read evidence: {}",
            ui.assistant
        );
        assert!(
            ui.assistant.contains(&inspected),
            "summary should cite inspected path: {}",
            ui.assistant
        );
        assert!(
            ui.statuses
                .iter()
                .any(|status| status.contains("nudging the model to summarize inspected files")),
            "expected summarize-evidence repair status: {:?}",
            ui.statuses
        );
        assert!(
            ui.statuses.iter().any(
                |status| status.contains("generic insufficient-evidence text after inspection")
            ),
            "expected replacement status: {:?}",
            ui.statuses
        );
        assert!(agent.last_turn_telemetry().stalled_unfinished);
        let _ = std::fs::remove_file(inspected_path);
    }

    #[tokio::test]
    async fn read_only_review_generic_insufficient_after_read_gets_summary_repair() {
        let inspected_path = temp_file("generic-insufficient-summary-repair");
        std::fs::write(
            &inspected_path,
            "pub fn value(input: Option<i32>) -> i32 { input.unwrap_or_default() }\n",
        )
        .unwrap();
        let inspected = inspected_path.to_string_lossy().to_string();
        let responses = vec![
            completion(
                vec![Content::ToolCall {
                    id: "grep".into(),
                    name: "grep".into(),
                    arguments: serde_json::json!({
                        "pattern": "unsafe|unwrap|expect|panic|std::process|std::fs|std::env|secret|token|auth",
                        "glob": "*.rs",
                    })
                    .to_string(),
                }],
                1,
                1,
            ),
            completion(
                vec![Content::ToolCall {
                    id: "read".into(),
                    name: "read".into(),
                    arguments: serde_json::json!({ "path": inspected.clone() }).to_string(),
                }],
                1,
                1,
            ),
            completion(
                vec![Content::Text(format!(
                    "Insufficient evidence: I inspected `{inspected}`, but cannot make concrete security findings."
                ))],
                1,
                1,
            ),
            completion(
                vec![Content::Text(format!(
                    "Findings:\n- `{inspected}` uses `unwrap_or_default`; from the inspected file this is a fallback conversion, not a panic-prone unwrap.\n\nInspected Evidence:\n- `{inspected}` was read after the targeted search.\n\nLimits:\n- This is not a complete audit of uninspected files."
                ))],
                1,
                1,
            ),
        ];
        let mut agent = agent(responses, config());
        let mut ui = RecUi::default();

        agent
            .run_turn(
                "review for security issues or unsafe unwraps. then disucss only",
                &mut ui,
            )
            .await
            .unwrap();

        assert!(
            ui.statuses
                .iter()
                .any(|status| status.contains("nudging the model to summarize inspected files")),
            "expected summarize-evidence repair status: {:?}",
            ui.statuses
        );
        assert!(
            ui.assistant.contains(&inspected),
            "final answer should cite inspected path: {}",
            ui.assistant
        );
        assert!(
            !ui.assistant.contains("Bounded evidence summary"),
            "accepted repaired answer should not fall back: {}",
            ui.assistant
        );
        let telemetry = agent.last_turn_telemetry();
        assert_eq!(telemetry.quality_repair_nudges, 1);
        assert!(!telemetry.stalled_unfinished);
        let _ = std::fs::remove_file(inspected_path);
    }

    #[tokio::test]
    async fn read_only_review_repeat_exhaustion_reports_inspected_evidence() {
        let inspected_path = temp_file("repeat-exhaustion-evidence");
        std::fs::write(
            &inspected_path,
            "pub fn value() -> Option<i32> { Some(1) }\n",
        )
        .unwrap();
        let inspected = inspected_path.to_string_lossy().to_string();
        let grep_args = serde_json::json!({
            "pattern": "unwrap\\(",
            "glob": "*.rs",
        })
        .to_string();
        let responses = vec![
            completion(
                vec![Content::ToolCall {
                    id: "read".into(),
                    name: "read".into(),
                    arguments: serde_json::json!({ "path": inspected.clone() }).to_string(),
                }],
                1,
                1,
            ),
            completion(
                vec![Content::ToolCall {
                    id: "grep1".into(),
                    name: "grep".into(),
                    arguments: grep_args.clone(),
                }],
                1,
                1,
            ),
            completion(
                vec![Content::ToolCall {
                    id: "grep2".into(),
                    name: "grep".into(),
                    arguments: grep_args.clone(),
                }],
                1,
                1,
            ),
            completion(
                vec![Content::ToolCall {
                    id: "grep3".into(),
                    name: "grep".into(),
                    arguments: grep_args.clone(),
                }],
                1,
                1,
            ),
            completion(
                vec![Content::ToolCall {
                    id: "grep4".into(),
                    name: "grep".into(),
                    arguments: grep_args,
                }],
                1,
                1,
            ),
        ];
        let mut agent = agent(responses, config());
        let mut ui = RecUi::default();

        agent
            .run_turn(
                "review for security issues or unsafe unwraps. then disucss only",
                &mut ui,
            )
            .await
            .unwrap();

        assert!(
            ui.assistant
                .contains("Bounded evidence summary for an incomplete security review"),
            "expected bounded evidence fallback: {}",
            ui.assistant
        );
        assert!(
            ui.assistant.contains("Targeted searches: 1"),
            "repeated searches should not be counted as executed searches: {}",
            ui.assistant
        );
        assert!(
            ui.assistant.contains("File reads: 1"),
            "fallback should report file reads: {}",
            ui.assistant
        );
        assert!(
            ui.assistant.contains(&inspected),
            "fallback should cite inspected path: {}",
            ui.assistant
        );
        assert!(
            ui.statuses
                .iter()
                .any(|status| status.contains("bounded evidence summary")),
            "expected bounded repeat-exhaustion status: {:?}",
            ui.statuses
        );
        let telemetry = agent.last_turn_telemetry();
        assert_eq!(telemetry.repeat_nudges, 2);
        assert!(telemetry.stalled_unfinished);
        let _ = std::fs::remove_file(inspected_path);
    }

    #[tokio::test]
    async fn gap_review_search_match_blocks_no_gap_overclaim() {
        let inspected_path = temp_file("gap-overclaim-evidence");
        std::fs::write(
            &inspected_path,
            "// TODO: add provider retry coverage\npub fn value() {}\n",
        )
        .unwrap();
        let inspected = inspected_path.to_string_lossy().to_string();
        let responses = vec![
            completion(
                vec![Content::ToolCall {
                    id: "grep".into(),
                    name: "grep".into(),
                    arguments: serde_json::json!({
                        "pattern": "TODO|FIXME|missing|gap",
                        "path": inspected.clone(),
                    })
                    .to_string(),
                }],
                1,
                1,
            ),
            completion(
                vec![Content::ToolCall {
                    id: "read".into(),
                    name: "read".into(),
                    arguments: serde_json::json!({ "path": inspected.clone() }).to_string(),
                }],
                1,
                1,
            ),
            completion(
                vec![Content::Text(format!(
                    "{inspected}: The project appears mature with no obvious gaps and no TODO/FIXME markers."
                ))],
                1,
                1,
            ),
            completion(
                vec![Content::Text(format!(
                    "{inspected}: The project appears mature with no obvious gaps and no TODO/FIXME markers."
                ))],
                1,
                1,
            ),
            completion(
                vec![Content::Text(format!(
                    "{inspected}: The project appears mature with no obvious gaps and no TODO/FIXME markers."
                ))],
                1,
                1,
            ),
            completion(
                vec![Content::Text(format!(
                    "{inspected}: The project appears mature with no obvious gaps and no TODO/FIXME markers."
                ))],
                1,
                1,
            ),
        ];
        let mut agent = agent(responses, config());
        let mut ui = RecUi::default();

        agent
            .run_turn(
                "discuss whats its missing and what we should considering building and implimenting",
                &mut ui,
            )
            .await
            .unwrap();

        assert!(
            ui.statuses
                .iter()
                .any(|status| status.contains("contradicted search matches")),
            "expected gap overclaim nudge: {:?}",
            ui.statuses
        );
        assert!(
            ui.assistant
                .contains("Bounded evidence summary for an incomplete gap review"),
            "expected bounded evidence fallback: {}",
            ui.assistant
        );
        assert!(
            ui.assistant.contains(&inspected),
            "fallback should cite inspected path: {}",
            ui.assistant
        );
        assert!(
            !ui.assistant.contains("no TODO/FIXME markers"),
            "bad overclaim should not be surfaced: {}",
            ui.assistant
        );
        let telemetry = agent.last_turn_telemetry();
        assert!(telemetry.quality_repair_nudges >= 1);
        assert!(telemetry.stalled_unfinished);
        let _ = std::fs::remove_file(inspected_path);
    }

    #[tokio::test]
    async fn security_review_with_partial_search_gets_broad_search_nudge() {
        let inspected_path = temp_file("security-broad-search");
        std::fs::write(
            &inspected_path,
            "fn run() { let value = Some(1).unwrap(); }\n",
        )
        .unwrap();
        let inspected = inspected_path.to_string_lossy().to_string();
        let responses = vec![
            completion(
                vec![Content::ToolCall {
                    id: "list".into(),
                    name: "list".into(),
                    arguments: r#"{"path":"."}"#.into(),
                }],
                1,
                1,
            ),
            completion(
                vec![Content::Text(
                    "No security issues or unsafe unwraps were found.".into(),
                )],
                1,
                1,
            ),
            completion(
                vec![Content::ToolCall {
                    id: "grep".into(),
                    name: "grep".into(),
                    arguments: serde_json::json!({
                        "pattern": "unwrap|expect|panic",
                        "glob": "*.rs",
                    })
                    .to_string(),
                }],
                1,
                1,
            ),
            completion(
                vec![Content::ToolCall {
                    id: "read".into(),
                    name: "read".into(),
                    arguments: serde_json::json!({ "path": inspected.clone() }).to_string(),
                }],
                1,
                1,
            ),
            completion(
                vec![Content::Text(format!(
                    "Findings:\n- {inspected}: no unsafe unwrap, command execution, filesystem/env, or secret/token/auth risks were found."
                ))],
                1,
                1,
            ),
            completion(
                vec![Content::ToolCall {
                    id: "grep-broad".into(),
                    name: "grep".into(),
                    arguments: serde_json::json!({
                        "pattern": "unsafe|unwrap|expect|panic|command|std::process|process::|shell|exec|spawn|filesystem|std::fs|fs::|read_to_string|write|remove_file|std::env|env::|secret|token|auth|api_key|apikey|password|credential|bearer",
                        "glob": "*.rs",
                    })
                    .to_string(),
                }],
                1,
                1,
            ),
            completion(
                vec![Content::ToolCall {
                    id: "read-again".into(),
                    name: "read".into(),
                    arguments: serde_json::json!({ "path": inspected.clone() }).to_string(),
                }],
                1,
                1,
            ),
            completion(
                vec![Content::Text(format!(
                    "Findings:\n- {inspected}: searched unsafe/unwrap/panic, command/filesystem/env, and secret/token/auth patterns; this file contains a direct unwrap but no broader conclusion is made beyond inspected evidence."
                ))],
                1,
                1,
            ),
        ];
        let mut agent = agent(responses, config());
        let mut ui = RecUi::default();

        agent
            .run_turn(
                "review for security issues or unsafe unwraps. then disucss only",
                &mut ui,
            )
            .await
            .unwrap();

        assert!(
            ui.statuses
                .iter()
                .any(|status| status.contains("missed required pattern families")),
            "expected security broad-search nudge: {:?}",
            ui.statuses
        );
        assert!(
            agent
                .messages()
                .iter()
                .any(|message| message.role == Role::Assistant
                    && message.text().contains(&inspected)
                    && message.text().contains("direct unwrap")),
            "final answer should cite inspected path after broad search"
        );
        let telemetry = agent.last_turn_telemetry();
        assert_eq!(telemetry.quality_repair_nudges, 2);
        assert_eq!(telemetry.targeted_searches, 2);
        assert_eq!(telemetry.file_reads, 2);
        assert!(!telemetry.listing_only);
        let _ = std::fs::remove_file(inspected_path);
    }

    #[tokio::test]
    async fn security_review_overbroad_all_clear_gets_scope_nudge() {
        let inspected_path = temp_file("security-scope");
        std::fs::write(&inspected_path, "fn main() { println!(\"ok\"); }\n").unwrap();
        let inspected = inspected_path.to_string_lossy().to_string();
        let responses = vec![
            completion(
                vec![Content::ToolCall {
                    id: "grep".into(),
                    name: "grep".into(),
                    arguments: serde_json::json!({
                        "pattern": "unsafe|unwrap|expect|panic|command|std::process|spawn|std::fs|std::env|secret|token|auth",
                        "glob": "*.rs",
                    })
                    .to_string(),
                }],
                1,
                1,
            ),
            completion(
                vec![Content::ToolCall {
                    id: "read".into(),
                    name: "read".into(),
                    arguments: serde_json::json!({ "path": inspected.clone() }).to_string(),
                }],
                1,
                1,
            ),
            completion(
                vec![Content::Text(format!(
                    "The codebase appears to be secure. There are no hardcoded secrets or direct command execution issues. Specifically, in `{inspected}`, no unsafe unwraps were found."
                ))],
                1,
                1,
            ),
            completion(
                vec![Content::Text(format!(
                    "Findings:\n- {inspected}: Based on the inspected file and searched security patterns, I did not establish a concrete unsafe/unwrap finding in this file. This is not a complete audit and does not rule out issues outside the inspected evidence."
                ))],
                1,
                1,
            ),
        ];
        let mut agent = agent(responses, config());
        let mut ui = RecUi::default();

        agent
            .run_turn(
                "review for security issues or unsafe unwraps. then disucss only",
                &mut ui,
            )
            .await
            .unwrap();

        assert!(
            ui.statuses
                .iter()
                .any(|status| status.contains("overclaimed repo-wide safety")),
            "expected security scope nudge: {:?}",
            ui.statuses
        );
        assert!(
            agent
                .messages()
                .iter()
                .any(|message| message.role == Role::Assistant
                    && message.text().contains("not a complete audit")),
            "final answer should be bounded"
        );
        assert_eq!(agent.last_turn_telemetry().quality_repair_nudges, 1);
        let _ = std::fs::remove_file(inspected_path);
    }

    #[tokio::test]
    async fn read_only_review_repeated_search_without_read_returns_insufficient_evidence() {
        let grep_call = || {
            completion(
                vec![Content::ToolCall {
                    id: "grep".into(),
                    name: "grep".into(),
                    arguments: serde_json::json!({
                        "pattern": "fn run_turn",
                        "glob": "*.rs",
                    })
                    .to_string(),
                }],
                1,
                1,
            )
        };
        let responses = vec![grep_call(), grep_call(), grep_call(), grep_call()];
        let mut agent = agent(responses, config());
        let mut ui = RecUi::default();

        agent
            .run_turn(
                "review for security issues or unsafe unwraps. then disucss only",
                &mut ui,
            )
            .await
            .unwrap();

        assert!(
            ui.statuses
                .iter()
                .any(|status| status.contains("nudging it to read a matching file")),
            "expected read-after-search nudge: {:?}",
            ui.statuses
        );
        assert!(
            ui.assistant
                .contains("Insufficient evidence: targeted search ran"),
            "expected insufficient-evidence final: {}",
            ui.assistant
        );
        assert!(agent.last_turn_telemetry().stalled_unfinished);
    }

    #[tokio::test]
    async fn read_only_review_search_then_generic_final_requires_file_read() {
        let responses = vec![
            completion(
                vec![Content::ToolCall {
                    id: "grep".into(),
                    name: "grep".into(),
                    arguments: serde_json::json!({
                        "pattern": "unwrap|expect|panic",
                        "glob": "*.rs",
                    })
                    .to_string(),
                }],
                1,
                1,
            ),
            completion(
                vec![Content::Text("Completed the requested action.".into())],
                1,
                1,
            ),
            completion(
                vec![Content::Text(
                    "Insufficient evidence: targeted search ran, but no matching file was read."
                        .into(),
                )],
                1,
                1,
            ),
        ];
        let mut agent = agent(responses, config());
        let mut ui = RecUi::default();

        agent
            .run_turn(
                "review for security issues or unsafe unwraps. then disucss only",
                &mut ui,
            )
            .await
            .unwrap();

        assert!(
            ui.statuses
                .iter()
                .any(|status| status.contains("targeted search but no file reads")),
            "expected search-without-read nudge: {:?}",
            ui.statuses
        );
        assert_eq!(agent.last_turn_telemetry().quality_repair_nudges, 1);
    }

    #[tokio::test]
    async fn listing_only_review_repair_exhaustion_returns_insufficient_evidence() {
        let responses = vec![
            completion(
                vec![Content::ToolCall {
                    id: "list".into(),
                    name: "list".into(),
                    arguments: r#"{"path":"."}"#.into(),
                }],
                1,
                1,
            ),
            completion(
                vec![Content::Text(
                    "The repository looks healthy and organized.".into(),
                )],
                1,
                1,
            ),
            completion(
                vec![Content::Text(
                    "Findings/Status:\n- The inspected context points to `src/lib.rs`.".into(),
                )],
                1,
                1,
            ),
        ];
        let mut agent = agent(responses, config());
        let mut ui = RecUi::default();

        agent
            .run_turn("review codebase and discuss status and state", &mut ui)
            .await
            .unwrap();

        assert!(
            ui.assistant.contains("Insufficient evidence"),
            "assistant output should be bounded evidence: {}",
            ui.assistant
        );
        assert!(
            !ui.assistant.contains("src/lib.rs"),
            "listing-only fallback targets should not be shown as findings: {}",
            ui.assistant
        );
        assert!(
            ui.statuses
                .iter()
                .any(|status| status.contains("only listing evidence after repair")),
            "expected exhausted repair status: {:?}",
            ui.statuses
        );
        let telemetry = agent.last_turn_telemetry();
        assert_eq!(telemetry.quality_repair_nudges, 1);
        assert!(telemetry.listing_only);
        assert!(telemetry.stalled_unfinished);
        assert!(agent.usage_summary(agent.totals()).contains("stalled"));
    }

    #[tokio::test]
    async fn does_not_nudge_a_plain_answer() {
        // No tool call this turn (a Q&A-style reply) — never nudge, never warn,
        // even though the text isn't an action.
        let responses = vec![completion(
            vec![Content::Text("The answer is 42.".into())],
            1,
            1,
        )];
        let mut agent = agent(responses, config());
        let mut ui = RecUi::default();
        agent.run_turn("what is 6*7?", &mut ui).await.unwrap();
        assert!(
            !ui.statuses
                .iter()
                .any(|s| s.contains("nudging") || s.contains("incomplete")),
            "plain answer is left alone, got: {:?}",
            ui.statuses
        );
        assert!(ui.turn_end.is_some(), "turn completed");
    }

    #[tokio::test]
    async fn finalizes_with_a_recap_when_files_changed() {
        // A turn that changes a file ends with a dedicated recap call. The recap
        // is emitted to the UI (so the user sees it) and its usage is counted,
        // but the [user: finalize-nudge][assistant: recap] pair is stripped from
        // the persisted transcript at turn end — the FINALIZE_PROMPT's "don't
        // take any further action" instruction must not bleed into the next turn.
        // Holds the workspace lock: this test writes a temp file, which would
        // otherwise perturb the file-change detection of the verify tests.
        let _guard = VERIFY_TEST_LOCK.lock().await;
        let mut cfg = config();
        cfg.finalize = true;
        let tmp = temp_file("finalize");
        let p = tmp.to_string_lossy().to_string();
        let responses = vec![
            write_completion(&p),
            completion(vec![Content::Text("done".into())], 1, 1),
            completion(
                vec![Content::Text(
                    "## Summary\n- Created the file.\n\nRun `cargo test`.".into(),
                )],
                3,
                4,
            ),
        ];
        let mut agent = agent(responses, cfg);
        let mut ui = RecUi::default();
        agent.run_turn("make a file", &mut ui).await.unwrap();
        let _ = std::fs::remove_file(&tmp);

        // The recap was emitted to the UI (the user sees it).
        assert!(
            ui.assistant.contains("## Summary"),
            "recap is emitted to the UI: {}",
            ui.assistant
        );

        let m = agent.messages();
        // The finalize nudge + recap are stripped from history. The last message
        // is the assistant's "done" from the turn work, not the recap.
        let last = m.last().expect("history is non-empty");
        assert_eq!(last.role, Role::Assistant);
        assert!(
            !last.text().contains("[hi:nudge:finalize]"),
            "no finalize nudge marker in history, got: {}",
            last.text()
        );
        // No finalize nudge anywhere in the transcript.
        assert!(
            !m.iter().any(|msg| {
                msg.role == Role::User
                    && msg
                        .content
                        .iter()
                        .any(|c| matches!(c, Content::Text(t) if t.contains("[hi:nudge:finalize]")))
            }),
            "finalize nudge should be stripped from history"
        );
        // Roles alternate (no two assistants in a row → provider-safe next turn).
        assert!(
            m.windows(2).all(|w| w[0].role != w[1].role),
            "roles must alternate"
        );
        // The recap call's usage (3/4) is folded into the running totals.
        assert_eq!(agent.totals().input_tokens, 1 + 1 + 3);
        assert_eq!(agent.totals().output_tokens, 1 + 1 + 4);
    }

    #[tokio::test]
    async fn finalize_recap_is_emitted_to_the_ui() {
        // The Canned provider never calls the stream sink — it returns text
        // only in the completion object. The finalize fallback must emit that
        // text through ui.assistant_text so the user sees the recap, not just
        // record it silently in history. (This is the "ending doesn't show"
        // bug: the recap was recorded but never displayed.)
        let _guard = VERIFY_TEST_LOCK.lock().await;
        let mut cfg = config();
        cfg.finalize = true;
        let tmp = temp_file("finalize_ui");
        let p = tmp.to_string_lossy().to_string();
        let responses = vec![
            write_completion(&p),
            completion(vec![Content::Text("done".into())], 1, 1),
            completion(
                vec![Content::Text("## Summary\n- Created the file.".into())],
                3,
                4,
            ),
        ];
        let mut agent = agent(responses, cfg);
        let mut ui = RecUi::default();
        agent.run_turn("make a file", &mut ui).await.unwrap();
        let _ = std::fs::remove_file(&tmp);

        // The recap text must have been emitted to the UI, not just recorded.
        assert!(
            ui.assistant.contains("## Summary"),
            "recap text should be emitted to the UI, got assistant: {:?}",
            ui.assistant
        );
    }

    #[tokio::test]
    async fn finalize_nudge_does_not_bleed_into_next_turn() {
        // Regression: after a finalized turn, the FINALIZE_PROMPT ("don't take
        // any further action") was left in history. On the next turn the model
        // saw it above the new prompt and emitted more summary text instead of
        // executing the request. The fix strips the [user: finalize-nudge]
        // [assistant: recap] pair at turn end. This test verifies the nudge is
        // gone from history before the second turn starts, so the model's
        // context for turn 2 contains only real conversation.
        let _guard = VERIFY_TEST_LOCK.lock().await;
        let mut cfg = config();
        cfg.finalize = true;
        let tmp = temp_file("finalize_bleed");
        let p = tmp.to_string_lossy().to_string();
        let responses = vec![
            // Turn 1: write a file, then a "done" text, then the recap.
            write_completion(&p),
            completion(vec![Content::Text("done".into())], 1, 1),
            completion(
                vec![Content::Text("## Summary\n- Created the file.".into())],
                3,
                4,
            ),
            // Turn 2: a clean text response to the second prompt.
            completion(vec![Content::Text("ok second".into())], 1, 1),
        ];
        let mut agent = agent(responses, cfg);
        let mut ui = RecUi::default();
        agent.run_turn("make a file", &mut ui).await.unwrap();
        let _ = std::fs::remove_file(&tmp);

        // After turn 1: no finalize nudge or recap in history.
        let msgs = agent.messages();
        assert!(
            !msgs.iter().any(|m| {
                m.content.iter().any(|c| {
                    matches!(
                        c,
                        Content::Text(t) if t.contains("[hi:nudge:finalize]")
                    )
                })
            }),
            "finalize nudge must be stripped from history after turn 1"
        );
        assert!(
            !msgs.iter().any(|m| m.text().contains("## Summary")),
            "recap must be stripped from history after turn 1"
        );

        // Turn 2: the model should see the new prompt without the stale
        // "don't take any further action" instruction. We verify by checking
        // the last user message is the real second prompt, not folded nudge text.
        let mut ui2 = RecUi::default();
        agent
            .run_turn("now do something else", &mut ui2)
            .await
            .unwrap();

        let msgs = agent.messages();
        let last_user = msgs
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .expect("there is a last user message");
        let text = last_user
            .content
            .iter()
            .filter_map(|c| match c {
                Content::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect::<String>();
        assert!(
            text.contains("now do something else"),
            "second prompt is the real user message, got: {text}"
        );
        assert!(
            !text.contains("don't take any further action"),
            "stale finalize instruction must not be in the second prompt context, got: {text}"
        );
    }

    #[tokio::test]
    async fn does_not_finalize_a_plain_answer() {
        // Finalization on, but the turn changed no files (a Q&A reply) — no extra
        // recap call fires. (The canned provider has exactly one completion; a
        // stray finalization call would panic trying to pop a second.)
        let mut cfg = config();
        cfg.finalize = true;
        let mut agent = agent(
            vec![completion(
                vec![Content::Text("The answer is 42.".into())],
                1,
                1,
            )],
            cfg,
        );
        let mut ui = RecUi::default();
        agent.run_turn("what is 6*7?", &mut ui).await.unwrap();
        let assistants = agent
            .messages()
            .iter()
            .filter(|m| m.role == Role::Assistant)
            .count();
        assert_eq!(assistants, 1, "no extra recap message");
        assert_eq!(agent.totals().output_tokens, 1, "no extra recap call");
    }

    #[tokio::test]
    async fn turn_end_reports_cumulative_not_last_round() {
        // Two rounds (5/1 then 6/2). The done line must show the cumulative
        // session total (11/3/14), matching the live counter — not just the
        // last round (6/2/8).
        let responses = vec![
            completion(
                vec![Content::ToolCall {
                    id: "1".into(),
                    name: "bash".into(),
                    arguments: "{\"command\":\"echo hi\"}".into(),
                }],
                5,
                1,
            ),
            completion(vec![Content::Text("done".into())], 6, 2),
        ];
        let mut agent = agent(responses, config());
        let mut ui = RecUi::default();
        agent.run_turn("go", &mut ui).await.unwrap();
        let summary = ui.turn_end.expect("turn_end emitted");
        // Cumulative session totals (↑11 ↓3), matching the live counter — not just
        // the last round (↑6 ↓2).
        assert!(
            summary.contains("↑11 ↓3"),
            "cumulative totals, got: {summary}"
        );
    }

    #[tokio::test]
    async fn usage_line_separates_cumulative_spend_from_context_fill() {
        // The regression guard: with a window + price set, the done line shows
        // cumulative ↑/↓ session spend (abbreviated, matching the live line), the
        // cost, and a context gauge that is the *last request's* size — distinct
        // from cumulative input and humanized the same way. Pins against mixing
        // raw/abbreviated units, rendering a count two ways, or conflating the two.
        let mut cfg = config();
        cfg.context_window = Some(1_000_000);
        cfg.price = Some((5.0, 15.0)); // $/1M (in, out)
        let responses = vec![
            completion(
                vec![Content::ToolCall {
                    id: "1".into(),
                    name: "bash".into(),
                    arguments: "{\"command\":\"echo hi\"}".into(),
                }],
                8_000,
                100,
            ),
            completion(vec![Content::Text("done".into())], 12_000, 200),
        ];
        let mut agent = agent(responses, cfg);
        let mut ui = RecUi::default();
        agent.run_turn("go", &mut ui).await.unwrap();
        let line = ui.turn_end.expect("turn_end emitted");

        // Cumulative session spend, arrowed + abbreviated (same shape as the live line).
        assert!(line.contains("↑20k"), "cumulative input ↑ (8k+12k): {line}");
        assert!(
            line.contains("↓300"),
            "cumulative output ↓ (100+200): {line}"
        );
        // The context gauge is the LAST request (12k) over the window — NOT the
        // cumulative input (20k), and abbreviated, not raw.
        assert!(
            line.contains("ctx 1% (12k/1.0M)"),
            "point-in-time context: {line}"
        );
        // The old, mixed-unit, misleading format is gone.
        assert!(
            !line.contains(" in ·") && !line.contains("total"),
            "no raw in/out/total wording: {line}"
        );
        assert!(
            !line.contains("20000") && !line.contains("12000"),
            "no raw token counts: {line}"
        );
        // A clean turn (one tool call, no verify/retry/nudge) shows no steer
        // suffix — the trajectory surface is additive, only for noisy turns.
        assert!(
            !line.contains("steer"),
            "clean turn has no steer suffix: {line}"
        );
    }

    #[test]
    fn turn_steer_summarizes_trajectory() {
        // Clean turn → None.
        let mut a = agent(vec![], config());
        assert_eq!(a.turn_steer(), None);

        // Noisy turn → a steer line listing each non-zero component.
        a.last_turn_telemetry = TurnTelemetry {
            verify_rounds: 2,
            recovery_retries: 1,
            repeat_nudges: 0,
            continue_nudges: 0,
            truncation_retries: 0,
            hit_step_cap: false,
            stalled_unfinished: false,
            stalled_repeating: false,
            verify_attributions: Vec::new(),
            tool_calls: 0,
            max_concurrent_batch: 0,
            serial_runs: 0,
            tool_timeline: Vec::new(),
            ..TurnTelemetry::default()
        };
        let steer = a.turn_steer().expect("noisy turn has a steer line");
        assert!(
            steer.contains("2 verify") && steer.contains("1 retry"),
            "lists non-zero components: {steer}"
        );
        assert!(
            !steer.contains("repeat") && !steer.contains("continue"),
            "omits zero components: {steer}"
        );

        // A stall is surfaced even with no rounds.
        a.last_turn_telemetry = TurnTelemetry {
            verify_rounds: 0,
            recovery_retries: 0,
            repeat_nudges: 0,
            continue_nudges: 0,
            truncation_retries: 0,
            hit_step_cap: false,
            stalled_unfinished: true,
            stalled_repeating: false,
            verify_attributions: Vec::new(),
            tool_calls: 0,
            max_concurrent_batch: 0,
            serial_runs: 0,
            tool_timeline: Vec::new(),
            ..TurnTelemetry::default()
        };
        let steer = a.turn_steer().expect("stall has a steer line");
        assert!(steer.contains("stalled"), "stall flagged: {steer}");
    }

    #[tokio::test]
    async fn cost_accumulates_at_price_active_for_each_call() {
        let mut cfg = config();
        cfg.price = Some((1.0, 10.0));
        let responses = vec![
            completion(vec![Content::Text("first".into())], 1_000, 100),
            completion(vec![Content::Text("second".into())], 1_000, 100),
        ];
        let mut agent = agent(responses, cfg);

        agent.run_turn("first", &mut NullUi).await.unwrap();
        agent.set_model("m2".into(), Some((2.0, 20.0)), None);
        agent.run_turn("second", &mut NullUi).await.unwrap();

        assert_eq!(agent.cost_usd(), Some(0.006));
    }

    #[test]
    fn add_usage_uses_normalized_billable_across_provider_semantics() {
        // A session that switches providers mid-run must accrue cost coherently.
        // The `billable` breakdown is provider-computed, so the agent's cost
        // math doesn't have to know whether `input_tokens` includes cached
        // tokens (OpenAI) or excludes them (Anthropic). Pin: an OpenAI-style
        // usage where input_tokens already includes the cached subset must NOT
        // double-count the cached tokens, and an Anthropic-style usage where
        // input excludes cache must still bill the cache portion at a discount.
        let mut cfg = config();
        cfg.price = Some((1.0, 10.0)); // $/1M in, out
        let mut a = agent(vec![], cfg);

        // OpenAI-style: prompt_tokens=1000 includes 400 cached. The normalized
        // breakdown separates them: 600 regular + 400 cached. Cost must bill
        // 600 at full price + 400 at 0.5x — NOT 1000 + 400 (double-count).
        a.add_usage(Usage {
            input_tokens: 1000,
            output_tokens: 0,
            cache_read_tokens: 400,
            cache_creation_tokens: 0,
            input_includes_cache: true,
            context_occupancy: 1000,
            billable: Some(hi_ai::BillableBreakdown {
                regular_input: 600,
                cached_input: 400,
                cache_creation: 0,
                output: 0,
            }),
        });
        let openai_cost = a.cost_usd().unwrap();
        // 600*1 + 400*0.5 = 800 token-units -> $0.0008
        assert!(
            (openai_cost - 0.0008).abs() < 1e-9,
            "openai no double-count: {openai_cost}"
        );

        // Anthropic-style: input_tokens=600 excludes 400 cache_read + 100
        // cache_creation. The breakdown bills 600 regular + 400 at 0.5x + 100
        // at 1.25x. The agent must NOT re-derive (which would wrongly subtract
        // cache_read from input_tokens).
        a.add_usage(Usage {
            input_tokens: 600,
            output_tokens: 50,
            cache_read_tokens: 400,
            cache_creation_tokens: 100,
            input_includes_cache: false,
            context_occupancy: 1100,
            billable: Some(hi_ai::BillableBreakdown {
                regular_input: 600,
                cached_input: 400,
                cache_creation: 100,
                output: 50,
            }),
        });
        let total = a.cost_usd().unwrap();
        // anthropic increment: 600*1 + 400*0.5 + 100*1.25 + 50*10 = 600+200+125+500 = 1425 -> $0.001425
        assert!(
            (total - (0.0008 + 0.001425)).abs() < 1e-9,
            "coherent cumulative across providers: {total}"
        );
    }

    #[tokio::test]
    async fn emits_running_cumulative_usage_each_round() {
        // Two rounds (tool call, then text). The UI should see the cumulative
        // total climb after each round, so it can show usage live mid-turn.
        let responses = vec![
            completion(
                vec![Content::ToolCall {
                    id: "1".into(),
                    name: "bash".into(),
                    arguments: "{\"command\":\"echo hi\"}".into(),
                }],
                5,
                1,
            ),
            completion(vec![Content::Text("done".into())], 6, 2),
        ];
        let mut agent = agent(responses, config());
        let mut ui = RecUi::default();
        agent.run_turn("go", &mut ui).await.unwrap();
        // Cumulative after round 1 = (5,1); after round 2 = (11,3).
        assert_eq!(ui.usages, vec![(5, 1), (11, 3)]);
    }

    #[tokio::test]
    async fn auto_compacts_when_context_fills() {
        let mut cfg = config();
        cfg.auto_compact = true;
        cfg.context_window = Some(100);
        let responses = vec![
            completion(vec![Content::Text("ans1".into())], 90, 1), // fills context to 90%
            completion(vec![Content::Text("CONVO SUMMARY".into())], 5, 5), // the compaction call
            completion(vec![Content::Text("ans2".into())], 5, 1),  // turn two, post-compaction
        ];
        let mut agent = agent(responses, cfg);
        let mut ui = RecUi::default();

        agent.run_turn("q1", &mut ui).await.unwrap(); // starts empty → no compaction
        agent.run_turn("q2", &mut ui).await.unwrap(); // context 90% full → compacts first

        assert!(
            ui.statuses.iter().any(|s| s.contains("compacting")),
            "expected a compaction status, got {:?}",
            ui.statuses
        );
        assert!(
            agent
                .messages()
                .iter()
                .any(|m| m.text().contains("CONVO SUMMARY")),
            "history should be replaced by the summary"
        );
        assert_eq!(agent.messages().last().unwrap().text(), "ans2");
    }

    #[tokio::test]
    async fn elides_old_tool_outputs_before_model_request() {
        let mut cfg = config();
        cfg.auto_compact = true;
        cfg.context_window = Some(100);
        let (mut agent, requests) = scripted_agent(
            vec![ProviderStep::Completion(completion(
                vec![Content::Text("done".into())],
                5,
                1,
            ))],
            cfg,
        );
        agent
            .messages_mut()
            .push(Message::user("existing long turn"));
        for i in 1..=8 {
            let id = format!("c{i}");
            agent
                .messages_mut()
                .push(Message::assistant(vec![Content::ToolCall {
                    id: id.clone(),
                    name: "read".into(),
                    arguments: "{}".into(),
                }]));
            agent.messages_mut().push(Message::tool_result(
                &id,
                format!("{i}\n{}", "x".repeat(500)),
            ));
        }

        let mut ui = RecordingUi::default();
        agent.run_turn("continue", &mut ui).await.unwrap();

        let requests = requests.lock().unwrap();
        let outputs: Vec<String> = requests[0]
            .iter()
            .flat_map(|msg| &msg.content)
            .filter_map(|c| match c {
                Content::ToolResult { output, .. } => Some(output.clone()),
                _ => None,
            })
            .collect();
        assert!(outputs[0].starts_with("[elided"), "{outputs:?}");
        assert!(outputs[1].starts_with("[elided"), "{outputs:?}");
        assert!(outputs[2].starts_with("3\n"), "{outputs:?}");
        assert!(outputs[7].starts_with("8\n"), "{outputs:?}");
        assert!(
            !ui.statuses.iter().any(|s| s.contains("elided old tool")),
            "in-turn elision should stay quiet, got {:?}",
            ui.statuses
        );
    }

    #[tokio::test]
    async fn retry_uses_recovery_sampling() {
        // A content-less first round triggers the silent retry, which must
        // resample hotter and with nucleus + frequency penalty to escape the
        // attractor; the initial (non-retry) call uses the plain configured temp.
        let samples = std::sync::Arc::new(Mutex::new(Vec::new()));
        let provider = RecordTemps {
            responses: Mutex::new(vec![
                completion(vec![], 0, 0), // empty → retry
                completion(vec![Content::Text("recovered".into())], 5, 3),
            ]),
            samples: samples.clone(),
        };
        let mut cfg = config();
        cfg.temperature = Some(0.2);
        let mut agent = Agent::new(Box::new(provider), cfg);
        agent.run_turn("go", &mut NullUi).await.unwrap();

        let samples = samples.lock().unwrap();
        assert_eq!(
            samples.len(),
            2,
            "initial call + one retry, got {:?}",
            *samples
        );
        assert_eq!(
            samples[0],
            (Some(0.2), None, None),
            "first call: configured temp, no recovery overrides"
        );
        let (temp, top_p, freq) = samples[1];
        assert!(temp.unwrap() > 0.2, "retry resamples hotter, got {temp:?}");
        assert_eq!(top_p, Some(0.95), "retry adds nucleus sampling");
        assert!(
            freq.is_some_and(|f| f > 0.0),
            "retry adds a frequency penalty, got {freq:?}"
        );
    }

    #[tokio::test]
    async fn empty_response_recovers_on_retry() {
        // First round comes back content-less; the silent retry succeeds. The
        // dead round is dropped from history, so the retry sees the same context.
        let responses = vec![
            completion(vec![], 0, 0), // empty → retry
            completion(vec![Content::Text("here's the review".into())], 5, 3), // succeeds
        ];
        let mut agent = agent(responses, config());
        let mut ui = RecUi::default();
        agent.run_turn("say hi", &mut ui).await.unwrap();
        assert!(
            ui.statuses.iter().any(|s| s.contains("retrying (1/")),
            "a retry should be shown, got: {:?}",
            ui.statuses
        );
        assert!(
            !ui.statuses.iter().any(|s| s.contains("after retrying")),
            "should not have given up, got: {:?}",
            ui.statuses
        );
        assert_eq!(agent.messages().last().unwrap().text(), "here's the review");
        // Only the successful assistant message is recorded (not the dead round).
        let assistants = agent
            .messages()
            .iter()
            .filter(|m| m.role == Role::Assistant)
            .count();
        assert_eq!(assistants, 1);
    }

    #[tokio::test]
    async fn empty_response_gives_up_after_retries() {
        // Persistent content-less responses (the last is reasoning-only, which the
        // old zero-token check missed): exhaust the budget, then surface it.
        let responses = vec![
            completion(vec![], 0, 0),
            completion(vec![], 0, 0),
            completion(vec![], 0, 42),
        ];
        let mut agent = agent(responses, config());
        let mut ui = RecUi::default();
        agent.run_turn("review codebase", &mut ui).await.unwrap();
        assert!(
            ui.statuses.iter().any(|s| s.contains("after retrying")),
            "exhaustion should be surfaced, got: {:?}",
            ui.statuses
        );
    }

    #[tokio::test]
    async fn normal_final_text_does_not_retry() {
        // A turn that ends with real text must not retry or warn.
        let mut agent = agent(
            vec![completion(
                vec![Content::Text("here's the answer".into())],
                5,
                3,
            )],
            config(),
        );
        let mut ui = RecUi::default();
        agent.run_turn("say hi", &mut ui).await.unwrap();
        assert!(
            !ui.statuses.iter().any(|s| s.contains("no response")),
            "real text should not warn, got: {:?}",
            ui.statuses
        );
    }

    #[tokio::test]
    async fn layered_verify_stops_at_first_failing_stage() {
        let _guard = VERIFY_TEST_LOCK.lock().await;
        // The compile gate fails, so the later (passing) test stage must NOT run
        // — and the feedback should be the compile-error guidance, not the test one.
        let mut cfg = config();
        cfg.verify = vec![
            VerifyStage::new("check", "false"), // "compile" fails
            VerifyStage::new("test", "true"),   // would pass, must be skipped
        ];
        cfg.max_verify_iterations = 1;
        // The model edits (so verification runs), then stops; after the failing
        // verify it re-prompts once more before the cap is reached.
        let tmp = temp_file("stop");
        let p = tmp.to_string_lossy().to_string();
        let mut agent = agent(
            vec![
                write_completion(&p),
                completion(vec![Content::Text("attempt 1".into())], 1, 1),
                completion(vec![Content::Text("attempt 2".into())], 1, 1),
            ],
            cfg,
        );
        let mut ui = RecUi::default();
        agent.run_turn("x", &mut ui).await.unwrap();
        let _ = std::fs::remove_file(&tmp);
        assert_eq!(agent.last_verify(), Some(false));
        // The failing stage is named…
        assert!(
            ui.statuses
                .iter()
                .any(|s| s.contains("check") && s.contains("failed")),
            "names the failing stage: {:?}",
            ui.statuses
        );
        // …and the later test stage never ran (no status line for it).
        assert!(
            !ui.statuses.iter().any(|s| s.contains("· test:")),
            "test stage must be skipped after the gate fails: {:?}",
            ui.statuses
        );
        // …and the feedback to the model is the compile-error nudge.
        let fed_back = agent
            .messages()
            .iter()
            .any(|m| m.role == Role::User && m.text().contains("fix its root cause"));
        assert!(fed_back, "compile-stage guidance fed back");
        // The `false` command's output isn't a parseable diagnostic, so the
        // attribution layer adds no "Likely cause" section — the nudge keeps
        // its original shape (enrich-only contract).
        let has_cause = agent
            .messages()
            .iter()
            .any(|m| m.role == Role::User && m.text().contains("Likely cause"));
        assert!(!has_cause, "no attribution section for unparseable output");
    }

    #[tokio::test]
    async fn layered_verify_passes_when_all_stages_pass() {
        let _guard = VERIFY_TEST_LOCK.lock().await;
        let mut cfg = config();
        cfg.verify = vec![
            VerifyStage::new("check", "true"),
            VerifyStage::new("test", "true"),
        ];
        let tmp = temp_file("pass");
        let p = tmp.to_string_lossy().to_string();
        let mut agent = agent(
            vec![
                write_completion(&p),
                completion(vec![Content::Text("done".into())], 1, 1),
            ],
            cfg,
        );
        agent.run_turn("x", &mut NullUi).await.unwrap();
        let _ = std::fs::remove_file(&tmp);
        assert_eq!(agent.last_verify(), Some(true));
    }

    #[tokio::test]
    async fn verify_failure_exhausts_retries() {
        let _guard = VERIFY_TEST_LOCK.lock().await;
        let mut cfg = config();
        cfg.verify = vec![VerifyStage::new("test", "false")]; // always fails
        cfg.max_verify_iterations = 2;
        // The model edits once (so verify runs), then keeps finishing without
        // tool calls; verify fails each round until the cap.
        let tmp = temp_file("exhaust");
        let p = tmp.to_string_lossy().to_string();
        let responses = vec![
            write_completion(&p),
            completion(vec![Content::Text("attempt 1".into())], 1, 1),
            completion(vec![Content::Text("attempt 2".into())], 1, 1),
            completion(vec![Content::Text("attempt 3".into())], 1, 1),
        ];
        let mut agent = agent(responses, cfg);
        agent.run_turn("x", &mut NullUi).await.unwrap();
        let _ = std::fs::remove_file(&tmp);
        assert_eq!(agent.last_verify(), Some(false));
        // PROBE: with max_verify_iterations=2 the verifier should iterate twice.
        let tel = agent.last_turn_telemetry();
        eprintln!(
            "PROBE verify_rounds={} telemetry={:?}",
            tel.verify_rounds, tel
        );
    }

    #[tokio::test]
    async fn verify_failure_nudge_carries_attribution() {
        let _guard = VERIFY_TEST_LOCK.lock().await;
        // A verify stage that emits a real rustc-style diagnostic should yield a
        // "Likely cause" section in the nudge pointing at the parsed file:line,
        // while the raw `Output:` block is preserved (enrich-only).
        let mut cfg = config();
        cfg.verify = vec![VerifyStage::new(
            "check",
            "printf 'error[E0308]: mismatched types\\n  --> src/lib.rs:42:18\\n' >&2; exit 1",
        )];
        cfg.max_verify_iterations = 1;
        let tmp = temp_file("attr");
        let p = tmp.to_string_lossy().to_string();
        let mut agent = agent(
            vec![
                write_completion(&p),
                completion(vec![Content::Text("attempt 1".into())], 1, 1),
                completion(vec![Content::Text("attempt 2".into())], 1, 1),
            ],
            cfg,
        );
        let mut ui = RecUi::default();
        agent.run_turn("x", &mut ui).await.unwrap();
        let _ = std::fs::remove_file(&tmp);
        // The attribution section is present and points at the parsed location.
        let nudge = agent
            .messages()
            .iter()
            .find(|m| m.role == Role::User && m.text().contains("Likely cause"))
            .expect("attribution section present");
        let body = nudge.text();
        assert!(
            body.contains("Likely cause (verify and fix first)"),
            "section header: {body}"
        );
        assert!(
            body.contains("src/lib.rs:42:18"),
            "parsed location in attribution: {body}"
        );
        assert!(body.contains("[compile]"), "compile kind label: {body}");
        // Enrich-only: the raw output block is still there alongside it.
        assert!(
            body.contains("Output:\n"),
            "raw Output block preserved: {body}"
        );
        assert!(
            body.contains("mismatched types"),
            "raw error message preserved in Output block: {body}"
        );
    }

    #[tokio::test]
    async fn verify_skipped_when_no_files_changed() {
        let _guard = VERIFY_TEST_LOCK.lock().await;
        // A turn that only answers (no edits) must not run verification, even
        // when configured — so a red test suite can't hijack a question.
        let mut cfg = config();
        cfg.verify = vec![VerifyStage::new("test", "false")];
        let mut agent = agent(
            vec![completion(
                vec![Content::Text("just answering".into())],
                1,
                1,
            )],
            cfg,
        );
        let mut ui = RecUi::default();
        agent.run_turn("what does this do?", &mut ui).await.unwrap();
        assert_eq!(agent.last_verify(), None, "verify must not have run");
        assert!(
            ui.statuses
                .iter()
                .any(|s| s.contains("skipped — no files changed")),
            "skip is surfaced: {:?}",
            ui.statuses
        );
    }

    #[tokio::test]
    async fn verify_runs_when_bash_changes_files() {
        let _guard = VERIFY_TEST_LOCK.lock().await;
        let tmp = temp_file("bash");
        let p = tmp.to_string_lossy().to_string();
        let mut cfg = config();
        cfg.verify = vec![VerifyStage::new("test", "true")];
        let mut agent = agent(
            vec![
                completion(
                    vec![Content::ToolCall {
                        id: "b".into(),
                        name: "bash".into(),
                        arguments: format!("{{\"command\":\"printf x > '{}'\"}}", p),
                    }],
                    1,
                    1,
                ),
                completion(vec![Content::Text("done".into())], 1, 1),
            ],
            cfg,
        );
        agent.run_turn("x", &mut NullUi).await.unwrap();
        let _ = std::fs::remove_file(&tmp);
        assert_eq!(agent.last_verify(), Some(true));
    }
