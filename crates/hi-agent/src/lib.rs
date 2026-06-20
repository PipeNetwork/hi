//! The agent loop: user message → model → tool calls → results → repeat
//! until the model stops calling tools. No artificial step limit.

pub mod command;
pub mod session;
pub mod ui;

use anyhow::Result;
use hi_ai::{ChatRequest, Content, Message, Provider, StreamEvent, ToolSpec, Usage};
use hi_tools::{execute, tool_specs};

pub use command::Command;
pub use session::SessionSink;
pub use ui::{Ui, preview_args};

/// Auto-compact once the context window is at least this percent full.
const AUTO_COMPACT_PERCENT: u64 = 80;

const SYSTEM_PROMPT: &str = "\
You are hi, a coding agent running in the user's terminal, in their current \
working directory. You can read, write, and edit files and run shell commands \
via your tools. Work directly on the user's project. Prefer making the change \
over describing it. Keep responses concise. When the task is done, stop.";

/// The system message, optionally with project context appended.
fn build_system(project_context: Option<&str>) -> Message {
    match project_context {
        Some(context) if !context.trim().is_empty() => {
            Message::system(format!("{SYSTEM_PROMPT}\n\n{}", context.trim()))
        }
        _ => Message::system(SYSTEM_PROMPT),
    }
}

/// Per-session configuration the agent applies to every request.
pub struct AgentConfig {
    pub model: String,
    pub max_tokens: u32,
    pub temperature: Option<f32>,
    pub thinking_budget: Option<u32>,
    /// USD per 1M (input, output) tokens, when known — used for cost display.
    pub price: Option<(f64, f64)>,
    /// Model context window, when known — used to show how full it is.
    pub context_window: Option<u32>,
    /// Project context (e.g. from HI.md/AGENTS.md) appended to the system prompt.
    pub project_context: Option<String>,
    /// Shell command run after the model stops, to check the work. On failure
    /// its output is fed back and the model iterates (verification-in-the-loop).
    pub verify_command: Option<String>,
    /// Cap on verification retry rounds.
    pub max_verify_iterations: u32,
    /// Safety cap on model calls per turn, to stop runaway tool loops.
    pub max_steps: u32,
    /// When the context window fills past a threshold, summarize-and-reset
    /// before the next turn so a long session doesn't overflow the model.
    pub auto_compact: bool,
}

pub struct Agent {
    provider: Box<dyn Provider>,
    config: AgentConfig,
    messages: Vec<Message>,
    tools: Vec<ToolSpec>,
    session: Option<Box<dyn SessionSink>>,
    /// How many messages have already been handed to the session sink.
    persisted: usize,
    /// Running total of tokens across the session.
    totals: Usage,
    /// Whether the most recent turn's verification passed (None if not run).
    last_verify: Option<bool>,
    /// Input tokens of the most recent model call — a proxy for how full the
    /// context window is, used to decide when to auto-compact.
    context_used: u64,
}

impl Agent {
    /// Start a fresh session seeded with the system prompt.
    pub fn new(provider: Box<dyn Provider>, config: AgentConfig) -> Self {
        let system = build_system(config.project_context.as_deref());
        Self::with_messages(provider, config, vec![system], 0)
    }

    /// Resume from previously-saved history (which already includes the system
    /// prompt). The loaded messages are treated as already persisted.
    pub fn resume(provider: Box<dyn Provider>, config: AgentConfig, history: Vec<Message>) -> Self {
        let persisted = history.len();
        Self::with_messages(provider, config, history, persisted)
    }

    fn with_messages(
        provider: Box<dyn Provider>,
        config: AgentConfig,
        messages: Vec<Message>,
        persisted: usize,
    ) -> Self {
        Self {
            provider,
            config,
            messages,
            tools: tool_specs(),
            session: None,
            persisted,
            totals: Usage::default(),
            last_verify: None,
            context_used: 0,
        }
    }

    /// Attach a sink that records messages produced from here on.
    pub fn set_session(&mut self, session: Box<dyn SessionSink>) {
        self.session = Some(session);
    }

    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Discard messages back to `len` — used to drop an interrupted turn so the
    /// conversation stays consistent (no dangling user message).
    pub fn truncate_messages(&mut self, len: usize) {
        self.messages.truncate(len);
        self.persisted = self.persisted.min(self.messages.len());
    }

    /// Cumulative token usage across the session.
    pub fn totals(&self) -> &Usage {
        &self.totals
    }

    pub fn model(&self) -> &str {
        &self.config.model
    }

    /// Switch the model used for subsequent turns, refreshing the pricing and
    /// context-window metadata that drive the usage display.
    pub fn set_model(
        &mut self,
        model: String,
        price: Option<(f64, f64)>,
        context_window: Option<u32>,
    ) {
        self.config.model = model;
        self.config.price = price;
        self.config.context_window = context_window;
    }

    /// Reset the live context to just the system prompt. This is transient: it
    /// doesn't rewrite the session file, and the reset point isn't persisted, so
    /// resuming replays the full log.
    pub fn clear_history(&mut self) {
        self.messages = vec![build_system(self.config.project_context.as_deref())];
        self.persisted = self.messages.len();
    }

    /// Whether the most recent turn's verification passed (None if not run).
    pub fn last_verify(&self) -> Option<bool> {
        self.last_verify
    }

    /// The verify command turns iterate against, if any.
    pub fn verify_command(&self) -> Option<&str> {
        self.config.verify_command.as_deref()
    }

    /// Set or clear the verify command applied to subsequent turns.
    pub fn set_verify_command(&mut self, cmd: Option<String>) {
        self.config.verify_command = cmd;
    }

    /// Summarize the conversation so far via the model and reset the live
    /// context to just the system prompt plus that summary — reclaiming context
    /// on long sessions. Transient like [`clear_history`](Self::clear_history):
    /// the session file keeps the full log, so resuming replays everything.
    pub async fn compact(&mut self, ui: &mut dyn Ui) -> Result<()> {
        // Need at least one exchange beyond the system prompt to summarize.
        if self.messages.len() <= 1 {
            ui.status("nothing to compact yet");
            return Ok(());
        }
        ui.status("compacting the conversation…");

        let mut messages = self.messages.clone();
        messages.push(Message::user(
            "Summarize our conversation so far into a concise but complete handoff brief: \
             the task and goal, key decisions and constraints, files created or changed, \
             commands that matter, and any open or next steps. This summary will REPLACE \
             the history, so include everything needed to continue seamlessly. Output only \
             the summary.",
        ));

        let request = ChatRequest {
            model: self.config.model.clone(),
            messages,
            tools: Vec::new(), // summarizing — no tool use
            max_tokens: self.config.max_tokens,
            temperature: self.config.temperature,
            thinking_budget: None,
        };

        let mut summary = String::new();
        let mut sink = |event: StreamEvent| match event {
            StreamEvent::Text(text) => {
                summary.push_str(&text);
                ui.assistant_text(&text);
            }
            StreamEvent::Status(text) => ui.status(&text),
            StreamEvent::Reasoning(_) => {}
        };
        let completion = self.provider.stream(request, &mut sink).await?;
        ui.assistant_end();
        self.totals.input_tokens += completion.usage.input_tokens;
        self.totals.output_tokens += completion.usage.output_tokens;

        // Fall back to the final content if the provider didn't stream text.
        if summary.trim().is_empty() {
            for c in &completion.content {
                if let Content::Text(t) = c {
                    summary.push_str(t);
                }
            }
        }
        if summary.trim().is_empty() {
            ui.status("compaction produced no summary; keeping history");
            return Ok(());
        }

        let system = build_system(self.config.project_context.as_deref());
        self.messages = vec![
            system,
            Message::user(format!(
                "[Summary of the conversation so far]\n\n{}",
                summary.trim()
            )),
        ];
        self.persisted = self.messages.len();
        ui.status("✓ compacted — context reset to the summary");
        Ok(())
    }

    /// Run one user turn to completion, emitting output through `ui`.
    ///
    /// After the model stops calling tools, an optional verification command is
    /// run; if it fails, its output is fed back and the model iterates, up to
    /// `max_verify_iterations` rounds.
    pub async fn run_turn(&mut self, input: &str, ui: &mut dyn Ui) -> Result<()> {
        // If the context window is filling up, summarize-and-reset before adding
        // more, so the session keeps going instead of overflowing. Best-effort:
        // a failed compaction just leaves the history as-is.
        if self.config.auto_compact
            && let Some(window) = self.config.context_window
            && window > 0
            && self.context_used * 100 >= u64::from(window) * AUTO_COMPACT_PERCENT
        {
            ui.status(&format!(
                "context ~{}% full — compacting to free room",
                self.context_used * 100 / u64::from(window)
            ));
            let _ = self.compact(ui).await;
            self.context_used = 0;
        }

        self.messages.push(Message::user(input));
        self.last_verify = None;

        let verify = self.config.verify_command.clone();
        let max_verify = self.config.max_verify_iterations;
        let max_steps = self.config.max_steps;
        let mut verify_round = 0u32;
        let mut steps = 0u32;

        let final_usage = 'turn: loop {
            // Inner loop: model + tools until the model stops calling tools, or
            // the per-turn step cap is hit. Yields (usage, hit_step_cap).
            let mut round_usage = hi_ai::Usage::default();
            let hit_cap = loop {
                if steps >= max_steps {
                    break true;
                }
                steps += 1;

                let request = ChatRequest {
                    model: self.config.model.clone(),
                    messages: self.messages.clone(),
                    tools: self.tools.clone(),
                    max_tokens: self.config.max_tokens,
                    temperature: self.config.temperature,
                    thinking_budget: self.config.thinking_budget,
                };

                let mut sink = |event: StreamEvent| match event {
                    StreamEvent::Text(text) => ui.assistant_text(&text),
                    StreamEvent::Reasoning(text) => ui.assistant_reasoning(&text),
                    StreamEvent::Status(text) => ui.status(&text),
                };
                let completion = self.provider.stream(request, &mut sink).await?;
                ui.assistant_end();

                self.totals.input_tokens += completion.usage.input_tokens;
                self.totals.output_tokens += completion.usage.output_tokens;
                // Latest context fill, for the auto-compaction decision next turn.
                if completion.usage.input_tokens > 0 {
                    self.context_used = completion.usage.input_tokens;
                }

                let calls: Vec<(String, String, String)> = completion
                    .tool_calls()
                    .into_iter()
                    .map(|c| {
                        (
                            c.id.to_string(),
                            c.name.to_string(),
                            c.arguments.to_string(),
                        )
                    })
                    .collect();

                round_usage = completion.usage.clone();
                let produced_nothing = completion.content.is_empty();
                self.messages.push(Message::assistant(completion.content));

                if calls.is_empty() {
                    // The model stopped without any text or tool calls and with
                    // no output tokens — i.e. the provider streamed nothing back
                    // (an overloaded/rate-limited backend, or a dropped stream).
                    // Say so, otherwise the turn just ends on a blank screen.
                    if produced_nothing && round_usage.output_tokens == 0 {
                        ui.status(
                            "⚠ the model returned an empty response (no text or tool calls). \
                             The provider may be overloaded or rate-limiting — try again, or \
                             use /model to switch. /retry re-runs this prompt.",
                        );
                    }
                    break false;
                }
                for (id, name, arguments) in calls {
                    ui.tool_call(&name, &arguments);
                    let output = execute(&name, &arguments).await;
                    ui.tool_result(output.display.as_deref().unwrap_or(&output.content));
                    self.messages.push(Message::tool_result(id, output.content));
                }
            };

            if hit_cap {
                ui.status(&format!("reached step limit ({max_steps}); stopping turn"));
                break 'turn round_usage;
            }

            // Verification gate.
            match &verify {
                Some(cmd) if verify_round < max_verify => {
                    verify_round += 1;
                    ui.status(&format!(
                        "running verification ({verify_round}/{max_verify}): {cmd}"
                    ));
                    let (passed, output) = hi_tools::run_check(cmd).await;
                    if passed {
                        ui.status("✓ verification passed");
                        self.last_verify = Some(true);
                        break 'turn round_usage;
                    }
                    ui.status("✗ verification failed; iterating");
                    self.last_verify = Some(false);
                    self.messages.push(Message::user(format!(
                        "Verification failed. Command: `{cmd}`\n\nOutput:\n{output}\n\n\
                         The tests define the exact required behavior. Compare the expected \
                         and actual values above to work out the precise rule — including \
                         edge cases and tie-breaking — then edit the code so every check passes."
                    )));
                }
                // No verification, or retries exhausted: end the turn.
                _ => break 'turn round_usage,
            }
        };

        ui.turn_end(&self.usage_summary(&final_usage));
        self.persist()?;
        Ok(())
    }

    fn usage_summary(&self, usage: &hi_ai::Usage) -> String {
        let mut summary = format!(
            "[{} in · {} out · {} total",
            usage.input_tokens,
            usage.output_tokens,
            usage.total()
        );
        if let Some((input_price, output_price)) = self.config.price {
            let cost = (usage.input_tokens as f64 * input_price
                + usage.output_tokens as f64 * output_price)
                / 1_000_000.0;
            summary.push_str(&format!(" · ${cost:.4}"));
        }
        if let Some(window) = self.config.context_window {
            summary.push_str(&format!(
                " · {}k/{}k ctx",
                usage.input_tokens / 1000,
                window / 1000
            ));
        }
        summary.push(']');
        summary
    }

    fn persist(&mut self) -> Result<()> {
        if let Some(session) = self.session.as_mut() {
            session.record(&self.messages[self.persisted..])?;
            self.persisted = self.messages.len();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use hi_ai::{ChatRequest, Completion, Content, Provider, Role, StreamEvent, Usage};
    use std::sync::Mutex;

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

    struct NullUi;
    impl Ui for NullUi {
        fn assistant_text(&mut self, _: &str) {}
        fn assistant_reasoning(&mut self, _: &str) {}
        fn assistant_end(&mut self) {}
        fn tool_call(&mut self, _: &str, _: &str) {}
        fn tool_result(&mut self, _: &str) {}
        fn status(&mut self, _: &str) {}
        fn turn_end(&mut self, _: &str) {}
    }

    fn config() -> AgentConfig {
        AgentConfig {
            model: "m".into(),
            max_tokens: 100,
            temperature: None,
            thinking_budget: None,
            price: None,
            context_window: None,
            project_context: None,
            verify_command: None,
            max_verify_iterations: 2,
            max_steps: 50,
            auto_compact: false,
        }
    }

    fn completion(content: Vec<Content>, input: u64, output: u64) -> Completion {
        Completion {
            content,
            usage: Usage {
                input_tokens: input,
                output_tokens: output,
            },
            stop_reason: None,
        }
    }

    fn agent(responses: Vec<Completion>, cfg: AgentConfig) -> Agent {
        Agent::new(Box::new(Canned(Mutex::new(responses))), cfg)
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
    async fn compact_replaces_history_with_summary() {
        let responses = vec![completion(
            vec![Content::Text(
                "BRIEF: ported the parser; tests green".into(),
            )],
            7,
            5,
        )];
        let mut agent = agent(responses, config());
        // Some history to compact.
        agent.messages.push(Message::user("hello"));
        agent
            .messages
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
    }

    #[derive(Default)]
    struct RecUi {
        statuses: Vec<String>,
    }
    impl Ui for RecUi {
        fn assistant_text(&mut self, _: &str) {}
        fn assistant_reasoning(&mut self, _: &str) {}
        fn assistant_end(&mut self) {}
        fn tool_call(&mut self, _: &str, _: &str) {}
        fn tool_result(&mut self, _: &str) {}
        fn status(&mut self, t: &str) {
            self.statuses.push(t.to_string());
        }
        fn turn_end(&mut self, _: &str) {}
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
    async fn empty_completion_is_surfaced() {
        // Provider streams nothing back: empty content, zero tokens.
        let mut agent = agent(vec![completion(vec![], 0, 0)], config());
        let mut ui = RecUi::default();
        agent.run_turn("review codebase", &mut ui).await.unwrap();
        assert!(
            ui.statuses.iter().any(|s| s.contains("empty response")),
            "an empty completion should be surfaced to the user, got: {:?}",
            ui.statuses
        );
    }

    #[tokio::test]
    async fn verify_failure_exhausts_retries() {
        let mut cfg = config();
        cfg.verify_command = Some("false".into()); // always fails
        cfg.max_verify_iterations = 2;
        // Each round the model "finishes" (no tool calls), so verify runs.
        let responses = vec![
            completion(vec![Content::Text("attempt 1".into())], 1, 1),
            completion(vec![Content::Text("attempt 2".into())], 1, 1),
            completion(vec![Content::Text("attempt 3".into())], 1, 1),
        ];
        let mut agent = agent(responses, cfg);
        agent.run_turn("x", &mut NullUi).await.unwrap();
        assert_eq!(agent.last_verify(), Some(false));
    }
}
