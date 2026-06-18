//! The agent loop: user message → model → tool calls → results → repeat
//! until the model stops calling tools. No artificial step limit.

pub mod command;
pub mod session;
pub mod ui;

use anyhow::Result;
use pi_ai::{ChatRequest, Message, Provider, StreamEvent, ToolSpec, Usage};
use pi_tools::{execute, tool_specs};

pub use command::Command;
pub use session::SessionSink;
pub use ui::{Ui, preview_args};

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
        }
    }

    /// Attach a sink that records messages produced from here on.
    pub fn set_session(&mut self, session: Box<dyn SessionSink>) {
        self.session = Some(session);
    }

    pub fn messages(&self) -> &[Message] {
        &self.messages
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

    /// Run one user turn to completion, emitting output through `ui`.
    ///
    /// After the model stops calling tools, an optional verification command is
    /// run; if it fails, its output is fed back and the model iterates, up to
    /// `max_verify_iterations` rounds.
    pub async fn run_turn(&mut self, input: &str, ui: &mut dyn Ui) -> Result<()> {
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
            let mut round_usage = pi_ai::Usage::default();
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
                };
                let completion = self.provider.stream(request, &mut sink).await?;
                ui.assistant_end();

                self.totals.input_tokens += completion.usage.input_tokens;
                self.totals.output_tokens += completion.usage.output_tokens;

                let calls: Vec<(String, String, String)> = completion
                    .tool_calls()
                    .into_iter()
                    .map(|c| (c.id.to_string(), c.name.to_string(), c.arguments.to_string()))
                    .collect();

                round_usage = completion.usage.clone();
                self.messages.push(Message::assistant(completion.content));

                if calls.is_empty() {
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
                    ui.status(&format!("running verification ({verify_round}/{max_verify}): {cmd}"));
                    let (passed, output) = pi_tools::run_check(cmd).await;
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

    fn usage_summary(&self, usage: &pi_ai::Usage) -> String {
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
