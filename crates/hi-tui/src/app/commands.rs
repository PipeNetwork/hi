//! `App` methods: commands.


use ansi_to_tui::IntoText;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use hi_agent::{Agent, Command, command};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Text};

use crate::model_picker::ModelPicker;
use crate::render::dim;
use crate::util::{copy_to_clipboard, goal_feedback};
use crate::{
    App,
    TurnState, working_tree_diff_sync,
};

impl crate::App {

    /// Apply a pure editing/navigation key to the input line, shared by the
    /// idle input phase and the in-turn queue-entry path. Returns the submitted
    /// text on Enter (when non-empty); the caller decides whether to run it now
    /// or queue it. Phase-specific control keys (Ctrl-C/Ctrl-D/Esc) are handled
    /// by the caller, not here.
    pub(crate) fn edit_key(&mut self, key: &KeyEvent) -> Option<String> {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let alt = key.modifiers.contains(KeyModifiers::ALT);
        // --- Ctrl-R reverse history search mode ---
        // When active, keystrokes go to the search filter, not the input line.
        if let Some(search) = &mut self.history_search {
            match key.code {
                KeyCode::Enter => {
                    // Load the highlighted match into the input and submit it.
                    let idx = search.current();
                    self.history_search = None;
                    if let Some(i) = idx
                        && i < self.input.history.len()
                    {
                        self.input.set(&self.input.history[i].clone());
                        let line = self.input.submit();
                        if !line.trim().is_empty() {
                            return Some(line);
                        }
                    }
                    return None;
                }
                KeyCode::Esc => {
                    // On Esc, load the highlighted match for editing (don't submit).
                    if let Some(i) = search.current()
                        && i < self.input.history.len()
                    {
                        self.input.set(&self.input.history[i].clone());
                    }
                    self.history_search = None;
                    return None;
                }
                KeyCode::Char('r') if ctrl => {
                    // Cycle to the next match (like bash Ctrl-R).
                    search.next();
                    if let Some(i) = search.current()
                        && i < self.input.history.len()
                    {
                        self.input.set(&self.input.history[i].clone());
                    }
                    return None;
                }
                KeyCode::Char('s') if ctrl => {
                    search.prev();
                    if let Some(i) = search.current()
                        && i < self.input.history.len()
                    {
                        self.input.set(&self.input.history[i].clone());
                    }
                    return None;
                }
                KeyCode::Backspace => {
                    search.backspace(&self.input.history);
                    if let Some(i) = search.current()
                        && i < self.input.history.len()
                    {
                        self.input.set(&self.input.history[i].clone());
                    }
                    return None;
                }
                KeyCode::Up => {
                    search.prev();
                    if let Some(i) = search.current()
                        && i < self.input.history.len()
                    {
                        self.input.set(&self.input.history[i].clone());
                    }
                    return None;
                }
                KeyCode::Down => {
                    search.next();
                    if let Some(i) = search.current()
                        && i < self.input.history.len()
                    {
                        self.input.set(&self.input.history[i].clone());
                    }
                    return None;
                }
                KeyCode::Char(c) if !ctrl => {
                    search.insert(c, &self.input.history);
                    if let Some(i) = search.current()
                        && i < self.input.history.len()
                    {
                        self.input.set(&self.input.history[i].clone());
                    }
                    return None;
                }
                _ => return None,
            }
        }
        match key.code {
            // Alt+Enter inserts a newline (multi-line prompt without pasting); so
            // does a trailing backslash, for terminals that can't send Alt+Enter.
            KeyCode::Enter if alt => self.input.insert('\n'),
            KeyCode::Enter if self.input.continue_line() => {}
            KeyCode::Enter => {
                let line = self.input.submit();
                if !line.trim().is_empty() {
                    return Some(line);
                }
            }
            KeyCode::Char('u') if ctrl => self.input.kill_to_start(),
            KeyCode::Char('a') if ctrl => self.input.home(),
            KeyCode::Char('e') if ctrl => self.input.end(),
            // Toggle the working-tree diff panel. Refreshed when opened so it
            // reflects the current tree, not a stale snapshot. Fetched
            // synchronously (a `git diff` is fast and user-initiated) since the
            // key handler isn't async.
            KeyCode::Char('d') if ctrl => {
                self.show_diff = !self.show_diff;
                if self.show_diff {
                    self.diff_text = Some(working_tree_diff_sync());
                } else {
                    self.diff_text = None;
                }
            }
            // Toggle the agent-observability panel (Ctrl-? = Ctrl-Shift-/).
            // Shows the last turn's trajectory telemetry, tool-call count, and
            // context composition — read-only diagnostics for the agent's own
            // behavior.
            KeyCode::Char('?') if ctrl => {
                self.show_debug = !self.show_debug;
            }
            // Toggle reasoning (CoT) expansion: collapsed "thought for Ns"
            // summaries vs. the full thinking text. Off by default so reasoning
            // doesn't flood the transcript; Ctrl-T shows/hides all blocks.
            KeyCode::Char('t') if ctrl => {
                self.show_reasoning = !self.show_reasoning;
            }
            KeyCode::Home => self.input.home(),
            KeyCode::End => self.input.end(),
            // `?` on an empty input line toggles a keybindings help overlay;
            // when there's text, it's a normal character.
            KeyCode::Char('?') if !ctrl && self.input.is_empty() => {
                self.show_help = !self.show_help;
            }
            KeyCode::Char(c) if !ctrl => self.input.insert(c),
            KeyCode::Backspace => self.input.backspace(),
            KeyCode::Left => self.input.left(),
            KeyCode::Right => self.input.right(),
            KeyCode::Up => self.input.history_prev(),
            KeyCode::Down => self.input.history_next(),
            KeyCode::PageUp => self.scroll_up(5),
            KeyCode::PageDown => self.scroll_down(5),
            _ => {}
        }
        None
    }

    pub(crate) fn write_debug_log(&mut self) {
        let path = std::path::Path::new(".hi-debug.log");
        let mut body = String::new();
        body.push_str("# hi debug log\n\n");
        body.push_str("## status\n");
        body.push_str(&format!(
            "provider: {}\nmodel: {}\n",
            self.provider, self.model
        ));
        body.push_str(&format!("status: {}\n\n", self.status));
        body.push_str(&format!(
            "last_error: {}\nwaiting_for: {:?}\nstartup_notice: {}\nlast_turn_file_edits: {}\n\n",
            self.last_error.as_deref().unwrap_or("none"),
            self.waiting_for,
            self.startup_notice.as_deref().unwrap_or("none"),
            self.last_turn_had_file_edits
        ));
        body.push_str("## events\n");
        for event in &self.event_log {
            body.push_str(event);
            body.push('\n');
        }
        body.push_str("\n## transcript\n");
        body.push_str(&self.transcript_text());
        match std::fs::write(path, body) {
            Ok(()) => self.push(Line::styled("wrote debug log: .hi-debug.log", dim())),
            Err(err) => self.push(Line::styled(
                format!("log failed: {err}"),
                Style::default().fg(Color::Yellow),
            )),
        }
        self.follow();
    }

    pub(crate) fn copy(&mut self, arg: &str) {
        let text = match arg.trim() {
            "all" | "transcript" => self.transcript_text(),
            _ => self.last_assistant.trim().to_string(),
        };
        if text.is_empty() {
            self.push(Line::styled("nothing to copy yet", dim()));
        } else {
            match copy_to_clipboard(&text) {
                Ok(()) => self.push(Line::styled(format!("copied {} chars", text.len()), dim())),
                Err(err) => self.push(Line::styled(
                    format!("copy failed: {err}"),
                    Style::default().fg(Color::Yellow),
                )),
            }
        }
        self.follow();
    }

    pub(crate) fn handle_goal(&mut self, agent: &mut Agent, arg: &str) {
        // Apply the change first, then describe the resulting state. When
        // long-horizon agency is on, setting a goal creates a structured `Goal`
        // (a single sub-goal equal to the objective, which the model decomposes
        // as it works via `update_plan`); clearing drops both views.
        match arg.trim() {
            "" => {} // no argument — just report the current goal
            "clear" | "off" | "none" => {
                agent.set_goal(None);
                agent.set_structured_goal(None);
            }
            goal => {
                if agent.long_horizon() {
                    let accepted = agent.set_structured_goal(Some(hi_agent::Goal::new(
                        goal.to_string(),
                        vec![goal.to_string()],
                    )));
                    if !accepted {
                        agent.set_goal(Some(goal.to_string()));
                    }
                } else {
                    agent.set_goal(Some(goal.to_string()));
                }
            }
        }
        // Report whichever view is active.
        let (msg, prominent) = if let Some(g) = agent.structured_goal() {
            let done = g
                .sub_goals
                .iter()
                .filter(|s| s.status == hi_agent::GoalStatus::Done)
                .count();
            (
                format!(
                    "goal: {} — {}/{} sub-goals done",
                    g.objective,
                    done,
                    g.sub_goals.len()
                ),
                true,
            )
        } else {
            goal_feedback(arg, agent.goal())
        };
        // A set/clear is an applied change — show it plainly (green ✓), not dim,
        // so it's obvious it took effect. A bare `/goal` is just a read-out.
        let style = if prominent {
            Style::default().fg(Color::Green)
        } else {
            dim()
        };
        self.push(Line::styled(msg, style));
        self.follow();
    }

    pub(crate) async fn handle_command(&mut self, agent: &mut Agent, command: Command, registry: &hi_ai::Registry) {
        match command {
            Command::Quit => {}
            Command::Help => {
                for line in command::help_text().lines() {
                    self.push(Line::styled(line.to_string(), dim()));
                }
            }
            Command::Tokens => {
                let t = agent.totals();
                self.usage = (t.input_tokens, t.output_tokens);
                let cost = agent
                    .cost_usd()
                    .map(|c| format!(" · ${c:.4}"))
                    .unwrap_or_default();
                self.push(Line::styled(
                    format!(
                        "cumulative: {} in · {} out · {} total{}",
                        t.input_tokens,
                        t.output_tokens,
                        t.total(),
                        cost,
                    ),
                    dim(),
                ));
            }
            Command::Status => self.report_status(agent),
            Command::Log => self.write_debug_log(),
            Command::Model(id) => {
                if id.is_empty() {
                    // Open the interactive picker (filter + arrow-select) on the
                    // live served list — no static catalog fallback.
                    let current = self.model.clone();
                    let tags = self.served_tags();
                    let mut ids: Vec<String> = self.served.keys().cloned().collect();
                    ids.sort();
                    let caps = App::capabilities_map(registry, &ids);
                    if ids.is_empty() {
                        self.push(Line::styled(
                            "no models fetched yet — the endpoint didn't respond".to_string(),
                            dim(),
                        ));
                    } else {
                        self.picker =
                            Some(ModelPicker::new(ids, &current, tags, &self.served, &caps));
                    }
                } else {
                    let health = self.apply_model(agent, registry, &id);
                    self.push(Line::styled(format!("model set to {id}"), dim()));
                    if let Some(h) = health {
                        self.warn_degraded(&id, &h);
                    }
                }
            }
            Command::Clear => {
                let count = agent
                    .messages()
                    .iter()
                    .filter(|m| m.role != hi_ai::Role::System)
                    .count();
                agent.clear_history();
                self.transcript.clear();
                self.pending = None;
                self.code_lang = None;
                self.current_assistant.clear();
                self.last_assistant.clear();
                self.status.clear();
                self.last_turn_state = TurnState::Idle;
                self.push(Line::styled(
                    format!("cleared {count} messages — starting fresh"),
                    dim(),
                ));
            }
            Command::Verify(arg) => {
                let msg = match arg.trim() {
                    "" if agent.verify_is_on() => format!("verify: {}", agent.verify_summary()),
                    "" => "verify: off (set one with /verify <cmd>)".to_string(),
                    "off" | "none" | "clear" | "disable" => {
                        agent.set_verify_command(None);
                        "verification disabled".to_string()
                    }
                    cmd => {
                        agent.set_verify_command(Some(cmd.to_string()));
                        format!(
                            "verification on: `{cmd}` — runs after each turn, iterates on failure"
                        )
                    }
                };
                self.push(Line::styled(msg, dim()));
            }
            Command::Diff => {
                let out = hi_tools::working_tree_diff().await;
                let text = out.into_text().unwrap_or_else(|_| Text::from(out.clone()));
                for line in text.lines {
                    self.push(line);
                }
            }
            Command::Commit => {
                let out = hi_tools::commit().await;
                for line in out.lines() {
                    self.push(Line::styled(format!("── {line} ──"), dim()));
                }
            }
            Command::Copy(arg) => self.copy(&arg),
            Command::Goal(arg) => self.handle_goal(agent, &arg),
            Command::Context => {
                let breakdown = agent.context_breakdown();
                for line in breakdown.lines() {
                    self.push(Line::styled(line.to_string(), dim()));
                }
            }
            // Handled in the event loop (async / runs a turn / needs config); never reach here.
            Command::Prompt(_)
            | Command::Compact(_)
            | Command::Retry
            | Command::Edit
            | Command::Undo
            | Command::Init
            | Command::Provider(_) => {}
            Command::Version => {
                self.push(Line::styled(format!("hi {}", hi_agent::VERSION), dim()));
            }
            Command::Mcp => {
                let Some(url) = self.mcp_url.clone() else {
                    self.push(Line::styled(
                        "no MCP URL configured for this provider".to_string(),
                        Style::default().fg(Color::Yellow),
                    ));
                    return;
                };
                self.push(Line::styled("contacting MCP endpoint…".to_string(), dim()));
                let result: Result<_, anyhow::Error> = async {
                    let client = hi_ai::PipeMcpClient::new(url, self.api_key.clone());
                    let (server, protocol) = client.server_info().await?;
                    let tools = client.tools_list().await?;
                    let models = client.list_models().await?;
                    Ok((server, protocol, tools, models))
                }
                .await;
                match result {
                    Ok((server, protocol, tools, models)) => {
                        let url = self.mcp_url.as_deref().unwrap_or("");
                        self.push(Line::styled(
                            format!("mcp_url:  {url}"),
                            dim(),
                        ));
                        self.push(Line::styled(format!("server:   {server}"), dim()));
                        self.push(Line::styled(format!("protocol: {protocol}"), dim()));
                        self.push(Line::styled("tools:", dim()));
                        for tool in &tools {
                            let title = tool.title.as_deref().unwrap_or("");
                            if title.is_empty() {
                                self.push(Line::styled(format!("  {}", tool.name), dim()));
                            } else {
                                self.push(Line::styled(
                                    format!("  {}  - {}", tool.name, title),
                                    dim(),
                                ));
                            }
                        }
                        self.push(Line::styled(
                            format!("models:   {}", models.len()),
                            dim(),
                        ));
                        if let Some(model) = models.iter().find(|m| m.id == self.model) {
                            let health = model.health().unwrap_or("available");
                            let provider = model
                                .provider_label
                                .as_deref()
                                .unwrap_or("Pipe");
                            self.push(Line::styled(
                                format!("current:  {} · {} · {}", model.id, provider, health),
                                dim(),
                            ));
                        }
                    }
                    Err(err) => {
                        self.push(Line::styled(
                            format!("mcp inspection failed: {err:#}"),
                            Style::default().fg(Color::Yellow),
                        ));
                    }
                }
            }
            Command::Export(arg) => {
                let path = if arg.trim().is_empty() {
                    "transcript.md"
                } else {
                    arg.trim()
                };
                let content = agent.export_markdown();
                let count = agent
                    .messages()
                    .iter()
                    .filter(|m| m.role != hi_ai::Role::System)
                    .count();
                match std::fs::write(path, &content) {
                    Ok(()) => self.push(Line::styled(
                        format!("exported {count} messages to {path}"),
                        dim(),
                    )),
                    Err(err) => self.push(Line::styled(
                        format!("export failed: {err}"),
                        Style::default().fg(Color::Yellow),
                    )),
                }
            }
            Command::Unknown(name) => {
                self.push(Line::styled(
                    format!("unknown command /{name}; try /help"),
                    dim(),
                ));
            }
        }
        self.follow();
    }
}
