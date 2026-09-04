//! Reducer-backed transcript identities for presentation clients.
//!
//! Live UI events continue through the established renderer. When the v2
//! feature gate is enabled, this module reduces the same events into a shadow
//! projection and assigns stable block identities. An explicitly supplied
//! snapshot or exact-base patch is different: it has already passed the
//! reducer's integrity checks, so the TUI may rebuild its view from it. The
//! rendered widgets never feed state back into the reducer.

use std::collections::VecDeque;
use std::time::{Duration, SystemTime};

use hi_agent::{
    SessionEvent, SessionEventKind, SessionProjection, SessionProjectionPatch,
    SessionProjectionSnapshot, TranscriptBlock, TranscriptBlockId, TranscriptBlockKind,
    TranscriptBlockLifecycle, TranscriptBlockTerminal,
};
use ratatui::text::Line;

use crate::event::UiEvent;
use crate::{App, TranscriptEntry};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ProjectedBlockIdentity {
    pub id: String,
    pub terminal: Option<&'static str>,
}

pub(crate) struct PresentationProjection {
    enabled: bool,
    projection: SessionProjection,
    active_assistant: Option<TranscriptBlockId>,
    active_reasoning: Option<TranscriptBlockId>,
    active_tools: VecDeque<(String, TranscriptBlockId)>,
    last_patch: Option<blake3::Hash>,
    last_error: Option<String>,
}

impl Default for PresentationProjection {
    fn default() -> Self {
        Self {
            enabled: false,
            projection: SessionProjection::new(),
            active_assistant: None,
            active_reasoning: None,
            active_tools: VecDeque::new(),
            last_patch: None,
            last_error: None,
        }
    }
}

impl PresentationProjection {
    fn configure(&mut self, enabled: bool) {
        if enabled == self.enabled {
            return;
        }
        *self = Self {
            enabled,
            ..Self::default()
        };
    }

    fn snapshot(&self) -> SessionProjectionSnapshot {
        self.projection.snapshot()
    }

    fn prepare_patch(&self, events: Vec<SessionEvent>) -> Result<SessionProjectionPatch, String> {
        self.projection
            .prepare_patch(events)
            .map_err(|error| error.to_string())
    }

    fn install_snapshot(&mut self, snapshot: SessionProjectionSnapshot) -> Result<(), String> {
        let projection =
            SessionProjection::from_snapshot(snapshot).map_err(|error| error.to_string())?;
        self.enabled = true;
        self.projection = projection;
        self.last_patch = None;
        self.last_error = None;
        self.rebuild_open_blocks();
        Ok(())
    }

    fn apply_patch(&mut self, patch: SessionProjectionPatch) -> Result<bool, String> {
        let encoded = serde_json::to_vec(&patch).map_err(|error| error.to_string())?;
        let fingerprint = blake3::hash(&encoded);
        let current = self.projection.snapshot();
        if patch.target_revision == current.revision
            && patch.target_digest == current.digest
            && self.last_patch == Some(fingerprint)
        {
            return Ok(false);
        }
        self.projection
            .apply_patch(patch)
            .map_err(|error| error.to_string())?;
        self.enabled = true;
        self.last_patch = Some(fingerprint);
        self.last_error = None;
        self.rebuild_open_blocks();
        Ok(true)
    }

    fn rebuild_open_blocks(&mut self) {
        self.active_assistant = None;
        self.active_reasoning = None;
        self.active_tools.clear();
        for block in &self.projection.reducer().state().transcript_blocks {
            if block.lifecycle.is_terminal() {
                continue;
            }
            match block.kind {
                TranscriptBlockKind::Assistant => self.active_assistant = Some(block.id.clone()),
                TranscriptBlockKind::Reasoning => self.active_reasoning = Some(block.id.clone()),
                TranscriptBlockKind::Tool => self
                    .active_tools
                    .push_back((tool_name(&block.content), block.id.clone())),
                _ => {}
            }
        }
    }

    fn observe(&mut self, event: &UiEvent) -> Result<(), String> {
        if !self.enabled {
            return Ok(());
        }
        let result = self.observe_enabled(event);
        if let Err(error) = &result {
            self.last_error = Some(error.clone());
        }
        result
    }

    fn observe_enabled(&mut self, event: &UiEvent) -> Result<(), String> {
        match event {
            UiEvent::Text { text } => {
                self.settle_reasoning(TranscriptBlockTerminal::Completed)?;
                let active = self.active_assistant.clone();
                let id = active
                    .clone()
                    .unwrap_or_else(|| self.synthetic_id("assistant"));
                let event = if active.is_some() {
                    SessionEventKind::TranscriptBlockAppended {
                        block_id: id.clone(),
                        delta: text.clone(),
                    }
                } else {
                    SessionEventKind::TranscriptBlockOpened {
                        block_id: id.clone(),
                        kind: TranscriptBlockKind::Assistant,
                        content: text.clone(),
                    }
                };
                self.transact(vec![event])?;
                self.active_assistant = Some(id);
            }
            UiEvent::Reasoning { text } => {
                let active = self.active_reasoning.clone();
                let id = active
                    .clone()
                    .unwrap_or_else(|| self.synthetic_id("reasoning"));
                let event = if active.is_some() {
                    SessionEventKind::TranscriptBlockAppended {
                        block_id: id.clone(),
                        delta: text.clone(),
                    }
                } else {
                    SessionEventKind::TranscriptBlockOpened {
                        block_id: id.clone(),
                        kind: TranscriptBlockKind::Reasoning,
                        content: text.clone(),
                    }
                };
                self.transact(vec![event])?;
                self.active_reasoning = Some(id);
            }
            UiEvent::AssistantEnd => self.settle_narrative(TranscriptBlockTerminal::Completed)?,
            UiEvent::ToolCall { name, arguments } => {
                self.open_tool(name, arguments)?;
            }
            UiEvent::ToolResult { name, result } => {
                self.settle_tool(name, result, TranscriptBlockTerminal::Completed)?;
            }
            UiEvent::ToolStream { line, .. } => {
                if let Some((_, id)) = self.active_tools.back() {
                    self.transact(vec![SessionEventKind::TranscriptBlockAppended {
                        block_id: id.clone(),
                        delta: format!("\n{line}"),
                    }])?;
                }
            }
            UiEvent::TurnEnd { .. } => {
                self.settle_narrative(TranscriptBlockTerminal::Completed)?;
                self.settle_tools(TranscriptBlockTerminal::Failed)?;
            }
            UiEvent::TurnError { .. } => self.settle_all(TranscriptBlockTerminal::Failed)?,
            _ => {}
        }
        Ok(())
    }

    fn record_user_prompt(&mut self, content: String) -> Result<(), String> {
        if !self.enabled || content.trim().is_empty() {
            return Ok(());
        }
        let id = self.synthetic_id("user");
        self.transact(vec![SessionEventKind::TranscriptBlockRecorded {
            block_id: id,
            kind: TranscriptBlockKind::UserPrompt,
            content,
            terminal: TranscriptBlockTerminal::Completed,
        }])
    }

    fn open_tool(&mut self, name: &str, arguments: &str) -> Result<(), String> {
        self.settle_narrative(TranscriptBlockTerminal::Completed)?;
        let id = self.synthetic_id("tool");
        let content = tool_call_content(name, arguments);
        self.transact(vec![SessionEventKind::TranscriptBlockOpened {
            block_id: id.clone(),
            kind: TranscriptBlockKind::Tool,
            content,
        }])?;
        self.active_tools.push_back((name.to_owned(), id));
        Ok(())
    }

    fn settle_tool(
        &mut self,
        name: &str,
        result: &str,
        terminal: TranscriptBlockTerminal,
    ) -> Result<(), String> {
        let expected = tool_result_content(name, result);
        let id = self
            .active_tools
            .iter()
            .find(|(active_name, _)| active_name == name)
            .map(|(_, id)| id.clone())
            .or_else(|| self.active_tools.front().map(|(_, id)| id.clone()));
        let Some(id) = id else {
            let id = self.synthetic_id("tool");
            self.transact(vec![SessionEventKind::TranscriptBlockRecorded {
                block_id: id,
                kind: TranscriptBlockKind::Tool,
                content: expected,
                terminal,
            }])?;
            return Ok(());
        };
        self.transact(vec![
            SessionEventKind::TranscriptBlockReplaced {
                block_id: id.clone(),
                content: expected,
            },
            SessionEventKind::TranscriptBlockSettled {
                block_id: id.clone(),
                terminal,
            },
        ])?;
        self.active_tools.retain(|(_, active_id)| active_id != &id);
        Ok(())
    }

    fn settle_narrative(&mut self, terminal: TranscriptBlockTerminal) -> Result<(), String> {
        self.settle_reasoning(terminal)?;
        if let Some(id) = self.active_assistant.clone() {
            self.transact(vec![SessionEventKind::TranscriptBlockSettled {
                block_id: id,
                terminal,
            }])?;
            self.active_assistant = None;
        }
        Ok(())
    }

    fn settle_reasoning(&mut self, terminal: TranscriptBlockTerminal) -> Result<(), String> {
        if let Some(id) = self.active_reasoning.clone() {
            self.transact(vec![SessionEventKind::TranscriptBlockSettled {
                block_id: id,
                terminal,
            }])?;
            self.active_reasoning = None;
        }
        Ok(())
    }

    fn settle_all(&mut self, terminal: TranscriptBlockTerminal) -> Result<(), String> {
        self.settle_narrative(terminal)?;
        self.settle_tools(terminal)
    }

    fn settle_tools(&mut self, terminal: TranscriptBlockTerminal) -> Result<(), String> {
        let tools = self
            .active_tools
            .iter()
            .map(|(_, id)| SessionEventKind::TranscriptBlockSettled {
                block_id: id.clone(),
                terminal,
            })
            .collect::<Vec<_>>();
        if !tools.is_empty() {
            self.transact(tools)?;
            self.active_tools.clear();
        }
        Ok(())
    }

    fn transact(&mut self, kinds: Vec<SessionEventKind>) -> Result<(), String> {
        if kinds.is_empty() {
            return Ok(());
        }
        let events = kinds.into_iter().map(SessionEvent::new).collect();
        let patch = self
            .projection
            .prepare_patch(events)
            .map_err(|error| error.to_string())?;
        self.projection
            .apply_patch(patch)
            .map_err(|error| error.to_string())?;
        Ok(())
    }

    fn synthetic_id(&self, kind: &str) -> TranscriptBlockId {
        TranscriptBlockId::new(format!(
            "tui.{kind}.{}",
            self.projection.snapshot().revision.saturating_add(1)
        ))
        .expect("synthetic transcript IDs use a fixed safe alphabet")
    }

    fn identities_for(
        &self,
        transcript: &[TranscriptEntry],
    ) -> Vec<Option<ProjectedBlockIdentity>> {
        let mut identities = vec![None; transcript.len()];
        if !self.enabled {
            return identities;
        }
        let blocks = &self.projection.reducer().state().transcript_blocks;
        let mut before = blocks.len();
        for (entry_index, entry) in transcript.iter().enumerate().rev() {
            let Some(kind) = entry_kind(entry) else {
                continue;
            };
            let Some(block_index) = blocks[..before]
                .iter()
                .rposition(|block| block.kind == kind)
            else {
                continue;
            };
            let block = &blocks[block_index];
            identities[entry_index] = Some(ProjectedBlockIdentity {
                id: block.id.to_string(),
                terminal: terminal_name(&block.lifecycle),
            });
            before = block_index;
        }
        identities
    }
}

impl App {
    pub(crate) fn configure_session_projection_v2(&mut self, enabled: bool) {
        self.session_projection.configure(enabled);
    }

    pub(crate) fn reset_session_projection_v2(&mut self) {
        let enabled = self.session_projection.enabled;
        self.session_projection = PresentationProjection {
            enabled,
            ..PresentationProjection::default()
        };
        self.transcript.clear();
    }

    pub(crate) fn apply(&mut self, event: UiEvent) {
        if let Err(error) = self.try_apply(event) {
            tracing::warn!(%error, "session projection v2 rejected a UI lifecycle event");
            self.event_log
                .push(format!("session_projection_rejected {error}"));
        }
    }

    pub(crate) fn try_apply(&mut self, event: UiEvent) -> Result<(), String> {
        self.session_projection.observe(&event)?;
        self.apply_legacy(event);
        Ok(())
    }

    pub(crate) fn record_projected_user_prompt(&mut self, line: &Line<'_>) {
        let content = crate::render::line_text(line);
        let content = content
            .trim_start()
            .strip_prefix('❯')
            .or_else(|| content.trim_start().strip_prefix('>'))
            .unwrap_or(&content)
            .trim_start()
            .to_owned();
        if let Err(error) = self.session_projection.record_user_prompt(content) {
            tracing::warn!(%error, "session projection v2 rejected a user prompt block");
            self.event_log
                .push(format!("session_projection_rejected {error}"));
        }
    }

    pub(crate) fn session_projection_snapshot(&self) -> SessionProjectionSnapshot {
        self.session_projection.snapshot()
    }

    pub(crate) fn prepare_session_projection_patch(
        &self,
        events: Vec<SessionEvent>,
    ) -> Result<SessionProjectionPatch, String> {
        self.session_projection.prepare_patch(events)
    }

    pub(crate) fn apply_session_projection_patch(
        &mut self,
        patch: SessionProjectionPatch,
    ) -> Result<(), String> {
        if self.session_projection.apply_patch(patch)? {
            self.sync_session_projection_view();
        }
        Ok(())
    }

    pub(crate) fn install_session_projection_snapshot(
        &mut self,
        snapshot: SessionProjectionSnapshot,
    ) -> Result<(), String> {
        self.session_projection.install_snapshot(snapshot)?;
        self.sync_session_projection_view();
        Ok(())
    }

    pub(crate) fn projected_transcript_identities(&self) -> Vec<Option<ProjectedBlockIdentity>> {
        self.session_projection.identities_for(&self.transcript)
    }

    pub(crate) fn sync_session_projection_view(&mut self) {
        let state = self.session_projection.projection.reducer().state().clone();
        self.clear_projected_transcript_view();
        self.goal = state.goal;
        self.plan = state.plan;
        self.plan_drive_paused = state.plan_drive_paused;
        self.usage = (state.usage.input_tokens, state.usage.output_tokens);
        self.usage_estimated = state.usage.estimated;
        self.session_totals = state.usage;
        if !state.transcript_blocks.is_empty() {
            for block in state.transcript_blocks {
                self.push_projected_block(block);
            }
        } else {
            self.push_projected_messages(state.messages);
        }
        self.cap_transcript();
        self.follow();
    }

    fn push_projected_block(&mut self, block: TranscriptBlock) {
        let content = block.content;
        match block.kind {
            TranscriptBlockKind::UserPrompt => {
                self.transcript.push(TranscriptEntry::UserPrompt {
                    line: Line::raw(format!("❯ {content}")),
                    at: SystemTime::now(),
                });
            }
            TranscriptBlockKind::Assistant => self
                .transcript
                .push(TranscriptEntry::AssistantMessage { text: content }),
            TranscriptBlockKind::Reasoning => self.transcript.push(TranscriptEntry::Reasoning {
                text: content,
                elapsed: Duration::ZERO,
            }),
            TranscriptBlockKind::Tool => {
                let text = tool_payload(&content);
                let mut body = text
                    .lines()
                    .map(|line| Line::raw(line.to_owned()))
                    .collect::<Vec<_>>();
                if body.is_empty() {
                    body.push(Line::raw(""));
                }
                self.transcript.push(TranscriptEntry::ToolOutput {
                    body,
                    expanded: false,
                });
            }
            TranscriptBlockKind::Workflow
            | TranscriptBlockKind::Activity
            | TranscriptBlockKind::Notice => {
                self.transcript
                    .push(TranscriptEntry::Line(Line::raw(content)));
            }
        }
        self.bump_transcript();
    }

    fn push_projected_messages(&mut self, messages: Vec<hi_ai::Message>) {
        let mut tool_names = std::collections::BTreeMap::<String, String>::new();
        for message in messages {
            match message.role {
                hi_ai::Role::System => {}
                hi_ai::Role::User => {
                    let mut text = message.text();
                    if message
                        .content
                        .iter()
                        .any(|content| matches!(content, hi_ai::Content::Image { .. }))
                    {
                        text.push_str(if text.is_empty() {
                            "[image]"
                        } else {
                            "\n[image]"
                        });
                    }
                    if !text.trim().is_empty() {
                        self.transcript.push(TranscriptEntry::UserPrompt {
                            line: Line::raw(format!("❯ {text}")),
                            at: SystemTime::now(),
                        });
                        self.bump_transcript();
                    }
                }
                hi_ai::Role::Assistant => {
                    for content in message.content {
                        match content {
                            hi_ai::Content::Text(text) => {
                                self.apply_legacy(UiEvent::Text { text });
                            }
                            hi_ai::Content::Thinking { text, .. } => {
                                self.apply_legacy(UiEvent::Reasoning { text });
                            }
                            hi_ai::Content::ToolCall {
                                id,
                                name,
                                arguments,
                            } => {
                                tool_names.insert(id, name.clone());
                                self.apply_legacy(UiEvent::ToolCall { name, arguments });
                            }
                            hi_ai::Content::ToolResult { .. } | hi_ai::Content::Image { .. } => {}
                        }
                    }
                    self.apply_legacy(UiEvent::AssistantEnd);
                }
                hi_ai::Role::Tool => {
                    for content in message.content {
                        if let hi_ai::Content::ToolResult { call_id, output } = content {
                            let name = tool_names
                                .get(&call_id)
                                .cloned()
                                .unwrap_or_else(|| "tool".to_owned());
                            self.apply_legacy(UiEvent::ToolResult {
                                name,
                                result: output,
                            });
                        }
                    }
                }
            }
        }
    }

    fn clear_projected_transcript_view(&mut self) {
        self.transcript.clear();
        self.pending = None;
        self.reasoning_buffer.clear();
        self.reasoning_started = None;
        self.current_assistant.clear();
        self.current_assistant_streamed_bytes = 0;
        self.assistant_message_open = false;
        self.current_tool = None;
        self.current_tool_started = None;
        self.event_log.clear();
        self.last_assistant.clear();
        self.block_cursor = 0;
        self.scroll = 0;
        self.following = true;
        self.bump_transcript();
    }
}

fn entry_kind(entry: &TranscriptEntry) -> Option<TranscriptBlockKind> {
    match entry {
        TranscriptEntry::UserPrompt { .. } => Some(TranscriptBlockKind::UserPrompt),
        TranscriptEntry::Assistant(_) | TranscriptEntry::AssistantMessage { .. } => {
            Some(TranscriptBlockKind::Assistant)
        }
        TranscriptEntry::Reasoning { .. } => Some(TranscriptBlockKind::Reasoning),
        TranscriptEntry::Activity(_) | TranscriptEntry::ToolOutput { .. } => {
            Some(TranscriptBlockKind::Tool)
        }
        TranscriptEntry::Workflow { .. } => Some(TranscriptBlockKind::Workflow),
        TranscriptEntry::Line(_) | TranscriptEntry::Btw { .. } => None,
    }
}

fn terminal_name(lifecycle: &TranscriptBlockLifecycle) -> Option<&'static str> {
    match lifecycle {
        TranscriptBlockLifecycle::Open => None,
        TranscriptBlockLifecycle::Settled { terminal, .. } => Some(match terminal {
            TranscriptBlockTerminal::Completed => "completed",
            TranscriptBlockTerminal::Failed => "failed",
            TranscriptBlockTerminal::Cancelled => "cancelled",
            TranscriptBlockTerminal::Superseded => "superseded",
        }),
    }
}

fn tool_call_content(name: &str, arguments: &str) -> String {
    serde_json::json!({"phase": "call", "name": name, "payload": arguments}).to_string()
}

fn tool_result_content(name: &str, result: &str) -> String {
    serde_json::json!({"phase": "result", "name": name, "payload": result}).to_string()
}

fn tool_name(content: &str) -> String {
    serde_json::from_str::<serde_json::Value>(content)
        .ok()
        .and_then(|value| {
            value
                .get("name")
                .and_then(|name| name.as_str())
                .map(str::to_owned)
        })
        .unwrap_or_else(|| "tool".to_owned())
}

fn tool_payload(content: &str) -> String {
    serde_json::from_str::<serde_json::Value>(content)
        .ok()
        .and_then(|value| {
            value
                .get("payload")
                .and_then(|payload| payload.as_str())
                .map(str::to_owned)
        })
        .unwrap_or_else(|| content.to_owned())
}
