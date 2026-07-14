//! Memory distillation and workspace snapshot caching: `update_memory`
//! (session lessons → `.hi/memory.md`) and the snapshot cache helpers used by
//! verify/turn-end checks.

use std::sync::Arc;

use hi_ai::{ChatRequest, Content, Message, RequestProfile, Role, StreamEvent, ToolMode};

use crate::compaction;
use crate::memory::{
    cap_memory, extract_corrections, global_memory_file, memory_file_at, memory_prompt,
    split_layers, strip_header, unreferenced_bullets, verify_grounded, write_memory,
};
use crate::snapshot::FileFingerprint;
use crate::transcript::repair_invalid_tool_call_arguments_in_messages;
use crate::{ReviewStatus, TurnStatus, Ui, VerificationStatus};

impl crate::Agent {
    /// Distill durable, reusable lessons from this session into the project memory
    /// file (`.hi/memory.md`), then load it as context next session. Re-derives the
    /// *whole* capped list from the current memory + this session, so stale or wrong
    /// facts fall out instead of accreting (self-correcting against poisoning).
    ///
    /// One chat-only model call. Best-effort: a provider/IO error is surfaced as a
    /// status, never fatal (it runs at quit). Like [`summarize`](Self::summarize) it
    /// builds a throwaway message vec and does NOT record into the session history.
    pub async fn update_memory(&mut self, ui: &mut dyn Ui) {
        self.update_memory_at(memory_file_at(self.runtime.root()), ui)
            .await;
    }

    /// See [`update_memory`](Self::update_memory); the path is a parameter so tests
    /// can redirect the write to a temp file (no global env/cwd state).
    pub(crate) async fn update_memory_at(&mut self, path: std::path::PathBuf, ui: &mut dyn Ui) {
        // Read both memory layers, stripping the schema header so the distiller
        // sees only the bullets (and tolerates a missing/legacy header).
        let existing = std::fs::read_to_string(&path).unwrap_or_default();
        let existing = strip_header(&existing);
        let global_path = global_memory_file();
        let global_existing = std::fs::read_to_string(&global_path).unwrap_or_default();
        let global_existing = strip_header(&global_existing);

        ui.status("distilling session memory…");

        // Elide bulky tool outputs — the memory distillation only needs to
        // understand what was done, not re-read command output verbatim. Skip
        // any leading system message explicitly (robust against message order),
        // rather than assuming index 0 is system.
        let all = self.messages.as_slice();
        let start = all
            .iter()
            .position(|m| m.role != Role::System)
            .unwrap_or(all.len());
        let mut history: Vec<Message> = all[start..].to_vec();
        // Bound the replay window so a very long session doesn't blow up the
        // throwaway distillation call: keep the most recent turns, which carry
        // the durable lessons worth recording.
        const MEMORY_REPLAY_MAX: usize = 40;
        let window_start = history.len().saturating_sub(MEMORY_REPLAY_MAX);
        if window_start > 0 {
            history.drain(..window_start);
        }
        // Self-learning enrichment: surface corrections and unreferenced facts
        // so the distiller focuses on the highest-signal material.
        let corrections = extract_corrections(&history);
        let transcript_text: String = history.iter().map(|m| m.text()).collect();
        let recalled = unreferenced_bullets(&existing, &transcript_text);

        // Assistant conclusions are eligible only from the most recent turn
        // when it completed against deterministic verification. Earlier turns
        // in a long session may have been incomplete or unverified, so do not
        // replay them into the distiller. Explicit user corrections/preferences
        // remain eligible independently.
        let verified_turn = self.last_turn_outcome.as_ref().is_some_and(|outcome| {
            outcome.status == TurnStatus::Completed
                && outcome.verification == VerificationStatus::Passed
                && outcome.review != ReviewStatus::Objected
        });
        if verified_turn {
            let turn_start = history.iter().rposition(|message| {
                message.role == Role::User && !message.text().trim_start().starts_with("[hi:nudge:")
            });
            if let Some(turn_start) = turn_start {
                history.drain(..turn_start);
            } else {
                history.clear();
            }
        } else {
            history.clear();
        }
        if history.is_empty() && corrections.trim().is_empty() {
            ui.status("(memory unchanged — no verifier-backed turn or user correction)");
            return;
        }
        let len = history.len();
        compaction::elide_tool_outputs(&mut history, len);

        let mut messages = Vec::with_capacity(history.len() + 2);
        messages.push(self.minimal_system_message());
        messages.extend_from_slice(&history);
        let mut prompt = memory_prompt(&existing, &global_existing, &corrections, &recalled);
        if !verified_turn {
            prompt = format!(
                "No verifier-backed assistant turn is available. Preserve existing facts and use ONLY explicit user corrections or preferences below; do not promote assistant claims.\n\n{prompt}"
            );
        }
        messages.push(Message::user(prompt));
        repair_invalid_tool_call_arguments_in_messages(&mut messages);

        let request = ChatRequest {
            model: self.config.model.clone(),
            messages: Arc::from(messages),
            tools: Arc::new([]), // distilling — no tool use
            max_tokens: 1024,    // throwaway call — memory notes are short
            temperature: self.config.temperature,
            top_p: None,
            frequency_penalty: None,
            thinking_budget: None,
            reasoning_effort: None,
            profile: RequestProfile {
                compat: self.config.compat,
                tool_mode: ToolMode::ChatOnly,
                stream_usage: None,
            },
        };

        let mut memory = String::new();
        let mut sink = |event: StreamEvent| match event {
            StreamEvent::Text(text) => {
                memory.push_str(&text);
                ui.assistant_text(&text);
            }
            StreamEvent::Status(text) => ui.status(&text),
            StreamEvent::Reasoning(_) => {}
        };
        let completion = match self.provider.stream(request, &mut sink).await {
            Ok(completion) => completion,
            Err(err) => {
                self.add_side_error_usage(&err);
                // Flush any partially-streamed memory text before the status.
                ui.assistant_end();
                let _ = self.persist();
                ui.status(&format!("(couldn't update memory: {err})"));
                return;
            }
        };
        self.add_side_usage(completion.usage);
        let _ = self.persist();

        // Fall back to the final content if the provider didn't stream text.
        // Emit it through the UI before assistant_end so the user sees the
        // distilled memory even when the provider returned text only in the
        // completion object (not via stream deltas).
        if memory.trim().is_empty() {
            for c in &completion.content {
                if let Content::Text(t) = c {
                    memory.push_str(t);
                    ui.assistant_text(t);
                }
            }
        }
        ui.assistant_end();
        let memory = cap_memory(&memory);
        if memory.is_empty() {
            return; // nothing durable to save
        }

        // Groundedness: drop distilled bullets that reference paths or commands
        // which don't resolve in the current workspace — a hallucinated path or
        // a stale build command is worse than no memory. Global-routed bullets
        // (global: prefix) are exempt since they may reference other projects.
        let memory = verify_grounded(&memory);
        if memory.trim().is_empty() {
            return; // nothing left after grounding
        }

        // Hierarchical routing: split `global:`-prefixed bullets out to the
        // user-level memory file; the rest stays in project memory.
        let (project_body, global_body) = split_layers(&memory);

        // Publish each layer atomically + exclusively (temp file + rename under
        // an O_EXCL lock). A concurrent distillation in the same dir is skipped;
        // its revision loses to whichever process took the lock first.
        let mut saved_notes = 0usize;
        let mut wrote_global = false;
        if !project_body.trim().is_empty() {
            match write_memory(&path, &project_body) {
                Ok(notes) => saved_notes += notes,
                Err(status) => ui.status(&format!("({status})")),
            }
        }
        if !global_body.trim().is_empty() {
            match write_memory(&global_path, &global_body) {
                Ok(notes) => {
                    saved_notes += notes;
                    wrote_global = true;
                }
                Err(status) => ui.status(&format!("({status})")),
            }
        }
        if saved_notes > 0 {
            let where_to = if wrote_global {
                "project + global memory"
            } else {
                "project memory"
            };
            ui.status(&format!(
                "✓ saved {saved_notes} memory note(s) to {where_to}"
            ));
        }
    }

    /// Get the workspace snapshot, using the cached version when available.
    /// The cache is valid until invalidated by [`invalidate_snapshot`].
    pub(crate) async fn snapshot_cached(
        &mut self,
    ) -> anyhow::Result<std::collections::BTreeMap<String, FileFingerprint>> {
        self.snapshot_cache.get(self.runtime.root()).await
    }

    /// Invalidate the snapshot cache — called after any mutating tool.
    pub(crate) fn invalidate_snapshot(&mut self) {
        self.snapshot_cache.invalidate();
    }
}
