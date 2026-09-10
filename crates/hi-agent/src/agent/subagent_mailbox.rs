//! Follow-up messages for owned subagents (`send_subagent_message`).
//!
//! Running children share an [`InterjectionInbox`] so a parent can steer the
//! current turn at the next safe point. Finished children are kept so a later
//! message can wake them as a continuation task (grok-build coordinator wake).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::InterjectionInbox;

#[derive(Clone, Debug)]
pub(crate) struct FinishedSubagent {
    pub kind: String,
    pub last_prompt: String,
    pub last_output: String,
    pub attempt_id: String,
}

#[derive(Clone, Default)]
pub(crate) struct SubagentMailbox {
    running: Arc<Mutex<HashMap<String, InterjectionInbox>>>,
    finished: Arc<Mutex<HashMap<String, FinishedSubagent>>>,
}

#[derive(Debug)]
pub(crate) enum SendSubagentResult {
    Accepted {
        message_id: String,
    },
    NotFound,
    Finished {
        kind: String,
        last_prompt: String,
        last_output: String,
    },
}

impl SubagentMailbox {
    pub(crate) fn register_running(&self, id: &str, inbox: InterjectionInbox) {
        if let Ok(mut running) = self.running.lock() {
            running.insert(id.to_string(), inbox);
        }
        if let Ok(mut finished) = self.finished.lock() {
            finished.remove(id);
        }
    }

    pub(crate) fn mark_finished(
        &self,
        id: &str,
        kind: String,
        last_prompt: String,
        last_output: String,
    ) {
        if let Ok(mut running) = self.running.lock() {
            running.remove(id);
        }
        if let Ok(mut finished) = self.finished.lock() {
            finished.insert(
                id.to_string(),
                FinishedSubagent {
                    kind,
                    last_prompt,
                    last_output,
                    attempt_id: uuid::Uuid::new_v4().to_string(),
                },
            );
        }
    }

    pub(crate) fn send(&self, id: &str, text: &str) -> SendSubagentResult {
        if let Ok(running) = self.running.lock()
            && let Some(inbox) = running.get(id)
        {
            inbox.push(text);
            return SendSubagentResult::Accepted {
                message_id: format!("{id}:{}", inbox.pending().len()),
            };
        }
        if let Ok(finished) = self.finished.lock()
            && let Some(child) = finished.get(id)
        {
            return SendSubagentResult::Finished {
                kind: child.kind.clone(),
                last_prompt: child.last_prompt.clone(),
                last_output: child.last_output.clone(),
            };
        }
        SendSubagentResult::NotFound
    }

    pub(crate) fn snapshot_lines(&self) -> Vec<String> {
        let mut lines = Vec::new();
        if let Ok(running) = self.running.lock() {
            for id in running.keys() {
                lines.push(format!("- {id} (running)"));
            }
        }
        if let Ok(finished) = self.finished.lock() {
            for (id, child) in finished.iter() {
                lines.push(format!(
                    "- {id} (finished {}, attempt {}, wake with send_subagent_message)",
                    child.kind, child.attempt_id
                ));
            }
        }
        lines
    }

    /// One model-facing block for finished children (prompt + clipped output).
    pub(crate) fn wake_digest(&self) -> Option<String> {
        let finished = self.finished.lock().ok()?;
        if finished.is_empty() {
            return None;
        }
        let mut out = String::from("[Finished-child wake digest]\n");
        for (id, child) in finished.iter() {
            let prompt = clip(&child.last_prompt, 120);
            let output = clip(&child.last_output, 400);
            out.push_str(&format!(
                "- {id} ({}, attempt {}): {prompt}\n  {output}\n",
                child.kind, child.attempt_id
            ));
        }
        Some(out.trim_end().to_string())
    }
}

fn clip(text: &str, max: usize) -> String {
    let t = text.trim();
    if t.chars().count() <= max {
        return t.to_string();
    }
    let clipped: String = t.chars().take(max.saturating_sub(1)).collect();
    format!("{clipped}…")
}

impl crate::Agent {
    pub(crate) async fn handle_send_subagent_message(
        &mut self,
        arguments: &str,
        ui: &mut dyn crate::Ui,
    ) -> hi_tools::ToolOutcome {
        let parsed = match serde_json::from_str::<serde_json::Value>(arguments) {
            Ok(v) => v,
            Err(_) => {
                return tool_outcome(
                    "send_subagent_message error: invalid JSON arguments",
                    hi_tools::ToolStatus::Failed,
                );
            }
        };
        let id = parsed
            .get("subagent_id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim();
        let text = parsed
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim();
        if id.is_empty() || text.is_empty() {
            return tool_outcome(
                "send_subagent_message error: subagent_id and text are required",
                hi_tools::ToolStatus::Failed,
            );
        }
        match self.subagent_mailbox.send(id, text) {
            SendSubagentResult::Accepted { message_id } => {
                ui.status(&format!("steered subagent {id}"));
                tool_outcome(
                    format!("Message accepted (message_id: {message_id})."),
                    hi_tools::ToolStatus::Succeeded,
                )
            }
            SendSubagentResult::NotFound => tool_outcome(
                format!("Subagent `{id}` not found or not owned by this session."),
                hi_tools::ToolStatus::Failed,
            ),
            SendSubagentResult::Finished {
                kind,
                last_prompt,
                last_output,
            } => {
                let follow = format!(
                    "You previously completed this task:\n\n{last_prompt}\n\nPrevious result:\n{last_output}\n\nFollow-up from the parent:\n{text}"
                );
                let spawn = serde_json::json!({
                    "description": format!("wake {id}"),
                    "prompt": follow,
                    "subagent_type": kind,
                });
                let outcome = self.handle_task(&spawn.to_string(), ui).await;
                let mut content = format!("Woke finished subagent `{id}` as a continuation.\n");
                content.push_str(&outcome.content);
                let mut woken = outcome;
                woken.content = content;
                woken
            }
        }
    }
}

fn tool_outcome(content: impl Into<String>, status: hi_tools::ToolStatus) -> hi_tools::ToolOutcome {
    hi_tools::ToolOutcome {
        content: content.into(),
        display: None,
        plan: None,
        status,
        process: None,
        background: None,
        effects: hi_tools::ToolEffects::default(),
        truncation: hi_tools::TruncationState::Complete,
        images: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn running_child_accepts_follow_up() {
        let mailbox = SubagentMailbox::default();
        let inbox = InterjectionInbox::default();
        mailbox.register_running("task_1", inbox.clone());
        match mailbox.send("task_1", "please also check tests") {
            SendSubagentResult::Accepted { .. } => {}
            other => panic!("expected accepted, got {other:?}"),
        }
        assert_eq!(inbox.drain(), vec!["please also check tests"]);
    }

    #[test]
    fn finished_child_can_be_woken() {
        let mailbox = SubagentMailbox::default();
        mailbox.mark_finished(
            "explore-1",
            "explore".into(),
            "find TODOs".into(),
            "none found".into(),
        );
        match mailbox.send("explore-1", "look in src/") {
            SendSubagentResult::Finished {
                kind, last_output, ..
            } => {
                assert_eq!(kind, "explore");
                assert_eq!(last_output, "none found");
            }
            other => panic!("expected finished, got {other:?}"),
        }
        let digest = mailbox.wake_digest().expect("digest");
        assert!(digest.contains("explore-1"));
        assert!(digest.contains("attempt"));
        assert!(digest.contains("none found"));
    }
}
