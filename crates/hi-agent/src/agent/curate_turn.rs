//! Verifier-gated skill auto-curation — the ACE "evolving playbook" idea.
//!
//! After a turn PASSES verification, make one throwaway chat-only model call that
//! inspects that turn's trajectory and — only if it contains a genuinely reusable,
//! general technique — emits a `SKILL.md`, which we persist as a learned skill.
//! The verifier is the gate, so a weak local model can't poison the playbook: a
//! skill is only ever written for a turn the ground-truth check already accepted.
//! The prompt biases hard toward silence, and a per-session cap bounds spam.

use std::sync::Arc;

use hi_ai::{ChatRequest, Content, Message, RequestProfile, Role, StreamEvent, ToolMode};

use crate::Ui;
use crate::compaction;
use crate::skills::{self, skill_roots};
use crate::transcript::repair_invalid_tool_call_arguments_in_messages;

/// Cap on skills auto-curated per session, so a long run of verified turns can't
/// flood `.hi/skills/` even if each one tempts a lesson.
pub(crate) const MAX_AUTO_SKILLS_PER_SESSION: u32 = 3;

/// Replay window (recent messages of the just-passed turn) handed to the curator.
const CURATE_REPLAY_MAX: usize = 30;

impl crate::Agent {
    /// After a verified turn, distill any reusable technique into a learned skill.
    ///
    /// Best-effort: a provider/IO error is surfaced as a status, never fatal. The
    /// caller gates on the success signal (`last_verify == Some(true)` + changed
    /// files), `config.curate_skills`, and the per-session cap; this method also
    /// re-checks the cap defensively. Like [`update_memory`](Self::update_memory)
    /// it builds a throwaway message vec and does NOT record into session history.
    pub(crate) async fn curate_turn_end(&mut self, turn_start: usize, ui: &mut dyn Ui) {
        if self.auto_skills_written >= MAX_AUTO_SKILLS_PER_SESSION {
            return;
        }
        // Just this turn's trajectory (user prompt → tool calls → results), with
        // bulky tool outputs elided — the curator needs the shape of what worked,
        // not the verbatim command output.
        let all = self.messages.as_slice();
        if turn_start >= all.len() {
            return;
        }
        let mut history: Vec<Message> = all[turn_start..]
            .iter()
            .filter(|m| m.role != Role::System)
            .cloned()
            .collect();
        let window_start = history.len().saturating_sub(CURATE_REPLAY_MAX);
        if window_start > 0 {
            history.drain(..window_start);
        }
        if history.is_empty() {
            return;
        }
        let len = history.len();
        compaction::elide_tool_outputs(&mut history, len);

        let existing = skills::learned_skills_context().unwrap_or_default();

        let mut messages = Vec::with_capacity(history.len() + 2);
        messages.push(self.minimal_system_message());
        messages.extend_from_slice(&history);
        messages.push(Message::user(curate_prompt(&existing)));
        repair_invalid_tool_call_arguments_in_messages(&mut messages);

        let request = ChatRequest {
            model: self.config.model.clone(),
            user_turn: false,
            canonical_objective: None,
            messages: Arc::from(messages),
            tools: Arc::new([]), // curating — no tool use
            max_tokens: 512,     // a skill is short
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

        ui.status("curating skill…");
        let mut out = String::new();
        let mut sink = |event: StreamEvent| match event {
            StreamEvent::Text(text) => out.push_str(&text),
            StreamEvent::Status(text) => ui.status(&text),
            StreamEvent::Reasoning(_) => {}
        };
        let completion = match self.provider.stream(request, &mut sink).await {
            Ok(completion) => completion,
            Err(err) => {
                self.add_side_error_usage(&err);
                ui.status(&format!("(couldn't curate skill: {err})"));
                return;
            }
        };
        self.add_side_usage(completion.usage);
        let _ = self.persist();
        if out.trim().is_empty() {
            for c in &completion.content {
                if let Content::Text(text) = c {
                    out.push_str(text);
                }
            }
        }

        // Silence bias: the model emits nothing (or no valid frontmatter) when
        // there's no general lesson — only persist a well-formed SKILL.md.
        let Some(skill) = parse_skill_markdown(&out) else {
            return;
        };
        match skills::write_skill(
            &skill_roots(),
            &skill.scope,
            &skill.name,
            &skill.description,
            &skill.body,
        ) {
            Ok(Some(path)) => {
                self.auto_skills_written += 1;
                ui.status(&format!("✓ curated skill → {}", path.display()));
            }
            Ok(None) => {} // a skill by this name already exists
            Err(err) => ui.status(&format!("(skill not saved: {err})")),
        }
    }
}

struct ParsedSkill {
    name: String,
    description: String,
    scope: String,
    body: String,
}

/// Curation prompt. Heavily biased toward outputting nothing, so a routine turn
/// with no transferable lesson doesn't produce a junk skill.
fn curate_prompt(existing_index: &str) -> String {
    let existing = if existing_index.trim().is_empty() {
        String::new()
    } else {
        format!("\n\nSkills already recorded (do NOT duplicate any of these):\n{existing_index}")
    };
    format!(
        "The task above was just verified as correct by an automated check.\n\n\
         If — and ONLY if — the trajectory contains a REUSABLE, GENERAL technique that would help on \
         FUTURE, DIFFERENT tasks (not tied to this task's specific files, names, or values), record it \
         as a learned skill. Most turns have no such lesson; in that case output NOTHING AT ALL.\n\n\
         When there IS a genuine transferable technique, output exactly one SKILL.md and nothing else, \
         in this exact format:\n\
         ---\n\
         name: <short Title Case name>\n\
         description: <one sentence: when to use it>\n\
         scope: project\n\
         ---\n\
         # <name>\n\n\
         <a few lines: when to use it, the procedure, and how to verify>\n\n\
         Rules: describe the transferable METHOD, never this specific task. Use scope `global` only if \
         the technique is repo-independent. Keep the whole file under ~30 lines. If in any doubt, output \
         nothing.{existing}"
    )
}

/// Extract a `SKILL.md` (frontmatter + body) from model output that may have
/// surrounding prose or a code fence. Returns `None` unless a `---`-delimited
/// frontmatter block with a non-empty `name` is present (the silence path).
fn parse_skill_markdown(text: &str) -> Option<ParsedSkill> {
    let start = text.find("---")?;
    let lines: Vec<&str> = text[start..].lines().collect();
    if lines.first()?.trim() != "---" {
        return None;
    }
    let (mut name, mut description, mut scope) = (None, None, "project".to_string());
    let mut close = None;
    for (i, line) in lines.iter().enumerate().skip(1) {
        let trimmed = line.trim();
        if trimmed == "---" {
            close = Some(i);
            break;
        }
        if let Some((key, value)) = trimmed.split_once(':') {
            let value = value.trim().trim_matches('"').trim_matches('\'').trim();
            match key.trim() {
                "name" if !value.is_empty() => name = Some(value.to_string()),
                "description" if !value.is_empty() => description = Some(value.to_string()),
                "scope" if !value.is_empty() => scope = value.to_string(),
                _ => {}
            }
        }
    }
    let name = name?;
    let close = close?;
    let body = lines[close + 1..]
        .join("\n")
        .trim()
        .trim_end_matches("```")
        .trim()
        .to_string();
    Some(ParsedSkill {
        name,
        description: description.unwrap_or_default(),
        scope,
        body,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_skill_with_prose_and_fence() {
        let out = "Sure, here is a reusable technique:\n\n\
                   ```markdown\n\
                   ---\n\
                   name: Bisect A Flaky Test\n\
                   description: Narrow a flaky test by bisecting recent commits.\n\
                   scope: global\n\
                   ---\n\
                   # Bisect A Flaky Test\n\n\
                   Run git bisect with the test as the check.\n\
                   ```\n";
        let skill = parse_skill_markdown(out).unwrap();
        assert_eq!(skill.name, "Bisect A Flaky Test");
        assert_eq!(skill.scope, "global");
        assert!(skill.description.starts_with("Narrow a flaky test"));
        assert!(skill.body.contains("git bisect"));
        assert!(!skill.body.contains("```"));
    }

    #[test]
    fn silence_when_no_frontmatter() {
        assert!(parse_skill_markdown("").is_none());
        assert!(parse_skill_markdown("No reusable lesson here.").is_none());
        // A bare horizontal rule with no `name:` is not a skill.
        assert!(parse_skill_markdown("---\njust text\n---").is_none());
    }
}
