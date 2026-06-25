//! System-prompt composition for the agent loop.

use hi_ai::Message;

/// Ending instruction when no separate finalization step runs: the model itself
/// must produce the closing recap.
const SELF_RECAP_INSTRUCTION: &str = " When the task is done, stop and end with a short recap so \
the user has the full picture: a one-line headline of what you accomplished, then — for any \
non-trivial change — a brief bullet list of the key edits (grouped by file) and the exact \
command(s) to run or test it. Write it in past tense, covering only what you actually did; don't \
restate the plan or pad it. For a trivial change or a plain question, a single line is enough.";

/// Ending instruction when a finalization step will write the recap: the model
/// shouldn't duplicate it, just confirm completion.
const DEFERRED_RECAP_INSTRUCTION: &str = " When the task is done, stop. A separate step will write \
the final summary for the user, so you don't need to compose a full recap yourself — just make \
sure the work is actually complete and finish with at most a one-line note.";

/// Builds a system message by composing optional sections.
///
/// Usage:
/// ```ignore
/// SystemPrompt::new()
///     .with_project_context(ctx)
///     .with_goal(goal)
///     .with_finalize(true)
///     .build()
/// ```
pub(crate) struct SystemPrompt {
    project_context: Option<String>,
    goal: Option<String>,
    goal_state: Option<String>,
    decisions: Option<String>,
    finalize: bool,
}

impl SystemPrompt {
    pub(crate) fn new() -> Self {
        Self {
            project_context: None,
            goal: None,
            goal_state: None,
            decisions: None,
            finalize: false,
        }
    }

    pub(crate) fn with_project_context(mut self, context: Option<&str>) -> Self {
        self.project_context = context
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        self
    }

    pub(crate) fn with_goal(mut self, goal: Option<&str>) -> Self {
        self.goal = goal.map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
        self
    }

    /// Inject the long-horizon goal state (already rendered, including its own
    /// header) — the objective, the sub-goal checklist with statuses, and any
    /// retry notes on the active sub-goal. Survives compaction (system message).
    pub(crate) fn with_goal_state(mut self, section: Option<&str>) -> Self {
        self.goal_state = section.map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
        self
    }

    /// Inject the durable decision-log section (already rendered, including its
    /// own header) so it survives compaction verbatim — it's part of the system
    /// message, which compaction preserves, not the summarizable history.
    pub(crate) fn with_decisions(mut self, section: Option<&str>) -> Self {
        self.decisions = section.map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
        self
    }

    pub(crate) fn with_finalize(mut self, finalize: bool) -> Self {
        self.finalize = finalize;
        self
    }

    pub(crate) fn build(self) -> Message {
        let mut text = super::SYSTEM_PROMPT.to_string();
        text.push_str(if self.finalize {
            DEFERRED_RECAP_INSTRUCTION
        } else {
            SELF_RECAP_INSTRUCTION
        });
        // Ground the model in its real location so it doesn't guess paths (a wrong
        // `/home/user`, scaffolding under `/tmp`, copying from directories that don't
        // exist) and wander out of the project. Each shell command runs from here in
        // a fresh shell, so `cd` never persists — say so explicitly.
        if let Ok(cwd) = std::env::current_dir() {
            text.push_str(&format!(
                "\n\nYour working directory is `{}` — work here. Every shell command runs from \
                 this directory in a fresh shell, so `cd` does NOT persist between commands. Use \
                 paths relative to it; do not `cd` into, copy from, or create directories elsewhere.",
                cwd.display()
            ));
        }
        if let Some(context) = self.project_context {
            text.push_str("\n\n");
            text.push_str(&context);
        }
        if let Some(goal) = self.goal {
            text.push_str("\n\n[Current session goal]\n");
            text.push_str(&goal);
        }
        if let Some(goal_state) = self.goal_state {
            text.push_str(&goal_state);
        }
        if let Some(decisions) = self.decisions {
            text.push_str(&decisions);
        }
        Message::system(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_prompt_grounds_the_working_directory() {
        // The model must be told where it actually is, so it doesn't invent paths
        // (e.g. /home/user), cd elsewhere, or scaffold a new project.
        let sys = SystemPrompt::new().build();
        let text = sys.text();
        let cwd = std::env::current_dir().unwrap().display().to_string();
        assert!(text.contains(&cwd), "names the working directory: {text}");
        assert!(
            text.contains("does NOT persist"),
            "warns that cd doesn't persist"
        );
    }
}
