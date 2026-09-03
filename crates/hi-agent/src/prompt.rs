//! System-prompt composition for the agent loop.

use hi_ai::Message;

/// Ending instruction when no separate finalization step runs: the model itself
/// must produce the closing recap.
const SELF_RECAP_INSTRUCTION: &str = " When the task is done, stop and end with a short recap so \
the user has the full picture: a one-line headline of what you accomplished, then — for any \
non-trivial change — a brief bullet list of the key edits (grouped by file) and the exact \
command(s) to run or test it. Write it in past tense, covering only what you actually did; don't \
restate the plan or pad it. For a trivial change or a plain question, a single line is enough.";

/// Ending instruction when a finalization step may expand the recap. The main
/// model must still leave a concrete result; otherwise a failed side call can
/// turn a completed coding run into an apparently answerless stall.
const DEFERRED_RECAP_INSTRUCTION: &str = " When the task is done, stop and finish with one concrete \
sentence naming what changed and the verification result. A separate step may expand that summary, \
but never answer with a generic completion claim such as 'completed the requested action'.";

/// Builds the **stable** system message: identity, rules, working directory,
/// and durable project guides (HI.md/skills). Deliberately excludes anything
/// that changes turn-to-turn — task-ranked memory, the task context index,
/// session goal/goal state, and the decision log ride in the per-turn context
/// block attached to the user message instead (see
/// [`crate::Agent::volatile_context_block`]). Keeping message[0] byte-stable
/// is what lets provider prompt caches (explicit breakpoints and implicit
/// prefix caches alike) hit across the many model rounds of a session:
/// rebuilding it every round was observed to hold cache hits under 4%.
///
/// Usage:
/// ```ignore
/// SystemPrompt::new()
///     .with_project_context(ctx)
///     .with_finalize(true)
///     .build()
/// ```
pub(crate) struct SystemPrompt {
    workspace_root: Option<std::path::PathBuf>,
    project_context: Option<String>,
    standing_rules: Option<String>,
    finalize: bool,
}

impl SystemPrompt {
    pub(crate) fn new() -> Self {
        Self {
            workspace_root: None,
            project_context: None,
            standing_rules: None,
            finalize: false,
        }
    }

    pub(crate) fn with_workspace_root(mut self, root: &std::path::Path) -> Self {
        self.workspace_root = Some(root.to_path_buf());
        self
    }

    pub(crate) fn with_project_context(mut self, context: Option<&str>) -> Self {
        self.project_context = context
            .map(|s| clip_chars(s.trim(), MAX_PROJECT_CONTEXT_CHARS))
            .filter(|s| !s.is_empty());
        self
    }

    pub(crate) fn with_standing_rules(mut self, rules: Option<&str>) -> Self {
        self.standing_rules = rules
            .map(|s| clip_chars(s.trim(), MAX_STANDING_RULES_CHARS))
            .filter(|s| !s.is_empty());
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
        if let Some(cwd) = self.workspace_root.or_else(|| std::env::current_dir().ok()) {
            text.push_str(&format!(
                "\n\nYour working directory is `{}` — work here. Every shell command runs from \
                 this directory in a fresh shell, so `cd` does NOT persist between commands. Use \
                 paths relative to it; do not `cd` into, copy from, or create directories elsewhere.",
                cwd.display()
            ));
        }
        if let Some(rules) = self.standing_rules {
            text.push_str("\n\n# User standing rules (from me.md)\n");
            text.push_str(&rules);
        }
        if let Some(context) = self.project_context {
            text.push_str("\n\n");
            text.push_str(&context);
        }
        Message::system(text)
    }
}

/// Last-line defense: project context is also clipped at load, but config
/// can be injected from tests or other frontends.
const MAX_PROJECT_CONTEXT_CHARS: usize = 16_000;
const MAX_STANDING_RULES_CHARS: usize = 8_000;

fn clip_chars(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let clipped: String = text.chars().take(max.saturating_sub(1)).collect();
    format!("{clipped}…")
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

    #[test]
    fn system_prompt_steers_incremental_implementation() {
        let sys = SystemPrompt::new().build();
        let text = sys.text();
        assert!(text.contains("standard-library solutions"));
        assert!(text.contains("coherent chunks"));
        assert!(text.contains("targeted syntax/build/test command"));
        assert!(
            text.contains("repo_map") && text.contains("find_symbol"),
            "steers orientation tools: {text}"
        );
        assert!(
            text.contains("explore") && text.contains("delegate"),
            "steers subagent tools: {text}"
        );
        assert!(
            text.contains("Prefer `edit`") || text.contains("prefer `edit`"),
            "steers edit vs write/patch: {text}"
        );
        assert!(
            text.contains("`git add .`")
                && text.contains("`git add -A`")
                && text.contains("force-push"),
            "standing git rules: {text}"
        );
        assert!(
            text.contains("three failed attempts") && text.contains("tool set"),
            "escalate instead of looping: {text}"
        );
        assert!(
            text.contains("untrusted data")
                && text.contains("not instructions")
                && text.contains("MCP payloads"),
            "treats tool/web/MCP content as data: {text}"
        );
    }

    #[test]
    fn system_prompt_clips_a_huge_project_guide() {
        let bomb = "TOKENBOMB ".repeat(5_000);
        let sys = SystemPrompt::new()
            .with_project_context(Some(&bomb))
            .build();
        let text = sys.text();
        assert!(
            text.chars().count() < bomb.chars().count(),
            "system prompt must not absorb an unbounded HI.md: {}",
            text.chars().count()
        );
        assert!(
            text.chars().count() <= MAX_PROJECT_CONTEXT_CHARS + 4_000,
            "clipped guide plus identity must stay bounded: {}",
            text.chars().count()
        );
        assert!(
            !text.contains(&"TOKENBOMB ".repeat(2_000)),
            "huge guide must be clipped"
        );
    }

    #[test]
    fn system_prompt_includes_standing_rules_and_clips_them() {
        let sys = SystemPrompt::new()
            .with_standing_rules(Some("Prefer small diffs.\nNever force-push."))
            .build();
        let text = sys.text();
        assert!(text.contains("# User standing rules (from me.md)"));
        assert!(text.contains("Prefer small diffs."));
        let bomb = "TOKENBOMB ".repeat(5_000);
        let clipped = SystemPrompt::new()
            .with_standing_rules(Some(&bomb))
            .build()
            .text();
        assert!(
            clipped.chars().count() < bomb.chars().count(),
            "me.md must be clipped"
        );
        assert!(
            !clipped.contains(&"TOKENBOMB ".repeat(2_000)),
            "huge me.md must be clipped"
        );
    }
}
