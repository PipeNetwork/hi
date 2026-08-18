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
    finalize: bool,
}

impl SystemPrompt {
    pub(crate) fn new() -> Self {
        Self {
            workspace_root: None,
            project_context: None,
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
}
