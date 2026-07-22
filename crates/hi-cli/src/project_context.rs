//! Project guide and hierarchical memory context loaded into the agent.

pub(crate) fn load_project_context() -> Option<String> {
    const FILES: &[&str] = &["HI.md", "AGENTS.md"];
    let mut parts = Vec::new();
    for name in FILES {
        if let Ok(text) = std::fs::read_to_string(name) {
            let text = text.trim();
            if !text.is_empty() {
                parts.push(format!("# Project context (from {name})\n{text}"));
            }
        }
    }
    // Memory is injected live by hi-agent (task-ranked, refreshed each turn and
    // after coding-fact writes). Do not bake a static snapshot here — that
    // frozen the session-start file and crowded the prompt with unranked bullets.
    // Repository structure is also supplied per task by hi-agent's ranked
    // context index / repo_map seed.
    if let Some(section) = hi_agent::learned_skills_context() {
        parts.push(section);
    }
    (!parts.is_empty()).then(|| parts.join("\n\n"))
}

/// Whether auto-memory is active for this session: on unless `--no-memory`, and
/// off when the session isn't saved (`--no-save`) since memory is persistence.
pub(crate) fn auto_memory_enabled(no_memory: bool, no_save: bool) -> bool {
    !no_memory && !no_save
}

/// Build the `# Memory` context section from the saved memory file's contents,
/// or `None` when it's empty/whitespace (so a blank file adds nothing).
///
/// Kept for unit tests / callers that still want a static wrap; production
/// injection goes through `hi_agent::memory_section_for_task` (task-ranked).
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn memory_context(text: &str) -> Option<String> {
    let text = text.trim();
    (!text.is_empty()).then(|| format!("# Memory (from past sessions)\n{text}"))
}
