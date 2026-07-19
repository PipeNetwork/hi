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
    // Memory distilled from past sessions (auto-maintained at session end).
    // Hierarchical: project memory (annotated for stale paths/commands) + a
    // global user-level layer for cross-project preferences.
    let project = hi_agent::read_project_annotated();
    let global = hi_agent::read_global_memory();
    let mem = render_memory_layers(&project, &global);
    if let Some(section) = memory_context(&mem) {
        parts.push(section);
    }
    // Repository structure is supplied per task by hi-agent's deterministic,
    // ranked context index. Do not also inject the old alphabetical repo map:
    // it consumed every request and could crowd out task-relevant files.
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
pub(crate) fn memory_context(text: &str) -> Option<String> {
    let text = text.trim();
    (!text.is_empty()).then(|| format!("# Memory (from past sessions)\n{text}"))
}

/// Render the hierarchical memory layers into a single context block.
///
/// Project bullets are emitted first (annotated with stale-path warnings on
/// render), then global user-level bullets under a sub-heading. Either layer
/// may be empty.
pub(crate) fn render_memory_layers(project: &[hi_agent::AnnotatedBullet], global: &str) -> String {
    let mut out = String::new();
    for b in project {
        out.push_str(&b.render());
        out.push('\n');
    }
    let global = global.trim();
    if !global.is_empty() {
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str("## User-level (global)\n");
        out.push_str(global);
        out.push('\n');
    }
    out
}

