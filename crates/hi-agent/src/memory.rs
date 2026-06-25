//! Project memory file location, distillation prompt, and capping helpers.

/// Backstop cap on the distilled memory file. The prompt does the real shaping
/// (≤ ~20 short bullets); this just stops a runaway response from bloating the
/// file — and thus every future session's context.
const MEMORY_MAX_CHARS: usize = 2_000;

/// Where the project memory lives — `.hi/memory.md` under the working directory,
/// overridable via `HI_MEMORY_FILE` (which also makes the file IO testable). The
/// frontend reads the same path to load it as context.
pub fn memory_file() -> std::path::PathBuf {
    std::env::var_os("HI_MEMORY_FILE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::Path::new(".hi").join("memory.md"))
}

/// The session-end distillation prompt, folding in the current memory so the
/// model revises it (merge / de-dupe / drop-stale) instead of appending.
pub(crate) fn memory_prompt(existing: &str) -> String {
    let existing = if existing.trim().is_empty() {
        "(empty)"
    } else {
        existing.trim()
    };
    format!(
        "This coding session is ending. Maintain a small, durable memory for future work \
         in this project — reusable notes, not a transcript.\n\nCurrent saved memory:\n\
         ---\n{existing}\n---\n\nRevise it using only what THIS session actually \
         established: keep facts that save time next time — project conventions, key \
         decisions and constraints, non-obvious gotchas, important file locations, and \
         the exact build/test/run commands that matter. Drop anything transient, already \
         obvious from the code or HI.md, or now outdated. Merge and de-duplicate. Output \
         ONLY the updated memory as at most ~20 short bullet points (a few words to one \
         line each), no preamble. If nothing durable is worth keeping, output the current \
         memory unchanged (or nothing if it was empty)."
    )
}

/// Trim and cap the distilled memory at [`MEMORY_MAX_CHARS`], cutting back to the
/// last whole line so a bullet isn't sliced mid-word. Empty in → empty out.
pub(crate) fn cap_memory(s: &str) -> String {
    let s = s.trim();
    if s.chars().count() <= MEMORY_MAX_CHARS {
        return s.to_string();
    }
    let kept: String = s.chars().take(MEMORY_MAX_CHARS).collect();
    let kept = kept
        .rsplit_once('\n')
        .map(|(head, _)| head)
        .unwrap_or(&kept);
    format!("{}\n… (memory truncated)", kept.trim_end())
}

/// Whether to distill session memory at quit: only when enabled *and* the model
/// actually produced output this session, so an empty or command-only session
/// writes nothing. Shared by both frontends so the rule can't drift between them.
pub fn should_distill_memory(enabled: bool, output_tokens: u64) -> bool {
    enabled && output_tokens > 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cap_memory_trims_and_backstops() {
        assert_eq!(cap_memory("  - a\n- b  "), "- a\n- b"); // trimmed, under budget
        assert_eq!(cap_memory("   "), ""); // empty in → empty out
        let big = "- a durable note\n".repeat(1000); // ≫ MEMORY_MAX_CHARS
        let capped = cap_memory(&big);
        assert!(
            capped.chars().count() <= MEMORY_MAX_CHARS + 40,
            "backstopped"
        );
        assert!(capped.ends_with("(memory truncated)"));
    }

    #[test]
    fn memory_prompt_folds_in_existing_memory() {
        let p = memory_prompt("- 4-space indent");
        assert!(p.contains("- 4-space indent"), "includes current memory");
        assert!(p.contains("Current saved memory"));
        assert!(memory_prompt("   ").contains("(empty)"), "blank → (empty)");
    }

    #[test]
    fn should_distill_memory_gates_on_enabled_and_work() {
        assert!(should_distill_memory(true, 1), "enabled + work → distill");
        assert!(!should_distill_memory(true, 0), "no model output → skip");
        assert!(!should_distill_memory(false, 100), "disabled → skip");
    }
}
