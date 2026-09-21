//! What a review-drive audit turn may not do.
//!
//! The audit and re-audit prompts say "read-only", and a live re-audit still
//! rewrote `src/main.rs` with the missing `/topic` feature, added a test,
//! called `update_plan`, and then reported the feature as missing. Prompts
//! are advice; this is the guard. Write tools are denied with a result that
//! says what to do instead, shell commands the policy proves mutating
//! (`sed -i`, `rm`, `git checkout`, …) are denied the same way, and a turn
//! that keeps trying is stopped. Test runs (`cargo test … | tail`) are not
//! provably read-only and stay allowed: the prompt permits them.

use hi_tools::shell_policy::{ShellPolicyBasis, classify_shell_tool_arguments};
use hi_tools::{ToolEffects, ToolOutcome, ToolStatus, TruncationState};

/// Tools that change the tree or the plan.
pub const AUDIT_WRITE_TOOLS: &[&str] =
    &["write", "edit", "multi_edit", "apply_patch", "update_plan"];

/// Denied write calls in one audit turn before the turn is stopped.
pub const AUDIT_WRITE_DENIALS_BEFORE_STOP: u32 = 3;

/// Error kind and message for the stop after repeated denials.
pub const AUDIT_WRITE_STORM_KIND: &str = "audit_writes";
pub const AUDIT_WRITE_STORM_MSG: &str =
    "audit turn kept calling write tools after they were denied; stopping the turn";

/// Why an audit turn may not run this call, or `None` when it may.
pub fn audit_denial(name: &str, arguments: &str) -> Option<String> {
    if AUDIT_WRITE_TOOLS.contains(&name) {
        return Some(format!("`{name}` was not run"));
    }
    if name == "bash"
        && classify_shell_tool_arguments(arguments).basis == ShellPolicyBasis::KnownMutation
    {
        return Some("that shell command was not run (it modifies the tree)".to_string());
    }
    None
}

/// The tool result handed to the model for a denied call: the reason, and
/// how to report the change instead so the reply still ends in the block.
pub fn audit_denied_outcome(reason: &str) -> ToolOutcome {
    ToolOutcome {
        content: format!(
            "[hi:review] {reason}: this is a read-only audit turn. \
Report instead of changing anything: a defect in implemented code is a `finding:` row, \
an unimplemented plan/spec item is a `missing` or `partial` `coverage:` row. \
Do not write a plan or next steps for it; end your reply with the <review> block."
        ),
        display: None,
        plan: None,
        status: ToolStatus::Denied,
        process: None,
        background: None,
        effects: ToolEffects::default(),
        truncation: TruncationState::Complete,
        images: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_tools_and_proven_mutations_are_denied_reads_and_tests_are_not() {
        for name in AUDIT_WRITE_TOOLS {
            assert!(audit_denial(name, "{}").is_some(), "{name}");
        }
        assert_eq!(
            audit_denial("edit", r#"{"path":"src/main.rs"}"#).as_deref(),
            Some("`edit` was not run")
        );
        for name in ["read", "grep", "glob", "list", "repo_map", "diagnostics"] {
            assert_eq!(audit_denial(name, "{}"), None, "{name}");
        }
        let bash = |command: &str| {
            audit_denial(
                "bash",
                &serde_json::json!({ "command": command }).to_string(),
            )
        };
        assert!(bash("sed -i 's/a/b/' src/main.rs").is_some());
        assert!(bash("rm -rf target").is_some());
        assert!(bash("git checkout -- src/main.rs").is_some());
        assert_eq!(bash("cargo test --offline 2>&1 | tail -30"), None);
        assert_eq!(bash("cargo test --offline"), None);
        assert_eq!(bash("sed -n '10,20p' src/main.rs"), None);
        assert_eq!(bash("git diff --stat"), None);
        assert_eq!(
            audit_denial("bash", "not json"),
            None,
            "fail-closed classification is not a proven mutation"
        );
    }

    #[test]
    fn denied_outcome_is_denied_without_effects_and_says_what_to_do() {
        let outcome = audit_denied_outcome("`write` was not run");
        assert_eq!(outcome.status, ToolStatus::Denied);
        assert!(!outcome.effects.mutation_applied);
        assert!(outcome.plan.is_none());
        assert!(
            outcome
                .content
                .starts_with("[hi:review] `write` was not run: this is a read-only audit turn.")
        );
        assert!(
            outcome
                .content
                .contains("`missing` or `partial` `coverage:` row")
        );
        assert!(
            outcome
                .content
                .ends_with("end your reply with the <review> block.")
        );
        assert!(!hi_tools::is_probe_refusal(&outcome.content));
    }
}
