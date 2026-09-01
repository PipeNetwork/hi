//! Synthetic denial messages and waiting/dry-run classification.

use crate::{ConfirmationResult, PARKED_TOOL_RESULT};

#[cfg(test)]
use crate::agent::turn::helpers::synthetic_tool_outcome;

/// Whether a call is compatible with a turn that is merely waiting on live
/// background work: the poll itself, plan bookkeeping, and non-mutating,
/// non-validating shell probes (log tails, size checks, process listings).
/// Real work — edits, builds, tests, file reads — is deliberately not
/// wait-flavored, so interleaved genuine progress resets the waiting streak.
pub(super) fn wait_flavored_call(
    name: &str,
    arguments: &str,
    output: &hi_tools::ToolOutcome,
) -> bool {
    match name {
        "bash_output" | "update_plan" => true,
        "bash" => {
            !output.effects.mutation_applied
                && !crate::steering::implementation_tool_call_validates(name, arguments)
        }
        _ => false,
    }
}

/// Human-readable "planned action" line for a dry-run tool call. Pure so the
/// dry-run path is unit-testable without a live agent/runtime.
pub(super) fn dry_run_message(name: &str, path: &str, mutates: bool) -> String {
    let target = if path.is_empty() {
        String::new()
    } else {
        format!(" on {path}")
    };
    let kind = if mutates { "mutating" } else { "read-only" };
    format!("[dry-run] would run `{name}`{target} ({kind}; not executed)")
}

#[cfg(test)]
mod dry_run_tests {
    use super::*;

    #[test]
    fn mutating_call_reports_path_and_mutation() {
        let msg = dry_run_message("edit", "src/main.rs", true);
        assert_eq!(
            msg,
            "[dry-run] would run `edit` on src/main.rs (mutating; not executed)"
        );
    }

    #[test]
    fn read_only_call_reports_read_only() {
        let msg = dry_run_message("read", "src/main.rs", false);
        assert_eq!(
            msg,
            "[dry-run] would run `read` on src/main.rs (read-only; not executed)"
        );
    }

    #[test]
    fn call_without_path_omits_target() {
        let msg = dry_run_message("bash", "", true);
        assert_eq!(msg, "[dry-run] would run `bash` (mutating; not executed)");
    }
}

#[cfg(test)]
mod wait_flavored_tests {
    use super::*;

    fn outcome(mutation_applied: bool) -> hi_tools::ToolOutcome {
        let mut output = synthetic_tool_outcome("ok".into(), hi_tools::ToolStatus::Succeeded);
        output.effects.mutation_applied = mutation_applied;
        output
    }

    #[test]
    fn polls_bookkeeping_and_status_probes_are_wait_flavored_but_real_work_is_not() {
        assert!(wait_flavored_call(
            "bash_output",
            r#"{"id":"sh_1"}"#,
            &outcome(false)
        ));
        assert!(wait_flavored_call("update_plan", "{}", &outcome(false)));
        assert!(wait_flavored_call(
            "bash",
            r#"{"command":"tail -c 200 /tmp/download.log"}"#,
            &outcome(false)
        ));
        // Real work resets the waiting streak: mutations, validation runs,
        // and file reads are progress, not babysitting.
        assert!(!wait_flavored_call(
            "bash",
            r#"{"command":"echo hi > f.txt"}"#,
            &outcome(true)
        ));
        assert!(!wait_flavored_call(
            "bash",
            r#"{"command":"cargo test"}"#,
            &outcome(false)
        ));
        assert!(!wait_flavored_call(
            "read",
            r#"{"path":"src/main.rs"}"#,
            &outcome(false)
        ));
        assert!(!wait_flavored_call("edit", "{}", &outcome(true)));
    }
}

pub(super) fn parked_or_denied_shell(
    decision: &ConfirmationResult,
) -> (String, hi_tools::ToolStatus) {
    match decision {
        ConfirmationResult::Parked => (
            PARKED_TOOL_RESULT.to_string(),
            hi_tools::ToolStatus::Cancelled,
        ),
        ConfirmationResult::Unavailable => (
            "Shell mutation skipped: confirmation required, but this frontend cannot answer it; rerun interactively or disable --confirm-edits.".into(),
            hi_tools::ToolStatus::Denied,
        ),
        _ => (
            "Shell mutation skipped by user (not run).".into(),
            hi_tools::ToolStatus::Denied,
        ),
    }
}

pub(super) fn parked_or_denied_delegate(
    decision: &ConfirmationResult,
) -> (String, hi_tools::ToolStatus) {
    match decision {
        ConfirmationResult::Parked => (
            PARKED_TOOL_RESULT.to_string(),
            hi_tools::ToolStatus::Cancelled,
        ),
        ConfirmationResult::Unavailable => (
            "Delegate skipped: confirmation required, but this frontend cannot answer it.".into(),
            hi_tools::ToolStatus::Denied,
        ),
        _ => (
            "Delegate skipped by user (no changes applied).".into(),
            hi_tools::ToolStatus::Denied,
        ),
    }
}
