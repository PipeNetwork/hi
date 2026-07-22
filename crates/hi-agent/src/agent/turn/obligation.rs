//! Coding-turn verification obligation.
//!
//! A mutation-shaped turn that changed code should not claim "done" without
//! green verify evidence when a verification pipeline is configured. This module
//! decides when to fire a one-shot re-entry nudge before Settle.

use crate::config::VerificationMode;
use crate::task_contract::{TaskContract, TaskIntent};
use crate::verify::is_prose_only_path;

/// Why the turn still owes deterministic verification evidence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ObligationReason {
    /// Code changed, stages are configured, but no stage produced a green seal.
    UnverifiedMutation,
    /// Last verify attempt failed and the repair budget is exhausted.
    FailedVerify,
}

impl ObligationReason {
    pub(crate) fn nudge_body(&self) -> String {
        match self {
            Self::UnverifiedMutation => "\
Verification obligation: this turn changed code but never produced a green \
verification seal. Before finishing, run the project's check/test command \
(or the configured verify stages) and fix any failures — or say explicitly \
why verification does not apply. Do not claim the task is done without \
evidence."
                .into(),
            Self::FailedVerify => "\
Verification obligation: the last verification attempt failed and the repair \
budget is exhausted. Either make a concrete fix and leave evidence the next \
turn can verify, or stop and report what is still broken. Do not claim the \
task is done."
                .into(),
        }
    }

    pub(crate) fn ui_status(&self) -> &'static str {
        match self {
            Self::UnverifiedMutation => {
                "verification obligation — code changed without a green seal; nudging once"
            }
            Self::FailedVerify => {
                "verification obligation — last check failed; nudging once before settle"
            }
        }
    }
}

/// Decide whether a coding turn still owes verify evidence before Settle.
///
/// Returns `None` when obligation does not apply (read-only, prose-only,
/// verification disabled, already green, etc.).
pub(crate) fn coding_verify_obligation(
    contract: Option<&TaskContract>,
    verification_mode: &VerificationMode,
    expected_mutation: bool,
    changed_files: &[String],
    mutation_seen: bool,
    last_verify: Option<bool>,
    verify_executions: usize,
) -> Option<ObligationReason> {
    // No configured pipeline → nothing to obligate.
    if matches!(verification_mode, VerificationMode::Disabled) {
        return None;
    }
    // Already green.
    if last_verify == Some(true) {
        return None;
    }

    let coding_turn = expected_mutation
        || contract.is_some_and(|c| {
            c.intent == TaskIntent::Mutation || c.explicit_mutation || mutation_seen
        })
        || mutation_seen;

    if !coding_turn {
        return None;
    }

    let code_touched = mutation_seen || changed_files.iter().any(|path| !is_prose_only_path(path));

    if !code_touched {
        return None;
    }

    // Prose-only net change with no mutation_seen → not a coding obligation.
    if !mutation_seen
        && !changed_files.is_empty()
        && changed_files.iter().all(|path| is_prose_only_path(path))
    {
        return None;
    }

    if last_verify == Some(false) {
        return Some(ObligationReason::FailedVerify);
    }

    // Unverified (last_verify is None): code changed but nothing sealed green.
    // Auto with zero executions usually means "no pipeline detected" → honest
    // NotApplicable, not an obligation. Explicit always has stages the user
    // asked for; any prior execution without a seal is also a real gap.
    match verification_mode {
        VerificationMode::Disabled => None,
        VerificationMode::Auto if verify_executions == 0 => None,
        VerificationMode::Auto | VerificationMode::Explicit(_) => {
            Some(ObligationReason::UnverifiedMutation)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::VerifyStage;

    fn mutation_contract() -> TaskContract {
        let mut c = TaskContract::derive("fix the login bug", VerificationMode::Auto);
        c.intent = TaskIntent::Mutation;
        c.explicit_mutation = true;
        c
    }

    #[test]
    fn green_verify_clears_obligation() {
        assert_eq!(
            coding_verify_obligation(
                Some(&mutation_contract()),
                &VerificationMode::Auto,
                true,
                &["src/a.rs".into()],
                true,
                Some(true),
                1,
            ),
            None
        );
    }

    #[test]
    fn failed_verify_is_obligation() {
        assert_eq!(
            coding_verify_obligation(
                Some(&mutation_contract()),
                &VerificationMode::Explicit(vec![VerifyStage::new("t", "true")]),
                true,
                &["src/a.rs".into()],
                true,
                Some(false),
                2,
            ),
            Some(ObligationReason::FailedVerify)
        );
    }

    #[test]
    fn unverified_mutation_is_obligation() {
        // Explicit pipeline + mutation + no seal → obligate.
        assert_eq!(
            coding_verify_obligation(
                Some(&mutation_contract()),
                &VerificationMode::Explicit(vec![VerifyStage::new("t", "true")]),
                true,
                &["src/a.rs".into()],
                true,
                None,
                0,
            ),
            Some(ObligationReason::UnverifiedMutation)
        );
        // Auto with executions but no seal → obligate.
        assert_eq!(
            coding_verify_obligation(
                Some(&mutation_contract()),
                &VerificationMode::Auto,
                true,
                &["src/a.rs".into()],
                true,
                None,
                1,
            ),
            Some(ObligationReason::UnverifiedMutation)
        );
        // Auto with zero executions → no pipeline detected, not an obligation.
        assert_eq!(
            coding_verify_obligation(
                Some(&mutation_contract()),
                &VerificationMode::Auto,
                true,
                &["src/a.rs".into()],
                true,
                None,
                0,
            ),
            None
        );
    }

    #[test]
    fn read_only_no_obligation() {
        let c = TaskContract::derive("what does main do?", VerificationMode::Auto);
        assert_eq!(
            coding_verify_obligation(
                Some(&c),
                &VerificationMode::Auto,
                false,
                &[],
                false,
                None,
                0,
            ),
            None
        );
    }

    #[test]
    fn disabled_verify_no_obligation() {
        assert_eq!(
            coding_verify_obligation(
                Some(&mutation_contract()),
                &VerificationMode::Disabled,
                true,
                &["src/a.rs".into()],
                true,
                None,
                0,
            ),
            None
        );
    }

    #[test]
    fn prose_only_no_obligation() {
        assert_eq!(
            coding_verify_obligation(
                Some(&mutation_contract()),
                &VerificationMode::Auto,
                true,
                &["README.md".into()],
                false,
                None,
                0,
            ),
            None
        );
    }
}
