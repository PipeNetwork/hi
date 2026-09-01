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
    /// Code changed and auto-detection found nothing that exercises it. Failing
    /// to *find* a check is not evidence that none is needed — and here the
    /// model is the only party that can supply one, because it just wrote the
    /// code. Observed: a new module with no test suite completed clean without
    /// ever being executed, and the defect (an exception on the first call)
    /// would have surfaced from running it once.
    NoExecutableCheck,
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
            Self::NoExecutableCheck => "\
Verification obligation: this turn wrote code that was never executed, and no \
existing check covers it. Run it now — a smoke test, a direct invocation, a \
short script that exercises the new behavior end to end, whatever is cheapest \
— and show the output. Prefer the shape the task itself implies (if it \
describes multiple processes, ranks, or a server, exercise that shape, not \
just an import). Fix what it reveals. If running it truly is not possible \
here, say plainly why."
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
            Self::NoExecutableCheck => {
                "verification obligation — new code was never executed; asking for a check once"
            }
        }
    }
}

/// Decide whether a coding turn still owes verify evidence before Settle.
///
/// Returns `None` when obligation does not apply (read-only, prose-only,
/// verification disabled, already green, etc.).
#[allow(clippy::too_many_arguments)] // verify obligation carries each gating input explicitly
pub(crate) fn coding_verify_obligation(
    contract: Option<&TaskContract>,
    verification_mode: &VerificationMode,
    expected_mutation: bool,
    changed_files: &[String],
    mutation_seen: bool,
    last_verify: Option<bool>,
    verify_executions: usize,
    validation_after_last_mutation: bool,
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
    // Auto with zero executions means auto-detection found no pipeline for this
    // code. That is exactly when the model must supply the check itself — it is
    // the only party that can, having just written the code — so it is an
    // obligation of its own kind rather than a free pass. Explicit always has
    // stages the user asked for; any prior execution without a seal is a real gap.
    match verification_mode {
        VerificationMode::Disabled => None,
        VerificationMode::Auto if verify_executions == 0 => {
            (!validation_after_last_mutation).then_some(ObligationReason::NoExecutableCheck)
        }
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
                false,
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
                false,
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
                false,
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
                false,
            ),
            Some(ObligationReason::UnverifiedMutation)
        );
        // Auto with zero executions → auto-detection found nothing that
        // exercises the new code, so the model owes an executable check.
        // Previously this was a free pass, which let code that had never run
        // once report "no applicable checks".
        assert_eq!(
            coding_verify_obligation(
                Some(&mutation_contract()),
                &VerificationMode::Auto,
                true,
                &["src/a.rs".into()],
                true,
                None,
                0,
                false,
            ),
            Some(ObligationReason::NoExecutableCheck)
        );
        // A successful model-run smoke check is valid evidence when the
        // automatic pipeline has no stage of its own; do not make the model
        // repeat the same command during the obligation round.
        assert_eq!(
            coding_verify_obligation(
                Some(&mutation_contract()),
                &VerificationMode::Auto,
                true,
                &["src/a.rs".into()],
                true,
                None,
                0,
                true,
            ),
            None
        );
    }

    #[test]
    fn no_executable_check_nudge_asks_for_a_run_not_a_claim() {
        let body = ObligationReason::NoExecutableCheck.nudge_body();
        assert!(body.contains("never executed"), "{body}");
        // It must ask for evidence in the shape the task implies, so a
        // multi-process/multi-rank task is not "checked" by an import alone.
        assert!(body.contains("ranks") || body.contains("shape"), "{body}");
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
                false,
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
                false,
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
                false,
            ),
            None
        );
    }
}
