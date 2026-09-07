//! Post-verification workspace settlement: keep or invalidate a green verify
//! when the tree moves after the check.

use crate::domain::VerifyEvidence;
use crate::outcome::ReviewStatus;
use crate::ui::Ui;

const DEFAULT_INVALIDATED: &str =
    "workspace changed after verification; the previous pass was invalidated";

/// Compare the immutable checked inputs with current canonical inputs. A
/// bookkeeping revision alone cannot alter or replace the original seal.
///
/// Returns `true` when a green verify was wiped (caller may roll back goal state).
pub(super) fn reconcile_verified_revision(
    evidence: &mut VerifyEvidence,
    independent_review_status: &mut ReviewStatus,
    current_digest: String,
    ui: &mut dyn Ui,
) -> bool {
    reconcile_verified_revision_with_message(
        evidence,
        independent_review_status,
        current_digest,
        ui,
        DEFAULT_INVALIDATED,
    )
}

/// Same as [`reconcile_verified_revision`] with a custom invalidation status line.
pub(super) fn reconcile_verified_revision_with_message(
    evidence: &mut VerifyEvidence,
    independent_review_status: &mut ReviewStatus,
    current_digest: String,
    ui: &mut dyn Ui,
    invalidated_message: &str,
) -> bool {
    // Only a Passed verdict (which carries bound evidence) can drift.
    let VerifyEvidence::Passed { revision, digest } = evidence else {
        return false;
    };
    if digest == &current_digest {
        return false;
    }
    *evidence = VerifyEvidence::Invalidated {
        revision: *revision,
        digest: digest.clone(),
    };
    if *independent_review_status == ReviewStatus::Passed {
        *independent_review_status = ReviewStatus::Unavailable;
    }
    ui.status(invalidated_message);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::outcome::ReviewStatus;
    use crate::ui::Ui;

    struct NullUi;
    impl Ui for NullUi {
        fn assistant_text(&mut self, _: &str) {}
        fn assistant_reasoning(&mut self, _: &str) {}
        fn assistant_end(&mut self) {}
        fn tool_call(&mut self, _: &str, _: &str) {}
        fn tool_result(&mut self, _: &str, _: &str) {}
        fn status(&mut self, _: &str) {}
        fn turn_end(&mut self, _: &str) {}
    }

    #[test]
    fn changed_inputs_invalidate_pass_and_keep_original_seal() {
        let mut evidence = VerifyEvidence::pass(1, "checked".into());
        let mut review = ReviewStatus::Passed;
        assert!(reconcile_verified_revision(
            &mut evidence,
            &mut review,
            "changed".into(),
            &mut NullUi,
        ));
        assert_eq!(
            evidence,
            VerifyEvidence::Invalidated {
                revision: 1,
                digest: "checked".into(),
            }
        );
        assert_eq!(review, ReviewStatus::Unavailable);
        // Returning to the old bytes cannot resurrect an invalidated pass.
        assert!(!reconcile_verified_revision(
            &mut evidence,
            &mut review,
            "checked".into(),
            &mut NullUi,
        ));
        assert!(evidence.invalidated());
    }

    #[test]
    fn identical_inputs_preserve_original_seal() {
        let mut evidence = VerifyEvidence::pass(1, "checked".into());
        let mut review = ReviewStatus::Passed;
        assert!(!reconcile_verified_revision(
            &mut evidence,
            &mut review,
            "checked".into(),
            &mut NullUi,
        ));
        assert_eq!(evidence, VerifyEvidence::pass(1, "checked".into()));
        assert_eq!(review, ReviewStatus::Passed);
    }

    #[test]
    fn failed_checks_remain_failed() {
        let mut evidence = VerifyEvidence::fail();
        let mut review = ReviewStatus::NotRequired;
        assert!(!reconcile_verified_revision(
            &mut evidence,
            &mut review,
            "changed".into(),
            &mut NullUi,
        ));
        assert_eq!(evidence, VerifyEvidence::fail());
    }
}
