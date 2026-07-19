//! Post-verification workspace settlement: keep or invalidate a green verify
//! when the tree moves after the check.

use hi_tools::FileChange;
use crate::outcome::ReviewStatus;
use crate::ui::Ui;

use super::helpers::post_verify_delta_is_benign;

const DEFAULT_INVALIDATED: &str =
    "workspace changed after verification; the previous pass was invalidated";

/// Compare the sealed verify revision to the current ledger head. Prose-only
/// deltas keep the pass (and refresh the sealed head); code/config deltas wipe it.
///
/// Returns `true` when a green verify was wiped (caller may roll back goal state).
pub(super) fn reconcile_verified_revision(
    last_verify: &mut Option<bool>,
    verified_at: &mut Option<(u64, String)>,
    independent_review_status: &mut ReviewStatus,
    current_revision: u64,
    current_digest: String,
    delta_since_verified: &[FileChange],
    ui: &mut dyn Ui,
) -> bool {
    reconcile_verified_revision_with_message(
        last_verify,
        verified_at,
        independent_review_status,
        current_revision,
        current_digest,
        delta_since_verified,
        ui,
        DEFAULT_INVALIDATED,
    )
}

/// Same as [`reconcile_verified_revision`] with a custom invalidation status line.
pub(super) fn reconcile_verified_revision_with_message(
    last_verify: &mut Option<bool>,
    verified_at: &mut Option<(u64, String)>,
    independent_review_status: &mut ReviewStatus,
    current_revision: u64,
    current_digest: String,
    delta_since_verified: &[FileChange],
    ui: &mut dyn Ui,
    invalidated_message: &str,
) -> bool {
    if *last_verify != Some(true) {
        return false;
    }
    let drifted = verified_at
        .as_ref()
        .is_none_or(|(revision, digest)| *revision != current_revision || digest != &current_digest);
    if !drifted {
        return false;
    }
    // Only code/config deltas after the pass wipe it. Prose-only writes
    // (learned skills, docs) are outside the auto pipeline and must not
    // flip a green turn into "incomplete · unverified changes".
    if post_verify_delta_is_benign(delta_since_verified) {
        // Keep the pass; refresh the sealed revision to the new head so
        // later settlement checks compare against the prose write too.
        *verified_at = Some((current_revision, current_digest));
        false
    } else {
        *last_verify = None;
        *verified_at = None;
        if *independent_review_status == ReviewStatus::Passed {
            *independent_review_status = ReviewStatus::Unavailable;
        }
        ui.status(invalidated_message);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::outcome::ReviewStatus;
    use crate::ui::Ui;
    use hi_tools::{FileChange, FileChangeKind};

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

    fn change(path: &str) -> FileChange {
        FileChange {
            path: path.into(),
            kind: FileChangeKind::Modify,
            before_digest: None,
            after_digest: Some("a".into()),
            before_len: None,
            after_len: Some(1),
            before_mode: None,
            after_mode: None,
        }
    }

    #[test]
    fn prose_delta_keeps_pass_and_refreshes_head() {
        let mut last = Some(true);
        let mut verified = Some((1, "old".into()));
        let mut review = ReviewStatus::Passed;
        let mut ui = NullUi;
        let wiped = reconcile_verified_revision(
            &mut last,
            &mut verified,
            &mut review,
            2,
            "new".into(),
            &[change("README.md")],
            &mut ui,
        );
        assert!(!wiped);
        assert_eq!(last, Some(true));
        assert_eq!(verified, Some((2, "new".into())));
        assert_eq!(review, ReviewStatus::Passed);
    }

    #[test]
    fn code_delta_wipes_pass() {
        let mut last = Some(true);
        let mut verified = Some((1, "old".into()));
        let mut review = ReviewStatus::Passed;
        let mut ui = NullUi;
        let wiped = reconcile_verified_revision(
            &mut last,
            &mut verified,
            &mut review,
            2,
            "new".into(),
            &[change("src/lib.rs")],
            &mut ui,
        );
        assert!(wiped);
        assert_eq!(last, None);
        assert_eq!(verified, None);
        assert_eq!(review, ReviewStatus::Unavailable);
    }

    #[test]
    fn no_op_when_not_green() {
        let mut last = Some(false);
        let mut verified = None;
        let mut review = ReviewStatus::NotRequired;
        let mut ui = NullUi;
        let wiped = reconcile_verified_revision(
            &mut last,
            &mut verified,
            &mut review,
            9,
            "x".into(),
            &[change("a.rs")],
            &mut ui,
        );
        assert!(!wiped);
        assert_eq!(last, Some(false));
    }
}
