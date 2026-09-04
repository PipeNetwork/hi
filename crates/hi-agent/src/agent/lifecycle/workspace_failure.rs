/// Retain deterministic evidence across a later, unrelated turn failure only
/// when a pass is still bound to the current workspace. Provider/session
/// failures are not verifier infrastructure failures, and therefore default to
/// `Unverified` rather than manufacturing `InfrastructureError`.
pub(super) fn verification_after_turn_failure(
    evidence: &crate::domain::VerifyEvidence,
    workspace_reconciled: bool,
    current_revision: u64,
    current_digest: &str,
) -> (crate::VerificationStatus, Option<String>) {
    if !workspace_reconciled {
        return (crate::VerificationStatus::Unverified, None);
    }
    match evidence {
        crate::domain::VerifyEvidence::Passed { revision, digest }
            if *revision == current_revision && digest == current_digest =>
        {
            (crate::VerificationStatus::Passed, Some(digest.clone()))
        }
        crate::domain::VerifyEvidence::Failed => (crate::VerificationStatus::Failed, None),
        crate::domain::VerifyEvidence::None | crate::domain::VerifyEvidence::Passed { .. } => {
            (crate::VerificationStatus::Unverified, None)
        }
    }
}
