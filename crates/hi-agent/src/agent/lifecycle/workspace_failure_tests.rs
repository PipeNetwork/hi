use super::verification_after_turn_failure;
use crate::VerificationStatus;
use crate::domain::VerifyEvidence;

#[test]
fn only_a_current_revision_pass_survives_later_infrastructure_failure() {
    let pass = VerifyEvidence::pass(7, "current".into());
    assert_eq!(
        verification_after_turn_failure(&pass, true, 7, "current"),
        (VerificationStatus::Passed, Some("current".into()))
    );
    assert_eq!(
        verification_after_turn_failure(&pass, true, 8, "changed"),
        (VerificationStatus::Unverified, None)
    );
    assert_eq!(
        verification_after_turn_failure(&pass, false, 7, "current"),
        (VerificationStatus::Unverified, None)
    );
}
