use super::*;

#[test]
fn feature_controls_only_the_transcript_buffer_guard() {
    let placeholder = "Completed the requested action.";
    assert!(could_be_generic_completion_prefix(placeholder));
    assert_eq!(
        should_buffer_generic_completion_prefix(placeholder),
        generic_completion_guards_enabled()
    );
}
