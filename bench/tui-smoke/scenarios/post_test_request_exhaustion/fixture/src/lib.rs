#[test]
fn existing_behavior_passes() {
    assert_eq!("retained".strip_prefix("re"), Some("tained"));
}
