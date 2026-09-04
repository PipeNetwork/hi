use super::feature_enabled_for_value;

#[test]
fn defaults_on_for_every_build() {
    for value in [None, Some("on"), Some("true")] {
        assert!(feature_enabled_for_value(value));
    }
}

#[test]
fn only_an_explicit_environment_value_disables_folder_trust() {
    for value in ["off", "0", "false", "no", "", " OFF "] {
        assert!(!feature_enabled_for_value(Some(value)), "{value:?}");
    }
}
