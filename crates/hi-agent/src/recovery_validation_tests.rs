use super::*;

fn failures() -> Option<BTreeSet<String>> {
    Some(BTreeSet::from(["suite::broad_failure".into()]))
}

#[test]
fn unresolved_diagnostics_distinguish_stale_and_current_workspace_evidence() {
    let mut state = TaskRecoveryState::default();
    state.observe_validation(
        "cargo check",
        "before-edit",
        Some(BTreeSet::from([
            "diag:error[E0382]: use of partially moved value `cmd`\0src/server.rs".into(),
        ])),
        false,
        false,
    );
    let stale = state.unresolved_validation_summary("after-edit").unwrap();
    assert!(stale.contains("Earlier failure at workspace revision before-edit"));
    assert!(stale.contains("current revision remains unverified"));
    assert!(stale.contains("src/server.rs"));
    assert!(!stale.contains('\0'));
    let current = state.unresolved_validation_summary("before-edit").unwrap();
    assert!(current.contains("failed for the current workspace revision"));
    assert!(!current.contains("workspace changed"));
}

#[test]
fn narrow_success_cannot_clear_a_broader_failure_even_after_inputs_change() {
    let mut state = TaskRecoveryState::default();
    state.observe_validation("full suite", "revision-a", failures(), false, false);
    state.observe_validation("focused suite", "revision-a", None, true, false);
    assert_eq!(
        state.unresolved_validation_status("revision-a"),
        Some(ValidationResult::Failed)
    );
    assert_eq!(
        state.unresolved_validation_status("revision-b"),
        Some(ValidationResult::Deferred)
    );
    assert!(
        state
            .unresolved_validation_summary("current")
            .unwrap()
            .contains("suite::broad_failure")
    );
    state.observe_validation("full suite", "revision-b", None, true, false);
    assert_eq!(state.unresolved_validation_status("revision-b"), None);
}

#[test]
fn regressing_a_historical_pass_reopens_failure_without_reearning_credit() {
    let mut state = TaskRecoveryState::default();
    state.observe_validation("full suite", "revision-a", failures(), false, true);
    state.intervene("repair");
    state.observe_validation("full suite", "revision-b", None, true, true);
    assert_eq!(state.remaining, 3);
    state.intervene("another correction");
    state.observe_validation("full suite", "revision-c", failures(), false, true);
    assert_eq!(
        state.unresolved_validation_status("revision-c"),
        Some(ValidationResult::Failed)
    );
    state.observe_validation("full suite", "revision-d", None, true, true);
    assert_eq!(state.unresolved_validation_status("revision-d"), None);
    assert_eq!(
        state.remaining, 2,
        "an old best pass cannot replenish recovery again"
    );
}

#[test]
fn current_failure_survives_restore_and_explicit_clearing_stays_cleared() {
    let mut state = TaskRecoveryState::default();
    state.observe_validation("full suite", "revision-a", failures(), false, false);
    let restore = |state: &TaskRecoveryState| -> TaskRecoveryState {
        serde_json::from_value(serde_json::to_value(state).unwrap()).unwrap()
    };
    let mut restored = restore(&state);
    assert_eq!(
        restored.unresolved_validation_status("revision-a"),
        Some(ValidationResult::Failed)
    );
    restored.observe_validation("full suite", "revision-b", None, true, false);
    assert_eq!(
        restore(&restored).unresolved_validation_status("revision-b"),
        None
    );
}

#[test]
fn legacy_failure_migrates_and_ambiguous_pass_history_requires_revalidation() {
    let mut state = TaskRecoveryState::default();
    state.observe_validation("full suite", "revision-a", failures(), false, false);
    let legacy = |state: &TaskRecoveryState| {
        let mut value = serde_json::to_value(state).unwrap();
        value["schema_version"] = 1.into();
        value["validations"]["full suite"]
            .as_object_mut()
            .unwrap()
            .remove("current_failure");
        value
    };
    let mut restored: TaskRecoveryState = serde_json::from_value(legacy(&state)).unwrap();
    restored.migrate();
    restored.validate().unwrap();
    assert_eq!(restored.schema_version, RECOVERY_SCHEMA_VERSION);
    assert_eq!(
        restored.unresolved_validation_status("revision-a"),
        Some(ValidationResult::Failed)
    );

    state.observe_validation("full suite", "revision-b", None, true, false);
    let mut restored: TaskRecoveryState = serde_json::from_value(legacy(&state)).unwrap();
    assert_eq!(
        restored.unresolved_validation_status("revision-a"),
        Some(ValidationResult::Deferred)
    );
    assert!(
        restored
            .unresolved_validation_summary("current")
            .unwrap()
            .contains("revalidation is required")
    );
    restored.observe_validation("full suite", "revision-c", None, true, false);
    assert_eq!(restored.unresolved_validation_status("revision-c"), None);
}

#[test]
fn a_real_pass_can_resolve_an_unrecognized_failure_once() {
    let mut state = TaskRecoveryState::default();
    state.observe_validation("custom check", "a", None, false, false);
    state.intervene("repair");
    state.observe_validation("custom check", "b", None, true, false);
    assert_eq!(state.remaining, 3);
    state.intervene("repair");
    state.observe_validation("custom check", "c", None, false, false);
    state.observe_validation("custom check", "d", None, true, false);
    assert_eq!(state.remaining, 2);
    assert_eq!(state.unresolved_validation_status("d"), None);
}
