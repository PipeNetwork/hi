use super::*;

fn failures(names: &[&str]) -> Option<BTreeSet<String>> {
    Some(names.iter().map(|name| (*name).to_owned()).collect())
}

#[test]
fn intermediate_compile_failures_do_not_spend_or_exhaust_recovery() {
    let mut state = TaskRecoveryState::default();
    // Adding an enum variant, its parser, and handlers can report the
    // same missing arm several times, including across a resumed turn.
    for (index, diagnostic) in [
        "missing arm",
        "missing arm",
        "missing method",
        "missing arm",
        "missing arm",
    ]
    .into_iter()
    .enumerate()
    {
        let observation = ValidationObservation {
            execution_id: format!("edit-{index}"),
            scope: "cargo check".into(),
            input_revision: format!("rev-{index}"),
            status: ValidationResult::Failed,
            diagnostics: failures(&[diagnostic]),
            required_stage: false,
            equivalent_scopes: Vec::new(),
        };
        state.observe_feedback(&observation);
        assert!(!state.exhausted);
        assert!(!state.pending_validation_correction);
        assert_eq!(state.remaining, state.limit);
        assert_eq!(
            state.unresolved_validation_status(&observation.input_revision),
            Some(ValidationResult::Failed)
        );
        state = serde_json::from_str(&serde_json::to_string(&state).unwrap()).unwrap();
    }
    let final_check = ValidationObservation::command(
        "final-check".into(),
        "cargo check",
        "finished".into(),
        ValidationResult::Passed,
        "",
        std::path::Path::new("."),
        true,
    );
    state.observe(&final_check);
    assert_eq!(state.unresolved_validation_status("finished"), None);
}

#[test]
fn feedback_does_not_hide_failures_or_cancel_an_existing_repair() {
    let mut state = TaskRecoveryState::default();
    state.request_correction("required test failed");
    let failure = ValidationObservation::command(
        "fast-check".into(),
        "cargo check",
        "rev".into(),
        ValidationResult::Failed,
        "error[E0382]: partial move\n --> src/server.rs:482:4",
        std::path::Path::new("."),
        false,
    );
    state.observe_feedback(&failure);
    assert!(state.pending_validation_correction);
    assert_eq!(
        state.unresolved_validation_status("rev"),
        Some(ValidationResult::Failed)
    );
    assert_eq!(
        state.unresolved_validation_status("changed"),
        Some(ValidationResult::Deferred)
    );
    let mut final_failure = failure.clone();
    final_failure.execution_id = "final-check".into();
    final_failure.required_stage = true;
    state.observe(&final_failure);
    assert!(
        !state.exhausted,
        "the first final check is not a feedback cycle"
    );
    assert!(state.pending_validation_correction);
    assert!(state.intervene("fix check"));
    final_failure.execution_id = "repeated-final-check".into();
    state.observe(&final_failure);
    assert!(
        state.exhausted,
        "explicit failed verification loops remain bounded"
    );
}

#[test]
fn changing_reasons_or_restoring_a_session_cannot_replenish_recovery() {
    let mut state = TaskRecoveryState::default();
    for reason in ["review", "protocol", "verification"] {
        assert!(state.intervene(reason));
    }
    let mut restored: TaskRecoveryState =
        serde_json::from_str(&serde_json::to_string(&state).unwrap()).unwrap();
    assert!(!restored.intervene("a different label"));
    restored.observe_requested_mutation();
    assert!(restored.exhausted);
}

#[test]
fn terminal_observations_retain_the_stop_cause_and_never_schedule_corrections() {
    let mut state = TaskRecoveryState::default();
    state.request_correction("previous repair");
    state.stop("provider terminated recovery: 4/4 sends");
    for revision in ["old", "current", "old"] {
        state.observe(&ValidationObservation::command(
            format!("check-{revision}-{}", state.observed_executions.len()),
            "cargo check",
            revision.into(),
            ValidationResult::Failed,
            "error[E0382]: partial move\n --> src/server.rs:482:4",
            std::path::Path::new("."),
            true,
        ));
        assert_eq!(
            state.unresolved_validation_status(revision),
            Some(ValidationResult::Failed)
        );
        assert!(!state.pending_validation_correction);
        assert!(!state.intervene("late repair"));
        state.stop("later generic exhaustion");
        assert_eq!(
            state.last_reason.as_deref(),
            Some("provider terminated recovery: 4/4 sends")
        );
    }
    let mut restored: TaskRecoveryState =
        serde_json::from_str(&serde_json::to_string(&state).unwrap()).unwrap();
    restored.observe(&ValidationObservation::command(
        "final-pass".into(),
        "cargo check",
        "repaired".into(),
        ValidationResult::Passed,
        "",
        std::path::Path::new("."),
        true,
    ));
    assert_eq!(restored.unresolved_validation_status("repaired"), None);
    assert!(restored.exhausted);
    assert_eq!(restored.remaining, 0);
    assert!(!restored.pending_validation_correction);
    assert_eq!(
        restored.last_reason.as_deref(),
        Some("provider terminated recovery: 4/4 sends")
    );
}

#[test]
fn changed_diagnostics_on_fresh_revisions_replenish() {
    let mut state = TaskRecoveryState::default();
    state.observe_validation("full", "a", failures(&["x", "y"]), false, true);
    assert!(state.intervene("fix"));
    state.observe_validation("narrow", "b", None, true, false);
    assert_eq!(state.remaining, 2);
    state.observe_validation("full", "b", failures(&["z", "w", "v"]), false, true);
    assert_eq!(state.remaining, 3);
    assert!(state.intervene("fix"));
    state.observe_validation("full", "c", failures(&["x", "y"]), false, true);
    state.observe_validation("full", "d", failures(&["x"]), false, true);
    assert!(
        !state.exhausted,
        "diagnostics can recur on different source revisions"
    );
    assert_eq!(state.remaining, 3);
    state.observe_validation("full", "a", failures(&["x", "y"]), false, true);
    assert!(
        state.exhausted,
        "revisiting the same failed workspace still stops"
    );
}

#[test]
fn alternating_failed_inputs_stop() {
    let mut state = TaskRecoveryState::default();
    state.observe_validation("test", "a", failures(&["x"]), false, true);
    assert!(state.intervene("fix"));
    state.observe_validation("test", "b", failures(&["x"]), false, true);
    assert!(state.intervene("fix again"));
    state.observe_validation("test", "a", failures(&["x"]), false, true);
    assert!(state.exhausted);
}

#[test]
fn diagnostic_positions_and_output_order_do_not_replenish_recovery() {
    let mut state = TaskRecoveryState::default();
    let observe = |id: &str, revision: &str, output: &str| {
        ValidationObservation::command(
            id.into(),
            "cargo test --workspace",
            revision.into(),
            ValidationResult::Failed,
            output,
            std::path::Path::new("."),
            true,
        )
    };
    state.observe(&observe(
        "first",
        "a",
        "error[E0308]: mismatched types\n --> src/main.rs:10:2",
    ));
    assert!(state.intervene("repair"));
    state.observe(&observe(
        "second",
        "b",
        "error[E0308]: mismatched types\n --> src/main.rs:90:8",
    ));
    assert_eq!(
        state.remaining, 2,
        "moved source positions do not constitute progress"
    );
    state.observe(&observe(
        "second",
        "b",
        "error[E0308]: mismatched types\n --> src/main.rs:90:8",
    ));
    assert!(
        !state.exhausted,
        "duplicate completion callbacks are ignored"
    );
}

#[test]
fn repeated_infrastructure_and_new_unrelated_green_checks_do_not_buy_repairs() {
    let mut state = TaskRecoveryState::default();
    assert!(state.intervene("repair"));
    for (id, status) in [
        ("infra", ValidationResult::Infrastructure),
        ("defer", ValidationResult::Deferred),
        ("pass", ValidationResult::Passed),
    ] {
        state.observe(&ValidationObservation::command(
            id.into(),
            "unrelated narrow check",
            "a".into(),
            status,
            "",
            std::path::Path::new("."),
            false,
        ));
    }
    assert_eq!(state.remaining, 2);
    assert!(!state.exhausted);
}

#[test]
fn pending_corrections_survive_resume_and_do_not_reopen_exhaustion() {
    let mut state = TaskRecoveryState::default();
    for _ in 0..3 {
        state.request_correction("reviewer objection");
        assert!(!state.exhausted);
        assert!(state.intervene("reviewer objection"));
    }
    let mut restored: TaskRecoveryState =
        serde_json::from_str(&serde_json::to_string(&state).unwrap()).unwrap();
    assert!(restored.pending_validation_correction);
    restored.request_correction("different reviewer label");
    assert!(restored.exhausted);
    restored.observe_required_effect("same already terminal goal".into());
    assert!(restored.exhausted);
}

#[test]
fn a_required_effect_is_credited_once() {
    let mut state = TaskRecoveryState::default();
    assert!(state.intervene("repair"));
    state.observe_required_effect("goal:fixed-objective:0".into());
    assert_eq!(state.remaining, 3);
    assert!(state.intervene("repair"));
    state.observe_required_effect("goal:fixed-objective:0".into());
    assert_eq!(state.remaining, 2);
}

#[test]
fn malformed_or_future_recovery_is_rejected() {
    let mut state = TaskRecoveryState::default();
    state.schema_version += 1;
    assert!(state.validate().is_err());
    state = TaskRecoveryState::default();
    state.remaining += 1;
    assert!(state.validate().is_err());
}
