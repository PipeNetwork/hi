use super::common::{
    IsolatedWorkspace, ProviderStep, RecUi, bash_completion, completion, scripted_agent,
};
use super::*;

fn edit_state(before: &str, value: &str) -> Completion {
    completion(
        vec![Content::ToolCall {
            id: format!("state-{value}"),
            name: "apply_patch".into(),
            arguments: serde_json::json!({"patch":format!("*** Begin Patch\n*** Update File: state.rs\n@@\n-{before}\n+{value}\n*** End Patch")}).to_string(),
        }],
        1,
        1,
    )
}

async fn exhausted_edit_case(
    final_state: &str,
    verification_enabled: bool,
    verification_cap_spent: bool,
) {
    let workspace = IsolatedWorkspace::new("terminal-recovery-verification");
    let receipts = tempfile::NamedTempFile::new().unwrap();
    let receipt_path = serde_json::to_string(receipts.path().to_str().unwrap()).unwrap();
    std::fs::write(workspace.path("state.rs"), "baseline\n").unwrap();
    // An unchanged failure set keeps this fixture focused on exhaustion.
    // Newly revealed diagnostics correctly earn another repair attempt.
    let diagnostic_state = if verification_cap_spent && final_state == "still-broken" {
        "'still-broken'"
    } else {
        "state"
    };
    std::fs::write(workspace.path("validate.py"), format!(
        "from pathlib import Path\nimport sys\nstate = Path('state.rs').read_text().strip()\nwith open({receipt_path}, 'a') as receipt: receipt.write(state + '\\n')\nif state == 'unavailable':\n print('Operation not permitted')\n sys.exit(1)\nif state != 'fixed':\n print('test current::' + {diagnostic_state} + ' ... FAILED')\n sys.exit(1)\nprint('test result: ok. 1 passed; 0 failed')\n"
    )).unwrap();
    let command = "python3 validate.py";
    let mut cfg = workspace.config();
    cfg.loop_limits.max_recovery_interventions = 1;
    if verification_cap_spent {
        cfg.gates.max_verify_repairs = 0;
    }
    cfg.gates.verification = if verification_enabled {
        VerificationMode::Explicit(vec![VerifyStage::new("retained edits", command)])
    } else {
        VerificationMode::Disabled
    };
    // A passing terminal check must not enter either optional model path.
    cfg.gates.review = ReviewPolicy::Always;
    cfg.memory.curate_skills = true;
    cfg.memory.suggest_next_prompt = true;
    cfg.memory.finalize = true;
    let current_check_failed = verification_cap_spent && final_state == "still-broken";
    // Invalid tool JSON is format steering: consecutive retries, then a
    // reserved Auto implementation text-tool fallback, then settlement.
    // One error used to end the turn because it shared the recovery budget.
    let protocol_exhaustion = || {
        (0..crate::MAX_TOOL_PROTOCOL_RETRIES)
            .map(|_| ProviderStep::Error(ProviderErrorKind::ToolProtocol))
    };
    let steps = if verification_cap_spent {
        let mut steps = vec![
            ProviderStep::Completion(edit_state("baseline", "broken")),
            ProviderStep::Completion(bash_completion("python3 -c 'assert 2 + 2 == 4'")),
            ProviderStep::Completion(completion(vec![Content::Text("Updated state.rs with the requested implementation and checked the edited source. The source is ready for the configured verification stage.".into())], 1, 1)),
            ProviderStep::Completion(edit_state("broken", final_state)),
        ];
        if current_check_failed {
            steps.push(ProviderStep::Completion(bash_completion(command)));
        } else {
            steps.extend(protocol_exhaustion());
        }
        steps
    } else {
        let mut steps = vec![
            ProviderStep::Completion(edit_state("baseline", "broken")),
            ProviderStep::Completion(bash_completion(command)),
            ProviderStep::Completion(edit_state("broken", final_state)),
        ];
        steps.extend(protocol_exhaustion());
        steps
    };
    let (mut subject, requests) = scripted_agent(steps, cfg);
    let mut ui = RecUi::default();
    let outcome = subject
        .run_turn(
            "build the requested implementation in state.rs and check it",
            &mut ui,
        )
        .await
        .unwrap();
    let protocol_sends = crate::MAX_TOOL_PROTOCOL_RETRIES as usize;
    assert_eq!(
        requests.lock().unwrap().len(),
        if current_check_failed {
            5
        } else if verification_cap_spent {
            4 + protocol_sends
        } else {
            3 + protocol_sends
        },
        "no model request after protocol and recovery exhaustion"
    );
    assert_eq!(
        std::fs::read_to_string(workspace.path("state.rs"))
            .unwrap()
            .trim(),
        final_state
    );
    assert!(subject.task_recovery().exhausted);
    assert_eq!(subject.task_recovery().remaining, 0);
    assert_eq!(subject.task_recovery().interventions, 1);
    if current_check_failed {
        assert_eq!(outcome.status, TurnStatus::Failed);
        assert_eq!(outcome.stop_reason, TurnStopReason::NoProgress);
    } else if verification_enabled && !verification_cap_spent && final_state == "fixed" {
        assert_eq!(outcome.status, TurnStatus::Completed);
        assert_ne!(
            outcome.stop_reason,
            TurnStopReason::NoProgress,
            "protocol after retained edits is leftover, not a failed stall"
        );
    } else if verification_enabled && final_state == "still-broken" {
        assert_eq!(outcome.status, TurnStatus::Failed);
        assert_eq!(outcome.stop_reason, TurnStopReason::VerificationFailed);
    } else {
        assert_ne!(
            outcome.stop_reason,
            TurnStopReason::NoProgress,
            "protocol after mutations must not be no_progress: {outcome:?}"
        );
    }
    assert_eq!(
        subject.last_turn_telemetry().verify_rounds,
        u32::from(verification_enabled)
    );
    let checks = std::fs::read_to_string(receipts.path()).unwrap();
    assert_eq!(
        checks.lines().filter(|line| *line == final_state).count(),
        usize::from(current_check_failed || (verification_enabled && !verification_cap_spent)),
        "only one final check of these exact bytes"
    );
    match (verification_enabled && !verification_cap_spent, final_state) {
        (true, "fixed") => {
            assert_eq!(outcome.verification, VerificationStatus::Passed);
            assert_eq!(subject.last_verify(), Some(true));
            assert_eq!(
                outcome.verified_workspace_revision.as_deref(),
                Some(subject.runtime.ledger().workspace_revision().as_str())
            );
            assert!(
                ui.assistant.contains("Final verification passed")
                    || ui
                        .statuses
                        .iter()
                        .any(|status| status.to_ascii_lowercase().contains("verification passed")),
                "terminal check must still run: assistant={} statuses={:?}",
                ui.assistant,
                ui.statuses
            );
            assert!(!ui.assistant.contains("current::broken"));
        }
        (_, "still-broken") => {
            assert_eq!(outcome.verification, VerificationStatus::Failed);
            assert_eq!(subject.last_verify(), Some(false));
            assert!(ui.assistant.contains("current::still-broken"));
            assert!(
                ui.assistant
                    .contains("failed for the current workspace revision")
            );
            if current_check_failed {
                assert!(
                    ui.statuses
                        .iter()
                        .any(|status| status.contains("latest applicable check still fails"))
                );
                assert!(
                    !ui.statuses
                        .iter()
                        .any(|status| status.contains("retained edits remain unverified"))
                );
            }
        }
        _ => {
            assert_eq!(outcome.verification, VerificationStatus::Unverified);
            assert!(outcome.verified_workspace_revision.is_none());
            assert!(
                ui.assistant
                    .contains("Earlier failure at workspace revision")
            );
            assert!(ui.assistant.contains("current revision remains unverified"));
            assert!(ui.assistant.contains("current::broken"));
        }
    }
}

#[tokio::test]
async fn exhausted_recovery_still_verifies_the_last_corrective_edit() {
    exhausted_edit_case("fixed", true, false).await;
}

#[tokio::test]
async fn exhausted_recovery_records_a_fresh_failure_without_another_repair() {
    exhausted_edit_case("still-broken", true, false).await;
}

#[tokio::test]
async fn exhausted_recovery_labels_old_failures_when_final_check_is_unavailable() {
    exhausted_edit_case("unavailable", true, false).await;
}

#[tokio::test]
async fn exhausted_recovery_respects_disabled_verification_and_labels_stale_evidence() {
    exhausted_edit_case("fixed", false, false).await;
}

#[tokio::test]
async fn exhausted_recovery_does_not_extend_an_explicit_verification_cap() {
    exhausted_edit_case("fixed", true, true).await;
}

#[tokio::test]
async fn exhausted_verification_cap_preserves_a_current_failed_check_in_status() {
    exhausted_edit_case("still-broken", true, true).await;
}
