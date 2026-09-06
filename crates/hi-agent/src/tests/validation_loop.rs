use super::common::*;
use super::*;

#[tokio::test]
async fn changing_validation_output_selectors_cannot_loop_on_the_same_result() {
    // Live regression: DeepSeek ran `cargo clippy | grep src/main.rs:N | head`
    // hundreds of times, incrementing N while every call returned no output.
    // The varying selector defeated exact-call comparison, and validation was
    // allowed to reset progress forever. Use the lightweight validation
    // fixture here so this test does not contend on Cargo's workspace lock.
    let mut responses = Vec::new();
    for line in 1605..1611 {
        responses.push(bash_completion(&format!(
            "true # validate 2>&1 | grep 'src/main.rs:{line}' | head -40"
        )));
    }
    // If the convergence guard regresses, the provider reaches this answer
    // and the turn falsely succeeds instead of reporting typed no-progress.
    responses.push(completion(
        vec![Content::Text(
            "Incorrectly escaped the validation loop.".into(),
        )],
        1,
        1,
    ));

    let mut cfg = config();
    cfg.loop_limits.max_repeat_nudges = 1;
    cfg.loop_limits.max_keep_working = 0;
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();

    let outcome = agent
        .run_turn("run the validation and report the result", &mut ui)
        .await
        .unwrap();

    assert_eq!(outcome.status, crate::TurnStatus::Failed);
    assert_eq!(outcome.stop_reason, crate::TurnStopReason::NoProgress);
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("validation for unchanged workspace bytes")),
        "expected a validation-specific convergence nudge: {:?}",
        ui.statuses
    );
    assert!(
        ui.tool_results
            .iter()
            .filter(|(name, _)| name == "bash")
            .count()
            <= 3,
        "selector churn must be bounded before the scripted tail is consumed: {:?}",
        ui.tool_results
    );
    assert!(!ui.assistant.contains("Incorrectly escaped"));
}

#[tokio::test]
async fn weak_companion_call_cannot_hide_a_repeated_validation_result() {
    let mut responses = Vec::new();
    for line in 1605..1611 {
        responses.push(completion(
            vec![
                Content::ToolCall {
                    id: format!("validate-{line}"),
                    name: "bash".into(),
                    arguments: serde_json::json!({
                        "command": format!(
                            "true # validate 2>&1 | grep 'src/main.rs:{line}' | head -40"
                        )
                    })
                    .to_string(),
                },
                Content::ToolCall {
                    id: format!("noise-{line}"),
                    name: "bash".into(),
                    arguments: serde_json::json!({
                        "command": format!("echo checking-line-{line}")
                    })
                    .to_string(),
                },
            ],
            1,
            1,
        ));
    }

    let mut cfg = config();
    cfg.loop_limits.max_repeat_nudges = 1;
    cfg.loop_limits.max_keep_working = 0;
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    let outcome = agent
        .run_turn("run the validation and report the result", &mut ui)
        .await
        .unwrap();

    assert_eq!(outcome.stop_reason, crate::TurnStopReason::NoProgress);
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("validation for unchanged workspace bytes"))
    );
    assert!(ui.tool_results.len() <= 6, "batch churn was not bounded");
}
