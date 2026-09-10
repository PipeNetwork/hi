use super::*;

#[tokio::test]
async fn skeptic_context_includes_prior_notes_for_anti_ratchet() {
    // On a re-review the skeptic must see the step's prior notes so it can
    // confirm they're addressed rather than raising fresh objections.
    let workspace = IsolatedWorkspace::new("goal-skeptic-prior-notes");
    let mut cfg = workspace.config();
    cfg.subagents.long_horizon = true;
    cfg.gates.verification =
        crate::VerificationMode::Explicit(vec![VerifyStage::new("test", "true")]);
    cfg.subagents.skeptic_model = Some("skeptic".into());
    cfg.gates.review = ReviewPolicy::Off;
    let changed = workspace.path("changed.rs");
    let steps = vec![
        ProviderStep::Completion(write_content_completion(
            &changed.to_string_lossy(),
            "a substantial implementation body, comfortably past the trivial-diff exemption",
        )),
        ProviderStep::Completion(completion(vec![Content::Text("done".into())], 1, 1)),
        ProviderStep::Completion(completion(vec![Content::Text("APPROVE".into())], 1, 1)),
    ];
    let (mut agent, requests) = scripted_agent(steps, cfg);
    let mut goal = Goal::new("ship it", vec!["step one".into(), "step two".into()]);
    goal.team = true;
    goal.sub_goals[0]
        .notes
        .push("reviewer objected — address then continue:\nthe empty-input case crashes".into());
    agent.set_structured_goal(Some(goal)).unwrap();

    agent.run_turn("go", &mut RecUi::default()).await.unwrap();

    let recorded = requests.lock().unwrap();
    let skeptic_request = recorded
        .last()
        .unwrap()
        .iter()
        .map(Message::text)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        skeptic_request.contains("Prior review notes on this step"),
        "anti-ratchet section present: {skeptic_request}"
    );
    assert!(
        skeptic_request.contains("empty-input case crashes"),
        "the prior note itself is shown"
    );
    assert!(
        skeptic_request.contains("the bar does not rise"),
        "anti-ratchet contract stated in context"
    );
}

#[tokio::test]
async fn completion_audit_input_names_the_round() {
    let workspace = IsolatedWorkspace::new("goal-audit-round-line");
    std::fs::write(workspace.path("plan.md"), quant_plan_doc()).unwrap();
    let changed = workspace.path("changed.rs");
    let (mut agent, requests) = scripted_agent(
        vec![
            ProviderStep::Completion(write_content_completion(
                &changed.to_string_lossy(),
                "a substantial implementation body, comfortably past the trivial-diff exemption",
            )),
            ProviderStep::Completion(completion(vec![Content::Text("done".into())], 1, 1)),
            ProviderStep::Completion(completion(vec![Content::Text("COMPLETE".into())], 1, 1)),
        ],
        audit_cfg(&workspace),
    );
    let mut goal = single_step_goal();
    goal.audit_rounds = 2;
    agent.set_structured_goal(Some(goal)).unwrap();

    agent.run_turn("go", &mut RecUi::default()).await.unwrap();

    let recorded = requests.lock().unwrap();
    let audit_request = recorded
        .iter()
        .map(|messages| {
            messages
                .iter()
                .map(Message::text)
                .collect::<Vec<_>>()
                .join("\n")
        })
        .find(|text| text.contains("Audit round:"))
        .unwrap_or_default();
    assert!(
        audit_request.contains("Audit round: 2"),
        "round number anchors the anti-ratchet rule: {audit_request}"
    );
    assert!(
        audit_request.contains("the bar does NOT rise between rounds"),
        "auditor prompt carries the anti-ratchet contract"
    );
}
