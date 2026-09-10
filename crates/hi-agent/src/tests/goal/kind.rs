use super::*;

const STRUCTURED_DECOMPOSITION: &str = "## Goal kind\n\
code-change\n\
\n\
## Acceptance criteria\n\
1. binary fake-quantization with group-128 scales is implemented\n\
\n\
## Verification plan\n\
1. exercise the shipped quantization path on a representative tensor\n\
\n\
## Task checklist\n\
Implement binary fake-quantization with group-128 scales\n\
Implement CUDA GEMV decode kernels\n\
Add teacher distillation losses\n";

#[tokio::test]
async fn planner_structured_output_carries_kind_and_criteria() {
    let workspace = IsolatedWorkspace::new("planner-structured");
    std::fs::write(workspace.path("plan.md"), quant_plan_doc()).unwrap();
    let mut cfg = workspace.config();
    cfg.subagents.long_horizon = true;
    cfg.subagents.planner_model = Some("planner".into());
    let (mut agent, _) = scripted_agent(
        vec![ProviderStep::Completion(completion(
            vec![Content::Text(STRUCTURED_DECOMPOSITION.into())],
            1,
            1,
        ))],
        cfg,
    );

    let plan = agent
        .decompose_goal("review plan.md and fully build this")
        .await
        .expect("structured planner output is a valid decomposition");

    assert_eq!(plan.kind, Some(crate::GoalKind::CodeChange));
    assert_eq!(plan.milestones.len(), 3);
    assert!(
        plan.acceptance
            .iter()
            .any(|c| c.contains("fake-quantization")),
        "acceptance: {:?}",
        plan.acceptance
    );
    assert!(
        plan.verification
            .iter()
            .any(|v| v.contains("quantization path")),
        "verification: {:?}",
        plan.verification
    );
    let goal = Goal::from_goal_plan("review plan.md and fully build this", plan);
    assert_eq!(goal.kind, crate::GoalKind::CodeChange);
    assert!(!goal.acceptance.is_empty());
    assert!(!goal.verification.is_empty());
}

#[tokio::test]
async fn analysis_goal_advances_on_a_cited_write_up() {
    let workspace = IsolatedWorkspace::new("goal-analysis-writeup");
    let mut cfg = workspace.config();
    cfg.subagents.long_horizon = true;
    cfg.gates.verification = crate::VerificationMode::Disabled;
    let update_plan = completion(
        vec![Content::ToolCall {
            id: "up".into(),
            name: "update_plan".into(),
            arguments: serde_json::json!({
                "steps": [
                    {"title": "name the request path", "status": "done"},
                    {"title": "list failure modes", "status": "active"},
                ]
            })
            .to_string(),
        }],
        1,
        1,
    );
    let write_up = "The auth middleware wraps inbound requests in src/auth.rs:40 by \
calling `require_session` before the handler. Unauthenticated calls return 401; a missing CSRF \
header returns 403. That is the whole request path.";
    let responses = vec![
        update_plan,
        completion(vec![Content::Text(write_up.into())], 1, 1),
    ];
    let mut agent = agent(responses, cfg);
    let mut goal = Goal::new(
        "explain how the auth middleware works",
        vec!["name the request path".into(), "list failure modes".into()],
    );
    assert_eq!(goal.kind, crate::GoalKind::Analysis);
    goal.team = false;
    agent.set_structured_goal(Some(goal)).unwrap();
    agent.run_turn("go", &mut RecUi::default()).await.unwrap();
    let goal = agent.structured_goal().expect("goal still set");
    assert_eq!(
        goal.sub_goals[0].status,
        GoalStatus::Done,
        "analysis write-up must complete the claimed step: {:?}",
        goal.sub_goals
    );
    assert_eq!(goal.active_index(), Some(1));
}

#[tokio::test]
async fn team_panel_majority_objects() {
    let workspace = IsolatedWorkspace::new("goal-panel-majority");
    let mut cfg = workspace.config();
    cfg.subagents.long_horizon = true;
    cfg.subagents.skeptic_count = 3;
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
        ProviderStep::Completion(completion(
            vec![Content::Text("OBJECT\n- missing CSRF path".into())],
            1,
            1,
        )),
        ProviderStep::Completion(completion(
            vec![Content::Text("OBJECT\n- stub in handler".into())],
            1,
            1,
        )),
        ProviderStep::Completion(completion(vec![Content::Text("APPROVE".into())], 1, 1)),
    ];
    let (mut agent, _) = scripted_agent(steps, cfg);
    let mut goal = Goal::new("ship it", vec!["step one".into(), "step two".into()]);
    goal.team = true;
    agent.set_structured_goal(Some(goal)).unwrap();
    agent.run_turn("go", &mut RecUi::default()).await.unwrap();
    let goal = agent.structured_goal().expect("goal");
    assert_eq!(
        goal.active_index(),
        Some(0),
        "majority object keeps the step"
    );
    assert_eq!(goal.skeptic_objections, 1);
    assert!(
        goal.last_gaps.contains("CSRF") || goal.last_gaps.contains("stub"),
        "gaps persisted for the next drive: {}",
        goal.last_gaps
    );
}

#[tokio::test]
async fn analysis_goal_does_not_advance_on_a_short_done() {
    let workspace = IsolatedWorkspace::new("goal-analysis-short");
    let mut cfg = workspace.config();
    cfg.subagents.long_horizon = true;
    cfg.gates.verification = crate::VerificationMode::Disabled;
    let update_plan = completion(
        vec![Content::ToolCall {
            id: "up".into(),
            name: "update_plan".into(),
            arguments: serde_json::json!({
                "steps": [
                    {"title": "name the request path", "status": "done"},
                    {"title": "list failure modes", "status": "active"},
                ]
            })
            .to_string(),
        }],
        1,
        1,
    );
    let responses = vec![
        update_plan,
        completion(vec![Content::Text("Done.".into())], 1, 1),
    ];
    let mut agent = agent(responses, cfg);
    let mut goal = Goal::new(
        "explain how the auth middleware works",
        vec!["name the request path".into(), "list failure modes".into()],
    );
    goal.team = false;
    agent.set_structured_goal(Some(goal)).unwrap();
    agent.run_turn("go", &mut RecUi::default()).await.unwrap();
    let goal = agent.structured_goal().expect("goal still set");
    assert_eq!(
        goal.sub_goals[0].status,
        GoalStatus::Active,
        "a stub recap must not complete analysis: {:?}",
        goal.sub_goals
    );
}

#[tokio::test]
async fn analysis_completion_audit_judges_the_write_up_not_the_repo() {
    let workspace = IsolatedWorkspace::new("goal-analysis-audit");
    let mut cfg = workspace.config();
    cfg.subagents.long_horizon = true;
    cfg.gates.verification = crate::VerificationMode::Disabled;
    cfg.gates.review = ReviewPolicy::Off;
    let write_up = "The auth middleware wraps inbound requests in src/auth.rs:40 by \
calling `require_session` before the handler. Unauthenticated calls return 401; a missing CSRF \
header returns 403. That is the whole request path.";
    let update_plan = completion(
        vec![Content::ToolCall {
            id: "up".into(),
            name: "update_plan".into(),
            arguments: serde_json::json!({
                "steps": [{"title": "name the request path", "status": "done"}]
            })
            .to_string(),
        }],
        1,
        1,
    );
    let (mut agent, requests) = scripted_agent(
        vec![
            ProviderStep::Completion(update_plan),
            ProviderStep::Completion(completion(vec![Content::Text(write_up.into())], 1, 1)),
            ProviderStep::Completion(completion(vec![Content::Text("COMPLETE".into())], 1, 1)),
            ProviderStep::Completion(completion(
                vec![Content::Text(
                    "Delivered the auth middleware write-up.".into(),
                )],
                1,
                1,
            )),
        ],
        cfg,
    );
    let mut goal = Goal::new(
        "explain how the auth middleware works",
        vec!["name the request path".into()],
    );
    goal.team = false;
    agent.set_structured_goal(Some(goal)).unwrap();
    agent.run_turn("go", &mut RecUi::default()).await.unwrap();

    let goal = agent.structured_goal().unwrap();
    assert_eq!(goal.status, GoalStatus::Done);
    assert!(goal.objective_complete);
    let audit_request = requests
        .lock()
        .unwrap()
        .iter()
        .map(|messages| {
            messages
                .iter()
                .map(Message::text)
                .collect::<Vec<_>>()
                .join("\n")
        })
        .find(|text| text.contains("Agent write-up:") || text.contains("Audit round:"))
        .unwrap_or_default();
    assert!(
        audit_request.contains("cited write-up"),
        "analysis auditor, not the repo auditor: {audit_request}"
    );
    assert!(audit_request.contains("Kind: analysis"), "{audit_request}");
    assert!(audit_request.contains("Agent write-up:"), "{audit_request}");
    assert!(
        !audit_request.contains("Repository files (path, bytes):"),
        "repo listing would invite invented code work: {audit_request}"
    );
}

#[tokio::test]
async fn analysis_completion_audit_reopens_on_an_unanswered_question() {
    let workspace = IsolatedWorkspace::new("goal-analysis-audit-gap");
    let mut cfg = workspace.config();
    cfg.subagents.long_horizon = true;
    cfg.gates.verification = crate::VerificationMode::Disabled;
    cfg.gates.review = ReviewPolicy::Off;
    let write_up = "The auth middleware wraps inbound requests in src/auth.rs:40 by \
calling `require_session` before the handler. Unauthenticated calls return 401; a missing CSRF \
header returns 403. That is the whole request path.";
    let update_plan = completion(
        vec![Content::ToolCall {
            id: "up".into(),
            name: "update_plan".into(),
            arguments: serde_json::json!({
                "steps": [{"title": "name the request path", "status": "done"}]
            })
            .to_string(),
        }],
        1,
        1,
    );
    let mut agent = agent(
        vec![
            update_plan,
            completion(vec![Content::Text(write_up.into())], 1, 1),
            completion(
                vec![Content::Text("Cite the CSRF failure path".into())],
                1,
                1,
            ),
        ],
        cfg,
    );
    let mut goal = Goal::new(
        "explain how the auth middleware works",
        vec!["name the request path".into()],
    );
    goal.team = false;
    agent.set_structured_goal(Some(goal)).unwrap();
    agent.run_turn("go", &mut RecUi::default()).await.unwrap();

    let goal = agent.structured_goal().unwrap();
    assert_eq!(goal.status, GoalStatus::Active);
    assert!(
        goal.sub_goals
            .iter()
            .any(|step| step.description.contains("CSRF")),
        "unanswered analysis gap must reopen the goal: {:?}",
        goal.sub_goals
    );
}

#[tokio::test]
async fn identical_panel_gaps_pause_the_drive() {
    let workspace = IsolatedWorkspace::new("goal-same-gaps-stall");
    let mut cfg = workspace.config();
    cfg.subagents.long_horizon = true;
    cfg.gates.verification =
        crate::VerificationMode::Explicit(vec![VerifyStage::new("test", "true")]);
    cfg.subagents.skeptic_model = Some("skeptic".into());
    cfg.gates.review = ReviewPolicy::Off;
    let changed = workspace.path("changed.rs");
    let body = "a substantial implementation body, comfortably past the trivial-diff exemption";
    let object = || {
        completion(
            vec![Content::Text("OBJECT\n- missing CSRF path".into())],
            1,
            1,
        )
    };
    let steps = vec![
        ProviderStep::Completion(write_content_completion(&changed.to_string_lossy(), body)),
        ProviderStep::Completion(completion(vec![Content::Text("done".into())], 1, 1)),
        ProviderStep::Completion(object()),
        ProviderStep::Completion(write_content_completion(
            &changed.to_string_lossy(),
            &format!(
                "{body} round two: {}",
                "more implementation evidence to clear the trivial-diff exemption. ".repeat(4)
            ),
        )),
        ProviderStep::Completion(completion(vec![Content::Text("done".into())], 1, 1)),
        ProviderStep::Completion(object()),
    ];
    let (mut agent, _) = scripted_agent(steps, cfg);
    let mut goal = Goal::new("ship it", vec!["step one".into(), "step two".into()]);
    goal.team = true;
    agent.set_structured_goal(Some(goal)).unwrap();
    let mut ui = RecUi::default();

    agent.run_turn("go", &mut ui).await.unwrap();
    let goal = agent.structured_goal().expect("goal");
    assert!(
        goal.should_auto_drive(),
        "first identical panel still drives"
    );
    assert_eq!(goal.skeptic_objections, 1);

    agent.run_turn("go again", &mut ui).await.unwrap();
    let goal = agent.structured_goal().expect("goal");
    assert!(
        !goal.should_auto_drive(),
        "second identical panel must park the drive"
    );
    assert_eq!(goal.pause_reason, GoalPauseReason::Stall);
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("same verifier gaps twice")),
        "statuses: {:?}",
        ui.statuses
    );
}
