use super::*;

fn temp_root(label: &str) -> std::path::PathBuf {
    let root = std::env::temp_dir().join(format!(
        "hi-plan-goal-{label}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&root).unwrap();
    root.canonicalize().unwrap()
}

#[test]
fn parses_and_cleans_planner_output() {
    let raw = "1. Add the parser module\n2) Wire it into main\n- Add a test\n* Update docs\n";
    assert_eq!(
        parse_sub_goals(raw),
        vec![
            "Add the parser module",
            "Wire it into main",
            "Add a test",
            "Update docs",
        ]
    );
}

#[test]
fn drops_blank_lines_without_truncating_large_plans() {
    // Cross the historical 120-milestone ceiling, with blanks interspersed.
    let mut raw = String::from("first\n\n  \n");
    for i in 0..125 {
        raw.push_str(&format!("step {i}\n"));
    }
    let out = parse_sub_goals(&raw);
    assert_eq!(out.len(), 126, "valid planned work must not disappear");
    assert_eq!(out.first().map(String::as_str), Some("first"));
    assert_eq!(out.last().map(String::as_str), Some("step 124"));
}

#[test]
fn single_line_stays_one_step() {
    assert_eq!(
        parse_sub_goals("Fix the off-by-one in count()\n"),
        vec!["Fix the off-by-one in count()"]
    );
}

#[test]
fn empty_output_yields_nothing() {
    assert!(parse_sub_goals("   \n\n").is_empty());
}

#[test]
fn planner_output_falls_back_to_line_list_and_drops_kind_prefix() {
    let plan = parse_planner_output(
        "KIND: code-change\n\
         Add the parser module\n\
         Wire it into main\n",
    );
    assert_eq!(plan.kind, Some(crate::GoalKind::CodeChange));
    assert_eq!(
        plan.milestones,
        vec!["Add the parser module", "Wire it into main"]
    );
}

#[test]
fn planner_preserves_objective_tail_past_the_old_boundary() {
    let root = temp_root("huge-objective");
    const OLD_OBJECTIVE_LIMIT: usize = 32 * 1024;
    let objective = format!(
        "{}FINAL PLANNER REQUIREMENT MUST SURVIVE",
        "O".repeat(OLD_OBJECTIVE_LIMIT + 200)
    );
    let input = planner_input(&root, &objective);
    assert_eq!(input.text, objective);
    assert!(
        input
            .text
            .ends_with("FINAL PLANNER REQUIREMENT MUST SURVIVE")
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn planner_reads_explicit_workspace_plan_before_decomposing() {
    let root = temp_root("referenced-plan");
    std::fs::write(
        root.join("plan.md"),
        "Implement the parser, wire the CLI, and pass the acceptance suite.",
    )
    .unwrap();
    let input = planner_input(&root, "review the plan.md document and fully build this");
    assert!(input.text.contains("<workspace-document path=\"plan.md\">"));
    assert!(input.text.contains("wire the CLI"));
    assert_eq!(input.docs.len(), 1);
    assert_eq!(input.docs[0].0, "plan.md");
    assert!(input.docs[0].1.contains("wire the CLI"));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn planner_referenced_files_cannot_escape_workspace() {
    let parent = temp_root("contained");
    let root = parent.join("workspace");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(parent.join("secret.md"), "outside-secret-marker").unwrap();
    let input = planner_input(&root, "review ../secret.md and build it");
    assert!(!input.text.contains("outside-secret-marker"));
    assert!(input.docs.is_empty());
    std::fs::remove_dir_all(parent).unwrap();
}

#[test]
fn read_review_milestones_are_filtered() {
    let steps = vec![
        "Read the supplied workspace documents to identify requirements".to_string(),
        "Investigate the lifecycle evidence and record the conclusion".to_string(),
        "Review plan.md and implement the parser".to_string(),
        "Implement the tensor-inventory crate".to_string(),
    ];
    let kept = drop_meta_milestones(steps);
    assert_eq!(kept.len(), 2, "pure-read milestone dropped: {kept:?}");
    assert!(
        kept[0].contains("implement the parser"),
        "read+implement kept"
    );
    // Never empties the list.
    let all_read = vec!["Read the documents".to_string()];
    assert_eq!(drop_meta_milestones(all_read.clone()), all_read);

    for title in [
        "Inspect the current request lifecycle",
        "Investigate the lifecycle evidence",
        "Audit cancellation behavior",
        "Audit build logs",
        "Audit: build logs",
        "Review recent build output",
        "Inspect patch behavior",
        "Review the update strategy",
        "Trace the provider request path",
        "Understand fixture behavior",
        "Review the updated fixture behavior",
    ] {
        assert!(
            is_meta_milestone(title),
            "{title:?} must stay investigative"
        );
    }
    assert!(
        !is_meta_milestone("Investigate the lifecycle and fix cancellation cleanup"),
        "a later implementation clause must keep the step mutation-aware"
    );
    for title in [
        "Review current code before implementing the parser",
        "Audit logs, then wire cancellation cleanup",
        "Inspect the service and persist the repaired state",
    ] {
        assert!(
            !is_meta_milestone(title),
            "{title:?} contains a later implementation action"
        );
    }
}

#[test]
fn validation_only_milestones_are_filtered() {
    // The qtest failure: an executor-appended "Final workspace validation"
    // milestone is unwinnable (honest no-edit validation turns classify as
    // stalls) and killed a 20/21-done goal.
    for title in [
        "Validate the full workspace",
        "Verify the application runs its primary workflow without errors",
        "Confirm the release artifacts are complete",
        "Run the full test suite and confirm everything passes",
        "Rerun the integration suite",
        "Re-run the cancellation regression",
        "Execute all smoke scenarios",
        "Check the persisted session",
        "Test the primary workflow",
        "Perform release verification",
        "Final workspace validation",
        "Full validation of all components",
        "End-to-end verification",
        "Overall workspace validation",
    ] {
        assert!(is_validation_milestone(title), "{title:?} is validation");
        assert!(is_meta_milestone(title), "{title:?} is meta-work");
        assert!(
            plan_step_requires_execution_evidence(title),
            "{title:?} must be reopened when its effects are rolled back"
        );
    }
    // Real work survives — an implementation verb keeps the line.
    assert!(!is_meta_milestone(
        "Run the full test suite and fix any failing tests"
    ));
    assert!(!is_validation_milestone(
        "Run the full test suite and fix any failing tests"
    ));
    assert!(!is_meta_milestone("Write and run integration tests"));
    assert!(!is_meta_milestone(
        "Implement the tensor-inventory crate with validation"
    ));
    // Filter never empties the list.
    let only_meta = vec!["Final workspace validation".to_string()];
    assert_eq!(drop_meta_milestones(only_meta.clone()), only_meta);
}

fn quant_doc() -> Vec<(String, String)> {
    vec![(
        "plan.md".to_string(),
        "Quantization-aware training for the GLM transformer: binary and ternary \
         fake-quantization with group-128 scales, teacher distillation losses, CUDA \
         GEMV decode kernels, artifact packing manifests, expert coverage tracking, \
         progressive quantization schedules, inference runtime backends. Quantization \
         kernels, distillation, quantization schedules, teacher logits, expert routing, \
         GEMV kernels, artifact manifests, runtime backends, transformer layers."
            .to_string(),
    )]
}

#[test]
fn doc_overlap_accepts_grounded_decomposition() {
    let steps = vec![
        "Implement binary fake-quantization with group-128 scales".to_string(),
        "Implement CUDA GEMV decode kernels".to_string(),
        "Add teacher distillation losses".to_string(),
        "Run the full acceptance suite".to_string(), // generic line tolerated
    ];
    assert!(decomposition_grounded(&steps, &quant_doc()).is_ok());
}

#[test]
fn doc_overlap_rejects_generic_web_plan() {
    // The observed production failure: a quant-training doc decomposed into
    // generic web-app milestones.
    let steps = vec![
        "Implement all missing frontend UI components and pages".to_string(),
        "Set up authentication and API endpoints".to_string(),
        "Add client-side state management".to_string(),
    ];
    let unmatched = decomposition_grounded(&steps, &quant_doc()).unwrap_err();
    assert_eq!(unmatched.len(), 3, "all three named in the retry message");
}

#[test]
fn doc_overlap_skipped_for_tiny_docs_and_no_docs() {
    let steps = vec!["Anything at all".to_string()];
    let tiny = vec![("note.md".to_string(), "fix the bug".to_string())];
    assert!(decomposition_grounded(&steps, &tiny).is_ok());
    assert!(decomposition_grounded(&steps, &[]).is_ok());
}
