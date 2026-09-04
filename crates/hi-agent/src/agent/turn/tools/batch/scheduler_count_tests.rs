use super::{
    ProgramRunGuard, normalize_unsupported_plan_completion, plan_step_requires_execution_evidence,
    saturating_add_scheduler_count,
};
use hi_tools::{PlanStatus, PlanStep};

fn step(title: &str, status: PlanStatus) -> PlanStep {
    PlanStep {
        title: title.into(),
        status,
    }
}

#[test]
fn implementation_plan_completion_requires_execution_evidence() {
    let title = "Build VoteInstruction::Vote transaction from tower decision";
    assert!(plan_step_requires_execution_evidence(title));
    assert!(plan_step_requires_execution_evidence(
        "Wire vote transaction signing"
    ));
    assert!(plan_step_requires_execution_evidence(
        "Persist vote state in SQLite"
    ));
    assert!(plan_step_requires_execution_evidence(
        "Run the final test suite"
    ));
    assert!(
        !plan_step_requires_execution_evidence("Understand fixture behavior"),
        "a substring of a read-only subject must not look like an implementation verb"
    );
    let current = vec![step(title, PlanStatus::Active)];
    let mut proposed = vec![step(title, PlanStatus::Done)];
    let arguments = serde_json::json!({
        "steps": [{"title": title, "status": "done"}]
    })
    .to_string();

    assert_eq!(
        normalize_unsupported_plan_completion(&current, &mut proposed, &arguments, false,),
        vec![0]
    );
    assert_eq!(proposed[0].status, PlanStatus::Active);

    let mut with_execution = vec![step(title, PlanStatus::Done)];
    assert_eq!(
        normalize_unsupported_plan_completion(&current, &mut with_execution, &arguments, true,),
        Vec::<usize>::new()
    );
    assert_eq!(with_execution[0].status, PlanStatus::Done);

    let current = vec![
        step("Build parser", PlanStatus::Active),
        step("Persist parser state", PlanStatus::Pending),
    ];
    let mut bulk_done = vec![
        step("Build parser", PlanStatus::Done),
        step("Persist parser state", PlanStatus::Done),
    ];
    let arguments = serde_json::json!({
        "steps": [
            {"title": "Build parser", "status": "done"},
            {"title": "Persist parser state", "status": "done"}
        ]
    })
    .to_string();
    assert_eq!(
        normalize_unsupported_plan_completion(&current, &mut bulk_done, &arguments, true,),
        vec![1],
        "one turn-global mutation must not self-certify every plan step"
    );
    assert_eq!(bulk_done[0].status, PlanStatus::Done);
    assert_eq!(bulk_done[1].status, PlanStatus::Pending);
}

#[test]
fn explicit_per_step_no_change_evidence_can_complete_implementation() {
    let title = "Implement compatibility shim";
    let current = vec![step(title, PlanStatus::Active)];
    let mut proposed = vec![step(title, PlanStatus::Done)];
    let arguments = serde_json::json!({
        "steps": [{
            "title": title,
            "status": "done",
            "completion_evidence": "Existing implementation already handles both formats; focused test passed."
        }]
    })
    .to_string();

    assert_eq!(
        normalize_unsupported_plan_completion(&current, &mut proposed, &arguments, false,),
        Vec::<usize>::new()
    );
    assert_eq!(proposed[0].status, PlanStatus::Done);

    for generic in ["done", "already done", "no changes required"] {
        let mut unsupported = vec![step(title, PlanStatus::Done)];
        let arguments = serde_json::json!({
            "steps": [{
                "title": title,
                "status": "done",
                "completion_evidence": generic
            }]
        })
        .to_string();
        assert_eq!(
            normalize_unsupported_plan_completion(&current, &mut unsupported, &arguments, false,),
            vec![0],
            "generic evidence {generic:?} bypassed the guard"
        );
        assert_eq!(unsupported[0].status, PlanStatus::Active);
    }
}

#[test]
fn read_only_or_previously_done_steps_are_not_reopened() {
    let mut inspection = vec![step("Inspect the vote path", PlanStatus::Done)];
    assert_eq!(
        normalize_unsupported_plan_completion(
            &[step("Inspect the vote path", PlanStatus::Active)],
            &mut inspection,
            r#"{"steps":[{"title":"Inspect the vote path","status":"done"}]}"#,
            false,
        ),
        Vec::<usize>::new()
    );

    let title = "Build vote transaction";
    let mut retained = vec![step(title, PlanStatus::Done)];
    assert_eq!(
        normalize_unsupported_plan_completion(
            &[step(title, PlanStatus::Done)],
            &mut retained,
            r#"{"steps":[{"title":"Build vote transaction","status":"done"}]}"#,
            false,
        ),
        Vec::<usize>::new()
    );
}

#[test]
fn scheduler_count_saturates_instead_of_wrapping() {
    let mut total = u32::MAX - 1;
    saturating_add_scheduler_count(&mut total, 2);
    assert_eq!(total, u32::MAX);
    saturating_add_scheduler_count(&mut total, 1);
    assert_eq!(total, u32::MAX);
}

#[test]
#[cfg(target_pointer_width = "64")]
fn scheduler_count_clamps_oversized_host_counts() {
    let mut total = 0;
    saturating_add_scheduler_count(&mut total, (u32::MAX as usize) + 1);
    assert_eq!(total, u32::MAX);
}

#[tokio::test]
async fn dropping_program_run_guard_cancels_unlimited_blocking_worker() {
    let cancel = tokio_util::sync::CancellationToken::new();
    let (host_tx, _host_rx) = tokio::sync::mpsc::unbounded_channel();
    let params = hi_workflow::ProgramRunParams {
        source: "loop {}".into(),
        host_tx,
        cancel: cancel.clone(),
        max_ops: hi_workflow::ProgramRunParams::DEFAULT_MAX_OPS,
        max_calls: None,
    };
    let worker = tokio::task::spawn_blocking(move || hi_workflow::run_program(params));
    let watcher = tokio::spawn(std::future::pending::<()>());
    let guard = ProgramRunGuard::new(cancel.clone(), watcher);

    tokio::task::yield_now().await;
    drop(guard);

    assert!(cancel.is_cancelled());
    let outcome = tokio::time::timeout(std::time::Duration::from_secs(2), worker)
        .await
        .expect("drop cancellation should stop the unlimited program")
        .expect("program worker should join cleanly");
    assert!(matches!(
        outcome,
        hi_workflow::ProgramOutcome::Cancelled { .. }
    ));
}
