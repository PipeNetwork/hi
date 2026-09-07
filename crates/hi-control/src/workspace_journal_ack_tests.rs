use super::*;

#[tokio::test(flavor = "current_thread")]
async fn dropped_seal_waiter_retains_late_acknowledgement_and_clears_fence() {
    let (_directory, store) = store();
    let inner: Arc<dyn WorkspaceController> = Arc::new(InMemoryWorkspaceController::new_local(
        "late-seal",
        "/work",
        "/state",
    ));
    let controller = Arc::new(
        JournaledWorkspaceController::attach(inner.clone(), Arc::new(store.clone())).unwrap(),
    );
    let job = controller
        .register_job(candidate_spec("candidate"))
        .await
        .unwrap();
    for completion in [
        JobCompletion::ReadyToMerge,
        JobCompletion::Merging,
        JobCompletion::Settling,
    ] {
        assert_eq!(
            controller
                .seal_job(
                    job.job_id.clone(),
                    JobTerminal {
                        completion,
                        detail: None,
                        artifacts: Vec::new(),
                    }
                )
                .await
                .status,
            JobSealStatus::Sealed
        );
    }
    let blocker = rusqlite::Connection::open(store.path()).unwrap();
    blocker.execute_batch("BEGIN IMMEDIATE").unwrap();
    let mut changes = controller.subscribe();
    let waiter = {
        let controller = controller.clone();
        let id = job.job_id.clone();
        tokio::spawn(async move {
            controller
                .seal_job(
                    id,
                    JobTerminal {
                        completion: JobCompletion::Succeeded,
                        detail: None,
                        artifacts: Vec::new(),
                    },
                )
                .await
        })
    };
    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        while inner.job_state(&job.job_id) != Some(JobState::Succeeded) {
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        }
        assert_eq!(controller.job_state(&job.job_id), Some(JobState::Settling));
    })
    .await
    .unwrap();
    waiter.abort();
    assert!(waiter.await.unwrap_err().is_cancelled());
    assert!(controller.status().active_jobs.contains(&job.job_id));
    blocker.execute_batch("ROLLBACK").unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        while controller.job_state(&job.job_id) != Some(JobState::Succeeded) {
            changes.changed().await.unwrap();
        }
    })
    .await
    .unwrap();
    assert!(!controller.status().active_jobs.contains(&job.job_id));
    assert_eq!(
        store.get_job(job.job_id.as_str()).unwrap().unwrap().state,
        ControlJobState::Succeeded
    );
    assert_eq!(
        controller.journal_health().state,
        JournalHealthState::Healthy
    );
}
