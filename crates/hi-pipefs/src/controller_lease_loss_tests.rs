use super::*;

#[tokio::test]
async fn pushed_lease_uncertainty_closes_admission_until_authority_is_reproven() {
    let (_temporary, controller, session, _server) = subject(false).await;
    session.loss_tx.send_replace(PipeFsLeaseStatus::Uncertain);
    tokio::time::timeout(Duration::from_secs(2), async {
        let mut status = controller.subscribe();
        while status.borrow().state != WorkspaceState::LeaseUncertain {
            status.changed().await.unwrap();
        }
    })
    .await
    .unwrap();

    let denied = controller
        .begin(MutationIntent::workspace("blocked while uncertain"))
        .await
        .unwrap_err();
    assert_eq!(denied.reason, AdmissionDeniedReason::NotReady);
    assert!(controller.status().active_operation.is_none());
    assert!(controller.status().recovery_id.is_none());

    session.loss_tx.send_replace(PipeFsLeaseStatus::Valid);
    tokio::time::timeout(Duration::from_secs(2), async {
        let mut status = controller.subscribe();
        while status.borrow().state != WorkspaceState::Ready {
            status.changed().await.unwrap();
        }
    })
    .await
    .unwrap();
    let permit = controller
        .begin(MutationIntent::workspace("allowed after refresh"))
        .await
        .unwrap();
    let outcome = controller
        .settle(permit, ExecutionReport::succeeded(None))
        .await;
    assert_eq!(outcome.status, SettlementStatus::NoChange);
}

#[tokio::test]
async fn pushed_lease_loss_without_active_operation_retains_dirty_recovery() {
    let (_temporary, controller, session, _server) = subject(false).await;
    controller
        .inner
        .workspace
        .mutation_started(Some(vec!["unsettled.txt".into()]))
        .await
        .unwrap();

    session.loss_tx.send_replace(PipeFsLeaseStatus::Lost);
    let recovery = tokio::time::timeout(Duration::from_secs(2), async {
        let mut status = controller.subscribe();
        loop {
            let current = status.borrow_and_update().clone();
            if current.state == WorkspaceState::RecoveryRequired {
                break current
                    .recovery_id
                    .expect("dirty lease loss has recovery ID");
            }
            status.changed().await.unwrap();
        }
    })
    .await
    .unwrap();

    let marker = controller
        .binding()
        .state_root
        .parent()
        .unwrap()
        .join("recovery-required");
    assert!(marker.is_file());
    assert_eq!(
        controller.reconcile(recovery).await.status,
        RecoveryStatus::Rejected,
        "unmatched dirty evidence must never call remote settlement"
    );
    assert_eq!(controller.status().state, WorkspaceState::RecoveryRequired);
}

#[tokio::test]
async fn pushed_lease_loss_fences_an_active_operation_as_recovery_required() {
    let (_temporary, controller, session, _server) = subject(false).await;
    let permit = controller
        .begin(MutationIntent::workspace("active writer"))
        .await
        .unwrap();

    session.loss_tx.send_replace(PipeFsLeaseStatus::Lost);
    tokio::time::timeout(Duration::from_secs(2), async {
        let mut status = controller.subscribe();
        while status.borrow().state != WorkspaceState::RecoveryRequired {
            status.changed().await.unwrap();
        }
    })
    .await
    .unwrap();

    let status = controller.status();
    assert!(status.recovery_id.is_some());
    assert_eq!(status.active_operation, None);
    assert!(
        controller
            .binding()
            .state_root
            .parent()
            .unwrap()
            .join("recovery-required")
            .is_file()
    );
    drop(permit);
    assert_eq!(controller.status().state, WorkspaceState::RecoveryRequired);
}
