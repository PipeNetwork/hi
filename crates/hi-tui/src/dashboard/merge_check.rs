use super::*;

pub(super) async fn check(worktree: PathBuf, base: String, verify: Option<String>) -> RowDone {
    let changed = worktree::changed_files_async(&worktree, &base)
        .await
        .map_err(|error| format!("{error:#}"));
    let verified = match (&changed, verify.as_deref()) {
        (Ok(changed), Some(verify)) if !changed.is_empty() => {
            worktree::verify_passes_async(&worktree, verify, None).await
        }
        (Ok(_), _) => true,
        (Err(_), _) => false,
    };
    RowDone::MergeCheck { changed, verified }
}

pub(super) async fn force(worktree: PathBuf, base: String, destination: PathBuf) -> RowDone {
    let changed = match worktree::changed_files_async(&worktree, &base).await {
        Ok(changed) => changed,
        Err(error) => {
            return RowDone::ForceMerge {
                changed: Vec::new(),
                result: Err(format!("could not inspect changes: {error:#}")),
            };
        }
    };
    let result = if changed.is_empty() {
        Ok(())
    } else {
        worktree::apply_changes_to_async(&worktree, &base, &destination, None)
            .await
            .map(|_| ())
            .map_err(|error| format!("{error:#}"))
    };
    RowDone::ForceMerge { changed, result }
}

pub(super) fn finish_failure(app: &mut App, idx: usize, error: String) {
    let Some(row) = app.fleet.get_mut(idx) else {
        return;
    };
    row.state = RowState::Failed;
    row.merge = MergeState::VerifyFailed;
    row.started = None;
    row.activity.clear();
    let message = format!(
        "could not inspect worktree changes: {error}; candidate retained at {}",
        row.worktree.display()
    );
    row.push_line(format!("✗ {message}"));
    // Keep pending work parked. Neither an empty-diff success nor another
    // autonomous turn is justified while the candidate cannot be inspected.
    let completion = finish_workflow_agent(row, false, message);
    settle_workflow_reply(app, completion);
    flag_attention(app, idx);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn failed_inspection_cannot_report_an_empty_success_or_force_apply() {
        let candidate = tempfile::tempdir().unwrap();
        let destination = tempfile::tempdir().unwrap();
        std::fs::write(candidate.path().join("source.rs"), "new work").unwrap();
        let done = check(candidate.path().into(), "missing-base".into(), None).await;
        assert!(matches!(
            done,
            RowDone::MergeCheck {
                changed: Err(_),
                verified: false
            }
        ));
        let done = force(
            candidate.path().into(),
            "missing-base".into(),
            destination.path().into(),
        )
        .await;
        assert!(matches!(done, RowDone::ForceMerge { result: Err(_), .. }));
        assert_eq!(
            std::fs::read_to_string(candidate.path().join("source.rs")).unwrap(),
            "new work"
        );
        assert!(!destination.path().join("source.rs").exists());
    }

    #[tokio::test]
    async fn failed_inspection_retains_candidate_and_reports_workflow_failure() {
        let candidate = tempfile::tempdir().unwrap();
        std::fs::write(candidate.path().join("source.rs"), "new work").unwrap();
        let mut app = crate::tests::test_app("openai", "test");
        let mut row = super::super::tests::row();
        let (tx, rx) = oneshot::channel();
        row.workflow_reply = Some(tx);
        row.worktree = candidate.path().into();
        row.session = candidate.path().join("session.jsonl");
        row.pending.push_back("next task".into());
        row.changed.push("previous.rs".into());
        app.fleet.push(row);
        finish_failure(&mut app, 0, "index is locked".into());
        assert!(!rx.await.unwrap().unwrap().success);
        assert!(app.fleet[0].state == RowState::Failed);
        assert!(app.fleet[0].merge == MergeState::VerifyFailed);
        assert_eq!(app.fleet[0].changed, ["previous.rs"]);
        assert_eq!(app.fleet[0].pending.len(), 1);
        assert!(candidate.path().join("source.rs").exists());
        assert!(
            app.fleet[0]
                .tail
                .iter()
                .any(|line| line.contains("index is locked"))
        );
    }
}
