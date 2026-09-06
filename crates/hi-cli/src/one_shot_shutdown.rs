use anyhow::Result;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Strategy {
    Settle,
    PreserveRecoveryFence,
}

pub(crate) fn strategy(outcome: Option<&hi_agent::TurnOutcome>) -> Strategy {
    if outcome.is_some_and(|outcome| {
        outcome.stop_reason == hi_agent::TurnStopReason::WorkspaceRecoveryRequired
    }) {
        Strategy::PreserveRecoveryFence
    } else {
        Strategy::Settle
    }
}

pub(crate) async fn prepare_agent(
    agent: &mut hi_agent::Agent,
    keep_background: bool,
    strategy: Strategy,
) -> Result<()> {
    match strategy {
        Strategy::PreserveRecoveryFence => agent.quiesce_after_blocked_workspace_admission().await,
        Strategy::Settle if keep_background => agent.release_background_services().await,
        Strategy::Settle => agent.settle_workspace_for_exit().await.map(|_| ()),
    }
}

pub(crate) async fn finish_pipefs(
    host: Option<&crate::pipefs::PipeFsHost>,
    agent: &mut hi_agent::Agent,
    strategy: Strategy,
) -> Result<()> {
    let Some(host) = host else {
        return Ok(());
    };
    let result = match strategy {
        Strategy::PreserveRecoveryFence => host.fenced_exit(agent).await,
        Strategy::Settle => host.clean_exit(agent).await,
    };
    if let Err(error) = result {
        eprintln!("\x1b[31mPipeFS exit blocked: {error:#}; recovery cache was retained\x1b[0m");
        std::process::exit(3);
    }
    Ok(())
}

pub(crate) fn exit_code(
    result: &anyhow::Result<hi_agent::TurnOutcome>,
    failed_outcome: Option<&hi_agent::TurnOutcome>,
    allow_unverified: bool,
    leftover_remains: bool,
) -> i32 {
    result
        .as_ref()
        .ok()
        .or(failed_outcome)
        .map(|outcome| {
            crate::report::one_shot_exit_code(outcome, allow_unverified, leftover_remains)
        })
        .unwrap_or(3)
}

#[cfg(test)]
mod tests {
    use super::{Strategy, strategy};

    fn blocked(reason: hi_agent::TurnStopReason) -> hi_agent::TurnOutcome {
        hi_agent::TurnOutcome {
            status: hi_agent::TurnStatus::Blocked,
            verification: hi_agent::VerificationStatus::Unverified,
            review: hi_agent::ReviewStatus::NotRequired,
            stop_reason: reason,
            changed_files: Vec::new(),
            verified_workspace_revision: None,
            effective_route: hi_agent::EffectiveModelRoute {
                provider: None,
                model: "model".into(),
            },
            review_same_model: false,
            leftover: None,
            plan_leftover: None,
        }
    }

    #[test]
    fn only_durable_recovery_uses_fenced_shutdown() {
        let recovery = blocked(hi_agent::TurnStopReason::WorkspaceRecoveryRequired);
        assert_eq!(strategy(Some(&recovery)), Strategy::PreserveRecoveryFence);

        let active_writer = blocked(hi_agent::TurnStopReason::WorkspaceNotReady);
        assert_eq!(
            strategy(Some(&active_writer)),
            Strategy::Settle,
            "a same-process live writer must be reaped and durably settled"
        );
        assert_eq!(strategy(None), Strategy::Settle);
    }

    #[test]
    fn typed_failed_outcome_controls_exit_code() {
        let recovery = blocked(hi_agent::TurnStopReason::WorkspaceRecoveryRequired);
        let result = Err(anyhow::anyhow!("workspace admission denied"));
        assert_eq!(super::exit_code(&result, Some(&recovery), false, false), 1);
    }
}
