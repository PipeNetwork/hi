//! Workspace binding checks for detached delegate candidates.

impl crate::Agent {
    pub(crate) fn delegate_runner_matches_workspace(&self) -> bool {
        !self.pipefs_workspace_active()
            || (self.workspace_controller_capabilities().candidate_apply
                && self
                    .subagents
                    .delegate_runner
                    .as_ref()
                    .is_some_and(|runner| {
                        runner.is_bound_to_workspace(self.runtime.root(), self.runtime.state_root())
                    }))
    }

    pub(super) fn bind_delegate_runner_workspace(&self) {
        if let Some(runner) = &self.subagents.delegate_runner
            && !runner.bind_workspace(self.runtime.root(), self.runtime.state_root())
        {
            tracing::warn!(
                "delegate runner could not bind the new workspace; delegate remains disabled"
            );
        }
    }
}
