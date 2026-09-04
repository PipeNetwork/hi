use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, RwLock};

use super::{WorkspaceCoordination, local_controller, resolved_job_limits};

impl WorkspaceCoordination {
    #[cfg(test)]
    pub(crate) fn new_local(workspace_root: &Path, state_root: &Path) -> Self {
        Self::new_local_with_settings(
            workspace_root,
            state_root,
            hi_workspace::ResolvedHarnessSettings::default(),
        )
    }

    pub(crate) fn new_local_with_settings(
        workspace_root: &Path,
        state_root: &Path,
        harness: hi_workspace::ResolvedHarnessSettings,
    ) -> Self {
        let job_limits = resolved_job_limits(&harness);
        Self {
            controller: Arc::new(RwLock::new(local_controller(
                workspace_root,
                state_root,
                0,
                job_limits,
            ))),
            active: Arc::new(Mutex::new(None)),
            admission: Arc::new(super::admission::WorkspaceAdmissionGate::default()),
            controller_settles_backend: Arc::new(AtomicBool::new(false)),
            harness,
        }
    }

    pub(crate) fn harness_settings(&self) -> &hi_workspace::ResolvedHarnessSettings {
        &self.harness
    }
}
