//! Admission gate shared by foreground mutations and background job adapters.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Result, anyhow, ensure};

use super::WorkspaceCoordination;

pub(super) struct WorkspaceAdmissionGate {
    lock: Arc<tokio::sync::RwLock<()>>,
    generation: AtomicU64,
    #[cfg(test)]
    waiting_readers: std::sync::atomic::AtomicUsize,
}

impl Default for WorkspaceAdmissionGate {
    fn default() -> Self {
        Self {
            lock: Arc::new(tokio::sync::RwLock::new(())),
            generation: AtomicU64::new(0),
            #[cfg(test)]
            waiting_readers: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}

pub(crate) struct WorkspaceRebindAdmission {
    gate: Arc<WorkspaceAdmissionGate>,
    _exclusive: tokio::sync::OwnedRwLockWriteGuard<()>,
}

impl WorkspaceAdmissionGate {
    pub(super) async fn read(&self) -> tokio::sync::OwnedRwLockReadGuard<()> {
        #[cfg(test)]
        self.waiting_readers.fetch_add(1, Ordering::AcqRel);
        let guard = self.lock.clone().read_owned().await;
        #[cfg(test)]
        self.waiting_readers.fetch_sub(1, Ordering::AcqRel);
        guard
    }

    pub(super) async fn close(self: &Arc<Self>) -> WorkspaceRebindAdmission {
        let exclusive = self.lock.clone().write_owned().await;
        WorkspaceRebindAdmission {
            gate: self.clone(),
            _exclusive: exclusive,
        }
    }

    pub(super) fn try_close(self: &Arc<Self>) -> Result<WorkspaceRebindAdmission> {
        let exclusive = self
            .lock
            .clone()
            .try_write_owned()
            .map_err(|_| anyhow!("workspace admission is busy"))?;
        Ok(WorkspaceRebindAdmission {
            gate: self.clone(),
            _exclusive: exclusive,
        })
    }

    pub(super) fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    #[cfg(test)]
    pub(super) fn waiting_readers(&self) -> usize {
        self.waiting_readers.load(Ordering::Acquire)
    }
}

impl WorkspaceCoordination {
    pub(super) async fn acquire_admission(&self) -> tokio::sync::OwnedRwLockReadGuard<()> {
        self.admission.read().await
    }

    pub(crate) async fn close_admission_for_rebind(&self) -> WorkspaceRebindAdmission {
        self.admission.close().await
    }

    pub(super) fn admission_generation(&self) -> u64 {
        self.admission.generation()
    }

    pub(super) fn admission_generation_is_current(&self, generation: u64) -> bool {
        self.admission_generation() == generation
    }

    pub(crate) fn install_local_during_rebind(
        &self,
        workspace_root: &std::path::Path,
        state_root: &std::path::Path,
        admission: &WorkspaceRebindAdmission,
    ) -> Result<()> {
        let epoch = self.controller().binding().epoch.saturating_add(1);
        let controller = super::local_controller(
            workspace_root,
            state_root,
            epoch,
            super::resolved_job_limits(&self.harness),
        );
        self.replace_during_rebind(controller, false, admission)
    }

    pub(super) fn replace_during_rebind(
        &self,
        controller: Arc<dyn hi_workspace::WorkspaceController>,
        controller_settles_backend: bool,
        admission: &WorkspaceRebindAdmission,
    ) -> Result<()> {
        ensure!(
            Arc::ptr_eq(&self.admission, &admission.gate),
            "rebind admission guard belongs to another workspace coordinator"
        );
        let next_generation = self
            .admission_generation()
            .checked_add(1)
            .ok_or_else(|| anyhow!("workspace admission generation overflow"))?;
        self.replace_while_admission_closed(controller, controller_settles_backend)?;
        // The exclusive gate makes the controller pointer and lifecycle
        // generation one publication boundary for every waiting admission.
        self.admission
            .generation
            .store(next_generation, Ordering::Release);
        Ok(())
    }

    pub(super) fn replace_with_admission_closed(
        &self,
        controller: Arc<dyn hi_workspace::WorkspaceController>,
        controller_settles_backend: bool,
    ) -> Result<()> {
        let admission = self.admission.try_close()?;
        ensure!(Arc::ptr_eq(&self.admission, &admission.gate));
        self.replace_while_admission_closed(controller, controller_settles_backend)
    }

    #[cfg(test)]
    pub(super) fn admission_waiting_readers(&self) -> usize {
        self.admission.waiting_readers()
    }
}
