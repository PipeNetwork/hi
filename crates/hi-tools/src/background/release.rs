//! Durable ownership handoff for processes intentionally kept after Hi exits.

use super::*;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BackgroundReleaseSummary {
    pub released: Vec<String>,
    pub settlement_pending: Vec<String>,
}

impl BackgroundRegistry {
    /// Durably relinquish requested processes that are still running.
    ///
    /// This is the final `--keep-background` handoff. A managed job is sealed
    /// `Orphaned` before its registry entry is removed. Auto-backgrounded and
    /// already-terminal processes remain owned so callers must reap and settle
    /// them normally; a terminal callback that wins the race is never rewritten
    /// as an orphan.
    pub async fn release_requested_running(&self) -> Result<BackgroundReleaseSummary> {
        if self.quiescing.swap(true, Ordering::AcqRel) {
            bail!("another workspace lifecycle operation is already in progress");
        }
        let _reset_quiescing = ResetQuiescing(&self.quiescing);

        // `reserve_slot` holds the process-map lock while incrementing the
        // reservation count. Acquiring that lock after closing admission makes
        // every in-flight spawn visible here; do not clear around a child that
        // has been admitted but not inserted yet.
        let mut processes = {
            let processes = self.processes.lock().unwrap();
            if self.reserved_slots.load(Ordering::Acquire) != 0 {
                bail!("a background process is still being started");
            }
            processes
                .iter()
                .filter(|(_, process)| {
                    process.origin == BgOrigin::Requested
                        && process.inner.lock().unwrap().native_running()
                })
                .map(|(id, process)| (id.clone(), Arc::clone(process)))
                .collect::<Vec<_>>()
        };
        processes.sort_by_key(|(id, _)| id_num(id));

        let mut released = Vec::with_capacity(processes.len());
        let mut settlement_pending = Vec::new();
        for (id, process) in &processes {
            // The initial snapshot only closes the spawn race. Re-check the
            // native state at the lifecycle-publication boundary: the driver
            // sets `native_exited` and publishes its real terminal callback
            // while holding this same async gate. Never hold `inner`'s sync
            // mutex across the journal await.
            let _terminal_publication = process.terminal_publication.lock().await;
            if !process.inner.lock().unwrap().native_running() {
                continue;
            }
            let Some(job) = &process.managed_job else {
                process.ownership_released.store(true, Ordering::Release);
                self.processes.lock().unwrap().remove(id);
                released.push(id.clone());
                continue;
            };
            let publication = job
                .observe(
                    crate::BackgroundJobTerminal::Orphaned,
                    Some(
                        "background process ownership was relinquished by --keep-background".into(),
                    ),
                )
                .await
                .map_err(|error| {
                    anyhow::anyhow!(
                        "could not durably orphan background process {id} before release: {error}"
                    )
                })?;
            if publication != crate::BackgroundJobPublication::Published {
                settlement_pending.push(id.clone());
                continue;
            }
            // Once the durable journal has acknowledged relinquishment, this
            // registry must stop owning the process immediately. If a later
            // handoff fails, Drop must not kill an earlier process whose job
            // already says Orphaned.
            process.ownership_released.store(true, Ordering::Release);
            self.processes.lock().unwrap().remove(id);
            released.push(id.clone());
        }

        // Admission remains closed through the acknowledgement loop. Auto jobs
        // and processes already terminal remain owned until normal reap and
        // settlement completes.
        Ok(BackgroundReleaseSummary {
            released,
            settlement_pending,
        })
    }
}

struct ResetQuiescing<'a>(&'a std::sync::atomic::AtomicBool);

impl Drop for ResetQuiescing<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}
