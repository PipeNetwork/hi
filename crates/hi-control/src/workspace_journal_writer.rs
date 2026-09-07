//! Bounded, ordered execution of complete projection operations off Tokio workers.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, Weak, mpsc};

use tokio::sync::{OwnedSemaphorePermit, Semaphore, oneshot};

use crate::{ControlError, Result, WorkspaceProjectionJournal, WorkspaceProjectionStore};

const CAPACITY: usize = 128;
type Work = Box<dyn FnOnce() + Send>;

struct Inner {
    sender: mpsc::Sender<Work>,
    capacity: Arc<Semaphore>,
}

#[derive(Clone)]
pub(crate) struct JournalWriter(Arc<Inner>);

#[derive(Debug)]
pub(crate) struct JournalReservation(OwnedSemaphorePermit);

impl JournalWriter {
    pub(crate) fn for_store(store: &Arc<dyn WorkspaceProjectionStore>) -> Self {
        static WRITERS: OnceLock<Mutex<HashMap<String, Weak<Inner>>>> = OnceLock::new();
        let key = store
            .writer_key()
            .unwrap_or_else(|| format!("store:{:p}", Arc::as_ptr(store)));
        let mut writers = WRITERS
            .get_or_init(Mutex::default)
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        if let Some(existing) = writers.get(&key).and_then(Weak::upgrade) {
            return Self(existing);
        }
        let (sender, receiver) = mpsc::channel::<Work>();
        let inner = Arc::new(Inner {
            sender,
            capacity: Arc::new(Semaphore::new(CAPACITY)),
        });
        std::thread::Builder::new()
            .name("hi-journal-writer".into())
            .spawn(move || {
                while let Ok(work) = receiver.recv() {
                    work();
                }
            })
            .expect("start workspace journal writer");
        writers.retain(|_, writer| writer.strong_count() != 0);
        writers.insert(key, Arc::downgrade(&inner));
        Self(inner)
    }

    /// Reserve before admission. This bounds running plus queued work, so
    /// saturation cannot manufacture an admitted effect with no journal slot.
    pub(crate) fn reserve(&self) -> Result<JournalReservation> {
        self.0.capacity.clone().try_acquire_owned().map(JournalReservation)
            .map_err(|_| ControlError::Invalid("workspace journal queue is full (128 accepted operations); retry admission after pending writes settle".into()))
    }

    pub(crate) async fn reserve_settlement(&self) -> Result<JournalReservation> {
        self.0
            .capacity
            .clone()
            .acquire_owned()
            .await
            .map(JournalReservation)
            .map_err(|_| ControlError::Invalid("workspace journal writer closed admission".into()))
    }

    pub(crate) async fn run<T: Send + 'static>(
        &self,
        reservation: JournalReservation,
        work: impl FnOnce() -> Result<T> + Send + 'static,
    ) -> Result<T> {
        let (reply, acknowledged) = oneshot::channel();
        self.0
            .sender
            .send(Box::new(move || {
                let _capacity = reservation.0;
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(work))
                    .unwrap_or_else(|_| {
                        Err(ControlError::Invalid(
                            "workspace journal writer operation panicked".into(),
                        ))
                    });
                // Accepted work completes even if its waiter was cancelled. The
                // closure owns publication/health updates as well as its commit.
                let _ = reply.send(result);
            }))
            .map_err(|_| ControlError::Invalid("workspace journal writer stopped".into()))?;
        acknowledged.await.map_err(|_| {
            ControlError::Invalid("workspace journal acknowledgment was lost".into())
        })?
    }
}

impl WorkspaceProjectionJournal {
    pub(crate) fn reserve_async(&self) -> Result<JournalReservation> {
        self.writer.reserve()
    }

    pub(crate) async fn run_reserved<T: Send + 'static>(
        &self,
        reservation: JournalReservation,
        operation: impl FnOnce(WorkspaceProjectionJournal) -> Result<T> + Send + 'static,
    ) -> Result<T> {
        let journal = self.clone();
        self.writer
            .run(reservation, move || operation(journal))
            .await
    }

    pub(crate) async fn run_when_available<T: Send + 'static>(
        &self,
        operation: impl FnOnce(WorkspaceProjectionJournal) -> Result<T> + Send + 'static,
    ) -> Result<T> {
        self.run_reserved(self.writer.reserve_settlement().await?, operation)
            .await
    }

    #[cfg(test)]
    pub(crate) async fn run_async<T: Send + 'static>(
        &self,
        operation: impl FnOnce(WorkspaceProjectionJournal) -> Result<T> + Send + 'static,
    ) -> Result<T> {
        self.run_reserved(self.reserve_async()?, operation).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hi_workspace::{InMemoryWorkspaceController, WorkspaceController};
    use std::time::Duration;

    fn journal() -> (
        tempfile::TempDir,
        crate::ControlStore,
        WorkspaceProjectionJournal,
    ) {
        let dir = tempfile::tempdir().unwrap();
        let store = crate::ControlStore::open(dir.path().join("journal.sqlite3")).unwrap();
        let journal = WorkspaceProjectionJournal::from_control_store(store.clone());
        (dir, store, journal)
    }

    #[test]
    fn admission_capacity_is_bounded_and_shared_by_database() {
        let (_dir, store, journal) = journal();
        let second = WorkspaceProjectionJournal::from_control_store(store);
        let reservations = (0..CAPACITY)
            .map(|_| journal.reserve_async().unwrap())
            .collect::<Vec<_>>();
        assert!(
            second
                .reserve_async()
                .unwrap_err()
                .to_string()
                .contains("queue is full")
        );
        drop(reservations);
        assert!(second.reserve_async().is_ok());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn locked_sqlite_does_not_block_timers_or_execution_cancellation() {
        let (_dir, store, journal) = journal();
        let blocker = rusqlite::Connection::open(store.path()).unwrap();
        blocker.execute_batch("BEGIN IMMEDIATE").unwrap();
        let local = InMemoryWorkspaceController::new_local("writer-test", "/work", "/state");
        let binding = local.binding();
        let status = local.status();
        let capabilities = local.capabilities();
        let (entered, started) = oneshot::channel();
        let work = tokio::spawn(async move {
            journal
                .run_async(move |journal| {
                    let _ = entered.send(());
                    journal.record_binding(&binding, &status, &capabilities)
                })
                .await
        });
        started.await.unwrap();
        let execution = tokio::spawn(std::future::pending::<()>());
        execution.abort();
        tokio::time::timeout(Duration::from_millis(200), async {
            assert!(execution.await.unwrap_err().is_cancelled());
            tokio::time::sleep(Duration::from_millis(10)).await;
        })
        .await
        .expect("SQLite must not block the current-thread runtime");
        assert!(
            !work.is_finished(),
            "writer is still waiting for the real SQLite lock"
        );
        blocker.execute_batch("ROLLBACK").unwrap();
        tokio::time::timeout(Duration::from_secs(2), work)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn accepted_write_survives_dropped_waiter() {
        let (_dir, store, journal) = journal();
        let local = InMemoryWorkspaceController::new_local("late-ack", "/work", "/state");
        let binding = local.binding();
        let id = binding.binding_id.to_string();
        let status = local.status();
        let capabilities = local.capabilities();
        let (release, gate) = std::sync::mpsc::channel();
        let (entered, started) = oneshot::channel();
        let (committed, done) = oneshot::channel();
        let waiter = tokio::spawn(async move {
            journal
                .run_async(move |journal| {
                    let _ = entered.send(());
                    gate.recv().unwrap();
                    journal.record_binding(&binding, &status, &capabilities)?;
                    let _ = committed.send(());
                    Ok(())
                })
                .await
        });
        started.await.unwrap();
        waiter.abort();
        assert!(waiter.await.unwrap_err().is_cancelled());
        release.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(2), done)
            .await
            .unwrap()
            .unwrap();
        assert!(store.get_workspace_binding(&id).unwrap().is_some());
    }
}
