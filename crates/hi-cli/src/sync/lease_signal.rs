use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use hi_pipefs::PipeFsLeaseStatus;

#[derive(Clone)]
pub(super) struct LeaseLossSignal {
    inner: Arc<Inner>,
}

struct Inner {
    status: AtomicU8,
    changes: tokio::sync::watch::Sender<PipeFsLeaseStatus>,
}

impl LeaseLossSignal {
    pub(super) fn new() -> Self {
        Self {
            inner: Arc::new(Inner {
                status: AtomicU8::new(Self::encode(PipeFsLeaseStatus::Valid)),
                changes: tokio::sync::watch::channel(PipeFsLeaseStatus::Valid).0,
            }),
        }
    }

    pub(super) fn is_lost(&self) -> bool {
        Self::decode(self.inner.status.load(Ordering::Acquire)) == PipeFsLeaseStatus::Lost
    }

    pub(super) fn mark_uncertain(&self) {
        if self
            .inner
            .status
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                (current != Self::encode(PipeFsLeaseStatus::Lost))
                    .then_some(Self::encode(PipeFsLeaseStatus::Uncertain))
            })
            .is_ok()
        {
            self.inner
                .changes
                .send_replace(PipeFsLeaseStatus::Uncertain);
        }
    }

    pub(super) fn mark_lost(&self) {
        self.inner
            .status
            .store(Self::encode(PipeFsLeaseStatus::Lost), Ordering::Release);
        self.inner.changes.send_replace(PipeFsLeaseStatus::Lost);
    }

    pub(super) fn mark_synchronously_confirmed(&self) {
        if self
            .inner
            .status
            .compare_exchange(
                Self::encode(PipeFsLeaseStatus::Uncertain),
                Self::encode(PipeFsLeaseStatus::Valid),
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
        {
            self.inner.changes.send_replace(PipeFsLeaseStatus::Valid);
        }
    }

    pub(super) fn reset(&self) {
        self.inner
            .status
            .store(Self::encode(PipeFsLeaseStatus::Valid), Ordering::Release);
        self.inner.changes.send_replace(PipeFsLeaseStatus::Valid);
    }

    pub(super) fn subscribe(&self) -> tokio::sync::watch::Receiver<PipeFsLeaseStatus> {
        self.inner.changes.subscribe()
    }

    const fn encode(status: PipeFsLeaseStatus) -> u8 {
        match status {
            PipeFsLeaseStatus::Valid => 0,
            PipeFsLeaseStatus::Uncertain => 1,
            PipeFsLeaseStatus::Lost => 2,
        }
    }

    const fn decode(status: u8) -> PipeFsLeaseStatus {
        match status {
            1 => PipeFsLeaseStatus::Uncertain,
            2 => PipeFsLeaseStatus::Lost,
            _ => PipeFsLeaseStatus::Valid,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn publishes_uncertainty_loss_and_reset() {
        let signal = LeaseLossSignal::new();
        let mut changes = signal.subscribe();
        signal.mark_uncertain();
        changes.changed().await.unwrap();
        assert_eq!(*changes.borrow_and_update(), PipeFsLeaseStatus::Uncertain);
        assert!(!signal.is_lost());
        signal.mark_synchronously_confirmed();
        changes.changed().await.unwrap();
        assert_eq!(*changes.borrow_and_update(), PipeFsLeaseStatus::Valid);
        signal.mark_lost();
        changes.changed().await.unwrap();
        assert_eq!(*changes.borrow_and_update(), PipeFsLeaseStatus::Lost);
        assert!(signal.is_lost());
        signal.mark_uncertain();
        assert_eq!(*changes.borrow(), PipeFsLeaseStatus::Lost);
        signal.reset();
        changes.changed().await.unwrap();
        assert_eq!(*changes.borrow_and_update(), PipeFsLeaseStatus::Valid);
        assert!(!signal.is_lost());
    }
}
