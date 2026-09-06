use tokio_util::sync::CancellationToken;

/// Cancels the cooperative Rhai worker before dropping its watcher.
pub(super) struct ProgramRunGuard {
    cancel: CancellationToken,
    _watcher: tokio_util::task::AbortOnDropHandle<()>,
}

impl ProgramRunGuard {
    pub(super) fn new(cancel: CancellationToken, watcher: tokio::task::JoinHandle<()>) -> Self {
        Self {
            cancel,
            _watcher: tokio_util::task::AbortOnDropHandle::new(watcher),
        }
    }
}

impl Drop for ProgramRunGuard {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}
