//! Mid-turn interjection queue. Frontends clone a handle and push while a turn
//! runs; the loop drains at model-round boundaries. Waiters (wait_tasks /
//! poll_wait) subscribe so a follow-up can abort a parked wait.

use std::collections::VecDeque;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

use tokio::sync::Notify;

/// Cloneable mid-turn interjection queue, drained by the turn loop at safe points.
/// Cheap to clone because the queue is shared.
#[derive(Clone)]
pub struct InterjectionInbox {
    inner: Arc<Inner>,
}

struct Inner {
    queue: Mutex<VecDeque<String>>,
    notify: Arc<Notify>,
    /// Sticky with the queue so a wait that starts after `push` still aborts.
    /// `Notify` permits from `notify_waiters` are not sticky.
    abort_pending: Arc<AtomicBool>,
}

/// Prefix tagging an interjected message as a `/btw` side question. The preferred
/// path is [`crate::Agent::btw_dispatcher`] →
/// [`crate::BtwDispatcher::ask`] which answers **immediately** with its own model
/// calls. The inbox tag remains for tests and frontends that only have the
/// interjection queue; the loop drains it as a fallback. A control char keeps
/// it out of the visible transcript and collision-free with real user text.
pub const BTW_INTERJECTION_PREFIX: &str = "\u{1}btw:";

impl Default for InterjectionInbox {
    fn default() -> Self {
        Self {
            inner: Arc::new(Inner {
                queue: Mutex::new(VecDeque::new()),
                notify: Arc::new(Notify::new()),
                abort_pending: Arc::new(AtomicBool::new(false)),
            }),
        }
    }
}

impl InterjectionInbox {
    /// Queue a user message to be injected into the running turn. Empty/
    /// whitespace-only messages are ignored.
    pub fn push(&self, message: impl Into<String>) {
        let message = message.into();
        if message.trim().is_empty() {
            return;
        }
        if let Ok(mut queue) = self.inner.queue.lock() {
            queue.push_back(message);
        }
        self.inner.abort_pending.store(true, Ordering::Release);
        self.inner.notify.notify_waiters();
    }

    /// Wake waiters without enqueueing (empty-Enter send-now when the line
    /// is already in the inbox).
    pub fn notify_waiters(&self) {
        if self.has_pending() {
            self.inner.abort_pending.store(true, Ordering::Release);
        }
        self.inner.notify.notify_waiters();
    }

    /// Shared notify handle so process/task waits can abort on a follow-up.
    pub fn notify_handle(&self) -> Arc<Notify> {
        self.inner.notify.clone()
    }

    pub fn abort_pending_flag(&self) -> Arc<AtomicBool> {
        self.inner.abort_pending.clone()
    }

    /// Take all queued messages, leaving the queue empty.
    pub fn drain(&self) -> Vec<String> {
        let drained = self
            .inner
            .queue
            .lock()
            .map(|mut queue| queue.drain(..).collect())
            .unwrap_or_default();
        if !self.has_pending() {
            self.inner.abort_pending.store(false, Ordering::Release);
        }
        drained
    }

    /// Snapshot of messages still waiting (for UI; does not consume).
    pub fn pending(&self) -> Vec<String> {
        self.inner
            .queue
            .lock()
            .map(|queue| queue.iter().cloned().collect())
            .unwrap_or_default()
    }

    pub fn has_pending(&self) -> bool {
        self.inner
            .queue
            .lock()
            .map(|queue| !queue.is_empty())
            .unwrap_or(false)
    }

    /// Resolve once a message is queued. Returns immediately if one is already waiting.
    pub async fn wait_pending(&self) {
        loop {
            let notified = self.inner.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.has_pending() {
                return;
            }
            notified.await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn push_drain_and_ignore_empty() {
        let inbox = InterjectionInbox::default();
        assert!(!inbox.has_pending());
        inbox.push("  ");
        inbox.push("focus on the parser");
        inbox.push("and add a test");
        assert!(inbox.has_pending());
        assert_eq!(
            inbox.pending(),
            vec!["focus on the parser", "and add a test"]
        );
        let drained = inbox.drain();
        assert_eq!(drained, vec!["focus on the parser", "and add a test"]);
        assert!(!inbox.has_pending());
    }

    #[tokio::test]
    async fn wait_pending_wakes_on_push() {
        let inbox = InterjectionInbox::default();
        let waiter = inbox.clone();
        let handle = tokio::spawn(async move {
            waiter.wait_pending().await;
        });
        tokio::time::sleep(Duration::from_millis(20)).await;
        inbox.push("send now");
        tokio::time::timeout(Duration::from_secs(1), handle)
            .await
            .expect("wait_pending must not hang")
            .expect("task");
        assert_eq!(inbox.drain(), vec!["send now"]);
    }

    #[test]
    fn drain_clears_abort_pending_so_later_waits_are_not_stuck_idle() {
        let inbox = InterjectionInbox::default();
        inbox.push("follow-up");
        assert!(inbox.abort_pending_flag().load(Ordering::Acquire));
        let _ = inbox.drain();
        assert!(!inbox.abort_pending_flag().load(Ordering::Acquire));
    }
}
