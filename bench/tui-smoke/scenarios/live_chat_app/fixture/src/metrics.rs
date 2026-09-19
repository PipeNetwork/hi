//! Lightweight process-wide counters, exposed as plain atomics.
//!
//! These are deliberately dependency-free: a handful of `AtomicU64`s that the
//! hot paths bump with `fetch_add(1, Relaxed)`. There is no exporter yet — the
//! values are readable in a debugger or via a future `/metrics` endpoint — but
//! the counters are the right shape to feed a real metrics pipeline later.

use std::sync::atomic::{AtomicU64, Ordering};

/// Process-wide counters. All fields are `Relaxed` because they are only ever
/// bumped and read for observability; exact ordering is irrelevant.
#[derive(Debug, Default)]
pub struct Metrics {
    /// Total accepted TCP (line-protocol) connections.
    pub tcp_connections: AtomicU64,
    /// Total accepted websocket bridge connections.
    pub ws_connections: AtomicU64,
    /// Total channel messages broadcast to at least one subscriber.
    pub messages_broadcast: AtomicU64,
    /// Total direct messages delivered to a connected user.
    pub dms_delivered: AtomicU64,
    /// Total direct messages stored for an offline user.
    pub dms_stored: AtomicU64,
    /// Total auth attempts (login + register) that were rate-limited.
    pub auth_rate_limited: AtomicU64,
    /// Total slow consumers closed because their outbound queue filled.
    pub slow_consumers: AtomicU64,
    /// Total rows deleted by the retention task.
    pub messages_pruned: AtomicU64,
    /// Total bridge tickets minted.
    pub tickets_minted: AtomicU64,
    /// Total bridge tickets redeemed.
    pub tickets_redeemed: AtomicU64,
}

impl Metrics {
    /// Bump a counter by one. Kept for callers that hold a reference to a
    /// specific field; the hot paths use `fetch_add` directly on the static.
    #[allow(dead_code)]
    pub fn bump(&self, field: &AtomicU64) {
        field.fetch_add(1, Ordering::Relaxed);
    }
}

/// The single process-wide metrics instance.
pub static METRICS: Metrics = Metrics {
    tcp_connections: AtomicU64::new(0),
    ws_connections: AtomicU64::new(0),
    messages_broadcast: AtomicU64::new(0),
    dms_delivered: AtomicU64::new(0),
    dms_stored: AtomicU64::new(0),
    auth_rate_limited: AtomicU64::new(0),
    slow_consumers: AtomicU64::new(0),
    messages_pruned: AtomicU64::new(0),
    tickets_minted: AtomicU64::new(0),
    tickets_redeemed: AtomicU64::new(0),
};
