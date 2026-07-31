//! Privacy-safe, bounded-cardinality reliability observability.

use std::sync::atomic::{AtomicU64, Ordering};

use serde::Serialize;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReliabilityEvent {
    RuntimeRegistration,
    Reconnect,
    Replay { count: u64 },
    ReplayGap,
    ReplayDedup,
    HeartbeatFailure,
    HeartbeatConflict,
    HttpRetry,
    HttpFreshPoolEscape,
    HttpDeadline,
    BreakerTransition,
    AnnouncementFetchFailure,
    AnnouncementCacheFailure,
    AnnouncementValidationFailure,
    UpdateCheckFailure,
    UpdateSignatureFailure,
    ProtocolVersionMismatch,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct MetricsSnapshot {
    pub runtime_registrations: u64,
    pub reconnects: u64,
    pub replayed_events: u64,
    pub replay_gaps: u64,
    pub replay_deduplicated: u64,
    pub heartbeat_failures: u64,
    pub heartbeat_conflicts: u64,
    pub http_retries: u64,
    pub http_fresh_pool_escapes: u64,
    pub http_deadlines: u64,
    pub breaker_transitions: u64,
    pub announcement_fetch_failures: u64,
    pub announcement_cache_failures: u64,
    pub announcement_validation_failures: u64,
    pub update_check_failures: u64,
    pub update_signature_failures: u64,
    pub protocol_version_mismatches: u64,
}

macro_rules! counters {
    ($($name:ident),+ $(,)?) => { $(static $name: AtomicU64 = AtomicU64::new(0);)+ };
}
counters!(
    RUNTIME_REGISTRATIONS,
    RECONNECTS,
    REPLAYED_EVENTS,
    REPLAY_GAPS,
    REPLAY_DEDUPLICATED,
    HEARTBEAT_FAILURES,
    HEARTBEAT_CONFLICTS,
    HTTP_RETRIES,
    HTTP_FRESH_POOL_ESCAPES,
    HTTP_DEADLINES,
    BREAKER_TRANSITIONS,
    ANNOUNCEMENT_FETCH_FAILURES,
    ANNOUNCEMENT_CACHE_FAILURES,
    ANNOUNCEMENT_VALIDATION_FAILURES,
    UPDATE_CHECK_FAILURES,
    UPDATE_SIGNATURE_FAILURES,
    PROTOCOL_VERSION_MISMATCHES,
);

fn increment(counter: &AtomicU64, count: u64) {
    let _ = counter.try_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
        Some(value.saturating_add(count))
    });
}

/// Records only a fixed event kind and numeric values. No caller content is accepted.
pub fn record(event: ReliabilityEvent) {
    let (counter, count, kind) = match event {
        ReliabilityEvent::RuntimeRegistration => {
            (&RUNTIME_REGISTRATIONS, 1, "runtime_registration")
        }
        ReliabilityEvent::Reconnect => (&RECONNECTS, 1, "reconnect"),
        ReliabilityEvent::Replay { count } => (&REPLAYED_EVENTS, count, "replay"),
        ReliabilityEvent::ReplayGap => (&REPLAY_GAPS, 1, "replay_gap"),
        ReliabilityEvent::ReplayDedup => (&REPLAY_DEDUPLICATED, 1, "replay_dedup"),
        ReliabilityEvent::HeartbeatFailure => (&HEARTBEAT_FAILURES, 1, "heartbeat_failure"),
        ReliabilityEvent::HeartbeatConflict => (&HEARTBEAT_CONFLICTS, 1, "heartbeat_conflict"),
        ReliabilityEvent::HttpRetry => (&HTTP_RETRIES, 1, "http_retry"),
        ReliabilityEvent::HttpFreshPoolEscape => {
            (&HTTP_FRESH_POOL_ESCAPES, 1, "http_fresh_pool_escape")
        }
        ReliabilityEvent::HttpDeadline => (&HTTP_DEADLINES, 1, "http_deadline"),
        ReliabilityEvent::BreakerTransition => (&BREAKER_TRANSITIONS, 1, "breaker_transition"),
        ReliabilityEvent::AnnouncementFetchFailure => (
            &ANNOUNCEMENT_FETCH_FAILURES,
            1,
            "announcement_fetch_failure",
        ),
        ReliabilityEvent::AnnouncementCacheFailure => (
            &ANNOUNCEMENT_CACHE_FAILURES,
            1,
            "announcement_cache_failure",
        ),
        ReliabilityEvent::AnnouncementValidationFailure => (
            &ANNOUNCEMENT_VALIDATION_FAILURES,
            1,
            "announcement_validation_failure",
        ),
        ReliabilityEvent::UpdateCheckFailure => (&UPDATE_CHECK_FAILURES, 1, "update_check_failure"),
        ReliabilityEvent::UpdateSignatureFailure => {
            (&UPDATE_SIGNATURE_FAILURES, 1, "update_signature_failure")
        }
        ReliabilityEvent::ProtocolVersionMismatch => {
            (&PROTOCOL_VERSION_MISMATCHES, 1, "protocol_version_mismatch")
        }
    };
    increment(counter, count);
    tracing::event!(target: "hi::reliability", tracing::Level::INFO, event.kind = kind, event.count = count);
}

pub fn snapshot() -> MetricsSnapshot {
    macro_rules! load {
        ($name:ident) => {
            $name.load(Ordering::Relaxed)
        };
    }
    MetricsSnapshot {
        runtime_registrations: load!(RUNTIME_REGISTRATIONS),
        reconnects: load!(RECONNECTS),
        replayed_events: load!(REPLAYED_EVENTS),
        replay_gaps: load!(REPLAY_GAPS),
        replay_deduplicated: load!(REPLAY_DEDUPLICATED),
        heartbeat_failures: load!(HEARTBEAT_FAILURES),
        heartbeat_conflicts: load!(HEARTBEAT_CONFLICTS),
        http_retries: load!(HTTP_RETRIES),
        http_fresh_pool_escapes: load!(HTTP_FRESH_POOL_ESCAPES),
        http_deadlines: load!(HTTP_DEADLINES),
        breaker_transitions: load!(BREAKER_TRANSITIONS),
        announcement_fetch_failures: load!(ANNOUNCEMENT_FETCH_FAILURES),
        announcement_cache_failures: load!(ANNOUNCEMENT_CACHE_FAILURES),
        announcement_validation_failures: load!(ANNOUNCEMENT_VALIDATION_FAILURES),
        update_check_failures: load!(UPDATE_CHECK_FAILURES),
        update_signature_failures: load!(UPDATE_SIGNATURE_FAILURES),
        protocol_version_mismatches: load!(PROTOCOL_VERSION_MISMATCHES),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_counts_events_without_content_fields() {
        let before = snapshot();
        record(ReliabilityEvent::Replay { count: 3 });
        record(ReliabilityEvent::HttpRetry);
        let after = snapshot();
        assert_eq!(after.replayed_events, before.replayed_events + 3);
        assert_eq!(after.http_retries, before.http_retries + 1);
        let debug = format!("{after:?}");
        for forbidden in ["prompt", "tool_args", "api_key", "https://"] {
            assert!(!debug.contains(forbidden));
        }
    }
}
