//! Per-turn registry for speculative program calls.
//!
//! The registry is intentionally independent of the tool dispatcher. A
//! shadow task may publish a result here, but only the real program execution
//! can claim it. Exact keys and a single-claim bit prevent stale or duplicate
//! work from entering the transcript.

use std::collections::HashMap;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};
use std::time::{Duration, Instant};

use futures_util::future::{BoxFuture, FutureExt, Shared};
use hi_workflow::ProgramToolResult;
use tokio_util::sync::CancellationToken;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(crate) struct SpeculationKey {
    pub turn_id: String,
    pub program_call_id: String,
    pub tool_occurrence: usize,
    pub tool_name: String,
    pub canonical_arguments: String,
    pub workspace_context_generation: u64,
    pub ledger_revision: u64,
    pub external_freshness_epoch: u64,
}

impl SpeculationKey {
    #[allow(
        clippy::too_many_arguments,
        reason = "matches the explicit registry key schema"
    )]
    pub(crate) fn new(
        turn_id: impl Into<String>,
        program_call_id: impl Into<String>,
        tool_occurrence: usize,
        tool_name: impl Into<String>,
        arguments: &str,
        workspace_context_generation: u64,
        ledger_revision: u64,
        external_freshness_epoch: u64,
    ) -> Self {
        Self {
            turn_id: turn_id.into(),
            program_call_id: program_call_id.into(),
            tool_occurrence,
            tool_name: tool_name.into(),
            canonical_arguments: canonical_json(arguments),
            workspace_context_generation,
            ledger_revision,
            external_freshness_epoch,
        }
    }
}

type SharedResult = Shared<BoxFuture<'static, Result<ProgramToolResult, String>>>;

struct Entry {
    result: SharedResult,
    cancel: CancellationToken,
    external: bool,
    expires_at: Option<Instant>,
    claimed: bool,
}

#[derive(Default)]
struct Counters {
    launched: AtomicU64,
    completed: AtomicU64,
    claimed: AtomicU64,
    exact_misses: AtomicU64,
    invalidated: AtomicU64,
    cancelled: AtomicU64,
}

/// Cheap copy of registry counters for turn telemetry.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SpeculationTelemetry {
    pub launched: u64,
    pub completed: u64,
    pub claimed: u64,
    pub exact_misses: u64,
    pub invalidated: u64,
    pub cancelled: u64,
}

struct RegistryState {
    entries: HashMap<SpeculationKey, Entry>,
    counters: Counters,
}

/// A turn-scoped registry. It can be cloned into streaming/shadow tasks and
/// is dropped with the turn owner; no global cache is used.
#[derive(Clone)]
pub(crate) struct SpeculationRegistry {
    state: Arc<Mutex<RegistryState>>,
    max_calls: usize,
    max_external_calls: usize,
    external_ttl: Duration,
}

impl SpeculationRegistry {
    pub(crate) fn new(max_calls: usize, max_external_calls: usize, external_ttl: Duration) -> Self {
        Self {
            state: Arc::new(Mutex::new(RegistryState {
                entries: HashMap::new(),
                counters: Counters::default(),
            })),
            max_calls,
            max_external_calls,
            external_ttl,
        }
    }

    pub(crate) fn launch<F>(&self, key: SpeculationKey, external: bool, work: F) -> bool
    where
        F: std::future::Future<Output = Result<ProgramToolResult, String>> + Send + 'static,
    {
        let mut state = self.state.lock().expect("speculation registry poisoned");
        reap_expired(&mut state);
        if state.entries.contains_key(&key) || state.entries.len() >= self.max_calls {
            return false;
        }
        let external_count = state
            .entries
            .values()
            .filter(|entry| entry.external)
            .count();
        if external && external_count >= self.max_external_calls {
            return false;
        }
        let cancel = CancellationToken::new();
        let task_cancel = cancel.clone();
        let result = async move {
            tokio::select! {
                _ = task_cancel.cancelled() => Err("speculative call cancelled".to_string()),
                output = work => output,
            }
        }
        .boxed()
        .shared();
        // Poll one shared clone immediately. Without this eager task, the
        // first real claim would merely start the work and speculation would
        // have no opportunity to overlap with the rest of the program.
        tokio::spawn(result.clone());
        state.entries.insert(
            key,
            Entry {
                result,
                cancel,
                external,
                expires_at: external.then(|| Instant::now() + self.external_ttl),
                claimed: false,
            },
        );
        state.counters.launched.fetch_add(1, Ordering::Relaxed);
        true
    }

    /// Claim the exact result once. Waiting occurs without holding the
    /// registry lock, so invalidation and cancellation remain responsive.
    #[allow(
        dead_code,
        reason = "kept as the non-cancellable registry API for tests and callers"
    )]
    pub(crate) async fn claim_exact(
        &self,
        key: &SpeculationKey,
    ) -> Option<Result<ProgramToolResult, String>> {
        self.claim_exact_cancelled(key, None).await
    }

    pub(crate) async fn claim_exact_cancelled(
        &self,
        key: &SpeculationKey,
        cancel: Option<&CancellationToken>,
    ) -> Option<Result<ProgramToolResult, String>> {
        let shared = {
            let mut state = self.state.lock().expect("speculation registry poisoned");
            reap_expired(&mut state);
            let Some(entry) = state.entries.get_mut(key) else {
                state.counters.exact_misses.fetch_add(1, Ordering::Relaxed);
                return None;
            };
            if entry.claimed {
                state.counters.exact_misses.fetch_add(1, Ordering::Relaxed);
                return None;
            }
            entry.claimed = true;
            entry.result.clone()
        };
        let result = if let Some(cancel) = cancel {
            tokio::select! {
                _ = cancel.cancelled() => {
                    self.invalidate(key);
                    return Some(Err("speculative call cancelled".to_string()));
                }
                result = shared => result,
            }
        } else {
            shared.await
        };
        let mut state = self.state.lock().expect("speculation registry poisoned");
        state.entries.remove(key);
        state.counters.claimed.fetch_add(1, Ordering::Relaxed);
        if result.is_ok() {
            state.counters.completed.fetch_add(1, Ordering::Relaxed);
        }
        Some(result)
    }

    pub(crate) fn invalidate(&self, key: &SpeculationKey) {
        let mut state = self.state.lock().expect("speculation registry poisoned");
        if let Some(entry) = state.entries.remove(key) {
            entry.cancel.cancel();
            state.counters.invalidated.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(crate) fn invalidate_all(&self) {
        let keys = {
            let state = self.state.lock().expect("speculation registry poisoned");
            state.entries.keys().cloned().collect::<Vec<_>>()
        };
        for key in keys {
            self.invalidate(&key);
        }
    }

    pub(crate) fn cancel_all(&self) {
        let mut state = self.state.lock().expect("speculation registry poisoned");
        let entries = std::mem::take(&mut state.entries);
        for entry in entries.into_values() {
            entry.cancel.cancel();
            state.counters.cancelled.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(crate) fn telemetry(&self) -> SpeculationTelemetry {
        let state = self.state.lock().expect("speculation registry poisoned");
        SpeculationTelemetry {
            launched: state.counters.launched.load(Ordering::Relaxed),
            completed: state.counters.completed.load(Ordering::Relaxed),
            claimed: state.counters.claimed.load(Ordering::Relaxed),
            exact_misses: state.counters.exact_misses.load(Ordering::Relaxed),
            invalidated: state.counters.invalidated.load(Ordering::Relaxed),
            cancelled: state.counters.cancelled.load(Ordering::Relaxed),
        }
    }
}

fn reap_expired(state: &mut RegistryState) {
    let expired: Vec<_> = state
        .entries
        .iter()
        .filter(|(_, entry)| {
            entry
                .expires_at
                .is_some_and(|expires| Instant::now() >= expires)
        })
        .map(|(key, _)| key.clone())
        .collect();
    for key in expired {
        if let Some(entry) = state.entries.remove(&key) {
            entry.cancel.cancel();
            state.counters.invalidated.fetch_add(1, Ordering::Relaxed);
        }
    }
}

fn canonical_json(arguments: &str) -> String {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(arguments) else {
        return arguments.trim().to_string();
    };
    serde_json::to_string(&sort_json(value)).unwrap_or_else(|_| arguments.trim().to_string())
}

fn sort_json(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let mut sorted = serde_json::Map::new();
            let mut entries: Vec<_> = map.into_iter().collect();
            entries.sort_by(|left, right| left.0.cmp(&right.0));
            for (key, value) in entries {
                sorted.insert(key, sort_json(value));
            }
            serde_json::Value::Object(sorted)
        }
        serde_json::Value::Array(values) => {
            serde_json::Value::Array(values.into_iter().map(sort_json).collect())
        }
        value => value,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    fn key(arguments: &str) -> SpeculationKey {
        SpeculationKey::new("turn", "program", 0, "read", arguments, 1, 2, 3)
    }

    #[tokio::test]
    async fn canonical_arguments_claim_once() {
        let registry = SpeculationRegistry::new(2, 1, Duration::from_secs(30));
        let first = key(r#"{"b":2,"a":1}"#);
        let second = key(r#"{"a":1,"b":2}"#);
        assert!(registry.launch(first.clone(), false, async {
            Ok(ProgramToolResult {
                index: 0,
                name: "read".into(),
                status: "succeeded".into(),
                output: "ok".into(),
            })
        }));
        assert!(registry.claim_exact(&second).await.is_some());
        assert!(registry.claim_exact(&first).await.is_none());
        assert_eq!(registry.telemetry().claimed, 1);
    }

    #[tokio::test]
    async fn cancellation_stops_in_flight_work() {
        let registry = SpeculationRegistry::new(2, 1, Duration::from_secs(30));
        let started = Arc::new(AtomicBool::new(false));
        let started_for_task = started.clone();
        let item = key(r#"{"path":"x"}"#);
        assert!(registry.launch(item.clone(), false, async move {
            started_for_task.store(true, Ordering::Relaxed);
            tokio::time::sleep(Duration::from_secs(30)).await;
            Ok(ProgramToolResult {
                index: 0,
                name: "read".into(),
                status: "succeeded".into(),
                output: String::new(),
            })
        }));
        while !started.load(Ordering::Relaxed) {
            tokio::task::yield_now().await;
        }
        registry.cancel_all();
        assert!(registry.claim_exact(&item).await.is_none());
        assert_eq!(registry.telemetry().cancelled, 1);
    }
}
