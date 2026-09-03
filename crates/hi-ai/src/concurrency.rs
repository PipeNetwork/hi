//! Shared concurrency limiting for provider inference requests.

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use tokio::sync::{Mutex, Notify};

use crate::{
    ChatRequest, Completion, Provider, ProviderErrorKind, ServedModel, StreamEvent,
    provider_error_kind,
};

/// Default number of in-process provider streams admitted concurrently.
pub const DEFAULT_PROVIDER_REQUEST_CONCURRENCY: usize = 8;
const DEFAULT_FOREGROUND_RESERVED: usize = 1;
const RECOVERY_SUCCESSES: usize = 4;
const ADAPTIVE_COOLDOWN: Duration = Duration::from_secs(10);

/// Bounded request-concurrency policy. Auxiliary requests share the same hard
/// cap but cannot consume the reserved foreground slots.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProviderConcurrencyConfig {
    pub max_concurrent: usize,
    pub foreground_reserved: usize,
    pub adaptive: bool,
}

impl Default for ProviderConcurrencyConfig {
    fn default() -> Self {
        Self {
            max_concurrent: DEFAULT_PROVIDER_REQUEST_CONCURRENCY,
            foreground_reserved: DEFAULT_FOREGROUND_RESERVED,
            adaptive: true,
        }
    }
}

impl ProviderConcurrencyConfig {
    pub fn validate(self) -> Result<Self> {
        if self.max_concurrent == 0 {
            return Err(anyhow!("provider request concurrency must be at least one"));
        }
        if self.foreground_reserved >= self.max_concurrent {
            return Err(anyhow!(
                "foreground-reserved provider slots must be below total concurrency"
            ));
        }
        Ok(self)
    }
}

#[derive(Debug)]
struct LimitState {
    in_flight: usize,
    auxiliary_in_flight: usize,
    current_limit: usize,
    success_streak: usize,
    last_throttle: Option<Instant>,
}

struct RequestPermit {
    state: Arc<Mutex<LimitState>>,
    notify: Arc<Notify>,
    auxiliary: bool,
}

impl Drop for RequestPermit {
    fn drop(&mut self) {
        let state = self.state.clone();
        let notify = self.notify.clone();
        let auxiliary = self.auxiliary;
        tokio::spawn(async move {
            let mut state = state.lock().await;
            state.in_flight = state.in_flight.saturating_sub(1);
            if auxiliary {
                state.auxiliary_in_flight = state.auxiliary_in_flight.saturating_sub(1);
            }
            drop(state);
            notify.notify_waiters();
        });
    }
}

/// Provider decorator that applies backpressure across all clones sharing it.
///
/// The permit is held for the full streamed inference request and is released
/// automatically on success, error, or cancellation. Metadata discovery is not
/// limited because it does not consume a model generation slot.
pub struct ConcurrencyLimitedProvider {
    inner: Box<dyn Provider>,
    config: ProviderConcurrencyConfig,
    state: Arc<Mutex<LimitState>>,
    notify: Arc<Notify>,
}

impl ConcurrencyLimitedProvider {
    pub fn new(inner: Box<dyn Provider>, max_concurrent: usize) -> Result<Self> {
        Self::with_config(
            inner,
            ProviderConcurrencyConfig {
                max_concurrent,
                foreground_reserved: usize::from(max_concurrent > 1),
                adaptive: true,
            },
        )
    }

    pub fn with_config(
        inner: Box<dyn Provider>,
        config: ProviderConcurrencyConfig,
    ) -> Result<Self> {
        let config = config.validate()?;
        Ok(Self {
            inner,
            config,
            state: Arc::new(Mutex::new(LimitState {
                in_flight: 0,
                auxiliary_in_flight: 0,
                current_limit: config.max_concurrent,
                success_streak: 0,
                last_throttle: None,
            })),
            notify: Arc::new(Notify::new()),
        })
    }

    async fn acquire(&self, auxiliary: bool) -> RequestPermit {
        loop {
            let notified = self.notify.notified();
            {
                let mut state = self.state.lock().await;
                // Adaptive throttling must not collapse the entire pool into the
                // foreground reservation. An auxiliary request may be awaited by
                // the foreground turn itself (finalization, compaction, review),
                // so leaving it zero admissible slots deadlocks the turn and also
                // prevents the successes that would recover the adaptive limit.
                // Heal an already-degraded state here as well as enforcing the
                // floor in `record_result`, so restored/legacy state cannot strand
                // an auxiliary waiter forever.
                let productive_floor = self
                    .config
                    .foreground_reserved
                    .saturating_add(1)
                    .min(self.config.max_concurrent);
                state.current_limit = state.current_limit.max(productive_floor);
                let auxiliary_limit = state
                    .current_limit
                    .saturating_sub(self.config.foreground_reserved);
                let admitted = state.in_flight < state.current_limit
                    && (!auxiliary || state.auxiliary_in_flight < auxiliary_limit);
                if admitted {
                    state.in_flight += 1;
                    if auxiliary {
                        state.auxiliary_in_flight += 1;
                    }
                    return RequestPermit {
                        state: self.state.clone(),
                        notify: self.notify.clone(),
                        auxiliary,
                    };
                }
            }
            notified.await;
        }
    }

    async fn record_result(&self, result: &Result<Completion>) {
        if !self.config.adaptive {
            return;
        }
        let mut state = self.state.lock().await;
        let throttled = result.as_ref().err().is_some_and(|error| {
            matches!(
                provider_error_kind(error),
                Some(ProviderErrorKind::RateLimit | ProviderErrorKind::CapacityUnavailable)
            )
        });
        if throttled {
            let productive_floor = self
                .config
                .foreground_reserved
                .saturating_add(1)
                .min(self.config.max_concurrent);
            state.current_limit = state.current_limit.saturating_sub(1).max(productive_floor);
            state.success_streak = 0;
            state.last_throttle = Some(Instant::now());
        } else if result.is_ok() {
            state.success_streak += 1;
            let cooled_down = state
                .last_throttle
                .is_none_or(|at| at.elapsed() >= ADAPTIVE_COOLDOWN);
            if cooled_down
                && state.success_streak >= RECOVERY_SUCCESSES
                && state.current_limit < self.config.max_concurrent
            {
                state.current_limit += 1;
                state.success_streak = 0;
            }
        } else {
            state.success_streak = 0;
        }
        drop(state);
        self.notify.notify_waiters();
    }
}

#[async_trait]
impl Provider for ConcurrencyLimitedProvider {
    fn capabilities(&self) -> crate::provider::ProviderCapabilities {
        self.inner.capabilities()
    }

    async fn stream(
        &self,
        request: ChatRequest,
        sink: &mut (dyn FnMut(StreamEvent) + Send),
    ) -> Result<Completion> {
        let auxiliary = !request.user_turn;
        let _permit = self.acquire(auxiliary).await;
        let result = self.inner.stream(request, sink).await;
        self.record_result(&result).await;
        result
    }

    async fn list_models(&self) -> Result<Vec<ServedModel>> {
        self.inner.list_models().await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use futures_util::FutureExt;

    use super::*;
    use crate::{Content, RequestProfile, Usage};
    use tokio::sync::Semaphore;

    struct BlockingProvider {
        active: AtomicUsize,
        peak: AtomicUsize,
        release: Semaphore,
    }

    #[async_trait]
    impl Provider for BlockingProvider {
        async fn stream(
            &self,
            _request: ChatRequest,
            _sink: &mut (dyn FnMut(StreamEvent) + Send),
        ) -> Result<Completion> {
            let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(active, Ordering::SeqCst);
            let permit = self.release.acquire().await.unwrap();
            permit.forget();
            self.active.fetch_sub(1, Ordering::SeqCst);
            Ok(Completion {
                content: vec![Content::Text("ok".into())],
                usage: Usage::default(),
                stop_reason: None,
                refusal: None,
                tool_call_channel: crate::ToolCallChannel::None,
            })
        }
    }

    struct NeverProvider;

    #[async_trait]
    impl Provider for NeverProvider {
        async fn stream(
            &self,
            _request: ChatRequest,
            _sink: &mut (dyn FnMut(StreamEvent) + Send),
        ) -> Result<Completion> {
            unreachable!()
        }
    }

    fn request() -> ChatRequest {
        ChatRequest {
            model: "test".into(),
            request_id: None,
            user_turn: true,
            canonical_objective: None,
            messages: Vec::new().into(),
            tools: Vec::new().into(),
            max_tokens: 16,
            temperature: None,
            top_p: None,
            frequency_penalty: None,
            thinking_budget: None,
            reasoning_effort: None,
            retry_attempt: 0,
            profile: RequestProfile::default(),
        }
    }

    #[tokio::test]
    async fn bounds_streams_across_shared_provider() {
        let inner = Arc::new(BlockingProvider {
            active: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            release: Semaphore::new(0),
        });
        struct Shared(Arc<BlockingProvider>);
        #[async_trait]
        impl Provider for Shared {
            async fn stream(
                &self,
                request: ChatRequest,
                sink: &mut (dyn FnMut(StreamEvent) + Send),
            ) -> Result<Completion> {
                self.0.stream(request, sink).await
            }
        }
        let provider: Arc<dyn Provider> =
            Arc::new(ConcurrencyLimitedProvider::new(Box::new(Shared(inner.clone())), 2).unwrap());
        let mut tasks = Vec::new();
        for _ in 0..4 {
            let provider = provider.clone();
            tasks.push(tokio::spawn(async move {
                provider.stream(request(), &mut |_| {}).await.unwrap();
            }));
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        assert_eq!(inner.peak.load(Ordering::SeqCst), 2);
        inner.release.add_permits(4);
        for task in tasks {
            task.await.unwrap();
        }
        assert_eq!(inner.peak.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn auxiliary_requests_leave_a_foreground_slot() {
        let inner = Arc::new(BlockingProvider {
            active: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            release: Semaphore::new(0),
        });
        struct Shared(Arc<BlockingProvider>);
        #[async_trait]
        impl Provider for Shared {
            async fn stream(
                &self,
                request: ChatRequest,
                sink: &mut (dyn FnMut(StreamEvent) + Send),
            ) -> Result<Completion> {
                self.0.stream(request, sink).await
            }
        }
        let provider: Arc<dyn Provider> = Arc::new(
            ConcurrencyLimitedProvider::with_config(
                Box::new(Shared(inner.clone())),
                ProviderConcurrencyConfig {
                    max_concurrent: 3,
                    foreground_reserved: 1,
                    adaptive: false,
                },
            )
            .unwrap(),
        );
        let mut auxiliary = request();
        auxiliary.user_turn = false;
        let mut tasks = Vec::new();
        for _ in 0..3 {
            let provider = provider.clone();
            let request = auxiliary.clone();
            tasks.push(tokio::spawn(async move {
                provider.stream(request, &mut |_| {}).await.unwrap();
            }));
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
        assert_eq!(inner.active.load(Ordering::SeqCst), 2);
        let foreground = provider.clone();
        tasks.push(tokio::spawn(async move {
            foreground.stream(request(), &mut |_| {}).await.unwrap();
        }));
        tokio::time::sleep(Duration::from_millis(25)).await;
        assert_eq!(inner.active.load(Ordering::SeqCst), 3);
        inner.release.add_permits(4);
        for task in tasks {
            task.await.unwrap();
        }
    }

    #[tokio::test]
    async fn adaptive_throttling_preserves_an_auxiliary_slot() {
        let provider = ConcurrencyLimitedProvider::with_config(
            Box::new(NeverProvider),
            ProviderConcurrencyConfig {
                max_concurrent: 8,
                foreground_reserved: 1,
                adaptive: true,
            },
        )
        .unwrap();
        let throttled: Result<Completion> = Err(crate::ProviderError::new(
            ProviderErrorKind::CapacityUnavailable,
            "test throttle",
        )
        .into());

        for _ in 0..16 {
            provider.record_result(&throttled).await;
        }

        let state = provider.state.lock().await;
        assert_eq!(state.current_limit, 2);
        assert_eq!(
            state
                .current_limit
                .saturating_sub(provider.config.foreground_reserved),
            1,
            "adaptive backoff must retain one auxiliary slot"
        );
    }

    #[tokio::test]
    async fn degraded_limit_self_heals_and_keeps_the_foreground_reservation() {
        let provider = ConcurrencyLimitedProvider::with_config(
            Box::new(NeverProvider),
            ProviderConcurrencyConfig {
                max_concurrent: 3,
                foreground_reserved: 1,
                adaptive: true,
            },
        )
        .unwrap();
        provider.state.lock().await.current_limit = 1;

        let auxiliary = provider
            .acquire(true)
            .now_or_never()
            .expect("an auxiliary request must not deadlock at the reserved limit");
        {
            let state = provider.state.lock().await;
            assert_eq!(state.current_limit, 2);
            assert_eq!(state.in_flight, 1);
            assert_eq!(state.auxiliary_in_flight, 1);
        }

        assert!(
            provider.acquire(true).now_or_never().is_none(),
            "a second auxiliary request must leave the foreground slot reserved"
        );
        let foreground = provider
            .acquire(false)
            .now_or_never()
            .expect("the reserved foreground slot must remain available");
        {
            let state = provider.state.lock().await;
            assert_eq!(state.in_flight, 2);
            assert_eq!(state.auxiliary_in_flight, 1);
        }

        drop(foreground);
        drop(auxiliary);
    }

    #[test]
    fn rejects_zero_concurrency() {
        struct Never;
        #[async_trait]
        impl Provider for Never {
            async fn stream(
                &self,
                _request: ChatRequest,
                _sink: &mut (dyn FnMut(StreamEvent) + Send),
            ) -> Result<Completion> {
                unreachable!()
            }
        }
        assert!(ConcurrencyLimitedProvider::new(Box::new(Never), 0).is_err());
    }

    #[test]
    fn rejects_invalid_reservation() {
        struct Never;
        #[async_trait]
        impl Provider for Never {
            async fn stream(
                &self,
                _request: ChatRequest,
                _sink: &mut (dyn FnMut(StreamEvent) + Send),
            ) -> Result<Completion> {
                unreachable!()
            }
        }
        assert!(
            ConcurrencyLimitedProvider::with_config(
                Box::new(Never),
                ProviderConcurrencyConfig {
                    max_concurrent: 2,
                    foreground_reserved: 2,
                    adaptive: false,
                },
            )
            .is_err()
        );
    }
}
