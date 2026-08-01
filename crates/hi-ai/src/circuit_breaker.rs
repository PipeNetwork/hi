//! Circuit breaker — sliding-window-with-min-samples algorithm.
//!
//! Inspired by grok-build's `xai-circuit-breaker` crate. The breaker trips
//! when `sample_count >= min_samples AND error_rate >= error_rate_threshold`
//! over the live window. Three states: `Closed` (normal), `Open` (rejecting),
//! `HalfOpen` (probing).
//!
//! Integrated into [`crate::fallback::FallbackProvider`] so a flaky backend
//! is automatically skipped after repeated failures, without waiting for
//! each call to time out.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Tri-state circuit-breaker status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreakerState {
    /// Normal operation — requests pass through.
    Closed,
    /// Rejecting all requests — too many recent failures.
    Open,
    /// Probing — allowing a limited number of test requests to see if the
    /// backend has recovered.
    HalfOpen,
}

/// Outcome of a request, fed back to the breaker via [`CircuitBreaker::record`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Success,
    Failure,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BreakerEvent {
    pub previous: BreakerState,
    pub current: BreakerState,
    pub rejected: bool,
}

pub trait BreakerObserver: Send + Sync {
    fn on_breaker_event(&self, event: &BreakerEvent);
}

/// Configuration for a [`CircuitBreaker`].
#[derive(Debug, Clone)]
pub struct BreakerConfig {
    /// How long the sliding window covers.
    pub window_duration: Duration,
    /// Minimum samples before the breaker can trip.
    pub min_samples: usize,
    /// Error rate (0.0–1.0) that trips the breaker.
    pub error_rate_threshold: f64,
    /// How long the breaker stays open before transitioning to half-open.
    pub open_duration: Duration,
    /// Maximum concurrent probes in half-open state.
    pub half_open_max_probes: usize,
    /// How long a granted probe may stay unrecorded before its slot is
    /// reclaimed (the request future was cancelled). Must comfortably exceed
    /// normal request latency — a streaming LLM probe can legitimately run for
    /// minutes, and reclaiming a live probe's slot admits an extra concurrent
    /// probe against a recovering backend.
    pub half_open_probe_timeout: Duration,
}

impl Default for BreakerConfig {
    fn default() -> Self {
        Self::client()
    }
}

impl BreakerConfig {
    /// Server preset: stricter trip threshold, short cool-down.
    pub fn server() -> Self {
        Self {
            window_duration: Duration::from_secs(60),
            min_samples: 10,
            error_rate_threshold: 0.5,
            open_duration: Duration::from_secs(10),
            half_open_max_probes: 1,
            half_open_probe_timeout: Duration::from_secs(300),
        }
    }

    /// Client preset: fewer samples, longer cool-down.
    pub fn client() -> Self {
        Self {
            window_duration: Duration::from_secs(60),
            min_samples: 5,
            error_rate_threshold: 0.5,
            open_duration: Duration::from_secs(60),
            half_open_max_probes: 1,
            half_open_probe_timeout: Duration::from_secs(300),
        }
    }
}

/// A circuit breaker for a single backend. Tracks success/failure outcomes
/// over a sliding window and transitions between [`BreakerState`]s.
pub struct CircuitBreaker {
    config: BreakerConfig,
    state: Mutex<BreakerInner>,
    observer: Option<Arc<dyn BreakerObserver>>,
}

struct BreakerInner {
    state: BreakerState,
    opened_at: Option<Instant>,
    half_open_probes: usize,
    /// When the most recent half-open probe was granted. Probes whose outcome
    /// is never recorded (the request future was cancelled) are reclaimed via
    /// this timestamp; otherwise one cancelled probe locks the backend out for
    /// the rest of the process.
    half_open_probe_granted_at: Option<Instant>,
    window: VecDeque<(Instant, bool)>,
}

impl BreakerInner {
    fn grant_probe(&mut self, max_probes: usize, probe_timeout: Duration) -> Result<(), String> {
        if self.half_open_probes < max_probes {
            self.half_open_probes += 1;
            self.half_open_probe_granted_at = Some(Instant::now());
            Ok(())
        } else if self
            .half_open_probe_granted_at
            .is_none_or(|granted| granted.elapsed() >= probe_timeout)
        {
            // A probe's outcome was never recorded — its request future was
            // dropped (user cancelled the turn). Reclaim ONE slot instead of
            // rejecting forever: probe count stays at the max so the
            // concurrency bound holds even if the old probe turns out to be
            // alive, and the refreshed timestamp restarts the timeout.
            self.half_open_probe_granted_at = Some(Instant::now());
            Ok(())
        } else {
            Err("circuit breaker half-open; probe slots exhausted".into())
        }
    }
}

impl CircuitBreaker {
    pub fn new(config: BreakerConfig) -> Self {
        Self {
            config,
            state: Mutex::new(BreakerInner {
                state: BreakerState::Closed,
                opened_at: None,
                half_open_probes: 0,
                half_open_probe_granted_at: None,
                window: VecDeque::new(),
            }),
            observer: None,
        }
    }

    pub fn with_observer(mut self, observer: Arc<dyn BreakerObserver>) -> Self {
        self.observer = Some(observer);
        self
    }

    fn observe(&self, previous: BreakerState, current: BreakerState, rejected: bool) {
        if previous != current {
            hi_observability::record(hi_observability::ReliabilityEvent::BreakerTransition);
            tracing::info!(
                target: "hi::reliability",
                event_kind = "breaker_transition",
                previous_state = ?previous,
                current_state = ?current,
                rejected,
            );
        }
        // Invoke external code without holding the breaker's state lock.
        if let Some(observer) = &self.observer {
            observer.on_breaker_event(&BreakerEvent {
                previous,
                current,
                rejected,
            });
        }
    }

    /// Current state of the breaker.
    pub fn state(&self) -> BreakerState {
        self.state.lock().unwrap().state
    }

    /// Whether the breaker is open (rejecting requests).
    pub fn is_open(&self) -> bool {
        matches!(self.state(), BreakerState::Open)
    }

    /// Check if a request is allowed. Returns `Ok(())` if allowed, or an error
    /// message if the breaker is open. In half-open, allows up to
    /// `half_open_max_probes` concurrent probes.
    pub fn check(&self) -> Result<(), String> {
        let mut inner = self.state.lock().unwrap();
        let previous = inner.state;
        let result = match inner.state {
            BreakerState::Closed => Ok(()),
            BreakerState::Open => {
                // Check if cool-down has elapsed → transition to half-open.
                if let Some(opened) = inner.opened_at {
                    if opened.elapsed() >= self.config.open_duration {
                        inner.state = BreakerState::HalfOpen;
                        inner.half_open_probes = 0;
                        // Fall through to half-open logic.
                    } else {
                        let error = format!(
                            "circuit breaker open; retry after {:.1}s",
                            self.config.open_duration.as_secs_f64()
                                - opened.elapsed().as_secs_f64()
                        );
                        drop(inner);
                        self.observe(previous, previous, true);
                        return Err(error);
                    }
                }
                // Half-open logic (fall-through from open→half-open transition).
                inner.grant_probe(
                    self.config.half_open_max_probes,
                    self.config.half_open_probe_timeout,
                )
            }
            BreakerState::HalfOpen => inner.grant_probe(
                self.config.half_open_max_probes,
                self.config.half_open_probe_timeout,
            ),
        };
        let current = inner.state;
        let rejected = result.is_err();
        drop(inner);
        if current != previous || rejected {
            self.observe(previous, current, rejected);
        }
        result
    }

    /// Record the outcome of a request. Transitions states as needed.
    pub fn record(&self, outcome: Outcome) {
        let mut inner = self.state.lock().unwrap();
        let previous = inner.state;
        let now = Instant::now();
        let is_success = outcome == Outcome::Success;

        // Evict expired entries from the sliding window. `checked_sub`: the
        // monotonic clock starts near zero at boot, and `now - window` panics
        // when the process starts within the first window_duration of boot.
        if let Some(cutoff) = now.checked_sub(self.config.window_duration) {
            while let Some(&(ts, _)) = inner.window.front() {
                if ts < cutoff {
                    inner.window.pop_front();
                } else {
                    break;
                }
            }
        }

        // Record the outcome.
        inner.window.push_back((now, is_success));

        match inner.state {
            BreakerState::HalfOpen => {
                if is_success {
                    // Probe succeeded → close the breaker. Drop the window
                    // samples from the outage: recovery starts with a clean
                    // slate, otherwise the very next request re-trips on stale
                    // failures and the backend flaps once per open_duration for
                    // the rest of the window.
                    inner.state = BreakerState::Closed;
                    inner.opened_at = None;
                    inner.half_open_probes = 0;
                    inner.half_open_probe_granted_at = None;
                    inner.window.clear();
                } else {
                    // Probe failed → re-open.
                    inner.state = BreakerState::Open;
                    inner.opened_at = Some(now);
                    inner.half_open_probes = 0;
                    inner.half_open_probe_granted_at = None;
                }
            }
            BreakerState::Closed => {
                // Check if we should trip the breaker.
                let samples = inner.window.len();
                if samples >= self.config.min_samples {
                    let failures = inner.window.iter().filter(|(_, ok)| !ok).count();
                    let error_rate = failures as f64 / samples as f64;
                    if error_rate >= self.config.error_rate_threshold {
                        inner.state = BreakerState::Open;
                        inner.opened_at = Some(now);
                    }
                }
            }
            BreakerState::Open => {
                // Already open — the outcome was from a probe that slipped through
                // or a race. Just record it.
            }
        }
        let current = inner.state;
        drop(inner);
        if current != previous {
            self.observe(previous, current, false);
        }
    }

    /// Current error rate over the sliding window (0.0–1.0).
    pub fn error_rate(&self) -> f64 {
        let inner = self.state.lock().unwrap();
        let samples = inner.window.len();
        if samples == 0 {
            return 0.0;
        }
        let failures = inner.window.iter().filter(|(_, ok)| !ok).count();
        failures as f64 / samples as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;
    use std::thread;

    struct RecordingObserver(StdMutex<Vec<BreakerEvent>>);

    impl BreakerObserver for RecordingObserver {
        fn on_breaker_event(&self, event: &BreakerEvent) {
            self.0.lock().unwrap().push(*event);
        }
    }

    #[test]
    fn closed_allows_requests() {
        let breaker = CircuitBreaker::new(BreakerConfig::client());
        assert_eq!(breaker.state(), BreakerState::Closed);
        assert!(breaker.check().is_ok());
    }

    #[test]
    fn observer_sees_transition_and_rejection() {
        let observer = Arc::new(RecordingObserver(StdMutex::new(Vec::new())));
        let breaker = CircuitBreaker::new(BreakerConfig {
            min_samples: 1,
            ..BreakerConfig::client()
        })
        .with_observer(observer.clone());

        breaker.record(Outcome::Failure);
        assert!(breaker.check().is_err());

        assert_eq!(observer.0.lock().unwrap().len(), 2);
        assert_eq!(observer.0.lock().unwrap()[0].current, BreakerState::Open);
        assert!(observer.0.lock().unwrap()[1].rejected);
    }

    #[test]
    fn trips_after_enough_failures() {
        let config = BreakerConfig {
            min_samples: 3,
            error_rate_threshold: 0.5,
            ..BreakerConfig::client()
        };
        let breaker = CircuitBreaker::new(config);

        // Record 3 failures (meets min_samples, 100% error rate).
        breaker.record(Outcome::Failure);
        breaker.record(Outcome::Failure);
        assert_eq!(breaker.state(), BreakerState::Closed); // only 2 samples

        breaker.record(Outcome::Failure);
        assert_eq!(breaker.state(), BreakerState::Open);
        assert!(breaker.check().is_err());
    }

    #[test]
    fn does_not_trip_below_min_samples() {
        let config = BreakerConfig {
            min_samples: 10,
            ..BreakerConfig::client()
        };
        let breaker = CircuitBreaker::new(config);

        for _ in 0..5 {
            breaker.record(Outcome::Failure);
        }
        assert_eq!(breaker.state(), BreakerState::Closed);
    }

    #[test]
    fn does_not_trip_below_error_rate_threshold() {
        let config = BreakerConfig {
            min_samples: 4,
            error_rate_threshold: 0.75,
            ..BreakerConfig::client()
        };
        let breaker = CircuitBreaker::new(config);

        // 50% error rate, below 75% threshold.
        breaker.record(Outcome::Success);
        breaker.record(Outcome::Failure);
        breaker.record(Outcome::Success);
        breaker.record(Outcome::Failure);
        assert_eq!(breaker.state(), BreakerState::Closed);
    }

    #[test]
    fn half_open_closes_on_success() {
        let config = BreakerConfig {
            min_samples: 2,
            open_duration: Duration::from_millis(10),
            ..BreakerConfig::client()
        };
        let breaker = CircuitBreaker::new(config);

        breaker.record(Outcome::Failure);
        breaker.record(Outcome::Failure);
        assert_eq!(breaker.state(), BreakerState::Open);

        // Wait for cool-down.
        thread::sleep(Duration::from_millis(20));

        // check() transitions to half-open and allows a probe.
        assert!(breaker.check().is_ok());
        assert_eq!(breaker.state(), BreakerState::HalfOpen);

        // Probe succeeds → breaker closes.
        breaker.record(Outcome::Success);
        assert_eq!(breaker.state(), BreakerState::Closed);
    }

    #[test]
    fn half_open_reopens_on_failure() {
        let config = BreakerConfig {
            min_samples: 2,
            open_duration: Duration::from_millis(10),
            ..BreakerConfig::client()
        };
        let breaker = CircuitBreaker::new(config);

        breaker.record(Outcome::Failure);
        breaker.record(Outcome::Failure);
        assert_eq!(breaker.state(), BreakerState::Open);

        thread::sleep(Duration::from_millis(20));
        assert!(breaker.check().is_ok());
        assert_eq!(breaker.state(), BreakerState::HalfOpen);

        // Probe fails → breaker re-opens.
        breaker.record(Outcome::Failure);
        assert_eq!(breaker.state(), BreakerState::Open);
    }

    #[test]
    fn cancelled_probe_slot_is_reclaimed_after_probe_timeout() {
        let config = BreakerConfig {
            min_samples: 2,
            open_duration: Duration::from_millis(10),
            // Wide enough that the back-to-back check() pair below cannot
            // straddle it even on a stalled CI machine.
            half_open_probe_timeout: Duration::from_secs(2),
            ..BreakerConfig::client()
        };
        let breaker = CircuitBreaker::new(config);
        breaker.record(Outcome::Failure);
        breaker.record(Outcome::Failure);
        thread::sleep(Duration::from_millis(20));
        // Probe granted, but its outcome is never recorded (cancelled turn).
        assert!(breaker.check().is_ok());
        assert!(breaker.check().is_err());
        thread::sleep(Duration::from_millis(2_100));
        // The slot must be reclaimed, not locked out for the process lifetime.
        assert!(breaker.check().is_ok());
    }

    #[test]
    fn recovery_clears_window_so_success_does_not_retrip() {
        let config = BreakerConfig {
            min_samples: 2,
            open_duration: Duration::from_millis(10),
            ..BreakerConfig::client()
        };
        let breaker = CircuitBreaker::new(config);
        breaker.record(Outcome::Failure);
        breaker.record(Outcome::Failure);
        assert_eq!(breaker.state(), BreakerState::Open);
        thread::sleep(Duration::from_millis(20));
        assert!(breaker.check().is_ok());
        breaker.record(Outcome::Success);
        assert_eq!(breaker.state(), BreakerState::Closed);
        // Stale outage samples must not re-trip the breaker on a success.
        breaker.record(Outcome::Success);
        breaker.record(Outcome::Success);
        assert_eq!(breaker.state(), BreakerState::Closed);
    }

    #[test]
    fn error_rate_reflects_window() {
        let breaker = CircuitBreaker::new(BreakerConfig::client());
        breaker.record(Outcome::Success);
        breaker.record(Outcome::Success);
        breaker.record(Outcome::Failure);
        assert!((breaker.error_rate() - 1.0 / 3.0).abs() < 0.01);
    }
}
