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
use std::sync::Mutex;
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
        }
    }
}

/// A circuit breaker for a single backend. Tracks success/failure outcomes
/// over a sliding window and transitions between [`BreakerState`]s.
pub struct CircuitBreaker {
    config: BreakerConfig,
    state: Mutex<BreakerInner>,
}

struct BreakerInner {
    state: BreakerState,
    opened_at: Option<Instant>,
    half_open_probes: usize,
    window: VecDeque<(Instant, bool)>,
}

impl CircuitBreaker {
    pub fn new(config: BreakerConfig) -> Self {
        Self {
            config,
            state: Mutex::new(BreakerInner {
                state: BreakerState::Closed,
                opened_at: None,
                half_open_probes: 0,
                window: VecDeque::new(),
            }),
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
        match inner.state {
            BreakerState::Closed => Ok(()),
            BreakerState::Open => {
                // Check if cool-down has elapsed → transition to half-open.
                if let Some(opened) = inner.opened_at {
                    if opened.elapsed() >= self.config.open_duration {
                        inner.state = BreakerState::HalfOpen;
                        inner.half_open_probes = 0;
                        // Fall through to half-open logic.
                    } else {
                        return Err(format!(
                            "circuit breaker open; retry after {:.1}s",
                            self.config.open_duration.as_secs_f64()
                                - opened.elapsed().as_secs_f64()
                        ));
                    }
                }
                // Half-open logic (fall-through from open→half-open transition).
                if inner.half_open_probes < self.config.half_open_max_probes {
                    inner.half_open_probes += 1;
                    Ok(())
                } else {
                    Err("circuit breaker half-open; probe slots exhausted".into())
                }
            }
            BreakerState::HalfOpen => {
                if inner.half_open_probes < self.config.half_open_max_probes {
                    inner.half_open_probes += 1;
                    Ok(())
                } else {
                    Err("circuit breaker half-open; probe slots exhausted".into())
                }
            }
        }
    }

    /// Record the outcome of a request. Transitions states as needed.
    pub fn record(&self, outcome: Outcome) {
        let mut inner = self.state.lock().unwrap();
        let now = Instant::now();
        let is_success = outcome == Outcome::Success;

        // Evict expired entries from the sliding window.
        let cutoff = now - self.config.window_duration;
        while let Some(&(ts, _)) = inner.window.front() {
            if ts < cutoff {
                inner.window.pop_front();
            } else {
                break;
            }
        }

        // Record the outcome.
        inner.window.push_back((now, is_success));

        match inner.state {
            BreakerState::HalfOpen => {
                if is_success {
                    // Probe succeeded → close the breaker.
                    inner.state = BreakerState::Closed;
                    inner.opened_at = None;
                    inner.half_open_probes = 0;
                } else {
                    // Probe failed → re-open.
                    inner.state = BreakerState::Open;
                    inner.opened_at = Some(now);
                    inner.half_open_probes = 0;
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
    use std::thread;

    #[test]
    fn closed_allows_requests() {
        let breaker = CircuitBreaker::new(BreakerConfig::client());
        assert_eq!(breaker.state(), BreakerState::Closed);
        assert!(breaker.check().is_ok());
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
    fn error_rate_reflects_window() {
        let breaker = CircuitBreaker::new(BreakerConfig::client());
        breaker.record(Outcome::Success);
        breaker.record(Outcome::Success);
        breaker.record(Outcome::Failure);
        assert!((breaker.error_rate() - 1.0 / 3.0).abs() < 0.01);
    }
}
