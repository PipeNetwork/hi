//! Account for completed responses discarded during a compatibility repair.

use crate::{ProviderError, Usage};

pub(super) fn include_previous(mut latest: Usage, previous: Usage) -> Usage {
    let rate_limits = latest.rate_limits.or(previous.rate_limits);
    // Usage::add sums billed tokens while preserving the receiver's context
    // occupancy. Its rate-limit rule assumes chronological inputs, so restore
    // the most recent snapshot after adding this earlier attempt.
    latest.add(previous);
    latest.rate_limits = rate_limits;
    latest
}

pub(super) fn error_with_previous(mut error: ProviderError, previous: Usage) -> ProviderError {
    error.usage = include_previous(error.usage, previous);
    error
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RateLimitBucket, RateLimitState};

    #[test]
    fn sums_billed_tokens_but_keeps_latest_context_and_rate_limits() {
        let prior = Usage {
            input_tokens: 100,
            output_tokens: 10,
            cache_read_tokens: 40,
            cache_creation_tokens: 2,
            input_includes_cache: true,
            context_occupancy: 100,
            rate_limits: Some(RateLimitState {
                requests_min: RateLimitBucket {
                    remaining: 9,
                    ..Default::default()
                },
                ..Default::default()
            }),
            estimated: true,
        };
        let latest = Usage {
            input_tokens: 200,
            output_tokens: 20,
            cache_read_tokens: 80,
            cache_creation_tokens: 3,
            input_includes_cache: true,
            context_occupancy: 200,
            rate_limits: Some(RateLimitState {
                requests_min: RateLimitBucket {
                    remaining: 8,
                    ..Default::default()
                },
                ..Default::default()
            }),
            estimated: false,
        };
        let total = include_previous(latest, prior);
        assert_eq!(total.input_tokens, 300);
        assert_eq!(total.output_tokens, 30);
        assert_eq!(total.cache_read_tokens, 120);
        assert_eq!(total.cache_creation_tokens, 5);
        assert_eq!(total.context_occupancy, 200);
        assert_eq!(total.rate_limits, latest.rate_limits);
        assert!(total.input_includes_cache);
        assert!(total.estimated);
    }
}
