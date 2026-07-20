//! Provider retry budgets, **review-answer** repair counters, and output-cap backoff.
//!
//! [`ReviewRepairState`] budgets quality nudges during [`super::phase::TurnPhase::Steer`].
//! It is **not** the workspace compile/lint/test loop — that is
//! [`crate::verify::WorkspaceRepairVerifier`] under
//! [`super::phase::TurnPhase::WorkspaceRepair`].

use std::collections::BTreeMap;

use hi_ai::{OutputCapError, ToolSpec};

use crate::steering::{EvidenceTracker, ReviewRepairMode};

pub(super) const MAX_TRANSIENT_ROUTE_RETRIES: u32 = 2;
pub(super) const TRANSIENT_ROUTE_RETRY_DELAYS: [u64; 2] = [2, 5];
pub(super) const MAX_TRANSIENT_ROUTE_RETRY_DELAY_SECS: u64 = 30;
/// Shared budget for ordinary 429s and temporary provider overload/capacity blips.
/// Exponential schedule so a sticky throttle has room to clear without hammering.
pub(super) const MAX_PROVIDER_OVERLOAD_RETRIES: u32 = 8;
pub(super) const PROVIDER_OVERLOAD_RETRY_DELAYS: [u64; 8] = [1, 2, 4, 8, 16, 32, 64, 120];
pub(super) const MAX_PROVIDER_OVERLOAD_RETRY_DELAY_SECS: u64 = 120;
pub(super) const MIN_OUTPUT_CAP_RETRY_TOKENS: u32 = 512;
pub(super) const INCOMPLETE_STATUS: &str = "turn stopped incomplete";

/// Per-turn budgets for **review-answer** repair modes (Steer phase).
///
/// Separate from [`crate::verify::WorkspaceRepairVerifier`]'s `max_rounds`.
#[derive(Default)]
pub(super) struct ReviewRepairState {
    pub(super) counts: BTreeMap<String, u32>,
    pub(super) exhaustion_reason: String,
}

impl ReviewRepairState {
    pub(super) fn count(&self, mode: ReviewRepairMode) -> u32 {
        self.counts.get(mode.key()).copied().unwrap_or(0)
    }

    pub(super) fn has_budget(
        &self,
        mode: ReviewRepairMode,
        budgets: &crate::config::ReviewRepairBudgets,
    ) -> bool {
        self.count(mode) < mode.limit_with(budgets)
    }

    pub(super) fn spend(
        &mut self,
        mode: ReviewRepairMode,
        evidence: &mut EvidenceTracker,
        budgets: &crate::config::ReviewRepairBudgets,
    ) -> bool {
        if !self.has_budget(mode, budgets) {
            return false;
        }
        let entry = self.counts.entry(mode.key().to_string()).or_insert(0);
        *entry = (*entry).saturating_add(1);
        evidence.quality_repair_nudges = evidence.quality_repair_nudges.saturating_add(1);
        true
    }

    pub(super) fn note(&mut self, mode: ReviewRepairMode) {
        let entry = self.counts.entry(mode.key().to_string()).or_insert(0);
        *entry = (*entry).saturating_add(1);
    }

    pub(super) fn exhausted(&mut self, mode: ReviewRepairMode) -> &'static str {
        let reason = mode.exhaustion_key();
        self.exhaustion_reason = reason.to_string();
        reason
    }
}

#[derive(Default)]
pub(super) struct TurnRetryState {
    pub(super) request_too_large_retried: bool,
    pub(super) output_cap_retry_attempted: bool,
    pub(super) transient_route_retries: u32,
    pub(super) provider_overload_retries: u32,
    pub(super) protocol_retries: u32,
    /// Cumulative invalid tool turns this turn — unlike `protocol_retries`, this
    /// never resets on valid output, so an alternating valid/invalid loop still
    /// trips the [`crate::MAX_TOOL_PROTOCOL_FAILURES`] circuit-breaker.
    pub(super) protocol_failures_total: u32,
    pub(super) protocol_text_fallbacks: u32,
}

impl TurnRetryState {
    pub(super) fn record_provider_success(&mut self) {
        self.output_cap_retry_attempted = false;
        self.transient_route_retries = 0;
        self.provider_overload_retries = 0;
    }
}

pub(super) fn output_cap_retry_tokens(current: u32, cap: OutputCapError) -> Option<u32> {
    let next = if let Some(available) = cap.available_output_tokens {
        available.min(current.saturating_sub(1))
    } else if current > 1024 {
        (current / 2).max(1024)
    } else {
        return None;
    };
    (next >= MIN_OUTPUT_CAP_RETRY_TOKENS && next < current).then_some(next)
}

pub(super) fn transient_route_retry_delay(retry: u32, err: &anyhow::Error) -> std::time::Duration {
    provider_retry_delay(
        retry,
        err,
        &TRANSIENT_ROUTE_RETRY_DELAYS,
        MAX_TRANSIENT_ROUTE_RETRY_DELAY_SECS,
    )
}

pub(super) fn provider_overload_retry_delay(
    retry: u32,
    err: &anyhow::Error,
) -> std::time::Duration {
    provider_retry_delay(
        retry,
        err,
        &PROVIDER_OVERLOAD_RETRY_DELAYS,
        MAX_PROVIDER_OVERLOAD_RETRY_DELAY_SECS,
    )
}

pub(super) fn provider_retry_delay(
    retry: u32,
    err: &anyhow::Error,
    default_delays: &[u64],
    max_delay_secs: u64,
) -> std::time::Duration {
    let default = default_delays
        .get(retry.saturating_sub(1) as usize)
        .copied()
        .unwrap_or(*default_delays.last().unwrap_or(&5));
    // Prefer the provider's Retry-After when it asks us to wait; treat an explicit
    // 0 as "retry immediately" (common on overload blips / tests). Missing values
    // fall through to the exponential table.
    let secs = match hi_ai::provider_retry_after_seconds(err) {
        Some(0) => 0,
        Some(secs) => secs.max(default).min(max_delay_secs),
        None => default.min(max_delay_secs),
    };
    if secs == 0 {
        return std::time::Duration::ZERO;
    }
    let jitter_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| u64::from(duration.subsec_millis()) % 250)
        .unwrap_or(0);
    std::time::Duration::from_secs(secs) + std::time::Duration::from_millis(jitter_ms)
}

/// Rate limits and temporary overload/capacity errors share the extended backoff budget.
pub(super) fn provider_error_is_backoff_retryable(err: &anyhow::Error) -> bool {
    matches!(
        hi_ai::provider_error_kind(err),
        Some(hi_ai::ProviderErrorKind::RateLimit)
    ) || hi_ai::provider_error_is_temporary_overload(err)
}

pub(super) fn delay_label(delay: std::time::Duration) -> String {
    if delay.is_zero() {
        "now".to_string()
    } else {
        format!("{}s", delay.as_secs())
    }
}

pub(super) fn estimate_tool_schema_tokens(tools: &[ToolSpec]) -> u64 {
    tools
        .iter()
        .map(|tool| {
            hi_ai::estimate_text_tokens(&tool.name)
                + hi_ai::estimate_text_tokens(&tool.description)
                + hi_ai::estimate_text_tokens(&tool.parameters.to_string())
        })
        .sum()
}
