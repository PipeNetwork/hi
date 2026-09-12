//! Plan-boundary compaction candidates priced against cache-write cost.
//!
//! Completing a plan step is a candidate, not a command. Compaction runs when
//! expected remaining requests repay the cache-write, or when occupancy is
//! inside a reserved window of the model limit. The existing high-water
//! occupancy compact remains the window-protection backstop.

use hi_tools::PlanStatus;

/// Tokens reserved at the end of the window that force compaction even when
/// economics would defer. Conservative hi default — not an external constant.
pub const DEFAULT_WINDOW_RESERVE_TOKENS: u64 = 8_192;
/// First compaction may use a longer remaining-request horizon.
pub const DEFAULT_FIRST_COMPACTION_REQUEST_SCALE: f64 = 2.0;
/// Later compactions need this margin over breakeven.
pub const DEFAULT_SUBSEQUENT_COMPACTION_MARGIN: f64 = 1.5;
/// Conservative cache-write/read cost ratio. Not NVIDIA's 12.5.
pub const DEFAULT_CACHE_WRITE_READ_RATIO: f64 = 8.0;
/// Estimated tokens in a native compaction summary.
pub const DEFAULT_MEMO_TOKENS: u64 = 1_000;
/// Recent tail kept verbatim when estimating archive size.
pub const DEFAULT_KEEP_RECENT_TOKENS: u64 = 20_000;

pub const PLAN_BOUNDARY_COMPACT_REORIENT: &str = "Plan-boundary compaction finished. The parent task is still active. Re-orient from the workspace and the remaining plan; do not assume prior reads are still in context.";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompactionReason {
    Economic,
    WindowProtection,
    DeferredEconomic,
    DeferredSubsequentMargin,
    DeferredCarriedDebt,
    HorizonUnavailable,
    CacheRatioUnavailable,
    NonPositiveSaving,
}

#[derive(Clone, Copy, Debug)]
pub struct CompactionEconomics {
    pub remaining_request_scale: f64,
    pub window_reserve_tokens: u64,
    pub first_compaction_request_scale: f64,
    pub subsequent_compaction_margin: f64,
}

impl Default for CompactionEconomics {
    fn default() -> Self {
        Self {
            remaining_request_scale: 1.0,
            window_reserve_tokens: DEFAULT_WINDOW_RESERVE_TOKENS,
            first_compaction_request_scale: DEFAULT_FIRST_COMPACTION_REQUEST_SCALE,
            subsequent_compaction_margin: DEFAULT_SUBSEQUENT_COMPACTION_MARGIN,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompactionDecision {
    pub compact: bool,
    pub reason: CompactionReason,
    pub breakeven_requests: Option<f64>,
    pub expected_remaining_requests: Option<f64>,
}

#[derive(Clone, Debug, Default)]
pub struct OnlineCompactState {
    pub completed_boundary_request_counts: Vec<u64>,
    pub requests_this_boundary: u64,
    pub native_compaction_count: u64,
    pub cache_debt_tokens: f64,
    pub positive_context_delta_total: u64,
    pub positive_context_delta_count: u64,
    pub last_context_tokens: u64,
    pub pending_boundary: bool,
    pub pending_reorient: bool,
}

impl OnlineCompactState {
    pub fn record_provider_request(&mut self, context_tokens: u64) {
        self.requests_this_boundary = self.requests_this_boundary.saturating_add(1);
        if context_tokens > self.last_context_tokens {
            self.positive_context_delta_total = self
                .positive_context_delta_total
                .saturating_add(context_tokens - self.last_context_tokens);
            self.positive_context_delta_count = self.positive_context_delta_count.saturating_add(1);
        }
        self.last_context_tokens = context_tokens;
    }

    pub fn record_boundary(&mut self) {
        self.completed_boundary_request_counts
            .push(self.requests_this_boundary.max(1));
        self.requests_this_boundary = 0;
        self.pending_boundary = true;
    }

    pub fn record_compaction(&mut self, write_tokens: u64, ratio: f64, saving_tokens: u64) {
        self.native_compaction_count = self.native_compaction_count.saturating_add(1);
        self.cache_debt_tokens += write_tokens as f64 * (ratio - 1.0).max(0.0);
        self.cache_debt_tokens = (self.cache_debt_tokens - saving_tokens as f64).max(0.0);
        self.pending_boundary = false;
        self.pending_reorient = true;
        self.requests_this_boundary = 0;
    }
}

pub fn newly_completed_plan_step(
    previous: &[hi_tools::PlanStep],
    next: &[hi_tools::PlanStep],
) -> bool {
    next.iter().any(|step| {
        step.status == PlanStatus::Done
            && previous
                .iter()
                .any(|old| old.title == step.title && old.status != PlanStatus::Done)
    })
}

pub fn remaining_plan_boundaries(plan: &[hi_tools::PlanStep]) -> u64 {
    plan.iter()
        .filter(|step| step.status != PlanStatus::Done)
        .count() as u64
}

pub fn decide_compaction(input: DecideCompactionInput) -> CompactionDecision {
    let saving_tokens = input.archive_tokens.saturating_sub(input.memo_tokens);
    let incremental_ratio = input
        .cache_write_read_ratio
        .map(|ratio| (ratio - 1.0).max(0.0));
    let breakeven = if saving_tokens > 0 {
        incremental_ratio.map(|ratio| (input.write_tokens as f64 * ratio) / saving_tokens as f64)
    } else {
        None
    };
    let combined_breakeven = if saving_tokens > 0 {
        incremental_ratio.map(|ratio| {
            (input.carried_debt_tokens + input.write_tokens as f64 * ratio) / saving_tokens as f64
        })
    } else {
        None
    };

    let expected_remaining = if input.completed_boundary_request_counts.is_empty() {
        None
    } else {
        let mean = input.completed_boundary_request_counts.iter().sum::<u64>() as f64
            / input.completed_boundary_request_counts.len() as f64;
        let unbounded = 1.0
            + (mean * input.remaining_boundaries as f64 * input.economics.remaining_request_scale)
                .floor();
        let window_bound = match (
            input.context_window_tokens,
            input.average_context_token_increment,
        ) {
            (Some(window), Some(increment)) if increment > 0 => {
                Some(window.saturating_sub(input.context_tokens) / increment)
            }
            _ => None,
        };
        Some(match window_bound {
            Some(bound) => unbounded.min(bound as f64),
            None => unbounded,
        })
    };

    let first = input.prior_compaction_count == 0;
    let effective_horizon = expected_remaining.map(|remaining| {
        if first {
            let scaled = remaining * input.economics.first_compaction_request_scale;
            match (
                input.context_window_tokens,
                input.average_context_token_increment,
            ) {
                (Some(window), Some(increment)) if increment > 0 => {
                    scaled.min((window.saturating_sub(input.context_tokens) / increment) as f64)
                }
                _ => scaled,
            }
        } else {
            remaining
        }
    });

    let window_protection = input.context_window_tokens.is_some_and(|window| {
        input.context_tokens >= window.saturating_sub(input.economics.window_reserve_tokens)
    });
    let base_economic = expected_remaining.is_some_and(|remaining| {
        remaining > 0.0 && breakeven.is_some_and(|need| need <= remaining)
    });
    let first_economic = first
        && effective_horizon.is_some_and(|remaining| {
            remaining > 0.0 && breakeven.is_some_and(|need| need <= remaining)
        });
    let subsequent_margin = !first
        && expected_remaining.is_some_and(|remaining| {
            breakeven.is_some_and(|need| {
                need * input.economics.subsequent_compaction_margin <= remaining
            })
        });
    let carried_debt_ok = !first
        && expected_remaining
            .is_some_and(|remaining| combined_breakeven.is_some_and(|need| need <= remaining));
    let economic = if first {
        first_economic
    } else {
        base_economic && subsequent_margin && carried_debt_ok
    };
    let compressible = saving_tokens > 0;
    let compact = window_protection || (compressible && economic);
    let reason = if window_protection {
        CompactionReason::WindowProtection
    } else if !compressible {
        CompactionReason::NonPositiveSaving
    } else if economic {
        CompactionReason::Economic
    } else if expected_remaining.is_none() {
        CompactionReason::HorizonUnavailable
    } else if breakeven.is_none() {
        CompactionReason::CacheRatioUnavailable
    } else if !first && base_economic && !subsequent_margin {
        CompactionReason::DeferredSubsequentMargin
    } else if !first && base_economic && !carried_debt_ok {
        CompactionReason::DeferredCarriedDebt
    } else {
        CompactionReason::DeferredEconomic
    };

    CompactionDecision {
        compact,
        reason,
        breakeven_requests: breakeven,
        expected_remaining_requests: expected_remaining,
    }
}

pub struct DecideCompactionInput<'a> {
    pub write_tokens: u64,
    pub archive_tokens: u64,
    pub memo_tokens: u64,
    pub context_tokens: u64,
    pub completed_boundary_request_counts: &'a [u64],
    pub remaining_boundaries: u64,
    pub average_context_token_increment: Option<u64>,
    pub context_window_tokens: Option<u64>,
    pub prior_compaction_count: u64,
    pub carried_debt_tokens: f64,
    pub cache_write_read_ratio: Option<f64>,
    pub economics: CompactionEconomics,
}

impl crate::Agent {
    pub(crate) async fn maybe_plan_boundary_compact(
        &mut self,
        ui: &mut dyn crate::Ui,
    ) -> anyhow::Result<bool> {
        if !self.config.memory.online_context_compact || !self.online_compact.pending_boundary {
            return Ok(false);
        }
        let write_tokens = crate::compaction::estimate_tokens(self.messages.as_slice());
        let archive_tokens = write_tokens.saturating_sub(DEFAULT_KEEP_RECENT_TOKENS);
        let increment = self
            .online_compact
            .positive_context_delta_total
            .checked_div(self.online_compact.positive_context_delta_count);
        let decision = decide_compaction(DecideCompactionInput {
            write_tokens,
            archive_tokens,
            memo_tokens: DEFAULT_MEMO_TOKENS,
            context_tokens: write_tokens.max(self.report.context_used),
            completed_boundary_request_counts: &self
                .online_compact
                .completed_boundary_request_counts,
            remaining_boundaries: remaining_plan_boundaries(self.goals.plan()),
            average_context_token_increment: increment,
            context_window_tokens: self.config.routing.context_window.map(u64::from),
            prior_compaction_count: self.online_compact.native_compaction_count,
            carried_debt_tokens: self.online_compact.cache_debt_tokens,
            cache_write_read_ratio: Some(self.config.memory.cache_write_read_ratio),
            economics: CompactionEconomics::default(),
        });
        if !decision.compact {
            self.online_compact.pending_boundary = false;
            return Ok(false);
        }
        ui.status("plan-boundary compact — reclaiming completed subtask context");
        self.compact(ui).await?;
        self.online_compact.record_compaction(
            write_tokens,
            self.config.memory.cache_write_read_ratio,
            archive_tokens.saturating_sub(DEFAULT_MEMO_TOKENS),
        );
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hi_tools::{PlanStatus, PlanStep};

    fn base_input<'a>(counts: &'a [u64]) -> DecideCompactionInput<'a> {
        DecideCompactionInput {
            write_tokens: 8_000,
            archive_tokens: 6_000,
            memo_tokens: DEFAULT_MEMO_TOKENS,
            context_tokens: 8_000,
            completed_boundary_request_counts: counts,
            remaining_boundaries: 4,
            average_context_token_increment: Some(500),
            context_window_tokens: Some(100_000),
            prior_compaction_count: 0,
            carried_debt_tokens: 0.0,
            cache_write_read_ratio: Some(DEFAULT_CACHE_WRITE_READ_RATIO),
            economics: CompactionEconomics::default(),
        }
    }

    #[test]
    fn compact_when_remaining_work_repays_the_write() {
        let counts = [4, 5, 4];
        let decision = decide_compaction(base_input(&counts));
        assert!(decision.compact, "{decision:?}");
        assert_eq!(decision.reason, CompactionReason::Economic);
    }

    #[test]
    fn defer_when_horizon_cannot_repay() {
        let counts = [1];
        let mut input = base_input(&counts);
        input.remaining_boundaries = 0;
        input.archive_tokens = 1_200;
        input.prior_compaction_count = 1;
        input.context_window_tokens = Some(200_000);
        input.context_tokens = 8_000;
        let decision = decide_compaction(input);
        assert!(!decision.compact, "{decision:?}");
        assert!(
            matches!(
                decision.reason,
                CompactionReason::DeferredEconomic
                    | CompactionReason::DeferredSubsequentMargin
                    | CompactionReason::DeferredCarriedDebt
            ),
            "{decision:?}"
        );
    }

    #[test]
    fn window_pressure_compacts_even_when_economics_defer() {
        let counts = [1];
        let mut input = base_input(&counts);
        input.remaining_boundaries = 0;
        input.prior_compaction_count = 1;
        input.context_window_tokens = Some(10_000);
        input.context_tokens = 9_500;
        let decision = decide_compaction(input);
        assert!(decision.compact, "{decision:?}");
        assert_eq!(decision.reason, CompactionReason::WindowProtection);
    }

    #[test]
    fn newly_completed_step_is_a_candidate() {
        let previous = vec![PlanStep {
            title: "edit".into(),
            status: PlanStatus::Active,
        }];
        let next = vec![PlanStep {
            title: "edit".into(),
            status: PlanStatus::Done,
        }];
        assert!(newly_completed_plan_step(&previous, &next));
        assert!(!newly_completed_plan_step(&next, &next));
    }
}
