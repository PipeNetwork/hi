//! Public Outcome Contract HTTP client for interactive `hi`.
//!
//! Interactive `hi` is a **client** of `/v1/tasks` and siblings. It must not
//! host the RSI control plane. Unknown request fields are rejected
//! server-side, so this crate serializes only the live OpenAPI kernel.

mod client;
mod error;
mod types;

pub use client::{OutcomeClient, OutcomeClientConfig};
pub use error::{OutcomeError, OutcomeErrorKind};
pub use types::*;

/// Clamp a USD spend cap to the public task contract (`$0.01`–`$25`).
pub fn clamp_cost_usd(usd: f64) -> f64 {
    if !usd.is_finite() {
        return TASK_MIN_COST_USD;
    }
    usd.clamp(TASK_MIN_COST_USD, TASK_MAX_COST_USD)
}

/// Clamp a deadline to the public task contract (30s–3600s, default 1800s).
pub fn clamp_deadline_secs(secs: Option<u64>) -> u64 {
    secs.unwrap_or(TASK_DEFAULT_DEADLINE_SECS)
        .clamp(TASK_MIN_DEADLINE_SECS, TASK_MAX_DEADLINE_SECS)
}

/// Convert RSI micro-USD spend into a public `maximum_cost_usd`.
pub fn cost_usd_from_microusd(microusd: u64) -> f64 {
    clamp_cost_usd(microusd as f64 / 1_000_000.0)
}

/// Normalize an origin that may be `https://host`, `https://host/`, or `https://host/v1`.
pub fn normalize_origin(url: &str) -> String {
    let trimmed = url.trim().trim_end_matches('/');
    trimmed
        .strip_suffix("/v1")
        .unwrap_or(trimmed)
        .trim_end_matches('/')
        .to_string()
}

#[cfg(test)]
mod kernel_tests {
    use super::*;

    fn fixture(name: &str) -> &'static str {
        match name {
            "create" => include_str!("../fixtures/outcome-kernel/create.json"),
            "quote" => include_str!("../fixtures/outcome-kernel/quote.json"),
            "repair" => include_str!("../fixtures/outcome-kernel/repair.json"),
            "receipt" => include_str!("../fixtures/outcome-kernel/receipt.json"),
            "ledger" => include_str!("../fixtures/outcome-kernel/ledger.json"),
            "verification" => include_str!("../fixtures/outcome-kernel/verification.json"),
            _ => panic!("unknown outcome kernel fixture {name}"),
        }
    }

    #[test]
    fn golden_kernel_parses_and_rejects_unknown_fields() {
        assert_eq!(RECEIPTS_VERIFY_PATH, "/v1/receipts/verify");
        assert_eq!(HI_WORKER_HEARTBEAT_SERVICE, "rsi-hi-worker");
        assert_eq!(TASKS_PATH, "/v1/tasks");
        let _: TaskCreateRequest = serde_json::from_str(fixture("create")).unwrap();
        let _: QuoteCreateRequest = serde_json::from_str(fixture("quote")).unwrap();
        let _: RepairCreateRequest = serde_json::from_str(fixture("repair")).unwrap();
        let receipt: ReceiptVerifyRequest = serde_json::from_str(fixture("receipt")).unwrap();
        assert_eq!(receipt.task_id.as_deref(), Some("task_1"));
        let ledger: LedgerEventCreateRequest = serde_json::from_str(fixture("ledger")).unwrap();
        assert_eq!(ledger.outcome, "succeeded");
        let _: VerificationCreateRequest = serde_json::from_str(fixture("verification")).unwrap();
        let mut unknown: serde_json::Value = serde_json::from_str(fixture("create")).unwrap();
        unknown
            .as_object_mut()
            .unwrap()
            .insert("webhook_url".into(), serde_json::json!("https://example"));
        assert!(serde_json::from_value::<TaskCreateRequest>(unknown).is_err());
    }
}
