//! Quote-checked diagnostic reduction. Off unless explicitly enabled.
//!
//! A receipt replaces the source log only when every quote is a contiguous
//! byte-for-byte substring of an archived copy, the receipt is smaller than
//! the source, and success/failure matches the observed outcome. Any check
//! failure leaves the original log. Deterministic [`crate::condense`] stays first.

use sha2::{Digest, Sha256};

pub const EVIDENCE_RECEIPT_PREFIX: &str = "hi_evidence_receipt_v1";
pub const REDUCER_RECEIPT_SCHEMA: &str = "hi_evidence_receipt_v1";
const MAX_EVIDENCE_ITEMS: usize = 12;
const MAX_QUOTE_CHARS: usize = 400;
const DEFAULT_MIN_BYTES: usize = 8_192;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvidenceKind {
    Fatal,
    Failure,
    Warning,
    Target,
    Summary,
}

impl EvidenceKind {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "fatal" => Some(Self::Fatal),
            "failure" => Some(Self::Failure),
            "warning" => Some(Self::Warning),
            "target" => Some(Self::Target),
            "summary" => Some(Self::Summary),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Fatal => "fatal",
            Self::Failure => "failure",
            Self::Warning => "warning",
            Self::Target => "target",
            Self::Summary => "summary",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedEvidence {
    pub kind: EvidenceKind,
    pub line: Option<usize>,
    pub quote: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidatedReceipt {
    pub status: &'static str,
    pub uncertain: bool,
    pub evidence: Vec<VerifiedEvidence>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReceiptValidation {
    Ok(ValidatedReceipt),
    Rejected(&'static str),
}

#[derive(Clone, Debug)]
pub struct EvidenceReducerConfig {
    pub enabled: bool,
    pub min_bytes: usize,
}

impl Default for EvidenceReducerConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            min_bytes: DEFAULT_MIN_BYTES,
        }
    }
}

/// Apply quote-checked reduction. `receipt_json` is the reducer model's JSON
/// object. Failures return the original `source`.
pub fn apply_quote_checked_reduction(
    config: &EvidenceReducerConfig,
    source: &str,
    is_error: bool,
    receipt_json: &str,
) -> String {
    if !config.enabled {
        return source.to_string();
    }
    match reduce_or_reject(config, source, is_error, receipt_json) {
        Ok(receipt) => receipt,
        Err(_) => source.to_string(),
    }
}

/// Nested reducer/model errors fail open and keep the original log.
pub fn reduce_from_reducer_result(
    config: &EvidenceReducerConfig,
    source: &str,
    is_error: bool,
    reducer_result: Result<String, &'static str>,
) -> String {
    match reducer_result {
        Ok(json) => apply_quote_checked_reduction(config, source, is_error, &json),
        Err(_) => source.to_string(),
    }
}

/// Quote-checked reduction after deterministic condense. Disabled, missing, or
/// invalid receipts keep `condensed`.
pub fn after_condense(
    condensed: &str,
    is_error: bool,
    config: &EvidenceReducerConfig,
    receipt_json: Result<String, &'static str>,
) -> String {
    if !config.enabled {
        return condensed.to_string();
    }
    reduce_from_reducer_result(config, condensed, is_error, receipt_json)
}

/// Canned nested-reducer callback. Production has no live model; tests supply JSON.
pub type EvidenceReducerHook =
    std::sync::Arc<dyn Fn(&str, bool) -> Result<String, &'static str> + Send + Sync>;

pub fn reduce_or_reject(
    config: &EvidenceReducerConfig,
    source: &str,
    is_error: bool,
    receipt_json: &str,
) -> Result<String, &'static str> {
    if !config.enabled {
        return Err("disabled");
    }
    if source.len() < config.min_bytes {
        return Err("below-min-bytes");
    }
    let archive_hash = sha256_hex(source.as_bytes());
    let checked = validate_receipt(receipt_json, source, &archive_hash, is_error);
    let ReceiptValidation::Ok(validated) = checked else {
        return match checked {
            ReceiptValidation::Rejected(reason) => Err(reason),
            ReceiptValidation::Ok(_) => unreachable!(),
        };
    };
    let receipt = receipt_text(source, &archive_hash, &validated);
    if receipt.len() >= source.len() {
        return Err("receipt-not-smaller");
    }
    Ok(receipt)
}

pub fn validate_receipt(
    raw: &str,
    source: &str,
    archive_hash: &str,
    is_error: bool,
) -> ReceiptValidation {
    let parsed: serde_json::Value = match serde_json::from_str(raw) {
        Ok(value) => value,
        Err(_) => return ReceiptValidation::Rejected("invalid-json"),
    };
    let expected_status = if is_error { "failure" } else { "success" };
    let evidence_value = parsed.get("evidence").and_then(|value| value.as_array());
    if parsed.get("schema").and_then(|value| value.as_str()) != Some(REDUCER_RECEIPT_SCHEMA)
        || parsed.get("source_sha256").and_then(|value| value.as_str()) != Some(archive_hash)
        || parsed.get("status").and_then(|value| value.as_str()) != Some(expected_status)
        || !parsed
            .get("uncertain")
            .is_some_and(|value| value.is_boolean())
        || evidence_value.is_none_or(|items| items.len() > MAX_EVIDENCE_ITEMS)
    {
        return ReceiptValidation::Rejected("schema-mismatch");
    }
    let mut evidence = Vec::new();
    let mut seen = HashSet::new();
    for item in evidence_value.unwrap() {
        let kind = item
            .get("kind")
            .and_then(|value| value.as_str())
            .and_then(EvidenceKind::parse);
        let quote = item.get("quote").and_then(|value| value.as_str());
        let (Some(kind), Some(quote)) = (kind, quote) else {
            return ReceiptValidation::Rejected("unverifiable-quote");
        };
        if quote.is_empty() || quote.chars().count() > MAX_QUOTE_CHARS || !source.contains(quote) {
            return ReceiptValidation::Rejected("unverifiable-quote");
        }
        let key = format!("{}\0{quote}", kind.as_str());
        if !seen.insert(key) {
            continue;
        }
        evidence.push(VerifiedEvidence {
            kind,
            line: line_number_of(source, quote),
            quote: quote.to_string(),
        });
    }
    if is_error
        && has_failure_signal(source)
        && !evidence
            .iter()
            .any(|item| matches!(item.kind, EvidenceKind::Fatal | EvidenceKind::Failure))
    {
        return ReceiptValidation::Rejected("missing-failure-evidence");
    }
    ReceiptValidation::Ok(ValidatedReceipt {
        status: expected_status,
        uncertain: parsed
            .get("uncertain")
            .and_then(|value| value.as_bool())
            .unwrap_or(false),
        evidence,
    })
}

pub fn receipt_text(source: &str, archive_hash: &str, validated: &ValidatedReceipt) -> String {
    let mut lines = vec![
        EVIDENCE_RECEIPT_PREFIX.to_string(),
        format!("status={}", validated.status),
        format!("uncertain={}", validated.uncertain),
        format!("source_sha256={archive_hash}"),
        format!("source_bytes={}", source.len()),
        format!("source_lines={}", source.lines().count()),
        "verified_evidence:".into(),
    ];
    if validated.evidence.is_empty() {
        lines.push("- none".into());
    } else {
        for item in &validated.evidence {
            lines.push(format!(
                "- kind={} line={} quote={}",
                item.kind.as_str(),
                item.line.unwrap_or(0),
                serde_json::to_string(&item.quote).unwrap_or_default()
            ));
        }
    }
    lines.join("\n")
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn line_number_of(source: &str, quote: &str) -> Option<usize> {
    let index = source.find(quote)?;
    Some(
        source[..index]
            .bytes()
            .filter(|byte| *byte == b'\n')
            .count()
            + 1,
    )
}

fn has_failure_signal(source: &str) -> bool {
    source.lines().any(|line| {
        let trimmed = line.trim();
        trimmed.starts_with("error:")
            || trimmed.starts_with("error[")
            || trimmed.contains("FAILED")
            || trimmed.contains("FAILURES")
            || trimmed.contains("panic")
    })
}

use std::collections::HashSet;

#[cfg(test)]
mod tests {
    use super::*;

    fn source_log() -> String {
        let mut body = String::from("running 3 tests\nerror: mismatch\nFAILED tests::it_breaks\n");
        body.push_str(&"ok noise\n".repeat(400));
        body
    }

    fn receipt(source: &str, quote: &str, status: &str) -> String {
        let hash = sha256_hex(source.as_bytes());
        serde_json::json!({
            "schema": REDUCER_RECEIPT_SCHEMA,
            "source_sha256": hash,
            "status": status,
            "uncertain": false,
            "evidence": [{"kind": "failure", "quote": quote}]
        })
        .to_string()
    }

    #[test]
    fn exact_quotes_are_accepted() {
        let source = source_log();
        let json = receipt(&source, "error: mismatch", "failure");
        let config = EvidenceReducerConfig {
            enabled: true,
            min_bytes: 16,
        };
        let reduced = apply_quote_checked_reduction(&config, &source, true, &json);
        assert!(reduced.starts_with(EVIDENCE_RECEIPT_PREFIX), "{reduced}");
        assert!(reduced.contains("error: mismatch"), "{reduced}");
        assert!(reduced.len() < source.len());
    }

    #[test]
    fn mutated_quote_keeps_original() {
        let source = source_log();
        let json = receipt(&source, "error: invented", "failure");
        let config = EvidenceReducerConfig {
            enabled: true,
            min_bytes: 16,
        };
        let reduced = apply_quote_checked_reduction(&config, &source, true, &json);
        assert_eq!(reduced, source);
        assert_eq!(
            validate_receipt(&json, &source, &sha256_hex(source.as_bytes()), true),
            ReceiptValidation::Rejected("unverifiable-quote")
        );
    }

    #[test]
    fn failing_log_without_failure_evidence_is_rejected() {
        let source = source_log();
        let hash = sha256_hex(source.as_bytes());
        let json = serde_json::json!({
            "schema": REDUCER_RECEIPT_SCHEMA,
            "source_sha256": hash,
            "status": "failure",
            "uncertain": false,
            "evidence": [{"kind": "summary", "quote": "running 3 tests"}]
        })
        .to_string();
        assert_eq!(
            validate_receipt(&json, &source, &hash, true),
            ReceiptValidation::Rejected("missing-failure-evidence")
        );
        let config = EvidenceReducerConfig {
            enabled: true,
            min_bytes: 16,
        };
        assert_eq!(
            apply_quote_checked_reduction(&config, &source, true, &json),
            source
        );
    }

    #[test]
    fn receipt_not_smaller_is_rejected() {
        let source = "error: x\nFAILED\n";
        let json = receipt(source, "error: x", "failure");
        let config = EvidenceReducerConfig {
            enabled: true,
            min_bytes: 1,
        };
        let result = reduce_or_reject(&config, source, true, &json);
        assert_eq!(result, Err("receipt-not-smaller"));
        assert_eq!(
            apply_quote_checked_reduction(&config, source, true, &json),
            source
        );
    }

    #[test]
    fn reducer_model_error_keeps_original() {
        let source = source_log();
        let config = EvidenceReducerConfig {
            enabled: true,
            min_bytes: 16,
        };
        assert_eq!(
            reduce_from_reducer_result(&config, &source, true, Err("model-call-exception")),
            source
        );
    }

    #[test]
    fn disabled_path_leaves_condenser_output_unchanged() {
        let source = source_log();
        let json = receipt(&source, "error: mismatch", "failure");
        let config = EvidenceReducerConfig::default();
        assert!(!config.enabled);
        assert_eq!(
            apply_quote_checked_reduction(&config, &source, true, &json),
            source
        );
    }

    #[test]
    fn after_condense_fail_open_without_receipt_or_when_disabled() {
        let condensed = source_log();
        let json = receipt(&condensed, "error: mismatch", "failure");
        let disabled = EvidenceReducerConfig::default();
        assert_eq!(
            after_condense(&condensed, true, &disabled, Ok(json.clone())),
            condensed
        );
        let enabled = EvidenceReducerConfig {
            enabled: true,
            min_bytes: 16,
        };
        assert_eq!(
            after_condense(&condensed, true, &enabled, Err("no-nested-model")),
            condensed
        );
        let reduced = after_condense(&condensed, true, &enabled, Ok(json));
        assert!(reduced.starts_with(EVIDENCE_RECEIPT_PREFIX), "{reduced}");
        assert!(reduced.contains("error: mismatch"), "{reduced}");
        assert!(reduced.len() < condensed.len());
    }
}
