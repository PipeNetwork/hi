//! x402 payment protocol for Pipenetwork Chat Completions.
//!
//! Types and codecs mirror the server (`ipop` `api-error::x402`). Solana
//! signing lives in `hi-x402`; this crate only parses quotes, encodes
//! `payment-signature`, and validates amount/network/mint before a settler
//! is invoked.

use std::sync::Mutex;

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64_STANDARD};
use serde::{Deserialize, Serialize};

pub const X402_PAYMENT_REQUIRED_HEADER: &str = "payment-required";
pub const X402_PAYMENT_SIGNATURE_HEADER: &str = "payment-signature";
pub const X402_PAYMENT_RESPONSE_HEADER: &str = "payment-response";

pub const X402_VERSION: i32 = 2;
pub const X402_SCHEME_EXACT: &str = "exact";
pub const X402_SOLANA_MAINNET: &str = "solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp";
pub const X402_USDC_MINT_MAINNET: &str = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v";
pub const X402_CREDIT_TOKEN_PREFIX: &str = "x402_";
pub const X402_MEMO_PREFIX: &str = "x402_";
pub const X402_USDC_DECIMALS: u32 = 6;
pub const X402_MIN_TOPUP_MINOR: i64 = 10_000;
pub const X402_USDC_MINOR_PER_UNIT: i64 = 1_000_000;
pub const X402_DEFAULT_MAX_USD: f64 = 1.0;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct X402PaymentRequired {
    pub x402_version: i32,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub error: String,
    pub resource: X402Resource,
    pub accepts: Vec<X402PaymentRequirements>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct X402Resource {
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct X402PaymentRequirements {
    pub scheme: String,
    pub network: String,
    pub amount: String,
    pub asset: String,
    pub pay_to: String,
    pub max_timeout_seconds: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct X402PaymentPayload {
    pub x402_version: i32,
    pub accepted: X402PaymentRequirements,
    pub payload: X402SettlePayload,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum X402SettlePayload {
    Signature { signature: String },
    Transaction { transaction: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct X402PaymentResponse {
    pub success: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transaction: Option<String>,
    pub network: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payer: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra: Option<serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct X402QuoteSummary {
    pub usd: f64,
    pub amount_minor: i64,
    pub pay_to: String,
    pub mint: String,
    pub memo: String,
    pub quote_id: String,
    pub network: String,
    pub max_timeout_seconds: u64,
}

impl X402QuoteSummary {
    pub fn prompt_text(&self) -> String {
        format!(
            "Pay ${:.6} USDC ({} minor units) to {} on {}.\nMint: {}\nMemo: {}\nQuote: {}",
            self.usd,
            self.amount_minor,
            self.pay_to,
            self.network,
            self.mint,
            self.memo,
            self.quote_id
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum X402CodecError {
    Json(String),
    Base64(String),
}

impl std::fmt::Display for X402CodecError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Json(message) | Self::Base64(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for X402CodecError {}

fn encode_json_header<T: Serialize>(value: &T) -> Result<String, X402CodecError> {
    let json =
        serde_json::to_vec(value).map_err(|error| X402CodecError::Json(error.to_string()))?;
    Ok(BASE64_STANDARD.encode(json))
}

fn decode_json_header<T: for<'de> Deserialize<'de>>(header: &str) -> Result<T, X402CodecError> {
    let trimmed = header.trim();
    let bytes = BASE64_STANDARD
        .decode(trimmed)
        .map_err(|error| X402CodecError::Base64(error.to_string()))?;
    serde_json::from_slice(&bytes).map_err(|error| X402CodecError::Json(error.to_string()))
}

pub fn encode_payment_required_header(
    value: &X402PaymentRequired,
) -> Result<String, X402CodecError> {
    encode_json_header(value)
}

pub fn decode_payment_required_header(header: &str) -> Result<X402PaymentRequired, X402CodecError> {
    decode_json_header(header)
}

pub fn encode_payment_payload_header(value: &X402PaymentPayload) -> Result<String, X402CodecError> {
    encode_json_header(value)
}

pub fn decode_payment_payload_header(header: &str) -> Result<X402PaymentPayload, X402CodecError> {
    decode_json_header(header)
}

pub fn encode_payment_response_header(
    value: &X402PaymentResponse,
) -> Result<String, X402CodecError> {
    encode_json_header(value)
}

pub fn decode_payment_response_header(header: &str) -> Result<X402PaymentResponse, X402CodecError> {
    decode_json_header(header)
}

impl X402PaymentRequirements {
    pub fn quote_id(&self) -> Option<&str> {
        self.extra
            .as_ref()
            .and_then(|extra| extra.get("quoteId"))
            .and_then(|value| value.as_str())
    }

    pub fn memo(&self) -> Option<&str> {
        self.extra
            .as_ref()
            .and_then(|extra| extra.get("memo"))
            .and_then(|value| value.as_str())
    }

    pub fn estimated_usd(&self) -> Option<f64> {
        self.extra
            .as_ref()
            .and_then(|extra| extra.get("estimatedUsd"))
            .and_then(|value| match value {
                serde_json::Value::String(text) => text.parse().ok(),
                serde_json::Value::Number(number) => number.as_f64(),
                _ => None,
            })
    }

    pub fn amount_minor(&self) -> Option<i64> {
        self.amount.parse().ok()
    }
}

impl X402PaymentResponse {
    pub fn credit_token(&self) -> Option<&str> {
        self.extra
            .as_ref()
            .and_then(|extra| extra.get("credit_token"))
            .and_then(|value| value.as_str())
            .filter(|token| token.starts_with(X402_CREDIT_TOKEN_PREFIX))
    }
}

impl X402PaymentRequired {
    pub fn first_accept(&self) -> Option<&X402PaymentRequirements> {
        self.accepts.first()
    }
}

/// Parse a 402 body or `payment-required` header into a quote.
pub fn parse_payment_required(
    header: Option<&str>,
    body: &str,
) -> Result<X402PaymentRequired, X402CodecError> {
    if let Some(header) = header.filter(|value| !value.trim().is_empty()) {
        return decode_payment_required_header(header);
    }
    serde_json::from_str(body).map_err(|error| X402CodecError::Json(error.to_string()))
}

pub fn quote_summary(requirements: &X402PaymentRequirements) -> Result<X402QuoteSummary> {
    let amount_minor = requirements
        .amount_minor()
        .context("x402 quote amount is not an integer")?;
    let memo = requirements
        .memo()
        .or_else(|| requirements.quote_id().map(|_| ""))
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    let quote_id = requirements
        .quote_id()
        .map(str::to_string)
        .or_else(|| {
            requirements.memo().and_then(|memo| {
                memo.strip_prefix(X402_MEMO_PREFIX)
                    .filter(|id| !id.is_empty())
                    .map(str::to_string)
            })
        })
        .context("x402 quote is missing extra.quoteId / extra.memo")?;
    let memo = memo.unwrap_or_else(|| format!("{X402_MEMO_PREFIX}{quote_id}"));
    let usd = requirements
        .estimated_usd()
        .unwrap_or(amount_minor as f64 / X402_USDC_MINOR_PER_UNIT as f64);
    Ok(X402QuoteSummary {
        usd,
        amount_minor,
        pay_to: requirements.pay_to.clone(),
        mint: requirements.asset.clone(),
        memo,
        quote_id,
        network: requirements.network.clone(),
        max_timeout_seconds: requirements.max_timeout_seconds,
    })
}

/// Fail closed before any chain send when the quote is the wrong network/mint,
/// dust, missing memo, or over the local USD cap.
pub fn validate_quote(
    requirements: &X402PaymentRequirements,
    max_usd: f64,
) -> Result<X402QuoteSummary> {
    if requirements.scheme != X402_SCHEME_EXACT {
        bail!(
            "x402 quote scheme {:?} is not supported (need {X402_SCHEME_EXACT})",
            requirements.scheme
        );
    }
    if requirements.network != X402_SOLANA_MAINNET {
        bail!(
            "x402 quote network {:?} is not Solana mainnet ({X402_SOLANA_MAINNET})",
            requirements.network
        );
    }
    if requirements.asset != X402_USDC_MINT_MAINNET {
        bail!(
            "x402 quote mint {:?} is not USDC ({X402_USDC_MINT_MAINNET})",
            requirements.asset
        );
    }
    let summary = quote_summary(requirements)?;
    if summary.amount_minor < X402_MIN_TOPUP_MINOR {
        bail!(
            "x402 quote amount {} is below the ${:.2} minimum",
            summary.amount_minor,
            X402_MIN_TOPUP_MINOR as f64 / X402_USDC_MINOR_PER_UNIT as f64
        );
    }
    let cap = if max_usd.is_finite() && max_usd > 0.0 {
        max_usd
    } else {
        X402_DEFAULT_MAX_USD
    };
    let amount_usd = summary.amount_minor as f64 / X402_USDC_MINOR_PER_UNIT as f64;
    if summary.usd > cap + f64::EPSILON || amount_usd > cap + f64::EPSILON {
        bail!(
            "x402 quote ${:.6} ({} minor units) exceeds HI_X402_MAX_USD ${:.2} — shrink the turn or raise the cap",
            summary.usd.max(amount_usd),
            summary.amount_minor,
            cap
        );
    }
    Ok(summary)
}

pub fn signature_payload(
    accepted: X402PaymentRequirements,
    signature: impl Into<String>,
) -> X402PaymentPayload {
    X402PaymentPayload {
        x402_version: X402_VERSION,
        accepted,
        payload: X402SettlePayload::Signature {
            signature: signature.into(),
        },
    }
}

pub fn is_unconfirmed_conflict(status: u16, body: &str) -> bool {
    status == 409 && body.to_ascii_lowercase().contains("not confirmed yet")
}

pub fn credit_token_from_header(header: Option<&str>) -> Option<String> {
    let header = header?.trim();
    if header.is_empty() {
        return None;
    }
    decode_payment_response_header(header)
        .ok()?
        .credit_token()
        .map(str::to_string)
}

pub fn credit_token_from_json(body: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(body).ok()?;
    value
        .pointer("/ipop/payment/credit_token")
        .and_then(|token| token.as_str())
        .or_else(|| value.get("credit_token").and_then(|token| token.as_str()))
        .filter(|token| token.starts_with(X402_CREDIT_TOKEN_PREFIX))
        .map(str::to_string)
}

/// Pays a validated quote and returns the broadcast Solana signature.
#[async_trait]
pub trait X402Settler: Send + Sync {
    async fn settle(&self, requirements: &X402PaymentRequirements) -> Result<String>;
}

/// Interactive confirm / paste-signature frontend.
#[async_trait]
pub trait X402Confirmer: Send + Sync {
    async fn confirm(&self, quote: &X402QuoteSummary) -> Result<bool>;

    async fn prompt_signature(&self) -> Result<String> {
        bail!("no signature paste frontend is configured; set HI_X402_KEYPAIR")
    }
}

pub struct AutoX402Confirmer;

#[async_trait]
impl X402Confirmer for AutoX402Confirmer {
    async fn confirm(&self, _quote: &X402QuoteSummary) -> Result<bool> {
        Ok(true)
    }
}

/// Pending TUI/REPL prompt issued by a settler while `OpenAiProvider` is blocked.
pub enum X402UserPrompt {
    Confirm {
        quote: X402QuoteSummary,
        reply: tokio::sync::oneshot::Sender<bool>,
    },
    PasteSignature {
        reply: tokio::sync::oneshot::Sender<Option<String>>,
    },
}

pub struct X402ConfirmRequest {
    pub prompt: X402UserPrompt,
}

/// Shared queue so the TUI event loop can show a confirm overlay while the
/// provider hop waits on-chain.
#[derive(Default)]
pub struct X402ConfirmBroker {
    pending: Mutex<Option<X402UserPrompt>>,
}

impl X402ConfirmBroker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn take(&self) -> Option<X402UserPrompt> {
        self.pending.lock().ok()?.take()
    }

    fn submit(&self, prompt: X402UserPrompt) {
        if let Ok(mut pending) = self.pending.lock() {
            *pending = Some(prompt);
        }
    }
}

#[async_trait]
impl X402Confirmer for X402ConfirmBroker {
    async fn confirm(&self, quote: &X402QuoteSummary) -> Result<bool> {
        let (reply, rx) = tokio::sync::oneshot::channel();
        self.submit(X402UserPrompt::Confirm {
            quote: quote.clone(),
            reply,
        });
        Ok(rx.await.unwrap_or(false))
    }

    async fn prompt_signature(&self) -> Result<String> {
        let (reply, rx) = tokio::sync::oneshot::channel();
        self.submit(X402UserPrompt::PasteSignature { reply });
        rx.await
            .ok()
            .flatten()
            .filter(|value| !value.trim().is_empty())
            .context("x402 signature paste cancelled")
    }
}

#[async_trait]
impl X402Confirmer for std::sync::Arc<X402ConfirmBroker> {
    async fn confirm(&self, quote: &X402QuoteSummary) -> Result<bool> {
        (**self).confirm(quote).await
    }

    async fn prompt_signature(&self) -> Result<String> {
        (**self).prompt_signature().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_required() -> X402PaymentRequired {
        X402PaymentRequired {
            x402_version: X402_VERSION,
            error: "PAYMENT-SIGNATURE header is required".to_string(),
            resource: X402Resource {
                url: "https://api.pipenetwork.ai/v1/chat/completions".to_string(),
                description: Some("Chat completion".to_string()),
                mime_type: Some("application/json".to_string()),
            },
            accepts: vec![X402PaymentRequirements {
                scheme: X402_SCHEME_EXACT.to_string(),
                network: X402_SOLANA_MAINNET.to_string(),
                amount: "20000".to_string(),
                asset: X402_USDC_MINT_MAINNET.to_string(),
                pay_to: "Treasury111111111111111111111111111111111".to_string(),
                max_timeout_seconds: 180,
                extra: Some(serde_json::json!({
                    "memo": "x402_abc",
                    "quoteId": "abc",
                    "estimatedUsd": "0.02",
                    "maxUsd": "0.02"
                })),
            }],
        }
    }

    #[test]
    fn payment_required_header_round_trips() {
        let required = sample_required();
        let encoded = encode_payment_required_header(&required).unwrap();
        let decoded = decode_payment_required_header(&encoded).unwrap();
        assert_eq!(decoded, required);
        assert_eq!(decoded.accepts[0].quote_id(), Some("abc"));
    }

    #[test]
    fn validate_quote_accepts_mainnet_usdc_under_cap() {
        let summary = validate_quote(&sample_required().accepts[0], 1.0).unwrap();
        assert_eq!(summary.amount_minor, 20_000);
        assert_eq!(summary.memo, "x402_abc");
    }

    #[test]
    fn validate_quote_rejects_over_cap() {
        let mut requirements = sample_required().accepts.remove(0);
        requirements.amount = "2000000".into();
        requirements.extra = Some(serde_json::json!({
            "memo": "x402_abc",
            "quoteId": "abc",
            "estimatedUsd": "2.00"
        }));
        let error = validate_quote(&requirements, 1.0).unwrap_err().to_string();
        assert!(error.contains("HI_X402_MAX_USD"), "{error}");
    }

    #[test]
    fn validate_quote_rejects_missing_memo_and_quote_id() {
        let mut requirements = sample_required().accepts.remove(0);
        requirements.extra = Some(serde_json::json!({ "estimatedUsd": "0.02" }));
        let error = validate_quote(&requirements, 1.0).unwrap_err().to_string();
        assert!(
            error.contains("quoteId") || error.contains("memo"),
            "{error}"
        );
    }

    #[test]
    fn credit_token_is_read_from_payment_response_extra() {
        let encoded = encode_payment_response_header(&X402PaymentResponse {
            success: true,
            transaction: Some("sig".into()),
            network: X402_SOLANA_MAINNET.into(),
            payer: Some("payer".into()),
            extra: Some(serde_json::json!({
                "credit_token": "x402_live_test",
                "credit_remaining_usd": "0.98"
            })),
        })
        .unwrap();
        assert_eq!(
            credit_token_from_header(Some(&encoded)).as_deref(),
            Some("x402_live_test")
        );
    }

    #[test]
    fn credit_token_is_read_from_json_body_fallback() {
        let body = serde_json::json!({
            "ipop": { "payment": { "credit_token": "x402_from_body" } }
        });
        assert_eq!(
            credit_token_from_json(&body.to_string()).as_deref(),
            Some("x402_from_body")
        );
    }
}
