//! A [`Provider`] that tries a chain of backends in order, moving to the next
//! when one errors or returns nothing, carrying usage from failed attempts into
//! the final result.

use anyhow::Result;
use async_trait::async_trait;

use crate::circuit_breaker::{BreakerConfig, CircuitBreaker, Outcome};
use crate::provider::{
    Provider, ProviderError, ServedModel, provider_error_affects_health,
    provider_error_is_fallback_eligible, provider_error_kind, provider_error_usage,
};
use crate::types::{ChatRequest, Completion, StreamEvent, Usage};

/// One link in the fallback chain: a built provider plus the model id to request
/// from it (each backend names its models differently).
pub struct Backend {
    pub provider: Box<dyn Provider>,
    pub model: String,
    /// A short human label for status messages, e.g. "ollama/qwen2.5:7b".
    pub label: String,
}

/// Tries each [`Backend`] in turn. A backend "fails" if it returns an error or a
/// completion with no content (no text and no tool calls) — the symptom of an
/// overloaded model. The first backend that produces real output wins; if all
/// fail, the last result (error or empty) is returned so the caller still sees a
/// definitive outcome.
///
/// Each backend has an associated [`CircuitBreaker`] — after repeated failures
/// the breaker opens and the backend is skipped without waiting for a timeout,
/// reducing wasted API calls during provider outages.
pub struct FallbackProvider {
    chain: Vec<Backend>,
    breakers: Vec<CircuitBreaker>,
}

impl FallbackProvider {
    /// Build from a non-empty chain. With a single backend it's a thin pass-through.
    pub fn new(chain: Vec<Backend>) -> Result<Self> {
        Self::with_config(chain, BreakerConfig::client())
    }

    /// Build with a custom circuit breaker config for each backend.
    pub fn with_config(chain: Vec<Backend>, breaker_config: BreakerConfig) -> Result<Self> {
        anyhow::ensure!(!chain.is_empty(), "fallback chain must not be empty");
        let breakers = chain
            .iter()
            .map(|_| CircuitBreaker::new(breaker_config.clone()))
            .collect();
        Ok(Self { chain, breakers })
    }
}

#[async_trait]
impl Provider for FallbackProvider {
    async fn stream(
        &self,
        request: ChatRequest,
        sink: &mut (dyn FnMut(StreamEvent) + Send),
    ) -> Result<Completion> {
        let last = self.chain.len().saturating_sub(1);
        let mut prior_usage = Usage::default();
        for (i, backend) in self.chain.iter().enumerate() {
            let is_last = i == last;

            // Check the circuit breaker — skip this backend if it's open.
            if let Err(reason) = self.breakers[i].check() {
                if is_last {
                    return Err(anyhow::anyhow!("{reason}"));
                }
                let next = &self.chain[i + 1];
                sink(StreamEvent::Status(format!(
                    "{} circuit breaker open — skipping to {}",
                    backend.label, next.label
                )));
                continue;
            }

            let mut req = request.clone();
            req.model = backend.model.clone();

            match backend.provider.stream(req, sink).await {
                Ok(mut completion) if !completion.content.is_empty() || is_last => {
                    self.breakers[i].record(Outcome::Success);
                    if !prior_usage.is_zero() {
                        // Fold the failed/empty earlier attempts' token counts
                        // into the winner. `Usage::add` sums only the token counts
                        // + `estimated` and leaves `context_occupancy` /
                        // `input_includes_cache` untouched (so the winner's stay),
                        // but it would let a stale prior rate-limit snapshot
                        // overwrite the winner's — so preserve the winner's
                        // occupancy/cache/rate-limit scalars explicitly (these
                        // drive the context gauge; taking them from a zeroed prior
                        // attempt would mis-count cache tokens and trip early
                        // auto-compaction).
                        let winner_context = completion.usage.context_occupancy;
                        let winner_includes_cache = completion.usage.input_includes_cache;
                        let winner_rate_limits = completion.usage.rate_limits;
                        let prior_rate_limits = prior_usage.rate_limits;
                        completion.usage.add(prior_usage);
                        completion.usage.context_occupancy = winner_context;
                        completion.usage.input_includes_cache = winner_includes_cache;
                        completion.usage.rate_limits = winner_rate_limits.or(prior_rate_limits);
                    }
                    return Ok(completion);
                }
                Ok(empty) => {
                    // Empty/model-output failures may justify trying another
                    // backend, but they do not mean this backend is unhealthy.
                    self.breakers[i].record(Outcome::Success);
                    prior_usage.add(empty.usage);
                    let next = &self.chain[i + 1];
                    sink(StreamEvent::Status(format!(
                        "{} returned nothing — falling back to {}",
                        backend.label, next.label
                    )));
                }
                Err(err) if is_last => {
                    self.breakers[i].record(if provider_error_affects_health(&err) {
                        Outcome::Failure
                    } else {
                        Outcome::Success
                    });
                    prior_usage.add(provider_error_usage(&err));
                    if prior_usage.is_zero() {
                        return Err(err);
                    }
                    return Err(error_with_usage(&err, prior_usage).into());
                }
                Err(err) => {
                    self.breakers[i].record(if provider_error_affects_health(&err) {
                        Outcome::Failure
                    } else {
                        Outcome::Success
                    });
                    prior_usage.add(provider_error_usage(&err));
                    if !provider_error_is_fallback_eligible(&err) {
                        return Err(error_with_usage(&err, prior_usage).into());
                    }
                    let next = &self.chain[i + 1];
                    sink(StreamEvent::Status(format!(
                        "{} failed ({err}) — falling back to {}",
                        backend.label, next.label
                    )));
                }
            }
        }
        // The loop always returns on the last backend; reaching here means
        // the chain was empty, which `new` prevents — but fail gracefully
        // rather than panicking so a future caller can't wedge the turn.
        anyhow::bail!("fallback chain exhausted without producing a result")
    }

    async fn list_models(&self) -> Result<Vec<ServedModel>> {
        match self.chain.first() {
            Some(backend) => backend.provider.list_models().await,
            None => Ok(Vec::new()),
        }
    }
}

fn error_with_usage(err: &anyhow::Error, usage: Usage) -> ProviderError {
    err.downcast_ref::<ProviderError>()
        .cloned()
        .unwrap_or_else(|| {
            ProviderError::new(
                provider_error_kind(err).unwrap_or(crate::provider::ProviderErrorKind::Other),
                err.to_string(),
            )
        })
        .with_usage(usage)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::ProviderErrorKind;
    use crate::types::{Completion, Content, Usage};
    use std::sync::Mutex;

    /// Returns one canned completion per call, in order.
    struct Canned(Mutex<Vec<Result<Completion>>>);

    #[async_trait]
    impl Provider for Canned {
        async fn stream(
            &self,
            _req: ChatRequest,
            _sink: &mut (dyn FnMut(StreamEvent) + Send),
        ) -> Result<Completion> {
            self.0.lock().unwrap().remove(0)
        }
    }

    fn empty() -> Completion {
        Completion::default()
    }

    fn empty_with_usage(input: u64, output: u64) -> Completion {
        Completion {
            content: Vec::new(),
            usage: Usage {
                input_tokens: input,
                output_tokens: output,
                context_occupancy: input,
                ..Usage::default()
            },
            stop_reason: None,
        }
    }

    fn text_with_usage(s: &str, input: u64, output: u64) -> Completion {
        Completion {
            content: vec![Content::Text(s.into())],
            usage: Usage {
                input_tokens: input,
                output_tokens: output,
                context_occupancy: input,
                ..Usage::default()
            },
            stop_reason: None,
        }
    }

    fn text(s: &str) -> Completion {
        text_with_usage(s, 0, 0)
    }

    fn first_text(c: &Completion) -> &str {
        match c.content.first() {
            Some(Content::Text(t)) => t,
            _ => "",
        }
    }

    fn backend(label: &str, results: Vec<Result<Completion>>) -> Backend {
        Backend {
            provider: Box::new(Canned(Mutex::new(results))),
            model: "m".into(),
            label: label.into(),
        }
    }

    fn req() -> ChatRequest {
        ChatRequest {
            model: "primary".into(),
            request_id: None,
            retry_attempt: 0,
            user_turn: false,
            canonical_objective: None,
            messages: vec![].into(),
            tools: vec![].into(),
            max_tokens: 16,
            temperature: None,
            top_p: None,
            frequency_penalty: None,
            thinking_budget: None,
            reasoning_effort: None,
            profile: Default::default(),
        }
    }

    #[tokio::test]
    async fn falls_back_past_empty_and_errored_backends() {
        let mut statuses = Vec::new();
        let mut sink = |e: StreamEvent| {
            if let StreamEvent::Status(s) = e {
                statuses.push(s);
            }
        };
        let fp = FallbackProvider::new(vec![
            backend("primary", vec![Ok(empty())]), // returns nothing
            backend(
                "mid",
                vec![Err(
                    ProviderError::new(ProviderErrorKind::Outage, "503").into()
                )],
            ), // eligible provider failure
            backend("local", vec![Ok(text("hello from local"))]), // wins
        ])
        .unwrap();
        let out = fp.stream(req(), &mut sink).await.unwrap();
        assert_eq!(out.content.len(), 1);
        assert_eq!(first_text(&out), "hello from local");
        // Two fallbacks announced.
        assert_eq!(statuses.len(), 2, "statuses: {statuses:?}");
        assert!(statuses[0].contains("falling back to mid"));
        assert!(statuses[1].contains("falling back to local"));
    }

    #[tokio::test]
    async fn fallback_preserves_usage_from_prior_attempts() {
        let mut sink = |_e: StreamEvent| {};
        let fp = FallbackProvider::new(vec![
            backend("primary", vec![Ok(empty_with_usage(10, 2))]),
            backend("local", vec![Ok(text_with_usage("winner", 3, 4))]),
        ])
        .unwrap();
        let out = fp.stream(req(), &mut sink).await.unwrap();
        assert_eq!(first_text(&out), "winner");
        assert_eq!(out.usage.input_tokens, 13);
        assert_eq!(out.usage.output_tokens, 6);
    }

    #[tokio::test]
    async fn first_healthy_backend_wins_without_fallback() {
        let mut sink = |_e: StreamEvent| {};
        let fp = FallbackProvider::new(vec![
            backend("primary", vec![Ok(text("direct"))]),
            backend("local", vec![Ok(text("unused"))]),
        ])
        .unwrap();
        let out = fp.stream(req(), &mut sink).await.unwrap();
        assert_eq!(first_text(&out), "direct");
    }

    #[tokio::test]
    async fn explicit_non_retryable_error_does_not_fallback() {
        let mut statuses = Vec::new();
        let mut sink = |event: StreamEvent| {
            if let StreamEvent::Status(status) = event {
                statuses.push(status);
            }
        };
        let rejected = ProviderError::new(ProviderErrorKind::Outage, "payload rejected")
            .with_api_contract(Some("service_unavailable".to_string()), Some(false), None);
        let fp = FallbackProvider::new(vec![
            backend("primary", vec![Err(rejected.into())]),
            backend("local", vec![Ok(text("must not run"))]),
        ])
        .unwrap();
        let err = fp.stream(req(), &mut sink).await.unwrap_err();
        assert_eq!(provider_error_kind(&err), Some(ProviderErrorKind::Outage));
        assert!(statuses.is_empty());
    }

    #[tokio::test]
    async fn returns_last_result_when_all_fail() {
        let mut sink = |_e: StreamEvent| {};
        let fp = FallbackProvider::new(vec![
            backend("primary", vec![Ok(empty())]),
            backend("local", vec![Ok(empty())]),
        ])
        .unwrap();
        // All empty → the last (empty) completion is returned, not an error.
        let out = fp.stream(req(), &mut sink).await.unwrap();
        assert!(out.content.is_empty());
    }
}
