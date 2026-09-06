use std::sync::atomic::Ordering;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use hi_ai::{
    CapabilityRoute, ChatRequest, Completion, Provider, ProviderCapabilities,
    ProviderCapabilityCandidate, ProviderRequestContext, ServedModel, StreamEvent,
    endpoint_capability_route,
};

use super::RsiRemoteProvider;

#[cfg(test)]
mod tests;

#[derive(Clone, Debug, Eq, PartialEq)]
enum SealedRsiRoute {
    Remote(String),
    Inner,
}

impl RsiRemoteProvider {
    fn remote_capability_candidate(&self) -> ProviderCapabilityCandidate {
        ProviderCapabilityCandidate::new(
            CapabilityRoute::new(
                endpoint_capability_route("rsi_remote", &self.settings.base_url),
                format!("server-selected/{}", self.settings.channel()),
            ),
            ProviderCapabilities::default(),
        )
    }

    fn sealed_route(&self, request: &ChatRequest) -> Result<Option<SealedRsiRoute>> {
        let Some(envelope) = request.tool_envelope.as_deref() else {
            return Ok(None);
        };
        let remote_route = self.remote_capability_candidate().target.route;
        let members = envelope
            .payload
            .pointer("/provider/capability_record/members")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| anyhow!("sealed provider capability members are missing"))?;
        for member in members {
            let target = member
                .pointer("/target")
                .ok_or_else(|| anyhow!("sealed provider capability target is missing"))?;
            if target.get("route").and_then(serde_json::Value::as_str)
                == Some(remote_route.as_str())
            {
                let model = target
                    .get("model")
                    .and_then(serde_json::Value::as_str)
                    .ok_or_else(|| anyhow!("sealed RSI route model is missing"))?;
                let channel = model
                    .strip_prefix("server-selected/")
                    .ok_or_else(|| anyhow!("sealed RSI route model is invalid"))?;
                if matches!(channel, "stable" | "beta") {
                    return Ok(Some(SealedRsiRoute::Remote(channel.to_string())));
                }
                return Err(anyhow!("sealed RSI channel is invalid"));
            }
        }
        Ok(Some(SealedRsiRoute::Inner))
    }
}

#[async_trait]
impl Provider for RsiRemoteProvider {
    fn capabilities(&self) -> ProviderCapabilities {
        let inner = self.inner.capabilities();
        if self.enabled.load(Ordering::SeqCst) {
            // The general declaration spans remote primary and inner auxiliary
            // calls. Request-aware sealing below selects the exact path.
            inner.conservative_intersection(&ProviderCapabilities::default())
        } else {
            inner
        }
    }

    fn capability_candidates(&self, route: &str, model: &str) -> Vec<ProviderCapabilityCandidate> {
        let mut candidates = self.inner.capability_candidates(route, model);
        if self.enabled.load(Ordering::SeqCst) {
            candidates.push(self.remote_capability_candidate());
        }
        candidates
    }

    fn capability_candidates_for_request(
        &self,
        route: &str,
        model: &str,
        context: ProviderRequestContext<'_>,
    ) -> Vec<ProviderCapabilityCandidate> {
        if self.enabled.load(Ordering::SeqCst) && context.user_turn {
            vec![self.remote_capability_candidate()]
        } else {
            self.inner
                .capability_candidates_for_request(route, model, context)
        }
    }

    async fn stream(
        &self,
        request: ChatRequest,
        sink: &mut (dyn FnMut(StreamEvent) + Send),
    ) -> Result<Completion> {
        // Auxiliary requests (compaction, memory, planning, or finalization) stay on their
        // normal route. The primary user turn is remote even in chat-only tool mode.
        if !request.user_turn {
            return self.inner.stream(request, sink).await;
        }
        let sealed_route = self.sealed_route(&request)?.unwrap_or_else(|| {
            if self.enabled.load(Ordering::SeqCst) {
                SealedRsiRoute::Remote(self.settings.channel().to_string())
            } else {
                SealedRsiRoute::Inner
            }
        });
        match sealed_route {
            SealedRsiRoute::Inner => self.inner.stream(request, sink).await,
            SealedRsiRoute::Remote(channel) => self.remote_stream(request, sink, &channel).await,
        }
    }

    async fn list_models(&self) -> Result<Vec<ServedModel>> {
        self.inner.list_models().await
    }
}
