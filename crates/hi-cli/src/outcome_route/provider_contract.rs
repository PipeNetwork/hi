use anyhow::{Result, anyhow};
use async_trait::async_trait;
use hi_agent::TaskContract;
use hi_ai::{
    CapabilityRoute, ChatRequest, Completion, Provider, ProviderCapabilities,
    ProviderCapabilityCandidate, ProviderRequestContext, ServedModel, StreamEvent,
    endpoint_capability_route,
};

use super::{OutcomeRouteProvider, fail_open_warning, should_submit_outcome};

#[cfg(test)]
mod tests;

impl OutcomeRouteProvider {
    fn remote_capability_candidate(&self) -> ProviderCapabilityCandidate {
        ProviderCapabilityCandidate::new(
            CapabilityRoute::new(
                endpoint_capability_route("outcome_task", self.client.origin()),
                self.offer.as_route(),
            ),
            ProviderCapabilities::default(),
        )
    }

    fn request_may_use_outcome(&self, context: ProviderRequestContext<'_>) -> bool {
        if !context.user_turn {
            return false;
        }
        let Some(objective) = context.canonical_objective else {
            // Unknown primary requests must include the remote route rather
            // than inherit capabilities that route cannot honor.
            return true;
        };
        let contract = TaskContract::derive(objective, self.quality.verification.clone());
        should_submit_outcome(self.mode, true, self.has_cargo(), &contract)
    }

    fn sealed_outcome_intent(&self, request: &ChatRequest) -> Result<Option<bool>> {
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
            let route = member
                .pointer("/target/route")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| anyhow!("sealed provider capability route is missing"))?;
            if route == remote_route {
                return Ok(Some(true));
            }
        }
        Ok(Some(false))
    }
}

#[async_trait]
impl Provider for OutcomeRouteProvider {
    fn capabilities(&self) -> ProviderCapabilities {
        // The general declaration covers both task execution and inner
        // fail-open/passthrough. Request-aware sealing selects the exact set.
        self.inner
            .capabilities()
            .conservative_intersection(&ProviderCapabilities::default())
    }

    fn capability_candidates(&self, route: &str, model: &str) -> Vec<ProviderCapabilityCandidate> {
        let mut candidates = self.inner.capability_candidates(route, model);
        candidates.push(self.remote_capability_candidate());
        candidates
    }

    fn capability_candidates_for_request(
        &self,
        route: &str,
        model: &str,
        context: ProviderRequestContext<'_>,
    ) -> Vec<ProviderCapabilityCandidate> {
        let mut candidates = self
            .inner
            .capability_candidates_for_request(route, model, context);
        if self.request_may_use_outcome(context) {
            // Task admission/runtime failures fail open to `inner`, so both
            // backends can receive this exact request and must be sealed.
            candidates.push(self.remote_capability_candidate());
        }
        candidates
    }

    async fn stream(
        &self,
        request: ChatRequest,
        sink: &mut (dyn FnMut(StreamEvent) + Send),
    ) -> Result<Completion> {
        if !request.user_turn {
            return self.inner.stream(request, sink).await;
        }
        if self.sealed_outcome_intent(&request)? == Some(false) {
            return self.inner.stream(request, sink).await;
        }
        match self.submit_turn(request.clone(), sink).await {
            Ok(completion) => Ok(completion),
            Err(error) if error.is_fail_open() => {
                if error.message != "turn is not an Outcome code.change" {
                    sink(StreamEvent::Warning(fail_open_warning(&error)));
                }
                self.inner.stream(request, sink).await
            }
            Err(error) => Err(anyhow::anyhow!(error)),
        }
    }

    async fn list_models(&self) -> Result<Vec<ServedModel>> {
        self.inner.list_models().await
    }
}
