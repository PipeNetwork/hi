//! Exact provider-capability receipts for detached candidate turns.
//!
//! A candidate may make several primary model requests. The receipt stored on
//! the candidate must describe the capabilities that actually governed those
//! accepted requests, rather than a speculative lookup performed before the
//! child builds its canonical objective and sealed envelope.

use std::sync::{Arc, Mutex};

use hi_ai::{ChatRequest, Completion, Provider, ProviderRequestContext, StreamEvent};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct AcceptedCandidateRoute {
    pub(super) requested_model: String,
    pub(super) provider: hi_tools::envelope::ProviderEnvelope,
}

#[derive(Default)]
struct ReceiptState {
    accepted_primary_requests: u64,
    receipt: Option<AcceptedCandidateRoute>,
    error: Option<String>,
}

/// Shared recorder installed immediately outside the candidate's real
/// provider. It observes the exact request after the child has selected tools
/// and sealed capabilities, and records it only when the transport returns a
/// completion.
#[derive(Clone)]
pub(super) struct CandidateCapabilityRecorder {
    expected_objective: Arc<str>,
    state: Arc<Mutex<ReceiptState>>,
}

impl CandidateCapabilityRecorder {
    pub(super) fn new(expected_objective: String) -> Self {
        Self {
            expected_objective: Arc::from(expected_objective),
            state: Arc::new(Mutex::new(ReceiptState::default())),
        }
    }

    pub(super) fn wrap(&self, provider: Arc<dyn Provider>) -> Arc<dyn Provider> {
        Arc::new(ReceiptProvider {
            inner: provider,
            recorder: self.clone(),
        })
    }

    pub(super) fn finish(&self) -> Result<AcceptedCandidateRoute, String> {
        let state = self
            .state
            .lock()
            .map_err(|_| "candidate provider receipt state was poisoned".to_string())?;
        if let Some(error) = &state.error {
            return Err(error.clone());
        }
        if state.accepted_primary_requests == 0 {
            return Err("candidate completed without an accepted primary provider request".into());
        }
        state
            .receipt
            .clone()
            .ok_or_else(|| "candidate accepted request had no capability receipt".into())
    }

    fn record(&self, request: &ChatRequest) {
        if !request.user_turn {
            return;
        }
        let parsed = accepted_route(request, self.expected_objective.as_ref());
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        state.accepted_primary_requests = state.accepted_primary_requests.saturating_add(1);
        let receipt = match parsed {
            Ok(receipt) => receipt,
            Err(error) => {
                state.error.get_or_insert(error);
                return;
            }
        };
        if let Some(first) = &state.receipt
            && first != &receipt
        {
            state.error.get_or_insert_with(|| {
                "candidate provider capability identity changed between accepted primary requests; rerun the candidate on one stable route".into()
            });
            return;
        }
        state.receipt = Some(receipt);
    }
}

fn accepted_route(
    request: &ChatRequest,
    expected_objective: &str,
) -> Result<AcceptedCandidateRoute, String> {
    if request.canonical_objective.as_deref() != Some(expected_objective) {
        return Err(
            "candidate provider request was not bound to the detached child's canonical objective"
                .into(),
        );
    }
    let attached = request
        .tool_envelope
        .as_ref()
        .ok_or_else(|| "candidate provider request had no sealed tool envelope".to_string())?;
    let payload: hi_tools::envelope::ToolEnvelopePayload =
        serde_json::from_value(attached.payload.clone()).map_err(|error| {
            format!("candidate provider request envelope could not be decoded: {error}")
        })?;
    let execution = hi_tools::envelope::ToolEnvelope {
        digest: attached.digest.clone(),
        payload,
    };
    if !execution.digest_is_valid() {
        return Err("candidate provider request envelope failed its digest check".into());
    }
    let provider = execution.payload.provider;
    if provider.requested_model != request.model {
        return Err(
            "candidate provider request model did not match its sealed capability record".into(),
        );
    }
    Ok(AcceptedCandidateRoute {
        requested_model: request.model.clone(),
        provider,
    })
}

struct ReceiptProvider {
    inner: Arc<dyn Provider>,
    recorder: CandidateCapabilityRecorder,
}

#[async_trait::async_trait]
impl Provider for ReceiptProvider {
    async fn stream(
        &self,
        request: ChatRequest,
        sink: &mut (dyn FnMut(StreamEvent) + Send),
    ) -> anyhow::Result<Completion> {
        let receipt_request = request.clone();
        let completion = self.inner.stream(request, sink).await?;
        self.recorder.record(&receipt_request);
        Ok(completion)
    }

    fn capabilities(&self) -> hi_ai::ProviderCapabilities {
        self.inner.capabilities()
    }

    fn capability_candidates(
        &self,
        route: &str,
        model: &str,
    ) -> Vec<hi_ai::ProviderCapabilityCandidate> {
        self.inner.capability_candidates(route, model)
    }

    fn capability_candidates_for_request(
        &self,
        route: &str,
        model: &str,
        context: ProviderRequestContext<'_>,
    ) -> Vec<hi_ai::ProviderCapabilityCandidate> {
        self.inner
            .capability_candidates_for_request(route, model, context)
    }

    async fn list_models(&self) -> anyhow::Result<Vec<hi_ai::ServedModel>> {
        self.inner.list_models().await
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use hi_ai::{Message, ProviderCapabilities, RequestProfile, ToolMode};
    use hi_tools::envelope::{
        ProviderEnvelope, ToolEnvelope, ToolEnvelopeContext, ToolEnvelopeLimits, WorkspaceEnvelope,
        WorkspaceTrust,
    };
    use hi_workspace::{WorkspaceAuthority, WorkspaceVersion};

    use super::*;

    struct CompletingProvider;

    #[async_trait::async_trait]
    impl Provider for CompletingProvider {
        async fn stream(
            &self,
            _request: ChatRequest,
            _sink: &mut (dyn FnMut(StreamEvent) + Send),
        ) -> anyhow::Result<Completion> {
            Ok(Completion::default())
        }
    }

    fn request(objective: &str, revision: &str, user_turn: bool) -> ChatRequest {
        let model = "candidate-model";
        let mut capabilities = ProviderCapabilities::native_tools(true);
        capabilities.actual_model_revision = Some(revision.into());
        let effective = hi_ai::EffectiveProviderCapabilities::conservative(
            hi_ai::CapabilityRoute::new("candidate-route", model),
            capabilities,
        );
        let envelope = ToolEnvelope::build(
            &[],
            ToolEnvelopeContext {
                provider: ProviderEnvelope::from_capability_record(effective),
                workspace: WorkspaceEnvelope {
                    authority: WorkspaceAuthority::Local,
                    binding_id: "candidate-binding".into(),
                    epoch: 7,
                    version: WorkspaceVersion::Unknown,
                },
                trust: WorkspaceTrust::Untrusted,
                permissions: BTreeSet::new(),
                limits: ToolEnvelopeLimits {
                    max_output_tokens: 64,
                    max_parallel_calls: 1,
                    max_calls_per_round: 1,
                    max_inline_output_bytes: 1024,
                    max_tool_argument_bytes: 1024,
                },
                tool_mode: ToolMode::ChatOnly,
                execution_mode: ToolMode::ChatOnly,
                tool_versions: BTreeMap::new(),
            },
        );
        ChatRequest {
            execution: Default::default(),
            model: model.into(),
            request_id: None,
            retry_attempt: 0,
            user_turn,
            canonical_objective: user_turn.then(|| objective.into()),
            messages: vec![Message::user(objective)].into(),
            tools: Vec::new().into(),
            tool_envelope: Some(Arc::new(hi_ai::RequestToolEnvelope {
                digest: envelope.digest,
                payload: serde_json::to_value(envelope.payload).unwrap(),
            })),
            max_tokens: 64,
            temperature: None,
            top_p: None,
            frequency_penalty: None,
            thinking_budget: None,
            reasoning_effort: None,
            profile: RequestProfile {
                tool_mode: ToolMode::ChatOnly,
                ..RequestProfile::default()
            },
        }
    }

    #[tokio::test]
    async fn receipt_comes_from_exact_accepted_child_objective() {
        let raw_prompt = "change the file";
        let child_prompt = format!("detached child guard\n\nTask: {raw_prompt}");
        let recorder = CandidateCapabilityRecorder::new(child_prompt.clone());
        let provider = recorder.wrap(Arc::new(CompletingProvider));
        let mut sink = |_| {};

        provider
            .stream(request(&child_prompt, "child-revision", true), &mut sink)
            .await
            .unwrap();

        let receipt = recorder.finish().unwrap();
        assert_eq!(receipt.requested_model, "candidate-model");
        assert_eq!(
            receipt.provider.actual_model_revision.as_deref(),
            Some("child-revision")
        );
        let stale_raw =
            accepted_route(&request(raw_prompt, "raw-revision", true), raw_prompt).unwrap();
        assert_ne!(
            receipt.provider.capability_digest,
            stale_raw.provider.capability_digest
        );
    }

    #[tokio::test]
    async fn changing_capabilities_between_accepted_requests_fails_closed() {
        let objective = "stable detached objective";
        let recorder = CandidateCapabilityRecorder::new(objective.into());
        let provider = recorder.wrap(Arc::new(CompletingProvider));
        let mut sink = |_| {};
        provider
            .stream(request(objective, "revision-a", true), &mut sink)
            .await
            .unwrap();
        provider
            .stream(request(objective, "revision-b", true), &mut sink)
            .await
            .unwrap();

        let error = recorder.finish().unwrap_err();
        assert!(error.contains("changed between accepted primary requests"));
    }

    #[tokio::test]
    async fn auxiliary_requests_cannot_replace_the_candidate_receipt() {
        let objective = "stable detached objective";
        let recorder = CandidateCapabilityRecorder::new(objective.into());
        let provider = recorder.wrap(Arc::new(CompletingProvider));
        let mut sink = |_| {};
        provider
            .stream(request(objective, "primary", true), &mut sink)
            .await
            .unwrap();
        provider
            .stream(request("auxiliary", "auxiliary", false), &mut sink)
            .await
            .unwrap();

        assert_eq!(
            recorder
                .finish()
                .unwrap()
                .provider
                .actual_model_revision
                .as_deref(),
            Some("primary")
        );
    }
}
