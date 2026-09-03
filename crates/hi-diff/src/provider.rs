use std::time::Instant;

use anyhow::Result;
use hi_ai::{ChatRequest, Completion, Content, Provider, StreamEvent};
use tokio::task::JoinSet;

use crate::{
    ApiOutcome, ApiTarget, CaseVerdict, EquivalenceContract, ToolCallRecord, compare_response,
};

/// Fan one canonical request out to several configured providers. The request
/// is cloned in memory for each target; credentials stay inside the provider
/// objects and are never serialized by `hi-diff`.
pub async fn run_provider_targets(
    case_id: impl Into<String>,
    request: ChatRequest,
    targets: Vec<(ApiTarget, Box<dyn Provider>)>,
    contract: &EquivalenceContract,
) -> Result<CaseVerdict> {
    anyhow::ensure!(
        targets.len() >= 2,
        "provider differential runs need at least two targets"
    );
    let mut jobs = JoinSet::new();
    for (target, provider) in targets {
        let mut request = request.clone();
        request.model = target.model.clone();
        jobs.spawn(async move {
            let started = Instant::now();
            let mut text = String::new();
            let mut reasoning = String::new();
            let mut sink = |event: StreamEvent| match event {
                StreamEvent::Text(delta) => text.push_str(&delta),
                StreamEvent::Reasoning(delta) => reasoning.push_str(&delta),
                StreamEvent::WireAudit(_) => {}
                StreamEvent::Status(_) => {}
                StreamEvent::Warning(_) => {}
                StreamEvent::ToolCallDelta { .. } => {}
            };
            let result = provider.stream(request, &mut sink).await;
            let outcome = match result {
                Ok(completion) => completion_to_outcome(
                    &completion,
                    text,
                    reasoning,
                    started.elapsed().as_millis() as u64,
                ),
                Err(error) => ApiOutcome {
                    text,
                    json: None,
                    tool_calls: Vec::new(),
                    finish_reason: None,
                    error_category: Some(error.to_string()),
                    input_tokens: 0,
                    output_tokens: 0,
                    latency_ms: started.elapsed().as_millis() as u64,
                    schema_valid: None,
                },
            };
            (target.name, outcome)
        });
    }
    let mut outcomes = Vec::new();
    while let Some(result) = jobs.join_next().await {
        outcomes.push(result?);
    }
    outcomes.sort_by(|left, right| left.0.cmp(&right.0));
    Ok(compare_response(case_id, &outcomes, contract))
}

fn completion_to_outcome(
    completion: &Completion,
    streamed_text: String,
    reasoning: String,
    latency_ms: u64,
) -> ApiOutcome {
    let mut text = streamed_text;
    if text.is_empty() {
        text = completion
            .content
            .iter()
            .filter_map(|content| match content {
                Content::Text(value) => Some(value.as_str()),
                _ => None,
            })
            .collect::<String>();
    }
    let tool_calls = completion
        .tool_calls()
        .into_iter()
        .map(|call| ToolCallRecord {
            name: call.name.to_string(),
            arguments: serde_json::from_str(call.arguments)
                .unwrap_or_else(|_| serde_json::Value::String(call.arguments.to_string())),
        })
        .collect();
    let json = serde_json::from_str(&text).ok();
    let _ = reasoning;
    ApiOutcome {
        text,
        json,
        tool_calls,
        finish_reason: completion.stop_reason.clone(),
        error_category: None,
        input_tokens: completion.usage.input_tokens,
        output_tokens: completion.usage.output_tokens,
        latency_ms,
        schema_valid: None,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use async_trait::async_trait;
    use hi_ai::{ChatRequest, Completion, Content, Message, Provider, RequestProfile, StreamEvent};

    use super::*;
    use crate::{ApiTarget, DiffMode, EquivalenceContract};

    struct MockProvider {
        text: &'static str,
        seen_models: Arc<Mutex<Vec<String>>>,
        active: Arc<AtomicUsize>,
        max_active: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl Provider for MockProvider {
        async fn stream(
            &self,
            request: ChatRequest,
            sink: &mut (dyn FnMut(StreamEvent) + Send),
        ) -> anyhow::Result<Completion> {
            self.seen_models.lock().unwrap().push(request.model);
            let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.max_active.fetch_max(active, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(10)).await;
            sink(StreamEvent::Text(self.text.to_string()));
            self.active.fetch_sub(1, Ordering::SeqCst);
            Ok(Completion {
                content: vec![Content::Text(self.text.to_string())],
                usage: Default::default(),
                stop_reason: Some("stop".into()),
                ..Completion::default()
            })
        }
    }

    fn request() -> ChatRequest {
        ChatRequest {
            model: "placeholder".into(),
            request_id: Some("test-request".into()),
            retry_attempt: 0,
            user_turn: true,
            canonical_objective: Some("same prompt".into()),
            messages: Arc::new(vec![Message::user("same prompt")]),
            tools: Arc::from(Vec::new()),
            max_tokens: 128,
            temperature: Some(0.0),
            top_p: None,
            frequency_penalty: None,
            thinking_budget: None,
            reasoning_effort: None,
            profile: RequestProfile::default(),
        }
    }

    fn target(name: &str, model: &str) -> ApiTarget {
        ApiTarget {
            name: name.into(),
            profile: "pipenetwork".into(),
            model: model.into(),
            provider: "pipenetwork".into(),
        }
    }

    #[tokio::test]
    async fn fans_out_same_request_concurrently_and_compares_normalized_text() {
        let seen_models = Arc::new(Mutex::new(Vec::new()));
        let active = Arc::new(AtomicUsize::new(0));
        let max_active = Arc::new(AtomicUsize::new(0));
        let provider = |text| MockProvider {
            text,
            seen_models: seen_models.clone(),
            active: active.clone(),
            max_active: max_active.clone(),
        };
        let contract = EquivalenceContract {
            mode: DiffMode::ApiResponse,
            normalize_whitespace: true,
            ..Default::default()
        };
        let verdict = run_provider_targets(
            "case-1",
            request(),
            vec![
                (
                    target("glm", "pipe/glm-5.2"),
                    Box::new(provider("same  text")),
                ),
                (
                    target("kimi", "pipe/kimi3"),
                    Box::new(provider("same text")),
                ),
            ],
            &contract,
        )
        .await
        .unwrap();

        assert_eq!(verdict.verdict, crate::Verdict::Equivalent);
        assert_eq!(max_active.load(Ordering::SeqCst), 2);
        let mut models = seen_models.lock().unwrap().clone();
        models.sort();
        assert_eq!(models, ["pipe/glm-5.2", "pipe/kimi3"]);
    }
}
